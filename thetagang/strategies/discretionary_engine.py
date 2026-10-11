"""Discretionary strategy: manage the option positions already held and open the
trades listed in strategies.discretionary.trades.

Enabled with run.strategies = ["discretionary"]. It:
  - manages every USD stock/option symbol held (portfolio.symbols are overrides),
  - rolls/closes credit verticals as whole spreads (4-leg / 2-leg combos) with the
    same [roll_when] rules as standalone options; calendar/diagonal legs are left
    alone, except short calls fully covered by shares,
  - prints SPX beta-weighted greeks, stress P&L, delta by symbol and SPX delta = 0
    proposals, and sizes the configured trades to SPX delta 0
    (DiscretionaryRiskMixin in discretionary_risk.py; pure math in thetagang.risk),
  - shows held long VIX calls and sells them above a VIX threshold.
Standalone short options roll/close through OptionsStrategyEngine, which asks this
engine to skip spread legs and applies the manager strike rule.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from ib_async import PortfolioItem, util
from ib_async.contract import ComboLeg, Contract, Index, Option, Stock
from ib_async.order import LimitOrder
from rich.table import Table

from thetagang import log
from thetagang.config import Config
from thetagang.fmt import dfmt, ifmt, pfmt
from thetagang.ibkr import IBKR, TickerField
from thetagang.options import option_dte
from thetagang.strategies.discretionary_risk import (
    ENTRY_ORDER_REF,
    MANAGE_ORDER_REF,
    DiscretionaryRiskMixin,
    quoted_price,
)
from thetagang.trading_operations import (
    NoValidContractsError,
    OptionChainScanner,
    OrderOperations,
)
from thetagang.util import (
    get_lower_price,
    position_pnl,
)

__all__ = [
    "ENTRY_ORDER_REF",
    "DiscretionaryEngine",
    "is_covered_call",
    "quoted_price",
    "roll_strike_limit",
]

# Option roots whose underlying is an index, not a stock: rolls look the underlying
# up as Stock(symbol), which can't resolve for these, so they are never managed.
# ponytail: fixed list; add roots here if you trade other index options.
INDEX_SYMBOLS = {
    "SPX",
    "SPXW",
    "XSP",
    "NDX",
    "NDXP",
    "RUT",
    "RUTW",
    "DJX",
    "XND",
    "OEX",
}


def roll_strike_limit(
    right: str,
    strike: float,
    captured: float,
    itm: bool,
    configured: float | None,
) -> float:
    """Symmetric roll strike bound for calls and puts (discretionary strategy).

    The new strike may move against you by at most the premium already captured
    per share; an ITM position only rolls to the same or a safer strike. Puts get
    an upper bound (lower is safer), calls a lower bound (higher is safer). A
    configured strike_limit still applies."""
    if right.upper().startswith("P"):
        limit = max(strike, strike + captured)
        if itm:
            limit = min(limit, strike)
        if configured:
            limit = min(limit, configured)
    else:
        limit = min(strike, strike - captured)
        if itm:
            limit = max(limit, strike)
        if configured:
            limit = max(limit, configured)
    return round(limit, 2)


def is_covered_call(
    right: str,
    symbol: str,
    portfolio_positions: dict[str, list[PortfolioItem]] | None,
) -> bool:
    """A call on a symbol where shares are owned: rolls with the wheel's covered
    call rule (strike >= share cost, high-water mark if enabled).
    ponytail: any long shares count; the cost floor only raises strikes, so a
    partly covered call is rolled more conservatively, never less."""
    return right.upper().startswith("C") and any(
        isinstance(p.contract, Stock) and p.position > 0
        for p in (portfolio_positions or {}).get(symbol, [])
    )


def vix_hedges_table(
    hedges: list[PortfolioItem],
    vix: float,
    threshold: float | None,
    close: bool,
    handed_off: bool,
) -> Table:
    """Long VIX calls held, with the action this run takes on them."""
    caption = f"VIX {'n/a' if util.isNan(vix) else f'{vix:.2f}'}; " + (
        f"all are sold when VIX > {threshold}"
        if threshold
        else "close_hedges_when_vix_exceeds not set: never closed"
    )
    if handed_off:
        caption += "; buying and selling done by the VIX call hedging stage"
    table = Table(title="VIX call hedges", caption=caption, caption_justify="left")
    for column in ("Contract", "Qty", "DTE", "Price", "P&L", "Action"):
        table.add_column(column, justify="left" if column == "Contract" else "right")
    for p in hedges:
        table.add_row(
            p.contract.localSymbol,
            ifmt(int(p.position)),
            str(option_dte(p.contract.lastTradeDateOrContractMonth)),
            dfmt(p.marketPrice),
            pfmt(position_pnl(p), 1),
            "[red]Close" if close else "Keep",
        )
    return table


class DiscretionaryEngine(DiscretionaryRiskMixin):
    def __init__(
        self,
        *,
        config: Config,
        ibkr: IBKR,
        order_ops: OrderOperations,
        option_scanner: OptionChainScanner,
        qualified_contracts: dict[int, Contract],
        put_is_itm: Callable[[Contract], Awaitable[bool]],
        call_is_itm: Callable[[Contract], Awaitable[bool]],
        get_primary_exchange: Callable[[str], str],
    ) -> None:
        self.config = config
        self.ibkr = ibkr
        self.order_ops = order_ops
        self.option_scanner = option_scanner
        self.qualified_contracts = qualified_contracts
        self.put_is_itm = put_is_itm
        self.call_is_itm = call_is_itm
        self.get_primary_exchange = get_primary_exchange
        self.account_number = config.runtime.account.number
        # [portfolio.symbols.X] blocks from the config, kept as overrides: read at
        # the first discover_symbols, which then replaces portfolio.symbols on
        # every run (reconnects)
        self.symbol_overrides: dict[str, Any] = {}
        self._overrides_loaded = False
        # set once discover_symbols has seen real positions (reconnect guard)
        self.discovered_symbols = False
        # orders planned by plan_trades, enqueued in the post_discretionary stage
        self.planned_orders: list[tuple[Contract, LimitOrder]] = []
        # SPX delta = 0 targets per symbol for the positions summary (display only)
        self.zero_delta: dict[str, tuple[str, ...]] = {}
        # delta (shares), gamma, theta, vega per position (conId) and per symbol.
        # Filled with the risk tables, shown in the symbol and position tables
        self.position_greeks: dict[int, tuple[float, float, float, float]] = {}
        self.symbol_greeks: dict[str, tuple[float, float, float, float] | None] = {}
        self.contract_greeks: dict[int, tuple[float, float, float, float]] = {}
        # betas barely move within a day: reuse them across reconnects/reruns
        self.beta_cache: dict[str, float] = {}
        self.beta_cache_day: date | None = None
        self.spx_closes: dict[Any, float] = {}

    # ----------------------------------------------------------------- symbols

    def held_positions(
        self, portfolio: list[PortfolioItem] | None = None
    ) -> list[PortfolioItem]:
        """Positions the bot works with: this account, non-zero, with a cost basis."""
        if portfolio is None:
            portfolio = self.ibkr.portfolio(account=self.account_number)
        return [
            item
            for item in portfolio
            if item.account == self.account_number
            and item.position != 0
            and item.averageCost != 0
        ]

    def discover_symbols(self) -> None:
        """Manage every symbol held in the account (USD stock or options, except
        index and VIX options); [portfolio.symbols.X] blocks are overrides."""
        from thetagang.config_models import SymbolConfig

        if not self._overrides_loaded:
            self.symbol_overrides = dict(self.config.portfolio.symbols)
            self._overrides_loaded = True
        exchanges: dict[str, str] = {}
        held: list[str] = []
        for item in self.held_positions():
            c = item.contract
            if c.symbol in INDEX_SYMBOLS:
                log.notice(
                    f"{c.localSymbol}: index option, not managed (roll it by hand)"
                )
            if (
                c.symbol in INDEX_SYMBOLS
                or c.symbol == "VIX"
                or c.currency != "USD"
                or c.secType not in ("STK", "OPT")
            ):
                continue
            if c.secType == "STK":
                exchanges[c.symbol] = c.primaryExchange or c.exchange
            if c.symbol not in held:
                held.append(c.symbol)

        if not held:
            if self.discovered_symbols:
                # right after a reconnect the portfolio can be empty until IB syncs
                log.warning(
                    "Portfolio looks empty (not synced yet?), keeping the symbols "
                    "from the last run"
                )
                return
            log.notice("No USD stock or option positions, nothing to manage")
            self.config.portfolio.symbols = {}
            return
        self.discovered_symbols = True

        symbols: dict[str, Any] = {}
        for symbol in sorted(held):
            base = self.symbol_overrides.get(symbol) or SymbolConfig(weight=0.0)
            if not base.primary_exchange and exchanges.get(symbol):
                base = base.model_copy(update={"primary_exchange": exchanges[symbol]})
            symbols[symbol] = base
        self.config.portfolio.symbols = symbols

    # ------------------------------------------------------------ spread legs

    @staticmethod
    def find_credit_verticals(
        positions: list[PortfolioItem],
    ) -> list[tuple[PortfolioItem, PortfolioItem]]:
        """Pair each short option with a long of the same right/expiry/qty that makes a
        credit vertical (long put below the short put, long call above the short call)."""
        options = [p for p in positions if isinstance(p.contract, Option)]
        longs = [p for p in options if p.position > 0]
        pairs: list[tuple[PortfolioItem, PortfolioItem]] = []
        for short in (p for p in options if p.position < 0):
            c = short.contract
            is_put = c.right.upper().startswith("P")
            candidates = [
                p
                for p in longs
                if p.contract.right == c.right
                and p.contract.lastTradeDateOrContractMonth
                == c.lastTradeDateOrContractMonth
                and math.isclose(p.position, -short.position)
                and (
                    p.contract.strike < c.strike
                    if is_put
                    else p.contract.strike > c.strike
                )
            ]
            if candidates:
                long = min(candidates, key=lambda p: abs(p.contract.strike - c.strike))
                longs.remove(long)
                pairs.append((short, long))
        return pairs

    def split_option_legs(
        self, positions: list[PortfolioItem]
    ) -> tuple[
        list[tuple[PortfolioItem, PortfolioItem]], list[PortfolioItem], set[str]
    ]:
        """A symbol's options as (credit verticals, options not in one, rights whose
        leftover options have both shorts and longs = calendar/diagonal legs).
        The one place that decides what counts as a spread leg.

        Covered calls: if the shares in `positions` cover every leftover short call,
        a long call on the symbol can't be protecting them, so calls are not
        treated as calendar/diagonal legs (the short calls roll on their own)."""
        options = [p for p in positions if isinstance(p.contract, Option)]
        pairs = self.find_credit_verticals(options)
        paired = {id(p) for pair in pairs for p in pair}
        unpaired = [p for p in options if id(p) not in paired]
        mixed = {
            right
            for right in {p.contract.right for p in unpaired}
            if any(p.position < 0 and p.contract.right == right for p in unpaired)
            and any(p.position > 0 and p.contract.right == right for p in unpaired)
        }
        shares = sum(p.position for p in positions if isinstance(p.contract, Stock))
        for right in [r for r in mixed if r.upper().startswith("C")]:
            covered_shares_needed = sum(
                -p.position * float(p.contract.multiplier or 100)
                for p in unpaired
                if p.contract.right == right and p.position < 0
            )
            if shares >= covered_shares_needed:
                mixed.discard(right)
        return pairs, unpaired, mixed

    def without_spread_legs(
        self,
        shorts: list[PortfolioItem],
        portfolio_positions: dict[str, list[PortfolioItem]],
    ) -> list[PortfolioItem]:
        """Drop short options that are spread legs from the standalone roll/close
        checks: credit verticals are handled by roll_spreads, calendar/diagonal
        legs are left alone."""
        legs_by_symbol = {
            symbol: self.split_option_legs(portfolio_positions.get(symbol, []))
            for symbol in {s.contract.symbol for s in shorts}
        }
        kept = []
        for short in shorts:
            c = short.contract
            pairs, unpaired, mixed = legs_by_symbol.get(c.symbol, ([], [], set()))
            vertical = next((lg for sh, lg in pairs if sh is short), None)
            if vertical is None and c.right not in mixed:
                kept.append(short)  # standalone: rolls on its own
                continue
            longs = [
                p for p in unpaired if p.position > 0 and p.contract.right == c.right
            ]
            dte = option_dte(c.lastTradeDateOrContractMonth)
            if vertical is not None and self.spread_pnl(short, vertical) is None:
                log.warning(
                    f"{c.localSymbol} ({dte} DTE): vertical with "
                    f"{vertical.contract.localSymbol} has no net credit in its cost "
                    "basis, so its P&L can't be measured: left alone - manage by hand"
                )
            elif vertical is not None:
                log.notice(
                    f"{c.localSymbol} ({dte} DTE): vertical with "
                    f"{vertical.contract.localSymbol}, rolled together as a spread "
                    "when [roll_when] is met"
                )
            else:
                names = ", ".join(lg.contract.localSymbol for lg in longs)
                log.warning(
                    f"{c.localSymbol} ({dte} DTE): calendar/diagonal with {names}, "
                    "left alone - manage by hand"
                )
        return kept

    # ----------------------------------------------------------- roll rules

    @staticmethod
    def spread_pnl(short: PortfolioItem, long: PortfolioItem) -> float | None:
        """Combined P&L as a fraction of the net credit received; None if no credit."""
        credit = abs(short.averageCost * short.position) - abs(
            long.averageCost * long.position
        )
        if credit <= 0:
            return None
        return (short.unrealizedPNL + long.unrealizedPNL) / credit

    def roll_rule_chain(self, right: str, itm: bool, dte: int, pnl: float) -> bool:
        """[roll_when] decision for a credit vertical, in the same order as
        OptionsStrategyEngine.put_can_be_rolled / call_can_be_rolled (a test keeps
        them in sync): always_when_itm, itm, max_dte, then dte/min_pnl or pnl.
        Excess-contract checks are wheel-only and never apply here."""
        rw = self.config.strategies.wheel.defaults.roll_when
        kind = rw.calls if right.upper().startswith("C") else rw.puts
        if itm and kind.always_when_itm:
            return True
        if itm and not kind.itm:
            return False
        if rw.max_dte and dte > rw.max_dte:
            return False
        return (dte <= rw.dte and pnl >= rw.min_pnl) or pnl >= rw.pnl

    def close_rule_allows(self, pnl: float) -> bool:
        """Close when P&L is above roll_when.close_at_pnl (options and spreads)."""
        close_at_pnl = self.config.strategies.wheel.defaults.roll_when.close_at_pnl
        return bool(close_at_pnl) and pnl > close_at_pnl

    def close_if_unable_to_roll_allows(self, symbol: str, dte: int, pnl: float) -> bool:
        """After a failed roll: close if profitable and within roll_when.max_dte."""
        max_dte = self.config.strategies.wheel.defaults.roll_when.max_dte
        return bool(
            self.config.close_if_unable_to_roll(symbol)
            and max_dte
            and dte <= max_dte
            and pnl > 0
        )

    async def short_leg_is_itm(self, contract: Contract) -> bool:
        if contract.right.upper().startswith("C"):
            return await self.call_is_itm(contract)
        return await self.put_is_itm(contract)

    # --------------------------------------------------------------- orders

    async def leg_price(self, contract: Contract, exchange: str) -> float:
        """Mid (or market) price of an option leg; raises if there is no quote, since
        midpoint_or_market_price falls back to 0.0, which would misprice a combo."""
        ticker = await self.ibkr.get_ticker_for_contract(
            Contract(
                conId=contract.conId,
                localSymbol=contract.localSymbol,
                exchange=exchange,
            ),
            required_fields=[],
            optional_fields=[TickerField.MIDPOINT, TickerField.MARKET_PRICE],
        )
        return quoted_price(ticker, contract.localSymbol)

    def combo_order(
        self,
        symbol: str,
        legs: list[tuple[Contract, str]],
        qty: int,
        price: float,
        exchange: str,
    ) -> tuple[Contract, LimitOrder]:
        """A BUY combo (negative price = net credit) over (contract, action) legs."""
        for c, _ in legs:
            self.qualified_contracts[c.conId] = c
        combo = Contract(
            secType="BAG",
            symbol=symbol,
            currency="USD",
            exchange=exchange,
            comboLegs=[
                ComboLeg(conId=c.conId, ratio=1, exchange=exchange, action=action)
                for c, action in legs
            ],
        )
        order = self.order_ops.create_limit_order(
            action="BUY",
            quantity=qty,
            limit_price=round(price, 2),
            use_default_algo=False,
            order_ref=MANAGE_ORDER_REF,  # entries overwrite it with ENTRY_ORDER_REF
            transmit=True,
        )
        return combo, order

    async def roll_spreads(
        self, portfolio_positions: dict[str, list[PortfolioItem]]
    ) -> None:
        """Roll credit verticals as one 4-leg combo, or close them as one 2-leg combo,
        with the same [roll_when] rules as standalone options."""
        exchange = self.order_ops.get_order_exchange()
        for symbol, positions in portfolio_positions.items():
            if symbol == "VIX" or not self.config.trading_is_allowed(symbol):
                continue
            for short, long in self.find_credit_verticals(positions):
                name = short.contract.localSymbol
                pnl = self.spread_pnl(short, long)
                if pnl is None:
                    continue  # no net credit: warned about in without_spread_legs
                dte = option_dte(short.contract.lastTradeDateOrContractMonth)
                try:
                    itm = await self.short_leg_is_itm(short.contract)
                    # same order as standalone options: roll first, otherwise close
                    if self.roll_rule_chain(short.contract.right, itm, dte, pnl):
                        try:
                            await self.roll_spread(symbol, short, long, exchange, itm)
                            continue
                        except NoValidContractsError as e:
                            log.error(f"{name}: spread roll failed: {e}")
                            if not self.close_if_unable_to_roll_allows(
                                symbol, dte, pnl
                            ):
                                continue
                            log.warning(f"{name}: unable to roll spread, closing it")
                    elif not self.close_rule_allows(pnl):
                        continue
                    await self.close_spread(symbol, short, long, exchange)
                except Exception as e:  # noqa: BLE001
                    # never let one spread abort the run before orders are submitted
                    log.error(f"{name}: spread roll/close failed: {e!r}")

    async def close_spread(
        self, symbol: str, short: PortfolioItem, long: PortfolioItem, exchange: str
    ) -> None:
        """Close a credit vertical as one 2-leg combo: buy back short, sell long."""
        old_short, old_long = short.contract, long.contract
        # BUY combo price = cost to close (debit), same convention as rolls
        price = await self.leg_price(old_short, exchange) - await self.leg_price(
            old_long, exchange
        )
        qty = math.floor(abs(short.position))
        combo, order = self.combo_order(
            symbol, [(old_short, "BUY"), (old_long, "SELL")], qty, price, exchange
        )
        log.info(
            f"{symbol}: Closing {old_short.right} spread "
            f"{old_short.strike}/{old_long.strike} {old_short.lastTradeDateOrContractMonth} "
            f"price={dfmt(price, 3)} qty={qty}"
        )
        self.order_ops.enqueue_order(combo, order)

    async def roll_spread(
        self,
        symbol: str,
        short: PortfolioItem,
        long: PortfolioItem,
        exchange: str,
        itm: bool,
    ) -> None:
        """Roll a credit vertical as one 4-leg combo, keeping its width."""
        old_short, old_long = short.contract, long.contract
        kind = "calls" if old_short.right.upper().startswith("C") else "puts"
        roll_when = self.config.strategies.wheel.defaults.roll_when
        credit_only = getattr(roll_when, kind).credit_only
        minimum_credit = self.config.runtime.orders.minimum_credit
        old_net = await self.leg_price(old_short, exchange) - await self.leg_price(
            old_long, exchange
        )
        # same strike rule as standalone rolls, on the spread's net premium
        multiplier = float(old_short.multiplier or 100)
        captured = (short.averageCost - long.averageCost) / multiplier - old_net
        strike_limit = roll_strike_limit(
            old_short.right,
            old_short.strike,
            captured,
            itm,
            self.config.get_strike_limit(symbol, old_short.right),
        )
        new_short, new_long, new_net = await self._new_spread(
            symbol, old_short, old_long, exchange, strike_limit, old_net, credit_only
        )
        # BUY combo: negative price = net credit, same convention as single rolls
        price = old_net - new_net
        if credit_only and price > -minimum_credit:
            raise NoValidContractsError(
                f"roll would cost {dfmt(price, 3)}, credit_only needs a credit of "
                f"at least {dfmt(minimum_credit, 2)}"
            )
        qty = math.floor(abs(short.position))  # the whole spread
        legs = [
            (old_short, "BUY"),
            (old_long, "SELL"),
            (new_short, "SELL"),
            (new_long, "BUY"),
        ]
        log.info(
            f"{symbol}: Rolling {old_short.right} spread "
            f"{old_short.strike}/{old_long.strike} "
            f"{old_short.lastTradeDateOrContractMonth} -> "
            f"{new_short.strike}/{new_long.strike} "
            f"{new_short.lastTradeDateOrContractMonth} price={dfmt(price, 3)} qty={qty}"
        )
        self.order_ops.enqueue_order(
            *self.combo_order(symbol, legs, qty, price, exchange)
        )

    async def _new_spread(
        self,
        symbol: str,
        old_short: Contract,
        old_long: Contract,
        exchange: str,
        strike_limit: float,
        old_net: float,
        credit_only: bool,
    ) -> tuple[Contract, Contract, float]:
        """New short at the target delta, long at the same width: (short, long, net
        mid). With credit_only the new short must pay for the buyback, the new long
        and the minimum credit; the long's cost is only known after a search, so
        search again with the bar raised by it (at most 3 searches)."""
        minimum_credit = self.config.runtime.orders.minimum_credit
        underlying = Stock(
            symbol, exchange, "USD", primaryExchange=self.get_primary_exchange(symbol)
        )
        long_cost = 0.0
        found: tuple[Contract, Contract, float] | None = None
        # the new short must not be either old leg; overlapping picks are added here
        excluded = [(old_long.strike, old_long.lastTradeDateOrContractMonth)]
        for _ in range(3):
            floor = (
                old_net + minimum_credit + long_cost if credit_only else minimum_credit
            )
            ticker = await self._find_new_short(
                underlying, old_short, strike_limit, floor, excluded
            )
            new_short = ticker.contract
            if not new_short:
                raise NoValidContractsError("no contract for the new short leg")
            # same width: long put below / long call above the short strike
            new_long = await self.qualify_strike(
                new_short,
                round(new_short.strike + (old_long.strike - old_short.strike), 2),
                exchange,
            )
            if {new_short.conId, new_long.conId} & {old_short.conId, old_long.conId}:
                # e.g. same expiry, short moved by exactly the width: the combo would
                # trade one conId twice (IB rejects it), so try the next candidate
                excluded.append(
                    (new_short.strike, new_short.lastTradeDateOrContractMonth)
                )
                continue
            long_price = await self.leg_price(new_long, exchange)
            new_net = quoted_price(ticker, new_short.localSymbol) - long_price
            found = (new_short, new_long, new_net)
            enough_credit = old_net - new_net <= -minimum_credit
            if not credit_only or enough_credit or long_price <= long_cost:
                break  # done, or raising the bar can't help any more
            long_cost = long_price
        if found is None:
            raise NoValidContractsError(
                "every candidate overlapped the spread being rolled"
            )
        return found

    async def _find_new_short(
        self,
        underlying: Stock,
        old_short: Contract,
        strike_limit: float,
        floor: float,
        excluded: list[tuple[float, str]],
    ) -> Any:
        """Ticker of the new short leg: target delta, at least `floor` premium, not
        earlier than the old short's expiry, not an excluded strike/expiry."""
        return await self.option_scanner.find_eligible_contracts(
            underlying,
            old_short.right,
            strike_limit,
            minimum_price=lambda: floor,
            exclude_expirations_before=old_short.lastTradeDateOrContractMonth,
            exclude_exp_strike=(
                old_short.strike,
                old_short.lastTradeDateOrContractMonth,
            ),
            also_exclude=excluded,
        )

    # ------------------------------------------------------------- VIX hedge

    async def close_vix_hedges(self) -> None:
        """Show the long VIX calls held (the hedge) and sell them all once VIX is
        above strategies.vix_call_hedge.close_hedges_when_vix_exceeds. Never buys.
        When the vix_call_hedge stage runs it buys and sells them itself: the table
        is still shown, but selling is left to that stage so they aren't sold twice."""
        handed_off = self.config.strategies.vix_call_hedge.enabled and any(
            s.id == "post_vix_call_hedge" and s.enabled
            for s in self.config.run.resolved_stages()
        )
        threshold = self.config.strategies.vix_call_hedge.close_hedges_when_vix_exceeds
        hedges = [
            p
            for p in self.held_positions()
            if p.contract.symbol == "VIX"
            and p.contract.secType == "OPT"
            and p.contract.right.upper().startswith("C")
            and p.position > 0
        ]
        if not hedges:
            log.info("VIX: no long VIX calls held")
            return
        try:
            vix_ticker = await self.ibkr.get_ticker_for_contract(
                Index("VIX", "CBOE", "USD")
            )
            vix = vix_ticker.marketPrice()
        except Exception as e:  # noqa: BLE001
            log.error(f"VIX: no VIX price ({e!r}), keeping the hedges this run")
            vix = float("nan")
        close = bool(threshold) and not util.isNan(vix) and vix > threshold
        log.print(vix_hedges_table(hedges, vix, threshold, close, handed_off))
        if close and not handed_off:
            await self._sell_vix_hedges(hedges)

    async def _sell_vix_hedges(self, hedges: list[PortfolioItem]) -> None:
        try:
            for position in hedges:
                position.contract.exchange = self.order_ops.get_order_exchange()
                ticker = await self.ibkr.get_ticker_for_contract(position.contract)
                try:
                    quoted_price(ticker, position.contract.localSymbol)
                except RuntimeError as e:
                    # never sell a hedge at the 0.0 no-quote fallback
                    log.error(f"VIX: {e}, not closing it this run")
                    continue
                order = self.order_ops.create_limit_order(
                    action="SELL",
                    quantity=abs(position.position),
                    limit_price=self.order_ops.round_vix_price(get_lower_price(ticker)),
                    order_ref=MANAGE_ORDER_REF,
                    transmit=True,
                )
                self.order_ops.enqueue_order(
                    ticker.contract or position.contract, order
                )
        except Exception as e:  # noqa: BLE001
            log.error(f"VIX: closing hedges failed: {e!r}")

    async def run_post_stage(
        self, portfolio_positions: dict[str, list[PortfolioItem]]
    ) -> None:
        """post_discretionary stage: credit verticals, VIX hedges, then the new
        trades planned while building the risk tables."""
        await self.roll_spreads(portfolio_positions)
        await self.close_vix_hedges()
        if self.planned_orders and self.order_ops.orders.records():
            # trades were sized on the book before these rolls/closes, which
            # change SPX delta: wait for them to fill and re-size next run
            log.warning(
                "Rolls/closes queued this run change SPX delta: new discretionary "
                "trades wait for the next run"
            )
        else:
            for contract, order in self.planned_orders:
                self.order_ops.enqueue_order(contract, order)
        self.planned_orders = []
