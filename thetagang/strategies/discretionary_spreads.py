"""Option spreads for the discretionary strategy, as one combo order each.

DiscretionarySpreadsMixin is mixed into DiscretionaryEngine. It decides what is a
spread leg (credit and debit verticals; calendar/diagonal legs are left alone),
rolls/closes credit verticals with the same [roll_when] rules as standalone
options, and manages long (debit) verticals: close at the take-profit, roll out
near expiry.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ib_async import PortfolioItem
from ib_async.contract import ComboLeg, Contract, Option, Stock
from ib_async.order import LimitOrder

from thetagang import log
from thetagang.config import DISCRETIONARY_STRUCTURES, Config
from thetagang.fmt import dfmt
from thetagang.ibkr import IBKR, TickerField
from thetagang.options import option_dte
from thetagang.risk import is_debit_vertical
from thetagang.strategies.discretionary_trades import (
    MANAGE_ORDER_REF,
    quoted_price,
)
from thetagang.trading_operations import (
    NoValidContractsError,
    OptionChainScanner,
    OrderOperations,
)


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


def find_verticals(
    positions: list[PortfolioItem], *, debit: bool
) -> list[tuple[PortfolioItem, PortfolioItem]]:
    """(short, long) verticals: same right, expiry and quantity. Credit: the long
    is further out of the money (put below / call above the short); debit: the
    long is closer to the money (put above / call below the short)."""
    options = [p for p in positions if isinstance(p.contract, Option)]
    longs = [p for p in options if p.position > 0]
    pairs: list[tuple[PortfolioItem, PortfolioItem]] = []
    for short in (p for p in options if p.position < 0):
        c = short.contract
        candidates = [
            p
            for p in longs
            if p.contract.right == c.right
            and p.contract.lastTradeDateOrContractMonth
            == c.lastTradeDateOrContractMonth
            and math.isclose(p.position, -short.position)
            and p.contract.strike != c.strike
            and is_debit_vertical(short, p) == debit
        ]
        if candidates:
            long = min(candidates, key=lambda p: abs(p.contract.strike - c.strike))
            longs.remove(long)
            pairs.append((short, long))
    return pairs


def short_call_shares(options: list[PortfolioItem]) -> float:
    """Shares needed to cover the short calls among `options`."""
    return sum(
        -p.position * float(p.contract.multiplier or 100)
        for p in options
        if p.contract.right.upper().startswith("C") and p.position < 0
    )


def debit_spread_pnl(short: PortfolioItem, long: PortfolioItem) -> float | None:
    """Combined P&L as a fraction of the net debit paid; None if no debit."""
    debit = abs(long.averageCost * long.position) - abs(
        short.averageCost * short.position
    )
    if debit <= 0:
        return None
    return (short.unrealizedPNL + long.unrealizedPNL) / debit


class DiscretionarySpreadsMixin:
    """Needs the DiscretionaryEngine attributes declared below."""

    config: Config
    ibkr: IBKR
    order_ops: OrderOperations
    option_scanner: OptionChainScanner
    qualified_contracts: dict[int, Contract]
    put_is_itm: Callable[[Contract], Awaitable[bool]]
    call_is_itm: Callable[[Contract], Awaitable[bool]]
    get_primary_exchange: Callable[[str], str]

    if TYPE_CHECKING:  # from DiscretionaryTradesMixin

        async def qualify_strike(
            self, like: Contract, strike: float, exchange: str
        ) -> Contract: ...

    # ------------------------------------------------------------ spread legs

    @staticmethod
    def find_credit_verticals(
        positions: list[PortfolioItem],
    ) -> list[tuple[PortfolioItem, PortfolioItem]]:
        """Pair each short option with a long of the same right/expiry/qty that makes a
        credit vertical (long put below the short put, long call above the short call)."""
        return find_verticals(positions, debit=False)

    @staticmethod
    def find_debit_verticals(
        positions: list[PortfolioItem],
    ) -> list[tuple[PortfolioItem, PortfolioItem]]:
        """(short, long) debit verticals: long call below the short call, long put
        above the short put; same expiry and quantity."""
        return find_verticals(positions, debit=True)

    def split_option_legs(
        self, positions: list[PortfolioItem]
    ) -> tuple[
        list[tuple[PortfolioItem, PortfolioItem]], list[PortfolioItem], set[str]
    ]:
        """A symbol's options as (verticals - credit and debit, options not in one,
        rights whose leftover options have both shorts and longs = calendar/diagonal
        legs).
        The one place that decides what counts as a spread leg.

        Credit verticals are paired first; debit verticals only from what is left,
        so no leg is ever in two spreads.

        Covered calls: if the shares in `positions` cover every leftover short call,
        a long call on the symbol can't be protecting them, so calls are neither
        paired as debit spreads nor treated as calendar/diagonal legs (the short
        calls roll on their own)."""
        options = [p for p in positions if isinstance(p.contract, Option)]
        shares = sum(p.position for p in positions if isinstance(p.contract, Stock))
        pairs = self.find_credit_verticals(options)
        credit_legs = {id(p) for pair in pairs for p in pair}
        leftover = [p for p in options if id(p) not in credit_legs]
        if shares >= short_call_shares(leftover) > 0:
            leftover = [
                p for p in leftover if not p.contract.right.upper().startswith("C")
            ]
        pairs += self.find_debit_verticals(leftover)
        paired = {id(p) for pair in pairs for p in pair}
        unpaired = [p for p in options if id(p) not in paired]
        mixed = {
            right
            for right in {p.contract.right for p in unpaired}
            if any(p.position < 0 and p.contract.right == right for p in unpaired)
            and any(p.position > 0 and p.contract.right == right for p in unpaired)
        }
        for right in [r for r in mixed if r.upper().startswith("C")]:
            if shares >= short_call_shares(unpaired):
                mixed.discard(right)
        return pairs, unpaired, mixed

    def without_spread_legs(
        self,
        shorts: list[PortfolioItem],
        portfolio_positions: dict[str, list[PortfolioItem]],
    ) -> list[PortfolioItem]:
        """Drop short options that are spread legs from the standalone roll/close
        checks: verticals are handled by roll_spreads / manage_debit_spreads,
        calendar/diagonal legs are left alone."""
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
            if vertical is not None and is_debit_vertical(short, vertical):
                log.notice(
                    f"{c.localSymbol} ({dte} DTE): debit spread with "
                    f"{vertical.contract.localSymbol}, managed together: closed at "
                    "the take-profit, rolled out near expiry"
                )
            elif vertical is not None and self.spread_pnl(short, vertical) is None:
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
        self,
        symbol: str,
        short: PortfolioItem,
        long: PortfolioItem,
        exchange: str,
        price: float | None = None,
    ) -> None:
        """Close a vertical (credit or debit) as one 2-leg combo: buy back the
        short, sell the long. `price` = short mid - long mid, fetched if not given."""
        old_short, old_long = short.contract, long.contract
        # BUY combo price = cost to close (debit), same convention as rolls
        if price is None:
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

    # ------------------------------------------------------- debit spreads

    async def manage_debit_spreads(
        self, portfolio_positions: dict[str, list[PortfolioItem]]
    ) -> None:
        """Long (debit) verticals as one combo: close at the take-profit (value >=
        debit_spread_take_profit x width), else roll out at roll_when.dte."""
        cfg = self.config.strategies.discretionary
        roll_dte = self.config.strategies.wheel.defaults.roll_when.dte
        exchange = self.order_ops.get_order_exchange()
        for symbol, positions in portfolio_positions.items():
            if symbol == "VIX" or not self.config.trading_is_allowed(symbol):
                continue
            # from split_option_legs: the same pairing every other check uses
            spreads = self.split_option_legs(positions)[0]
            for short, long in (p for p in spreads if is_debit_vertical(*p)):
                name = long.contract.localSymbol
                width = abs(long.contract.strike - short.contract.strike)
                try:
                    value = await self.leg_price(
                        long.contract, exchange
                    ) - await self.leg_price(short.contract, exchange)
                    dte = option_dte(long.contract.lastTradeDateOrContractMonth)
                    if value >= cfg.debit_spread_take_profit * width:
                        log.info(f"{name}: debit spread at the take-profit")
                        await self.close_spread(symbol, short, long, exchange, -value)
                    elif dte <= roll_dte:
                        try:
                            await self.roll_debit_spread(
                                symbol, short, long, exchange, value, width
                            )
                        except NoValidContractsError as e:
                            pnl = debit_spread_pnl(short, long)
                            if pnl is None or not self.close_if_unable_to_roll_allows(
                                symbol, dte, pnl
                            ):
                                raise
                            log.warning(f"{name}: unable to roll ({e}), closing it")
                            await self.close_spread(
                                symbol, short, long, exchange, -value
                            )
                except NoValidContractsError as e:
                    log.warning(f"{name}: debit spread not rolled ({e}) - roll by hand")
                except Exception as e:  # noqa: BLE001
                    log.error(f"{name}: debit spread close/roll failed: {e!r}")

    def _debit_spread_target(
        self, symbol: str, right: str
    ) -> tuple[float | None, int | None]:
        """Delta and DTE of the [[strategies.discretionary.trades]] debit spread
        for this symbol and right, so rolls keep the spread where it was opened;
        (None, None) = the roll defaults."""
        for trade in self.config.strategies.discretionary.trades:
            trade_right, sign, second = DISCRETIONARY_STRUCTURES[trade.structure]
            if (
                trade.symbol == symbol
                and second is not None
                and sign > 0
                and right.upper().startswith(trade_right)
            ):
                return trade.delta, trade.dte
        return None, None

    async def roll_debit_spread(
        self,
        symbol: str,
        short: PortfolioItem,
        long: PortfolioItem,
        exchange: str,
        old_value: float,
        width: float,
    ) -> None:
        """Sell the old spread and buy one in a later expiry as one 4-leg combo:
        same width, new long leg at the delta/DTE of the configured debit-spread
        trade for the symbol (else the roll defaults). Skipped if the net debit is
        above debit_spread_max_roll_debit x width."""
        old_long, old_short = long.contract, short.contract
        underlying = Stock(
            symbol, exchange, "USD", primaryExchange=self.get_primary_exchange(symbol)
        )
        target_delta, target_dte = self._debit_spread_target(symbol, old_long.right)
        excluded = [(old_short.strike, old_short.lastTradeDateOrContractMonth)]
        for _ in range(3):
            ticker = await self.option_scanner.find_eligible_contracts(
                underlying,
                old_long.right,
                None,
                minimum_price=lambda: 0.01,
                target_delta=target_delta,
                target_dte=target_dte,
                exclude_expirations_before=old_long.lastTradeDateOrContractMonth,
                exclude_exp_strike=(
                    old_long.strike,
                    old_long.lastTradeDateOrContractMonth,
                ),
                also_exclude=excluded,
            )
            new_long = ticker.contract
            if not new_long:
                raise NoValidContractsError("no contract for the new long leg")
            new_short = await self.qualify_strike(
                new_long,
                round(new_long.strike + (old_short.strike - old_long.strike), 2),
                exchange,
            )
            if not {new_long.conId, new_short.conId} & {
                old_long.conId,
                old_short.conId,
            }:
                break
            # the combo would trade one conId twice: try the next candidate
            excluded.append((new_long.strike, new_long.lastTradeDateOrContractMonth))
        else:
            raise NoValidContractsError("every candidate overlapped the old spread")
        new_value = quoted_price(ticker, new_long.localSymbol) - await self.leg_price(
            new_short, exchange
        )
        price = new_value - old_value  # BUY combo: positive = net debit
        max_debit = self.config.strategies.discretionary.debit_spread_max_roll_debit
        if price > max_debit * width:
            raise NoValidContractsError(
                f"roll would cost {dfmt(price, 3)}, more than "
                f"{max_debit:g} x width {dfmt(width, 2)}"
            )
        legs = [
            (old_long, "SELL"),
            (old_short, "BUY"),
            (new_long, "BUY"),
            (new_short, "SELL"),
        ]
        qty = math.floor(long.position)
        log.info(
            f"{symbol}: Rolling debit spread {old_long.strike}/{old_short.strike}"
            f"{old_long.right} {old_long.lastTradeDateOrContractMonth} -> "
            f"{new_long.strike}/{new_short.strike} "
            f"{new_long.lastTradeDateOrContractMonth} price={dfmt(price, 3)} qty={qty}"
        )
        self.order_ops.enqueue_order(
            *self.combo_order(symbol, legs, qty, price, exchange)
        )
