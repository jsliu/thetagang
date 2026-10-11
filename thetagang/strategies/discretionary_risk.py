"""Risk tables and new-trade planning for the discretionary strategy.

DiscretionaryRiskMixin is mixed into DiscretionaryEngine. It fetches betas and
option greeks, prints the SPX beta-weighted greeks / stress / delta-by-symbol
tables, the SPX delta = 0 target weights per symbol (display only), and sizes
strategies.discretionary.trades so SPX delta goes to 0 (real orders, enqueued by
the post_discretionary stage). Pure math lives in thetagang.risk.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ib_async import PortfolioItem, Ticker, util
from ib_async.contract import Contract, Index, Option, Stock
from ib_async.order import LimitOrder
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.table import Table

from thetagang import log
from thetagang.config import (
    DISCRETIONARY_STRUCTURES,
    Config,
    DiscretionaryTradeConfig,
)
from thetagang.fmt import dfmt, ifmt
from thetagang.ibkr import IBKR, TickerField
from thetagang.risk import (
    Adjuster,
    Book,
    Totals,
    aggregate_book,
    breakdown_table,
    compute_beta,
    contract_greeks,
    greeks_table,
    per_contract,
    position_greeks,
    side_weight,
    single_adjuster,
    size_adjustment,
    spread_adjuster,
    stock_trade,
    stress_table,
    symbol_greeks,
    valid_greeks,
    zero_delta_targets,
)
from thetagang.trading_operations import (
    NoValidContractsError,
    OptionChainScanner,
    OrderOperations,
)
from thetagang.util import (
    count_long_option_positions,
    count_short_option_positions,
    midpoint_or_market_price,
    weighted_avg_long_strike,
    weighted_avg_short_strike,
)

# Concurrent IB market-data/history requests for the risk tables (IB pacing limits)
RISK_TABLE_CONCURRENCY = 8

# Index option roots -> the index to take beta history from (SPX/XSP: beta 1).
# ponytail: exchanges as IB lists them; a wrong one falls back to beta 1 with a warning
INDEX_HISTORY = {
    "NDX": ("NDX", "NASDAQ"),
    "NDXP": ("NDX", "NASDAQ"),
    "XND": ("XND", "NASDAQ"),
    "RUT": ("RUT", "RUSSELL"),
    "RUTW": ("RUT", "RUSSELL"),
    "DJX": ("DJX", "CBOE"),
    "OEX": ("OEX", "CBOE"),
}


def quoted_price(ticker: Ticker, name: str) -> float:
    """midpoint_or_market_price, but raise instead of returning the 0.0 fallback."""
    price = midpoint_or_market_price(ticker)
    if not price or util.isNan(price) or price <= 0:
        raise RuntimeError(f"no quote for {name}")
    return price


# orderRef of orders that open discretionary trades: stale ones are always
# cancelled at startup so a rerun can't open the same trade twice
ENTRY_ORDER_REF = "thetagang-discretionary-entry"
# orderRef of spread rolls/closes and VIX hedge sells: also always cancelled at
# startup, so a rerun with cancel_orders = false can't send them twice
MANAGE_ORDER_REF = "thetagang-discretionary-manage"
DISCRETIONARY_ORDER_REFS = (ENTRY_ORDER_REF, MANAGE_ORDER_REF)


@dataclass
class TradeLeg:
    contract: Contract
    sign: int  # +1 long, -1 short, per unit of the trade
    greeks: Any
    price: float  # mid


@dataclass
class PlannedTrade:
    trade: DiscretionaryTradeConfig
    legs: list[TradeLeg] = field(default_factory=list)
    unit_share_delta: float = 0.0
    unit_plain_dollars: float = 0.0
    unit_beta_dollars: float = 0.0
    unit_spx: float = 0.0
    unit_gamma: float = 0.0
    unit_theta: float = 0.0
    unit_vega: float = 0.0
    units: int = 0
    note: str = ""


def size_plans(plans: list[PlannedTrade], need: float) -> None:
    """Split `need` (SPX delta) across the priced trades by weight, in whole units,
    never in the wrong direction, capped by max_contracts. A capped trade's
    remainder is not passed to the others."""
    usable = [p for p in plans if p.legs and p.unit_spx and p.unit_spx * need > 0]
    for p in plans:
        if p.legs and p.unit_spx and p not in usable:
            p.note = p.note or "would move SPX delta the wrong way"
    total_weight = sum(p.trade.weight for p in usable)
    for p in usable:
        target = need * p.trade.weight / total_weight
        p.units = max(round(target / p.unit_spx), 0)
        cap = p.trade.max_contracts
        if cap is not None and p.units > cap:
            p.units = cap
            p.note = p.note or f"capped at max_contracts={cap}"
        if p.units == 0:
            p.note = p.note or "less than one contract needed"


class DiscretionaryRiskMixin:
    """Needs the DiscretionaryEngine attributes and helpers declared below."""

    config: Config
    ibkr: IBKR
    order_ops: OrderOperations
    option_scanner: OptionChainScanner
    qualified_contracts: dict[int, Contract]
    account_number: str
    planned_orders: list[tuple[Contract, LimitOrder]]
    zero_delta: dict[str, tuple[str, ...]]
    position_greeks: dict[int, tuple[float, float, float, float]]
    symbol_greeks: dict[str, tuple[float, float, float, float] | None]
    contract_greeks: dict[int, tuple[float, float, float, float]]
    beta_cache: dict[str, float]
    beta_cache_day: date | None
    spx_closes: dict[Any, float]

    symbol_overrides: dict[str, Any]

    def symbol_config(self, symbol: str) -> Any:
        """[portfolio.symbols.X] for a symbol, held or not (discover_symbols keeps
        only held symbols in portfolio.symbols; overrides keep the rest)."""
        return self.config.portfolio.symbols.get(symbol) or self.symbol_overrides.get(
            symbol
        )

    # implemented by DiscretionaryEngine
    def held_positions(
        self, portfolio: list[PortfolioItem] | None = None
    ) -> list[PortfolioItem]:
        raise NotImplementedError

    def split_option_legs(
        self, positions: list[PortfolioItem]
    ) -> tuple[
        list[tuple[PortfolioItem, PortfolioItem]], list[PortfolioItem], set[str]
    ]:
        raise NotImplementedError

    def combo_order(
        self,
        symbol: str,
        legs: list[tuple[Contract, str]],
        qty: int,
        price: float,
        exchange: str,
    ) -> tuple[Contract, LimitOrder]:
        raise NotImplementedError

    # ----------------------------------------------------------- adjusters

    def has_calendar_or_diagonal_legs(self, options: list[PortfolioItem]) -> bool:
        return bool(self.split_option_legs(options)[2])

    def option_adjusters(
        self,
        options: list[PortfolioItem],
        greeks: dict[int, Any],
        beta: float,
        spx_price: float,
    ) -> list[Adjuster]:
        """Positions in one symbol that the zero-delta proposal may resize.

        Credit verticals are resized as whole spreads (same width); a vertical with
        a leg lacking greeks is skipped, never sized as a single option. Calendar/
        diagonal legs are left alone. Other options are resized one contract at a
        time if they have greeks and at least 0.01 delta."""

        def has(item: PortfolioItem) -> bool:
            return valid_greeks(greeks.get(item.contract.conId))

        def unit(item: PortfolioItem) -> tuple[float, float, float, float]:
            return per_contract(item, greeks[item.contract.conId], beta, spx_price)

        pairs, unpaired, mixed = self.split_option_legs(options)
        found = [
            spread_adjuster(short, long, unit(short), unit(long))
            for short, long in pairs
            if has(short) and has(long)
        ] + [
            single_adjuster(item, unit(item))
            for item in unpaired
            if item.contract.right not in mixed and has(item)
        ]
        return [a for a in found if a is not None]

    # ------------------------------------------------------------ risk tables

    async def portfolio_greeks_table(self) -> list[RenderableType]:
        """Risk tables for the whole account: greeks (now and after the planned
        discretionary trades, or the SPX delta = 0 proposals when no trades are
        configured), stress P&L, delta by symbol, and the trade plan. Also fills
        self.zero_delta."""
        self.zero_delta = {}  # fresh each run: closed symbols must not keep old rows
        self.position_greeks, self.symbol_greeks, self.contract_greeks = {}, {}, {}
        items, spx_price, betas, greeks = await self.fetch_risk_inputs()
        book = aggregate_book(items, betas, greeks, spx_price)
        self.position_greeks = position_greeks(items, greeks)
        self.symbol_greeks = symbol_greeks(items, self.position_greeks)
        self.contract_greeks = contract_greeks(items, greeks)
        after = self.zero_delta_proposals(book, items, betas, greeks, spx_price)
        plan_table, after_trades = await self.plan_trades(book, betas, spx_price)
        if plan_table is not None:  # real orders: "After trades" shows these instead
            after = after_trades
        tables: list[RenderableType] = [
            greeks_table(book, after, spx_price),
            stress_table(book.legs),
            breakdown_table(book, betas),
        ]
        if plan_table is not None:
            tables.append(plan_table)
        return tables

    async def risk_panel(self, account_table: Table) -> RenderableType:
        """Account summary next to the greeks and stress tables; breakdown and the
        trade plan below."""
        self.planned_orders = []  # never carry a plan over from an earlier run
        try:
            greeks, scenarios, *rest = await self.portfolio_greeks_table()
            return Group(Columns([account_table, greeks, scenarios]), *rest)
        except Exception as e:  # noqa: BLE001
            # display-only: never block managing positions on missing market data
            log.warning(f"Could not compute portfolio greeks: {e!r}")
            self.planned_orders = []  # no risk numbers, no new trades
            return account_table

    def _history_contract(self, symbol: str) -> Contract:
        if symbol in INDEX_HISTORY:
            index, exchange = INDEX_HISTORY[symbol]
            return Index(index, exchange, "USD")
        sconfig = self.symbol_config(symbol)
        return Stock(
            symbol,
            "SMART",
            "USD",
            primaryExchange=sconfig.primary_exchange if sconfig else "",
        )

    async def beta(
        self, symbol: str, limit: asyncio.Semaphore | None = None
    ) -> float | None:
        """1-year beta vs SPX (cached for the day); None if there is no usable
        history. Needs self.spx_closes, loaded by fetch_risk_inputs."""
        if symbol in ("SPX", "SPXW", "XSP"):
            return 1.0
        if symbol in self.beta_cache:
            return self.beta_cache[symbol]
        try:
            if limit is None:
                bars = await self.ibkr.request_historical_data(
                    self._history_contract(symbol), "1 Y"
                )
            else:
                async with limit:
                    bars = await self.ibkr.request_historical_data(
                        self._history_contract(symbol), "1 Y"
                    )
            beta = compute_beta({b.date: b.close for b in bars}, self.spx_closes)
        except Exception:  # noqa: BLE001
            beta = None
        if beta is not None:
            self.beta_cache[symbol] = beta
        return beta

    async def fetch_risk_inputs(
        self,
    ) -> tuple[list[PortfolioItem], float, dict[str, float], dict[int, Any]]:
        """USD stock/option positions, SPX price, 1-year betas and option greeks.
        IB requests are capped at RISK_TABLE_CONCURRENCY to respect pacing limits."""
        items = self._risk_items(self.ibkr.portfolio(account=self.account_number))
        spx = Index("SPX", "CBOE", "USD")
        spx_ticker = await self.ibkr.get_ticker_for_contract(
            spx, required_fields=[TickerField.MARKET_PRICE], optional_fields=[]
        )
        spx_price = spx_ticker.marketPrice()
        if self.beta_cache_day != date.today():  # noqa: DTZ011
            closes = {
                b.date: b.close
                for b in await self.ibkr.request_historical_data(spx, "1 Y")
            }
            if not closes:
                raise RuntimeError("no SPX history from IB")
            # only now: a failed load must be retried on the next run
            self.spx_closes, self.beta_cache = closes, {}
            self.beta_cache_day = date.today()  # noqa: DTZ011
        limit = asyncio.Semaphore(RISK_TABLE_CONCURRENCY)
        betas = await self._betas(sorted({i.contract.symbol for i in items}), limit)
        greeks = await self._option_greeks(
            [i for i in items if i.contract.secType == "OPT"], limit
        )
        return items, spx_price, betas, greeks

    def _risk_items(self, portfolio: list[PortfolioItem]) -> list[PortfolioItem]:
        """Held USD stock and options, without VIX options (VIX moves against the
        market and prices off VIX futures: delta x VIX x beta is not SPX exposure)."""
        no_cost = [
            i.contract.localSymbol or i.contract.symbol
            for i in portfolio
            if i.position and i.averageCost == 0
        ]
        if no_cost:
            log.warning(
                "No cost basis, left out of the risk tables and symbol discovery "
                "(rolls/closes may still touch them): " + ", ".join(no_cost)
            )
        items = [
            i
            for i in self.held_positions(portfolio)
            if i.contract.currency == "USD" and i.contract.secType in ("STK", "OPT")
        ]
        vix = [i for i in items if i.contract.symbol == "VIX"]
        if vix:
            log.notice(
                f"{len(vix)} VIX option position(s) excluded from the SPX risk tables "
                "(VIX has a negative, non-linear relation to SPX)"
            )
        return [i for i in items if i.contract.symbol != "VIX"]

    async def _betas(
        self, symbols: list[str], limit: asyncio.Semaphore
    ) -> dict[str, float]:
        found = await asyncio.gather(*(self.beta(s, limit) for s in symbols))
        fallback = [s for s, b in zip(symbols, found) if b is None]
        if fallback:
            log.warning(
                "Beta defaulted to 1.0 (no usable history): " + ", ".join(fallback)
            )
        # ponytail: unknown beta treated as 1; retried next run
        return {s: (1.0 if b is None else b) for s, b in zip(symbols, found)}

    async def _option_greeks(
        self, options: list[PortfolioItem], limit: asyncio.Semaphore
    ) -> dict[int, Any]:
        """IB model greeks by conId; None where IB returned none."""

        async def greeks_for(item: PortfolioItem) -> Any:
            try:
                async with limit:
                    ticker = await self.ibkr.get_ticker_for_contract(
                        Contract(
                            conId=item.contract.conId,
                            localSymbol=item.contract.localSymbol,
                            exchange=self.order_ops.get_order_exchange(),
                        ),
                        required_fields=[],
                        optional_fields=[TickerField.GREEKS],
                    )
                return ticker.modelGreeks
            except Exception:  # noqa: BLE001
                return None  # counted as "Options without greeks"

        results = await asyncio.gather(*map(greeks_for, options))
        return {i.contract.conId: g for i, g in zip(options, results)}

    def zero_delta_proposals(
        self,
        book: Book,
        items: list[PortfolioItem],
        betas: dict[str, float],
        greeks: dict[int, Any],
        spx_price: float,
    ) -> Totals:
        """SPX delta = 0 targets and trades per symbol (stored in self.zero_delta for
        the positions summary). Returns the greeks after the suggested trades."""
        plain_now = {sym: plain for sym, (plain, _) in book.by_symbol.items()}
        current = {sym: weighted for sym, (_, weighted) in book.by_symbol.items()}
        plain_target = zero_delta_targets(plain_now, betas)
        target = {sym: betas[sym] * d for sym, d in plain_target.items()}
        gross_now = sum(abs(b) for b in current.values()) or 1.0
        gross_target = sum(abs(b) for b in target.values()) or 1.0
        after = Totals(**vars(book.totals))
        for symbol, now in current.items():
            price = book.prices.get(symbol, 0.0)
            weights = (
                side_weight(now, gross_now),
                side_weight(target[symbol], gross_target),
                dfmt(target[symbol], 0),
            )
            if price <= 1e-9:
                self.zero_delta[symbol] = (*weights, "n/a", "n/a")
                continue
            shares = (plain_target[symbol] - plain_now[symbol]) / price
            symbol_items = [i for i in items if i.contract.symbol == symbol]
            traded, trade, (gamma, theta, vega) = self._symbol_adjustment(
                symbol_items, shares, greeks, betas[symbol], spx_price
            )
            after.gamma += gamma
            after.theta += theta
            after.vega += vega
            after.plain_delta_dollars += traded * price
            after.delta_dollars += traded * price * betas[symbol]
            self.zero_delta[symbol] = (*weights, f"{shares:+,.0f}", trade)
        return after

    def _symbol_adjustment(
        self,
        symbol_items: list[PortfolioItem],
        shares: float,
        greeks: dict[int, Any],
        beta: float,
        spx_price: float,
    ) -> tuple[float, str, tuple[float, float, float]]:
        """The trade toward `shares` of delta in one symbol: (share delta traded,
        description, gamma/theta/vega added). Resizes the position with the most
        delta; stock-only symbols trade shares; never flips a position."""
        adjusters = self.option_adjusters(symbol_items, greeks, beta, spx_price)
        if adjusters:
            main = max(adjusters, key=lambda a: a.exposure)
            units, trade = size_adjustment(main, shares)
            added = (units * main.gamma, units * main.theta, units * main.vega)
            return units * main.delta, trade, added
        stock = [i.position for i in symbol_items if i.contract.secType == "STK"]
        if stock:
            traded = stock_trade(shares, stock[0])
            return traded, (f"{traded:+d} shares" if traded else "0"), (0, 0, 0)
        reason = (
            "none (calendar/diagonal legs)"
            if self.has_calendar_or_diagonal_legs(symbol_items)
            else "none (delta < 0.01)"
        )
        return 0.0, reason, (0.0, 0.0, 0.0)

    # ------------------------------------------------------- new trades

    async def _trade_legs(self, trade: DiscretionaryTradeConfig) -> list[TradeLeg]:
        """The contracts for one configured trade: the main leg at the target delta
        and, for a vertical, the second leg `width` away in the same expiry."""
        right, main_sign, second = DISCRETIONARY_STRUCTURES[trade.structure]
        sconfig = self.symbol_config(trade.symbol)
        underlying = Stock(
            trade.symbol,
            self.order_ops.get_order_exchange(),
            "USD",
            primaryExchange=sconfig.primary_exchange if sconfig else "",
        )
        ticker = await self.option_scanner.find_eligible_contracts(
            underlying,
            right,
            None,
            minimum_price=lambda: 0.01,
            target_dte=trade.dte,
            target_delta=trade.delta,
            # a new position: open interest must be known, as for the wheel
            allow_missing_open_interest=False,
        )
        main = ticker.contract
        if not main:
            raise NoValidContractsError("no contract at the target delta")
        legs = [
            TradeLeg(
                main,
                main_sign,
                ticker.modelGreeks,
                quoted_price(ticker, main.localSymbol),
            )
        ]
        if second is not None:
            offset, sign = second
            legs.append(await self._second_leg(main, offset * (trade.width or 0), sign))
        if not all(valid_greeks(leg.greeks) for leg in legs):
            raise NoValidContractsError("no option greeks from IB")
        return legs

    async def _second_leg(self, main: Contract, offset: float, sign: int) -> TradeLeg:
        """The vertical's other leg: same expiry and class, strike moved by offset."""
        other = await self.qualify_strike(
            main, round(main.strike + offset, 2), self.order_ops.get_order_exchange()
        )
        ticker = await self.ibkr.get_ticker_for_contract(
            other,
            required_fields=[],
            optional_fields=[
                TickerField.GREEKS,
                TickerField.MIDPOINT,
                TickerField.MARKET_PRICE,
            ],
        )
        return TradeLeg(
            other, sign, ticker.modelGreeks, quoted_price(ticker, other.localSymbol)
        )

    async def qualify_strike(
        self, like: Contract, strike: float, exchange: str
    ) -> Contract:
        """The option like `like` (same symbol, right, expiry, class) at `strike`."""
        qualified = await self.ibkr.qualify_contracts(
            Option(
                like.symbol,
                like.lastTradeDateOrContractMonth,
                strike,
                like.right,
                exchange,
                multiplier=like.multiplier,
                currency="USD",
                tradingClass=like.tradingClass,
            )
        )
        found = qualified[0] if qualified else None
        if not found or not found.conId:
            raise NoValidContractsError(
                f"no {like.right} at strike {strike} for "
                f"{like.lastTradeDateOrContractMonth}"
            )
        return found

    async def plan_trades(
        self, book: Book, betas: dict[str, float], spx_price: float
    ) -> tuple[Table | None, Totals]:
        """Size strategies.discretionary.trades so the book's SPX delta goes to 0.
        Existing positions are taken as they are and never resized. Orders are kept
        in self.planned_orders for the post_discretionary stage."""
        after = Totals(**vars(book.totals))
        cfg = self.config.strategies.discretionary
        if not cfg.trades:
            return None, after
        now = book.totals.delta_dollars / spx_price
        if book.missing:
            # SPX delta is incomplete without them: sizing against it could
            # open the wrong amount, so trade nothing (and skip the chain scans)
            blocked = f"{book.missing} option(s) without greeks: no new trades"
        elif abs(now) <= cfg.delta_band:
            blocked = f"SPX delta {now:+.2f} is within the band of {cfg.delta_band:g}"
        else:
            blocked = ""
        if blocked:
            plans = [PlannedTrade(t, note=blocked) for t in cfg.trades]
        else:
            # ponytail: one trade at a time to stay within IB pacing limits
            plans = [await self._price_trade(t, spx_price) for t in cfg.trades]
            size_plans(plans, need=-now)

        for p in plans:
            if not p.units:
                continue
            after.delta_dollars += p.units * p.unit_beta_dollars
            after.plain_delta_dollars += p.units * p.unit_plain_dollars
            after.gamma += p.units * p.unit_gamma
            after.theta += p.units * p.unit_theta
            after.vega += p.units * p.unit_vega
            self.planned_orders.append(self._entry_order(p))
        return self._plan_table(plans, now, after.delta_dollars / spx_price), after

    async def _price_trade(
        self, trade: DiscretionaryTradeConfig, spx_price: float
    ) -> PlannedTrade:
        """Pick one unit's contracts and its greeks; a note says why if it can't."""
        plan = PlannedTrade(trade)
        sconfig = self.symbol_config(trade.symbol)
        if sconfig is not None and sconfig.no_trading:
            plan.note = "no_trading is set"
            return plan
        try:
            plan.legs = await self._trade_legs(trade)
        except Exception as e:  # noqa: BLE001
            plan.note = f"skipped: {e}"
            return plan
        beta = await self.beta(trade.symbol)
        if beta is None:
            beta = 1.0
            plan.note = "beta defaulted to 1.0"
        for leg in plan.legs:
            g, size = leg.greeks, leg.sign * float(leg.contract.multiplier or 100)
            plan.unit_share_delta += g.delta * size
            plan.unit_gamma += g.gamma * size * (beta * g.undPrice / spx_price) ** 2
            plan.unit_theta += g.theta * size
            plan.unit_vega += g.vega * size
        plan.unit_plain_dollars = plan.unit_share_delta * plan.legs[0].greeks.undPrice
        plan.unit_beta_dollars = plan.unit_plain_dollars * beta
        plan.unit_spx = plan.unit_beta_dollars / spx_price
        return plan

    def _entry_order(self, plan: PlannedTrade) -> tuple[Contract, LimitOrder]:
        """Single leg: BUY/SELL at mid. Vertical: one BUY combo (negative = credit).
        Tagged ENTRY_ORDER_REF so a rerun cancels it if it is still working."""
        if len(plan.legs) == 1:
            leg = plan.legs[0]
            order = self.order_ops.create_limit_order(
                action="BUY" if leg.sign > 0 else "SELL",
                quantity=plan.units,
                limit_price=round(leg.price, 2),
                order_ref=ENTRY_ORDER_REF,
                transmit=True,
            )
            self.qualified_contracts[leg.contract.conId] = leg.contract
            return leg.contract, order
        price = sum(leg.sign * leg.price for leg in plan.legs)
        combo, order = self.combo_order(
            plan.trade.symbol,
            [(leg.contract, "BUY" if leg.sign > 0 else "SELL") for leg in plan.legs],
            plan.units,
            price,
            self.order_ops.get_order_exchange(),
        )
        order.orderRef = ENTRY_ORDER_REF
        return combo, order

    @staticmethod
    def _plan_table(plans: list[PlannedTrade], now: float, after: float) -> Table:
        table = Table(
            title=f"Discretionary trades: SPX delta {now:+.2f} -> {after:+.2f}",
            caption=(
                "Existing positions stay as they are; only these trades are sized. "
                "Contracts are whole units (spreads: whole spreads), never in the "
                "wrong direction. Orders go out in the post_discretionary stage "
                "(dry run: shown only)."
            ),
            caption_justify="left",
        )
        for column in ("Trade", "Contracts", "SPX Δ / unit", "Weight", "Open", "Note"):
            table.add_column(
                column,
                justify="left" if column in ("Trade", "Contracts", "Note") else "right",
            )
        for p in plans:
            t = p.trade
            table.add_row(
                f"{t.symbol} {t.structure}",
                " / ".join(
                    f"{'+' if leg.sign > 0 else '-'}{leg.contract.localSymbol}"
                    for leg in p.legs
                )
                or "-",
                f"{p.unit_spx:+.3f}" if p.legs else "-",
                f"{t.weight:g}",
                str(p.units),
                p.note,
            )
        return table

    # ----------------------------------------------------------------- tables

    ZERO_DELTA_CAPTION = (
        "Targets for SPX delta = 0 AND delta $ (no beta) = 0: each symbol changes by "
        "the smallest proportional amount that zeroes both totals (weighted least "
        "squares), so the hedge holds even if betas are off. Weight = |beta delta $| / sum of "
        "|beta delta $|, so weights are positive and sum to 100%; long/short is the "
        "market side. At SPX delta = 0, long and short weights each sum to 50%. "
        "Share-equiv change = shares "
        "of stock, or option delta x 100 per contract, to add (+) or remove (-). "
        "Net target contracts adjusts the position with the most delta in that symbol, "
        "rounded to whole contracts and never flipping a position: a single option "
        "(+ buy, - sell), or a whole credit vertical (open/close N spreads, same width). "
        "Calendar/diagonal legs and options with delta < 0.01 are left alone; stock-only "
        "symbols trade shares. "
        "Suggestion only, no orders. If all symbols lean one way the rule flattens "
        "everything: use an index hedge instead."
    )

    def symbols_table(self) -> Table:
        """Managed symbols: net greeks of everything held in the symbol, and the
        SPX delta = 0 target weight."""
        table = Table(
            title="Managed symbols",
            caption=(
                "Greeks here are the net of all stock and option positions in the "
                "symbol (whole positions):\n"
                "  Delta: share-equivalents (stock = number of shares).\n"
                "  Gamma: how much that delta changes for a $1 move in the stock.\n"
                "  Theta: $ per day from time decay (positive means you earn it).\n"
                "  Vega: $ change for a 1-point rise in implied volatility.\n"
                "n/a: an option in the symbol has no greeks from IB."
            ),
            caption_justify="left",
        )
        for column in (
            "Symbol",
            "Delta",
            "Gamma",
            "Theta",
            "Vega",
            "Target weight (SPX Δ=0)",
        ):
            table.add_column(column, justify="left" if column == "Symbol" else "right")
        for symbol in self.config.portfolio.symbols:
            g = self.symbol_greeks.get(symbol)
            cells = (
                ["n/a"] * 4
                if g is None
                else [
                    f"{g[0]:+,.1f}",
                    f"{g[1]:+,.2f}",
                    dfmt(g[2], 0),
                    dfmt(g[3], 0),
                ]
            )
            table.add_row(symbol, *cells, self.zero_delta.get(symbol, ("n/a",) * 5)[1])
        return table

    def positions_summary_table(
        self, portfolio_positions: dict[str, list[PortfolioItem]]
    ) -> Table:
        """What is held per symbol, with the SPX delta = 0 targets and trades."""
        table = Table(
            title="Positions summary",
            show_edge=False,
            caption=self.ZERO_DELTA_CAPTION,
            caption_justify="left",
        )
        for column in (
            "Symbol",
            "Shares",
            "Short puts",
            "Long puts",
            "Short calls",
            "Long calls",
            "Weight now",
            "Target weight (SPX Δ=0)",
            "Target beta delta $",
            "Share-equiv change",
            "Net target contracts",
        ):
            table.add_column(column, justify="left" if column == "Symbol" else "right")
        for symbol in self.config.portfolio.symbols:
            positions = portfolio_positions.get(symbol, [])
            shares = sum(p.position for p in positions if isinstance(p.contract, Stock))
            table.add_row(
                symbol,
                ifmt(math.floor(shares)),
                ifmt(count_short_option_positions(positions, "P")),
                ifmt(count_long_option_positions(positions, "P")),
                ifmt(count_short_option_positions(positions, "C")),
                ifmt(count_long_option_positions(positions, "C")),
                *self.zero_delta.get(symbol, ("n/a",) * 5),
            )
            table.add_row(  # average strikes
                "",
                "",
                dfmt(weighted_avg_short_strike(positions, "P")),
                dfmt(weighted_avg_long_strike(positions, "P")),
                dfmt(weighted_avg_short_strike(positions, "C")),
                dfmt(weighted_avg_long_strike(positions, "C")),
            )
            table.add_section()
        return table
