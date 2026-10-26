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
from datetime import date
from typing import TYPE_CHECKING, Any

from ib_async import PortfolioItem
from ib_async.contract import Contract, Index, Stock
from ib_async.order import LimitOrder
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.table import Table

from thetagang import log
from thetagang.config import (
    Config,
)
from thetagang.fmt import dfmt, ifmt
from thetagang.ibkr import IBKR, TickerField
from thetagang.risk import (
    Adjuster,
    Book,
    DailyAttribution,
    Totals,
    aggregate_book,
    breakdown_table,
    compute_beta,
    contract_greeks,
    daily_pnl_by_symbol,
    greeks_table,
    option_volga,
    option_years,
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
from thetagang.strategies.discretionary_trades import Resize
from thetagang.trading_operations import (
    OptionChainScanner,
    OrderOperations,
)
from thetagang.util import (
    count_long_option_positions,
    count_short_option_positions,
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


class DiscretionaryRiskMixin:
    """Needs the DiscretionaryEngine attributes and helpers declared below."""

    config: Config
    ibkr: IBKR
    order_ops: OrderOperations
    option_scanner: OptionChainScanner
    qualified_contracts: dict[int, Contract]
    account_number: str
    planned_orders: list[tuple[Contract, LimitOrder]]
    resizes: list[Resize]
    resize_note: str
    zero_delta: dict[str, tuple[str, ...]]
    position_greeks: dict[int, tuple[float, float, float, float]]
    symbol_greeks: dict[str, tuple[float, float, float, float] | None]
    symbol_volga: dict[str, float]
    contract_greeks: dict[int, tuple[float, float, float, float]]
    beta_cache: dict[str, float]
    beta_cache_day: date | None
    spx_closes: dict[Any, float]
    symbol_closes: dict[str, dict[Any, float]]

    symbol_overrides: dict[str, Any]

    def symbol_config(self, symbol: str) -> Any:
        """[portfolio.symbols.X] for a symbol, held or not (discover_symbols keeps
        only held symbols in portfolio.symbols; overrides keep the rest)."""
        return self.config.portfolio.symbols.get(symbol) or self.symbol_overrides.get(
            symbol
        )

    if TYPE_CHECKING:  # implemented by DiscretionaryEngine / DiscretionarySpreadsMixin

        async def plan_trades(
            self, book: Book, betas: dict[str, float], spx_price: float
        ) -> tuple[Table | None, Totals]: ...

        async def _resize_orders(self, book: Book, spx_price: float) -> None: ...

        def held_positions(
            self, portfolio: list[PortfolioItem] | None = None
        ) -> list[PortfolioItem]: ...

        async def leg_price(self, contract: Contract, exchange: str) -> float: ...

        def split_option_legs(
            self, positions: list[PortfolioItem]
        ) -> tuple[
            list[tuple[PortfolioItem, PortfolioItem]], list[PortfolioItem], set[str]
        ]: ...

        def combo_order(
            self,
            symbol: str,
            legs: list[tuple[Contract, str]],
            qty: int,
            price: float,
            exchange: str,
        ) -> tuple[Contract, LimitOrder]: ...

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

        Verticals (credit or debit) are resized as whole spreads (same width); one with
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
        self.resize_note = ""
        self.resizes = []
        self.position_greeks, self.symbol_greeks, self.contract_greeks = {}, {}, {}
        items, spx_price, betas, greeks = await self.fetch_risk_inputs()
        book = aggregate_book(items, betas, greeks, spx_price)
        self.symbol_volga = book.volga_by_symbol
        self.position_greeks = position_greeks(items, greeks)
        self.symbol_greeks = symbol_greeks(items, self.position_greeks)
        self.contract_greeks = contract_greeks(items, greeks)
        after = self.zero_delta_proposals(book, items, betas, greeks, spx_price)
        plan_table, after_trades = await self.plan_trades(book, betas, spx_price)
        if self.config.strategies.discretionary.place_target_orders:
            await self._resize_orders(book, spx_price)
        if plan_table is not None:  # real orders: "After trades" shows these instead
            after = after_trades
        tables: list[RenderableType] = [
            greeks_table(book, after, spx_price),
            stress_table(book.legs),
            breakdown_table(
                book,
                betas,
                self.symbol_greeks,
                await self._daily_pnl(items, book.prices, spx_price, greeks),
            ),
        ]
        if plan_table is not None:
            tables.append(plan_table)
        return tables

    async def _daily_pnl(
        self,
        items: list[PortfolioItem],
        prices: dict[str, float],
        spx_price: float,
        greeks: dict[int, Any],
    ) -> DailyAttribution | None:
        """Daily P&L attribution inputs, or None when IB's daily P&L isn't
        available - display only, never blocks a run."""
        try:
            today = date.today()  # noqa: DTZ011
            previous = [
                close for day, close in sorted(self.spx_closes.items()) if day < today
            ]
            if not previous or not spx_price:
                return None
            daily = await self.ibkr.daily_pnl(
                self.account_number, [i.contract.conId for i in items]
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"Daily P&L by symbol unavailable: {e!r}")
            return None
        if not daily:
            log.warning("Daily P&L by symbol unavailable: IB reported no daily P&L")
            return None
        spx_move = spx_price / previous[-1] - 1
        return DailyAttribution(
            daily_pnl_by_symbol(items, daily),
            self._symbol_moves(prices, spx_move),
            spx_move,
            self._vega_by_symbol(items, greeks),
        )

    def _symbol_moves(
        self, prices: dict[str, float], spx_move: float
    ) -> dict[str, float]:
        """Actual % move today per symbol from the beta history's closes; falls
        back to the SPX move where a symbol has none (its beta also defaulted to
        1)."""
        today = date.today()  # noqa: DTZ011
        out: dict[str, float] = {}
        for symbol, price in prices.items():
            closes = self.symbol_closes.get(symbol)
            previous = (
                [close for day, close in sorted(closes.items()) if day < today]
                if closes
                else []
            )
            out[symbol] = price / previous[-1] - 1 if previous and price else spx_move
        return out

    def _vega_by_symbol(
        self, items: list[PortfolioItem], greeks: dict[int, Any]
    ) -> dict[str, float]:
        """Vega P&L ($) per symbol from each option's implied-vol change since
        the day's first observation. An option seen for the first time today is
        snapshotted at its current IV (no measured change yet); later runs use
        the recorded IV as the baseline. Without a data store the change is 0."""
        store = self.order_ops.data_store
        today = date.today().isoformat()  # noqa: DTZ011
        snapshot = store.get_iv_snapshot(today) if store else {}
        merged = dict(snapshot)
        out: dict[str, float] = {}
        for item in items:
            if item.contract.secType != "OPT":
                continue
            con_id = item.contract.conId
            g = greeks.get(con_id)
            if g is None or not valid_greeks(g):
                continue
            try:
                iv = float(g.impliedVol)
            except (TypeError, ValueError):
                continue
            if math.isnan(iv):
                continue
            prev = snapshot.get(con_id)
            if prev is None:
                merged[con_id] = iv  # first observation today
                continue
            position_vega = self.position_greeks.get(con_id)
            if position_vega is None:
                continue
            out[item.contract.symbol] = out.get(item.contract.symbol, 0.0) + (
                position_vega[3] * (iv - prev)
            )
        if store and len(merged) > len(snapshot):
            store.record_iv_snapshot(today, merged)
        return out

    async def risk_panel(self, account_table: Table) -> RenderableType:
        """Account summary next to the greeks table; stress P&L, the breakdown and
        the trade plan below."""
        self.planned_orders = []  # never carry a plan over from an earlier run
        try:
            greeks, scenarios, *rest = await self.portfolio_greeks_table()
            # account summary and greeks side by side; stress P&L below them
            return Group(Columns([account_table, greeks]), scenarios, *rest)
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
            closes = {b.date: b.close for b in bars}
            self.symbol_closes[symbol] = closes
            beta = compute_beta(closes, self.spx_closes)
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
            sconfig = self.symbol_config(symbol)
            if sconfig is not None and sconfig.no_trading:
                self.zero_delta[symbol] = (
                    *weights,
                    f"{shares:+,.0f}",
                    "none (no_trading)",
                )
                continue
            symbol_items = [i for i in items if i.contract.symbol == symbol]
            traded, trade, (gamma, theta, vega, volga), resize = (
                self._symbol_adjustment(
                    symbol_items, shares, greeks, betas[symbol], spx_price
                )
            )
            if resize is not None:
                self.resizes.append(resize)
            after.gamma += gamma
            after.theta += theta
            after.vega += vega
            after.volga += volga
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
    ) -> tuple[float, str, tuple[float, float, float, float], Resize | None]:
        """The trade toward `shares` of delta in one symbol: (share delta traded,
        description, gamma/theta/vega added, the trade itself). Resizes the
        position with the most delta; stock-only symbols trade shares; never flips
        a position."""
        symbol = symbol_items[0].contract.symbol
        adjusters = self.option_adjusters(symbol_items, greeks, beta, spx_price)
        if adjusters:
            main = max(adjusters, key=lambda a: a.exposure)
            units, trade = size_adjustment(main, shares)
            # volga per unit: long leg minus short leg for a spread
            signs = (-1, 1) if main.is_spread else (1,)
            unit_volga = sum(
                s
                * option_volga(
                    greeks[i.contract.conId], i.contract.strike, option_years(i)
                )
                * float(i.contract.multiplier or 100)
                for s, i in zip(signs, main.legs)
            )
            added = (
                units * main.gamma,
                units * main.theta,
                units * main.vega,
                units * unit_volga,
            )
            resize = Resize(symbol, units, adjuster=main) if units else None
            return units * main.delta, trade, added, resize
        stock = [i for i in symbol_items if i.contract.secType == "STK"]
        if stock:
            traded = stock_trade(shares, stock[0].position)
            # never sell the shares that cover short calls
            covering = sum(
                -i.position * float(i.contract.multiplier or 100)
                for i in symbol_items
                if i.contract.secType == "OPT"
                and i.contract.right.upper().startswith("C")
                and i.position < 0
            )
            traded = max(traded, -int(max(stock[0].position - covering, 0)))
            resize = Resize(symbol, traded, stock=stock[0]) if traded else None
            text = f"{traded:+d} shares" if traded else "0"
            return traded, text, (0.0, 0.0, 0.0, 0.0), resize
        reason = (
            "none (calendar/diagonal legs)"
            if self.has_calendar_or_diagonal_legs(symbol_items)
            else "none (delta < 0.01)"
        )
        return 0.0, reason, (0.0, 0.0, 0.0, 0.0), None

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
        "(+ buy, - sell), or a whole vertical (open/close N spreads, same width). "
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
                "  Volga: change in that vega ($) for a 1-point rise in implied volatility.\n"
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
            "Volga",
            "Target weight (SPX Δ=0)",
        ):
            table.add_column(column, justify="left" if column == "Symbol" else "right")
        for symbol in self.config.portfolio.symbols:
            g = self.symbol_greeks.get(symbol)
            cells = (
                ["n/a"] * 5
                if g is None
                else [
                    f"{g[0]:+,.1f}",
                    f"{g[1]:+,.2f}",
                    dfmt(g[2], 0),
                    dfmt(g[3], 0),
                    dfmt(self.symbol_volga.get(symbol, 0.0), 0),
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
            caption=(
                self.ZERO_DELTA_CAPTION.replace(
                    "Suggestion only, no orders.",
                    self.resize_note
                    or "place_target_orders is on, but there were no risk numbers "
                    "this run: no orders.",
                )
                if self.config.strategies.discretionary.place_target_orders
                else self.ZERO_DELTA_CAPTION
            ),
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
