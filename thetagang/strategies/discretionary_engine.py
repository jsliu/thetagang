"""Discretionary strategy: manage the option positions already held and open the
trades listed in strategies.discretionary.trades.

Enabled with run.strategies = ["discretionary"]. It:
  - manages every USD stock/option symbol held (portfolio.symbols are overrides),
  - rolls/closes credit verticals as whole spreads (4-leg / 2-leg combos) with the
    same [roll_when] rules as standalone options, and long (debit) verticals as
    whole spreads with their own take-profit / roll rules; calendar/diagonal legs
    are left alone, except short calls fully covered by shares,
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
from ib_async.contract import Contract, Index, Stock
from ib_async.order import LimitOrder
from rich.table import Table

from thetagang import log
from thetagang.config import Config
from thetagang.fmt import contract_label, dfmt, ifmt, pfmt
from thetagang.ibkr import IBKR
from thetagang.options import option_dte
from thetagang.strategies.discretionary_risk import DiscretionaryRiskMixin
from thetagang.strategies.discretionary_spreads import (
    DiscretionarySpreadsMixin,
    roll_strike_limit,
)
from thetagang.strategies.discretionary_trades import (
    ENTRY_ORDER_REF,
    MANAGE_ORDER_REF,
    DiscretionaryTradesMixin,
    quoted_price,
)
from thetagang.trading_operations import (
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
    for column in (
        "Contract",
        "Qty",
        "DTE",
        "Price",
        "Unrealized P&L",
        "P&L",
        "Action",
    ):
        table.add_column(column, justify="left" if column == "Contract" else "right")
    for p in hedges:
        table.add_row(
            contract_label(p.contract),
            ifmt(int(p.position)),
            str(option_dte(p.contract.lastTradeDateOrContractMonth)),
            dfmt(p.marketPrice),
            dfmt(p.unrealizedPNL, 0),  # same as the Portfolio positions table
            pfmt(position_pnl(p), 1),
            "[red]Close" if close else "Keep",
        )
    return table


def options_table(
    kind: str,
    shorts: list[PortfolioItem],
    rollable: list[PortfolioItem],
    closeable: list[PortfolioItem],
    longs: list[PortfolioItem],
    deltas: dict[int, float],
    take_profit: float | None,
    alert_delta: float,
) -> Table:
    """Standalone puts or calls (spread legs are in the Spreads table): shorts
    with what this run does and the take-profit buy-back price; longs with a
    roll-or-close reminder once |delta| >= alert_delta (no orders for longs)."""
    rolled = {id(p) for p in rollable}
    closed = {id(p) for p in closeable}

    def count(items: list[PortfolioItem]) -> int:
        return math.floor(sum(abs(p.position) for p in items))

    alerts = 0
    table = Table(title=f"{kind.capitalize()} (standalone)", caption_justify="left")
    for column in (
        "Contract",
        "Side",
        "Qty",
        "DTE",
        "Delta",
        "P&L",
        "Take profit at",
        "Action",
    ):
        table.add_column(column, justify="left" if column == "Contract" else "right")
    for p in sorted(
        shorts + longs, key=lambda p: (p.position > 0, contract_label(p.contract))
    ):
        delta = deltas.get(p.contract.conId)
        if p.position < 0:
            premium = abs(p.averageCost) / float(p.contract.multiplier or 100)
            target = (
                f"buy <= {dfmt(premium * (1 - take_profit))}"
                if premium and take_profit
                else "n/a"
            )
            action = (
                "[blue]Roll"
                if id(p) in rolled
                else "[green]Close"
                if id(p) in closed
                else "Keep"
            )
        else:
            target = "-"
            if delta is not None and abs(delta) >= alert_delta:
                alerts += 1
                action = f"[yellow]Roll or close by hand (|Δ| >= {alert_delta:g})"
            else:
                action = "Hold"
        table.add_row(
            contract_label(p.contract),
            "short" if p.position < 0 else "long",
            ifmt(int(abs(p.position))),
            str(option_dte(p.contract.lastTradeDateOrContractMonth)),
            "n/a" if delta is None else f"{delta:+.2f}",
            pfmt(position_pnl(p), 1),
            target,
            action,
        )
    table.caption = (
        f"Shorts: {count(rollable)} to roll, {count(closeable)} to close, "
        f"{count(shorts) - count(rollable) - count(closeable)} kept (contracts). "
        f"Longs: {alerts} to roll or close by hand (|delta| >= {alert_delta:g}, "
        "long_option_alert_delta). P&L: of the premium. Take profit at: buy the "
        "short back at or below this price (per share) - the bot closes at "
        + (f"{take_profit:.0%}" if take_profit else "n/a (unset)")
        + " of the premium (roll_when.close_at_pnl)."
    )
    return table


class DiscretionaryEngine(
    DiscretionarySpreadsMixin, DiscretionaryTradesMixin, DiscretionaryRiskMixin
):
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
        # "Net target contracts" trades of this run (orders if place_target_orders)
        self.resizes: list[Any] = []
        self.resize_note = ""  # what place_target_orders did this run (caption)
        self.spread_rows: list[tuple[str, ...]] = []  # Spreads table, per run
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
        # per-symbol daily closes (from the beta history), used to turn today's
        # moves into a greek P&L attribution
        self.symbol_closes: dict[str, dict[Any, float]] = {}

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

    def standalone_longs(
        self, portfolio_positions: dict[str, list[PortfolioItem]], right: str
    ) -> list[PortfolioItem]:
        """Long puts or calls not in a vertical or calendar/diagonal (VIX hedges
        have their own table)."""
        longs: list[PortfolioItem] = []
        for symbol, positions in portfolio_positions.items():
            if symbol == "VIX":
                continue
            _, unpaired, mixed = self.split_option_legs(positions)
            longs += [
                p
                for p in unpaired
                if p.position > 0
                and p.contract.right.upper().startswith(right)
                and p.contract.right not in mixed
            ]
        return longs

    async def run_post_stage(
        self, portfolio_positions: dict[str, list[PortfolioItem]]
    ) -> None:
        """post_discretionary stage: credit and debit verticals, VIX hedges, then
        the orders planned while building the risk tables."""
        self.spread_rows = []
        await self.roll_spreads(portfolio_positions)
        await self.manage_debit_spreads(portfolio_positions)
        table = self.spreads_table()
        if table is None:
            log.info("No credit or debit spreads held")
        else:
            log.print(table)
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
