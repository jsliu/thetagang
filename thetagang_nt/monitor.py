"""PortfolioMonitor: a read-only Nautilus Actor that turns the account into
Prometheus metrics for Grafana (deploy/grafana).

Every `interval_secs` it reads the net positions from the cache, quotes them,
computes SPX beta-weighted greeks and stress P&L, and swaps in a new metric
snapshot. Daily betas (1 year of daily closes vs SPX) are also written to the
cache under BETAS_KEY for the strategy."""

from __future__ import annotations

import json
from datetime import date, timedelta

from nautilus_trader.common.actor import Actor
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily

from thetagang_nt import aio
from thetagang_nt.config import MonitorConfig
from thetagang_nt.portfolio import (
    INDEX_CONTRACTS,
    SPX,
    VIX,
    VIXMO,
    Book,
    build_book,
    held_legs,
    is_index,
    price,
    stock_contract,
    stress_pnl,
    underlying_id,
)
from thetagang_nt.rules import Leg, compute_beta, short_pnl

BETAS_KEY = "thetagang.betas"


class _Snapshot:
    """Prometheus collector serving the last complete set of metrics, so a
    scrape never sees a half-built refresh."""

    def __init__(self) -> None:
        self.families: list[GaugeMetricFamily] = []

    def collect(self) -> list[GaugeMetricFamily]:
        return self.families


class _Families:
    def __init__(self) -> None:
        self.by_name: dict[str, GaugeMetricFamily] = {}

    def set(self, name: str, help_: str, value: float, **labels: str) -> None:
        fam = self.by_name.get(name)
        if fam is None:
            fam = self.by_name[name] = GaugeMetricFamily(
                f"thetagang_{name}", help_, labels=list(labels)
            )
        fam.add_metric(list(labels.values()), float(value))


def equity_symbols(symbols: set[str]) -> list[str]:
    return sorted(s for s in symbols if not is_index(s))


class PortfolioMonitor(Actor):
    def __init__(self, config: MonitorConfig) -> None:
        super().__init__(config)
        self._snapshot = _Snapshot()
        self._closes: dict[InstrumentId, dict[date, float]] = {}
        self._betas: dict[str, float] = {}
        self._beta_day: date | None = None
        self._busy = False

    def on_start(self) -> None:
        if self.config.publish:
            start_http_server(self.config.metrics_port, self.config.metrics_addr)
            REGISTRY.register(self._snapshot)
            self.log.info(
                f"Metrics on http://{self.config.metrics_addr}:"
                f"{self.config.metrics_port}/metrics"
            )
        aio.spawn(self, self._boot(), "monitor boot")

    def on_stop(self) -> None:
        if self.config.publish:
            REGISTRY.unregister(self._snapshot)
        self.clock.cancel_timers()

    async def _boot(self) -> None:
        await aio.load_contracts(self, INDEX_CONTRACTS)
        self.clock.set_timer(
            "monitor",
            timedelta(seconds=self.config.interval_secs),
            callback=self._on_timer,
        )
        await self._refresh()

    def _on_timer(self, _event: object) -> None:
        if not self._busy:  # a slow refresh (IB pacing) never piles up
            aio.spawn(self, self._refresh(), "monitor refresh")

    def on_historical_data(self, data: object) -> None:
        if isinstance(data, Bar):
            day = unix_nanos_to_dt(data.ts_event).date()
            closes = self._closes.setdefault(data.bar_type.instrument_id, {})
            closes[day] = data.close.as_double()

    # ------------------------------------------------------------- refresh

    async def _refresh(self) -> None:
        self._busy = True
        try:
            await self._update()
        finally:
            self._busy = False

    async def _update(self) -> None:
        today = self.clock.utc_now().date()
        symbols = {leg.symbol for leg in held_legs(self.cache)}
        unloaded = [
            s
            for s in equity_symbols(symbols)
            if self.cache.instrument(underlying_id(s)) is None
        ]
        if unloaded:
            await aio.load_contracts(
                self,
                [stock_contract(s) for s in unloaded],
            )
        ids = {leg.instrument_id for leg in held_legs(self.cache)}
        ids |= {SPX, VIX, VIXMO} | {underlying_id(s) for s in symbols}
        await aio.wait_for_prices(self, ids, timeout=10)
        legs = held_legs(self.cache)  # again, now with prices
        if self._beta_day != today:
            await self._load_betas(equity_symbols(symbols), today)
        if not self.config.publish:  # --role trade: the betas are all the strategy needs
            return

        f = _Families()
        for name, iid in (("spx", SPX), ("vix", VIX), ("vixmo", VIXMO)):
            value = price(self.cache, iid)
            if value is not None:
                f.set(f"{name}_price", f"{name.upper()} index level", value)
        self._account(f)
        self._positions(f, legs, today)
        spx = price(self.cache, SPX)
        if spx:
            book = build_book(self.cache, self.greeks, legs, self._betas, spx, today)
            self._risk(f, legs, book, today)
        f.set("orders_open", "Open orders", len(self.cache.orders_open()))
        f.set("last_update_timestamp_seconds", "Last refresh", self.clock.timestamp())
        self._snapshot.families = list(f.by_name.values())

    async def _load_betas(self, symbols: list[str], today: date) -> None:
        start = self.clock.utc_now() - timedelta(days=365)
        for iid in [SPX] + [underlying_id(s) for s in symbols]:
            if self.cache.instrument(iid) is None:
                continue  # IB couldn't resolve it: beta falls back to 1
            try:
                await aio.request_bars(
                    self, BarType.from_str(f"{iid}-1-DAY-LAST-EXTERNAL"), start
                )
            except (TimeoutError, ValueError) as e:
                self.log.warning(f"No daily history for {iid}: {e!r}")
        spx = self._closes.get(SPX, {})
        betas = {}
        for s in symbols:
            beta = compute_beta(self._closes.get(underlying_id(s), {}), spx)
            if beta is None:
                self.log.warning(f"{s}: beta defaulted to 1.0 (no usable history)")
            betas[s] = 1.0 if beta is None else beta
        if spx:  # only a successful SPX load counts as today's betas
            self._betas, self._beta_day = betas, today
            self.cache.add(BETAS_KEY, json.dumps(betas).encode())

    # ------------------------------------------------------------ metrics

    def _account(self, f: _Families) -> None:
        for acc in self.cache.accounts():
            total = acc.balance_total(USD)
            if total is None:
                continue
            a = acc.id.value
            f.set(
                "net_liquidation", "Net liquidation (USD)", total.as_double(), account=a
            )
            free, locked = acc.balance_free(USD), acc.balance_locked(USD)
            f.set(
                "available_funds", "Available funds (USD)", free.as_double(), account=a
            )
            f.set(
                "margin_used",
                "Net liq - available funds",
                locked.as_double(),
                account=a,
            )

    def _positions(self, f: _Families, legs: list[Leg], today: date) -> None:
        unrealized = 0.0
        for leg in legs:
            labels = {
                "instrument": leg.instrument_id.symbol.value,
                "symbol": leg.symbol,
                "kind": {"C": "call", "P": "put"}.get(leg.right, "stock"),
            }
            f.set("position_quantity", "Signed contracts or shares", leg.qty, **labels)
            f.set("position_avg_price", "Cost basis per share", leg.avg_px, **labels)
            if leg.mid is None:
                continue
            size = leg.qty * leg.multiplier
            pnl = (leg.mid - leg.avg_px) * size
            unrealized += pnl
            f.set("position_price", "Mid per share", leg.mid, **labels)
            f.set("position_unrealized_pnl", "Unrealized P&L (USD)", pnl, **labels)
            f.set(
                "position_market_value", "Market value (USD)", leg.mid * size, **labels
            )
            if not leg.is_option:
                continue
            f.set("position_dte", "Days to expiry", leg.dte(today), **labels)
            if leg.qty < 0:
                ratio = short_pnl(leg)
            else:
                ratio = (leg.mid - leg.avg_px) / leg.avg_px if leg.avg_px else None
            if ratio is not None:
                f.set(
                    "position_pnl_ratio", "P&L as a share of premium", ratio, **labels
                )
        f.set("unrealized_pnl", "Unrealized P&L of priced positions", unrealized)

    def _risk(self, f: _Families, legs: list[Leg], book: Book, today: date) -> None:
        f.set("spx_delta", "SPX beta-weighted delta (SPX units)", book.spx_delta)
        f.set("beta_delta_dollars", "Beta-weighted delta $", book.beta_dollars)
        f.set("delta_dollars", "Delta $ without beta", book.plain_dollars)
        f.set("gamma_spx", "SPX delta change per SPX point", book.gamma)
        f.set("theta", "Theta ($/day)", book.theta)
        f.set("vega", "Vega ($ per vol point)", book.vega)
        f.set("options_missing_greeks", "Positions without greeks", len(book.missing))
        for s, r in book.by_symbol.items():
            f.set("symbol_beta", "1y beta vs SPX", r.beta, symbol=s)
            f.set(
                "symbol_delta_dollars",
                "Delta $ without beta",
                r.plain_dollars,
                symbol=s,
            )
            f.set("symbol_beta_delta_dollars", "Beta delta $", r.beta_dollars, symbol=s)
            f.set(
                "symbol_delta_shares",
                "Share-equivalent delta",
                r.delta_shares,
                symbol=s,
            )
            f.set("symbol_gamma", "Share delta per $1 move", r.gamma, symbol=s)
            f.set("symbol_theta", "Theta ($/day)", r.theta, symbol=s)
            f.set("symbol_vega", "Vega ($ per vol point)", r.vega, symbol=s)
        alert = self.config.long_option_alert_delta
        alerts = 0
        for leg in legs:
            g = book.greeks.get(leg.instrument_id)
            if g is None:
                continue
            name = leg.instrument_id.symbol.value
            f.set("position_delta", "Option delta per share", g.delta, instrument=name)
            f.set("position_iv", "Implied volatility", g.vol, instrument=name)
            alerts += leg.qty > 0 and abs(g.delta) >= alert
        f.set("long_option_alerts", f"Long options with |delta| >= {alert}", alerts)
        for move in self.config.stress_moves:
            for basis, use_beta in (("beta", True), ("one", False)):
                f.set(
                    "stress_pnl",
                    "P&L if SPX moves (options repriced at current IV)",
                    stress_pnl(self.cache, legs, book, move, use_beta, today),
                    move=f"{move:+.0%}",
                    betas=basis,
                )
