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
from thetagang_nt.config import DiscretionaryConfig, MonitorConfig
from thetagang_nt.ib_extra import AccountExtras
from thetagang_nt.portfolio import (
    INDEX_CONTRACTS,
    SPX,
    VIX,
    VIX_ROOT,
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
from thetagang_nt.rules import (
    Action,
    Leg,
    RollWhen,
    Unit,
    compute_beta,
    leg_name,
    option_label,
    position_actions,
    short_pnl,
    side_weight,
    symbol_adjustment,
    vix_hedge_action,
    weighted_strike,
    zero_delta_targets,
)

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

    def set(self, name: str, help_: str, value: float, /, **labels: str) -> None:
        # positional-only: a label may itself be called `name`
        fam = self.by_name.get(name)
        if fam is None:
            fam = self.by_name[name] = GaugeMetricFamily(
                f"thetagang_{name}", help_, labels=list(labels)
            )
        fam.add_metric(list(labels.values()), float(value))


def take_profit_text(a: Action) -> str:
    """As the old bot printed it: "buy <= $0.10", "sell >= $4.40", "n/a", "-"."""
    if a.kind == "long" or a.kind == "calendar/diagonal":
        return "-"
    if a.take_profit is None:
        return "n/a"
    return f"{a.take_profit_side} ${a.take_profit:,.2f}"


def action_text(action: str) -> str:
    """ "roll or close by hand (|delta| >= 0.8)" -> "Roll or close by hand (|Δ| >= 0.8)"."""
    action = action.replace("|delta|", "|Δ|")
    return action[:1].upper() + action[1:]


def equity_symbols(symbols: set[str]) -> list[str]:
    return sorted(s for s in symbols if not is_index(s))


class PortfolioMonitor(Actor):
    """`rules`: the [strategy] settings, so the dashboard shows what the strategy
    would do (it never trades here). `extras`: the read-only IB connection for
    account fields and daily P&L Nautilus doesn't expose."""

    def __init__(
        self,
        config: MonitorConfig,
        rules: DiscretionaryConfig | None = None,
        extras: AccountExtras | None = None,
    ) -> None:
        super().__init__(config)
        self._snapshot = _Snapshot()
        self._closes: dict[InstrumentId, dict[date, float]] = {}
        self._betas: dict[str, float] = {}
        self._beta_day: date | None = None
        self._busy = False
        self._rules = rules
        self._extras = extras
        self._iv_open: tuple[date | None, dict[InstrumentId, float]] = (None, {})

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
        if self._extras is not None:
            self._extras.close()
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
        if (
            not self.config.publish
        ):  # --role trade: the betas are all the strategy needs
            return

        summary, daily = await self._read_extras()
        f = _Families()
        for field, value in summary.items():
            f.set(
                "account_value",
                "Account value from IB (USD; cushion a fraction)",
                value,
                field=field,
            )
        for name, iid in (("spx", SPX), ("vix", VIX), ("vixmo", VIXMO)):
            value = price(self.cache, iid)
            if value is not None:
                f.set(f"{name}_price", f"{name.upper()} index level", value)
        self._account(f)
        self._positions(f, legs, today)
        spx = price(self.cache, SPX)
        self._summary(f, legs)
        if spx:
            book = build_book(self.cache, self.greeks, legs, self._betas, spx, today)
            self._risk(f, legs, book, today)
            self._proposals(f, legs, book, spx)
            self._actions(f, legs, book, today)
            if daily:
                self._attribution(f, legs, book, daily, spx, today)
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
                "instrument": option_label(leg),
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
        f.set("volga", "Volga (vega $ change per vol point)", book.volga)
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
            f.set("symbol_volga", "Volga (vega $ per vol point)", r.volga, symbol=s)
        alert = self.config.long_option_alert_delta
        alerts = 0
        for leg in legs:
            g = book.greeks.get(leg.instrument_id)
            if g is None:
                continue
            name = option_label(leg)
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

    # ------------------------------------------- what the old bot also showed

    async def _read_extras(self) -> tuple[dict[str, float], dict[str, float]]:
        if self._extras is None:
            return {}, {}
        try:
            return await self._extras.read()
        except Exception as e:  # noqa: BLE001 - only these fields are lost
            self.log.warning(f"Account extras unavailable: {e!r}")
            self._extras.close()  # reconnect next refresh
            return {}, {}

    def _summary(self, f: _Families, legs: list[Leg]) -> None:
        """Positions summary: shares, option counts and average strikes."""
        by_symbol: dict[str, list[Leg]] = {}
        for leg in legs:
            by_symbol.setdefault(leg.symbol, []).append(leg)
        for s, items in by_symbol.items():
            fields = {
                "shares": sum(x.qty for x in items if not x.is_option),
                "short_puts": sum(
                    -x.qty for x in items if x.right == "P" and x.qty < 0
                ),
                "long_puts": sum(x.qty for x in items if x.right == "P" and x.qty > 0),
                "short_calls": sum(
                    -x.qty for x in items if x.right == "C" and x.qty < 0
                ),
                "long_calls": sum(x.qty for x in items if x.right == "C" and x.qty > 0),
                "avg_short_put_strike": weighted_strike(items, "P", short=True),
                "avg_long_put_strike": weighted_strike(items, "P", short=False),
                "avg_short_call_strike": weighted_strike(items, "C", short=True),
                "avg_long_call_strike": weighted_strike(items, "C", short=False),
            }
            for field, value in fields.items():
                if value is not None:
                    f.set(
                        "symbol_summary",
                        "Positions summary per symbol",
                        value,
                        symbol=s,
                        field=field,
                    )

    def _units(
        self, legs: list[Leg], book: Book, spx: float
    ) -> dict[InstrumentId, Unit]:
        """Per contract held long: share delta, SPX gamma, theta, vega."""
        out = {}
        for leg in legs:
            g, row = book.greeks.get(leg.instrument_id), book.by_symbol.get(leg.symbol)
            if g is None or row is None:
                continue
            m = leg.multiplier
            out[leg.instrument_id] = (
                g.delta * m,
                g.gamma * m * (row.beta * g.underlying / spx) ** 2,
                g.theta * m,
                g.vega * m,
                g.volga * m,
            )
        return out

    def _no_trading(self, symbol: str) -> bool:
        sym = self._rules.symbols.get(symbol) if self._rules else None
        return bool(sym and sym.no_trading)

    def _proposals(self, f: _Families, legs: list[Leg], book: Book, spx: float) -> None:
        """SPX delta = 0 targets per symbol and the trade toward them (display
        only, as in the old positions summary), and the greeks after them."""
        unit = self._units(legs, book, spx)
        plain_now = {s: r.plain_dollars for s, r in book.by_symbol.items()}
        betas = {s: r.beta for s, r in book.by_symbol.items()}
        plain_target = zero_delta_targets(plain_now, betas)
        target = {s: betas[s] * v for s, v in plain_target.items()}
        gross_now = sum(abs(r.beta_dollars) for r in book.by_symbol.values()) or 1.0
        gross_target = sum(abs(v) for v in target.values()) or 1.0
        after = {
            "beta_delta_dollars": book.beta_dollars,
            "delta_dollars": book.plain_dollars,
            "gamma": book.gamma,
            "theta": book.theta,
            "vega": book.vega,
            "volga": book.volga,
        }
        for s, r in book.by_symbol.items():
            w_now, side_now = side_weight(r.beta_dollars, gross_now)
            w_target, side_target = side_weight(target[s], gross_target)
            for field, value in (
                ("weight_now", w_now),
                ("target_weight", w_target),
                ("target_beta_delta_dollars", target[s]),
            ):
                f.set(
                    "symbol_summary",
                    "Positions summary per symbol",
                    value,
                    symbol=s,
                    field=field,
                )
            if r.price <= 1e-9:
                trade = "n/a"
            else:
                shares = (plain_target[s] - plain_now[s]) / r.price
                f.set(
                    "symbol_summary",
                    "Positions summary per symbol",
                    shares,
                    symbol=s,
                    field="share_equiv_change",
                )
                if self._no_trading(s):
                    trade = "none (no_trading)"
                else:
                    traded, trade, (gamma, theta, vega, volga_) = symbol_adjustment(
                        [x for x in legs if x.symbol == s], shares, unit
                    )
                    after["gamma"] += gamma
                    after["theta"] += theta
                    after["vega"] += vega
                    after["volga"] += volga_
                    after["delta_dollars"] += traded * r.price
                    after["beta_delta_dollars"] += traded * r.price * r.beta
            f.set(
                "symbol_target",
                "SPX delta = 0 target per symbol (display only)",
                1,
                symbol=s,
                side_now=side_now,
                side_target=side_target,
                trade=trade,
            )
        now = {
            "beta_delta_dollars": book.beta_dollars,
            "delta_dollars": book.plain_dollars,
            "gamma": book.gamma,
            "theta": book.theta,
            "vega": book.vega,
            "volga": book.volga,
        }
        for scenario, values in (("now", now), ("after_proposals", after)):
            f.set(
                "greeks",
                "Portfolio greeks by scenario",
                values["beta_delta_dollars"] / spx,
                greek="spx_delta",
                scenario=scenario,
            )
            for greek, value in values.items():
                f.set(
                    "greeks",
                    "Portfolio greeks by scenario",
                    value,
                    greek=greek,
                    scenario=scenario,
                )

    def _actions(self, f: _Families, legs: list[Leg], book: Book, today: date) -> None:
        """What the strategy's rules would do with each position, and the price
        that takes profit; the VIX hedge's action. Uses [strategy] settings."""
        rules = self._rules
        rw = rules.roll_when.rules() if rules else RollWhen()
        debit_tp = rules.debit_spread_take_profit if rules else 0.8
        deltas = {iid: g.delta for iid, g in book.greeks.items()}
        label_delta = {
            option_label(x): deltas[x.instrument_id]
            for x in legs
            if x.instrument_id in deltas
        }
        by_symbol: dict[str, list[Leg]] = {}
        for leg in legs:
            if leg.is_option and leg.symbol != VIX_ROOT:
                by_symbol.setdefault(leg.symbol, []).append(leg)
        for s, items in by_symbol.items():
            und = price(self.cache, underlying_id(s))
            for a in position_actions(
                items,
                und,
                rw,
                debit_tp,
                self.config.long_option_alert_delta,
                deltas,
                today,
                self._no_trading(s),
            ):
                spread = a.kind.endswith("spread")
                table = "spreads" if spread else ("calls" if a.right == "C" else "puts")
                if spread:
                    side = a.kind.split()[0]  # credit / debit
                else:
                    side = "short" if a.qty < 0 else "long"
                f.set(
                    "action_info",
                    "What the strategy's rules would do (the old bot's tables)",
                    1,
                    name=a.name,
                    symbol=a.symbol.lstrip("^"),
                    table=table,
                    side=side,
                    take_profit=take_profit_text(a),
                    action=action_text(a.action),
                )
                for field, value in (
                    ("qty", abs(a.qty)),
                    ("dte", a.dte),
                    ("pnl_ratio", a.pnl),
                    ("delta", None if spread else label_delta.get(a.name)),
                ):
                    if value is not None:
                        f.set(
                            "action",
                            "Position or spread figures",
                            value,
                            name=a.name,
                            table=table,
                            field=field,
                        )
        if rules is None:
            return
        cfg = rules.vix_call_hedge
        held = sum(
            x.qty
            for x in legs
            if x.symbol == VIX_ROOT
            and x.right == "C"
            and x.qty > 0
            and x.dte(today) > cfg.ignore_dte
        )
        action, weight = vix_hedge_action(
            held, price(self.cache, VIX), price(self.cache, VIXMO), cfg
        )
        f.set("vix_hedge_action", "What the VIX call hedge would do", 1, action=action)
        f.set("vix_hedge", "VIX call hedge", held, field="held_calls")
        f.set("vix_hedge", "VIX call hedge", weight, field="weight")
        if cfg.close_hedges_when_vix_exceeds:
            f.set(
                "vix_hedge",
                "VIX call hedge",
                cfg.close_hedges_when_vix_exceeds,
                field="close_above",
            )

    def _previous_close(self, iid: InstrumentId, today: date) -> float | None:
        closes = self._closes.get(iid, {})
        days = [d for d in closes if d < today]
        return closes[max(days)] if days else None

    def _attribution(
        self,
        f: _Families,
        legs: list[Leg],
        book: Book,
        daily: dict[str, float],
        spx: float,
        today: date,
    ) -> None:
        """Today's P&L by symbol (IB's daily P&L) split into what SPX explains
        and the greeks: delta, gamma, theta, vega and the residual."""
        spx_prev = self._previous_close(SPX, today)
        if not spx_prev:
            return
        spx_move = spx / spx_prev - 1
        f.set("spx_move_today", "SPX move since the last close", spx_move)
        day, iv_open = self._iv_open
        if day != today:  # implied vols seen first today are the baseline
            self._iv_open = (today, {})
            iv_open = self._iv_open[1]
        pnl: dict[str, float] = {}
        vega: dict[str, float] = {}
        for leg in legs:
            name = leg_name(leg)
            if name in daily:
                pnl[leg.symbol] = pnl.get(leg.symbol, 0.0) + daily[name]
            g = book.greeks.get(leg.instrument_id)
            if g is not None:
                base = iv_open.setdefault(leg.instrument_id, g.vol)
                # vega is $ per vol point (0.01) per share; vols are fractions
                vega[leg.symbol] = (
                    vega.get(leg.symbol, 0.0)
                    + g.vega * leg.qty * leg.multiplier * (g.vol - base) * 100
                )
        total: dict[str, float] = {}
        for s, r in book.by_symbol.items():
            if s not in pnl:
                continue
            prev = self._previous_close(underlying_id(s), today)
            move = r.price / prev - 1 if prev and r.price else spx_move
            parts = {"pnl": pnl[s], "from_spx": r.beta_dollars * spx_move}
            parts["unexplained"] = parts["pnl"] - parts["from_spx"]
            parts["delta"] = r.plain_dollars * move
            parts["gamma"] = 0.5 * r.gamma * (r.price * move) ** 2
            parts["theta"] = r.theta
            parts["vega"] = vega.get(s, 0.0)
            parts["residual"] = (
                parts["pnl"]
                - parts["delta"]
                - parts["gamma"]
                - parts["theta"]
                - parts["vega"]
            )
            for part, value in parts.items():
                f.set(
                    "symbol_pnl",
                    "Today's P&L by symbol and its attribution",
                    value,
                    symbol=s,
                    part=part,
                )
                total[part] = total.get(part, 0.0) + value
        for part, value in total.items():
            f.set(
                "symbol_pnl",
                "Today's P&L by symbol and its attribution",
                value,
                symbol="Total",
                part=part,
            )
