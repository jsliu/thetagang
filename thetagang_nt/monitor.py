"""PortfolioMonitor: a read-only Nautilus Actor that turns the account into
Prometheus metrics for Grafana (deploy/grafana).

Every `interval_secs` it reads the net positions from the cache, quotes them,
computes SPX beta-weighted greeks and stress P&L, and swaps in a new metric
snapshot. Daily betas (1 year of daily closes vs SPX) are also written to the
cache under BETAS_KEY for the strategy."""

from __future__ import annotations

import asyncio
import json
import math
import traceback
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

from nautilus_trader.common.actor import Actor
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OptionKind
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import OptionContract
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily

from thetagang_nt import aio
from thetagang_nt.config import DiscretionaryConfig, MonitorConfig, PMConfig
from thetagang_nt.ib_extra import AccountExtras
from thetagang_nt.journal import (
    ForecastRow,
    IvRow,
    Journal,
    LegMark,
    LegRow,
    RollForecast,
    StrategyRow,
)
from thetagang_nt.pm.core.american import schedule
from thetagang_nt.pm.core.assess import assess
from thetagang_nt.pm.core.iv_model import IvHistory, IvModel, build
from thetagang_nt.pm.core.profit import capture_ratio, value_bounds
from thetagang_nt.pm.core.quality import Quote, is_wide, leg_flags, wide_legs
from thetagang_nt.pm.core.review import (
    P0,
    P1,
    P2,
    SHORT_PREMIUM,
    ShortLeg,
    StrategyState,
    manager,
)
from thetagang_nt.pm.core.rolls import (
    Candidate,
    compare,
    evaluate,
    generate,
    target_dtes,
)
from thetagang_nt.pm.core.scenarios import RATE as SCENARIO_RATE
from thetagang_nt.pm.core.scenarios import PriceModel, repriced
from thetagang_nt.pm.core.scenarios import grid as scenario_grid
from thetagang_nt.pm.core.strategy_id import Seen, auto_ids
from thetagang_nt.pm.core.surface import Smile, measure, sticky_gaps, strikes_around
from thetagang_nt.pm.core.valuation import forward
from thetagang_nt.pm.long_options import signals as long_options
from thetagang_nt.pm.long_options import vix as long_vix
from thetagang_nt.pm.short_options import signals as short_options
from thetagang_nt.pm.short_options.analysis import loss_multiple, premium_to_risk
from thetagang_nt.portfolio import (
    INDEX_CONTRACTS,
    SPX,
    VENUE,
    VIX,
    VIX_ROOT,
    VIXMO,
    Book,
    build_book,
    contract_units,
    curve,
    held_legs,
    is_index,
    net_liquidation,
    option_greeks,
    order_cost,
    order_text,
    parity_partner,
    price,
    proposal_legs,
    stock_contract,
    stress_pnl,
    symbols_with_open_orders,
    target_plan,
    underlying_id,
    vix_curve,
)
from thetagang_nt.rules import (
    BY_HAND,
    NEUTRALITY,
    Action,
    Leg,
    RollWhen,
    breakeven,
    breakeven_move,
    compute_beta,
    delta_band_dollars,
    expected_move,
    find_structures,
    gamma_pnl,
    leg_name,
    neutrality,
    option_label,
    outside_groups,
    position_actions,
    position_rows,
    realized_move,
    recommend,
    short_pnl,
    side_weight,
    structure_figures,
    structure_label,
    symbol_adjustment,
    vix_hedge_action,
    weighted_strike,
    zero_delta_targets,
)

BETAS_KEY = "thetagang.betas"
ROLL_QUOTE_POLLS = 20  # 0.25 s apart: how long a batch of candidates may take


def position_labels(leg: Leg) -> dict[str, str]:
    """The labels every per-position metric shares, so the table joins them."""
    return {
        "instrument": option_label(leg),
        "symbol": leg.symbol,
        "kind": {"C": "call", "P": "put"}.get(leg.right, "stock"),
    }


def publish_grid(f: _Families, sid: str, grid: Any, spec: Any) -> None:
    """The fixed-move grid cells, stress scenarios and per-horizon summaries
    (scenario_engine, stress_testing) of one strategy."""
    for (h, move, iv), pnl in grid.fixed.items():
        f.set(
            "pm_grid",
            "Scenario P&L ($): horizon (trading days), price move, IV shift (points)",
            pnl,
            strategy=sid,
            horizon=str(h),
            move=f"{move:g}",  # numbers, so the dashboard sorts them
            iv=f"{iv:g}",
        )
    for name, pnl in grid.stress.items():
        f.set("pm_stress", "Stress scenario P&L ($)", pnl, strategy=sid, name=name)
    step = sorted(x for x in spec.iv_points if x > 0)[0]
    for h in (1, 5, 10):
        for stat, value in grid.summary(h, step).items():
            f.set(
                "pm_grid_summary",
                "Scenario grid summary per horizon ($; counts; ratio)",
                value,
                strategy=sid,
                horizon=str(h),
                stat=stat,
            )


def short_leg(
    p: Leg,
    today: date,
    spot: float | None,
    quote: Quote | None,
    dividends: tuple[tuple[date, float], ...],
) -> ShortLeg:
    """A short leg, with what the ex-dividend check needs for a call."""
    coming = [
        (when, amount)
        for when, amount in dividends
        if today <= when and p.expiry and when < p.expiry
    ]
    extrinsic = None
    if spot and quote and quote.bid is not None and quote.ask is not None:
        intrinsic = (
            max(spot - p.strike, 0.0) if p.right == "C" else max(p.strike - spot, 0.0)
        )
        extrinsic = max((quote.bid + quote.ask) / 2 - intrinsic, 0.0)
    return ShortLeg(
        p.dte(today),
        p.strike,
        p.right,
        coming[0][1] if coming else None,
        (coming[0][0] - today).days if coming else None,
        extrinsic,
    )


def iv_rows(
    legs: Any, greeks: dict, quotes: dict, now: datetime, limits: Any
) -> list[IvRow]:
    """Implied vols worth keeping: no leg with a data flag, no bid or a wide
    spread, because one bad IV would corrupt the IV-change distribution."""
    out = []
    for p in legs:
        g = greeks.get(p.instrument_id)
        quote = quotes.get(p.instrument_id)
        und = getattr(g, "underlying", None)
        if leg_flags(p, quote, g, now, limits) or not und or not p.expiry:
            continue
        if not quote.bid or is_wide(quote, limits):  # the mid is a guess
            continue
        out.append(
            IvRow(
                str(p.instrument_id),
                p.symbol.lstrip("^"),
                p.expiry,
                p.strike,
                p.right,
                g.vol,
                und,
            )
        )
    return out


@dataclass(frozen=True)
class RollJob:
    """What a roll scan needs about one strategy not on HOLD."""

    symbol: str
    structure: Any
    underlying: float
    sigma_f: float | None
    front_dte: int
    initial_dte: int | None
    max_loss: float | None
    vix: bool
    iv_model: IvModel | None = None
    price_model: PriceModel | None = None


@dataclass(frozen=True)
class RollScan:
    at: datetime
    candidates: list[Candidate]


def publish_rolls(
    f: _Families,
    sid: str,
    u_hold: float | None,
    u_close: float | None,
    scan: RollScan | None,
    margin: float,
    now: datetime,
    u_hold_iv: float | None = None,
) -> tuple[str, str]:
    """HOLD / CLOSE / ROLL (roll_rules): the model's view of one strategy and
    its scanned candidates, compared with today's U_hold and U_close."""
    found = scan.candidates if scan else []
    choice = compare(u_hold, u_close, found, margin)
    with_iv = compare(
        u_hold_iv, u_close, [replace(c, u_new=c.u_new_iv) for c in found], margin
    )
    f.set(
        "pm_view",
        "The model's HOLD / CLOSE / ROLL view (reported, not a signal)",
        1,
        strategy=sid,
        view=choice.view,
        view_iv=with_iv.view,
        best=choice.best.label if choice.best else "-",
        best_iv=with_iv.best.label if with_iv.best else "-",
        scanned="yes" if scan else "no",
    )
    if scan:
        f.set(
            "pm_roll_scan_age",
            "Seconds since this strategy's roll candidates were quoted",
            (now - scan.at).total_seconds(),
            strategy=sid,
        )
    for c in found:
        f.set(
            "pm_roll_info",
            "A roll candidate; why it can't be recommended, if so",
            1,
            strategy=sid,
            candidate=c.label,
            fails=", ".join(c.fails) or "-",
        )
        u_roll = u_roll_iv = None
        if c.u_new is not None and u_close is not None:
            u_roll = c.u_new + u_close
        if c.u_new_iv is not None and u_close is not None:
            u_roll_iv = c.u_new_iv + u_close
        for field, value in (
            ("net_open", c.net_open),
            ("open_cost", c.open_cost),
            ("wcl_5", c.wcl.get(5)),
            ("wcl_10", c.wcl.get(10)),
            ("stress_loss", c.stress_loss),
            ("e_pnl", c.e_pnl),
            ("es", c.es),
            ("u_roll", u_roll),
            ("u_roll_iv", u_roll_iv),
            (
                "roll_score_iv",
                u_roll_iv - u_hold_iv
                if u_roll_iv is not None and u_hold_iv is not None
                else None,
            ),
            (
                "roll_score",
                u_roll - u_hold if u_roll is not None and u_hold is not None else None,
            ),
        ):
            if value is not None:
                f.set(
                    "pm_roll",
                    "A roll candidate's figures ($)",
                    value,
                    strategy=sid,
                    candidate=c.label,
                    field=field,
                )
    return choice.view, with_iv.view


def reliable_for_the_day(flags: tuple[str, ...]) -> bool:
    """The forward test keeps the day's last good values: after the close quotes
    go stale or vanish, and those must not overwrite them. A zero bid is real."""
    return not any("stale" in x or "missing" in x or "no greeks" in x for x in flags)


def roll_forecasts(sid: str, scan: RollScan) -> list[RollForecast]:
    return [
        RollForecast(
            sid,
            c.label,
            scan.at,
            c.net_open,
            c.open_cost,
            c.wcl.get(5),
            c.wcl.get(10),
            c.stress_loss,
            c.e_pnl,
            c.es,
            c.u_new,
            c.u_new_iv,
            ", ".join(c.fails) or "-",
            [
                {"contract": str(p.instrument_id), "qty": p.qty, "price": p.avg_px}
                for p in c.legs
            ],
        )
        for c in scan.candidates
    ]


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
    if a.kind in BY_HAND or a.kind in ("calendar/diagonal", "group"):
        return "-"
    if a.take_profit is None:
        return "n/a"
    return f"{a.take_profit_side} ${a.take_profit:,.2f}"


def action_text(action: str) -> str:
    """ "roll or close by hand (|delta| >= 0.8)" -> "Roll or close by hand (|Δ| >= 0.8)"."""
    action = action.replace("|delta|", "|Δ|")
    return action[:1].upper() + action[1:]


def signals_text(found: list[Any]) -> str:
    """One line per priority, P0 first: "P1 DTE band: 15 days left: review;
    loss multiple: ...". "-" when there are none."""
    by: dict[str, list[str]] = {}
    for x in found:
        by.setdefault(x.priority, []).append(f"{x.name}: {x.detail}")
    return "\n".join(f"{p} " + "; ".join(v) for p, v in sorted(by.items())) or "-"


def review_rank(found: list[Any]) -> tuple[str, int]:
    """(priority shown, sort rank) for the review queue. Bad data is a P0 that
    says "check the prices", not "act": the shown priority is the most urgent
    other signal ("data" if bad data is the only one, "-" if none), and rows
    with bad data sort just below clean rows of the same priority, so what you
    can act on comes first. The outcome is unchanged (close_rules)."""
    real = [x for x in found if x.name != "data unreliable"]
    top = min((x.priority for x in real), default="data" if found else "-")
    data = len(real) < len(found)
    return top, max(0, 2 * {"P0": 3, "P1": 2, "P2": 1}.get(top, 0) - data)


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
        pm: PMConfig | None = None,
        journal: Journal | None = None,
    ) -> None:
        super().__init__(config)
        self._pm = pm or PMConfig()
        self._journal = journal  # the Portfolio Manager journal (Postgres), or None
        self._pm_known: dict[str, Seen] = {}  # IDs without a journal
        self._pm_cls: dict[str, str | None] = {}  # last class, for class-change signals
        self._pm_last: dict[str, tuple] = {}  # last (outcome, signals): log changes
        self._snapshot = _Snapshot()
        self._closes: dict[InstrumentId, dict[date, float]] = {}
        self._betas: dict[str, float] = {}
        self._beta_day: date | None = None
        self._busy = False
        self._rules = rules
        # legs -> (what the strategy's rules would do, take-profit), set by _actions
        self._bot_actions: dict[frozenset[str], tuple[str, str]] = {}
        self._extras = extras
        self._iv_open: tuple[date | None, dict[InstrumentId, float]] = (None, {})
        self._vix_hedge = "-"  # what the VIX call hedge would do, set by _actions
        self._vix_chain_day: date | None = None  # VIX chain loaded for parity
        self._roll_jobs: dict[str, RollJob] = {}  # strategies to scan, this cycle
        self._rolls: dict[str, RollScan] = {}  # the last scan of each strategy
        self._roll_chain_day: dict[str, date] = {}  # chains loaded, per symbol
        self._dividends: dict[str, tuple[tuple[date, float], ...]] = {}
        self._direct: dict[InstrumentId, Quote] = {}  # read from IB directly
        # (symbol, expiry) -> when measured, the smile (None: too few good quotes)
        self._smiles: dict[tuple[str, date], tuple[datetime, Smile | None]] = {}
        self._smile_chain_day: dict[tuple[str, date], date] = {}
        self._iv_hist: dict[str, IvHistory] = {}  # two years of closes and IVs
        self._iv_day: dict[str, date] = {}  # when each was last requested
        self._iv_models: dict[tuple[str, date], IvModel | None] = {}

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

    async def _vix_partners(self, today: date) -> set[InstrumentId]:
        """The parity partner of every held VIX option (same expiry and strike,
        other right): its price gives the matching VIX future the option prices
        off. Loads the VIX chain over the held expiries, at most once a day."""
        held = [
            self.cache.instrument(x.instrument_id)
            for x in held_legs(self.cache)
            if x.symbol == VIX_ROOT and x.is_option
        ]
        held = [i for i in held if i is not None]
        missing = [i for i in held if parity_partner(self.cache, i) is None]
        if missing and self._vix_chain_day != today:
            self._vix_chain_day = today
            days = [(i.expiration_utc.date() - today).days for i in missing]
            chain = {
                "secType": "IND",
                "symbol": "VIX",
                "exchange": "CBOE",
                "build_options_chain": True,
                "min_expiry_days": max(min(days) - 1, 0),
                "max_expiry_days": max(days) + 1,
            }
            try:
                await aio.load_contracts(self, [chain])
            except Exception as e:  # noqa: BLE001 - VIX greeks are then missing
                self.log.warning(f"VIX chain not loaded: {e!r}")
        return {p.id for i in held if (p := parity_partner(self.cache, i))}

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
        ids |= await self._vix_partners(today)
        await aio.wait_for_prices(self, ids, timeout=10)
        legs = held_legs(self.cache)  # again, now with prices
        if self._beta_day != today:
            await self._load_betas(equity_symbols(symbols), today)
        if (
            not self.config.publish
        ):  # --role trade: the betas are all the strategy needs
            return

        await self._read_direct_quotes(legs)
        summary, daily = await self._read_extras()
        orders = await self._read_orders()
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
            self._proposals(f, legs, book, spx, today)
            self._actions(f, legs, book, today)
            self._structures(f, legs, book, today)
            if daily:
                self._attribution(f, legs, book, daily, spx, today)
        if orders is None:  # no side connection: the orders this node knows
            f.set("orders_open", "Open orders", len(self.cache.orders_open()))
        else:
            f.set("orders_open", "Open orders", len(orders))
            for o in orders:
                labels = {k: o[k] for k in ("id", "order", "type", "status", "client")}
                for field in ("qty", "limit", "filled"):
                    if o[field] is not None:
                        f.set(
                            "open_order",
                            "Open orders at IB",
                            o[field],
                            field=field,
                            **labels,
                        )
        f.set("last_update_timestamp_seconds", "Last refresh", self.clock.timestamp())
        self._snapshot.families = list(f.by_name.values())
        if self._pm.roll_scan:
            await self._scan_rolls(today)
        if self._pm.iv_model or self._pm.american_pricing:
            await self._load_underlying_data(symbols, today)
        if self._pm.surface and self._pm.american_pricing:
            await self._scan_surface(today)

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
        kept: dict[str, float] = {}  # $ realised on closed options, per underlying
        if self._journal is not None:
            try:
                kept = self._journal.kept_premium()
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"Journal unreadable, stock breakeven at cost: {e!r}")
        for leg, row, stripe in position_rows(legs):
            labels = position_labels(leg)
            und = price(self.cache, underlying_id(leg.symbol))
            if und is not None:
                f.set("position_underlying_price", "Underlying price", und, **labels)
            f.set("position_row", "Row in the old bot's positions order", row, **labels)
            f.set("position_stripe", "0/1, alternating per symbol", stripe, **labels)
            f.set("position_quantity", "Signed contracts or shares", leg.qty, **labels)
            f.set("position_avg_price", "Cost basis per share", leg.avg_px, **labels)
            f.set(
                "position_breakeven",
                "Underlying price at expiry with no P&L (this leg alone; stock:"
                " cost less what its closed options realised)",
                breakeven(leg.right, leg.strike, leg.avg_px)
                if leg.is_option
                else leg.avg_px - kept.get(leg.symbol, 0.0) / leg.qty,
                **labels,
            )
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
        day = expected_move(price(self.cache, VIX))
        cost = gamma_pnl(book.gamma, book.spx, day)
        f.set("expected_move", "Expected daily SPX move from VIX (fraction)", day)
        f.set("gamma_pnl_expected", "Gamma P&L on an expected day ($)", cost)
        f.set("net_carry", "Theta + expected-day gamma P&L ($/day)", book.theta + cost)
        even = breakeven_move(book.theta, book.gamma, book.spx)
        if even is not None:  # max: bigger days lose (short gamma); min: smaller do
            kind, move = even
            f.set(
                "breakeven_move",
                "SPX daily move where gamma P&L offsets a day's theta",
                move,
                kind=kind,
            )
        nlv = net_liquidation(self.cache)
        if nlv:
            cfg = self._rules or DiscretionaryConfig()
            band = delta_band_dollars(
                book.theta, nlv, cfg.delta_band.theta_days, cfg.delta_band.nav_pct
            )
            f.set("delta_band_dollars", "Neutral while |beta delta $| <= this", band)
            f.set(
                "delta_neutrality",
                "0 neutral, 1 drifting, 2 off neutral",
                NEUTRALITY.index(neutrality(book.beta_dollars, band)),
            )
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
            # delta hedging this symbol on a typical day of its own (last month's
            # realized move; at its implied vol the carry would be ~0 by pricing)
            move = realized_move(self._closes.get(underlying_id(s), {}))
            if move is None:
                move = abs(r.beta) * day
            hedge = gamma_pnl(r.gamma, r.price, move)
            f.set("symbol_daily_move", "Typical daily move (fraction)", move, symbol=s)
            f.set(
                "symbol_gamma_pnl", "Gamma P&L of delta hedging, a day", hedge, symbol=s
            )
            f.set(
                "symbol_net_carry",
                "Theta + gamma P&L, a day",
                r.theta + hedge,
                symbol=s,
            )
        for leg in legs:
            labels = position_labels(leg)
            if not leg.is_option:  # per share, as IB quotes it: stock delta is 1
                f.set("position_delta", "Delta per share", 1.0, **labels)
                continue
            g = book.greeks.get(leg.instrument_id)
            if g is None:
                continue
            f.set("position_delta", "Delta per share", g.delta, **labels)
            f.set("position_gamma", "Gamma per share", g.gamma, **labels)
            f.set("position_theta", "Theta per share ($/day)", g.theta, **labels)
            f.set("position_vega", "Vega per share ($ per vol point)", g.vega, **labels)
            f.set("position_iv", "Implied volatility", g.vol, **labels)
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

    async def _read_orders(self) -> list[dict[str, Any]] | None:
        """Every open order at IB, from any client; None without the side
        connection (or when it fails)."""
        if self._extras is None or not self._extras.ib.isConnected():
            return None
        try:
            return await self._extras.open_orders()
        except Exception as e:  # noqa: BLE001 - only the order list is lost
            self.log.warning(f"Open orders unavailable: {e!r}")
            return None

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

    def _no_trading(self, symbol: str) -> bool:
        sym = self._rules.symbols.get(symbol) if self._rules else None
        return bool(sym and sym.no_trading)

    def _proposals(
        self, f: _Families, legs: list[Leg], book: Book, spx: float, today: date
    ) -> None:
        """SPX delta = 0 targets per symbol and the trade toward them (display
        only, as in the old positions summary), and the greeks and P&L curves
        after them."""
        unit = contract_units(legs, book, spx)
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
                    _, trade, _ = symbol_adjustment(
                        [x for x in legs if x.symbol == s], shares, unit
                    )
            f.set(
                "symbol_target",
                "SPX delta = 0 target per symbol (display only)",
                1,
                symbol=s,
                side_now=side_now,
                side_target=side_target,
                trade=trade,
            )
        # the resize plan (the strategy's trade plan), with the same guards as its orders:
        # rows for the Trade plan table, and "after resizing held" in the greeks
        skip = {s for s in book.by_symbol if is_index(s) or self._no_trading(s)}
        busy = symbols_with_open_orders(self.cache)
        active = bool(self._rules and self._rules.trading.approve_trades == "auto")
        groups = self._rules.groups() if self._rules else {}
        plan_legs = outside_groups(legs, groups)
        planned = target_plan(self.cache, plan_legs, book, spx, skip, busy)
        for p in planned:
            f.set(
                "target_plan_units",
                "Resize plan per symbol: units (contracts, spreads or shares)",
                p.qty,
                trade=p.name,
                contracts=order_text(self.cache, p.combo, p.qty or p.want) or "-",
                note=p.note,
                status="active" if active else "preview",
            )
            if p.price is not None:
                f.set("target_plan_price", "Net mid per unit", p.price, trade=p.name)
                f.set(
                    "target_plan_cost",
                    "Units x net mid x multiplier (+ debit, - credit)",
                    order_cost(self.cache, p.combo, p.qty, p.price),
                    trade=p.name,
                )
            if p.spx_per_unit is not None:
                f.set(
                    "target_plan_unit_spx_delta",
                    "SPX delta per unit of a resize",
                    p.spx_per_unit,
                    trade=p.name,
                )
            for greek, value in p.add.items():
                after[greek] += value
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
        added = [leg for p in planned if p.qty for leg in proposal_legs(self.cache, p)]
        resized = build_book(
            self.cache, self.greeks, legs + added, self._betas, spx, today
        )
        vix = price(self.cache, VIX)
        for scenario, held, b in (
            ("now", legs, book),
            ("resized", legs + added, resized),
        ):
            for level, pnl in curve(self.cache, held, b, today).items():
                f.set(
                    "pnl_curve",
                    "Theoretical P&L at an SPX price (beta-weighted, options at current IV)",
                    pnl,
                    scenario=scenario,
                    spx=f"{level:.0f}",
                )
            for level, pnl in vix_curve(self.cache, held, b, vix, today).items():
                f.set(
                    "vix_pnl_curve",
                    "Theoretical P&L at a VIX level (SPX unchanged, every IV shifted)",
                    pnl,
                    scenario=scenario,
                    vix=f"{level:.2f}",
                )

    def _structures(
        self, f: _Families, legs: list[Leg], book: Book, today: date
    ) -> None:
        """One row per structure, its legs taken together (read-only)."""
        groups = self._rules.groups() if self._rules else {}
        by_symbol: dict[str, list[Leg]] = {}
        for leg in legs:
            if leg.symbol != VIX_ROOT:
                by_symbol.setdefault(leg.symbol, []).append(leg)

        def money(x: float | None) -> str:
            if x is None:
                return "n/a"  # legs at several expiries
            if math.isinf(x):
                return "unlimited"
            return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"

        cfg = self._rules or DiscretionaryConfig()
        entries = []  # for the Portfolio Manager: (symbol, spot, structure, figures, bot)
        for s, items in by_symbol.items():
            und = price(self.cache, underlying_id(s))
            for st in find_structures(items, groups):
                fig = structure_figures(st, book.greeks, today)
                name = structure_label(st)
                # what the bot's own rules would do: shown beside the PM's view
                rec = recommend(st, fig, und, book.greeks, cfg, today)
                if rec is None:
                    labels = frozenset(option_label(p) for p in st.legs)
                    bot = self._bot_actions.get(labels, ("-", "-"))[0]
                else:
                    bot = rec[0]
                f.set(
                    "structure_info",
                    "An option structure: legs held as one position",
                    1,
                    name=name,
                    symbol=s.lstrip("^"),
                    kind=f"group: {st.name}" if st.name else st.kind,
                    max_profit=money(fig.max_profit),
                    max_loss=money(fig.max_loss),
                    breakevens=", ".join(f"{x:,.2f}" for x in fig.breakevens) or "-",
                )
                entries.append((s, und, st, fig, action_text(bot)))
                for field, value in (
                    ("qty", fig.qty),
                    ("dte", fig.dte),
                    ("cost", fig.cost),
                    ("value", fig.value),
                    ("pnl", fig.pnl),
                    (
                        "pnl_ratio",
                        fig.pnl / abs(fig.cost) if fig.pnl and fig.cost else None,
                    ),
                    ("underlying", und),
                    ("delta", fig.delta),
                    ("gamma", fig.gamma),
                    ("theta", fig.theta),
                    ("vega", fig.vega),
                ):
                    if value is not None:
                        f.set(
                            "structure",
                            "An option structure's combined figures ($, shares)",
                            value,
                            name=name,
                            field=field,
                        )
        # long VIX calls (vault vix_options_management): priced on the matching
        # VIX future, so the spot is that future (the front leg's, for a calendar)
        vix_legs = [x for x in legs if x.symbol == VIX_ROOT and x.is_option]
        greeks = dict(book.greeks)
        for x in vix_legs:
            ins = self.cache.instrument(x.instrument_id)
            if ins is not None and (
                g := option_greeks(self.cache, self.greeks, ins, today)
            ):
                greeks[x.instrument_id] = g
        for x in legs:  # no bid: the cache has no price, so half the ask gives one
            direct = self._direct.get(x.instrument_id)
            if x.is_option and x.instrument_id not in greeks and direct:
                ins = self.cache.instrument(x.instrument_id)
                g = (
                    option_greeks(
                        self.cache,
                        self.greeks,
                        ins,
                        today,
                        premium=(direct.bid + direct.ask) / 2,
                    )
                    if ins
                    else None
                )
                if g is not None:
                    greeks[x.instrument_id] = g
        for st in find_structures(vix_legs, groups):
            front = greeks.get(
                min(st.legs, key=lambda p: p.expiry or today).instrument_id
            )
            fig = structure_figures(st, greeks, today)
            entries.append(
                (
                    VIX_ROOT,
                    front.underlying if front else None,
                    st,
                    fig,
                    self._vix_hedge,
                )
            )
        try:
            self._portfolio_manager(f, entries, greeks, today)
        except Exception as e:  # noqa: BLE001 - the rest of the dashboard never waits on it
            where = traceback.extract_tb(e.__traceback__)[-1]
            self.log.warning(
                f"Portfolio Manager skipped: {e!r} at {where.filename}:{where.lineno}"
            )

    async def _scan_rolls(self, today: date) -> None:
        """Quote the candidates of the strategy longest without a scan
        (roll_rules: the same structure, shifted in expiry and strike) and
        evaluate each as a new trade. Reported, never orders."""
        now = datetime.now(UTC)
        rescan = timedelta(minutes=self._pm.roll_rescan_mins)
        never = datetime.min.replace(tzinfo=UTC)
        when = {
            sid: self._rolls[sid].at if sid in self._rolls else never
            for sid, job in self._roll_jobs.items()
            # after its dividends and IV history: a scan before them is half a scan
            if self._iv_day.get(job.symbol) == today or job.vix
        }
        due = [sid for sid, at in when.items() if now - at >= rescan]
        if not due:
            return
        sid = min(due, key=when.__getitem__)
        job = self._roll_jobs[sid]
        lim = self._pm.roll_limits()
        found: list[Candidate] = []
        try:
            targets = target_dtes(job.front_dte, job.initial_dte, job.vix, lim)
            await self._load_roll_chain(job.symbol, targets, today)
            chain = {
                (
                    "C" if i.option_kind == OptionKind.CALL else "P",
                    i.expiration_utc.date(),
                    float(i.strike_price),
                ): i.id
                for i in self.cache.instruments(VENUE, underlying=job.symbol)
                if isinstance(i, OptionContract)
            }
            candidates = generate(
                job.structure.legs, chain, job.underlying, today, targets, lim
            )
            ids = {p.instrument_id for c in candidates for p in c}
            if job.vix:  # each option's future comes from put-call parity
                ins = [self.cache.instrument(i) for i in ids]
                ids |= {q.id for i in ins if i and (q := parity_partner(self.cache, i))}
            await self._fresh_quotes(ids)
            greeks = {}
            for iid in ids:
                ins = self.cache.instrument(iid)
                g = option_greeks(self.cache, self.greeks, ins, today) if ins else None
                if g is not None:
                    greeks[iid] = g
            quotes = {iid: self._quote(iid) for iid in ids}
            for legs in candidates:
                c = evaluate(
                    job.structure.kind,
                    legs,
                    greeks,
                    quotes,
                    today,
                    now,
                    0.0 if job.vix else SCENARIO_RATE,
                    None if job.vix else job.sigma_f,
                    self._pm.grid_spec(vix=job.vix),
                    self._pm.utility_limits(),
                    self._pm.quality(),
                    job.max_loss,
                    job.iv_model,
                    self._pm.iv_limits(),
                    job.price_model,
                )
                if c is not None:
                    found.append(c)
        except Exception as e:  # noqa: BLE001 - the dashboard never waits on it
            self.log.warning(f"Roll scan of {sid} skipped: {e!r}")
        # recorded even when empty, so one bad strategy can't hog every cycle
        self._rolls[sid] = RollScan(now, found)

    def _iv_model_of(
        self, symbol: str, sigma_f: float | None, today: date
    ) -> IvModel | None:
        """Built once a day per underlying: its history and forecast barely move."""
        key = (symbol, today)
        if key not in self._iv_models:
            history = self._iv_hist.get(symbol)
            if history is None:
                return None  # not requested yet: build when it arrives
            self._iv_models[key] = build(
                history, sigma_f, self._pm.utility_horizon, self._pm.iv_limits()
            )
        return self._iv_models[key]

    async def _scan_surface(self, today: date) -> None:
        """Measure the smile of the held expiry longest without one, once its
        underlying's dividends are known; one per cycle, a few contracts, then
        unsubscribed. VIX options and index options are left out."""
        now = datetime.now(UTC)
        rescan = timedelta(minutes=self._pm.roll_rescan_mins)
        held = {
            (x.symbol, x.expiry)
            for x in held_legs(self.cache)
            if x.is_option and x.expiry and x.symbol in self._dividends
        }
        self._smiles = {k: v for k, v in self._smiles.items() if k in held}
        never = datetime.min.replace(tzinfo=UTC)
        when = {k: self._smiles[k][0] if k in self._smiles else never for k in held}
        due = [k for k, at in when.items() if now - at >= rescan]
        if not due:
            return
        key = min(due, key=when.__getitem__)
        smile = None
        try:
            smile = await self._measure_smile(key[0], key[1], today)
        except Exception as e:  # noqa: BLE001 - the dashboard never waits on it
            self.log.warning(f"Smile of {key[0]} {key[1]} skipped: {e!r}")
        self._smiles[key] = (now, smile)

    async def _measure_smile(
        self, symbol: str, expiry: date, today: date
    ) -> Smile | None:
        und = price(self.cache, underlying_id(symbol))
        if not und:
            return None
        dte = (expiry - today).days
        if self._smile_chain_day.get((symbol, expiry)) != today:
            self._smile_chain_day[(symbol, expiry)] = today
            chain = {
                "build_options_chain": True,
                "min_expiry_days": max(dte - 1, 0),
                "max_expiry_days": dte + 1,
            }
            await aio.load_contracts(self, [stock_contract(symbol) | chain])
        options = {
            (i.option_kind == OptionKind.CALL, float(i.strike_price)): i.id
            for i in self.cache.instruments(VENUE, underlying=symbol)
            if isinstance(i, OptionContract) and i.expiration_utc.date() == expiry
        }
        strikes = strikes_around(
            sorted({k for _, k in options}), und, tuple(self._pm.surface_strikes)
        )
        # out of the money: the liquid side, without early-exercise noise
        picks = [
            (k, k >= und, options[(k >= und, k)])
            for k in strikes
            if (k >= und, k) in options
        ]
        await self._fresh_quotes({iid for _, _, iid in picks})
        return measure(
            [(k, call, self._quote(iid)) for k, call, iid in picks],
            und,
            dte / 365,
            PriceModel(self._dividends[symbol]).after(today, 0),
            self._pm.wide_spread,
        )

    def _price_model_of(self, symbol: str) -> PriceModel | None:
        """American for equity and ETF options once their dividends are known;
        until then, and for index options, European."""
        if not self._pm.american_pricing or symbol not in self._dividends:
            return None
        return PriceModel(self._dividends[symbol])

    async def _load_underlying_data(self, symbols: set[str], today: date) -> None:
        """Dividends and IV history, one underlying per cycle, once a day: IB
        paces historical requests (60 per 10 minutes). VIX options need a
        different IV model (M12) and pay no dividends."""
        if self._extras is None:
            return
        todo = sorted(s for s in symbols - {VIX_ROOT} if self._iv_day.get(s) != today)
        if not todo:
            return
        symbol = todo[0]
        self._iv_day[symbol] = today
        if self._pm.american_pricing and not is_index(symbol):
            try:
                self._dividends[symbol] = schedule(
                    *await self._extras.dividends(symbol), today
                )
            except Exception as e:  # noqa: BLE001 - stays European until it loads
                self.log.warning(f"Dividends of {symbol} not loaded: {e!r}")
        if not self._pm.iv_model:
            return
        try:
            closes, ivs = await self._extras.iv_history(symbol)
            self._iv_hist[symbol] = IvHistory(tuple(closes), tuple(ivs))
            self._iv_models.pop((symbol, today), None)
        except Exception as e:  # noqa: BLE001 - the dashboard never waits on it
            self.log.warning(f"IV history of {symbol} not loaded: {e!r}")

    async def _load_roll_chain(
        self, symbol: str, targets: list[int], today: date
    ) -> None:
        """The option chain around the target expiries, at most once a day."""
        if self._roll_chain_day.get(symbol) == today:
            return
        self._roll_chain_day[symbol] = today
        base = (
            {"secType": "IND", "symbol": symbol[1:], "exchange": "CBOE"}
            if is_index(symbol)
            else stock_contract(symbol)
        )
        chain = {
            "build_options_chain": True,
            "min_expiry_days": max(min(targets) - 15, 0),
            "max_expiry_days": max(targets) + 45,
        }
        await aio.load_contracts(self, [base | chain])

    async def _fresh_quotes(self, ids: set[InstrumentId]) -> None:
        """Subscribe a few at a time, wait for a quote newer than the request (the
        cache keeps old ones), unsubscribe: IB limits market data lines."""
        held = {x.instrument_id for x in held_legs(self.cache)}
        todo = sorted(ids - held, key=str)
        size = self._pm.roll_quote_lines
        for i in range(0, len(todo), size):
            batch = todo[i : i + size]
            since = self.clock.timestamp_ns()
            for iid in batch:
                aio.subscribe(self, iid)
            try:
                for _ in range(ROLL_QUOTE_POLLS):
                    ticks = [self.cache.quote_tick(iid) for iid in batch]
                    if all(q is not None and q.ts_event >= since for q in ticks):
                        break
                    await asyncio.sleep(0.25)
            finally:
                for iid in batch:
                    aio.unsubscribe(self, iid)

    async def _read_direct_quotes(self, legs: list[Leg]) -> None:
        """A few held options per cycle, oldest first, read from IB directly. The
        adapter drops a side IB marks -1, so it never sees an option with no bid,
        keeps an old bid after it has gone, and a quiet market looks stale."""
        if self._extras is None:
            return
        never = datetime.min.replace(tzinfo=UTC)
        options = [x for x in legs if x.is_option and x.expiry]
        options.sort(
            key=lambda x: (
                getattr(self._direct.get(x.instrument_id), "time", never) or never
            )
        )
        batch = options[: self._pm.roll_quote_lines]

        def key(x: Leg) -> tuple[str, str, float, str]:
            return (x.symbol.lstrip("^"), f"{x.expiry:%Y%m%d}", x.strike, x.right)

        try:
            got = await self._extras.quotes({key(x) for x in batch})
        except Exception as e:  # noqa: BLE001 - the adapter's quotes still stand
            self.log.warning(f"Direct quotes not read: {e!r}")
            return
        now = datetime.now(UTC)
        for x in batch:
            bid, ask = got.get(key(x), (None, None))
            tick = self.cache.quote_tick(x.instrument_id)
            if (
                bid is None
                and ask is not None
                and tick
                and tick.bid_price.as_double() > 0
            ):
                self.log.info(
                    f"{option_label(x)}: no bid at IB, the adapter still has "
                    f"{tick.bid_price.as_double():g}"
                )
            if ask is not None:  # no bid is a bid of 0; no ask is no market
                self._direct[x.instrument_id] = Quote(bid or 0.0, ask, now)
            else:
                self._direct.pop(x.instrument_id, None)
        held = {x.instrument_id for x in legs}
        self._direct = {k: v for k, v in self._direct.items() if k in held}

    def _quote(self, iid: InstrumentId) -> Quote:
        """Bid, ask and quote time: the adapter's, unless a recent direct read
        shows no bid (it keeps the old one), or is newer, or it has none."""
        q = self.cache.quote_tick(iid)
        adapter = Quote(None, None)
        if q is not None:
            bid, ask = q.bid_price.as_double(), q.ask_price.as_double()
            adapter = Quote(
                bid if bid >= 0 else None,
                ask if ask > 0 else None,
                datetime.fromtimestamp(q.ts_event / 1e9, tz=UTC),
            )
        direct = self._direct.get(iid)
        if direct is None or direct.time is None:
            return adapter
        age = (datetime.now(UTC) - direct.time).total_seconds()
        if age > self._pm.stale_secs:
            return adapter
        if direct.bid == 0 or adapter.bid is None or adapter.ask is None:
            return direct
        return direct if adapter.time is None or direct.time > adapter.time else adapter

    def _portfolio_manager(
        self, f: _Families, entries: list, greeks: dict, today: date
    ) -> None:
        """The Portfolio Manager (vault): assess each strategy, give it its stable
        ID, raise the review signals and publish them; then write the journal,
        and a decision-log row when a strategy's outcome or signals change."""
        now = datetime.now(UTC)
        quotes = {
            p.instrument_id: self._quote(p.instrument_id)
            for e in entries
            for p in e[2].legs
        }
        seen = [
            Seen(
                s, st.name or st.kind, frozenset(str(p.instrument_id) for p in st.legs)
            )
            for s, _, st, _, _ in entries
        ]
        known, entry = self._pm_known, {}
        if self._journal is not None:
            try:
                known = self._journal.known(today)
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"Journal unreadable, IDs from memory: {e!r}")
        ids = auto_ids(seen, known)
        if self._journal is not None:
            try:
                entry = self._journal.entry(ids)
            except Exception:  # noqa: BLE001 - logged above
                pass
        inputs = {x.id: x for x in self._pm.strategies}
        limits, quality = self._pm.limits(), self._pm.quality()
        short_limits, vix_limits = self._pm.short_limits(), self._pm.vix_limits()
        utility = self._pm.utility_limits()
        jobs: dict[str, RollJob] = {}
        ivs: list[IvRow] = []
        forecasts: list[ForecastRow] = []
        marks: list[LegMark] = []
        roll_rows: list[RollForecast] = []
        rescan = timedelta(minutes=self._pm.roll_rescan_mins)
        rows, decisions, counts = [], [], dict.fromkeys((P0, P1, P2), 0)
        for sid, (s, und, st, fig, bot) in zip(ids, entries, strict=True):
            closes = self._closes.get(underlying_id(s), {})
            move = realized_move(closes, self._pm.realized_vol_days)
            sigma_f = move * math.sqrt(252) if move is not None else None
            a = assess(
                st,
                fig,
                greeks,
                quotes,
                und,
                sigma_f,
                today,
                now,
                quality,
                self._pm.tolerance(),
                self._pm.grid_moves,
            )
            net_debit, initial_dte = entry.get(sid, (fig.cost, a.dte_front))
            x = inputs.get(sid)
            invalidation = None
            if x is not None and x.invalidation_below is not None:
                invalidation = ("below", x.invalidation_below)
            elif x is not None and x.invalidation_above is not None:
                invalidation = ("above", x.invalidation_above)
            v_floor, _ = value_bounds(fig)
            # the scenario grid (scenario_engine): VIX options on their future
            is_vix = s == VIX_ROOT
            price_model = self._price_model_of(s)
            grid = scenario_grid(
                st.legs,
                greeks,
                today,
                0.0 if is_vix else SCENARIO_RATE,
                None if is_vix else sigma_f,  # no sigma_F for VIX futures yet
                self._pm.grid_spec(vix=is_vix),
                price_model,
            )
            worst = grid.worst_loss() if grid else None
            stress = grid.stress_loss() if grid else None
            liquidity = (
                None if a.v_mid is None or a.v_exec is None else a.v_mid - a.v_exec
            )
            # capital at risk (risk_metrics): to the expiry floor for defined
            # risk; else the worst stress loss (no broker margin here)
            car = (
                a.v_exec - v_floor
                if a.v_exec is not None and v_floor is not None
                else stress
            )
            fwd = (
                forward(
                    grid,
                    liquidity,
                    sum(abs(p.qty) for p in st.legs),
                    max(car, 0.0),
                    utility,
                )
                if grid and liquidity is not None and car is not None
                else None
            )
            # the joint price and IV model, beside the IV-unchanged one
            iv_model = None if is_vix else self._iv_model_of(s, sigma_f, today)
            fwd_iv = None
            if iv_model and grid and liquidity is not None and car is not None:
                pnls = repriced(
                    st.legs,
                    greeks,
                    today,
                    SCENARIO_RATE,
                    iv_model.samples,
                    utility.horizon,
                    self._pm.iv_term_days,
                    price_model,
                )
                fwd_iv = forward(
                    grid,
                    liquidity,
                    sum(abs(p.qty) for p in st.legs),
                    max(car, 0.0),
                    utility,
                    pnls=pnls,
                )
            # how much the grid depends on sticky strike (volatility_scenarios)
            smiles = {
                e: m
                for e in {p.expiry for p in st.legs}
                if (m := self._smiles.get((s, e), (None, None))[1])
            }
            sticky = (
                sticky_gaps(
                    st.legs,
                    greeks,
                    today,
                    smiles,
                    tuple(self._pm.grid_moves),
                    utility.horizon,
                    price_model,
                )
                if price_model and smiles
                else None
            )
            # LossBudget = max acceptable loss + min(0, unrealized P&L)
            max_loss = x.max_loss if x and x.max_loss is not None else self._pm.max_loss
            budget = (
                max_loss + min(0.0, a.v_exec - net_debit)
                if max_loss is not None and a.v_exec is not None
                else max_loss
            )
            single = st.legs[0] if len(st.legs) == 1 and st.legs[0].qty > 0 else None
            g = greeks.get(single.instrument_id) if single else None
            state = StrategyState(
                cls=a.cls,
                data_flags=a.flags,
                net_debit=net_debit,
                value_exec=a.v_exec,
                max_profit=None
                if fig.max_profit is None or math.isinf(fig.max_profit)
                else fig.max_profit,
                dte_front=a.dte_front,
                initial_dte=initial_dte,
                spot=und,
                single_long_delta=g.delta if g else None,
                gt_h=a.eff.gt_h,
                remaining_reward_risk=a.remaining_rr,
                short_legs=tuple(
                    short_leg(
                        p,
                        today,
                        und,
                        quotes.get(p.instrument_id),
                        self._dividends.get(s, ()),
                    )
                    for p in st.legs
                    if p.qty < 0
                ),
                prev_cls=self._pm_cls.get(sid),
                thesis_valid=x.thesis_valid if x else None,
                invalidation=invalidation,
                worst_case_loss=worst,
                stress_loss=stress,
                loss_budget=budget,
                sigma_f=sigma_f,
                assignment_intended=x.assignment_intended if x else False,
                wide_legs=wide_legs(st.legs, quotes, quality),
            )
            # management category (how it was opened), not its convexity class
            category = manager(net_debit)
            short = category == SHORT_PREMIUM
            if short:
                found, out = short_options.decide(state, short_limits)
            elif s == VIX_ROOT:
                found, out = long_vix.decide(state, limits, vix_limits)
            else:
                found, out = long_options.decide(state, limits)
            top, rank = review_rank(found)
            for sig in found:
                counts[sig.priority] += 1
                f.set(
                    "pm_signal",
                    "A Portfolio Manager review signal",
                    1,
                    strategy=sid,
                    priority=sig.priority,
                    name=sig.name,
                    detail=sig.detail,
                )
            f.set(
                "pm_info",
                "A strategy as the Portfolio Manager sees it",
                1,
                strategy=sid,
                symbol=s.lstrip("^"),
                kind=st.name or st.kind,
                structure=structure_label(st),
                manager=category,
                cls=a.cls or "unknown",
                outcome=out,
                top=top,
                signals=signals_text(found),
                flags="; ".join(a.flags) or "-",
                bot=bot,
            )
            net = a.net or {}
            for field, value in (
                ("v_mid", a.v_mid),
                ("v_exec", a.v_exec),
                ("liquidity_cost", liquidity),
                ("v_model", grid.v_model if grid else None),
                (
                    "sticky_gap",
                    max(sticky[0].values(), key=abs) if sticky else None,
                ),
                ("sticky_coverage", sticky[1] if sticky else None),
                ("american", 1.0 if price_model else 0.0),
                ("stress_loss", stress),
                ("loss_budget", budget),
                ("capital_at_risk", car),
                ("e_pnl", fwd.e_pnl if fwd else None),
                ("es", fwd.es if fwd else None),
                ("rar", fwd.rar if fwd else None),
                ("u_hold", fwd.u_hold if fwd else None),
                ("u_close", fwd.u_close if fwd else None),
                ("e_pnl_iv", fwd_iv.e_pnl if fwd_iv else None),
                ("es_iv", fwd_iv.es if fwd_iv else None),
                ("u_hold_iv", fwd_iv.u_hold if fwd_iv else None),
                ("iv_rank", iv_model.rank if iv_model else None),
                ("iv_view", iv_model.view * 100 if iv_model else None),
                ("iv_samples", len(iv_model.samples) if iv_model else None),
                ("wcl_1", grid.wcl(1) if grid else None),
                ("wcl_5", grid.wcl(5) if grid else None),
                ("wcl_10", grid.wcl(10) if grid else None),
                ("net_debit", net_debit),
                ("pnl_exec", None if a.v_exec is None else a.v_exec - net_debit),
                ("delta", net.get("delta")),
                ("gamma", net.get("gamma")),
                ("theta", net.get("theta")),
                ("vega", net.get("vega")),
                ("gt", a.eff.gt),
                ("gt_1", a.eff.gt_h.get(1)),
                ("gt_5", a.eff.gt_h.get(5)),
                ("gt_10", a.eff.gt_h.get(10)),
                ("sigma_be_5", a.eff.sigma_be.get(5)),
                ("sigma_f", sigma_f),
                ("tg", a.eff.tg),
                ("remaining_rr", a.remaining_rr),
                ("worst_case_loss", worst),
                ("dte_front", a.dte_front),
                (
                    "dte_window",
                    None
                    if short
                    else vix_limits.review_dte
                    if s == VIX_ROOT
                    else long_options.dte_window(initial_dte, limits.dte_windows)
                    if initial_dte is not None
                    else None,
                ),
                # Short Options Manager: premium captured, loss multiple, and
                # the premium still to earn per $ of risk still carried
                (
                    "capture",
                    capture_ratio(a.v_exec, net_debit)
                    if short and a.v_exec is not None
                    else None,
                ),
                (
                    "loss_multiple",
                    loss_multiple(a.v_exec, net_debit)
                    if short and a.v_exec is not None
                    else None,
                ),
                (
                    "premium_to_risk",
                    premium_to_risk(a.v_exec, car)
                    if short and a.v_exec is not None
                    else None,
                ),
                ("priority", rank),
            ):
                if value is not None:
                    f.set(
                        "pm",
                        "A strategy's Portfolio Manager figures",
                        value,
                        strategy=sid,
                        field=field,
                    )
            if sticky:
                for move, gap in sticky[0].items():
                    f.set(
                        "pm_sticky",
                        "Sticky-delta minus sticky-strike P&L ($), by price move",
                        gap,
                        strategy=sid,
                        move=f"{move:g}",
                    )
            for expiry, smile in smiles.items():
                for moneyness, iv in smile.points:
                    f.set(
                        "pm_smile",
                        "Measured IV (0.30 = 30%), by strike's % distance from spot",
                        iv,
                        strategy=sid,
                        expiry=expiry.isoformat(),
                        move=f"{math.expm1(moneyness) * 100:.1f}",
                    )
            if grid:
                publish_grid(f, sid, grid, self._pm.grid_spec(vix=is_vix))
                if out != "HOLD" and und:
                    jobs[sid] = RollJob(
                        s,
                        st,
                        und,
                        sigma_f,
                        a.dte_front,
                        initial_dte,
                        max_loss,
                        is_vix,
                        iv_model,
                        price_model,
                    )
            scan = self._rolls.get(sid)
            if scan and (now - scan.at) > 4 * rescan:
                scan = None  # e.g. yesterday's quotes: not worth showing
            view, view_iv = publish_rolls(
                f,
                sid,
                fwd.u_hold if fwd else None,
                fwd.u_close if fwd else None,
                scan,
                self._pm.roll_margin,
                now,
                fwd_iv.u_hold if fwd_iv else None,
            )
            key = (out, tuple((x.priority, x.name) for x in found))
            if self._pm_last.get(sid) != key:
                decisions.append((sid, out, [x.__dict__ for x in found], bot))
                self._pm_last[sid] = key
            self._pm_cls[sid] = a.cls
            ivs += iv_rows(st.legs, greeks, quotes, now, quality)
            for p in st.legs:
                q = quotes.get(p.instrument_id)
                leg_g = greeks.get(p.instrument_id)
                if q is not None and q.ask is not None:
                    marks.append(
                        LegMark(
                            str(p.instrument_id),
                            sid,
                            p.qty,
                            q.bid,
                            q.ask,
                            getattr(leg_g, "underlying", None),
                            getattr(leg_g, "vol", None),
                        )
                    )
            if scan and scan.at.date() == today:
                roll_rows += roll_forecasts(sid, scan)
            if (
                a.v_mid is not None
                and a.v_exec is not None
                and grid
                and reliable_for_the_day(a.flags)
            ):
                forecasts.append(
                    ForecastRow(
                        sid,
                        "|".join(
                            sorted(f"{p.instrument_id}:{p.qty:g}" for p in st.legs)
                        ),
                        utility.horizon,
                        out,
                        view,
                        view_iv,
                        a.v_mid,
                        a.v_exec,
                        grid.v_model,
                        fwd.e_pnl if fwd else None,
                        fwd.es if fwd else None,
                        fwd_iv.e_pnl if fwd_iv else None,
                        fwd_iv.es if fwd_iv else None,
                        fwd.u_hold if fwd else None,
                        fwd_iv.u_hold if fwd_iv else None,
                        fwd.u_close if fwd else None,
                        grid.wcl(5),
                        grid.wcl(10),
                        stress,
                        budget,
                        iv_model.rank if iv_model else None,
                        iv_model.view * 100 if iv_model else None,
                        max(sticky[0].values(), key=abs) if sticky else None,
                        price_model is not None,
                    )
                )
            legs = tuple(
                LegRow(
                    str(p.instrument_id), option_label(p), p.qty, p.multiplier, p.avg_px
                )
                for p in st.legs
            )
            rows.append(StrategyRow(sid, s, st.name or st.kind, legs, fig.cost, und, a))
        for priority, n in counts.items():
            f.set(
                "pm_signals",
                "Open Portfolio Manager review signals",
                n,
                priority=priority,
            )
        f.set("pm_strategies", "Strategies the Portfolio Manager follows", len(rows))
        for name, value, vid, source, meaning in self._pm.params():
            f.set(
                "pm_param",
                "A setting the Portfolio Manager decides with ([pm])",
                1,
                name=name,
                value=value,
                vault_id=vid or "-",
                source=source,
                meaning=meaning,
            )
        self._pm_known = dict(zip(ids, seen, strict=True))
        self._roll_jobs = jobs
        self._rolls = {k: v for k, v in self._rolls.items() if k in ids}
        if self._journal is not None:
            try:
                self._journal.record(rows, inputs, today, now)
                self._journal.record_iv(ivs, today, now)
                self._journal.record_forecasts(forecasts, marks, roll_rows, today, now)
                if decisions:
                    self._journal.decide(decisions, now)
            except Exception as e:  # noqa: BLE001 - the dashboard never depends on it
                self._pm_last.clear()  # retry the decision rows next cycle
                self.log.warning(f"Journal not written: {e!r}")

    def _actions(self, f: _Families, legs: list[Leg], book: Book, today: date) -> None:
        """What the strategy's rules would do with each position, and the price
        that takes profit; the VIX hedge's action. Uses [strategy] settings."""
        rules = self._rules
        rw = rules.credit.roll_when.rules() if rules else RollWhen()
        debit_tp = rules.debit.take_profit if rules else 0.8
        debit_dte = rules.debit.roll_when.dte if rules else 10
        long_gain = rules.debit.long_take_profit_gain if rules else 1.0
        deltas = {iid: g.delta for iid, g in book.greeks.items()}
        roll_delta = rules.debit.roll_when.delta if rules else 0.0
        self._bot_actions = {}
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
                today,
                self._no_trading(s),
                debit_dte,
                long_gain,
                rules.groups() if rules else None,
                deltas,
                roll_delta,
            ):
                key = frozenset(n for n, _ in a.legs) or frozenset({a.name})
                self._bot_actions[key] = (a.action, take_profit_text(a))
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
        vixmo = price(self.cache, VIXMO)
        action, weight = vix_hedge_action(held, price(self.cache, VIX), vixmo, cfg)
        self._vix_hedge = f"VIX hedge: {action}"
        f.set("vix_hedge_action", "What the VIX call hedge would do", 1, action=action)
        f.set("vix_hedge", "VIX call hedge", held, field="held_calls")
        if vixmo is not None:  # without VIXMO the weight is unknown, not 0
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
            for part, value in parts.items():  # per symbol only: the dashboard sums
                f.set(
                    "symbol_pnl",
                    "Today's P&L by symbol and its attribution",
                    value,
                    symbol=s,
                    part=part,
                )
