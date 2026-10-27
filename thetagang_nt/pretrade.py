"""`thetagang-nt pretrade [TRADE ...]`: see what proposed trades do to the
portfolio's risk before sending them.

What it proposes, first that applies:
- ad-hoc TRADEs typed on the command line (rules.parse_trade: "sell 2 AA
  20261218 40P"), as one scenario;
- each [[pretrade.candidates]] entry, sized on its own to bring SPX delta to 0
  like the [[strategy.trades]]: alternatives to compare, you pick one;
- the strategy's own trade plan (new trades, or the resizes of what you hold).
Every "after" number comes from the same build_book / stress_pnl the monitor
uses, run on the held legs plus the proposed ones. It prints now vs each
scenario, publishes the same on strategy.pretrade_metrics_port (the Pre-trade
dashboard), asks what to send, re-prices at the confirmation, and sends through
the strategy's own order path (dry_run applies). Then it shuts down."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from prometheus_client import Gauge

from thetagang_nt import aio
from thetagang_nt.config import CandidateConfig, DiscretionaryConfig
from thetagang_nt.monitor import BETAS_KEY
from thetagang_nt.portfolio import (
    INDEX_CONTRACTS,
    SPX,
    VIX,
    VIXMO,
    Book,
    ComboLegs,
    build_book,
    held_legs,
    leg_of,
    net_liquidation,
    order_cost,
    order_text,
    price,
    vix_pnl,
    stock_contract,
    stress_pnl,
    underlying_id,
)
from thetagang_nt.rules import (
    NEUTRALITY,
    Leg,
    TradeSpec,
    breakeven_move,
    delta_band_dollars,
    expected_move,
    gamma_pnl,
    neutrality,
    parse_trade,
)
from thetagang_nt.strategy import NEW_YORK, Discretionary, NoContract

RISK = Gauge(
    "thetagang_pretrade_risk",
    "Risk now and after each scenario ('now', or the candidate's name)",
    ["measure", "scenario", "row"],
)
TRADE = Gauge(
    "thetagang_pretrade_trade",
    "Proposed orders per scenario",
    ["scenario", "trade", "contracts", "field"],
)
CURVE = Gauge(
    "thetagang_pretrade_curve",
    "Theoretical P&L of each scenario's portfolio at an SPX price (beta-weighted, "
    "options repriced at current IV)",
    ["scenario", "spx"],
)
CURVE_MOVES = tuple(i / 100 for i in range(-10, 11))  # SPX -10% .. +10%, 1% apart
VIX_CURVE = Gauge(
    "thetagang_pretrade_vix_curve",
    "Theoretical P&L of each scenario's portfolio at a VIX level (SPX unchanged, "
    "every IV shifted by the VIX change)",
    ["scenario", "vix"],
)
VIX_MOVES = tuple(range(-10, 21))  # VIX points: -10 .. +20, floored at 9
STATUS = Gauge("thetagang_pretrade_status", "Where the proposal stands", ["status"])
CHOICE = Gauge(
    "thetagang_pretrade_choice",
    "Your last answer to the pretrade prompt (value: when, unix secs; the "
    "dashboard keeps showing the latest one after pretrade exits)",
    ["choice"],
)
SCRAPE_HOLD_SECS = (
    60  # stay up a minute after the answer: Prometheus (15 s) scrapes the outcome
)
BETAS_TIMEOUT_SECS = 180  # the monitor actor loads a year of daily closes first


class PreTradeConfig(DiscretionaryConfig, frozen=True):
    adhoc: tuple[str, ...] = ()  # typed trades: one scenario, before the candidates
    candidates: tuple[CandidateConfig, ...] = ()  # [[pretrade.candidates]]
    stress_moves: tuple[float, ...] = (-0.10, -0.05, -0.01, 0.01, 0.05, 0.10)


@dataclass
class Proposal:
    name: str
    symbol: str
    combo: ComboLegs  # (instrument, ratio) per leg of one unit
    qty: int  # units (contracts or shares)
    unit_price: float | None  # net mid per unit; negative is a credit
    note: str = ""


# (label, metric, format) in report order
# the first seven, named as the main dashboard's Portfolio greeks table, are what
# the Pre-trade dashboard compares
MEASURES = [
    ("SPX delta", "spx_delta", "{:+.2f}"),
    ("Beta delta $", "beta_delta_dollars", "${:+,.0f}"),
    ("Delta $ (no beta)", "delta_dollars", "${:+,.0f}"),
    ("Gamma (Δ / pt)", "gamma", "{:+.4f}"),
    ("Theta $/day", "theta", "${:+,.0f}"),
    ("Vega $/vol pt", "vega", "${:+,.0f}"),
    ("Volga $/vol pt", "volga", "${:+,.0f}"),
    ("Neutral band +/-", "delta_band_dollars", "${:,.0f}"),
    ("Neutrality", "neutrality", "{}"),
    ("Distance from neutral (x band)", "band_multiple", "{:.1f}x"),
    ("Gamma P&L expected", "gamma_pnl_expected", "${:+,.0f}"),
    ("Net carry", "net_carry", "${:+,.0f}"),
    ("Break-even: max move % (short gamma)", "breakeven_max", "{:.2f}%"),
    ("Break-even: min move % (long gamma)", "breakeven_min", "{:.2f}%"),
]


def risk(
    cache: Any,
    legs: list[Leg],
    book: Book,
    cfg: Any,
    today: date,
    moves: tuple[float, ...] = (),
) -> dict[str, float]:
    """The headline risk of a book, as the dashboard shows it."""
    nlv = net_liquidation(cache) or 0.0
    band = delta_band_dollars(
        book.theta, nlv, cfg.delta_band_theta_days, cfg.delta_band_nav_pct
    )
    gamma_day = gamma_pnl(book.gamma, book.spx, expected_move(price(cache, VIX)))
    even = breakeven_move(book.theta, book.gamma, book.spx)
    out = {
        "spx_delta": book.spx_delta,
        "beta_delta_dollars": book.beta_dollars,
        "delta_dollars": book.plain_dollars,
        "gamma": book.gamma,
        "delta_band_dollars": band,
        "neutrality": NEUTRALITY.index(neutrality(book.beta_dollars, band)),
        # 0 = neutral at the centre, 1 = the band's edge, 2 = the drifting edge
        "band_multiple": abs(book.beta_dollars) / band if band > 0 else math.inf,
        "theta": book.theta,
        "vega": book.vega,
        "volga": book.volga,
        "gamma_pnl_expected": gamma_day,
        "net_carry": book.theta + gamma_day,
        # only one applies: bigger days lose (max) or smaller days lose (min)
        "breakeven_max": 100 * even[1] if even and even[0] == "max" else math.nan,
        "breakeven_min": 100 * even[1] if even and even[0] == "min" else math.nan,
        "positions_without_greeks": len(book.missing),
    }
    for m in moves:
        out[f"stress_{m:+.0%}"] = stress_pnl(cache, legs, book, m, True, today)
    return out


def fmt(key: str, value: float, spec: str) -> str:
    if key == "neutrality":
        return NEUTRALITY[int(value)]
    if math.isnan(value):
        return "-"
    return spec.format(value)


class PreTrade(Discretionary):
    def on_start(self) -> None:  # no timer: one proposal, then shut down
        aio.spawn(self, self._main(), "pretrade")

    async def _main(self) -> None:
        try:
            await self._propose_and_confirm()
        except Exception as e:  # noqa: BLE001 - always shut down, say why
            self.log.error(f"pretrade failed: {e!r}")
            print(f"\npretrade failed: {e}", flush=True)
        finally:
            await asyncio.sleep(SCRAPE_HOLD_SECS)  # the dashboard sees the outcome
            self.shutdown_system("pretrade done")

    async def _propose_and_confirm(self) -> None:
        cfg: PreTradeConfig = self.config
        today = self._today()
        await aio.load_contracts(self, INDEX_CONTRACTS)
        betas = await self._betas()
        legs = held_legs(self.cache)
        ids = {leg.instrument_id for leg in legs} | {SPX, VIX, VIXMO}
        ids |= {underlying_id(leg.symbol) for leg in legs}
        await aio.wait_for_prices(self, ids, cfg.quote_timeout_secs)
        legs = held_legs(self.cache)
        spx = price(self.cache, SPX)
        if not spx:
            raise NoContract("no SPX price")
        book = build_book(self.cache, self.greeks, legs, betas, spx, today)
        scenarios = await self._scenarios(legs, book, spx, betas, today)
        now_risk, afters = compare(
            self, legs, book, scenarios, betas, spx, today, cfg.stress_moves
        )
        publish(self.cache, scenarios, now_risk, afters, "awaiting your choice")
        print(self._report(scenarios, now_risk, afters), flush=True)
        choices = [n for n, ps in scenarios.items() if sendable(ps)]
        if not choices:
            STATUS.clear()
            STATUS.labels(status="nothing to send").set(1)
            print("Nothing to send.", flush=True)
            return
        dry = "(DRY RUN) " if cfg.dry_run else ""
        if len(scenarios) == 1:
            count = len(sendable(scenarios[choices[0]]))
            prompt = f"Send {dry}these {count} order(s)? [y/N] "
        else:
            menu = "  ".join(f"{k}={n}" for k, n in enumerate(choices, 1))
            prompt = f"Send {dry}which? 0=none, no trade  {menu}  (Enter: 0) "
        answer = await ask_within(prompt, cfg.proposal_max_age_secs)
        STATUS.clear()
        if answer is None:  # its prices are too old to send now: end the run
            STATUS.labels(status="no answer: nothing sent").set(1)
            mins = cfg.proposal_max_age_secs / 60
            print(f"\nNo answer within {mins:g} min: nothing sent.", flush=True)
            return
        answer = answer.strip().lower()
        if len(scenarios) == 1:
            chosen = choices[0] if answer == "y" else None
        else:
            pick = int(answer) if answer.isdigit() else 0
            chosen = choices[pick - 1] if 1 <= pick <= len(choices) else None
        stamp = self.clock.utc_now().astimezone(NEW_YORK).strftime("%m-%d %H:%M")
        if chosen is None:
            STATUS.labels(status="no trade: your choice").set(1)
            CHOICE.labels(choice=f"{stamp} no trade").set(time.time())
            print("No trade: nothing sent.", flush=True)
            return
        for p in sendable(scenarios[chosen]):  # at the mid now, not the one shown
            mids = [await self._price_of(iid) for iid, _ in p.combo]
            if any(m is None for m in mids):
                print(f"{p.name}: no quote now, not sent", flush=True)
                continue
            px = sum(r * m for (_, r), m in zip(p.combo, mids, strict=True))
            await self._submit(p.symbol, f"pretrade {p.name}", p.combo, p.qty, px)
            done = "logged, not sent (dry_run)" if cfg.dry_run else "sent"
            print(f"{order_text(self.cache, p.combo, p.qty)} @ {px:+.2f}: {done}")
        sent = "sent (dry run)" if cfg.dry_run else "sent"
        STATUS.labels(status=f"{sent}: {chosen}").set(1)
        CHOICE.labels(choice=f"{stamp} {sent}: {chosen}").set(time.time())
        print(
            f"Done. Exiting in {SCRAPE_HOLD_SECS} s.", flush=True
        )  # _main waits before shutting down: orders reach IB

    async def _scenarios(
        self,
        legs: list[Leg],
        book: Book,
        spx: float,
        betas: dict[str, float],
        today: date,
    ) -> dict[str, list[Proposal]]:
        """What to compare: the typed trades, else each candidate (sized on its
        own), else the strategy's own plan."""
        cfg: PreTradeConfig = self.config
        if cfg.adhoc:
            return {
                "ad hoc": [await self._adhoc(parse_trade(t), today) for t in cfg.adhoc]
            }
        self._check_greeks(book)
        busy = self._symbols_with_open_orders()
        if cfg.candidates:
            return await size_candidates(
                self, cfg.candidates, book, spx, betas, today, busy
            )
        planned = await self._plan(legs, book, spx, betas, today, busy)
        mode = "resize held" if cfg.place_target_orders else "strategy trades"
        return {mode: [as_proposal(p) for p in planned]}

    def _check_greeks(self, book: Book) -> None:
        blocking = self._blocking_missing(book)
        if blocking:
            names = ", ".join(i.symbol.value for i in blocking)
            raise NoContract(
                f"{len(blocking)} position(s) without greeks (no quote at IB): "
                f"{names}. SPX delta is incomplete, so nothing can be sized "
                "(or set ignore_missing_greeks for their symbols)"
            )

    async def _betas(self) -> dict[str, float]:
        deadline = time.monotonic() + BETAS_TIMEOUT_SECS
        while (raw := self.cache.get(BETAS_KEY)) is None:
            if time.monotonic() > deadline:
                self.log.warning("No betas from the monitor: betas default to 1")
                return {}
            await asyncio.sleep(1)
        return json.loads(raw)

    async def _adhoc(self, spec: TradeSpec, today: date) -> Proposal:
        """Look up and price one typed trade (see rules.parse_trade)."""
        name = " ".join(spec.text.split()[2:])
        if not spec.legs:  # shares
            await aio.load_contracts(self, [stock_contract(spec.symbol)])
            iid = underlying_id(spec.symbol)
            mid = await self._price_of(iid)
            unit = None if mid is None else spec.side * mid
            return Proposal(name, spec.symbol, [(iid, spec.side)], spec.qty, unit)
        assert spec.expiry is not None
        days = (spec.expiry - today).days
        await self._load_chain(spec.symbol, days, days)
        await self._price_of(underlying_id(spec.symbol))  # the greeks need it
        combo: ComboLegs = []
        total = 0.0
        for right, strike, ratio in spec.legs:
            iid = self._option_at(spec.symbol, right, spec.expiry, strike)
            label = f"{spec.symbol} {spec.expiry:%Y%m%d} {strike:g}{right}"
            if iid is None:
                return Proposal(name, spec.symbol, [], spec.qty, None, f"no {label}")
            mid = await self._price_of(iid)
            if mid is None:
                return Proposal(
                    name, spec.symbol, [], spec.qty, None, f"no quote: {label}"
                )
            combo.append((iid, spec.side * ratio))
            total += spec.side * ratio * mid
        return Proposal(name, spec.symbol, combo, spec.qty, total)

    def _report(
        self,
        scenarios: dict[str, list[Proposal]],
        now: dict[str, float],
        afters: dict[str, dict[str, float]],
    ) -> str:
        lines = [""]
        choices = [n for n, ps in scenarios.items() if sendable(ps)]
        for name, proposals in scenarios.items():
            tag = f"[{choices.index(name) + 1}] " if name in choices else "[-] "
            cost = afters[name]["cost"]
            lines.append(f"{tag}{name}   (cost ${cost:+,.0f})")
            for p in proposals:
                if p.unit_price is None:
                    lines.append(f"    {p.name}: not priced ({p.note or 'no quote'})")
                    continue
                kind = "debit" if p.unit_price > 0 else "credit"
                note = f"   ({p.note})" if p.note else ""
                order = order_text(self.cache, p.combo, p.qty or 1) if p.combo else ""
                lines.append(
                    f"    {order}   {p.qty} x {abs(p.unit_price):.2f} {kind}{note}"
                )
            lines.append("")
        names = list(afters)
        width = max(14, *(min(len(n), 18) + 2 for n in names))
        single = len(names) == 1
        head = "".join(f"{n[:18]:>{width}}" for n in names)
        lines.append(
            f"  {'':<24}{'Now':>{width}}{head}"
            + (f"{'Change':>{width}}" if single else "")
        )
        for label, key, spec in rows(now):
            cells = "".join(f"{fmt(key, afters[n][key], spec):>{width}}" for n in names)
            change = ""
            if single and key not in ("neutrality", "cost"):
                a, b = now[key], afters[names[0]][key]
                if not (math.isnan(a) or math.isnan(b)):
                    change = spec.format(b - a)
            lines.append(
                f"  {label:<24}{fmt(key, now[key], spec):>{width}}{cells}"
                + (f"{change:>{width}}" if single else "")
            )
        if now["positions_without_greeks"]:
            lines.append(
                f"\n  Note: {now['positions_without_greeks']:.0f} position(s) have no "
                "greeks (no quote at IB): the totals leave them out."
            )
        return "\n".join(lines) + "\n"


def sendable(proposals: list[Proposal]) -> list[Proposal]:
    """The proposals that are orders: some quantity, and a price."""
    return [p for p in proposals if p.qty and p.unit_price is not None]


def as_proposal(p: Any) -> Proposal:
    return Proposal(p.name, p.symbol, p.combo, p.qty, p.price, p.note)


def rows(now: dict[str, float]) -> list[tuple[str, str, str]]:
    """(label, key, format) in report order: the measures, the stress moves,
    then the cost."""
    stress = [
        (f"Stress SPX {k[7:]}", k, "${:+,.0f}") for k in now if k.startswith("stress_")
    ]
    return MEASURES + stress + [("Cost (+ pay, - receive)", "cost", "${:+,.0f}")]


def proposal_legs(cache: Any, p: Proposal) -> list[Leg]:
    """A proposal as positions, at mid (so no P&L yet)."""
    out = []
    for iid, ratio in p.combo:
        leg = leg_of(cache, iid, p.qty * ratio, price(cache, iid) or 0.0)
        if leg is not None:
            out.append(leg)
    return out


def compare(
    st: Any,
    legs: list[Leg],
    book: Book,
    scenarios: dict[str, list[Proposal]],
    betas: dict[str, float],
    spx: float,
    today: date,
    moves: tuple[float, ...] = (),
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Risk now, and after each scenario's orders (held legs + the proposal,
    through the same build_book the monitor uses)."""
    now = risk(st.cache, legs, book, st.config, today, moves) | {"cost": 0.0}
    vix = price(st.cache, VIX)
    now["curve"] = curve(st.cache, legs, book, today)
    now["vix_curve"] = vix_curve(st.cache, legs, book, vix, today)
    afters = {}
    for name, proposals in scenarios.items():
        added = [leg for p in sendable(proposals) for leg in proposal_legs(st.cache, p)]
        after = build_book(st.cache, st.greeks, legs + added, betas, spx, today)
        afters[name] = risk(st.cache, legs + added, after, st.config, today, moves)
        afters[name]["curve"] = curve(st.cache, legs + added, after, today)
        afters[name]["vix_curve"] = vix_curve(st.cache, legs + added, after, vix, today)
        afters[name]["cost"] = sum(
            order_cost(st.cache, p.combo, p.qty, p.unit_price)
            for p in sendable(proposals)
        )
    return now, afters


def curve(cache: Any, legs: list[Leg], book: Book, today: date) -> dict[float, float]:
    """{SPX price: theoretical P&L} for CURVE_MOVES (the stress model)."""
    return {
        round(book.spx * (1 + m), 2): stress_pnl(cache, legs, book, m, True, today)
        for m in CURVE_MOVES
    }


def vix_curve(
    cache: Any, legs: list[Leg], book: Book, vix: float | None, today: date
) -> dict[float, float]:
    """{VIX level: theoretical P&L} for VIX_MOVES; empty without a VIX price."""
    if not vix:
        return {}
    return {
        round(vix + d, 2): vix_pnl(cache, legs, book, d, today)
        for d in VIX_MOVES
        if vix + d >= 9
    }


def publish(
    cache: Any,
    scenarios: dict[str, list[Proposal]],
    now: dict[str, float],
    afters: dict[str, dict[str, float]],
    status: str,
) -> None:
    RISK.clear()
    TRADE.clear()
    STATUS.clear()
    CURVE.clear()
    VIX_CURVE.clear()
    for scenario, values in [("now", now), *afters.items()]:
        for level, pnl in values.get("curve", {}).items():
            CURVE.labels(scenario=scenario, spx=f"{level:.0f}").set(pnl)
        for level, pnl in values.get("vix_curve", {}).items():
            VIX_CURVE.labels(scenario=scenario, vix=f"{level:.2f}").set(pnl)
    for row, (label, key, _) in enumerate(rows(now)):
        for scenario, values in [("now", now), *afters.items()]:
            RISK.labels(measure=label, scenario=scenario, row=f"{row:02d}").set(
                values[key]
            )
    for scenario, proposals in scenarios.items():
        for p in proposals:
            order = order_text(cache, p.combo, p.qty) or p.note or "-"
            labels = {"scenario": scenario, "trade": p.name, "contracts": order}
            TRADE.labels(**labels, field="qty").set(p.qty)
            if p.unit_price is not None:
                TRADE.labels(**labels, field="price").set(p.unit_price)
                cost = order_cost(cache, p.combo, p.qty, p.unit_price)
                TRADE.labels(**labels, field="cost").set(cost)
    STATUS.labels(status=status).set(1)


async def size_candidates(
    st: Any,
    candidates: Any,
    book: Book,
    spx: float,
    betas: dict[str, float],
    today: date,
    busy: set[str],
) -> dict[str, list[Proposal]]:
    """Each candidate's trades sized together to bring SPX delta to 0, on its own."""
    out = {}
    for c in candidates:
        planned = await st._new_trades(list(c.trades), book, spx, betas, today, busy)
        out[c.name] = [as_proposal(p) for p in planned]
    return out


async def refresh(st: Any, candidates: Any) -> None:
    """The background comparison (the trading strategy, every refresh_mins): size
    and compare the candidates, publish them for the Pre-trade dashboard, then
    stop the quotes they needed. Never sends anything."""
    today = st._today()
    raw = st.cache.get(BETAS_KEY)
    spx = price(st.cache, SPX)
    if raw is None or not spx:
        return  # just started: the next refresh has them
    betas = json.loads(raw)
    legs = held_legs(st.cache)
    book = build_book(st.cache, st.greeks, legs, betas, spx, today)
    blocking = st._blocking_missing(book)
    stamp = st.clock.utc_now().astimezone(NEW_YORK).strftime("%H:%M")
    if blocking:
        names = ", ".join(i.symbol.value for i in blocking)
        STATUS.clear()
        STATUS.labels(status=f"{stamp}: not compared, no greeks for {names}").set(1)
        return
    busy = st._symbols_with_open_orders()
    scenarios = await size_candidates(st, candidates, book, spx, betas, today, busy)
    now, afters = compare(st, legs, book, scenarios, betas, spx, today)
    publish(st.cache, scenarios, now, afters, f"compared at {stamp} (background)")
    held = {leg.instrument_id for leg in legs}
    for proposals in scenarios.values():  # free the quote lines the pricing took
        for p in proposals:
            for iid, _ in p.combo:
                if iid not in held:
                    aio.unsubscribe(st, iid)


def ask(prompt: str) -> str:
    """The y/N answer; no input at all (closed stdin) counts as no."""
    try:
        return input(prompt)
    except EOFError:
        return ""


async def ask_within(prompt: str, timeout_secs: float) -> str | None:
    """ask() in a daemon thread (an unanswered prompt never keeps the process
    alive); None when there is no answer within the timeout."""
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[str] = loop.create_future()

    def read() -> None:
        text = ask(prompt)
        loop.call_soon_threadsafe(lambda: answer.done() or answer.set_result(text))

    threading.Thread(target=read, daemon=True).start()
    try:
        return await asyncio.wait_for(answer, timeout_secs)
    except TimeoutError:
        return None
