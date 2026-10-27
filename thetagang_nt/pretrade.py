"""`thetagang-nt pretrade [TRADE ...]`: see what proposed trades do to the
portfolio's risk before sending them.

Without TRADEs it proposes the [[strategy.trades]], sized to bring SPX delta to
0 exactly as the strategy would; with them it prices those ad-hoc trades
(rules.parse_trade: "sell 2 AA 20261218 40P"). Every "after" number comes from
the same build_book / stress_pnl the monitor uses, run on the held legs plus the
proposed ones. It prints now / after / change, publishes the same on
strategy.pretrade_metrics_port (the Pre-trade dashboard), asks before sending,
re-prices at the confirmation, and sends through the strategy's own order path
(dry_run applies). Then it shuts down."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from prometheus_client import Gauge

from thetagang_nt import aio
from thetagang_nt.config import DiscretionaryConfig
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
from thetagang_nt.strategy import Discretionary, NoContract

RISK = Gauge(
    "thetagang_pretrade_risk",
    "Risk now and after the proposal",
    ["measure", "scenario"],
)
TRADE = Gauge(
    "thetagang_pretrade_trade", "Proposed trades", ["trade", "contracts", "field"]
)
STATUS = Gauge("thetagang_pretrade_status", "Where the proposal stands", ["status"])
BETAS_TIMEOUT_SECS = 180  # the monitor actor loads a year of daily closes first


class PreTradeConfig(DiscretionaryConfig, frozen=True):
    adhoc: tuple[str, ...] = ()  # empty: propose the [[strategy.trades]]
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
MEASURES = [
    ("SPX delta", "spx_delta", "{:+.2f}"),
    ("Beta delta $", "beta_delta_dollars", "${:+,.0f}"),
    ("Neutral band +/-", "delta_band_dollars", "${:,.0f}"),
    ("Neutrality", "neutrality", "{}"),
    ("Theta / day", "theta", "${:+,.0f}"),
    ("Vega / vol pt", "vega", "${:+,.0f}"),
    ("Volga / vol pt", "volga", "${:+,.0f}"),
    ("Gamma P&L expected", "gamma_pnl_expected", "${:+,.0f}"),
    ("Net carry", "net_carry", "${:+,.0f}"),
    ("Break-even move", "breakeven_move", "{:.2%}"),
]


def risk(
    cache: Any, legs: list[Leg], book: Book, cfg: PreTradeConfig, today: date
) -> dict[str, float]:
    """The headline risk of a book, as the dashboard shows it."""
    nlv = net_liquidation(cache) or 0.0
    band = delta_band_dollars(
        book.theta, nlv, cfg.delta_band_theta_days, cfg.delta_band_nav_pct
    )
    gamma_day = gamma_pnl(book.gamma, book.spx, expected_move(price(cache, VIX)))
    move = breakeven_move(book.theta, book.gamma, book.spx)
    out = {
        "spx_delta": book.spx_delta,
        "beta_delta_dollars": book.beta_dollars,
        "delta_band_dollars": band,
        "neutrality": NEUTRALITY.index(neutrality(book.beta_dollars, band)),
        "theta": book.theta,
        "vega": book.vega,
        "volga": book.volga,
        "gamma_pnl_expected": gamma_day,
        "net_carry": book.theta + gamma_day,
        "breakeven_move": math.nan if move is None else move,
        "positions_without_greeks": len(book.missing),
    }
    for m in cfg.stress_moves:
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
        if cfg.adhoc:
            proposals = [await self._adhoc(parse_trade(t), today) for t in cfg.adhoc]
        else:
            proposals = await self._configured(legs, book, spx, betas, today)
        orders = [p for p in proposals if p.qty and p.unit_price is not None]
        added = [leg for p in orders for leg in self._legs(p)]
        after = build_book(self.cache, self.greeks, legs + added, betas, spx, today)
        now_risk = risk(self.cache, legs, book, cfg, today)
        after_risk = risk(self.cache, legs + added, after, cfg, today)
        self._publish(proposals, now_risk, after_risk)
        print(self._report(proposals, now_risk, after_risk), flush=True)
        if not orders:
            STATUS.clear()
            STATUS.labels(status="nothing to send").set(1)
            print("Nothing to send.", flush=True)
            return
        asked = time.monotonic()
        dry = "(DRY RUN) " if cfg.dry_run else ""
        prompt = f"Send {dry}these {len(orders)} order(s)? [y/N] "
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, ask, prompt)
        STATUS.clear()
        if answer.strip().lower() != "y":
            STATUS.labels(status="cancelled").set(1)
            print("Nothing sent.", flush=True)
            return
        if time.monotonic() - asked > cfg.proposal_max_age_secs:
            STATUS.labels(status="stale").set(1)
            print("Proposal too old: nothing sent. Run pretrade again.", flush=True)
            return
        for p in orders:  # at the mid now, not the one shown
            mids = [await self._price_of(iid) for iid, _ in p.combo]
            if any(m is None for m in mids):
                print(f"{p.name}: no quote now, not sent", flush=True)
                continue
            px = sum(r * m for (_, r), m in zip(p.combo, mids, strict=True))
            await self._submit(p.symbol, f"pretrade {p.name}", p.combo, p.qty, px)
            done = "logged, not sent (dry_run)" if cfg.dry_run else "sent"
            print(f"{order_text(self.cache, p.combo, p.qty)} @ {px:+.2f}: {done}")
        STATUS.labels(status="sent (dry run)" if cfg.dry_run else "sent").set(1)
        await asyncio.sleep(5)  # let the orders reach IB before shutting down
        print("Done.", flush=True)

    async def _betas(self) -> dict[str, float]:
        deadline = time.monotonic() + BETAS_TIMEOUT_SECS
        while (raw := self.cache.get(BETAS_KEY)) is None:
            if time.monotonic() > deadline:
                self.log.warning("No betas from the monitor: betas default to 1")
                return {}
            await asyncio.sleep(1)
        return json.loads(raw)

    async def _configured(
        self,
        legs: list[Leg],
        book: Book,
        spx: float,
        betas: dict[str, float],
        today: date,
    ) -> list[Proposal]:
        """The strategy's own trade plan: new trades, or with place_target_orders
        the held positions resized toward their targets."""
        blocking = self._blocking_missing(book)
        if blocking:
            names = ", ".join(i.symbol.value for i in blocking)
            raise NoContract(
                f"{len(blocking)} position(s) without greeks (no quote at IB): "
                f"{names}. SPX delta is incomplete, so the plan can't be sized "
                "(or set ignore_missing_greeks for their symbols)"
            )
        busy = self._symbols_with_open_orders()
        planned = await self._plan(legs, book, spx, betas, today, busy)
        return [
            Proposal(p.name, p.symbol, p.combo, p.qty, p.price, p.note) for p in planned
        ]

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

    def _legs(self, p: Proposal) -> list[Leg]:
        """The proposal as positions, at mid (so no P&L yet)."""
        out = []
        for iid, ratio in p.combo:
            leg = leg_of(self.cache, iid, p.qty * ratio, price(self.cache, iid) or 0.0)
            if leg is not None:
                out.append(leg)
        return out

    def _publish(
        self, proposals: list[Proposal], now: dict[str, float], after: dict[str, float]
    ) -> None:
        RISK.clear()
        TRADE.clear()
        STATUS.clear()
        for key, value in now.items():
            RISK.labels(measure=key, scenario="now").set(value)
            RISK.labels(measure=key, scenario="after").set(after[key])
        for p in proposals:
            contracts = order_text(self.cache, p.combo, p.qty) or p.note or "-"
            TRADE.labels(trade=p.name, contracts=contracts, field="qty").set(p.qty)
            if p.unit_price is not None:
                TRADE.labels(trade=p.name, contracts=contracts, field="price").set(
                    p.unit_price
                )
                total = order_cost(self.cache, p.combo, p.qty, p.unit_price)
                TRADE.labels(trade=p.name, contracts=contracts, field="cost").set(total)
        STATUS.labels(status="awaiting approval").set(1)

    def _report(
        self, proposals: list[Proposal], now: dict[str, float], after: dict[str, float]
    ) -> str:
        lines = ["", "Proposed trades:"]
        cost = 0.0
        for p in proposals:
            if p.unit_price is None:
                lines.append(f"  {p.name:<36} not priced: {p.note or 'no quote'}")
                continue
            total = order_cost(self.cache, p.combo, p.qty, p.unit_price)
            cost += total
            kind = "debit" if p.unit_price > 0 else "credit"
            note = f"  ({p.note})" if p.note else ""
            lines.append(
                f"  {p.name:<36} {p.qty:>5} x {abs(p.unit_price):>8.2f} {kind:<6}"
                f" ${total:>+10,.0f}{note}"
            )
            lines.append(f"  {'':<36} {order_text(self.cache, p.combo, p.qty)}")
        lines.append(f"  {'Total (+ cost, - credit)':<36} ${cost:>+,.0f}")
        lines += ["", f"  {'':<22}{'Now':>14}{'After':>14}{'Change':>14}"]
        stress = [
            (f"Stress SPX {k[7:]}", k, "${:+,.0f}")
            for k in now
            if k.startswith("stress_")
        ]
        for label, key, spec in MEASURES + stress:
            a, b = now[key], after[key]
            change = ""
            if key != "neutrality" and not (math.isnan(a) or math.isnan(b)):
                change = f"{b - a:+.2%}" if "%" in spec else spec.format(b - a)
            lines.append(
                f"  {label:<22}{fmt(key, a, spec):>14}{fmt(key, b, spec):>14}{change:>14}"
            )
        if now["positions_without_greeks"]:
            lines.append(
                f"\n  Note: {now['positions_without_greeks']:.0f} position(s) have no "
                "greeks (no quote at IB): the totals leave them out."
            )
        return "\n".join(lines) + "\n"


def ask(prompt: str) -> str:
    """The y/N answer; no input at all (closed stdin) counts as no."""
    try:
        return input(prompt)
    except EOFError:
        return ""
