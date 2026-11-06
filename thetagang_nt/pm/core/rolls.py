"""Roll candidates and the HOLD / CLOSE / ROLL comparison (vault: roll_rules,
transaction_costs, vix_roll_management). A roll is two trades: close the old
strategy, open a new one that must be attractive on its own; the old P&L plays
no part. Candidates are the same structure, shifted in expiry and strike.

Reported, not a signal, like the forward value it rests on: with zero drift
and IV unchanged (M8) a new trade rarely beats cash after its costs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any

from thetagang_nt.pm.core.efficiency import HORIZONS
from thetagang_nt.pm.core.iv_model import IvLimits, IvModel
from thetagang_nt.pm.core.profit import value_bounds
from thetagang_nt.pm.core.quality import (
    QualityLimits,
    Quote,
    leg_flags,
    values,
    wide_legs,
)
from thetagang_nt.pm.core.scenarios import GridSpec, PriceModel, grid, repriced
from thetagang_nt.pm.core.valuation import UtilityLimits, capital_at_risk, forward
from thetagang_nt.rules import Leg, Structure, structure_figures, structure_label

Chain = dict[tuple[str, date, float], Any]  # (right, expiry, strike) -> instrument
UNUSABLE = ("missing quote", "no greeks")


@dataclass(frozen=True)
class RollLimits:
    """Register IDs in brackets; all hypotheses, starting values chosen here
    except V1 (the vault leaves the candidate set open)."""

    dte_multiples: tuple[float, ...] = (1.0, 1.5, 2.0)  # [M13] of the original life
    min_life: int = 30  # [M13] days: a shorter or unknown life counts as this
    min_extension: int = 14  # [M13] days past the current front expiry, at least
    strike_shifts: tuple[float, ...] = (-0.05, 0.0, 0.05)  # [M13] share of underlying
    strike_tolerance: float = 0.03  # [M13] nearest listed strike within this share
    vix_dte: tuple[int, ...] = (30, 45, 60)  # [V1] the 30-60 day roll target
    margin: float = 10.0  # [M7] epsilon, $: a roll must beat HOLD by more than this


def target_dtes(
    front_dte: int, initial_dte: int | None, vix: bool, lim: RollLimits
) -> list[int]:
    """Days to the new front expiry: back to the strategy's original life and
    longer (VIX: V1), always at least min_extension past the current one."""
    if vix:
        wanted: list[int] = list(lim.vix_dte)
    else:
        life = max(initial_dte or 0, lim.min_life)
        wanted = [round(life * m) for m in lim.dte_multiples]
    return sorted({max(t, front_dte + lim.min_extension) for t in wanted})


def generate(
    legs: tuple[Leg, ...],
    chain: Chain,
    underlying: float,
    today: date,
    targets: list[int],
    lim: RollLimits,
) -> list[tuple[Leg, ...]]:
    """Every leg moved by the same days and the same strike shift, each snapped
    to a listed contract. Dropped: nothing near enough listed, a structure that
    collapses (two expiries or strikes landing on one), or the strategy itself.
    avg_px and mid are placeholders until the candidate is quoted."""
    front = min(p.expiry for p in legs if p.expiry)
    front_dte = (front - today).days
    held = frozenset((p.right, p.expiry, p.strike) for p in legs)
    out, seen = [], set()
    for target in targets:
        for shift in lim.strike_shifts:
            moved = []
            for p in legs:
                expiries = {e for r, e, _ in chain if r == p.right}
                if not expiries:
                    break
                wanted = p.expiry + timedelta(days=target - front_dte)
                expiry = min(expiries, key=lambda e: abs((e - wanted).days))
                strikes = [k for r, e, k in chain if r == p.right and e == expiry]
                goal = p.strike + shift * underlying
                strike = min(strikes, key=lambda k: abs(k - goal))
                if abs(strike - goal) > lim.strike_tolerance * underlying:
                    break
                moved.append(
                    replace(
                        p,
                        instrument_id=chain[(p.right, expiry, strike)],
                        expiry=expiry,
                        strike=strike,
                    )
                )
            else:
                keys = frozenset((p.right, p.expiry, p.strike) for p in moved)
                same_shape = (
                    len(keys) == len(legs)
                    and len({p.expiry for p in moved}) == len({p.expiry for p in legs})
                    and min(p.expiry for p in moved) > front
                )
                if same_shape and keys not in seen and keys != held:
                    seen.add(keys)
                    out.append(tuple(moved))
    return out


@dataclass(frozen=True)
class Candidate:
    label: str
    legs: tuple[Leg, ...]
    net_open: float  # $ to open at executable prices (+ paid, - received)
    open_cost: float  # $, half-spread crossing plus commissions, to open
    wcl: dict[int, float]
    stress_loss: float
    e_pnl: float | None = None
    es: float | None = None
    u_new: float | None = None  # its own utility, opening and later exit costs in
    fails: tuple[str, ...] = ()  # why it can't be recommended
    u_new_iv: float | None = None  # the same utility under the joint IV model


def evaluate(
    kind: str,
    legs: tuple[Leg, ...],
    greeks: dict[Any, Any],
    quotes: dict[Any, Quote],
    today: date,
    now: datetime,
    carry: float,
    sigma_f: float | None,
    spec: GridSpec,
    util: UtilityLimits,
    quality: QualityLimits,
    max_loss: float | None,
    iv_model: IvModel | None = None,
    iv_limits: IvLimits = IvLimits(),
    model: PriceModel | None = None,
) -> Candidate | None:
    """One candidate as a new trade, through the same grid and utility as HOLD;
    None if a leg has no usable quote or greeks."""
    flags = {
        flag
        for p in legs
        for flag in leg_flags(
            p, quotes.get(p.instrument_id), greeks.get(p.instrument_id), now, quality
        )
    }
    if flags & set(UNUSABLE):
        return None
    legs = tuple(
        replace(p, mid=(quotes[p.instrument_id].bid + quotes[p.instrument_id].ask) / 2)
        for p in legs
    )
    g = grid(legs, greeks, today, carry, sigma_f, spec, model)
    v_mid, v_exec = values(legs, quotes)
    if g is None or v_mid is None or v_exec is None:
        return None
    priced = tuple(
        replace(
            p,
            avg_px=quotes[p.instrument_id].ask
            if p.qty > 0
            else quotes[p.instrument_id].bid,
        )
        for p in legs
    )
    s = Structure(kind, priced)
    fig = structure_figures(s, greeks, today)
    contracts = sum(abs(p.qty) for p in legs)
    half_spread = v_mid - v_exec
    open_cost = half_spread + contracts * util.commission
    stress = g.stress_loss()
    car = capital_at_risk(v_exec, value_bounds(fig)[0], stress)
    fwd = forward(g, half_spread, contracts, max(car or 0.0, 0.0), util)
    u_new_iv = None
    if iv_model is not None:
        pnls = repriced(
            legs,
            greeks,
            today,
            carry,
            iv_model.samples,
            util.horizon,
            iv_limits.term_days,
            model,
        )
        fwd_iv = forward(
            g, half_spread, contracts, max(car or 0.0, 0.0), util, pnls=pnls
        )
        if fwd_iv:
            u_new_iv = fwd_iv.u_hold - open_cost
    fails = sorted(flags)
    if wide_legs(legs, quotes, quality):
        fails.append("wide spread")
    if max_loss is not None and max(g.worst_loss(), stress) > max_loss:
        fails.append("over max_loss")
    return Candidate(
        structure_label(s),
        priced,
        fig.cost,
        open_cost,
        {h: g.wcl(h) for h in HORIZONS},
        stress,
        fwd.e_pnl if fwd else None,
        fwd.es if fwd else None,
        fwd.u_hold - open_cost if fwd else None,  # forward counts the later exit
        tuple(fails),
        u_new_iv,
    )


@dataclass(frozen=True)
class Choice:
    view: str  # HOLD, CLOSE, ROLL or "-" without probabilities
    best: Candidate | None  # the highest-utility candidate that meets the limits
    u_roll: float | None  # its U_new - C_close
    roll_score: float | None  # U_roll - U_hold


def compare(
    u_hold: float | None,
    u_close: float | None,
    candidates: list[Candidate],
    margin: float,
) -> Choice:
    """ROLL only if the best candidate beats HOLD by more than the margin and
    beats cash; else the better of HOLD and CLOSE (roll_rules)."""
    if u_hold is None or u_close is None:
        return Choice("-", None, None, None)
    usable = [c for c in candidates if c.u_new is not None and not c.fails]
    best = max(usable, key=lambda c: c.u_new or 0.0) if usable else None
    if best is None or best.u_new is None:
        return Choice("HOLD" if u_hold >= u_close else "CLOSE", None, None, None)
    u_roll = best.u_new + u_close
    score = u_roll - u_hold
    if score > margin and u_roll > u_close:
        view = "ROLL"
    else:
        view = "HOLD" if u_hold >= u_close else "CLOSE"
    return Choice(view, best, u_roll, score)
