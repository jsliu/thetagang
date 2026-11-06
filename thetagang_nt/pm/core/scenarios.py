"""The scenario grid (vault: scenario_engine, option_repricing,
volatility_scenarios, stress_testing): full repricing of every leg under joint
changes in price, implied vol and elapsed time. Scenarios are sensitivity and
stress tests, not equally probable outcomes.

Each leg is priced with the generalised Black-Scholes model on its own
underlying, with the same inputs its greeks were implied with: carry = rate
for equity and index options (no dividends), carry 0 (Black-76) for VIX
options on their matching future. The baseline is that model at today's
inputs, never the market mid, so model-vs-market gaps never enter a P&L.

With a PriceModel (equity and ETF options, once their dividends are known) legs
are American with discrete dividends (pm.core.american), each leg's IV re-implied
from its mid by the same model, so the baseline stays near the mid. Index options
stay European. Sticky strike, parallel IV per leg; no surface yet
(volatility_surface)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any

from thetagang_nt.pm.core.american import american, implied_vol
from thetagang_nt.pm.core.efficiency import HORIZONS, calendar_days
from thetagang_nt.rules import Leg

RATE = 0.0425  # the greeks' flat rate (portfolio.RATE)
MIN_VOL = 0.01  # scenario IV floor


def _n(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bsm(
    und: float,
    strike: float,
    years: float,
    vol: float,
    call: bool,
    carry: float,
    rate: float = RATE,
) -> float:
    """Generalised Black-Scholes price per share; intrinsic at or after expiry.
    carry = rate: Black-Scholes; carry = 0: Black-76 on a future."""
    if years <= 0 or vol <= 0:
        return max(und - strike, 0.0) if call else max(strike - und, 0.0)
    sd = vol * math.sqrt(years)
    d1 = (math.log(und / strike) + (carry + 0.5 * vol * vol) * years) / sd
    d2 = d1 - sd
    fwd, disc = und * math.exp((carry - rate) * years), strike * math.exp(-rate * years)
    if call:
        return fwd * _n(d1) - disc * _n(d2)
    return disc * _n(-d2) - fwd * _n(-d1)


@dataclass(frozen=True)
class PriceModel:
    """American exercise, with the dividends to come: (ex-date, $ per share)."""

    dividends: tuple[tuple[date, float], ...] = ()

    def after(self, today: date, days: int) -> tuple[tuple[float, float], ...]:
        """Those still to come `days` from today, in years from then."""
        return tuple(
            (((when - today).days - days) / 365, amount)
            for when, amount in self.dividends
            if (when - today).days > days
        )


@dataclass(frozen=True)
class GridSpec:
    """Register IDs in brackets; all hypotheses."""

    moves: tuple[float, ...] = (-0.05, -0.02, 0.0, 0.02, 0.05)  # [M1] price
    iv_points: tuple[float, ...] = (-10, -5, 0, 5, 10)  # [M1] parallel IV shifts
    z: tuple[int, ...] = (-2, -1, 0, 1, 2)  # vol-scaled family, in Move_h
    vix: bool = False  # VIX regime scenarios instead of the equity tails


# [M11] VIX: futures moves skewed upward (VIX jumps up, falls gradually) and
# large option-IV shifts
VIX_SPEC = GridSpec(
    moves=(-0.20, -0.10, 0.0, 0.15, 0.30), iv_points=(-20, -10, 0, 10, 20), vix=True
)

# (price move, IV points for the front expiry, IV points for later expiries),
# repriced over 1 trading day (a gap). [M2] equity tails; Phase 2 term shocks.
# ponytail: fixed sizes; calibrate per underlying from history (vault M2).
STRESS = {
    "gap down": (-0.10, 15, 15),
    "gap up": (0.10, -10, -10),
    "event IV crush": (0.0, -20, -5),
    "front-month spike": (0.0, 15, 3),
}
# [M11] VIX regime scenarios; the vault describes them, these sizes are
# starting points chosen here: (front future move, later futures move,
# front IV points, later IV points)
VIX_STRESS = {
    "spike": (0.30, 0.15, 20, 10),
    "fade": (-0.20, -0.10, -20, -10),
    "regime shift": (0.20, 0.20, 10, 10),
}


@dataclass(frozen=True)
class Grid:
    """P&L ($) per cell against the model value now. fixed[(h, move, iv)],
    scaled[(h, z, iv)] (empty without sigma_f), stress[name]."""

    v_model: float
    fixed: dict[tuple[int, float, float], float]
    scaled: dict[tuple[int, int, float], float] = field(default_factory=dict)
    stress: dict[str, float] = field(default_factory=dict)

    def wcl(self, h: int) -> float:
        """Worst-case scenario loss at horizon h, both families (risk_metrics)."""
        cells = [v for k, v in self.fixed.items() if k[0] == h]
        cells += [v for k, v in self.scaled.items() if k[0] == h]
        return max(0.0, -min(cells))

    def worst_loss(self) -> float:
        """The largest WCL over the horizons: the P0 risk-budget check."""
        return max(self.wcl(h) for h in HORIZONS)

    def stress_loss(self) -> float:
        return max([0.0] + [-v for v in self.stress.values()])

    def summary(self, h: int, iv_step: float) -> dict[str, float]:
        """The vault's per-horizon summaries of the fixed family."""
        cells = [v for k, v in self.fixed.items() if k[0] == h]
        worst, best = min(cells), max(cells)
        adverse = [v for v in cells if v < 0]
        out = {
            "best": best,
            "worst": worst,
            "median": median(cells),
            "adverse_count": float(len(adverse)),
            "adverse_sum": sum(adverse),
            "time_decay": self.fixed[(h, 0.0, 0.0)],
            "iv_sensitivity": self.fixed[(h, 0.0, iv_step)]
            - self.fixed[(h, 0.0, -iv_step)],
        }
        if worst < 0:
            out["asymmetry"] = best / -worst
        return out


@dataclass(frozen=True)
class _Priced:
    leg: Leg
    und: float
    vol: float
    front: bool


def _value(
    priced: list[_Priced],
    today: date,
    days: int,
    move: Any,
    iv: Any,
    carry: float,
    model: PriceModel | None = None,
) -> float:
    """Strategy value ($) with `days` calendar days elapsed; move(p) and iv(p)
    give each leg's price move and IV points."""
    total = 0.0
    dividends = model.after(today, days) if model else ()
    for p in priced:
        years = (p.leg.dte(today) - days) / 365
        vol = max(p.vol + iv(p) / 100, MIN_VOL)
        und = p.und * (1 + move(p))
        call = p.leg.right == "C"
        if years <= 0 and days > 0:
            price = _settled(p, today, days, move(p), und, call)
        elif model:
            price = american(und, p.leg.strike, years, vol, call, RATE, dividends)
        else:
            price = bsm(und, p.leg.strike, years, vol, call, carry)
        total += p.leg.qty * p.leg.multiplier * price
    return total


def _settled(
    p: _Priced, today: date, days: int, x: float, und: float, call: bool
) -> float:
    """A leg that expires inside the horizon (scenario_engine): it settles on the
    price at its own expiry, the move scaled by sqrt(time) to that day. Out of the
    money it is gone; in the money it becomes stock, repriced at the scenario's
    price, so it keeps its exposure after expiry. An option expiring today cannot
    lose on a move that comes later."""
    left = max(p.leg.dte(today), 0)
    at_expiry = p.und * (1 + x * math.sqrt(left / days))
    strike = p.leg.strike
    if call:
        return und - strike if at_expiry > strike else 0.0
    return strike - und if at_expiry < strike else 0.0


def _priced(
    legs: tuple[Leg, ...],
    greeks: dict[Any, Any],
    today: date,
    model: PriceModel | None = None,
) -> list[_Priced] | None:
    """Each leg with its own underlying and implied vol; None if one is missing.
    With a model, the vol is re-implied from the leg's mid by that model; where
    no vol matches the mid, the leg keeps its European one."""
    priced = []
    dividends = model.after(today, 0) if model else ()
    front_expiry = min(p.expiry for p in legs if p.expiry)
    for p in legs:
        g = greeks.get(p.instrument_id)
        und, vol = getattr(g, "underlying", None), getattr(g, "vol", None)
        if not und or not vol or not p.is_option:
            return None
        if model and p.mid:
            vol = (
                implied_vol(
                    p.mid,
                    und,
                    p.strike,
                    p.dte(today) / 365,
                    p.right == "C",
                    RATE,
                    dividends,
                )
                or vol
            )
        priced.append(_Priced(p, und, vol, p.expiry == front_expiry))
    return priced


def repriced(
    legs: tuple[Leg, ...],
    greeks: dict[Any, Any],
    today: date,
    carry: float,
    samples: tuple[tuple[float, float], ...],
    h: int,
    term_days: int,
    model: PriceModel | None = None,
) -> list[float] | None:
    """P&L ($) of each (return, IV change) sample after h trading days, against
    the same model baseline as the grid. The IV change is for term_days to
    expiry; a longer-dated leg moves by sqrt(term_days / DTE) of it (M3)."""
    priced = _priced(legs, greeks, today, model)
    if priced is None:
        return None
    v_model = _value(priced, today, 0, lambda p: 0.0, lambda p: 0.0, carry, model)
    days = calendar_days(today, h)

    def share(p: _Priced) -> float:
        return min(1.0, math.sqrt(term_days / max(p.leg.dte(today), 1)))

    return [
        _value(
            priced,
            today,
            days,
            lambda p, r=r: r,
            lambda p, c=c: c * 100 * share(p),
            carry,
            model,
        )
        - v_model
        for r, c in samples
    ]


def grid(
    legs: tuple[Leg, ...],
    greeks: dict[Any, Any],
    today: date,
    carry: float,
    sigma_f: float | None,
    spec: GridSpec = GridSpec(),
    model: PriceModel | None = None,
) -> Grid | None:
    """The 3 horizons x price x IV grid, both families, and the stress
    scenarios; None if a leg has no underlying price or implied vol. Each leg
    moves on its own underlying (a VIX option on its own future)."""
    priced = _priced(legs, greeks, today, model)
    if priced is None:
        return None
    v_model = _value(priced, today, 0, lambda p: 0.0, lambda p: 0.0, carry, model)

    def pnl(days: int, move: Any, iv: Any) -> float:
        return _value(priced, today, days, move, iv, carry, model) - v_model

    moves = set(spec.moves)
    last_day = calendar_days(today, max(HORIZONS))
    if any(p.leg.dte(today) <= last_day for p in priced):
        # near expiry: the strikes themselves are price nodes (pin risk)
        lo, hi = min(spec.moves), max(spec.moves)
        und = priced[0].und
        moves |= {
            round(p.leg.strike / und - 1, 4)
            for p in priced
            if lo < p.leg.strike / und - 1 < hi
        }
    fixed, scaled = {}, {}
    for h in HORIZONS:
        days = calendar_days(today, h)
        for dv in spec.iv_points:
            for x in sorted(moves):
                fixed[(h, x, dv)] = pnl(days, lambda p, x=x: x, lambda p, dv=dv: dv)
            if sigma_f:
                step = sigma_f * math.sqrt(h / 252)
                for z in spec.z:
                    scaled[(h, z, dv)] = pnl(
                        days, lambda p, x=z * step: x, lambda p, dv=dv: dv
                    )
    one_day = calendar_days(today, 1)
    stress = {}
    if spec.vix:
        for name, (f_mv, b_mv, f_iv, b_iv) in VIX_STRESS.items():
            stress[name] = pnl(
                one_day,
                lambda p, a=f_mv, b=b_mv: a if p.front else b,
                lambda p, a=f_iv, b=b_iv: a if p.front else b,
            )
    else:
        for name, (mv, f_iv, b_iv) in STRESS.items():
            stress[name] = pnl(
                one_day,
                lambda p, mv=mv: mv,
                lambda p, a=f_iv, b=b_iv: a if p.front else b,
            )
    return Grid(v_model, fixed, scaled, stress)
