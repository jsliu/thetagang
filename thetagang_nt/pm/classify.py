"""Convexity classification (vault: convexity_classification).

By the signs of the strategy's net Gamma and Theta, never by whether it was
opened for a debit or a credit: a debit call spread is short gamma once the
underlying is above the middle of its strikes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from thetagang_nt.rules import Leg

LONG = "long convexity"  # class 1: + gamma, - theta
SHORT = "short convexity"  # class 2: - gamma, + theta
MIXED = "mixed"  # class 3: anything else, multi-expiry, or unstable sign

GRID_MOVES = (-0.05, -0.02, 0.0, 0.02, 0.05)  # the scenario grid's price nodes (M1)


@dataclass(frozen=True)
class Tolerance:
    """T7 (hypotheses): below these the sign is treated as zero, so the class
    doesn't flap. Units: shares per $1, $ per calendar day."""

    gamma: float = 0.01
    theta: float = 0.01


def net_greeks(
    legs: tuple[Leg, ...], greeks: dict[Any, Any]
) -> dict[str, float] | None:
    """Signed, multiplier-weighted sums (vault: portfolio_greeks); None if any
    leg has no greeks."""
    g = [greeks.get(p.instrument_id) for p in legs]
    if any(x is None for x in g):
        return None
    out = {}
    for attr in ("delta", "gamma", "theta", "vega"):
        out[attr] = sum(
            p.qty * p.multiplier * getattr(x, attr) for p, x in zip(legs, g)
        )
    return out


def bs_gamma(spot: float, strike: float, years: float, vol: float) -> float:
    """Black-Scholes gamma per share, no rates or dividends.
    ponytail: zero-rate BSM, enough for a sign check; the scenario engine
    (Phase 2) reprices with rates and dividends."""
    if spot <= 0 or strike <= 0 or years <= 0 or vol <= 0:
        return 0.0
    sd = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * sd * sd) / sd
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (spot * sd)


def classify(
    legs: tuple[Leg, ...],
    greeks: dict[Any, Any],
    spot: float | None,
    today: date,
    tol: Tolerance = Tolerance(),
) -> str | None:
    """LONG / SHORT / MIXED; None without greeks (data unreliable). Mixed if
    the legs expire on different dates, or if net gamma changes sign across
    the grid's price nodes (needs each leg's implied vol, `vol`)."""
    if len({p.expiry for p in legs}) > 1:
        return MIXED
    net = net_greeks(legs, greeks)
    if net is None:
        return None
    if net["gamma"] > tol.gamma and net["theta"] < -tol.theta:
        cls = LONG
    elif net["gamma"] < -tol.gamma and net["theta"] > tol.theta:
        cls = SHORT
    else:
        return MIXED
    vols = [getattr(greeks[p.instrument_id], "vol", None) for p in legs]
    if spot and all(v for v in vols):
        sign = 1 if cls == LONG else -1
        for move in GRID_MOVES:
            s = spot * (1 + move)
            gamma = sum(
                p.qty * p.multiplier * bs_gamma(s, p.strike, p.dte(today) / 365, v)
                for p, v in zip(legs, vols)
            )
            if gamma * sign <= 0:
                return MIXED
    return cls
