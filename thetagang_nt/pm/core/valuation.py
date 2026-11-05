"""Forward value over a horizon (vault: expected_pnl, risk_metrics, roll_rules):
expected P&L, Expected Shortfall, risk-adjusted return and the HOLD and CLOSE
utilities. Risk and expectation stay separate outputs; the utility only
combines them for the HOLD vs CLOSE comparison. Reported, not a signal: with
zero drift and IV unchanged, E[P&L] is near zero and HOLD scores below CLOSE
almost by construction, so it waits for a better probability model (M8).

Probability model (M8, a hypothesis): the price follows the forecast realised
vol with zero drift, discretised onto the vol-scaled nodes (z = -2..+2 Move_h,
each node the normal mass of its cell); implied vol unchanged. Without
sigma_f there are no probabilities and nothing here is computed.
ponytail: IV held fixed in the expectation; add an empirical h-day IV-change
distribution per underlying (vault expected_pnl) when the history exists."""

from __future__ import annotations

import math
from dataclasses import dataclass

from thetagang_nt.pm.core.scenarios import Grid


def _n(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def node_probabilities(z: tuple[int, ...]) -> dict[int, float]:
    """Normal mass of each node's cell, edges halfway between nodes, open ends."""
    zs = sorted(z)
    edges = [-math.inf] + [(a + b) / 2 for a, b in zip(zs, zs[1:])] + [math.inf]
    return {x: _n(hi) - _n(lo) for x, lo, hi in zip(zs, edges, edges[1:])}


def expected_shortfall(outcomes: list[tuple[float, float]], alpha: float) -> float:
    """ES_alpha from (probability, P&L) pairs, as a positive loss (risk_metrics):
    the probability-weighted mean of the worst alpha of outcomes."""
    total, acc = 0.0, 0.0
    for p, pnl in sorted(outcomes, key=lambda x: x[1]):
        take = min(p, alpha - acc)
        total += take * pnl
        acc += take
        if acc >= alpha:
            break
    return -total / alpha


@dataclass(frozen=True)
class UtilityLimits:
    """Register IDs in brackets; all hypotheses with starting values chosen
    here (the vault leaves them open)."""

    horizon: int = 5  # trading days the forward value is measured over
    alpha: float = 0.10  # [M7] Expected Shortfall tail
    risk_aversion: float = 0.1  # [M7] lambda: $ of utility per $ of ES
    capital_cost: float = 0.05  # [M7] kappa: per year, on capital at risk
    commission: float = 0.65  # [M9] $ per contract to trade


@dataclass(frozen=True)
class Forward:
    e_pnl: float  # $, expected P&L over the horizon
    es: float  # $, Expected Shortfall (positive = loss)
    rar: float | None  # E / ES
    u_hold: float
    u_close: float


def forward(
    g: Grid,
    liquidity_cost: float,
    contracts: float,
    capital_at_risk: float,
    lim: UtilityLimits = UtilityLimits(),
) -> Forward | None:
    """U_hold = E - lambda ES - C - kappa CaR h/252 and U_close = -C_close
    (roll_rules), with C_close the half-spread crossing plus commissions;
    holding pays the same exit cost later. None without the vol-scaled family."""
    h = lim.horizon
    probs = node_probabilities(tuple(sorted({k[1] for k in g.scaled if k[0] == h})))
    if not probs:
        return None
    outcomes = [(p, g.scaled[(h, z, 0)]) for z, p in probs.items()]
    e = sum(p * x for p, x in outcomes)
    es = expected_shortfall(outcomes, lim.alpha)
    c_close = liquidity_cost + abs(contracts) * lim.commission
    u_hold = (
        e
        - lim.risk_aversion * es
        - c_close
        - lim.capital_cost * capital_at_risk * h / 252
    )
    return Forward(e, es, e / es if es > 0 else None, u_hold, -c_close)
