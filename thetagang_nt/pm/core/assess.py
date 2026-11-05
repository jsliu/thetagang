"""One strategy's Phase 1 assessment (vault: position_quality): its values,
data flags, convexity class, net greeks, efficiency and remaining reward/risk,
from the pm modules. Pure: the monitor passes in quotes and greeks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from thetagang_nt.pm.core.classify import GRID_MOVES, Tolerance, classify, net_greeks
from thetagang_nt.pm.core.efficiency import Efficiency, efficiency
from thetagang_nt.pm.core.profit import remaining, value_bounds
from thetagang_nt.pm.core.quality import QualityLimits, Quote, leg_flags, values
from thetagang_nt.rules import Structure, StructureFigures, option_label


@dataclass(frozen=True)
class Assessment:
    v_mid: float | None  # $, at mid
    v_exec: float | None  # $, what closing now would fetch
    flags: tuple[str, ...]  # "AA 20261218 40P: stale quote", ...
    cls: str | None  # convexity class; None without greeks
    net: dict[str, float] | None  # delta (shares), gamma, theta ($/day), vega
    eff: Efficiency
    remaining_rr: float | None  # remaining reward / risk from V_exec
    dte_front: int  # days to the earliest expiry


def assess(
    s: Structure,
    fig: StructureFigures,
    greeks: dict[Any, Any],
    quotes: dict[Any, Quote],
    spot: float | None,
    sigma_f: float | None,
    today: date,
    now: datetime,
    limits: QualityLimits = QualityLimits(),
    tol: Tolerance = Tolerance(),
    moves: tuple[float, ...] = GRID_MOVES,
) -> Assessment:
    v_mid, v_exec = values(s.legs, quotes)
    flags = tuple(
        f"{option_label(p)}: {flag}"
        for p in s.legs
        for flag in leg_flags(
            p, quotes.get(p.instrument_id), greeks.get(p.instrument_id), now, limits
        )
    )
    cls = classify(s.legs, greeks, spot, today, tol, moves)
    net = net_greeks(s.legs, greeks)
    eff = (
        efficiency(cls, net["gamma"], net["theta"], spot, sigma_f, today)
        if net and spot
        else Efficiency()
    )
    rr = remaining(v_exec, *value_bounds(fig))[2] if v_exec is not None else None
    return Assessment(v_mid, v_exec, flags, cls, net, eff, rr, fig.dte)
