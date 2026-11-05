"""Gamma/Theta efficiency (vault: gamma_theta_efficiency, theta_decay).

For long convexity: how much gamma the strategy buys per $ of daily decay, and
whether, at the forecast realised volatility, gamma gains should cover decay
over h trading days (GT_h > 1). Raw GT scales with 1/S^2, so compare GT_h or
the breakeven volatility sigma_BE (comparable with implied vol) across
underlyings. Short convexity reports TG; mixed reports nothing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from thetagang_nt.pm.classify import LONG, SHORT

HORIZONS = (1, 5, 10)  # trading days


def calendar_days(today: date, h: int) -> int:
    """Calendar days from today to the close of the h-th trading day after it
    (1 trading day from a Friday = 3). ponytail: weekends only; exchange
    holidays not counted, add the NYSE calendar if it matters."""
    d, left = today, h
    while left:
        d += timedelta(days=1)
        if d.weekday() < 5:
            left -= 1
    return (d - today).days


@dataclass(frozen=True)
class Efficiency:
    gt: float | None = None  # long: gamma / |theta|, shares per $1 per $ a day
    gt_h: dict[int, float] = field(default_factory=dict)  # long: by horizon
    sigma_be: dict[int, float] = field(default_factory=dict)  # long: annualised
    tg: float | None = None  # short: theta / |gamma|


def efficiency(
    cls: str | None,
    gamma: float,
    theta: float,
    spot: float,
    sigma_f: float | None,
    today: date,
) -> Efficiency:
    """`gamma`, `theta`: the strategy's net (shares per $1; $ per calendar day).
    `sigma_f`: forecast annualised realised volatility (M6)."""
    if cls == SHORT and gamma:
        return Efficiency(tg=theta / abs(gamma))
    if cls != LONG or gamma <= 0 or theta >= 0 or spot <= 0:
        return Efficiency()
    decay = abs(theta)
    gt_h, sigma_be = {}, {}
    for h in HORIZONS:
        d = calendar_days(today, h)
        sigma_be[h] = math.sqrt(2 * decay * d * 252 / (gamma * spot**2 * h))
        if sigma_f is not None:
            gt_h[h] = 0.5 * gamma * spot**2 * sigma_f**2 * (h / 252) / (decay * d)
    return Efficiency(gt=gamma / decay, gt_h=gt_h, sigma_be=sigma_be)
