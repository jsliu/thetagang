"""A small volatility surface and sticky-delta model risk (vault:
volatility_surface, volatility_scenarios). One smile per held expiry, from a few
out-of-the-money strikes around the money, implied by the same American model as
the grid. The grid stays sticky strike (M4); this reports how much its P&L would
change if IV stayed with moneyness instead, the vault's model-risk indicator.

ponytail: moneyness on spot, not the forward, and linear interpolation with flat
ends, on a few strikes; a real fit, calendar and butterfly arbitrage checks and
other expiries wait until something decides on the surface. Reported only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from thetagang_nt.pm.core.american import implied_vol
from thetagang_nt.pm.core.efficiency import calendar_days
from thetagang_nt.pm.core.quality import Quote, spread_pct
from thetagang_nt.pm.core.scenarios import RATE, PriceModel, _priced, _value
from thetagang_nt.rules import Leg

MIN_POINTS = 3


@dataclass(frozen=True)
class Smile:
    """(ln(K / S), implied vol), by moneyness."""

    points: tuple[tuple[float, float], ...]

    def covers(self, m: float) -> bool:
        """Inside the quoted strikes; past them, iv_at is flat and shows no skew."""
        return self.points[0][0] <= m <= self.points[-1][0]

    def iv_at(self, m: float) -> float:
        """Linear between points; flat past the last quoted one."""
        pts = self.points
        if m <= pts[0][0]:
            return pts[0][1]
        if m >= pts[-1][0]:
            return pts[-1][1]
        for (m0, v0), (m1, v1) in zip(pts, pts[1:], strict=False):
            if m <= m1:
                return v0 + (v1 - v0) * (m - m0) / (m1 - m0)
        return pts[-1][1]  # unreachable: m < the last point


def strikes_around(
    listed: list[float], underlying: float, offsets: tuple[float, ...]
) -> list[float]:
    """The listed strikes nearest these moves from the underlying."""
    if not listed:
        return []
    return sorted(
        {min(listed, key=lambda k: abs(k - underlying * (1 + x))) for x in offsets}
    )


def measure(
    rows: list[tuple[float, bool, Quote]],
    underlying: float,
    years: float,
    dividends: tuple[tuple[float, float], ...],
    wide_spread: float,
) -> Smile | None:
    """A smile from (strike, is call, quote); wide or missing quotes are left out.
    None with fewer than MIN_POINTS."""
    points = {}
    for strike, call, quote in rows:
        pct = spread_pct(quote)
        if pct is None or pct > wide_spread or not quote.bid:
            continue
        mid = (quote.bid + quote.ask) / 2
        iv = implied_vol(mid, underlying, strike, years, call, RATE, dividends)
        if iv is not None:
            points[math.log(strike / underlying)] = iv
    if len(points) < MIN_POINTS:
        return None
    return Smile(tuple(sorted(points.items())))


def sticky_gaps(
    legs: tuple[Leg, ...],
    greeks: dict[Any, Any],
    today: date,
    smiles: dict[date, Smile],
    moves: tuple[float, ...],
    h: int,
    model: PriceModel,
) -> tuple[dict[float, float], float] | None:
    """{price move: sticky-delta P&L minus sticky-strike P&L} ($) after h trading
    days, IV otherwise unchanged, and the share of legs inside a quoted smile. A
    leg without a smile, or past its quoted strikes, shows little or no gap
    however skewed the market is, so a low share means the gap understates the
    risk. None if no expiry has a smile, or a leg can't be priced."""
    priced = _priced(legs, greeks, today, model)
    if priced is None:
        return None
    if not any(p.leg.expiry in smiles for p in priced):
        return None
    covered = sum(
        p.leg.expiry in smiles
        and smiles[p.leg.expiry].covers(math.log(p.leg.strike / p.und))
        for p in priced
    )
    days = calendar_days(today, h)

    def shift(p: Any, x: float) -> float:
        """IV points as the strike's moneyness moves along its expiry's smile."""
        smile = smiles.get(p.leg.expiry)
        if smile is None or p.und * (1 + x) <= 0:
            return 0.0
        before = math.log(p.leg.strike / p.und)
        after = math.log(p.leg.strike / (p.und * (1 + x)))
        return (smile.iv_at(after) - smile.iv_at(before)) * 100

    gaps = {}
    for x in moves:
        fixed = _value(
            priced, today, days, lambda p, x=x: x, lambda p: 0.0, RATE, model
        )
        moved = _value(
            priced,
            today,
            days,
            lambda p, x=x: x,
            lambda p, x=x: shift(p, x),
            RATE,
            model,
        )
        gaps[x] = moved - fixed
    return gaps, covered / len(priced)
