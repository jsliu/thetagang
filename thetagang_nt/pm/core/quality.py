"""Data quality and executable value (vault: data_quality_controls).

Selling a long leg hits the bid and buying back a short lifts the ask, so a
strategy's executable value V_exec sits below its mid value V_mid; the gap is
the cost of leaving now. A flag on any leg makes the whole strategy
"REVIEW - data unreliable" downstream (signals)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from thetagang_nt.rules import Leg


@dataclass(frozen=True)
class Quote:
    bid: float | None
    ask: float | None
    time: datetime | None = None  # when it was quoted, UTC


@dataclass(frozen=True)
class QualityLimits:
    """T9 (hypotheses; the vault leaves them "to be set")."""

    stale_secs: float = 300.0  # a quote older than this, in market hours
    wide_spread: float = 0.25  # (ask - bid) / mid above this


def spread_pct(quote: Quote) -> float | None:
    if quote.bid is None or quote.ask is None:
        return None
    mid = (quote.bid + quote.ask) / 2
    return (quote.ask - quote.bid) / mid if mid > 0 else None


def leg_flags(
    leg: Leg,
    quote: Quote | None,
    greeks: Any | None,
    now: datetime,
    limits: QualityLimits = QualityLimits(),
) -> tuple[str, ...]:
    """Why this leg's data can't be trusted; empty when it can. `greeks`: an
    object with delta, gamma and vol (implied), or None."""
    flags = []
    if quote is None or quote.bid is None or quote.ask is None or quote.bid > quote.ask:
        flags.append("missing quote")
    else:
        if (
            quote.time is not None
            and (now - quote.time).total_seconds() > limits.stale_secs
        ):
            flags.append("stale quote")
        if leg.qty > 0 and quote.bid == 0:
            flags.append("zero bid")  # the liquidation value may be unattainable
        spread = spread_pct(quote)
        if spread is not None and spread > limits.wide_spread:
            flags.append("wide spread")
    if greeks is None:
        flags.append("no greeks")
    else:
        if not -1 <= greeks.delta <= 1:
            flags.append("delta out of range")
        if greeks.gamma < 0:
            flags.append("negative gamma")
        vol = getattr(greeks, "vol", None)
        if vol is not None and not 0.01 <= vol <= 5:
            flags.append("implausible IV")
    if leg.multiplier != 100:
        flags.append("non-standard multiplier")  # adjusted or mini contract
    return tuple(flags)


def values(
    legs: tuple[Leg, ...], quotes: dict[Any, Quote]
) -> tuple[float | None, float | None]:
    """(V_mid, V_exec) of a strategy in $; (None, None) if any leg lacks a
    two-sided quote. Never mix the two in one P&L figure."""
    mid = exe = 0.0
    for p in legs:
        q = quotes.get(p.instrument_id)
        if q is None or q.bid is None or q.ask is None or q.bid > q.ask:
            return None, None
        n = p.qty * p.multiplier
        mid += n * (q.bid + q.ask) / 2
        exe += n * (q.bid if p.qty > 0 else q.ask)
    return mid, exe
