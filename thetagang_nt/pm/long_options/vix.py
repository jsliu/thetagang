"""Long VIX calls (vault: long_options_manager/instrument_specific_management,
vix_options_management): the Long Options Manager with three VIX rules.

- V2 VIX profit levels replace T1 (vix_profit_capture): modest, early capture,
  because VIX gains are spikes that mean-revert.
- V3 loss reviews, on the same return on debit (vix_risk_management): reviews,
  not stop-losses.
- V7 a fixed time review replaces the life-scaled DTE window T3
  (vix_roll_management): decide before the last trading session, since the
  option settles to the SOQ, which can gap from the last trade.

Everything else is the general manager's: P0 hard signals, a short leg near
expiry, and the P2 signals, read on the matching VIX future (the greeks'
underlying), not spot VIX. Not here yet: V4 premium-at-risk and V5
losing-streak limits and V6 regime (the vault sets no values), and the V1 roll
candidates (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass

from thetagang_nt.pm.core.review import (
    P1,
    Signal,
    StrategyState,
    hard_signals,
    outcome,
)
from thetagang_nt.pm.long_options.signals import (
    SignalLimits,
    drift_signals,
    near_expiry_signals,
)

PROFIT_STAGES = (
    "profit review",
    "evaluate early close",
    "consider taking profit",
    "strong profit-taking consideration",
)
LOSS_STAGES = ("loss review", "stronger case for closing or reducing")


@dataclass(frozen=True)
class VixLimits:
    """Register IDs in brackets; all hypotheses."""

    profit_levels: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30)  # [V2] return on debit
    loss_levels: tuple[float, ...] = (-0.25, -0.35)  # [V3] return on debit, at or below
    # [V7] the vault leaves it "to be set": two weeks, a starting point chosen
    # here so the review comes well before the last trading session
    review_dte: int = 14


def vix_signals(s: StrategyState, lim: VixLimits = VixLimits()) -> list[Signal]:
    """P1: V2 profit levels and V3 loss reviews on the return on debit at
    executable value, and the V7 time review."""
    out: list[Signal] = []
    if s.value_exec is not None and s.net_debit > 0:
        r = (s.value_exec - s.net_debit) / s.net_debit
        n = sum(r >= x for x in lim.profit_levels)
        if n:
            stage = PROFIT_STAGES[min(n, len(PROFIT_STAGES)) - 1]
            out.append(
                Signal(P1, "VIX profit level", f"{r:+.0%} on the debit: {stage}")
            )
        n = sum(r <= x for x in lim.loss_levels)
        if n:
            stage = LOSS_STAGES[min(n, len(LOSS_STAGES)) - 1]
            out.append(Signal(P1, "VIX loss review", f"{r:+.0%} on the debit: {stage}"))
    if s.dte_front is not None and s.dte_front <= lim.review_dte:
        out.append(
            Signal(
                P1,
                "VIX time review",
                f"{s.dte_front} days left (review at {lim.review_dte}):"
                " close or roll before the last trading session",
            )
        )
    return out


def decide(
    s: StrategyState,
    lim: SignalLimits = SignalLimits(),
    vix: VixLimits = VixLimits(),
) -> tuple[list[Signal], str]:
    """A long VIX strategy's decision: the general signals without T1 and T3,
    the VIX ones in their place, then the shared Phase 1 outcome."""
    found = (
        hard_signals(s)
        + vix_signals(s, vix)
        + near_expiry_signals(s, lim)
        + drift_signals(s, lim)
    )
    return found, outcome(found)
