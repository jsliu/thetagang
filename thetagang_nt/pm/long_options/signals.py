"""The Long Options Manager's review signals (vault: exit_signals,
long_options_manager_framework): strategies opened for a debit.

P0 are the shared hard signals (core.review); P1 profit on the debit (T1),
the DTE window scaled to the strategy's original life (T3) and a short leg near
expiry; P2 delta (T4, T5), class change, Gamma/Theta efficiency (T6) and low
remaining reward (T8). The outcome is the shared Phase 1 one."""

from __future__ import annotations

from dataclasses import dataclass

from thetagang_nt.pm.core.classify import LONG
from thetagang_nt.pm.core.review import (
    P1,
    P2,
    Signal,
    StrategyState,
    class_change,
    hard_signals,
    outcome,
)

# [T3] initial life (days) -> review when this many days remain; the vault's
# table, taking the start of each window (review earlier, not later)
DTE_WINDOWS = ((90, 28), (180, 60), (365, 120), (730, 270))


@dataclass(frozen=True)
class SignalLimits:
    """Register IDs in brackets; all hypotheses. Values the vault leaves "to
    be set" (T5, T8, near expiry) are starting points chosen here."""

    profit_levels: tuple[float, ...] = (0.5, 1.0, 1.5)  # [T1] return on debit
    spread_max_profit: float = 0.8  # [T1] defined-risk debit: share of max profit
    # [T2] credit captured; superseded by S1: credit strategies now go to the
    # Short Options Manager, so the monitor never reaches this branch
    credit_levels: tuple[float, ...] = (0.5, 0.75)
    delta_review: float = 0.8  # [T4] a single long option's |delta|
    delta_drift: float | None = None  # [T5] share of entry gross delta; None: off
    gt_review: float = 1.0  # [T6] long convexity: GT_h below this
    gt_horizon: int = 5  # trading days for T6
    remaining_reward_risk: float = 0.25  # [T8] below this
    near_expiry_days: int = 7  # a short leg this close to expiry...
    near_money: float = 0.02  # ...within this of its strike
    dte_windows: tuple[tuple[int, int], ...] = DTE_WINDOWS  # [T3]


def dte_window(
    initial_dte: int, windows: tuple[tuple[int, int], ...] = DTE_WINDOWS
) -> int:
    """Days left at which a strategy that started with `initial_dte` days is
    reviewed: the vault's table, interpolated; shorter lives scale like 3
    months, longer ones keep the LEAPS window."""
    first_life, first_window = windows[0]
    if initial_dte <= first_life:
        return round(initial_dte * first_window / first_life)
    for (life0, win0), (life1, win1) in zip(windows, windows[1:]):
        if initial_dte <= life1:
            return round(win0 + (win1 - win0) * (initial_dte - life0) / (life1 - life0))
    return windows[-1][1]


def _level(value: float, levels: tuple[float, ...]) -> int:
    """How many of the ascending levels `value` has reached."""
    return sum(value >= x for x in levels)


def signals(s: StrategyState, lim: SignalLimits = SignalLimits()) -> list[Signal]:
    # P0: hard
    out = hard_signals(s)
    # P1: profit levels, by how the strategy was opened
    if s.value_exec is not None and s.net_debit > 0:
        gain = (s.value_exec - s.net_debit) / s.net_debit
        n = _level(gain, lim.profit_levels)
        if n:
            stage = ("begin", "detailed", "high-priority")[min(n, 3) - 1]
            out.append(
                Signal(P1, "profit level", f"{gain:+.0%} on the debit: {stage} review")
            )
        if s.max_profit:
            share = (s.value_exec - s.net_debit) / s.max_profit
            if share >= lim.spread_max_profit:
                out.append(Signal(P1, "profit level", f"{share:.0%} of max profit"))
    elif s.value_exec is not None and s.net_debit < 0:
        captured = (-s.net_debit - (-s.value_exec)) / -s.net_debit
        if _level(captured, lim.credit_levels):
            out.append(
                Signal(P1, "profit level", f"{captured:.0%} of the credit captured")
            )
    # P1: time
    if s.dte_front is not None and s.initial_dte is not None:
        window = dte_window(s.initial_dte, lim.dte_windows)
        if s.dte_front <= window:
            out.append(
                Signal(
                    P1, "DTE window", f"{s.dte_front} days left (review at {window})"
                )
            )
    if s.spot:
        for leg in s.short_legs:
            close = abs(leg.strike / s.spot - 1) <= lim.near_money
            if leg.dte <= lim.near_expiry_days and close:
                out.append(
                    Signal(
                        P1,
                        "near expiry",
                        f"short {leg.strike:g} near the money, {leg.dte} days left",
                    )
                )
    # P2: drift and efficiency
    if s.single_long_delta is not None and abs(s.single_long_delta) > lim.delta_review:
        out.append(Signal(P2, "delta", f"|delta| {abs(s.single_long_delta):.2f}"))
    if (
        lim.delta_drift is not None
        and s.net_delta is not None
        and s.entry_delta is not None
        and s.entry_gross_delta
    ):
        drift = abs(s.net_delta - s.entry_delta)
        if drift > lim.delta_drift * s.entry_gross_delta:
            out.append(Signal(P2, "delta drift", f"{drift:,.0f} shares since entry"))
    out += class_change(s)
    gt = s.gt_h.get(lim.gt_horizon)
    if s.cls == LONG and gt is not None and gt < lim.gt_review:
        out.append(
            Signal(P2, "efficiency", f"GT_{lim.gt_horizon} {gt:.2f}: decay beats gamma")
        )
    rr = s.remaining_reward_risk
    if rr is not None and rr < lim.remaining_reward_risk:
        out.append(Signal(P2, "low remaining reward", f"reward/risk {rr:.2f}"))
    return out


def decide(
    s: StrategyState, lim: SignalLimits = SignalLimits()
) -> tuple[list[Signal], str]:
    """The Long Options Manager's decision: its signals, then the shared
    Phase 1 outcome (no ROLL until the phase 2 comparison exists)."""
    found = signals(s, lim)
    return found, outcome(found)
