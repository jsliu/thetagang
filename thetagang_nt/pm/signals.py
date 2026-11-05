"""Review signals and the Phase 1 outcome (vault: exit_signals, hold_rules,
close_rules).

A signal opens a review with a priority; it never decides or trades. P0 are
hard (thesis invalid, risk budget breached, data unreliable), P1 time and
profit levels, P2 drift and efficiency. Phase 1 outcome: P0 -> CLOSE review
(REVIEW only, if the data is the problem), none -> HOLD, else REVIEW."""

from __future__ import annotations

from dataclasses import dataclass, field

from thetagang_nt.pm.classify import LONG

P0, P1, P2 = "P0", "P1", "P2"


@dataclass(frozen=True)
class Signal:
    priority: str
    name: str
    detail: str


@dataclass(frozen=True)
class SignalLimits:
    """Register IDs in brackets; all hypotheses. Values the vault leaves "to
    be set" (T5, T8, near expiry) are starting points chosen here."""

    profit_levels: tuple[float, ...] = (0.5, 1.0, 1.5)  # [T1] return on debit
    spread_max_profit: float = 0.8  # [T1] defined-risk debit: share of max profit
    credit_levels: tuple[float, ...] = (0.5, 0.75)  # [T2] credit captured
    delta_review: float = 0.8  # [T4] a single long option's |delta|
    delta_drift: float | None = None  # [T5] share of entry gross delta; None: off
    gt_review: float = 1.0  # [T6] long convexity: GT_h below this
    gt_horizon: int = 5  # trading days for T6
    remaining_reward_risk: float = 0.25  # [T8] below this
    near_expiry_days: int = 7  # a short leg this close to expiry...
    near_money: float = 0.02  # ...within this of its strike


# [T3] initial life (days) -> review when this many days remain; the vault's
# table, taking the start of each window (review earlier, not later)
DTE_WINDOWS = ((90, 28), (180, 60), (365, 120), (730, 270))


def dte_window(initial_dte: int) -> int:
    """Days left at which a strategy that started with `initial_dte` days is
    reviewed: the vault's table, interpolated; shorter lives scale like 3
    months, longer ones keep the LEAPS window."""
    first_life, first_window = DTE_WINDOWS[0]
    if initial_dte <= first_life:
        return round(initial_dte * first_window / first_life)
    for (life0, win0), (life1, win1) in zip(DTE_WINDOWS, DTE_WINDOWS[1:]):
        if initial_dte <= life1:
            return round(win0 + (win1 - win0) * (initial_dte - life0) / (life1 - life0))
    return DTE_WINDOWS[-1][1]


@dataclass(frozen=True)
class ShortLeg:
    dte: int
    strike: float


@dataclass(frozen=True)
class StrategyState:
    """What signals need about one strategy; None = not known."""

    cls: str | None  # convexity class (classify); None: no greeks
    data_flags: tuple[str, ...] = ()  # quality.leg_flags, all legs
    net_debit: float = 0.0  # paid (+) or received (-), $
    value_exec: float | None = None  # executable value now, $
    max_profit: float | None = None  # at expiry, $ (finite, defined risk)
    dte_front: int | None = None  # days to the earliest expiry
    initial_dte: int | None = None  # days to it when opened
    spot: float | None = None
    single_long_delta: float | None = None  # per share, if one long option
    net_delta: float | None = None
    entry_delta: float | None = None
    entry_gross_delta: float | None = None
    gt_h: dict[int, float] = field(default_factory=dict)
    remaining_reward_risk: float | None = None
    short_legs: tuple[ShortLeg, ...] = ()
    prev_cls: str | None = None
    thesis_valid: bool | None = None  # your input
    invalidation: tuple[str, float] | None = None  # ("below" | "above", price)
    worst_case_loss: float | None = None  # $, positive
    loss_budget: float | None = None  # max acceptable loss, $, positive


def _level(value: float, levels: tuple[float, ...]) -> int:
    """How many of the ascending levels `value` has reached."""
    return sum(value >= x for x in levels)


def signals(s: StrategyState, lim: SignalLimits = SignalLimits()) -> list[Signal]:
    out: list[Signal] = []
    # P0: hard
    if s.data_flags:
        out.append(Signal(P0, "data unreliable", ", ".join(sorted(set(s.data_flags)))))
    if s.thesis_valid is False:
        out.append(Signal(P0, "thesis invalidated", "marked invalid"))
    if s.invalidation and s.spot is not None:
        side, level = s.invalidation
        if (side == "below" and s.spot < level) or (side == "above" and s.spot > level):
            out.append(
                Signal(
                    P0, "thesis invalidated", f"price {s.spot:,.2f} {side} {level:,.2f}"
                )
            )
    if (
        s.worst_case_loss is not None
        and s.loss_budget is not None
        and s.worst_case_loss > s.loss_budget
    ):
        out.append(
            Signal(
                P0,
                "risk budget breached",
                f"worst case -${s.worst_case_loss:,.0f} > budget ${s.loss_budget:,.0f}",
            )
        )
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
        window = dte_window(s.initial_dte)
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
    if s.prev_cls is not None and s.cls is not None and s.cls != s.prev_cls:
        out.append(Signal(P2, "class change", f"{s.prev_cls} -> {s.cls}"))
    gt = s.gt_h.get(lim.gt_horizon)
    if s.cls == LONG and gt is not None and gt < lim.gt_review:
        out.append(
            Signal(P2, "efficiency", f"GT_{lim.gt_horizon} {gt:.2f}: decay beats gamma")
        )
    rr = s.remaining_reward_risk
    if rr is not None and rr < lim.remaining_reward_risk:
        out.append(Signal(P2, "low remaining reward", f"reward/risk {rr:.2f}"))
    return out


def outcome(found: list[Signal]) -> str:
    """Phase 1 (hold_rules): any P0 -> CLOSE review, unless the only problem is
    the data (REVIEW - data unreliable); no signal -> HOLD; else REVIEW."""
    p0 = [x for x in found if x.priority == P0]
    if p0:
        if all(x.name == "data unreliable" for x in p0):
            return "REVIEW - data unreliable"
        return "CLOSE review"
    return "REVIEW" if found else "HOLD"
