"""The Short Options Manager's review signals (vault: short_exit_signals,
short_options_manager_framework): strategies opened for a credit, single- or
multi-leg, reviewed as one strategy.

P0 are the shared hard signals (core.review); P1 premium captured (S1), the
front short leg's DTE band (S2), the loss multiple (S3) and assignment exposure
(S4); P2 a tested short strike (S5) and class change. The outcome is the shared
Phase 1 one. Independent of the Long Options Manager: it imports only core."""

from __future__ import annotations

from dataclasses import dataclass

from thetagang_nt.pm.core.profit import capture_ratio
from thetagang_nt.pm.core.review import (
    P1,
    P2,
    Signal,
    StrategyState,
    class_change,
    ex_dividend,
    hard_signals,
    liquidity,
    outcome,
)
from thetagang_nt.pm.short_options.analysis import loss_multiple, moves_out
from thetagang_nt.rules import is_itm

# [S1] what reaching each capture level asks for
CAPTURE_ACTIONS = (
    "review",
    "strong consideration for closing",
    "default preference to close",
    "normally close",
)


@dataclass(frozen=True)
class ShortLimits:
    """Register IDs in brackets; all hypotheses. S3-S5 are "to be set" in the
    vault: S3 takes its research grid, S4 and S5 are starting points chosen
    here."""

    capture_levels: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)  # [S1] credit captured
    # [S2] front short leg's DTE at or below: review / prefer closing or an
    # adjustment review / generally prefer closing
    dte_bands: tuple[int, int, int] = (21, 14, 7)
    loss_multiples: tuple[float, ...] = (1.0, 2.0, 3.0)  # [S3]
    assignment_days: int = 7  # [S4] an in-the-money short leg this close to expiry
    ex_dividend_days: int = 10  # [S4] a short call this close to an ex-date
    tested_moves: float = 1.0  # [S5] spot within this many Move_h of a short strike
    tested_horizon: int = 5  # [S5] trading days in Move_h


def _level(value: float, levels: tuple[float, ...]) -> int:
    return sum(value >= x for x in levels)


def signals(s: StrategyState, lim: ShortLimits = ShortLimits()) -> list[Signal]:
    out = hard_signals(s)
    shorts = [x for x in s.short_legs if x.right in ("C", "P")]
    # P1 [S1]: premium captured
    if s.value_exec is not None and s.net_debit < 0:
        captured = capture_ratio(s.value_exec, s.net_debit)
        n = _level(captured, lim.capture_levels) if captured is not None else 0
        if n:
            action = CAPTURE_ACTIONS[min(n, len(CAPTURE_ACTIONS)) - 1]
            out.append(
                Signal(
                    P1, "premium captured", f"{captured:.0%} of the credit: {action}"
                )
            )
    # P1 [S2]: DTE band of the front short leg
    front = min((x.dte for x in shorts), default=s.dte_front)
    if front is not None:
        review, adjust, close = lim.dte_bands
        if front <= close:
            act = (
                "expiry or assignment intended"
                if s.assignment_intended
                else "generally prefer closing"
            )
        elif front <= adjust:
            act = "prefer closing or an active adjustment review"
        elif front <= review:
            act = "review"
        else:
            act = None
        if act:
            out.append(Signal(P1, "DTE band", f"{front} days left: {act}"))
    # P1 [S3]: the loss multiple
    if s.value_exec is not None:
        m = loss_multiple(s.value_exec, s.net_debit)
        n = _level(m, lim.loss_multiples) if m is not None else 0
        if n:
            out.append(
                Signal(
                    P1,
                    "loss multiple",
                    f"loss {m:.2f}x the credit (level {lim.loss_multiples[n - 1]:g})",
                )
            )
    # P1 [S4]: assignment exposure, near expiry here; before an ex-date below
    out += ex_dividend(s, lim.ex_dividend_days)
    if s.spot and not s.assignment_intended:
        for x in shorts:
            if x.dte <= lim.assignment_days and is_itm(x.right, x.strike, s.spot):
                out.append(
                    Signal(
                        P1,
                        "assignment exposure",
                        f"short {x.strike:g}{x.right} in the money, {x.dte} days left",
                    )
                )
    # P2 [S5]: a short strike tested (in the money counts)
    if s.spot and s.sigma_f:
        for x in shorts:
            k = moves_out(x.right, x.strike, s.spot, s.sigma_f, lim.tested_horizon)
            if k is not None and k <= lim.tested_moves:
                where = "in the money" if k < 0 else f"{k:.1f} moves away"
                out.append(
                    Signal(
                        P2,
                        "short strike tested",
                        f"short {x.strike:g}{x.right} {where}"
                        f" ({lim.tested_horizon}-day moves)",
                    )
                )
    out += class_change(s) + liquidity(s)
    return out


def decide(
    s: StrategyState, lim: ShortLimits = ShortLimits()
) -> tuple[list[Signal], str]:
    """The Short Options Manager's decision: its signals, then the shared
    Phase 1 outcome (no ROLL until the phase 2 comparison exists)."""
    found = signals(s, lim)
    return found, outcome(found)
