"""What both managers share when they review a strategy (vault: exit_signals,
short_exit_signals, hold_rules, close_rules): the signal record, the strategy's
state, the hard P0 signals and the class-change signal both signal tables list,
and the Phase 1 outcome.

A signal opens a review with a priority; it never decides or trades. Phase 1
outcome: P0 -> CLOSE review (REVIEW only, if the data is the problem), none ->
HOLD, else REVIEW.

Which manager reviews a strategy is its management category: how it was
opened (long premium for a debit, short premium for a credit). That is not its
convexity class, which comes from net greeks (classify)."""

from __future__ import annotations

from dataclasses import dataclass, field

P0, P1, P2 = "P0", "P1", "P2"

LONG_PREMIUM = "long options"  # opened for a debit: the Long Options Manager
SHORT_PREMIUM = "short options"  # opened for a credit: the Short Options Manager


def manager(net_debit: float) -> str:
    """The management category from the entry net debit ($, + paid)."""
    return SHORT_PREMIUM if net_debit < 0 else LONG_PREMIUM


@dataclass(frozen=True)
class Signal:
    priority: str
    name: str
    detail: str


@dataclass(frozen=True)
class ShortLeg:
    dte: int
    strike: float
    right: str = ""  # "C" or "P"; the Short Options Manager needs it
    dividend: float | None = None  # next dividend before its expiry, $ per share
    ex_days: int | None = None  # days to that ex-date
    extrinsic: float | None = None  # its value above intrinsic, $ per share


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
    worst_case_loss: float | None = None  # $, positive: the grid's worst WCL_h
    stress_loss: float | None = None  # $, positive: the worst stress scenario
    loss_budget: float | None = None  # max acceptable loss, $, positive
    sigma_f: float | None = None  # forecast annualised realised vol (M6)
    assignment_intended: bool = False  # your input: expiry/assignment is the plan


def _money(x: float) -> str:
    """$500, or -$485 once the loss is already past the max acceptable loss."""
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def hard_signals(s: StrategyState) -> list[Signal]:
    """P0, the same in both signal tables: data, thesis, risk budget."""
    out: list[Signal] = []
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
                f"worst case -${s.worst_case_loss:,.0f} > budget {_money(s.loss_budget)}",
            )
        )
    if (
        s.stress_loss is not None
        and s.loss_budget is not None
        and s.stress_loss > s.loss_budget
    ):
        out.append(
            Signal(
                P0,
                "risk budget breached",
                f"stress loss -${s.stress_loss:,.0f} > budget {_money(s.loss_budget)}",
            )
        )
    return out


def ex_dividend(s: StrategyState, window: int) -> list[Signal]:
    """P1 (assignment_risk, S4): an in-the-money short call whose extrinsic value
    is below the next dividend, close to the ex-date: exercising is rational for
    the holder. Any manager's: a debit spread's short leg can be assigned too."""
    if s.assignment_intended or not s.spot:
        return []
    return [
        Signal(
            P1,
            "assignment exposure",
            f"short {x.strike:g}C: ex-dividend in {x.ex_days} days,"
            f" ${x.dividend:.2f} > ${x.extrinsic:.2f} extrinsic",
        )
        for x in s.short_legs
        if x.right == "C"
        and x.strike < s.spot
        and x.dividend is not None
        and x.ex_days is not None
        and x.extrinsic is not None
        and x.ex_days <= window
        and x.dividend > x.extrinsic
    ]


def class_change(s: StrategyState) -> list[Signal]:
    """P2, in both tables: the convexity class differs from the last run."""
    if s.prev_cls is not None and s.cls is not None and s.cls != s.prev_cls:
        return [Signal(P2, "class change", f"{s.prev_cls} -> {s.cls}")]
    return []


def outcome(found: list[Signal]) -> str:
    """Phase 1 (hold_rules): any P0 -> CLOSE review, unless the only problem is
    the data (REVIEW - data unreliable); no signal -> HOLD; else REVIEW."""
    p0 = [x for x in found if x.priority == P0]
    if p0:
        if all(x.name == "data unreliable" for x in p0):
            return "REVIEW - data unreliable"
        return "CLOSE review"
    return "REVIEW" if found else "HOLD"
