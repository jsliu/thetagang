"""Profit and capital efficiency (vault: profit_efficiency).

Backward-looking measures (return on debit, capture) say how the trade has
gone; only the forward-looking remaining reward/risk, from today's executable
value, bears on whether to keep it (past P&L is sunk)."""

from __future__ import annotations

import math

from thetagang_nt.rules import StructureFigures


def return_on_debit(value: float, net_debit: float) -> float | None:
    """(value - debit) / debit, for strategies opened for a debit."""
    return (value - net_debit) / net_debit if net_debit > 0 else None


def capture_ratio(value: float, net_debit: float) -> float | None:
    """Share of the credit captured, at most 1, for strategies opened for a
    credit (net_debit < 0). `value` is negative: minus the buy-back cost."""
    if net_debit >= 0:
        return None
    credit = -net_debit
    return (credit - (-value)) / credit


def value_bounds(fig: StructureFigures) -> tuple[float | None, float | None]:
    """(V_floor, V_ceil): the strategy's value at expiry, worst and best; None
    where undefined (several expiries, or unlimited)."""

    def bound(x: float | None) -> float | None:
        return None if x is None or math.isinf(x) else fig.cost + x

    return bound(fig.max_loss), bound(fig.max_profit)


def remaining(
    v_exec: float, v_floor: float | None, v_ceil: float | None
) -> tuple[float | None, float | None, float | None]:
    """(remaining profit, remaining risk, remaining reward/risk) in $ from the
    executable value; None where a bound is undefined."""
    profit = None if v_ceil is None else v_ceil - v_exec
    risk = None if v_floor is None else v_exec - v_floor
    ratio = profit / risk if profit is not None and risk else None
    return profit, risk, ratio
