"""The Short Options Manager's position analysis (vault: premium_capture_model,
loss_control, short_exit_signals S5, scenario_engine Move_h). Pure; the share
of the credit captured is core.profit.capture_ratio (profit_efficiency owns it).

Sign convention: `value_exec` is the strategy's executable value in $ (negative
for a short-premium position: minus the cost to buy it back); `net_debit` is
paid (+) or received (-)."""

from __future__ import annotations

import math


def premium_to_risk(value_exec: float, worst_case_loss: float | None) -> float | None:
    """RemainingPremium / RemainingRisk = C_close / WCL_h: the premium still to
    earn per $ of risk still carried. Phase 1 WCL_h: the expiry value floor
    (defined risk only), until the scenario grid exists."""
    if not worst_case_loss or worst_case_loss <= 0:
        return None
    return -value_exec / worst_case_loss


def loss_multiple(value_exec: float, net_debit: float) -> float | None:
    """(CurrentCloseCost - InitialCredit) / InitialCredit: 0 at breakeven, 1
    when the loss equals the credit, negative in profit. None unless opened for
    a credit."""
    if net_debit >= 0:
        return None
    credit = -net_debit
    return (-value_exec - credit) / credit


def moves_out(
    right: str, strike: float, spot: float, sigma_f: float, horizon: int
) -> float | None:
    """How far spot is on the safe (out-of-the-money) side of a short strike,
    in Move_h = S * sigma_f * sqrt(h / 252); negative once in the money."""
    move = spot * sigma_f * math.sqrt(horizon / 252)
    if move <= 0 or right not in ("C", "P"):
        return None
    return (strike - spot if right == "C" else spot - strike) / move
