"""thetagang_nt: ported decision rules and config loading (run with the
nautilus extra installed: pytest tests_nt)."""

from datetime import date
from types import SimpleNamespace

import pytest

from thetagang_nt.rules import (
    Candidate,
    Leg,
    Plan,
    RollWhen,
    compute_beta,
    credit_pnl,
    debit_take_profit_value,
    pick_contract,
    roll_strike_limit,
    should_close,
    should_roll,
    size_plans,
    split_option_legs,
    vix_hedge_weight,
)

EXP = date(2026, 12, 18)


def opt(right: str, strike: float, qty: float, avg: float, mid: float | None) -> Leg:
    return Leg(f"{right}{strike}", "AA", right, strike, EXP, qty, avg, mid)


def stock(qty: float, avg: float = 30.0) -> Leg:
    return Leg("AA", "AA", "", 0.0, None, qty, avg, 31.0, 1.0)


def cand(strike: float, dte: int, delta: float, px: float) -> Candidate:
    return Candidate(strike, strike, EXP, dte, delta, px)


def test_split_pairs_credit_then_debit_and_flags_diagonals() -> None:
    short_put, long_put = opt("P", 30, -2, 1.0, 0.4), opt("P", 25, 2, 0.3, 0.1)
    debit_long, debit_short = opt("C", 35, 1, 2.0, 2.5), opt("C", 40, -1, 0.8, 1.0)
    pairs, unpaired, mixed = split_option_legs(
        [short_put, long_put, debit_long, debit_short]
    )
    assert pairs == [(short_put, long_put), (debit_short, debit_long)]
    assert unpaired == [] and mixed == set()

    # different quantities: no vertical; a short and a long put = diagonal
    legs = [opt("P", 30, -2, 1, 1), opt("P", 25, 1, 1, 1)]
    pairs, unpaired, mixed = split_option_legs(legs)
    assert pairs == [] and mixed == {"P"}


def test_covered_calls_never_pair_with_long_calls() -> None:
    short_call, long_call = opt("C", 40, -1, 1.0, 0.5), opt("C", 35, 1, 2.0, 2.0)
    pairs, _, mixed = split_option_legs([stock(100), short_call, long_call])
    assert pairs == [] and "C" not in mixed


def test_roll_rules_match_roll_when() -> None:
    rw = RollWhen(pnl=0.9, dte=10, min_pnl=0.7, close_at_pnl=0.9)
    assert should_roll(rw, "P", False, 5, 0.75)  # near expiry, enough P&L
    assert not should_roll(rw, "P", False, 5, 0.5)
    assert should_roll(rw, "P", False, 40, 0.95)  # P&L target
    assert not should_roll(RollWhen(itm={"C": True, "P": False}), "P", True, 1, 1.0)
    always = RollWhen(always_when_itm={"C": True, "P": False})
    assert should_roll(always, "C", True, 99, -5)
    assert not should_roll(RollWhen(max_dte=30), "P", False, 45, 0.99)
    assert should_close(rw, 0.95) and not should_close(rw, 0.9)


def test_roll_strike_limit_is_symmetric() -> None:
    # captured 1.0: a put may roll up to strike + 1, a call down to strike - 1
    assert roll_strike_limit("P", 30, 1.0, False, None) == 31
    assert roll_strike_limit("C", 30, 1.0, False, None) == 29
    assert roll_strike_limit("P", 30, 1.0, True, None) == 30  # ITM: same or safer
    assert roll_strike_limit("P", 30, 1.0, False, 30.5) == 30.5  # configured cap


def test_spread_pnl_and_take_profit() -> None:
    short, long = opt("P", 30, -1, 1.0, 0.3), opt("P", 25, 1, 0.2, 0.1)
    assert credit_pnl(short, long) == pytest.approx((0.8 - 0.2) / 0.8)
    # debit 2 on a width of 5: 80% of the max profit 3 -> value 4.4
    d_short, d_long = opt("C", 40, -1, 1.0, 1.0), opt("C", 35, 1, 3.0, 3.0)
    assert debit_take_profit_value(d_short, d_long, 0.8) == pytest.approx(4.4)


def test_pick_contract_nearest_expiry_then_highest_delta_under_target() -> None:
    cands = [
        cand(28, 30, -0.25, 0.6),
        cand(29, 30, -0.31, 0.8),
        cand(27, 30, -0.18, 0.4),
        cand(28, 37, -0.29, 0.9),
    ]
    assert pick_contract(cands, "P", 30, 0.3, 0.05).strike == 28
    # price floor 0.5: 27 is too cheap, 28 still fits
    assert pick_contract(cands, "P", 30, 0.3, 0.5).strike == 28
    # nothing under the target and min_price set: smallest delta above it
    assert pick_contract([cand(29, 30, -0.31, 0.8)], "P", 30, 0.3, 0.05).strike == 29
    assert pick_contract([cand(29, 30, -0.31, 0.8)], "P", 30, 0.3, 0.0) is None


def test_size_plans_by_weight_never_wrong_way() -> None:
    a = Plan(2.0, None, unit_spx=-0.5, priced=True)
    b = Plan(1.0, 3, unit_spx=-0.5, priced=True)
    wrong = Plan(1.0, None, unit_spx=0.5, priced=True)
    size_plans([a, b, wrong], need=-6.0)  # bring SPX delta down by 6
    assert (a.units, b.units, wrong.units) == (8, 3, 0)
    assert "capped" in b.note and "wrong way" in wrong.note


def test_vix_weight_uses_first_matching_band() -> None:
    bands = [
        SimpleNamespace(lower_bound=None, upper_bound=15.0, weight=0.0),
        SimpleNamespace(lower_bound=15.0, upper_bound=30.0, weight=0.01),
        SimpleNamespace(lower_bound=30.0, upper_bound=50.0, weight=0.005),
        SimpleNamespace(lower_bound=50.0, upper_bound=None, weight=0.0),
    ]
    weights = [vix_hedge_weight(bands, v) for v in (12, 15, 29.9, 40, 60)]
    assert weights == [0, 0.01, 0.01, 0.005, 0]


def test_beta_of_a_scaled_series() -> None:
    index = {i: 100 * (1.01 if i % 2 else 0.995) ** i for i in range(40)}
    stock_closes = {d: 50 * (v / 100) ** 2 for d, v in index.items()}  # 2x log returns
    assert compute_beta(stock_closes, index) == pytest.approx(2.0)
    assert compute_beta({1: 1.0}, index) is None


def test_sample_config_loads() -> None:
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.config import load, node_config

    app = load("thetagang.nt.toml")
    assert app.strategy is not None and app.strategy.dry_run
    assert app.strategy.roll_when.rules().close_at_pnl == 0.9
    assert node_config(app).trader_id.value == "THETAGANG-001"


def test_roles_split_monitoring_from_trading() -> None:
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.config import load, split

    app = load("thetagang.nt.toml")  # has a [strategy] section, client_id 11
    monitor, strategy, cid = split(app, "monitor")
    assert strategy is None and monitor.publish and cid == 12  # can never trade
    monitor, strategy, cid = split(app, "trade")
    assert strategy is app.strategy and not monitor.publish and cid == 11  # no doubled metrics
    monitor, strategy, cid = split(app, "all")
    assert strategy is app.strategy and monitor.publish and cid == 11
    with pytest.raises(ValueError, match="role"):
        split(app, "everything")
