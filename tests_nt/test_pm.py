"""thetagang_nt.pm: the Portfolio Manager's pure functions, checked against
hand calculations (vault: testing_strategy)."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from thetagang_nt.pm.classify import LONG, MIXED, SHORT, bs_gamma, classify, net_greeks
from thetagang_nt.pm.efficiency import calendar_days, efficiency
from thetagang_nt.pm.profit import (
    capture_ratio,
    remaining,
    return_on_debit,
    value_bounds,
)
from thetagang_nt.pm.quality import Quote, leg_flags, values
from thetagang_nt.pm.signals import (
    P0,
    P1,
    P2,
    ShortLeg,
    StrategyState,
    dte_window,
    outcome,
    signals,
)
from thetagang_nt.pm.strategy_id import Seen, auto_ids
from thetagang_nt.rules import Leg, find_structures, structure_figures

TODAY = date(2026, 11, 2)  # a Monday
EXP = TODAY + timedelta(days=30)
NOW = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)


def leg(iid, right, strike, qty, avg=1.0, mid=1.0, expiry=EXP, mult=100.0):
    return Leg(iid, "AA", right, strike, expiry, qty, avg, mid, mult)


def greeks(delta=0.5, gamma=0.05, theta=-0.03, vega=0.1, vol=0.3):
    return SimpleNamespace(delta=delta, gamma=gamma, theta=theta, vega=vega, vol=vol)


# ------------------------------------------------------------------ quality


def test_executable_value_sells_at_the_bid_and_buys_back_at_the_ask():
    spread = (leg("c40", "C", 40, 1), leg("c45", "C", 45, -1))
    quotes = {"c40": Quote(2.9, 3.1), "c45": Quote(0.9, 1.1)}
    assert values(spread, quotes) == pytest.approx(
        (200.0, 180.0)
    )  # mid 3 - 1; 2.9 - 1.1
    assert values(spread, {"c40": Quote(2.9, 3.1)}) == (None, None)  # a leg unquoted


def test_leg_flags_catch_unreliable_data():
    long = leg("c40", "C", 40, 1)
    ok = leg_flags(long, Quote(2.9, 3.1, NOW), greeks(), NOW)
    assert ok == ()
    assert leg_flags(long, Quote(None, 3.1), greeks(), NOW) == ("missing quote",)
    assert leg_flags(long, Quote(3.2, 3.1), greeks(), NOW) == (
        "missing quote",
    )  # crossed
    old = NOW - timedelta(minutes=10)
    assert "stale quote" in leg_flags(long, Quote(2.9, 3.1, old), greeks(), NOW)
    assert "zero bid" in leg_flags(long, Quote(0.0, 0.05), greeks(), NOW)
    assert "wide spread" in leg_flags(
        long, Quote(0.5, 1.0), greeks(), NOW
    )  # 67% of mid
    assert "no greeks" in leg_flags(long, Quote(2.9, 3.1), None, NOW)
    assert "implausible IV" in leg_flags(long, Quote(2.9, 3.1), greeks(vol=9.0), NOW)
    mini = leg("m", "C", 40, 1, mult=10.0)
    assert "non-standard multiplier" in leg_flags(mini, Quote(2.9, 3.1), greeks(), NOW)


# ----------------------------------------------------------- classification


def test_black_scholes_gamma_matches_a_hand_calculation():
    # S = K = 100, T = 1, vol 20%: d1 = 0.1, phi(0.1) = 0.396953 -> / (100 x 0.2)
    assert bs_gamma(100, 100, 1.0, 0.2) == pytest.approx(0.0198476, rel=1e-5)


def test_class_comes_from_greeks_not_from_the_debit():
    call = (leg("c", "C", 100, 1),)
    assert classify(call, {"c": greeks()}, 100.0, TODAY) == LONG

    def spread_greeks(spot, theta40, theta45):
        years = 30 / 365
        return {
            "c40": greeks(gamma=bs_gamma(spot, 40, years, 0.3), theta=theta40),
            "c45": greeks(gamma=bs_gamma(spot, 45, years, 0.3), theta=theta45),
        }

    spread = (leg("c40", "C", 40, 1), leg("c45", "C", 45, -1))  # bought for a debit
    # above the middle of the strikes the short 45 dominates: short gamma, + theta
    assert classify(spread, spread_greeks(46.0, -0.02, -0.04), 46.0, TODAY) == SHORT
    # below the middle it is long gamma now, but a +5% move flips the sign: mixed
    assert classify(spread, spread_greeks(41.0, -0.04, -0.02), 41.0, TODAY) == MIXED


def test_multi_expiry_is_mixed_and_missing_greeks_is_unknown():
    cal = (leg("s", "C", 40, -1), leg("l", "C", 40, 1, expiry=EXP + timedelta(days=60)))
    assert classify(cal, {"s": greeks(), "l": greeks()}, 40.0, TODAY) == MIXED
    assert classify((leg("c", "C", 40, 1),), {}, 40.0, TODAY) is None
    net = net_greeks((leg("c", "C", 40, 2),), {"c": greeks(delta=0.5, theta=-0.03)})
    assert net["delta"] == pytest.approx(100) and net["theta"] == pytest.approx(-6)


# ------------------------------------------------------------- efficiency


def test_calendar_days_count_weekends():
    assert calendar_days(TODAY, 1) == 1  # Monday -> Tuesday
    assert calendar_days(TODAY, 5) == 7  # -> next Monday
    assert calendar_days(TODAY, 10) == 14
    assert calendar_days(date(2026, 11, 6), 1) == 3  # Friday -> Monday


def test_gamma_theta_efficiency_and_breakeven_vol():
    # gamma 5 shares/$1, theta -$10/day, S 100, forecast vol 20%, h = 1 (1 calendar day)
    e = efficiency(LONG, 5.0, -10.0, 100.0, 0.2, TODAY)
    assert e.gt == pytest.approx(0.5)
    assert e.gt_h[1] == pytest.approx(0.5 * 5 * 100**2 * 0.04 / 252 / 10)  # 0.3968
    assert e.sigma_be[1] == pytest.approx(
        (2 * 10 * 252 / (5 * 100**2)) ** 0.5
    )  # 0.3175
    for h in (1, 5, 10):  # GT_h = (sigma_f / sigma_BE)^2
        assert e.gt_h[h] == pytest.approx((0.2 / e.sigma_be[h]) ** 2)
    assert efficiency(SHORT, -5.0, 10.0, 100.0, 0.2, TODAY).tg == pytest.approx(2.0)
    assert efficiency(MIXED, 5.0, -10.0, 100.0, 0.2, TODAY).gt is None


# ----------------------------------------------------------------- profit


def test_profit_measures_by_how_the_strategy_was_opened():
    assert return_on_debit(300, 200) == pytest.approx(0.5)
    assert return_on_debit(300, -100) is None
    assert capture_ratio(-40, -100) == pytest.approx(0.6)  # 100 credit, 40 to buy back
    assert capture_ratio(-40, 200) is None
    # a 40/45 call spread bought for 2.00: worth 0 to 500 at expiry
    spread = [
        leg("c40", "C", 40, 1, avg=3.0, mid=2.9),
        leg("c45", "C", 45, -1, avg=1.0, mid=1.1),
    ]
    (s,) = find_structures(spread)
    fig = structure_figures(s, {}, TODAY)
    assert value_bounds(fig) == pytest.approx((0.0, 500.0))
    profit, risk, ratio = remaining(180.0, *value_bounds(fig))
    assert (profit, risk) == pytest.approx((320.0, 180.0)) and ratio == pytest.approx(
        320 / 180
    )


# ---------------------------------------------------------------- signals


def test_dte_windows_follow_the_vault_table():
    assert [dte_window(d) for d in (90, 180, 365, 730, 1095)] == [28, 60, 120, 270, 270]
    assert dte_window(45) == 14  # shorter lives scale like 3 months
    assert dte_window(270) == 89  # between 6 and 12 months: interpolated


def names(found):
    return [(x.priority, x.name) for x in found]


def test_hard_signals_override_and_data_problems_only_ask_for_review():
    bad_data = signals(StrategyState(cls=None, data_flags=("missing quote",)))
    assert names(bad_data) == [(P0, "data unreliable")]
    assert outcome(bad_data) == "REVIEW - data unreliable"
    invalid = signals(StrategyState(cls=LONG, thesis_valid=False))
    assert outcome(invalid) == "CLOSE review"
    breached = signals(StrategyState(cls=SHORT, worst_case_loss=600, loss_budget=500))
    assert (
        names(breached) == [(P0, "risk budget breached")]
        and outcome(breached) == "CLOSE review"
    )
    crossed = signals(StrategyState(cls=LONG, spot=95.0, invalidation=("below", 100.0)))
    assert names(crossed) == [(P0, "thesis invalidated")]
    assert outcome(signals(StrategyState(cls=LONG))) == "HOLD"


def test_profit_levels_open_reviews_at_the_vault_levels():
    def detail(state):
        return [x.detail for x in signals(state) if x.name == "profit level"]

    assert detail(StrategyState(cls=LONG, net_debit=200, value_exec=290)) == []  # +45%
    assert (
        "begin review"
        in detail(StrategyState(cls=LONG, net_debit=200, value_exec=320))[0]
    )
    assert (
        "detailed review"
        in detail(StrategyState(cls=LONG, net_debit=200, value_exec=420))[0]
    )
    assert (
        "high-priority"
        in detail(StrategyState(cls=LONG, net_debit=200, value_exec=520))[0]
    )
    # a defined-risk debit spread: also 80% of its max profit
    spread = detail(
        StrategyState(cls=LONG, net_debit=200, value_exec=450, max_profit=300)
    )
    assert "83% of max profit" in spread
    # credit: 50 / 75% captured, never above 100%
    assert detail(StrategyState(cls=SHORT, net_debit=-100, value_exec=-40)) == [
        "60% of the credit captured"
    ]
    assert detail(StrategyState(cls=SHORT, net_debit=-100, value_exec=-60)) == []


def test_time_drift_and_efficiency_signals():
    year = StrategyState(cls=LONG, initial_dte=365, dte_front=100)
    assert names(signals(year)) == [(P1, "DTE window")]  # reviewed at 120 days left
    assert signals(StrategyState(cls=LONG, initial_dte=365, dte_front=130)) == []
    near = StrategyState(
        cls=SHORT, spot=101.0, short_legs=(ShortLeg(dte=3, strike=100.0),)
    )
    assert names(signals(near)) == [(P1, "near expiry")]
    assert names(signals(StrategyState(cls=LONG, single_long_delta=0.85))) == [
        (P2, "delta")
    ]
    assert signals(StrategyState(cls=LONG, single_long_delta=0.75)) == []
    assert names(signals(StrategyState(cls=MIXED, prev_cls=LONG))) == [
        (P2, "class change")
    ]
    slow = StrategyState(cls=LONG, gt_h={5: 0.8})
    assert names(signals(slow)) == [(P2, "efficiency")]
    thin = StrategyState(cls=LONG, remaining_reward_risk=0.2)
    assert names(signals(thin)) == [(P2, "low remaining reward")]
    assert outcome(signals(thin)) == "REVIEW"


# ----------------------------------------------------------- strategy ids


def test_strategy_ids_survive_rolls_and_new_ones_are_numbered():
    known = {"AA-debit-spread-1": Seen("AA", "debit spread", frozenset({1, 2}))}
    same = Seen("AA", "debit spread", frozenset({1, 2}))
    rolled = Seen("AA", "debit spread", frozenset({3, 4}))  # both legs replaced
    one_leg = Seen("AA", "debit spread", frozenset({1, 9}))  # one leg replaced
    assert auto_ids([same], known) == ["AA-debit-spread-1"]
    assert auto_ids([rolled], known) == ["AA-debit-spread-1"]
    assert auto_ids([one_leg], known) == ["AA-debit-spread-1"]
    new = [
        Seen("^SPX", "long put", frozenset({5})),
        Seen("^SPX", "long put", frozenset({6})),
    ]
    assert auto_ids(new, known) == ["SPX-long-put-1", "SPX-long-put-2"]
    # the known ID goes to the structure still holding its leg; the other is new
    assert auto_ids([rolled, same], known) == ["AA-debit-spread-2", "AA-debit-spread-1"]
