"""thetagang_nt.pm: the Portfolio Manager's pure functions, checked against
hand calculations (vault: testing_strategy)."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from thetagang_nt.pm.core.classify import (
    LONG,
    MIXED,
    SHORT,
    bs_gamma,
    classify,
    net_greeks,
)
from thetagang_nt.pm.core.efficiency import calendar_days, efficiency
from thetagang_nt.pm.core.profit import (
    capture_ratio,
    remaining,
    return_on_debit,
    value_bounds,
)
from thetagang_nt.pm.core.quality import Quote, leg_flags, values
from thetagang_nt.pm.core.review import (
    LONG_PREMIUM,
    P0,
    P1,
    P2,
    SHORT_PREMIUM,
    ShortLeg,
    StrategyState,
    manager,
    outcome,
)
from thetagang_nt.pm.core.strategy_id import Seen, auto_ids
from thetagang_nt.pm.long_options import signals as long_options
from thetagang_nt.pm.long_options.signals import dte_window, signals
from thetagang_nt.pm.short_options import signals as short_options
from thetagang_nt.pm.short_options.analysis import (
    loss_multiple,
    moves_out,
    premium_to_risk,
)
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


def test_every_decision_setting_comes_from_the_toml_and_is_listed():
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.config import PMConfig

    pm = PMConfig(gamma_tolerance=0.5, dte_windows=((100, 10), (200, 50)))
    listed = {name: (value, source) for name, value, _, source, _ in pm.params()}
    assert listed["gamma_tolerance"] == ("0.5", "your setting")
    assert listed["dte_windows"] == ("100d -> 10d; 200d -> 50d", "your setting")
    assert listed["near_money"][1] == "default"
    # the settings reach the functions that decide with them
    assert dte_window(150, pm.limits().dte_windows) == 30  # interpolated in your table
    assert pm.tolerance().gamma == 0.5
    call = (leg("c", "C", 100, 1),)
    flat = {"c": greeks(gamma=0.002, theta=-0.002)}  # x100: 0.2 gamma, -0.2 theta
    assert classify(call, flat, 100.0, TODAY) == LONG
    assert (
        classify(call, flat, 100.0, TODAY, pm.tolerance()) == MIXED
    )  # below 0.5: zero


# ------------------------------------------------- short options manager


def test_management_category_is_how_it_was_opened_not_its_convexity():
    assert manager(200.0) == LONG_PREMIUM
    assert manager(0.0) == LONG_PREMIUM
    assert manager(-100.0) == SHORT_PREMIUM
    # a debit spread that has turned short gamma stays with the long manager,
    # and its class comes from greeks: the managers read cls, never assume it
    assert manager(150.0) == LONG_PREMIUM


def test_short_analysis_matches_the_vault_formulas():
    # 100 credit, 40 to buy back: in profit, loss multiple -0.6
    assert loss_multiple(-40.0, -100.0) == pytest.approx(-0.6)
    # 250 to buy back on a 100 credit: lost 1.5x the credit
    assert loss_multiple(-250.0, -100.0) == pytest.approx(1.5)
    assert loss_multiple(-40.0, 200.0) is None  # opened for a debit
    # 40 of premium left against 460 of risk
    assert premium_to_risk(-40.0, 460.0) == pytest.approx(40 / 460)
    assert premium_to_risk(-40.0, None) is None and premium_to_risk(-40.0, 0) is None
    # Move_5 at S=100, sigma 20%: 100 * 0.2 * sqrt(5/252) = 2.817
    move = 100 * 0.2 * (5 / 252) ** 0.5
    assert moves_out("P", 95.0, 100.0, 0.2, 5) == pytest.approx(5 / move)
    assert moves_out("C", 105.0, 100.0, 0.2, 5) == pytest.approx(5 / move)
    assert moves_out("C", 98.0, 100.0, 0.2, 5) < 0  # in the money


def short_names(state, lim=None):
    return names(short_options.signals(state, lim or short_options.ShortLimits()))


def credit(value_exec, **kw):
    return StrategyState(cls=SHORT, net_debit=-100.0, value_exec=value_exec, **kw)


def test_premium_capture_levels_s1():
    def detail(v):
        return [
            x.detail
            for x in short_options.signals(credit(v))
            if x.name == "premium captured"
        ]

    assert detail(-60.0) == []  # 40% captured
    assert detail(-50.0) == ["50% of the credit: review"]
    assert detail(-30.0) == ["70% of the credit: strong consideration for closing"]
    assert detail(-20.0) == ["80% of the credit: default preference to close"]
    assert detail(-5.0) == ["95% of the credit: normally close"]


def test_dte_bands_s2_use_the_front_short_leg():
    def band(dte, **kw):
        legs = (ShortLeg(dte, 90.0, "P"), ShortLeg(dte + 30, 110.0, "C"))
        found = short_options.signals(StrategyState(cls=SHORT, short_legs=legs, **kw))
        return [x.detail for x in found if x.name == "DTE band"]

    assert band(22) == []
    assert band(21) == ["21 days left: review"]
    assert band(14) == ["14 days left: prefer closing or an active adjustment review"]
    assert band(7) == ["7 days left: generally prefer closing"]
    assert band(3, assignment_intended=True) == [
        "3 days left: expiry or assignment intended"
    ]
    # no short leg known: the strategy's front expiry
    assert names(short_options.signals(StrategyState(cls=SHORT, dte_front=10))) == [
        (P1, "DTE band")
    ]


def test_loss_multiple_s3_names_the_highest_level():
    found = short_options.signals(credit(-250.0))
    assert [x.detail for x in found if x.name == "loss multiple"] == [
        "loss 1.5x the credit (level 1)"
    ]
    assert [
        x.detail
        for x in short_options.signals(credit(-420.0))
        if x.name == "loss multiple"
    ] == ["loss 3.2x the credit (level 3)"]
    assert short_names(credit(-150.0)) == []  # 0.5x: below the first level


def test_assignment_s4_and_tested_strike_s5():
    itm_put = ShortLeg(3, 100.0, "P")
    s = StrategyState(cls=SHORT, spot=98.0, short_legs=(itm_put,), dte_front=40)
    assert (P1, "assignment exposure") in short_names(s)
    # intended (cash-secured put): no assignment review
    intended = StrategyState(
        cls=SHORT, spot=98.0, short_legs=(itm_put,), assignment_intended=True
    )
    assert (P1, "assignment exposure") not in short_names(intended)
    # out of the money is not exposed
    otm = StrategyState(cls=SHORT, spot=103.0, short_legs=(ShortLeg(3, 100.0, "P"),))
    assert (P1, "assignment exposure") not in short_names(otm)
    # S5: 2 points from a short put with Move_5 = 2.82: tested; 5 points: not
    near = StrategyState(
        cls=SHORT, spot=100.0, sigma_f=0.2, short_legs=(ShortLeg(40, 98.0, "P"),)
    )
    far = StrategyState(
        cls=SHORT, spot=100.0, sigma_f=0.2, short_legs=(ShortLeg(40, 95.0, "P"),)
    )
    assert short_names(near) == [(P2, "short strike tested")]
    assert short_names(far) == []
    # no volatility forecast: no S5
    assert (
        short_names(StrategyState(cls=SHORT, spot=100.0, short_legs=near.short_legs))
        == []
    )


def test_iron_condor_is_reviewed_as_one_strategy():
    """Four legs, one state: the tested call side raises one S5 signal, the
    untested put side none; capture is measured on the whole credit."""
    condor = StrategyState(
        cls=SHORT,
        net_debit=-300.0,
        value_exec=-120.0,  # 60% of the credit captured
        spot=104.0,
        sigma_f=0.2,
        dte_front=30,
        short_legs=(ShortLeg(30, 90.0, "P"), ShortLeg(30, 105.0, "C")),
    )
    found = short_options.signals(condor)
    assert names(found) == [
        (P1, "premium captured"),
        (P2, "short strike tested"),
    ]
    assert found[1].detail.startswith("short 105C")
    assert short_options.decide(condor)[1] == "REVIEW"


def test_short_manager_shares_the_hard_signals_and_outcome():
    bad = StrategyState(cls=None, net_debit=-100.0, data_flags=("missing quote",))
    assert short_options.decide(bad) == (
        long_options.signals(bad)[:1],
        "REVIEW - data unreliable",
    )
    breached = credit(-300.0, worst_case_loss=600.0, loss_budget=500.0)
    found, out = short_options.decide(breached)
    assert found[0].name == "risk budget breached" and out == "CLOSE review"
    assert short_options.decide(StrategyState(cls=SHORT)) == ([], "HOLD")
    assert short_names(StrategyState(cls=MIXED, prev_cls=SHORT)) == [
        (P2, "class change")
    ]


def test_the_managers_are_independent():
    """A credit strategy at 60% captured: the long manager (T2, superseded)
    and the short manager (S1) read it differently; neither imports the other."""
    s = credit(-40.0)
    assert [x.detail for x in long_options.signals(s)] == ["60% of the credit captured"]
    assert [x.detail for x in short_options.signals(s)] == ["60% of the credit: review"]
    from pathlib import Path

    assert "short_options" not in Path(long_options.__file__).read_text()
    assert "long_options" not in Path(short_options.__file__).read_text()


def test_short_settings_come_from_the_toml_and_are_listed():
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.config import PMConfig, PMStrategyConfig

    pm = PMConfig(capture_levels=(0.6,), short_dte_bands=(30, 20, 10))
    listed = {name: (value, source) for name, value, _, source, _ in pm.params()}
    assert listed["capture_levels"] == ("0.6", "your setting")
    assert listed["short_dte_bands"] == ("30, 20, 10", "your setting")
    assert listed["loss_multiples"] == ("1, 2, 3", "default")
    lim = pm.short_limits()
    assert lim.capture_levels == (0.6,) and lim.dte_bands == (30, 20, 10)
    assert short_names(credit(-50.0), lim) == []  # 50% < your 60%
    assert short_names(StrategyState(cls=SHORT, dte_front=25), lim) == [
        (P1, "DTE band")
    ]
    assert PMStrategyConfig(id="x").assignment_intended is False


def test_the_monitor_sends_each_strategy_to_its_manager():
    """Integration: the monitor's Portfolio Manager cycle on a debit call spread
    and a credit put spread, with fake quotes and greeks, no IB."""
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import PortfolioMonitor, _Families

    def g(delta):
        return greeks(delta=delta, gamma=0.02, theta=-0.02, vega=0.1, vol=0.3)

    legs = [
        leg("c100", "C", 100, 1, avg=3.0),
        leg("c105", "C", 105, -1, avg=1.0),  # debit 200
        leg("p100", "P", 100, -1, avg=3.0),
        leg("p95", "P", 95, 1, avg=1.0),  # credit 200
    ]
    book_greeks = {
        "c100": g(0.6),
        "c105": g(0.3),
        "p100": g(-0.4),
        "p95": g(-0.2),
    }
    quotes = {
        "c100": Quote(4.4, 4.6),
        "c105": Quote(1.4, 1.6),
        "p100": Quote(1.5, 1.7),  # buy back at 1.70
        "p95": Quote(0.9, 1.1),  # sell at 0.90: 80 to close, 60% captured
    }
    entries = []
    for st in find_structures(legs):
        fig = structure_figures(st, book_greeks, TODAY)
        entries.append(("AA", 103.0, st, fig, "-"))
    warnings = []
    fake = SimpleNamespace(
        _quote=quotes.get,  # no quote time: the monitor runs on the real clock
        _pm_known={},
        _journal=None,
        _pm=PMConfig(),
        _closes={},
        _pm_cls={},
        _pm_last={},
        log=SimpleNamespace(warning=warnings.append),
    )
    f = _Families()
    PortfolioMonitor._portfolio_manager(
        fake, f, entries, SimpleNamespace(greeks=book_greeks), TODAY
    )
    assert warnings == []
    info = {s.labels["kind"]: s.labels for s in f.by_name["pm_info"].samples}
    assert info["debit spread"]["manager"] == LONG_PREMIUM
    assert info["credit spread"]["manager"] == SHORT_PREMIUM
    assert info["debit spread"]["outcome"] == "HOLD"
    assert info["credit spread"]["signals"] == (
        "P1 premium captured: 60% of the credit: review"
    )
    fields = {
        (s.labels["strategy"].split("-")[1], s.labels["field"]): s.value
        for s in f.by_name["pm"].samples
    }
    assert fields[("credit", "capture")] == pytest.approx(0.6)
    assert fields[("credit", "loss_multiple")] == pytest.approx(-0.6)
    # 80 still to earn against 420 at risk (to the 500 floor of a 5-wide spread)
    assert fields[("credit", "premium_to_risk")] == pytest.approx(80 / 420)
    assert ("credit", "dte_window") not in fields  # T3 is the long manager's
    assert ("debit", "dte_window") in fields and ("debit", "capture") not in fields
