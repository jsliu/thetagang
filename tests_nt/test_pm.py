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
from thetagang_nt.pm.long_options import vix as long_vix
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
    # a wide market is real, not bad data: a liquidity signal, not a flag
    assert "wide spread" not in leg_flags(long, Quote(0.5, 1.0), greeks(), NOW)
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
        "loss 1.50x the credit (level 1)"
    ]
    assert [
        x.detail
        for x in short_options.signals(credit(-420.0))
        if x.name == "loss multiple"
    ] == ["loss 3.20x the credit (level 3)"]
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
        _rolls={},
        _iv_hist={},
        _iv_models={},
        _iv_model_of=lambda *a: None,
        _dividends={},
        _smiles={},
        _price_model_of=lambda *a: None,
        log=SimpleNamespace(warning=warnings.append),
    )
    f = _Families()
    PortfolioMonitor._portfolio_manager(fake, f, entries, book_greeks, TODAY)
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


def test_loss_multiple_text_never_rounds_past_its_level():
    # 1.96x is below level 2: one decimal would print "2.0x ... (level 1)"
    found = short_options.signals(credit(-296.39))
    assert [x.detail for x in found if x.name == "loss multiple"] == [
        "loss 1.96x the credit (level 1)"
    ]


def test_signals_print_one_line_per_priority():
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.monitor import signals_text
    from thetagang_nt.pm.core.review import Signal

    found = [
        Signal(P2, "short strike tested", "short 43C in the money"),
        Signal(P0, "data unreliable", "wide spread"),
        Signal(P1, "DTE band", "15 days left: review"),
        Signal(P1, "loss multiple", "loss 3.50x the credit (level 3)"),
    ]
    assert signals_text(found) == (
        "P0 data unreliable: wide spread\n"
        "P1 DTE band: 15 days left: review; loss multiple: loss 3.50x the credit (level 3)\n"
        "P2 short strike tested: short 43C in the money"
    )
    assert signals_text([]) == "-"


def test_review_queue_puts_what_you_can_act_on_first():
    """Bad data asks you to check prices, not to act: the shown priority is
    the most urgent other signal, and data-flagged rows sort just below clean
    ones of the same priority; data-only rows come last."""
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.monitor import review_rank
    from thetagang_nt.pm.core.review import Signal

    data = Signal(P0, "data unreliable", "wide spread")
    thesis = Signal(P0, "thesis invalidated", "marked invalid")
    profit = Signal(P1, "profit level", "+175% on the debit: high-priority review")
    drift = Signal(P2, "efficiency", "GT_5 0.37: decay beats gamma")
    ranks = {
        "real P0": review_rank([thesis]),
        "real P0, bad data": review_rank([data, thesis]),
        "P1": review_rank([profit]),
        "P1, bad data": review_rank([data, profit]),
        "P2": review_rank([drift]),
        "P2, bad data": review_rank([data, drift]),
        "bad data only": review_rank([data]),
        "none": review_rank([]),
    }
    assert [top for top, _ in ranks.values()] == [
        "P0",
        "P0",
        "P1",
        "P1",
        "P2",
        "P2",
        "data",
        "-",
    ]
    order = [rank for _, rank in ranks.values()]
    assert order == sorted(order, reverse=True) and len(set(order[:-1])) == 7
    # a data-flagged P1 still outranks every P2 and every data-only row
    assert ranks["P1, bad data"][1] > ranks["P2"][1] > ranks["bad data only"][1]


def test_a_default_risk_budget_applies_unless_a_strategy_sets_its_own():
    """[pm] max_loss covers every strategy; [[pm.strategies]] max_loss wins."""
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.config import PMConfig, PMStrategyConfig
    from thetagang_nt.monitor import PortfolioMonitor, _Families

    # a 40/45 call spread bought for 2.00, worth 2.80 at the bid; the scenario
    # grid's worst case (10 days, -5%, IV -10) is the P0 risk-budget check
    legs = [leg("c40", "C", 40, 1, avg=3.0), leg("c45", "C", 45, -1, avg=1.0)]
    quotes = {"c40": Quote(3.9, 4.1), "c45": Quote(0.9, 1.1)}
    g = {
        "c40": SimpleNamespace(**greeks(delta=0.6).__dict__, underlying=43.0),
        "c45": SimpleNamespace(**greeks(delta=0.3).__dict__, underlying=43.0),
    }
    (st,) = find_structures(legs)
    entries = [("AA", 43.0, st, structure_figures(st, g, TODAY), "-")]

    def run(pm):
        fake = SimpleNamespace(
            _quote=quotes.get,
            _pm_known={},
            _journal=None,
            _pm=pm,
            _closes={},
            _pm_cls={},
            _pm_last={},
            _rolls={},
            _iv_hist={},
            _iv_models={},
            _iv_model_of=lambda *a: None,
            _dividends={},
            _smiles={},
            _price_model_of=lambda *a: None,
            log=SimpleNamespace(warning=print),
        )
        f = _Families()
        PortfolioMonitor._portfolio_manager(fake, f, entries, g, TODAY)
        (info,) = f.by_name["pm_info"].samples
        fields = {x.labels["field"]: x.value for x in f.by_name["pm"].samples}
        return info.labels["outcome"], info.labels["top"], fields

    outcome_, top, fields = run(PMConfig())
    assert (outcome_, top) == ("HOLD", "-")  # no budget: off
    worst = fields["worst_case_loss"]
    assert 0 < worst < 280  # 10 days of -5%: less than the whole spread
    assert worst == max(fields[f"wcl_{h}"] for h in (1, 5, 10))
    assert run(PMConfig(max_loss=worst - 1))[:2] == ("CLOSE review", "P0")
    stress = fields["stress_loss"]  # the -10% gap is beyond the grid: also P0
    assert stress > worst
    assert run(PMConfig(max_loss=stress - 1))[:2] == ("CLOSE review", "P0")
    own = PMStrategyConfig(id="AA-debit-spread-1", max_loss=stress + 1)
    assert run(PMConfig(max_loss=worst - 1, strategies=[own]))[:2] == ("HOLD", "-")
    listed = {n: (v, s) for n, v, _, s, _ in PMConfig(max_loss=2000.0).params()}
    assert listed["max_loss"] == ("2000", "your setting")


def test_long_vix_calls_use_the_vix_levels_not_t1_and_t3():
    """V2 replaces T1 and V7 replaces T3 for VIX; V3 adds loss reviews; P0 and
    the P2 signals are the general ones (vault vix_options_management)."""

    def vix(value, dte=40, initial=60, **kw):
        return StrategyState(
            cls=LONG,
            net_debit=128.0,
            value_exec=value,
            dte_front=dte,
            initial_dte=initial,
            **kw,
        )

    def found(s):
        return [(x.priority, x.name, x.detail) for x in long_vix.decide(s)[0]]

    # +18.8% (the vault's example): V2 "evaluate early close"; T1 (+50%) is silent
    assert found(vix(152.0)) == [
        (P1, "VIX profit level", "+19% on the debit: evaluate early close")
    ]
    assert long_options.signals(vix(152.0)) == []
    assert found(vix(170.0))[0][2].endswith("strong profit-taking consideration")
    assert found(vix(135.0)) == []  # +5%: normal monitoring
    # V3: -25% review, -35% the stronger case; reviews, not stops
    assert found(vix(96.0))[0][2] == "-25% on the debit: loss review"
    assert found(vix(80.0))[0][2].endswith("stronger case for closing or reducing")
    # V7: fixed 14 days, not the life-scaled 60 -> 19 days of T3
    assert dte_window(60) == 19
    assert found(vix(135.0, dte=18)) == []
    assert found(vix(135.0, dte=14))[0][1] == "VIX time review"
    # shared: P0 hard signals, P2 delta on the matching future, the outcome
    s = vix(152.0, thesis_valid=False, single_long_delta=0.85)
    names = [(p, n) for p, n, _ in found(s)]
    assert (P0, "thesis invalidated") in names and (P2, "delta") in names
    assert long_vix.decide(s)[1] == "CLOSE review"
    tight = long_vix.VixLimits(profit_levels=(0.05,), review_dte=30)
    assert [x.name for x in long_vix.vix_signals(vix(135.0, dte=25), tight)] == [
        "VIX profit level",
        "VIX time review",
    ]


def test_vix_settings_come_from_the_toml_and_are_listed():
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.config import PMConfig

    pm = PMConfig(vix_profit_levels=(0.12,), vix_review_dte=10)
    listed = {name: (value, vid, src) for name, value, vid, src, _ in pm.params()}
    assert listed["vix_profit_levels"] == ("0.12", "V2", "your setting")
    assert listed["vix_loss_levels"] == ("-0.25, -0.35", "V3", "default")
    assert listed["vix_review_dte"] == ("10", "V7", "your setting")
    lim = pm.vix_limits()
    assert lim.profit_levels == (0.12,) and lim.review_dte == 10


def test_the_monitor_sends_vix_calls_to_the_vix_rules():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import PortfolioMonitor, _Families
    from thetagang_nt.portfolio import VIX_ROOT

    call = Leg("vxc", VIX_ROOT, "C", 30, EXP, 2, 1.28, 1.55)  # debit 256
    g = {"vxc": greeks(delta=0.4, gamma=0.05, theta=-0.03, vega=0.04, vol=0.9)}
    entries = []
    for st in find_structures([call]):
        fig = structure_figures(st, g, TODAY)
        entries.append((VIX_ROOT, 27.5, st, fig, "VIX hedge: hold"))
    fake = SimpleNamespace(
        _quote={"vxc": Quote(1.52, 1.58)}.get,
        _pm_known={},
        _journal=None,
        _pm=PMConfig(),
        _closes={},
        _pm_cls={},
        _pm_last={},
        _rolls={},
        _iv_hist={},
        _iv_models={},
        _iv_model_of=lambda *a: None,
        _dividends={},
        _smiles={},
        _price_model_of=lambda *a: None,
        log=SimpleNamespace(warning=print),
    )
    f = _Families()
    PortfolioMonitor._portfolio_manager(fake, f, entries, g, TODAY)
    (info,) = [s.labels for s in f.by_name["pm_info"].samples]
    assert info["symbol"] == "VIX" and info["manager"] == LONG_PREMIUM
    assert info["signals"].startswith("P1 VIX profit level: +19% on the debit")
    assert info["bot"] == "VIX hedge: hold"
    window = [
        s.value for s in f.by_name["pm"].samples if s.labels["field"] == "dte_window"
    ]
    assert window == [14]  # V7, not T3


def test_the_monitor_prices_vix_calls_on_the_matching_future(monkeypatch):
    """The monitor's live-data steps for a long VIX call, on a real Nautilus
    cache: load the parity partner (once a day), price the call on the matching
    future F = C - P + K, and hand it to the Portfolio Manager."""
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    import asyncio

    from nautilus_trader.cache.cache import Cache
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import QuoteTick
    from nautilus_trader.model.enums import AssetClass, OptionKind
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.instruments import OptionContract
    from nautilus_trader.model.objects import Price, Quantity

    from thetagang_nt import monitor
    from thetagang_nt.monitor import PortfolioMonitor, _Families
    from thetagang_nt.portfolio import VIX_ROOT

    expiry = datetime(2026, 12, 16, 21, tzinfo=timezone.utc)  # 44 days out
    cache = Cache()

    def option(right, kind):
        iid = InstrumentId.from_str(f"VIX261216{right}00030000.SMART")
        return OptionContract(
            iid,
            Symbol(iid.symbol.value),
            AssetClass.INDEX,
            USD,
            2,
            Price.from_str("0.01"),
            Quantity.from_int(100),
            Quantity.from_int(1),
            VIX_ROOT,
            kind,
            Price.from_str("30.00"),
            0,
            int(expiry.timestamp() * 1e9),
            0,
            0,
        )

    def quote(ins, bid, ask):
        size = Quantity.from_int(10)
        cache.add_quote_tick(
            QuoteTick(
                ins.id, Price.from_str(bid), Price.from_str(ask), size, size, 0, 0
            )
        )

    call, put = option("C", OptionKind.CALL), option("P", OptionKind.PUT)
    cache.add_instrument(call)
    quote(call, "1.50", "1.60")
    held = Leg(call.id, VIX_ROOT, "C", 30.0, expiry.date(), 2, 1.28, 1.55)
    monkeypatch.setattr(monitor, "held_legs", lambda _cache: [held])
    loads = []

    async def load_contracts(_actor, contracts):
        loads.append(contracts)
        cache.add_instrument(put)
        quote(put, "4.00", "4.10")

    monkeypatch.setattr(monitor.aio, "load_contracts", load_contracts)
    fake = SimpleNamespace(cache=cache, _vix_chain_day=None, log=None)

    # the put at the held strike is not loaded: load the chain over the expiry
    assert asyncio.run(PortfolioMonitor._vix_partners(fake, TODAY)) == {put.id}
    ((chain,),) = loads
    assert chain["symbol"] == "VIX" and chain["build_options_chain"]
    assert chain["min_expiry_days"] <= 44 <= chain["max_expiry_days"]
    assert asyncio.run(PortfolioMonitor._vix_partners(fake, TODAY)) == {put.id}
    assert len(loads) == 1  # found in the cache now; never twice a day anyway

    # priced on F = 1.55 - 4.05 + 30 = 27.50, not spot VIX, and sent to the PM
    sent = {}
    fake = SimpleNamespace(
        cache=cache,
        greeks=None,  # the VIX path prices with Black-76 itself
        _rules=None,
        _vix_hedge="VIX hedge: hold",
        _direct={},
        _portfolio_manager=lambda f, entries, greeks, today: sent.update(
            entries=entries, greeks=greeks
        ),
    )
    PortfolioMonitor._structures(
        fake, _Families(), [held], SimpleNamespace(greeks={}), TODAY
    )
    ((symbol, spot, st, fig, bot),) = sent["entries"]
    assert (symbol, st.kind, bot) == (VIX_ROOT, "long call", "VIX hedge: hold")
    assert spot == pytest.approx(27.5)
    g = sent["greeks"][call.id]
    assert g.underlying == pytest.approx(27.5)
    assert 0 < g.delta < 0.5 and g.gamma > 0 and g.theta < 0 and g.vol > 0
    assert fig.cost == pytest.approx(256.0)


# ------------------------------------------------------- phase 2: scenarios


def test_repricing_matches_textbook_prices():
    from thetagang_nt.pm.core.scenarios import bsm

    # Hull: S = K = 100, 1 year, 20% vol, 5% rate
    assert bsm(100, 100, 1, 0.2, True, 0.05, 0.05) == pytest.approx(10.4506, abs=1e-4)
    assert bsm(100, 100, 1, 0.2, False, 0.05, 0.05) == pytest.approx(5.5735, abs=1e-4)
    # Black-76 on a future (carry 0): e^-r (F N(d1) - K N(d2)), d1 = 0.1
    assert bsm(100, 100, 1, 0.2, True, 0.0, 0.05) == pytest.approx(7.5771, abs=1e-4)
    assert bsm(105, 100, 0, 0.2, True, 0.05) == 5.0  # expired: intrinsic
    assert bsm(95, 100, -0.01, 0.2, False, 0.05) == 5.0


def priced(iid, right, strike, qty, und, vol, expiry=EXP):
    p = leg(iid, right, strike, qty, expiry=expiry)
    return p, SimpleNamespace(underlying=und, vol=vol)


def test_scenario_grid_reprices_every_cell_from_a_model_baseline():
    from thetagang_nt.pm.core.scenarios import RATE, bsm, grid

    p, g = priced("c", "C", 100, 1, 100.0, 0.3)
    s = grid((p,), {"c": g}, TODAY, RATE, 0.3)
    assert len(s.fixed) == 3 * 5 * 5 and len(s.scaled) == 3 * 5 * 5
    assert s.v_model == pytest.approx(100 * bsm(100, 100, 30 / 365, 0.3, True, RATE))
    # a cell is repriced directly at its own horizon, never scaled from 1 day
    days = calendar_days(TODAY, 5)
    cell = 100 * bsm(95, 100, (30 - days) / 365, 0.2, True, RATE) - s.v_model
    assert s.fixed[(5, -0.05, -10)] == pytest.approx(cell)
    # WCL covers both families: -2 Move_5 (30% vol: -8.5%) beats the fixed -5%
    assert min(v for k, v in s.fixed.items() if k[0] == 5) == pytest.approx(cell)
    assert s.wcl(5) == pytest.approx(-s.scaled[(5, -2, -10)])
    assert s.wcl(5) > -cell
    assert s.worst_loss() == s.wcl(10) > s.wcl(1)
    summary = s.summary(5, 5)
    assert summary["time_decay"] < 0 < summary["iv_sensitivity"]
    assert summary["worst"] == pytest.approx(cell) and summary["best"] > 0
    assert set(s.stress) == {
        "gap down",
        "gap up",
        "event IV crush",
        "front-month spike",
    }
    # a lone long call: the front IV crush (-20) hurts more than the gap down,
    # where IV +15 offsets part of the -10% move
    assert s.stress_loss() == -s.stress["event IV crush"] > -s.stress["gap down"] > 0
    # no realised-vol forecast: no vol-scaled family
    assert grid((p,), {"c": g}, TODAY, RATE, None).scaled == {}
    # no implied vol: no grid at all
    assert (
        grid((p,), {"c": SimpleNamespace(underlying=100.0, vol=None)}, TODAY, RATE, 0.3)
        is None
    )


def test_near_expiry_legs_expire_inside_the_horizon_with_strikes_as_nodes():
    from thetagang_nt.pm.core.scenarios import RATE, grid

    soon = TODAY + timedelta(days=4)  # expires before the 5- and 10-day horizons
    short, g = priced("p", "P", 98, -1, 100.0, 0.3, expiry=soon)
    s = grid((short,), {"p": g}, TODAY, RATE, None)
    assert (5, -0.02, 0) in s.fixed  # the 98 strike is a price node (pin risk)
    # expired at intrinsic: at -5% the short 98 put owes 98 - 95 = 3 a share
    assert s.fixed[(10, -0.05, 10)] == pytest.approx(-300 - s.v_model)
    assert s.fixed[(10, -0.05, 10)] == s.fixed[(10, -0.05, -10)]  # IV is gone


def test_vix_grid_moves_each_leg_on_its_own_future():
    from thetagang_nt.pm.core.scenarios import VIX_SPEC, bsm, grid

    later = EXP + timedelta(days=28)
    front, gf = priced("f", "C", 25, 1, 22.0, 0.9)
    back, gb = priced("b", "C", 25, -1, 24.0, 0.8, expiry=later)
    s = grid((front, back), {"f": gf, "b": gb}, TODAY, 0.0, None, VIX_SPEC)
    assert {k[1] for k in s.fixed} == {-0.2, -0.1, 0.0, 0.15, 0.3}
    assert set(s.stress) == {"spike", "fade", "regime shift"}
    # spike: front future +30%, back +15%; front IV +20, back +10 (1 day)

    d = calendar_days(TODAY, 1)
    spike = (
        100 * bsm(22 * 1.3, 25, (30 - d) / 365, 1.1, True, 0.0)
        - 100 * bsm(24 * 1.15, 25, (58 - d) / 365, 0.9, True, 0.0)
        - s.v_model
    )
    assert s.stress["spike"] == pytest.approx(spike)


def test_expected_pnl_and_expected_shortfall_by_hand():
    from thetagang_nt.pm.core.scenarios import Grid
    from thetagang_nt.pm.core.valuation import (
        UtilityLimits,
        expected_shortfall,
        forward,
        node_probabilities,
    )

    probs = node_probabilities((-2, -1, 0, 1, 2))
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs[-2] == pytest.approx(0.0668, abs=1e-4)  # N(-1.5)
    assert probs[0] == pytest.approx(0.3829, abs=1e-4)  # N(0.5) - N(-0.5)
    # the worst 10%: all of the 5% at -100 and half of the 10% at -50
    outcomes = [(0.05, -100.0), (0.10, -50.0), (0.85, 10.0)]
    assert expected_shortfall(outcomes, 0.10) == pytest.approx(75.0)

    pnl = {-2: -200.0, -1: -80.0, 0: -10.0, 1: 60.0, 2: 150.0}
    g = Grid(0.0, {}, {(5, z, 0): v for z, v in pnl.items()})
    lim = UtilityLimits(risk_aversion=0.5, capital_cost=0.10, commission=1.0)
    fwd = forward(g, liquidity_cost=20.0, contracts=2, capital_at_risk=504.0, lim=lim)
    e = sum(probs[z] * v for z, v in pnl.items())
    es = expected_shortfall([(probs[z], v) for z, v in pnl.items()], 0.10)
    c_close = 20.0 + 2 * 1.0
    assert fwd.e_pnl == pytest.approx(e) and fwd.es == pytest.approx(es)
    assert fwd.u_close == -c_close
    assert fwd.u_hold == pytest.approx(e - 0.5 * es - c_close - 0.10 * 504 * 5 / 252)
    assert fwd.rar == pytest.approx(e / es)
    assert forward(Grid(0.0, {}), 20.0, 2, 504.0) is None  # no probabilities


def test_stress_loss_beyond_the_budget_is_p0():
    from thetagang_nt.pm.core.review import hard_signals

    over = StrategyState(cls=LONG, stress_loss=900.0, loss_budget=500.0)
    assert [x.detail for x in hard_signals(over)] == ["stress loss -$900 > budget $500"]


def test_a_used_up_budget_reads_as_a_negative_amount():
    from thetagang_nt.pm.core.review import hard_signals

    s = StrategyState(cls=LONG, worst_case_loss=300.0, loss_budget=-485.0)
    assert hard_signals(s)[0].detail == "worst case -$300 > budget -$485"


# ----------------------------------------------------- phase 2: roll candidates


def listed(rights="CP", strikes=(90, 95, 100, 105, 110), days=(30, 60, 90, 120)):
    return {
        (r, TODAY + timedelta(days=d), float(k)): f"{r}{d}-{k}"
        for r in rights
        for k in strikes
        for d in days
    }


def test_roll_targets_restore_the_original_life_and_vix_uses_v1():
    from thetagang_nt.pm.core.rolls import RollLimits, target_dtes

    lim = RollLimits()
    assert target_dtes(20, 60, False, lim) == [60, 90, 120]
    assert target_dtes(5, None, False, lim) == [30, 45, 60]  # unknown life: 30 days
    assert target_dtes(80, 60, False, lim) == [94, 120]  # past the front, at least
    assert target_dtes(10, 400, True, lim) == [30, 45, 60]  # V1, whatever the life


def test_roll_candidates_keep_the_structure_and_snap_to_listed_contracts():
    from thetagang_nt.pm.core.rolls import RollLimits, generate

    lim = RollLimits(strike_shifts=(-0.05, 0.0))
    near = TODAY + timedelta(days=30)
    spread = (
        leg("l", "C", 100, 2, expiry=near),
        leg("s", "C", 105, -2, expiry=near),
    )
    out = generate(spread, listed(), 100.0, TODAY, [60, 90], lim)
    assert len(out) == 4  # 2 expiries x 2 shifts
    labels = {tuple((p.strike, p.qty, p.expiry - TODAY) for p in c) for c in out}
    assert (
        (95.0, 2, timedelta(days=60)),
        (100.0, -2, timedelta(days=60)),
    ) in labels
    assert out[0][0].instrument_id == "C60-95"
    # a strike too far from anything listed drops the candidate; so does the
    # strategy itself and anything not later than today's front expiry
    sparse = {k: v for k, v in listed().items() if k[2] in (90.0, 110.0)}
    assert generate(spread, sparse, 100.0, TODAY, [60], RollLimits()) == []
    held_later = (leg("l", "C", 100, 1, expiry=TODAY + timedelta(days=120)),)
    assert generate(held_later, listed(), 100.0, TODAY, [60], RollLimits()) == []


def test_a_calendar_rolls_with_its_gap_and_never_collapses():
    from thetagang_nt.pm.core.rolls import RollLimits, generate

    lim = RollLimits(strike_shifts=(0.0,))
    calendar = (
        leg("b", "C", 100, 1, expiry=TODAY + timedelta(days=60)),
        leg("f", "C", 100, -1, expiry=TODAY + timedelta(days=30)),
    )
    ((back, front),) = generate(calendar, listed(), 100.0, TODAY, [60], lim)
    assert (back.expiry - TODAY).days == 90 and (front.expiry - TODAY).days == 60
    one_expiry = {
        k: v for k, v in listed().items() if k[1] == TODAY + timedelta(days=60)
    }
    assert generate(calendar, one_expiry, 100.0, TODAY, [60], lim) == []


def roll_candidate(max_loss=None, quotes=None):
    from thetagang_nt.pm.core.quality import QualityLimits
    from thetagang_nt.pm.core.rolls import evaluate
    from thetagang_nt.pm.core.scenarios import RATE, GridSpec
    from thetagang_nt.pm.core.valuation import UtilityLimits

    legs = (leg("l", "C", 100, 1), leg("s", "C", 105, -1))
    g = {
        "l": SimpleNamespace(**greeks(delta=0.5).__dict__, underlying=100.0),
        "s": SimpleNamespace(**greeks(delta=0.3).__dict__, underlying=100.0),
    }
    quotes = quotes or {"l": Quote(4.0, 4.2), "s": Quote(1.9, 2.1)}
    return evaluate(
        "debit spread",
        legs,
        g,
        quotes,
        TODAY,
        NOW,
        RATE,
        0.3,
        GridSpec(),
        UtilityLimits(commission=1.0),
        QualityLimits(),
        max_loss,
    )


def test_a_candidate_is_judged_as_a_new_trade_with_its_own_costs():
    c = roll_candidate()
    # opens at the ask of the long and the bid of the short: 4.20 - 1.90
    assert c.net_open == pytest.approx(230.0)
    # half-spreads of 10 + 10, plus 2 contracts at $1
    assert c.open_cost == pytest.approx(22.0)
    assert c.fails == () and c.u_new is not None and c.e_pnl is not None
    assert c.label == f"AA {EXP:%Y%m%d} +1x100C -1x105C"
    # over the budget, or wide: listed, but not recommendable
    assert roll_candidate(max_loss=1.0).fails == ("over max_loss",)
    wide = roll_candidate(quotes={"l": Quote(3.0, 5.0), "s": Quote(1.9, 2.1)})
    assert wide.fails == ("wide spread",)
    # no usable quote: not a candidate at all
    assert roll_candidate(quotes={"l": Quote(None, None), "s": Quote(1.9, 2.1)}) is None


def test_hold_close_roll_comparison_follows_roll_rules():
    from dataclasses import replace

    from thetagang_nt.pm.core.rolls import compare

    c = replace(roll_candidate(), u_new=5.0)
    # U_roll = 5 - 20 = -15: beats HOLD by 25 and beats U_close = -20
    choice = compare(-40.0, -20.0, [c], 10.0)
    assert (choice.view, choice.u_roll, choice.roll_score) == ("ROLL", -15.0, 25.0)
    assert choice.best is c
    # within the margin of HOLD: no roll, and HOLD beats closing
    assert compare(-18.0, -20.0, [c], 10.0).view == "HOLD"
    # the new trade must beat cash: U_new < 0 means U_roll < U_close
    poor = replace(c, u_new=-1.0)
    assert compare(-60.0, -20.0, [poor], 10.0).view == "CLOSE"
    # a candidate over its limits is never recommended, however good
    failing = replace(c, u_new=500.0, fails=("over max_loss",))
    assert compare(-60.0, -20.0, [failing], 10.0).best is None
    # no probabilities (VIX): no view
    assert compare(None, None, [c], 10.0).view == "-"


def test_the_monitor_scans_one_strategy_and_leaves_no_line_subscribed(monkeypatch):
    """A real Nautilus cache: the scan quotes the candidates of the strategy
    longest without one, unsubscribes them all, keeps held legs subscribed, and
    the dashboard gets the candidates and the model's view."""
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    import asyncio

    from nautilus_trader.cache.cache import Cache
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import QuoteTick
    from nautilus_trader.model.enums import AssetClass, OptionKind
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.instruments import OptionContract
    from nautilus_trader.model.objects import Price, Quantity

    from thetagang_nt import monitor
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import (
        PortfolioMonitor,
        RollJob,
        RollScan,
        _Families,
        publish_rolls,
    )

    cache = Cache()
    held_expiry = date(2026, 12, 2)
    later = date(2027, 1, 15)

    def option(expiry, strike):
        iid = InstrumentId.from_str(f"AA{expiry:%y%m%d}C{strike}.SMART")
        return OptionContract(
            iid,
            Symbol(iid.symbol.value),
            AssetClass.EQUITY,
            USD,
            2,
            Price.from_str("0.01"),
            Quantity.from_int(100),
            Quantity.from_int(1),
            "AA",
            OptionKind.CALL,
            Price.from_str(f"{strike}.00"),
            0,
            int(
                datetime(
                    expiry.year, expiry.month, expiry.day, 21, tzinfo=timezone.utc
                ).timestamp()
                * 1e9
            ),
            0,
            0,
        )

    held_ins, new_ins = option(held_expiry, 100), option(later, 100)
    for ins in (held_ins, new_ins):
        cache.add_instrument(ins)
    held = Leg(held_ins.id, "AA", "C", 100.0, held_expiry, 1, 3.0, 3.1)
    (st,) = find_structures([held])
    clock = {"now": 1000}
    subscribed = set()

    def quote(iid):
        size = Quantity.from_int(10)
        cache.add_quote_tick(
            QuoteTick(
                iid,
                Price.from_str("5.00"),
                Price.from_str("5.20"),
                size,
                size,
                clock["now"],
                clock["now"],
            )
        )

    def subscribe(_actor, iid):
        subscribed.add(iid)
        quote(iid)  # IB answers a new subscription with a fresh quote

    monkeypatch.setattr(monitor.aio, "subscribe", subscribe)
    monkeypatch.setattr(
        monitor.aio, "unsubscribe", lambda _a, iid: subscribed.discard(iid)
    )
    monkeypatch.setattr(monitor, "held_legs", lambda _c: [held])
    monkeypatch.setattr(
        monitor,
        "option_greeks",
        lambda _c, _g, ins, _t: SimpleNamespace(
            delta=0.5, gamma=0.05, theta=-0.03, vega=0.1, vol=0.3, underlying=100.0
        ),
    )
    loads = []

    async def load_contracts(_actor, contracts):
        loads.append(contracts)

    monkeypatch.setattr(monitor.aio, "load_contracts", load_contracts)
    warnings = []
    other_scan = RollScan(datetime.now(timezone.utc), [])
    job = RollJob("AA", st, 100.0, 0.3, 30, 60, 2000.0, False)
    fake = SimpleNamespace(
        cache=cache,
        greeks=None,
        clock=SimpleNamespace(timestamp_ns=lambda: clock["now"]),
        log=SimpleNamespace(warning=warnings.append),
        _pm=PMConfig(roll_strike_shifts=(0.0,), roll_dte_multiples=(1.25,)),
        _roll_jobs={"AA-long-call-1": job, "BB-long-call-1": job},
        _rolls={"BB-long-call-1": other_scan},  # scanned just now: not due
        _roll_chain_day={},
        _iv_day={"AA": TODAY},
        _quote=lambda iid: Quote(5.0, 5.2, datetime.now(timezone.utc)),
    )
    fake._load_roll_chain = lambda *a: PortfolioMonitor._load_roll_chain(fake, *a)
    fake._fresh_quotes = lambda ids: PortfolioMonitor._fresh_quotes(fake, ids)
    asyncio.run(PortfolioMonitor._scan_rolls(fake, TODAY))
    assert warnings == []
    assert subscribed == set()  # every candidate line given back
    ((chain,),) = loads
    assert chain["symbol"] == "AA" and chain["build_options_chain"]
    assert fake._rolls["BB-long-call-1"] is other_scan
    # not scanned before its dividends and IV history have loaded
    waiting = RollJob("CC", st, 100.0, 0.3, 30, 60, 2000.0, False)
    fake._roll_jobs["CC-long-call-1"] = waiting
    asyncio.run(PortfolioMonitor._scan_rolls(fake, TODAY))
    assert "CC-long-call-1" not in fake._rolls
    del fake._roll_jobs["CC-long-call-1"]
    ((c,)) = fake._rolls["AA-long-call-1"].candidates
    assert c.legs[0].instrument_id == new_ins.id  # the later expiry, same strike
    # the chain is loaded once a day, however many scans
    fake._rolls.clear()
    asyncio.run(PortfolioMonitor._scan_rolls(fake, TODAY))
    assert len(loads) == 1
    # a failing scan is recorded, so it can't hog every cycle
    fake._rolls.clear()
    fake._load_roll_chain = lambda *a: (_ for _ in ()).throw(ValueError("IB down"))
    asyncio.run(PortfolioMonitor._scan_rolls(fake, TODAY))
    assert fake._rolls["AA-long-call-1"].candidates == []
    assert len(warnings) == 1

    f = _Families()
    scan = RollScan(datetime.now(timezone.utc), [c])
    publish_rolls(
        f, "AA-long-call-1", -30.0, -20.0, scan, 10.0, datetime.now(timezone.utc)
    )
    (view,) = f.by_name["pm_view"].samples
    assert view.labels["scanned"] == "yes" and view.labels["best"] == c.label
    fields = {s.labels["field"] for s in f.by_name["pm_roll"].samples}
    assert {"net_open", "wcl_5", "u_roll", "roll_score"} <= fields


def test_iv_history_skips_legs_with_a_data_flag():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.monitor import iv_rows
    from thetagang_nt.pm.core.quality import QualityLimits

    good, wide, none = leg("g", "C", 40, 1), leg("w", "C", 45, 1), leg("n", "P", 35, 1)
    g = {
        "g": SimpleNamespace(**greeks(vol=0.31).__dict__, underlying=41.0),
        "w": SimpleNamespace(**greeks(vol=0.90).__dict__, underlying=41.0),
    }
    quotes = {"g": Quote(2.0, 2.1), "w": Quote(1.0, 2.0), "n": Quote(1.0, 1.1)}
    (row,) = iv_rows((good, wide, none), g, quotes, NOW, QualityLimits())
    assert (row.contract, row.iv, row.underlying) == ("g", 0.31, 41.0)


# ------------------------------------------------- phase 2: the joint IV model


def iv_history(n=400):
    """IV cycles between 0.2 and 0.4; the price falls when IV rises."""
    import math

    from thetagang_nt.pm.core.iv_model import IvHistory

    ivs = [0.3 + 0.1 * math.sin(t / 15) for t in range(n)]
    closes = [100 * math.exp(-0.5 * (x - 0.3) + 0.0005 * t) for t, x in enumerate(ivs)]
    return IvHistory(tuple(closes), tuple(ivs))


def test_iv_model_takes_no_price_view_and_matches_the_forecast():
    import math
    from statistics import fmean, pstdev

    from thetagang_nt.pm.core.iv_model import IvLimits, build

    m = build(iv_history(), 0.25, 5, IvLimits(shrink=1.0))
    log_r = [math.log(1 + r) for r, _ in m.samples]
    assert fmean(log_r) == pytest.approx(0.0, abs=1e-12)  # the upward drift is gone
    assert pstdev(log_r) == pytest.approx(0.25 * math.sqrt(5 / 252))
    # the link between price and IV survives: price down, IV up
    from statistics import correlation

    assert correlation(log_r, [c for _, c in m.samples]) < -0.5


def test_iv_model_uses_todays_third_of_the_trailing_year_and_shrinks_the_view():
    from statistics import fmean

    from thetagang_nt.pm.core.iv_model import IvHistory, IvLimits, build

    def sawtooth(n):
        """IV climbs slowly for 40 days, then crashes: high IV falls, low IV rises."""
        ivs = tuple(0.2 + 0.2 * (t % 40) / 40 for t in range(n))
        return IvHistory(iv_history(n).closes, ivs)

    high = sawtooth(400)  # the last day is the top of a climb
    low = sawtooth(405)  # the last day is just after a crash
    full = build(high, 0.25, 5, IvLimits(shrink=1.0))
    assert full.rank > 2 / 3 and full.conditioned
    assert full.view == pytest.approx(fmean(c for _, c in full.samples))
    assert full.view < 0
    assert build(low, 0.25, 5, IvLimits(shrink=1.0)).view > 0
    # shrink keeps half the expected change, and only moves the mean
    half = build(high, 0.25, 5, IvLimits(shrink=0.5))
    assert half.view == pytest.approx(full.view / 2)
    assert fmean(c for _, c in half.samples) == pytest.approx(full.view / 2)


def test_iv_model_falls_back_or_gives_up_on_thin_data():
    from thetagang_nt.pm.core.iv_model import IvHistory, IvLimits, build

    short = iv_history(100)  # under a year: no rank, all of history
    m = build(short, 0.25, 5)
    assert m.rank is None and not m.conditioned and len(m.samples) == 95
    # today's third too rare: all of history, not a handful of days
    m = build(iv_history(), 0.25, 5, IvLimits(min_samples=10_000))
    assert m is None or not m.conditioned
    m = build(iv_history(), 0.25, 5, IvLimits(min_samples=300))
    assert not m.conditioned and len(m.samples) == 395
    assert build(iv_history(), None, 5) is None  # no forecast
    assert build(IvHistory((100.0,) * 10, (0.2,) * 10), 0.25, 5) is None  # too short
    flat = IvHistory((100.0,) * 100, (0.2,) * 100)
    assert build(flat, 0.25, 5) is None  # no price moves to scale


def test_repricing_samples_moves_long_dated_legs_less():
    import math

    from thetagang_nt.pm.core.scenarios import RATE, bsm, repriced

    far = TODAY + timedelta(days=120)
    p, g = priced("c", "C", 100, 1, 100.0, 0.3, expiry=far)
    d = calendar_days(TODAY, 5)
    ((pnl),) = repriced((p,), {"c": g}, TODAY, RATE, ((0.02, 0.04),), 5, 30)
    base = 100 * bsm(100, 100, 120 / 365, 0.3, True, RATE)
    # 4 points at 30 days is sqrt(30/120) = half of that at 120 days: 2 points
    moved = 100 * bsm(102, 100, (120 - d) / 365, 0.32, True, RATE)
    assert pnl == pytest.approx(moved - base)
    assert math.sqrt(30 / 120) == 0.5


def test_forward_value_takes_equally_likely_pnls_from_the_joint_model():
    from thetagang_nt.pm.core.scenarios import Grid
    from thetagang_nt.pm.core.valuation import UtilityLimits, forward

    lim = UtilityLimits(alpha=0.25, risk_aversion=0.0, capital_cost=0.0, commission=0.0)
    fwd = forward(Grid(0.0, {}), 0.0, 1, 0.0, lim, pnls=[-100.0, 0.0, 50.0, 150.0])
    assert fwd.e_pnl == pytest.approx(25.0)
    assert fwd.es == pytest.approx(100.0)  # the worst quarter


def test_the_monitor_reports_both_views_side_by_side():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import PortfolioMonitor, _Families
    from thetagang_nt.pm.core.iv_model import build

    legs = [leg("c", "C", 100, 1, avg=3.0)]
    g = {"c": SimpleNamespace(**greeks().__dict__, underlying=100.0)}
    (st,) = find_structures(legs)
    entries = [("AA", 100.0, st, structure_figures(st, g, TODAY), "-")]
    model = build(iv_history(), 0.3, 5)
    closes = {TODAY - timedelta(days=i): 100.0 + (i % 3) for i in range(40)}
    from thetagang_nt.portfolio import underlying_id

    fake = SimpleNamespace(
        _quote={"c": Quote(3.0, 3.1)}.get,
        _pm_known={},
        _journal=None,
        _pm=PMConfig(),
        _closes={underlying_id("AA"): closes},
        _pm_cls={},
        _pm_last={},
        _rolls={},
        _iv_model_of=lambda *a: model,
        _dividends={},
        _smiles={},
        _price_model_of=lambda *a: None,
        log=SimpleNamespace(warning=print),
    )
    f = _Families()
    PortfolioMonitor._portfolio_manager(fake, f, entries, g, TODAY)
    fields = {s.labels["field"]: s.value for s in f.by_name["pm"].samples}
    assert {"e_pnl", "e_pnl_iv", "u_hold", "u_hold_iv", "iv_rank", "iv_view"} <= set(
        fields
    )
    assert fields["e_pnl"] != fields["e_pnl_iv"]
    (view,) = f.by_name["pm_view"].samples
    assert view.labels["view_iv"] in ("HOLD", "CLOSE")
    # the signals and the outcome are untouched: reported only
    (info,) = f.by_name["pm_info"].samples
    assert "iv" not in info.labels["signals"].lower().replace("divid", "")


def test_a_roll_candidate_gets_a_utility_under_the_iv_model_too():
    from dataclasses import replace

    from thetagang_nt.pm.core.iv_model import build
    from thetagang_nt.pm.core.quality import QualityLimits
    from thetagang_nt.pm.core.rolls import compare, evaluate
    from thetagang_nt.pm.core.scenarios import RATE, GridSpec
    from thetagang_nt.pm.core.valuation import UtilityLimits

    legs = (leg("l", "C", 100, 1), leg("s", "C", 105, -1))
    g = {
        "l": SimpleNamespace(**greeks(delta=0.5).__dict__, underlying=100.0),
        "s": SimpleNamespace(**greeks(delta=0.3).__dict__, underlying=100.0),
    }
    quotes = {"l": Quote(4.0, 4.2), "s": Quote(1.9, 2.1)}
    args = ("debit spread", legs, g, quotes, TODAY, NOW, RATE, 0.3, GridSpec())
    rest = (UtilityLimits(commission=1.0), QualityLimits(), None)
    without = evaluate(*args, *rest)
    assert without.u_new_iv is None
    c = evaluate(*args, *rest, build(iv_history(), 0.3, 5))
    assert c.u_new_iv is not None and c.u_new_iv != c.u_new
    assert c.u_new == without.u_new  # the IV-fixed figures don't change
    # the comparison under the IV model reads u_new_iv
    assert compare(-500.0, -20.0, [replace(c, u_new=c.u_new_iv)], 10.0).best is not None


# -------------------------------------- phase 2: American pricing and dividends


def binomial(s0, k, t, v, call, r, dividends=(), steps=800):
    """The reference: Cox-Ross-Rubinstein on the escrowed price (option_repricing)."""
    import math

    import numpy as np

    inside = [(x, d) for x, d in dividends if 0 < x < t]
    dt = t / steps
    escrowed = s0 - sum(d * math.exp(-r * x) for x, d in inside)
    u = math.exp(v * math.sqrt(dt))
    p = (math.exp(r * dt) - 1 / u) / (u - 1 / u)
    disc = math.exp(-r * dt)
    st = escrowed * u ** (2 * np.arange(steps + 1) - steps)
    value = np.maximum(st - k, 0) if call else np.maximum(k - st, 0)
    for i in range(steps - 1, -1, -1):
        now = i * dt
        pv = sum(d * math.exp(-r * (x - now)) for x, d in inside if x > now)
        st = escrowed * u ** (2 * np.arange(i + 1) - i) + pv
        value = disc * (p * value[1:] + (1 - p) * value[:-1])
        value = np.maximum(value, st - k if call else k - st)
    return float(value[0])


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("years", [30 / 365, 0.25, 1.0])
@pytest.mark.parametrize("vol", [0.25, 0.6])
@pytest.mark.parametrize("ratio", [0.85, 1.0, 1.15])
def test_american_prices_converge_to_a_binomial_tree(call, years, vol, ratio):
    from thetagang_nt.pm.core.american import american

    dividends = ((years / 2, 0.5),)
    tree = binomial(100, 100 * ratio, years, vol, call, 0.0425, dividends)
    model = american(100, 100 * ratio, years, vol, call, 0.0425, dividends)
    # measured on typical inputs; extreme ones reach about 4% of the option's value
    assert abs(model - tree) <= max(0.25, 0.03 * tree)


def test_american_price_properties():
    from thetagang_nt.pm.core.american import american, european, implied_vol

    r = 0.0425
    # no dividend in its life: an American call is the European one
    assert american(100, 100, 0.5, 0.3, True, r) == european(
        100, 100, 0.5, 0.3, True, r
    )
    assert american(100, 100, 0.5, 0.3, True, r, ((1.0, 2.0),)) == pytest.approx(
        european(100, 100, 0.5, 0.3, True, r)
    )
    # a put is worth more than the European one, and at least its intrinsic value
    assert american(100, 110, 1.0, 0.2, False, r) > european(
        100, 110, 1.0, 0.2, False, r
    )
    assert american(60, 100, 1.0, 0.2, False, r) == pytest.approx(40.0)
    # a dividend lowers a call and raises a put
    assert american(100, 100, 1.0, 0.3, True, r, ((0.5, 2.0),)) < american(
        100, 100, 1.0, 0.3, True, r
    )
    assert american(100, 100, 1.0, 0.3, False, r, ((0.5, 2.0),)) > american(
        100, 100, 1.0, 0.3, False, r
    )
    # expired: intrinsic
    assert american(105, 100, 0, 0.3, True, r) == 5.0
    # deep in the money with a dividend: never more than waiting for it is worth
    assert american(100, 130, 1.0, 0.2, False, r, ((0.5, 1.0),)) == pytest.approx(30.0)
    # implied vol round trip, and none where no vol can match
    p = american(100, 95, 0.5, 0.4, False, r, ((0.2, 1.0),))
    assert implied_vol(p, 100, 95, 0.5, False, r, ((0.2, 1.0),)) == pytest.approx(
        0.4, abs=1e-3
    )
    assert implied_vol(1.0, 100, 50, 0.5, True, r) is None  # below intrinsic


def test_dividend_schedule_repeats_what_ib_gives():
    from thetagang_nt.pm.core.american import schedule

    assert schedule(None, None, None, TODAY) == ()  # pays none
    assert schedule(TODAY - timedelta(days=1), 1.0, 4.0, TODAY) == ()  # stale
    next_date = TODAY + timedelta(days=10)
    s = schedule(next_date, 1.0, 4.0, TODAY, years=1)
    assert s[0] == (next_date, 1.0) and len(s) == 4  # quarterly
    assert (s[1][0] - s[0][0]).days == 91
    assert len(schedule(next_date, 0.15, 0.15, TODAY, years=1)) == 1  # yearly (KWEB)


def test_the_grid_prices_american_from_a_baseline_at_the_mid():
    from thetagang_nt.pm.core.scenarios import RATE, PriceModel, grid

    p = leg("p", "P", 100, 1, mid=6.0)
    g = {"p": SimpleNamespace(underlying=100.0, vol=0.5)}
    euro = grid((p,), g, TODAY, RATE, None)
    am = grid((p,), g, TODAY, RATE, None, model=PriceModel())
    # IV re-implied by the same model: the baseline is the mid
    assert am.v_model == pytest.approx(600.0, abs=1.0)
    assert euro.v_model != am.v_model
    # a dividend inside the horizon is gone from the later cells' future
    ex = TODAY + timedelta(days=3)
    paying = grid((p,), g, TODAY, RATE, None, model=PriceModel(((ex, 2.0),)))
    assert paying.v_model == pytest.approx(600.0, abs=1.0)
    assert paying.fixed[(10, 0.0, 0)] != am.fixed[(10, 0.0, 0)]
    # no vol can match the mid: the leg keeps the vol it came with
    low = leg("p", "P", 100, 1, mid=0.0001)
    assert grid((low,), g, TODAY, RATE, None, model=PriceModel()).v_model > 100


def test_s4_ex_dividend_check_in_both_managers():
    def state(cls=SHORT, **kw):
        leg_ = ShortLeg(
            20,
            105.0,
            "C",
            kw.pop("dividend", 0.63),
            kw.pop("ex_days", 7),
            kw.pop("extrinsic", 0.40),
        )
        return StrategyState(
            cls=cls, spot=kw.pop("spot", 110.0), short_legs=(leg_,), **kw
        )

    want = "short 105C: ex-dividend in 7 days, $0.63 > $0.40 extrinsic"
    for found in (short_options.signals(state()), long_options.signals(state())):
        assert want in [x.detail for x in found if x.priority == P1]

    def details(s):
        return [x.detail for x in short_options.signals(s) if "ex-dividend" in x.detail]

    assert (
        details(state(extrinsic=0.80)) == []
    )  # time value worth more than the dividend
    assert details(state(spot=100.0)) == []  # out of the money
    assert details(state(ex_days=30)) == []  # too far off
    assert details(state(dividend=None, ex_days=None)) == []  # pays none
    assert details(state(assignment_intended=True)) == []  # your plan
    put = ShortLeg(20, 115.0, "P", 0.63, 7, 0.10)
    assert details(StrategyState(cls=SHORT, spot=110.0, short_legs=(put,))) == []


def test_short_leg_takes_the_next_dividend_before_its_expiry():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.monitor import short_leg

    p = leg("c", "C", 105, -1)  # expires in 30 days
    soon = TODAY + timedelta(days=7)
    after = TODAY + timedelta(days=60)
    x = short_leg(p, TODAY, 110.0, Quote(5.3, 5.5), ((soon, 0.63), (after, 0.63)))
    assert (x.dividend, x.ex_days) == (0.63, 7)
    assert x.extrinsic == pytest.approx(0.4)  # mid 5.40 less intrinsic 5
    y = short_leg(p, TODAY, 110.0, Quote(5.3, 5.5), ((after, 0.63),))
    assert y.dividend is None and y.ex_days is None
    assert short_leg(p, TODAY, None, None, ()).extrinsic is None


# ----------------------------------------------- phase 2: the small IV surface


def test_smile_interpolates_with_flat_ends():
    from thetagang_nt.pm.core.surface import Smile

    s = Smile(((-0.1, 0.40), (0.0, 0.30), (0.1, 0.26)))
    assert s.iv_at(-0.05) == pytest.approx(0.35)
    assert s.iv_at(0.05) == pytest.approx(0.28)
    assert s.iv_at(-0.5) == 0.40 and s.iv_at(0.5) == 0.26  # flat past the quotes
    assert s.iv_at(0.0) == pytest.approx(0.30)


def test_smile_strikes_are_the_listed_ones_nearest_the_moves():
    from thetagang_nt.pm.core.surface import strikes_around

    listed = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0]
    off = (-0.1, -0.05, 0.0, 0.05, 0.1)
    assert strikes_around(listed, 100.0, off) == [90.0, 95.0, 100.0, 105.0, 110.0]
    assert strikes_around([100.0], 100.0, off) == [100.0]  # sparse: fewer strikes
    assert strikes_around([], 100.0, off) == []


def test_smile_recovers_the_vols_and_leaves_out_bad_quotes():
    import math

    from thetagang_nt.pm.core.american import american
    from thetagang_nt.pm.core.scenarios import RATE
    from thetagang_nt.pm.core.surface import measure

    years, divs = 60 / 365, ((20 / 365, 0.5),)

    def row(k, vol):
        call = k >= 100
        p = american(100, k, years, vol, call, RATE, divs)
        return (k, call, Quote(p * 0.99, p * 1.01))

    rows = [row(90.0, 0.40), row(100.0, 0.30), row(110.0, 0.26)]
    s = measure(rows, 100.0, years, divs, 0.25)
    assert [round(v, 2) for _, v in s.points] == [0.40, 0.30, 0.26]
    assert s.points[0][0] == pytest.approx(math.log(0.9))
    # wide, missing or zero-bid quotes don't count; too few points: no smile
    bad = [
        (95.0, False, Quote(0.5, 2.0)),
        (105.0, True, Quote(None, None)),
        (115.0, True, Quote(0.0, 0.1)),
    ]
    assert len(measure(rows + bad, 100.0, years, divs, 0.25).points) == 3
    assert measure(rows[:2] + bad, 100.0, years, divs, 0.25) is None


def test_sticky_delta_gap_follows_the_skew_and_the_position():
    from thetagang_nt.pm.core.scenarios import PriceModel
    from thetagang_nt.pm.core.surface import Smile, sticky_gaps

    # the usual skew: IV falls as the strike rises against spot
    skew = Smile(((-0.2, 0.40), (0.0, 0.30), (0.2, 0.20)))
    flat = Smile(((-0.2, 0.30), (0.0, 0.30), (0.2, 0.30)))
    p = leg("c", "C", 100, 1, mid=None)
    g = {"c": SimpleNamespace(underlying=100.0, vol=0.3)}
    moves = (-0.05, 0.0, 0.05)

    def gaps(legs, smiles):
        return sticky_gaps(legs, g, TODAY, smiles, moves, 5, PriceModel())

    long_gaps, share = gaps((p,), {EXP: skew})
    assert share == 1.0
    assert long_gaps[0.0] == pytest.approx(0.0)  # no move: the two rules agree
    # price up: the strike is now below spot, so sticky delta reads a higher IV
    assert long_gaps[0.05] > 0 > long_gaps[-0.05]
    short = leg("c", "C", 100, -1, mid=None)
    assert gaps((short,), {EXP: skew})[0][0.05] == pytest.approx(-long_gaps[0.05])
    assert gaps((p,), {EXP: flat})[0][0.05] == pytest.approx(0.0)  # no skew, no gap
    # a leg without a smile keeps its IV and shows in the coverage
    later = EXP + timedelta(days=30)
    other = leg("c", "C", 100, -1, mid=None, expiry=later)
    _, partial = gaps((p, other), {EXP: skew})
    assert partial == 0.5
    assert gaps((p,), {}) is None
    # far outside the quoted strikes: no gap, and the coverage says why
    far = leg("c", "C", 150, 1, mid=None)
    far_gaps, far_share = gaps((far,), {EXP: skew})
    assert far_share == 0.0 and far_gaps[0.05] == pytest.approx(0.0)


def test_the_monitor_measures_one_smile_per_cycle_and_publishes_the_gap(monkeypatch):
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    import asyncio

    from thetagang_nt import monitor
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import PortfolioMonitor, _Families
    from thetagang_nt.pm.core.scenarios import PriceModel
    from thetagang_nt.pm.core.surface import Smile

    held = [
        leg("c", "C", 100, 1, mid=3.0),
        leg("d", "C", 100, 1, mid=3.0, expiry=EXP + timedelta(days=30)),
    ]
    monkeypatch.setattr(monitor, "held_legs", lambda _c: held)
    done = []

    async def measure_smile(symbol, expiry, today):
        done.append((symbol, expiry))
        return Smile(((-0.1, 0.35), (0.0, 0.30), (0.1, 0.27)))

    fake = SimpleNamespace(
        cache=None,
        log=None,
        _pm=PMConfig(),
        _smiles={("AA", date(2020, 1, 1)): (NOW, None)},  # no longer held: dropped
        _dividends={"AA": ()},
        _measure_smile=measure_smile,
    )
    asyncio.run(PortfolioMonitor._scan_surface(fake, TODAY))
    assert len(done) == 1 and set(fake._smiles) == {done[0]}
    asyncio.run(PortfolioMonitor._scan_surface(fake, TODAY))
    assert len(done) == 2 and len(fake._smiles) == 2
    asyncio.run(PortfolioMonitor._scan_surface(fake, TODAY))
    assert len(done) == 2  # both fresh: nothing due
    fake._dividends = {}  # dividends not loaded: nothing measured
    fake._smiles = {}
    asyncio.run(PortfolioMonitor._scan_surface(fake, TODAY))
    assert len(done) == 2

    g = {"c": SimpleNamespace(**greeks().__dict__, underlying=100.0)}
    (st,) = find_structures(held[:1])
    entries = [("AA", 100.0, st, structure_figures(st, g, TODAY), "-")]
    smile = Smile(((-0.1, 0.35), (0.0, 0.30), (0.1, 0.27)))
    pm = SimpleNamespace(
        _quote={"c": Quote(3.0, 3.1)}.get,
        _pm_known={},
        _journal=None,
        _pm=PMConfig(),
        _closes={},
        _pm_cls={},
        _pm_last={},
        _rolls={},
        _smiles={("AA", EXP): (NOW, smile)},
        _dividends={"AA": ()},
        _iv_model_of=lambda *a: None,
        _price_model_of=lambda *a: PriceModel(),
        log=SimpleNamespace(warning=print),
    )
    f = _Families()
    PortfolioMonitor._portfolio_manager(pm, f, entries, g, TODAY)
    fields = {s.labels["field"]: s.value for s in f.by_name["pm"].samples}
    assert fields["sticky_coverage"] == 1.0 and fields["american"] == 1.0
    moves = {s.labels["move"] for s in f.by_name["pm_sticky"].samples}
    assert moves == {"-0.05", "-0.02", "0", "0.02", "0.05"}


# ------------------------------------------- no bid, stale quotes and liquidity


def test_no_bid_is_a_bid_of_zero_flagged_on_long_legs_only():
    short, long_ = leg("s", "C", 50, -1), leg("l", "C", 50, 1)
    q = Quote(0.0, 0.01, NOW)
    assert leg_flags(short, q, greeks(), NOW) == ()  # bought back at the ask
    assert leg_flags(long_, q, greeks(), NOW) == ("zero bid",)
    assert "missing quote" in leg_flags(short, Quote(0.5, None, NOW), greeks(), NOW)
    assert values((short,), {"s": q}) == pytest.approx((-0.5, -1.0))
    assert values((long_,), {"l": q}) == pytest.approx((0.5, 0.0))


def test_wide_spreads_skip_ticks_and_price_the_cost_to_cross():
    from thetagang_nt.pm.core.quality import is_wide, wide_legs

    assert is_wide(Quote(1.0, 3.9))
    assert not is_wide(Quote(0.02, 0.03))  # 40%, but one tick
    assert not is_wide(Quote(0.0, 0.01))  # no bid, a cent of ask
    assert is_wide(Quote(0.0, 0.75))  # no bid and a real ask
    assert not is_wide(Quote(2.0, 2.1))
    assert not is_wide(Quote(None, 0.5)) and not is_wide(None)
    p = leg("b", "P", 19, 5)
    ((label, pct, cost),) = wide_legs((p,), {"b": Quote(1.0, 3.9)})
    assert label == f"AA {EXP:%Y%m%d} 19P"
    assert pct == pytest.approx(2.9 / 2.45)
    assert cost == pytest.approx(725.0)  # half of 2.90, 5 contracts


def test_a_wide_spread_is_a_p2_signal_that_leaves_other_signals_working():
    from thetagang_nt.pm.core.review import P2 as _P2

    wide = (("BKSY 19P", 1.18, 725.0),)
    for found, out in (
        long_options.decide(StrategyState(cls=LONG, wide_legs=wide)),
        short_options.decide(StrategyState(cls=SHORT, wide_legs=wide)),
        long_vix.decide(StrategyState(cls=LONG, wide_legs=wide)),
    ):
        assert out == "REVIEW"  # not "REVIEW - data unreliable"
        assert [(x.priority, x.name) for x in found] == [(_P2, "wide spread")]
        assert "118%, $725 to cross" in found[0].detail
        assert "rough" in found[0].detail
    # a P0 behind a wide spread reaches CLOSE review; before, data hid nothing less
    s = StrategyState(cls=LONG, wide_legs=wide, thesis_valid=False)
    assert long_options.decide(s)[1] == "CLOSE review"


def test_quote_prefers_a_recent_direct_read_when_the_adapter_cant_know():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import PortfolioMonitor

    now = datetime.now(timezone.utc)

    def tick(bid, ask, age):
        return SimpleNamespace(
            bid_price=SimpleNamespace(as_double=lambda: bid),
            ask_price=SimpleNamespace(as_double=lambda: ask),
            ts_event=(now - timedelta(seconds=age)).timestamp() * 1e9,
        )

    def read(q, direct):
        fake = SimpleNamespace(
            cache=SimpleNamespace(quote_tick=lambda _: q),
            _direct={"x": direct} if direct else {},
            _pm=PMConfig(),
        )
        return PortfolioMonitor._quote(fake, "x")

    fresh = Quote(0.0, 0.05, now)
    # the adapter never saw the no-bid option at all
    assert read(None, fresh) is fresh
    # it keeps an old bid after IB shows none, even if its ask just ticked
    assert read(tick(0.5, 0.6, 1), fresh) is fresh
    # two-sided direct: the newer of the two wins, so a quiet quote isn't stale
    quiet = tick(1.0, 1.2, 900)
    two_sided = Quote(1.0, 1.2, now)
    assert read(quiet, two_sided) is two_sided
    newer = read(tick(1.1, 1.3, -5), two_sided)
    assert (newer.bid, newer.ask) == (1.1, 1.3)
    # a direct read older than the stale limit no longer counts
    old = Quote(0.0, 0.05, now - timedelta(seconds=900))
    assert read(tick(1.0, 1.2, 1), old).bid == 1.0
    assert read(None, None).bid is None


def test_direct_quotes_rotate_and_a_gone_market_is_dropped():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    import asyncio

    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import PortfolioMonitor

    legs = [leg(f"c{i}", "C", 100 + i, -1) for i in range(5)]
    asked = []

    async def quotes(wanted):
        asked.append(wanted)
        out = {k: (None, 0.05) for k in wanted}
        out.pop(("AA", f"{EXP:%Y%m%d}", 101, "C"), None)  # no market at all
        return out

    fake = SimpleNamespace(
        cache=SimpleNamespace(quote_tick=lambda _: None),
        log=SimpleNamespace(warning=print, info=print),
        _extras=SimpleNamespace(quotes=quotes),
        _direct={"gone": Quote(1.0, 1.1, NOW)},  # no longer held
        _pm=PMConfig(roll_quote_lines=2),
    )
    asyncio.run(PortfolioMonitor._read_direct_quotes(fake, legs))
    assert len(asked[0]) == 2
    assert fake._direct["c0"].bid == 0.0 and fake._direct["c0"].ask == 0.05
    assert "c1" not in fake._direct and "gone" not in fake._direct
    asyncio.run(PortfolioMonitor._read_direct_quotes(fake, legs))
    # oldest first: the ones not yet read, not the same two again
    assert ("AA", f"{EXP:%Y%m%d}", 100, "C") not in asked[1]
    asyncio.run(PortfolioMonitor._read_direct_quotes(fake, []))  # nothing held
    fake._extras = None
    asyncio.run(PortfolioMonitor._read_direct_quotes(fake, legs))  # no connection


def test_iv_history_skips_no_bid_and_wide_legs():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.monitor import iv_rows
    from thetagang_nt.pm.core.quality import QualityLimits

    short = leg("s", "C", 50, -1)
    g = {"s": SimpleNamespace(**greeks().__dict__, underlying=41.0)}
    assert iv_rows((short,), g, {"s": Quote(0.0, 0.01)}, NOW, QualityLimits()) == []
    assert iv_rows((short,), g, {"s": Quote(1.0, 3.0)}, NOW, QualityLimits()) == []
    assert len(iv_rows((short,), g, {"s": Quote(1.0, 1.1)}, NOW, QualityLimits())) == 1


def test_a_premium_gives_greeks_when_the_cache_has_no_price():
    pytest.importorskip("nautilus_trader")
    from nautilus_trader.cache.cache import Cache
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AssetClass, OptionKind
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.instruments import OptionContract
    from nautilus_trader.model.objects import Price, Quantity

    from thetagang_nt.portfolio import option_greeks

    iid = InstrumentId.from_str("AA261218C00050000.SMART")
    ins = OptionContract(
        iid,
        Symbol(iid.symbol.value),
        AssetClass.EQUITY,
        USD,
        2,
        Price.from_str("0.01"),
        Quantity.from_int(100),
        Quantity.from_int(1),
        "AA",
        OptionKind.CALL,
        Price.from_str("50.00"),
        0,
        int(datetime(2026, 12, 18, 21, tzinfo=timezone.utc).timestamp() * 1e9),
        0,
        0,
    )
    cache = Cache()
    cache.add_instrument(ins)
    assert option_greeks(cache, None, ins, TODAY) is None  # no price at all
    assert option_greeks(cache, None, ins, TODAY, premium=0.05) is None  # no spot


def test_a_leg_expiring_today_cannot_lose_on_a_later_move():
    from thetagang_nt.pm.core.scenarios import RATE, grid

    today_leg, g = priced("s", "C", 110, -2, 100.0, 0.5, expiry=TODAY)
    s = grid((today_leg,), {"s": g}, TODAY, RATE, 0.5)
    # out of the money and expiring now: nothing moves it before it is gone
    assert s.worst_loss() == pytest.approx(0.0)
    assert s.stress_loss() == pytest.approx(0.0)


def test_an_in_the_money_leg_at_expiry_becomes_stock_and_keeps_moving():
    from thetagang_nt.pm.core.scenarios import RATE, grid

    itm, g = priced("s", "C", 90, -1, 100.0, 0.5, expiry=TODAY)
    s = grid((itm,), {"s": g}, TODAY, RATE, None)
    # assigned today at 90, short stock: at +5% after 10 days it owes 105 - 90
    assert s.fixed[(10, 0.05, 0)] == pytest.approx(-1500.0 - s.v_model)
    # and gains if the stock then falls below the strike's value today
    assert s.fixed[(10, -0.05, 0)] == pytest.approx(-500.0 - s.v_model)


def test_a_leg_expiring_inside_the_horizon_settles_on_its_own_days_move():
    import math

    from thetagang_nt.pm.core.efficiency import calendar_days
    from thetagang_nt.pm.core.scenarios import RATE, grid

    # expires in 3 days; the 10-day horizon's +5% is only part way there by then
    soon = TODAY + timedelta(days=3)
    days = calendar_days(TODAY, 10)
    at_expiry = 100 * (1 + 0.05 * math.sqrt(3 / days))
    assert 100 < at_expiry < 104  # below a 104 strike at its expiry...
    otm, g = priced("s", "C", 104, -1, 100.0, 0.5, expiry=soon)
    s = grid((otm,), {"s": g}, TODAY, RATE, None)
    # ...so it expires worthless, though the horizon's price is above the strike
    assert s.fixed[(10, 0.05, 0)] == pytest.approx(-s.v_model)
    near, g2 = priced("s", "C", 101, -1, 100.0, 0.5, expiry=soon)
    s2 = grid((near,), {"s": g2}, TODAY, RATE, None)
    assert s2.fixed[(10, 0.05, 0)] == pytest.approx(-(105 - 101) * 100 - s2.v_model)


def test_no_bid_legs_skip_the_iv_sanity_check():
    short = leg("s", "C", 50, -1)
    assert leg_flags(short, Quote(0.0, 0.01, NOW), greeks(vol=9.0), NOW) == ()
    assert "implausible IV" in leg_flags(
        short, Quote(0.5, 0.6, NOW), greeks(vol=9.0), NOW
    )


# ----------------------------------------------------------- the forward test


def test_the_monitor_records_the_forward_test_only_on_good_quotes():
    pytest.importorskip("nautilus_trader")
    pytest.importorskip("psycopg")
    from thetagang_nt.config import PMConfig
    from thetagang_nt.monitor import (
        PortfolioMonitor,
        RollScan,
        _Families,
        reliable_for_the_day,
    )

    assert reliable_for_the_day(())
    assert reliable_for_the_day(("AA 40C: zero bid",))
    assert not reliable_for_the_day(("AA 40C: stale quote",))
    assert not reliable_for_the_day(("AA 40C: missing quote", "AA 40C: no greeks"))

    p = leg("c", "C", 100, 1, avg=3.0)
    g = {"c": SimpleNamespace(**greeks().__dict__, underlying=100.0)}
    (st,) = find_structures([p])
    entries = [("AA", 100.0, st, structure_figures(st, g, TODAY), "-")]
    written = {}

    class Journal:
        def known(self, _today):
            return {}

        def entry(self, _ids):
            return {}

        def record(self, *a):
            pass

        def decide(self, *a):
            pass

        def record_iv(self, *a):
            pass

        def record_forecasts(self, forecasts, marks, rolls, today, now):
            written.update(forecasts=forecasts, marks=marks, rolls=rolls)

    candidate = roll_candidate()

    def run(quote, scan_at):
        now = datetime.now(timezone.utc)
        fake = SimpleNamespace(
            _quote=lambda _iid: quote,
            _pm_known={},
            _journal=Journal(),
            _pm=PMConfig(),
            _closes={},
            _pm_cls={},
            _pm_last={},
            _rolls={"AA-long-call-1": RollScan(scan_at or now, [candidate])},
            _smiles={},
            _dividends={},
            _iv_model_of=lambda *a: None,
            _price_model_of=lambda *a: None,
            log=SimpleNamespace(warning=print),
        )
        PortfolioMonitor._portfolio_manager(fake, _Families(), entries, g, now.date())

    run(Quote(3.0, 3.1, datetime.now(timezone.utc)), None)
    (row,) = written["forecasts"]
    assert row.strategy_id == "AA-long-call-1" and row.legs_key == "c:1"
    assert row.v_exec == pytest.approx(300.0) and row.wcl_5 is not None
    assert row.view in ("HOLD", "CLOSE", "-")
    (mark,) = written["marks"]
    assert (mark.bid, mark.ask, mark.underlying) == (3.0, 3.1, 100.0)
    assert written["rolls"][0].candidate == candidate.label

    # after the close: stale quotes don't overwrite the day's forecast
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    run(Quote(3.0, 3.1, old), None)
    assert written["forecasts"] == []
    # yesterday's scan isn't today's candidate
    run(Quote(3.0, 3.1, datetime.now(timezone.utc)), old - timedelta(days=2))
    assert written["rolls"] == []
