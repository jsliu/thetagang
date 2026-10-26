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
    assert (
        strategy is app.strategy and not monitor.publish and cid == 11
    )  # no doubled metrics
    monitor, strategy, cid = split(app, "all")
    assert strategy is app.strategy and monitor.publish and cid == 11
    with pytest.raises(ValueError, match="role"):
        split(app, "everything")


# ------------------------------------------- dashboard: proposals and actions

from thetagang_nt.rules import (  # noqa: E402
    position_actions,
    side_weight,
    size_adjustment,
    stock_trade,
    symbol_adjustment,
    vix_hedge_action,
    zero_delta_targets,
)

TODAY = date(2026, 11, 2)  # EXP is 46 days out


def test_zero_delta_targets_zero_both_totals() -> None:
    now = {"A": 50_000.0, "B": -10_000.0, "C": 20_000.0}
    betas = {"A": 1.5, "B": 0.8, "C": 1.0}
    target = zero_delta_targets(now, betas)
    assert sum(target.values()) == pytest.approx(0, abs=1e-6)
    assert sum(betas[s] * v for s, v in target.items()) == pytest.approx(0, abs=1e-6)
    assert side_weight(-300.0, 1000.0) == (pytest.approx(0.3), "short")
    assert side_weight(0.1, 1000.0) == (0.0, "")


def test_net_target_contracts_never_flip_or_sell_covering_shares() -> None:
    put = opt("P", 30, -2, 1.0, 0.5)  # short 2 puts: +50 share delta per contract
    unit = {put.instrument_id: (-50.0, 0.0, 1.0, -2.0, 0.5)}
    traded, text, _ = symbol_adjustment(
        [put], -500.0, unit
    )  # wants -500: only 2 to buy back
    assert traded == -100.0 and text == "+2 P30"
    shares, call = stock(200), opt("C", 40, -1, 1.0, 0.5)
    traded, text, _ = symbol_adjustment(
        [shares, call], -500.0, {}
    )  # call has no greeks
    assert traded == -100 and text == "-100 shares"  # 100 of the 200 cover the call
    assert stock_trade(-300, 100) == -100 and stock_trade(50, 100) == 50


def test_actions_follow_roll_when_and_show_take_profit() -> None:
    rw = RollWhen(pnl=0.9, dte=10, min_pnl=0.7, close_at_pnl=0.9)

    def act(*legs: Leg, deltas: dict | None = None) -> dict:
        found = position_actions(list(legs), 40.0, rw, 0.8, 0.8, deltas or {}, TODAY)
        return {a.name: a for a in found}

    shorts = act(opt("P", 30, -1, 1.0, 0.05), opt("C", 50, -1, 2.0, 1.5))
    assert shorts["AA 20261218 30P"].action == "roll"  # 95% of the premium captured
    assert shorts["AA 20261218 30P"].take_profit == pytest.approx(
        0.1
    )  # buy back at <= 10% of 1.0
    assert shorts["AA 20261218 50C"].action == "keep"  # 25% captured
    long_call = act(opt("C", 60, 2, 1.0, 1.2), deltas={"C60": 0.85})
    assert long_call["AA 20261218 60C"].action.startswith("roll or close by hand")
    # a short and a long call of different sizes are a diagonal: left alone, as before
    mixed = act(opt("C", 50, -1, 2.0, 1.5), opt("C", 60, 2, 1.0, 1.2))
    assert {a.action for a in mixed.values()} == {"by hand (calendar/diagonal)"}


def test_spread_actions_and_index_options_are_left_alone() -> None:
    rw = RollWhen(pnl=0.9, dte=10, min_pnl=0.7, close_at_pnl=0.9)
    # debit put spread: paid 2 for a 5 wide spread now worth 4.5 -> past the 80% take-profit (4.4)
    debit = position_actions(
        [opt("P", 40, 1, 3.0, 5.0), opt("P", 35, -1, 1.0, 0.5)],
        38.0,
        rw,
        0.8,
        0.8,
        {},
        TODAY,
    )
    assert debit[0].kind == "debit spread" and debit[0].action == "close (take-profit)"
    assert debit[0].take_profit == pytest.approx(4.4)
    index_put = Leg("SPX P", "^SPX", "P", 6000.0, EXP, -1, 10.0, 1.0)
    assert (
        position_actions([index_put], 7000.0, rw, 0.8, 0.8, {}, TODAY)[0].action
        == "by hand (index option)"
    )


def test_vix_hedge_action() -> None:
    cfg = SimpleNamespace(
        enabled=True,
        close_hedges_when_vix_exceeds=30.0,
        allocation=[SimpleNamespace(lower_bound=15.0, upper_bound=30.0, weight=0.01)],
    )
    assert vix_hedge_action(5, 35.0, 33.0, cfg)[0] == "sell all (VIX above threshold)"
    assert vix_hedge_action(5, 20.0, 21.0, cfg)[0] == "keep"
    assert vix_hedge_action(0, 20.0, 21.0, cfg) == ("buy calls (1.0% of net liq)", 0.01)
    assert vix_hedge_action(0, 20.0, 40.0, cfg)[0] == "none (weight 0)"


def test_a_metric_label_may_be_called_name() -> None:
    """Regression: a `name` label once broke the whole monitor refresh."""
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.monitor import _Families

    f = _Families()
    f.set("action", "help", 2.0, name="AA 270115P40", field="qty")
    (sample,) = f.by_name["action"].samples
    assert (
        sample.labels == {"name": "AA 270115P40", "field": "qty"}
        and sample.value == 2.0
    )


def test_volga_matches_nautilus_black_scholes() -> None:
    """Volga = change in vega ($ per vol point) per vol point, checked by bumping
    the vol in NautilusTrader's own Black-Scholes."""
    nt = pytest.importorskip("nautilus_trader.core.nautilus_pyo3")
    from thetagang_nt.rules import volga

    spot, strike, years, vol, rate = 100.0, 120.0, 0.5, 0.30, 0.0425

    def vega(v: float) -> float:
        return nt.black_scholes_greeks(spot, rate, rate, v, True, strike, years).vega

    expected = vega(vol + 0.005) - vega(vol - 0.005)  # per vol point
    assert volga(vega(vol), spot, strike, years, vol, rate) == pytest.approx(
        expected, rel=2e-3
    )
    assert volga(1.0, spot, strike, years, float("nan"), rate) == 0.0


def test_dashboard_names_and_wording_follow_the_old_bot() -> None:
    pytest.importorskip("nautilus_trader")
    from thetagang_nt.monitor import action_text, take_profit_text
    from thetagang_nt.rules import Action, option_label, spread_name

    put, long_put = opt("P", 43, -1, 1.0, 0.5), opt("P", 40, 1, 0.4, 0.2)
    assert option_label(put) == "AA 20261218 43P"
    assert spread_name(put, long_put) == "AA 20261218 40/43P"
    index_put = Leg("x", "^SPX", "P", 6000.0, EXP, -1, 10.0, 1.0)
    assert option_label(index_put) == "SPX 20261218 6000P"
    short = Action(
        "short", "AA 20261218 43P", "AA", "P", -1, 40, 0.5, 0.1, "buy <=", "roll"
    )
    assert (
        take_profit_text(short) == "buy <= $0.10"
        and action_text(short.action) == "Roll"
    )
    hold = Action("long", "x", "AA", "C", 2, 40, None, None, "", "hold")
    assert take_profit_text(hold) == "-"
    alert = "roll or close by hand (|delta| >= 0.8)"
    assert action_text(alert) == "Roll or close by hand (|Δ| >= 0.8)"


def test_failed_implied_vol_counts_as_missing_greeks() -> None:
    """Regression: with no quote, Nautilus returned IV 1e-8 and delta 0; that must
    be "no greeks", never a zero-delta position."""
    pytest.importorskip("nautilus_trader")
    from thetagang_nt import portfolio

    from datetime import datetime, timezone

    expiry = datetime(2026, 12, 18, 21, tzinfo=timezone.utc)
    ins = SimpleNamespace(
        underlying="AA", id="AA C40", strike_price=40.0, expiration_utc=expiry
    )
    failed = SimpleNamespace(
        delta=0.0,
        gamma=0.0,
        theta=0.0,
        vega=0.0,
        vol=1e-8,
        underlying_price=40.0,
        price=0.0,
    )
    good = SimpleNamespace(
        delta=0.4,
        gamma=0.05,
        theta=-0.02,
        vega=0.08,
        vol=0.35,
        underlying_price=40.0,
        price=1.2,
    )

    def run(mid: float | None, g: SimpleNamespace) -> object:
        calc = SimpleNamespace(instrument_greeks=lambda iid: g)
        original = portfolio.price
        portfolio.price = lambda cache, iid: mid
        try:
            return portfolio.option_greeks(None, calc, ins, date(2026, 11, 2))
        finally:
            portfolio.price = original

    assert run(1.2, failed) is None  # the solver failed
    assert run(None, good) is None  # no price of its own
    assert run(1.2, good).delta == 0.4
