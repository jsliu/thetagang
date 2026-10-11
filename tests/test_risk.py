"""Pure risk math for the discretionary strategy (thetagang.risk)."""

import pytest

from thetagang.risk import compute_beta


def test_compute_beta():
    """Stock log returns that are exactly 2x the index give beta 2."""
    import numpy as np

    index = {d: 100 * np.exp(0.01 * ((d % 3) - 1) + 0.001 * d) for d in range(60)}
    stock = {d: v**2 for d, v in index.items()}
    assert compute_beta(stock, index) == pytest.approx(2.0)
    assert compute_beta({0: 1.0}, index) is None  # too little data


def test_bs_price_put_call_parity_and_stress():
    """With zero rates, call - put = spot - strike; short put loses more than linear."""
    from thetagang.risk import bs_price, scenario_pnl

    call = bs_price(100, 95, 0.25, 0.3, "C")
    put = bs_price(100, 95, 0.25, 0.3, "P")
    assert call - put == pytest.approx(5.0)
    assert 0 < put < 5 < call

    # one short 95 put, 100 shares per contract, IV 30%, 3 months, beta 2
    leg = (2.0, -100.0, 100.0, 95.0, 0.25, 0.3, "P", 0.0, 0.0)
    down_beta = scenario_pnl([leg], -0.10, use_beta=True)  # stock -20%
    down_flat = scenario_pnl([leg], -0.10, use_beta=False)  # stock -10%
    assert down_beta < down_flat < 0
    assert down_beta < 2 * down_flat  # convex: loss grows faster than the move


def test_scenario_pnl_survives_extreme_beta():
    """A thin-history beta that implies a >100% fall must not crash the stress table."""
    from thetagang.risk import scenario_pnl

    leg = (12.0, -100.0, 50.0, 45.0, 0.25, 0.4, "P", 0.0, 0.0)  # beta 12 x -10%
    assert scenario_pnl([leg], -0.10, use_beta=True) < 0


def test_zero_delta_targets():
    """Both totals reach zero: delta $ and beta x delta $."""
    from thetagang.risk import zero_delta_targets

    plain = {"HIGH": 50_000.0, "LOW": 50_000.0, "SHORT": -57_158.0}
    betas = {"HIGH": 2.0, "LOW": 0.5, "SHORT": 1.0}
    target = zero_delta_targets(plain, betas)
    assert sum(target.values()) == pytest.approx(0, abs=1e-6)
    assert sum(betas[s] * d for s, d in target.items()) == pytest.approx(0, abs=1e-6)
    # the high-beta long is cut more than the low-beta long
    assert target["HIGH"] < target["LOW"]

    # equal betas: the two conditions coincide and the same-% rule is recovered
    same = zero_delta_targets({"A": 100_000.0, "B": -57_158.0}, {"A": 1.0, "B": 1.0})
    assert sum(same.values()) == pytest.approx(0, abs=1e-6)
    assert 1 - same["A"] / 100_000 == pytest.approx(same["B"] / -57_158 - 1)
    assert zero_delta_targets({}, {}) == {}


def test_side_weight():
    """Weights are positive shares of gross exposure, labelled by market side."""
    from rich.text import Text

    from thetagang.risk import side_weight

    def plain(beta_dollars):
        return Text.from_markup(side_weight(beta_dollars, 145_478.0)).plain

    assert plain(72_739.0) == "50.0% long"
    assert plain(-72_739.0) == "50.0% short"
    assert plain(0.0) == "0.0%"


def test_size_adjustment_never_flips():
    """Closing is capped at what is held; spreads open/close as whole spreads."""
    from thetagang.risk import Adjuster, size_adjustment

    spread = Adjuster(30, "HOOD 20261120 50/45P", True, 15.0, 0, 0, 0, 2, -1)
    assert size_adjustment(spread, 30) == (2, "open 2 spread HOOD 20261120 50/45P")
    # needs -45 shares = 3 spreads, but only 2 are held
    assert size_adjustment(spread, -45) == (-2, "close 2 spread HOOD 20261120 50/45P")

    short_put = Adjuster(10, "HOOD P40", False, -10.0, 0, 0, 0, 1, 1)
    assert size_adjustment(short_put, -30) == (1, "+1 HOOD P40")  # buy back 1, not 3
    assert size_adjustment(short_put, 30) == (-3, "-3 HOOD P40")  # sell 3 more
    assert size_adjustment(short_put, 2) == (0, "0")


def test_stock_trade_never_flips():
    """Selling is capped at long shares held; buying back at shares short."""
    from thetagang.risk import stock_trade

    assert stock_trade(-300.4, 100) == -100  # only 100 to sell
    assert stock_trade(-40.6, 100) == -41
    assert stock_trade(250, -100) == 100  # only 100 short to buy back
    assert stock_trade(50, 100) == 50  # adding to a long is fine


def test_position_greeks_and_symbol_greeks():
    from types import SimpleNamespace

    from ib_async import Option, PortfolioItem, Stock

    from thetagang.risk import contract_greeks, position_greeks, symbol_greeks

    def opt(symbol, strike, right, qty, con_id):
        c = Option(symbol, "20991120", strike, right, "SMART", "100", "USD")
        c.conId = con_id
        return PortfolioItem(c, qty, 1, 0, 1, 0, 0, "A")

    stock = PortfolioItem(
        Stock("HOOD", "SMART", "USD", conId=1), 200, 50, 0, 40, 0, 0, "A"
    )
    put = opt("HOOD", 45.0, "P", -2, 2)
    call = opt("AMD", 150.0, "C", -1, 3)

    def g(delta):
        return SimpleNamespace(
            delta=delta,
            gamma=0.03,
            theta=-0.04,
            vega=0.06,
            undPrice=50.0,
            impliedVol=0.5,
        )

    greeks = {2: g(-0.25)}
    by_position = position_greeks([stock, put, call], greeks)
    assert by_position[1] == (200.0, 0.0, 0.0, 0.0)
    assert by_position[2] == pytest.approx((50.0, -6.0, 8.0, -12.0))
    assert 3 not in by_position  # no greeks from IB
    assert symbol_greeks([stock, put, call], by_position) == {
        "HOOD": pytest.approx((250.0, -6.0, 8.0, -12.0)),
        "AMD": None,  # its option has no greeks
    }
    per_contract = contract_greeks([stock, put, call], greeks)
    assert per_contract == {1: (1.0, 0.0, 0.0, 0.0), 2: (-0.25, 0.03, -0.04, 0.06)}
