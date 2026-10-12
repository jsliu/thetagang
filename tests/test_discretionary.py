"""Discretionary strategy: run.strategies = ["discretionary"]."""

import itertools
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from ib_async import Option, PortfolioItem, Stock, Ticker
from rich.table import Table
from rich.text import Text

from thetagang.config import Config
from thetagang.orders import Orders
from thetagang.strategies.discretionary_engine import (
    DiscretionaryEngine,
    quoted_price,
    roll_strike_limit,
)
from thetagang.strategies.options_engine import OptionsStrategyEngine
from thetagang.trading_operations import (
    NoValidContractsError,
    OrderOperations,
    open_interest_ok,
)

ACCOUNT = "DUX"


def _config_dict(run=None, portfolio=None):
    data = {
        "meta": {"schema_version": 2},
        "run": run or {"strategies": ["discretionary"]},
        "runtime": {
            "account": {"number": ACCOUNT},
            "option_chains": {"expirations": 4, "strikes": 10},
        },
        "strategies": {
            "wheel": {
                "defaults": {
                    "target": {"dte": 7, "minimum_open_interest": 1000},
                    "roll_when": {"dte": 3, "pnl": 0.9, "close_at_pnl": 0.95},
                }
            }
        },
    }
    if portfolio is not None:
        data["portfolio"] = portfolio
    return data


def _opt(symbol, expiry, strike, right, position, avg=1.0, upnl=0.0, con_id=0):
    c = Option(symbol, expiry, strike, right, "SMART", "100", "USD", conId=con_id)
    return PortfolioItem(c, position, 0, 0, avg, upnl, 0, ACCOUNT)


def _stock(symbol, position, price=50.0, avg=40.0, exchange=""):
    c = Stock(symbol, "SMART", "USD", primaryExchange=exchange)
    return PortfolioItem(c, position, price, price * position, avg, 0, 0, ACCOUNT)


def _ticker(mocker, price, contract=None):
    t = mocker.Mock(modelGreeks=None, contract=contract)
    t.midpoint.return_value = price
    t.marketPrice.return_value = price
    return t


@pytest.fixture
def config():
    return Config(**_config_dict())


@pytest.fixture
def engine(config, mocker):
    mocker.patch("thetagang.strategies.discretionary_engine.log")
    mocker.patch("thetagang.strategies.discretionary_risk.log")
    mocker.patch("thetagang.strategies.discretionary_spreads.log")
    mocker.patch("thetagang.strategies.discretionary_trades.log")
    orders = Orders()
    order_ops = OrderOperations(
        config=config, account_number=ACCOUNT, orders=orders, data_store=None
    )
    eng = DiscretionaryEngine(
        config=config,
        ibkr=mocker.Mock(),
        order_ops=order_ops,
        option_scanner=mocker.Mock(),
        qualified_contracts={},
        put_is_itm=mocker.AsyncMock(return_value=False),
        call_is_itm=mocker.AsyncMock(return_value=False),
        get_primary_exchange=lambda symbol: "NASDAQ",
    )
    return eng


# ----------------------------------------------------------------- config


def test_manager_strategy_compiles_to_roll_close_and_manager_stages():
    config = Config(**_config_dict())
    assert config.discretionary_mode is True
    assert [s.id for s in config.run.resolved_stages()] == [
        "options_roll_positions",
        "options_close_positions",
        "post_discretionary",
    ]
    assert config.portfolio.symbols == {}  # [portfolio] is optional


def test_manager_overrides_need_no_weight():
    config = Config(**_config_dict(portfolio={"symbols": {"TQQQ": {"max_dte": 45}}}))
    assert config.portfolio.symbols["TQQQ"].max_dte == 45
    assert config.portfolio.symbols["TQQQ"].weight == 0.0


def test_manager_cannot_run_with_position_opening_strategies():
    with pytest.raises(ValueError, match="discretionary together with"):
        Config(**_config_dict(run={"strategies": ["discretionary", "wheel"]}))


def test_wheel_still_requires_symbols_and_weights():
    def wheel(symbols):
        data = _config_dict(
            run={"strategies": ["wheel"]}, portfolio={"symbols": symbols}
        )
        data["runtime"]["account"]["margin_usage"] = 0.5  # still required
        return Config(**data)

    with pytest.raises(ValueError, match="At least one symbol"):
        wheel({})
    with pytest.raises(ValueError, match="Symbol weights must sum to 1.0"):
        wheel({"AAA": {"weight": 0.5}})
    assert wheel({"AAA": {"weight": 1.0}}).discretionary_mode is False


# --------------------------------------------------------------- symbols


def test_discover_symbols_manages_every_held_symbol(engine, mocker):
    from thetagang.config_models import SymbolConfig

    engine.config.portfolio.symbols = {"HOOD": SymbolConfig(weight=0.0, delta=0.2)}
    engine.ibkr.portfolio = mocker.Mock(
        return_value=[
            _stock("AMD", 100, exchange="NASDAQ"),
            _opt("HOOD", "20261120", 50.0, "P", -1),
            _opt("VIX", "20261118", 20.0, "C", 5),
            _opt("SPX", "20261120", 5000.0, "P", -1),
            PortfolioItem(Stock("SHOP", "SMART", "CAD"), 10, 0, 0, 1, 0, 0, ACCOUNT),
        ]
    )
    engine.discover_symbols()
    assert list(engine.config.portfolio.symbols) == ["AMD", "HOOD"]
    assert engine.config.portfolio.symbols["AMD"].primary_exchange == "NASDAQ"
    assert engine.config.portfolio.symbols["HOOD"].delta == 0.2  # override kept

    engine.ibkr.portfolio = mocker.Mock(return_value=[])  # reconnect, not synced
    engine.discover_symbols()
    assert list(engine.config.portfolio.symbols) == ["AMD", "HOOD"]


# ----------------------------------------------------------- spread legs


def test_find_credit_verticals_pairs_puts_and_calls():
    put_short = _opt("HOOD", "20261120", 50.0, "P", -2)
    put_long_far = _opt("HOOD", "20261120", 40.0, "P", 2)
    put_long_near = _opt("HOOD", "20261120", 45.0, "P", 2)
    call_short = _opt("HOOD", "20261120", 70.0, "C", -1)
    call_long = _opt("HOOD", "20261120", 75.0, "C", 1)
    debit_long = _opt("HOOD", "20261120", 65.0, "C", 1)  # below the 70 call
    pairs = DiscretionaryEngine.find_credit_verticals(
        [put_short, put_long_far, put_long_near, call_short, call_long, debit_long]
    )
    assert (put_short, put_long_near) in pairs
    assert (call_short, call_long) in pairs
    assert len(pairs) == 2


def test_without_spread_legs(engine):
    spread_short = _opt("HOOD", "20261120", 50.0, "P", -1, avg=200.0)
    spread_long = _opt("HOOD", "20261120", 45.0, "P", 1, avg=50.0)
    lone_put = _opt("HOOD", "20261120", 40.0, "P", -1)
    diag_short = _opt("RDDT", "20261120", 170.0, "C", -1)
    diag_long = _opt("RDDT", "20261218", 160.0, "C", 1)
    covered_short = _opt("DXYZ", "20261218", 75.0, "C", -1)
    covered_long = _opt("DXYZ", "20270115", 70.0, "C", 1)
    positions = {
        "HOOD": [spread_short, spread_long, lone_put],
        "RDDT": [diag_short, diag_long],
        "DXYZ": [_stock("DXYZ", 100), covered_short, covered_long],
    }
    kept = engine.without_spread_legs(
        [spread_short, lone_put, diag_short, covered_short], positions
    )
    # vertical -> roll_spreads; diagonal -> left alone; the lone put and the
    # share-covered call roll on their own
    assert kept == [lone_put, covered_short]


# ------------------------------------------------------------ roll rules


@pytest.mark.asyncio
@pytest.mark.parametrize("right", ["P", "C"])
async def test_spread_roll_rules_match_standalone_checks(config, engine, mocker, right):
    """The manager's spread rule chain must agree with OptionsStrategyEngine's
    put_can_be_rolled / call_can_be_rolled, so they can't drift apart."""
    options = OptionsStrategyEngine(
        config=config,
        ibkr=mocker.Mock(),
        option_scanner=mocker.Mock(),
        order_ops=engine.order_ops,
        services=mocker.Mock(),
        target_quantities={},
        has_excess_puts=set(),
        has_excess_calls=set(),
        qualified_contracts={},
    )
    rw = config.strategies.wheel.defaults.roll_when
    kind = rw.calls if right == "C" else rw.puts
    check = options.call_can_be_rolled if right == "C" else options.put_can_be_rolled
    is_itm = "call_is_itm" if right == "C" else "put_is_itm"
    mocker.patch("thetagang.strategies.options_engine.log")

    for itm, itm_ok, always, max_dte, dte, pnl in itertools.product(
        [False, True],
        [False, True],
        [False, True],
        [None, 20],
        [2, 30],
        [-0.1, 0.5, 0.95],
    ):
        kind.itm, kind.always_when_itm, rw.max_dte = itm_ok, always, max_dte
        setattr(options, is_itm, mocker.AsyncMock(return_value=itm))
        expiry = (date.today() + timedelta(days=dte)).strftime("%Y%m%d")  # noqa: DTZ011
        position = _opt("HOOD", expiry, 50.0, right, -1, avg=100.0, upnl=pnl * 100)
        expected = await check(position, Table())
        assert engine.roll_rule_chain(right, itm, dte, pnl) == expected, (
            itm,
            itm_ok,
            always,
            max_dte,
            dte,
            pnl,
        )


def test_roll_strike_limit_is_symmetric():
    assert roll_strike_limit("P", 50.0, 1.5, False, None) == 51.5
    assert roll_strike_limit("C", 50.0, 1.5, False, None) == 48.5
    assert roll_strike_limit("P", 50.0, 1.5, True, None) == 50.0
    assert roll_strike_limit("C", 50.0, 1.5, True, None) == 50.0
    assert roll_strike_limit("P", 50.0, -2.0, False, None) == 50.0
    assert roll_strike_limit("C", 50.0, 1.5, False, 49.0) == 49.0


def test_open_interest_missing_is_strict_for_the_wheel_only():
    t = Ticker()  # ib_async default: open interest is NaN
    assert not open_interest_ok(t, "C", 1000)  # wheel
    assert open_interest_ok(t, "C", 1000, missing_ok=True)  # discretionary strategy
    t.callOpenInterest = 250
    assert not open_interest_ok(t, "C", 1000, missing_ok=True)  # real low OI


def test_quoted_price_rejects_missing_quotes(mocker):
    with pytest.raises(RuntimeError, match="no quote"):
        quoted_price(_ticker(mocker, float("nan")), "HOOD P45")


# --------------------------------------------------------- spread orders


def _spread_mocks(engine, mocker, mids, picks, longs):
    async def get_ticker(contract, **kwargs):
        return _ticker(mocker, mids[contract.conId], contract)

    engine.ibkr.get_ticker_for_contract = get_ticker
    engine.ibkr.qualify_contracts = mocker.AsyncMock(
        side_effect=[[leg] for leg in longs]
    )
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        side_effect=[_ticker(mocker, mids[p.conId], p) for p in picks]
    )


@pytest.mark.asyncio
async def test_roll_spread_builds_four_leg_combo(engine, mocker):
    short = _opt("HOOD", "20991120", 50.0, "P", -2, con_id=1)
    long = _opt("HOOD", "20991120", 45.0, "P", 2, con_id=2)
    new_short = Option("HOOD", "20991218", 48.0, "P", "SMART", conId=3)
    new_long = Option("HOOD", "20991218", 43.0, "P", "SMART", conId=4)
    _spread_mocks(
        engine, mocker, {1: 0.40, 2: 0.10, 3: 1.50, 4: 0.60}, [new_short], [new_long]
    )

    await engine.roll_spread("HOOD", short, long, "SMART", False)

    ((combo, order, _),) = engine.order_ops.orders.records()
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (1, "BUY"),
        (2, "SELL"),
        (3, "SELL"),
        (4, "BUY"),
    ]
    assert order.totalQuantity == 2  # the whole spread
    assert order.lmtPrice == pytest.approx(-0.60)  # 0.30 to close, 0.90 to open
    search = engine.option_scanner.find_eligible_contracts.call_args
    assert search.args[2] == 50.0  # nothing captured: no worse strike
    assert (45.0, "20991120") in search.kwargs["also_exclude"]  # old long excluded


@pytest.mark.asyncio
async def test_roll_spread_credit_only_searches_again_then_rejects(engine, mocker):
    engine.config.strategies.wheel.defaults.roll_when.puts.credit_only = True
    engine.config.runtime.orders.minimum_credit = 0.05
    short = _opt("HOOD", "20991120", 50.0, "P", -1, con_id=1)
    long = _opt("HOOD", "20991120", 45.0, "P", 1, con_id=2)
    first = Option("HOOD", "20991218", 48.0, "P", "SMART", conId=3)
    first_long = Option("HOOD", "20991218", 43.0, "P", "SMART", conId=4)
    second = Option("HOOD", "20991218", 49.0, "P", "SMART", conId=5)
    second_long = Option("HOOD", "20991218", 44.0, "P", "SMART", conId=6)
    _spread_mocks(
        engine,
        mocker,
        {1: 0.40, 2: 0.10, 3: 0.35, 4: 0.60, 5: 0.90, 6: 0.60},
        [first, second],
        [first_long, second_long],
    )

    with pytest.raises(NoValidContractsError):
        await engine.roll_spread("HOOD", short, long, "SMART", False)
    floors = [
        c.kwargs["minimum_price"]()
        for c in engine.option_scanner.find_eligible_contracts.call_args_list
    ]
    assert floors == pytest.approx([0.35, 0.95])  # 2nd search adds the long's cost
    assert (
        engine.order_ops.orders.records() == []
    )  # never a debit disguised as a credit


@pytest.mark.asyncio
async def test_roll_spread_skips_overlapping_candidate(engine, mocker):
    short = _opt("HOOD", "20991120", 100.0, "P", -1, con_id=1)
    long = _opt("HOOD", "20991120", 95.0, "P", 1, con_id=2)
    overlap = Option("HOOD", "20991120", 105.0, "P", "SMART", conId=3)
    reused = Option("HOOD", "20991120", 100.0, "P", "SMART", conId=1)
    good = Option("HOOD", "20991218", 98.0, "P", "SMART", conId=5)
    good_long = Option("HOOD", "20991218", 93.0, "P", "SMART", conId=6)
    _spread_mocks(
        engine,
        mocker,
        {1: 0.40, 2: 0.10, 3: 1.50, 5: 1.20, 6: 0.50},
        [overlap, good],
        [reused, good_long],
    )

    await engine.roll_spread("HOOD", short, long, "SMART", False)
    ((combo, _, _),) = engine.order_ops.orders.records()
    assert [leg.conId for leg in combo.comboLegs] == [1, 2, 5, 6]


@pytest.mark.asyncio
async def test_roll_spreads_close_rules_and_error_containment(engine, mocker):
    expiry = (date.today() + timedelta(days=20)).strftime("%Y%m%d")  # noqa: DTZ011
    # credit = 200 - 50 = $150; upnl 140 -> 93% P&L
    short = _opt("HOOD", expiry, 50.0, "P", -1, avg=200.0, upnl=140.0)
    long = _opt("HOOD", expiry, 45.0, "P", 1, avg=50.0)
    engine.close_spread = mocker.AsyncMock()
    rw = engine.config.strategies.wheel.defaults.roll_when

    # roll rule (90%) fires, roll fails, fallback off (no max_dte): stays open
    engine.roll_spread = mocker.AsyncMock(side_effect=NoValidContractsError("none"))
    await engine.roll_spreads({"HOOD": [short, long]})
    engine.close_spread.assert_not_called()

    # with max_dte the fallback closes it
    rw.max_dte, rw.close_if_unable_to_roll = 30, True
    await engine.roll_spreads({"HOOD": [short, long]})
    engine.close_spread.assert_awaited_once()

    # any other error is contained
    engine.roll_spread = mocker.AsyncMock(side_effect=ZeroDivisionError("price 0"))
    await engine.roll_spreads({"HOOD": [short, long]})  # must not raise


@pytest.mark.asyncio
async def test_close_spread_builds_two_leg_combo(engine, mocker):
    short = _opt("HOOD", "20991120", 50.0, "P", -2, con_id=1)
    long = _opt("HOOD", "20991120", 45.0, "P", 2, con_id=2)
    _spread_mocks(engine, mocker, {1: 0.40, 2: 0.10}, [], [])
    await engine.close_spread("HOOD", short, long, "SMART")
    ((combo, order, _),) = engine.order_ops.orders.records()
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (1, "BUY"),
        (2, "SELL"),
    ]
    assert order.lmtPrice == pytest.approx(0.30)


# ------------------------------------------------------------ debit spreads


def test_find_debit_verticals():
    call_long = _opt("HOOD", "20261120", 65.0, "C", 1)
    call_short = _opt("HOOD", "20261120", 70.0, "C", -1)
    put_long = _opt("HOOD", "20261120", 55.0, "P", 2)
    put_short = _opt("HOOD", "20261120", 50.0, "P", -2)
    credit_long = _opt("HOOD", "20261120", 45.0, "P", 3)
    credit_short = _opt("HOOD", "20261120", 48.0, "P", -3)
    positions = [call_long, call_short, put_long, put_short, credit_long, credit_short]
    assert DiscretionaryEngine.find_debit_verticals(positions) == [
        (call_short, call_long),
        (put_short, put_long),
    ]


def test_debit_spread_short_leg_is_not_rolled_alone(engine, mocker):
    log = mocker.patch("thetagang.strategies.discretionary_spreads.log")
    short = _opt("HOOD", "20261120", 70.0, "C", -1)
    long = _opt("HOOD", "20261120", 65.0, "C", 1)
    assert engine.without_spread_legs([short], {"HOOD": [short, long]}) == []
    (message,) = log.notice.call_args.args
    assert "debit spread" in message and "calendar" not in message
    assert not engine.has_calendar_or_diagonal_legs([short, long])


@pytest.mark.asyncio
async def test_debit_spread_closed_as_one_combo_at_take_profit(engine, mocker):
    """Worth 4.20 of a 5-wide spread (>= 80%): sell long, buy short together."""
    long = _opt("HOOD", "20991120", 50.0, "C", 2, con_id=1)
    short = _opt("HOOD", "20991120", 55.0, "C", -2, con_id=2)
    _spread_mocks(engine, mocker, {1: 4.50, 2: 0.30}, [], [])
    await engine.manage_debit_spreads({"HOOD": [long, short]})
    ((combo, order, _),) = engine.order_ops.orders.records()
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (2, "BUY"),  # buy back the short
        (1, "SELL"),  # sell the long
    ]
    assert (order.action, order.totalQuantity) == ("BUY", 2)
    assert order.lmtPrice == pytest.approx(-4.20)  # credit


def _near_expiry():
    return (date.today() + timedelta(days=2)).strftime("%Y%m%d")  # noqa: DTZ011


@pytest.mark.asyncio
async def test_debit_spread_rolled_out_as_one_four_leg_combo(engine, mocker):
    long = _opt("HOOD", _near_expiry(), 50.0, "C", 1, con_id=1)
    short = _opt("HOOD", _near_expiry(), 55.0, "C", -1, con_id=2)
    new_long = Option("HOOD", "20991218", 52.0, "C", "SMART", conId=3)
    new_short = Option("HOOD", "20991218", 57.0, "C", "SMART", conId=4)
    _spread_mocks(
        engine, mocker, {1: 2.5, 2: 0.5, 3: 2.8, 4: 0.9}, [new_long], [new_short]
    )
    await engine.manage_debit_spreads({"HOOD": [long, short]})
    ((combo, order, _),) = engine.order_ops.orders.records()
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (1, "SELL"),
        (2, "BUY"),
        (3, "BUY"),
        (4, "SELL"),
    ]
    asked = engine.ibkr.qualify_contracts.call_args.args[0]
    assert asked.strike == 57.0  # same 5-point width
    assert order.lmtPrice == pytest.approx(-0.10)  # new 1.90 - old 2.00


@pytest.mark.asyncio
async def test_debit_spread_roll_skipped_above_max_debit(engine, mocker):
    """New spread 4.80 - old 2.00 = 2.80 debit > 0.5 x 5 width: not rolled."""
    long = _opt("HOOD", _near_expiry(), 50.0, "C", 1, con_id=1)
    short = _opt("HOOD", _near_expiry(), 55.0, "C", -1, con_id=2)
    new_long = Option("HOOD", "20991218", 52.0, "C", "SMART", conId=3)
    new_short = Option("HOOD", "20991218", 57.0, "C", "SMART", conId=4)
    _spread_mocks(
        engine, mocker, {1: 2.5, 2: 0.5, 3: 5.0, 4: 0.2}, [new_long], [new_short]
    )
    await engine.manage_debit_spreads({"HOOD": [long, short]})
    assert engine.order_ops.orders.records() == []


@pytest.mark.asyncio
async def test_credit_spread_short_is_never_also_a_debit_leg(engine, mocker):
    """-P100/+P95 is a credit vertical; the extra +P105 must not pair with -P100
    as a debit spread (that would buy the short back twice)."""
    short = _opt("HOOD", "20991120", 100.0, "P", -1, con_id=1)
    credit_long = _opt("HOOD", "20991120", 95.0, "P", 1, con_id=2)
    extra_long = _opt("HOOD", "20991120", 105.0, "P", 1, con_id=3)
    _spread_mocks(engine, mocker, {1: 1.0, 2: 0.5, 3: 9.0}, [], [])
    pairs = engine.split_option_legs([short, credit_long, extra_long])[0]
    assert pairs == [(short, credit_long)]
    await engine.manage_debit_spreads({"HOOD": [short, credit_long, extra_long]})
    assert engine.order_ops.orders.records() == []


@pytest.mark.asyncio
async def test_covered_call_is_not_a_debit_spread(engine, mocker):
    """100 shares cover the -C110, so +C100 isn't its debit-spread partner."""
    shares = _stock("HOOD", 100)
    covered = _opt("HOOD", "20991120", 110.0, "C", -1, con_id=1)
    long_call = _opt("HOOD", "20991120", 100.0, "C", 1, con_id=2)
    _spread_mocks(engine, mocker, {1: 0.1, 2: 9.9}, [], [])
    positions = [shares, covered, long_call]
    assert engine.split_option_legs(positions)[0] == []
    assert engine.without_spread_legs([covered], {"HOOD": positions}) == [covered]
    await engine.manage_debit_spreads({"HOOD": positions})
    assert engine.order_ops.orders.records() == []


@pytest.mark.asyncio
async def test_unrollable_debit_spread_closed_when_allowed(engine, mocker):
    """Roll too expensive, profitable, within max_dte, close_if_unable_to_roll:
    closed as one combo instead of being left to expire."""
    rw = engine.config.strategies.wheel.defaults.roll_when
    rw.close_if_unable_to_roll, rw.max_dte = True, 30
    long = _opt("HOOD", _near_expiry(), 50.0, "C", 1, avg=200.0, upnl=50.0, con_id=1)
    short = _opt("HOOD", _near_expiry(), 55.0, "C", -1, avg=50.0, con_id=2)
    new_long = Option("HOOD", "20991218", 52.0, "C", "SMART", conId=3)
    new_short = Option("HOOD", "20991218", 57.0, "C", "SMART", conId=4)
    _spread_mocks(
        engine, mocker, {1: 2.5, 2: 0.5, 3: 5.0, 4: 0.2}, [new_long], [new_short]
    )
    await engine.manage_debit_spreads({"HOOD": [long, short]})
    ((combo, _, _),) = engine.order_ops.orders.records()
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (2, "BUY"),  # buy back the short
        (1, "SELL"),  # sell the long
    ]


def test_resize_never_adds_short_options():
    from thetagang.risk import Adjuster
    from thetagang.strategies.discretionary_trades import Resize, adds_short_options

    short_put = _opt("HOOD", "20991120", 50.0, "P", -1)
    single = Adjuster(10, "HOOD P50", False, -10.0, 0, 0, 0, 1, 1, (short_put,))
    assert adds_short_options(Resize("HOOD", -2, single))  # sell 2 more puts
    assert not adds_short_options(Resize("HOOD", 1, single))  # buy one back
    credit = (
        _opt("HOOD", "20991120", 50.0, "P", -1),
        _opt("HOOD", "20991120", 45.0, "P", 1),
    )
    debit = (
        _opt("HOOD", "20991120", 50.0, "P", -1),
        _opt("HOOD", "20991120", 55.0, "P", 1),
    )
    spread = Adjuster(10, "s", True, 5.0, 0, 0, 0, 1, -1, credit)
    assert adds_short_options(Resize("HOOD", 1, spread))  # open more credit
    assert not adds_short_options(Resize("HOOD", -1, spread))  # close one
    paid = Adjuster(10, "d", True, -5.0, 0, 0, 0, 1, -1, debit)
    assert not adds_short_options(Resize("HOOD", 1, paid))  # debit: risk paid


def test_target_share_sale_keeps_covered_calls_covered(engine):
    """100 shares under a far-OTM short call: never suggested to sell them."""
    shares = _stock("HOOD", 100)
    call = _opt("HOOD", "20991120", 200.0, "C", -1, con_id=1)
    traded, _, _, resize = engine._symbol_adjustment(
        [shares, call], -100.0, {1: _greek(0.001)}, 1.0, 5000.0
    )
    assert (traded, resize) == (0, None)


# ------------------------------------------------------------ VIX hedges


@pytest.mark.asyncio
async def test_close_vix_hedges(engine, mocker):
    long_call = _opt("VIX", "20991118", 25.0, "C", 10, con_id=1)
    short_call = _opt("VIX", "20991118", 40.0, "C", -5, con_id=2)
    engine.ibkr.portfolio = mocker.Mock(return_value=[long_call, short_call])
    prices = {"vix": 28.0, "option": 4.123}

    async def tickers(contract, **kwargs):
        if contract.secType == "IND":
            return _ticker(mocker, prices["vix"])
        return _ticker(mocker, prices["option"], long_call.contract)

    engine.ibkr.get_ticker_for_contract = tickers
    engine.config.strategies.vix_call_hedge.close_hedges_when_vix_exceeds = 30.0

    await engine.close_vix_hedges()  # VIX below the threshold
    assert engine.order_ops.orders.records() == []

    prices["vix"] = 35.0
    await engine.close_vix_hedges()
    ((_, order, _),) = (
        engine.order_ops.orders.records()
    )  # the short call is never touched
    assert (order.action, order.totalQuantity, order.lmtPrice) == ("SELL", 10, 4.1)

    engine.order_ops.orders.remove_records(list(engine.order_ops.orders.records()))
    prices["option"] = float("nan")  # no quote: never sell at 0
    await engine.close_vix_hedges()
    assert engine.order_ops.orders.records() == []


def test_vix_hedges_table_shows_unrealized_pnl():
    from thetagang.strategies.discretionary_engine import vix_hedges_table

    hedge = _opt("VIX", "20991118", 25.0, "C", 10, avg=150.0, upnl=420.0)
    table = vix_hedges_table([hedge], 28.0, 30.0, False, False)
    row = {c.header: Text.from_markup(str(c._cells[0])).plain for c in table.columns}
    assert row["Unrealized P&L"] == "$420"


@pytest.mark.asyncio
async def test_vix_closes_not_handed_to_a_disabled_stage(engine, mocker):
    """run.stages can list post_vix_call_hedge disabled: then this stage sells."""
    engine.config.strategies.vix_call_hedge.enabled = True
    engine.config.strategies.vix_call_hedge.close_hedges_when_vix_exceeds = 30.0
    mocker.patch.object(
        type(engine.config.run),
        "resolved_stages",
        lambda self: [SimpleNamespace(id="post_vix_call_hedge", enabled=False)],
    )
    long_call = _opt("VIX", "20991118", 25.0, "C", 10, con_id=1)
    engine.ibkr.portfolio = mocker.Mock(return_value=[long_call])
    engine.ibkr.get_ticker_for_contract = mocker.AsyncMock(
        return_value=_ticker(mocker, 35.0, long_call.contract)
    )
    await engine.close_vix_hedges()
    ((_, order, _),) = engine.order_ops.orders.records()
    assert (order.action, order.totalQuantity) == ("SELL", 10)
    assert order.orderRef == "thetagang-discretionary-manage"


@pytest.mark.asyncio
async def test_vix_closes_left_to_enabled_vix_call_hedge_stage(engine, mocker):
    """With the vix_call_hedge stage enabled and scheduled, it closes the hedges;
    the discretionary stage must not queue a second sell."""
    engine.config = Config(
        **_config_dict(run={"strategies": ["discretionary", "vix_call_hedge"]})
    )
    engine.config.strategies.vix_call_hedge.enabled = True
    engine.config.strategies.vix_call_hedge.close_hedges_when_vix_exceeds = 30.0
    long_call = _opt("VIX", "20991118", 25.0, "C", 10, con_id=1)
    engine.ibkr.portfolio = mocker.Mock(return_value=[long_call])
    engine.ibkr.get_ticker_for_contract = mocker.AsyncMock(
        return_value=_ticker(mocker, 35.0, long_call.contract)
    )
    await engine.close_vix_hedges()
    assert engine.order_ops.orders.records() == []
    engine.ibkr.get_ticker_for_contract.assert_called_once()  # VIX price for the table


# ------------------------------------------------------------ risk tables


def _greek(delta, und=50.0):
    return SimpleNamespace(
        delta=delta, gamma=0.01, theta=-0.02, vega=0.05, undPrice=und, impliedVol=None
    )


def _book(engine, mocker, extra_tickers=None):
    """Beta 1 everywhere: AAPL 100 sh @ 200 + a HOOD short put (delta -0.3, und 50)
    = 21,500 beta $ = SPX delta +4.30 at SPX 5000. VIX calls never count."""
    put = _opt("HOOD", "20991120", 50.0, "P", -1, con_id=7)
    stock = _stock("AAPL", 100, price=200.0)
    vix_call = _opt("VIX", "20991118", 20.0, "C", 10, con_id=9)
    engine.ibkr.portfolio = mocker.Mock(return_value=[stock, put, vix_call])
    bars = [mocker.Mock(date=d, close=100 + d % 5) for d in range(60)]
    engine.ibkr.request_historical_data = mocker.AsyncMock(return_value=bars)
    extra = extra_tickers or {}

    async def tickers(contract, **kwargs):
        if contract.conId in extra:
            return extra[contract.conId]
        t = mocker.Mock(modelGreeks=_greek(-0.3))
        t.modelGreeks.gamma, t.modelGreeks.theta, t.modelGreeks.vega = 0.02, -0.05, 0.1
        t.marketPrice.return_value = 5000.0
        return t

    engine.ibkr.get_ticker_for_contract = tickers


def _option_ticker(mocker, contract, delta, und, mid):
    t = _ticker(mocker, mid, contract)
    t.modelGreeks = _greek(delta, und)
    return t


def _trades(engine, *trades, band=1.0):
    from thetagang.config import DiscretionaryStrategyConfig

    engine.config.strategies.discretionary = DiscretionaryStrategyConfig(
        delta_band=band, trades=list(trades)
    )


def _rows(table):
    cells = [
        [Text.from_markup(str(c)).plain for c in col._cells] for col in table.columns
    ]
    return list(zip(*cells))


@pytest.mark.asyncio
async def test_no_trades_configured_keeps_book(engine, mocker):
    _book(engine, mocker)
    await engine.portfolio_greeks_table()
    assert engine.planned_orders == []


@pytest.mark.asyncio
async def test_long_put_sized_to_zero_spx_delta(engine, mocker):
    """SPY put delta -0.5 at 500: -25,000 beta $ = -5 SPX per contract, so one
    contract takes +4.30 to -0.70; a long call (wrong way) gets nothing."""
    spy_put = Option("SPY", "20991120", 500.0, "P", "SMART", "100", "USD", conId=20)
    spy_put.localSymbol = "SPY P500"
    spy_call = Option("SPY", "20991120", 520.0, "C", "SMART", "100", "USD", conId=21)
    _book(engine, mocker)
    found = {
        "P": _option_ticker(mocker, spy_put, -0.5, 500.0, 6.0),
        "C": _option_ticker(mocker, spy_call, 0.3, 500.0, 3.0),
    }
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        side_effect=lambda und, right, *a, **k: found[right]
    )
    _trades(
        engine,
        {"symbol": "SPY", "structure": "long_put", "delta": 0.5},
        {"symbol": "SPY", "structure": "long_call"},
    )
    greeks_tbl, _, _, plan = await engine.portfolio_greeks_table()
    rows = {r[0]: r[1:] for r in _rows(greeks_tbl)}
    assert rows["SPX delta"] == ("4.30", "-0.70")
    plan_rows = _rows(plan)
    assert plan_rows[0][4] == "1"
    assert (plan_rows[1][4], plan_rows[1][5]) == (
        "0",
        "would move SPX delta the wrong way",
    )
    ((contract, order),) = engine.planned_orders
    assert contract is spy_put
    assert (order.action, order.totalQuantity, order.lmtPrice) == ("BUY", 1, 6.0)
    assert order.orderRef == "thetagang-discretionary-entry"

    await engine.run_post_stage({})  # dry run or not, the stage enqueues them
    assert [r[1] for r in engine.order_ops.orders.records()] == [order]
    assert engine.planned_orders == []


@pytest.mark.asyncio
async def test_call_credit_spread_combo_and_cap(engine, mocker):
    """Short 50C / long 55C: delta (-0.4 + 0.2) x 100 x 50 = -1,000 beta $ =
    -0.2 SPX per spread; 4.30 needs 22, capped at 5. One BUY combo at a credit."""
    short = Option("HOOD", "20991120", 50.0, "C", "SMART", "100", "USD", conId=30)
    long = Option("HOOD", "20991120", 55.0, "C", "SMART", "100", "USD", conId=31)
    _book(engine, mocker, {31: _option_ticker(mocker, long, 0.2, 50.0, 1.0)})
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        return_value=_option_ticker(mocker, short, 0.4, 50.0, 2.5)
    )
    engine.ibkr.qualify_contracts = mocker.AsyncMock(return_value=[long])
    _trades(
        engine,
        {
            "symbol": "HOOD",
            "structure": "call_credit_spread",
            "delta": 0.4,
            "width": 5,
            "max_contracts": 5,
        },
    )
    _, _, _, plan = await engine.portfolio_greeks_table()
    asked = engine.ibkr.qualify_contracts.call_args.args[0]
    assert (asked.strike, asked.right, asked.lastTradeDateOrContractMonth) == (
        55.0,
        "C",
        "20991120",
    )
    row = _rows(plan)[0]
    assert (row[2], row[4], row[5]) == ("-0.200", "5", "capped at max_contracts=5")
    ((combo, order),) = engine.planned_orders
    assert combo.secType == "BAG"
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (30, "SELL"),
        (31, "BUY"),
    ]
    assert (order.action, order.totalQuantity, order.lmtPrice) == ("BUY", 5, -1.5)
    assert order.orderRef == "thetagang-discretionary-entry"


@pytest.mark.asyncio
async def test_no_new_trades_inside_band(engine, mocker):
    spy_put = Option("SPY", "20991120", 500.0, "P", "SMART", "100", "USD", conId=20)
    _book(engine, mocker)
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        return_value=_option_ticker(mocker, spy_put, -0.5, 500.0, 6.0)
    )
    _trades(engine, {"symbol": "SPY", "structure": "long_put"}, band=5.0)
    _, _, _, plan = await engine.portfolio_greeks_table()
    assert _rows(plan)[0][4] == "0"
    assert "within the band" in _rows(plan)[0][5]
    assert engine.planned_orders == []


@pytest.mark.asyncio
async def test_no_new_trades_when_greeks_are_missing(engine, mocker):
    """An option without greeks leaves SPX delta incomplete: plan shown, no orders."""
    spy_put = Option("SPY", "20991120", 500.0, "P", "SMART", "100", "USD", conId=20)
    _book(engine, mocker)
    real = engine.ibkr.get_ticker_for_contract

    async def tickers(contract, **kwargs):
        t = await real(contract, **kwargs)
        if contract.conId == 7:  # the HOOD short put
            t.modelGreeks = None
        return t

    engine.ibkr.get_ticker_for_contract = tickers
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        return_value=_option_ticker(mocker, spy_put, -0.5, 500.0, 6.0)
    )
    _trades(engine, {"symbol": "SPY", "structure": "long_put"})
    _, _, _, plan = await engine.portfolio_greeks_table()
    row = _rows(plan)[0]
    assert (row[4], row[5]) == ("0", "1 option(s) without greeks: no new trades")
    assert engine.planned_orders == []


def _spy_put_plan(engine, mocker):
    spy_put = Option("SPY", "20991120", 500.0, "P", "SMART", "100", "USD", conId=20)
    _book(engine, mocker)
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        return_value=_option_ticker(mocker, spy_put, -0.5, 500.0, 6.0)
    )
    _trades(engine, {"symbol": "SPY", "structure": "long_put"})


@pytest.mark.asyncio
async def test_new_trades_wait_when_rolls_or_closes_are_queued(engine, mocker):
    """Trades sized before this run's rolls/closes would over-hedge: skip them."""
    _spy_put_plan(engine, mocker)
    await engine.portfolio_greeks_table()
    assert len(engine.planned_orders) == 1
    roll = engine.order_ops.create_limit_order(
        action="BUY", quantity=1, limit_price=1.0
    )
    engine.order_ops.enqueue_order(Option("HOOD", "20991120", 50.0, "P"), roll)
    await engine.run_post_stage({})
    assert [r[1] for r in engine.order_ops.orders.records()] == [roll]
    assert engine.planned_orders == []


@pytest.mark.asyncio
async def test_new_entries_require_open_interest(engine, mocker):
    _spy_put_plan(engine, mocker)
    await engine.portfolio_greeks_table()
    kwargs = engine.option_scanner.find_eligible_contracts.call_args.kwargs
    assert kwargs["allow_missing_open_interest"] is False


@pytest.mark.asyncio
async def test_no_trading_override_applies_to_symbols_not_held(engine, mocker):
    from thetagang.config_models import SymbolConfig

    _spy_put_plan(engine, mocker)
    engine.symbol_overrides = {"SPY": SymbolConfig(weight=0.0, no_trading=True)}
    _, _, _, plan = await engine.portfolio_greeks_table()
    assert _rows(plan)[0][5] == "no_trading is set"
    assert engine.planned_orders == []


@pytest.mark.asyncio
async def test_new_trades_ignore_max_dte(engine, mocker):
    """max_dte is a roll rule: new trades search with no expiry cap."""
    from thetagang.config_models import SymbolConfig

    _spy_put_plan(engine, mocker)
    engine.symbol_overrides = {"SPY": SymbolConfig(weight=0.0, max_dte=21)}
    await engine.portfolio_greeks_table()
    kwargs = engine.option_scanner.find_eligible_contracts.call_args.kwargs
    assert kwargs["max_dte"] == 0


@pytest.mark.asyncio
async def test_failed_spx_history_is_retried_next_run(engine, mocker):
    _book(engine, mocker)
    engine.ibkr.request_historical_data = mocker.AsyncMock(
        side_effect=RuntimeError("timeout")
    )
    with pytest.raises(RuntimeError):
        await engine.fetch_risk_inputs()
    assert engine.beta_cache_day is None  # not marked loaded for today


@pytest.mark.asyncio
async def test_trade_without_contract_is_skipped(engine, mocker):
    _book(engine, mocker)
    engine.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        side_effect=NoValidContractsError("none")
    )
    _trades(engine, {"symbol": "SPY", "structure": "long_put"})
    _, _, _, plan = await engine.portfolio_greeks_table()
    assert _rows(plan)[0][5] == "skipped: none"
    assert engine.planned_orders == []


def test_option_adjusters_respect_spreads(engine):
    """Verticals resize as spreads, calendar legs are skipped, singles stay single."""
    spread_short = _opt("HOOD", "20261120", 50.0, "P", -2, con_id=1)
    spread_long = _opt("HOOD", "20261120", 45.0, "P", 2, con_id=2)
    lone_put = _opt("HOOD", "20261120", 40.0, "P", -1, con_id=3)
    cal_short = _opt("HOOD", "20261120", 60.0, "C", -1, con_id=4)
    cal_long = _opt("HOOD", "20261218", 60.0, "C", 1, con_id=5)
    greeks = {
        1: _greek(-0.30),
        2: _greek(-0.15),
        3: _greek(-0.10),
        4: _greek(0.40),
        5: _greek(0.45),
    }

    adjusters = engine.option_adjusters(
        [spread_short, spread_long, lone_put, cal_short, cal_long], greeks, 1.0, 5000.0
    )
    by_kind = {a.is_spread: a for a in adjusters}
    assert len(adjusters) == 2  # calendar call legs are left alone

    spread = by_kind[True]
    assert spread.name == "HOOD 20261120 50/45P"
    assert spread.delta == pytest.approx(15.0)  # +30 short put, -15 long put
    assert spread.held == 2
    lone = by_kind[False]
    assert lone.delta == pytest.approx(-10.0) and lone.close_sign == 1


def test_option_adjusters_skip_spread_with_missing_greeks(engine):
    short = _opt("HOOD", "20261120", 100.0, "P", -1, con_id=1)
    long = _opt("HOOD", "20261120", 95.0, "P", 1, con_id=2)
    greeks = {1: _greek(-0.4), 2: None}
    assert engine.option_adjusters([short, long], greeks, 1.0, 5000.0) == []


@pytest.mark.asyncio
async def test_zero_delta_proposals_without_trades(engine, mocker):
    """No trades configured: "After trades" shows the SPX delta = 0 proposals;
    AAPL sells stock, HOOD buys back its short put, VIX gets no row."""
    _book(engine, mocker)
    greeks_tbl, _, _ = await engine.portfolio_greeks_table()
    rows = {r[0]: r[1:] for r in _rows(greeks_tbl)}
    assert rows["SPX delta"] == ("4.30", "0.00")
    assert rows["Theta ($/day)"][1] == "$0"  # the put is bought back
    plain = {
        s: tuple(Text.from_markup(c).plain for c in cells)
        for s, cells in engine.zero_delta.items()
    }
    assert plain["AAPL"][3:] == ("-100", "-100 shares")
    assert plain["HOOD"][3] == "-30"
    assert "VIX" not in plain
    assert engine.planned_orders == []  # proposals never place orders
    # short put, delta -0.3: +30 shares; gamma 0.02, theta -0.05, vega 0.1 x -100
    assert engine.position_greeks[7] == pytest.approx((30.0, -2.0, 5.0, -10.0))
    assert engine.symbol_greeks["HOOD"] == pytest.approx((30.0, -2.0, 5.0, -10.0))
    from thetagang.config_models import SymbolConfig

    engine.config.portfolio.symbols = {"HOOD": SymbolConfig(weight=0.0)}
    assert _rows(engine.symbols_table())[0][:5] == (
        "HOOD",
        "+30.0",
        "-2.00",
        "$5",
        "$-10",
    )
    assert engine.contract_greeks[7] == pytest.approx((-0.3, 0.02, -0.05, 0.1))


def _place_targets(engine, mocker, price=2.0):
    from thetagang.config import DiscretionaryStrategyConfig
    from thetagang.config_models import SymbolConfig

    engine.config.strategies.discretionary = DiscretionaryStrategyConfig(
        place_target_orders=True
    )
    engine.config.portfolio.symbols = {  # managed = held stock/option symbols
        s: SymbolConfig(weight=0.0) for s in ("AAPL", "HOOD")
    }
    engine.leg_price = mocker.AsyncMock(return_value=price)


@pytest.mark.asyncio
async def test_place_target_orders_turns_net_targets_into_orders(engine, mocker):
    """AAPL sells 100 shares, HOOD buys back its short put, both at mid."""
    _book(engine, mocker)
    _place_targets(engine, mocker)
    await engine.portfolio_greeks_table()
    orders = {c.symbol: o for c, o in engine.planned_orders}
    assert (orders["AAPL"].action, orders["AAPL"].totalQuantity) == ("SELL", 100)
    assert (orders["HOOD"].action, orders["HOOD"].totalQuantity) == ("BUY", 1)
    assert {o.orderRef for o in orders.values()} == {"thetagang-discretionary-resize"}
    assert orders["AAPL"].lmtPrice == 2.0


@pytest.mark.asyncio
async def test_target_orders_skip_no_trading_symbols(engine, mocker):
    from thetagang.config_models import SymbolConfig

    _book(engine, mocker)
    _place_targets(engine, mocker)
    engine.config.portfolio.symbols["HOOD"] = SymbolConfig(weight=0.0, no_trading=True)
    await engine.portfolio_greeks_table()
    assert [c.symbol for c, _ in engine.planned_orders] == ["AAPL"]
    assert engine.zero_delta["HOOD"][-1] == "none (no_trading)"


@pytest.mark.asyncio
async def test_target_orders_blocked_inside_band(engine, mocker):
    from thetagang.config import DiscretionaryStrategyConfig

    _book(engine, mocker)
    engine.leg_price = mocker.AsyncMock(return_value=2.0)
    engine.config.strategies.discretionary = DiscretionaryStrategyConfig(
        place_target_orders=True, delta_band=10.0
    )
    await engine.portfolio_greeks_table()
    assert engine.planned_orders == []  # SPX delta 4.30 is inside the band


@pytest.mark.asyncio
async def test_target_order_for_a_vertical_is_one_combo(engine, mocker):
    from thetagang.risk import Adjuster
    from thetagang.strategies.discretionary_trades import Resize

    short = _opt("HOOD", "20991120", 50.0, "P", -2, con_id=1)
    long = _opt("HOOD", "20991120", 45.0, "P", 2, con_id=2)
    spread = Adjuster(30, "HOOD 50/45P", True, 15.0, 0, 0, 0, 2, -1, (short, long))
    prices = {1: 2.5, 2: 1.0}
    engine.leg_price = mocker.AsyncMock(side_effect=lambda c, ex: prices[c.conId])
    combo, order = await engine._resize_order(Resize("HOOD", 2, spread), "SMART")
    assert [(leg.conId, leg.action) for leg in combo.comboLegs] == [
        (1, "SELL"),
        (2, "BUY"),
    ]  # open 2 more spreads for a credit
    assert (order.action, order.totalQuantity, order.lmtPrice) == ("BUY", 2, -1.5)
    combo, order = await engine._resize_order(Resize("HOOD", -1, spread), "SMART")
    assert [leg.action for leg in combo.comboLegs] == ["BUY", "SELL"]  # close 1
    assert order.lmtPrice == 1.5


@pytest.mark.asyncio
async def test_target_orders_skip_unmanaged_symbols(engine, mocker):
    """Index options (and anything discover_symbols leaves out) are never resized."""
    _book(engine, mocker)
    _place_targets(engine, mocker)
    del engine.config.portfolio.symbols["HOOD"]
    await engine.portfolio_greeks_table()
    assert [c.symbol for c, _ in engine.planned_orders] == ["AAPL"]


@pytest.mark.asyncio
async def test_debit_roll_uses_trade_target_and_skips_overlap(engine, mocker):
    """Rolls keep the configured debit spread's delta/DTE, and an overlapping
    first candidate is excluded and searched again."""
    from thetagang.config import DiscretionaryStrategyConfig

    engine.config.strategies.discretionary = DiscretionaryStrategyConfig(
        trades=[
            {
                "symbol": "HOOD",
                "structure": "call_debit_spread",
                "delta": 0.5,
                "dte": 60,
                "width": 5,
            }
        ]
    )
    long = _opt("HOOD", _near_expiry(), 50.0, "C", 1, con_id=1)
    short = _opt("HOOD", _near_expiry(), 55.0, "C", -1, con_id=2)
    overlap = Option("HOOD", _near_expiry(), 45.0, "C", "SMART", conId=5)
    new_long = Option("HOOD", "20991218", 52.0, "C", "SMART", conId=3)
    new_short = Option("HOOD", "20991218", 57.0, "C", "SMART", conId=4)
    old_long_again = Option("HOOD", _near_expiry(), 50.0, "C", "SMART", conId=1)
    _spread_mocks(
        engine,
        mocker,
        {1: 2.5, 2: 0.5, 3: 2.8, 4: 0.9, 5: 6.0},
        [overlap, new_long],
        [old_long_again, new_short],
    )
    await engine.manage_debit_spreads({"HOOD": [long, short]})
    calls = engine.option_scanner.find_eligible_contracts.call_args_list
    assert len(calls) == 2
    assert (calls[0].kwargs["target_delta"], calls[0].kwargs["target_dte"]) == (0.5, 60)
    assert (45.0, _near_expiry()) in calls[1].kwargs["also_exclude"]
    ((combo, _, _),) = engine.order_ops.orders.records()
    assert {leg.conId for leg in combo.comboLegs} == {1, 2, 3, 4}


@pytest.mark.asyncio
async def test_positions_caption_says_what_target_orders_did(engine, mocker):
    from thetagang.config import DiscretionaryStrategyConfig

    _book(engine, mocker)
    _place_targets(engine, mocker)
    await engine.portfolio_greeks_table()
    caption = str(engine.positions_summary_table({}).caption)
    assert "2 order(s) planned, 0 not placed" in caption
    engine.config.strategies.discretionary = DiscretionaryStrategyConfig(
        place_target_orders=True, delta_band=10.0
    )
    await engine.portfolio_greeks_table()
    caption = str(engine.positions_summary_table({}).caption)
    assert "no orders this run" in caption and "planned" not in caption


def test_target_orders_and_trades_are_exclusive():
    from pydantic import ValidationError

    from thetagang.config import DiscretionaryStrategyConfig

    with pytest.raises(ValidationError, match="not both"):
        DiscretionaryStrategyConfig(
            place_target_orders=True,
            trades=[{"symbol": "SPY", "structure": "long_put"}],
        )


def test_option_reprice_default_covers_options_only(config):
    """The discretionary default re-prices option orders of any symbol (VIX and
    symbols not held included); shares stay as placed; a block's own setting wins."""
    from ib_async import Contract

    from thetagang.config_models import SymbolConfig
    from thetagang.order_execution import OrderExecutionManager

    config.strategies.discretionary.adjust_price_after_delay = True
    config.portfolio.symbols = {
        "HOOD": SymbolConfig(weight=0.0, adjust_price_after_delay=False),
        "AAPL": SymbolConfig(weight=0.0),
    }
    manager = OrderExecutionManager.__new__(OrderExecutionManager)
    manager.config = config
    enabled = manager._legacy_adjust_enabled
    assert enabled(Option("VIX", "20991118", 25.0, "C"))
    assert enabled(Option("SPY", "20991120", 500.0, "P"))  # not held
    assert enabled(Contract(secType="BAG", symbol="AAPL"))  # spread combo
    assert not enabled(Stock("AAPL", "SMART", "USD"))  # shares
    assert not enabled(Option("HOOD", "20991120", 50.0, "P"))  # explicit false
    config.strategies.discretionary.adjust_price_after_delay = False
    assert not enabled(Option("VIX", "20991118", 25.0, "C"))


def test_discretionary_trade_config_validation():
    from pydantic import ValidationError

    from thetagang.config import DiscretionaryTradeConfig

    DiscretionaryTradeConfig(symbol="SPY", structure="put_credit_spread", width=5)
    with pytest.raises(ValidationError):  # spreads need a width
        DiscretionaryTradeConfig(symbol="SPY", structure="put_credit_spread")
    with pytest.raises(ValidationError):  # single legs take none
        DiscretionaryTradeConfig(symbol="SPY", structure="long_put", width=5)
    with pytest.raises(ValidationError):
        DiscretionaryTradeConfig.model_validate(
            {"symbol": "SPY", "structure": "iron_condor"}
        )
    with pytest.raises(ValidationError):
        DiscretionaryTradeConfig(symbol="SPY", structure="long_put", delta=1.5)


def test_positions_summary_table(engine):
    from thetagang.config_models import SymbolConfig

    engine.config.portfolio.symbols = {"HOOD": SymbolConfig(weight=0.0)}
    positions = {
        "HOOD": [
            _opt("HOOD", "20261120", 50.0, "P", -2),
            _opt("HOOD", "20261120", 45.0, "P", 2),
        ]
    }
    engine.config.portfolio.symbols["AMD"] = SymbolConfig(weight=0.0)
    engine.zero_delta = {
        "HOOD": ("60.0% long", "50.0% long", "$5,000", "-30", "+1 HOOD P50")
    }
    rows = _rows(engine.positions_summary_table(positions))
    assert rows[0][:6] == ("HOOD", "0", "2", "2", "0", "0")
    assert rows[0][6:] == ("60.0% long", "50.0% long", "$5,000", "-30", "+1 HOOD P50")
    assert rows[2][0] == "AMD" and rows[2][-1] == "n/a"  # no greeks/targets yet
    assert _rows(engine.symbols_table())[0][-1] == "50.0% long"


def test_compute_beta_reports_short_history():
    from thetagang.risk import compute_beta

    index = {d: 100 * np.exp(0.01 * ((d % 3) - 1) + 0.001 * d) for d in range(60)}
    assert compute_beta({0: 1.0}, index) is None


@pytest.mark.asyncio
async def test_standalone_roll_uses_manager_rules(config, engine, mocker):
    """In manager mode a standalone short put rolls the whole position (no
    buying-power cap) with the symmetric strike rule, and spread legs are skipped
    by the put check."""
    services = mocker.Mock()
    services.get_maximum_new_contracts_for = mocker.AsyncMock(return_value=1)
    options = OptionsStrategyEngine(
        config=config,
        ibkr=mocker.Mock(),
        option_scanner=mocker.Mock(),
        order_ops=engine.order_ops,
        services=services,
        target_quantities={},
        has_excess_puts=set(),
        has_excess_calls=set(),
        qualified_contracts={},
    )
    options.discretionary = engine
    options.put_is_itm = mocker.AsyncMock(return_value=False)
    position = _opt("HOOD", "20991120", 50.0, "P", -3, avg=200.0, con_id=1)
    new_put = Option("HOOD", "20991218", 51.0, "P", "SMART", conId=2)

    async def tickers(contract, **kwargs):
        return _ticker(mocker, 0.50, contract)

    mocker.patch.object(options.ibkr, "get_ticker_for_contract", side_effect=tickers)
    options.option_scanner.find_eligible_contracts = mocker.AsyncMock(
        return_value=_ticker(mocker, 1.20, new_put)
    )

    await options.roll_positions([position], "P", {})

    # premium $2.00/share, buyback $0.50: 1.50 captured -> up to the 51.50 strike
    assert options.option_scanner.find_eligible_contracts.call_args.args[2] == 51.5
    services.get_maximum_new_contracts_for.assert_not_called()
    ((_, order, _),) = engine.order_ops.orders.records()
    assert order.totalQuantity == 3  # the whole position

    # the put check skips a vertical's short leg
    spread_long = _opt("HOOD", "20991120", 45.0, "P", 3, avg=50.0)
    rollable, closeable, _ = await options.check_puts({"HOOD": [position, spread_long]})
    assert rollable == [] and closeable == []
