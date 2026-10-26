from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

from thetagang.monitor_config import load_monitor_config
from thetagang.monitor_metrics import (
    AccountSnapshot,
    MonitorSnapshot,
    PositionSnapshot,
    build_metric_families,
)


def _samples(families):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in families
        for sample in family.samples
    }


def test_metric_families_preserve_dashboard_contract_and_signed_risk() -> None:
    config = replace(
        load_monitor_config("thetagang.nt.toml"),
        long_option_alert_delta=0.8,
        stress_moves=(0.1,),
    )
    snapshot = MonitorSnapshot(
        accounts=(AccountSnapshot("DU123456", 10_000.0, 8_000.0),),
        positions=(
            PositionSnapshot(
                instrument="AAA",
                symbol="AAA",
                kind="stock",
                quantity=10.0,
                average_price=40.0,
                multiplier=1.0,
                market_price=50.0,
                expiry=None,
                strike=None,
                delta=None,
                gamma=None,
                theta=None,
                vega=None,
                underlying_price=None,
                implied_volatility=None,
                beta=2.0,
            ),
            PositionSnapshot(
                instrument="AAA-CALL-SHORT",
                symbol="AAA",
                kind="call",
                quantity=-1.0,
                average_price=5.0,
                multiplier=100.0,
                market_price=3.0,
                expiry=date(2026, 11, 25),
                strike=60.0,
                delta=0.5,
                gamma=0.01,
                theta=-0.02,
                vega=0.1,
                underlying_price=50.0,
                implied_volatility=0.2,
                beta=2.0,
            ),
            PositionSnapshot(
                instrument="AAA-CALL-LONG",
                symbol="AAA",
                kind="call",
                quantity=1.0,
                average_price=2.0,
                multiplier=100.0,
                market_price=4.0,
                expiry=date(2026, 11, 25),
                strike=55.0,
                delta=0.8,
                gamma=0.02,
                theta=-0.03,
                vega=0.15,
                underlying_price=50.0,
                implied_volatility=0.25,
                beta=2.0,
            ),
            PositionSnapshot(
                instrument="AAA-PUT-MISSING-GREEKS",
                symbol="AAA",
                kind="put",
                quantity=-1.0,
                average_price=2.0,
                multiplier=100.0,
                market_price=None,
                expiry=date(2026, 11, 25),
                strike=45.0,
                delta=None,
                gamma=None,
                theta=None,
                vega=None,
                underlying_price=50.0,
                implied_volatility=None,
                beta=2.0,
            ),
        ),
        index_prices={"spx": 100.0, "vix": 20.0, "vixmo": 100.0},
        open_orders=2,
        timestamp_seconds=datetime(2026, 10, 26, tzinfo=UTC).timestamp(),
    )

    families = build_metric_families(snapshot, config)
    samples = _samples(families)

    assert {family.name for family in families} == {
        "thetagang_spx_price",
        "thetagang_vix_price",
        "thetagang_vixmo_price",
        "thetagang_net_liquidation",
        "thetagang_available_funds",
        "thetagang_margin_used",
        "thetagang_position_quantity",
        "thetagang_position_avg_price",
        "thetagang_position_price",
        "thetagang_position_unrealized_pnl",
        "thetagang_position_market_value",
        "thetagang_position_dte",
        "thetagang_position_pnl_ratio",
        "thetagang_unrealized_pnl",
        "thetagang_spx_delta",
        "thetagang_beta_delta_dollars",
        "thetagang_delta_dollars",
        "thetagang_gamma_spx",
        "thetagang_theta",
        "thetagang_vega",
        "thetagang_options_missing_greeks",
        "thetagang_symbol_beta",
        "thetagang_symbol_delta_dollars",
        "thetagang_symbol_beta_delta_dollars",
        "thetagang_symbol_delta_shares",
        "thetagang_symbol_gamma",
        "thetagang_symbol_theta",
        "thetagang_symbol_vega",
        "thetagang_position_delta",
        "thetagang_position_iv",
        "thetagang_long_option_alerts",
        "thetagang_stress_pnl",
        "thetagang_orders_open",
        "thetagang_last_update_timestamp_seconds",
    }
    assert samples[("thetagang_net_liquidation", (("account", "DU123456"),))] == 10_000
    assert samples[("thetagang_available_funds", (("account", "DU123456"),))] == 8_000
    assert samples[("thetagang_margin_used", (("account", "DU123456"),))] == 2_000
    assert (
        samples[
            (
                "thetagang_position_unrealized_pnl",
                (("instrument", "AAA"), ("kind", "stock"), ("symbol", "AAA")),
            )
        ]
        == 100
    )
    assert (
        samples[
            (
                "thetagang_position_unrealized_pnl",
                (("instrument", "AAA-CALL-SHORT"), ("kind", "call"), ("symbol", "AAA")),
            )
        ]
        == 200
    )
    assert samples[("thetagang_unrealized_pnl", ())] == 500
    assert (
        samples[
            (
                "thetagang_position_dte",
                (("instrument", "AAA-CALL-SHORT"), ("kind", "call"), ("symbol", "AAA")),
            )
        ]
        == 30
    )
    assert (
        samples[
            (
                "thetagang_position_pnl_ratio",
                (("instrument", "AAA-CALL-SHORT"), ("kind", "call"), ("symbol", "AAA")),
            )
        ]
        == 0.4
    )
    assert samples[("thetagang_spx_delta", ())] == 40
    assert samples[("thetagang_beta_delta_dollars", ())] == 4_000
    assert samples[("thetagang_delta_dollars", ())] == 2_000
    assert samples[("thetagang_gamma_spx", ())] == 1
    assert samples[("thetagang_theta", ())] == -1
    assert samples[("thetagang_vega", ())] == 5
    assert samples[("thetagang_options_missing_greeks", ())] == 1
    assert samples[("thetagang_symbol_delta_shares", (("symbol", "AAA"),))] == 40
    assert (
        samples[("thetagang_position_delta", (("instrument", "AAA-CALL-LONG"),))] == 0.8
    )
    assert samples[("thetagang_long_option_alerts", ())] == 1
    assert samples[("thetagang_orders_open", ())] == 2
    assert (
        samples[("thetagang_last_update_timestamp_seconds", ())]
        == snapshot.timestamp_seconds
    )


def test_stress_pnl_shocks_stock_with_beta_and_one_bases() -> None:
    config = replace(load_monitor_config("thetagang.nt.toml"), stress_moves=(0.1,))
    stock = PositionSnapshot(
        instrument="AAA",
        symbol="AAA",
        kind="stock",
        quantity=10.0,
        average_price=40.0,
        multiplier=1.0,
        market_price=50.0,
        expiry=None,
        strike=None,
        delta=None,
        gamma=None,
        theta=None,
        vega=None,
        underlying_price=None,
        implied_volatility=None,
        beta=2.0,
    )
    snapshot = MonitorSnapshot(
        accounts=(),
        positions=(stock,),
        index_prices={"spx": 100.0},
        open_orders=0,
        timestamp_seconds=datetime(2026, 10, 26, tzinfo=UTC).timestamp(),
    )

    samples = _samples(build_metric_families(snapshot, config))

    assert (
        samples[("thetagang_stress_pnl", (("betas", "beta"), ("move", "+10%")))] == 100
    )
    assert samples[("thetagang_stress_pnl", (("betas", "one"), ("move", "+10%")))] == 50
