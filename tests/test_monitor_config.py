from __future__ import annotations

import sys

import pytest

from thetagang.monitor_config import load_monitor_config


def test_loads_sample_monitor_settings_without_nautilus() -> None:
    config = load_monitor_config("thetagang.nt.toml")

    assert config.host == "127.0.0.1"
    assert config.port == 7497
    assert config.client_id == 11
    assert config.account_id is None
    assert config.market_data_type == "REALTIME"
    assert config.interval_secs == 60
    assert config.metrics_addr == "0.0.0.0"
    assert config.metrics_port == 9464
    assert config.long_option_alert_delta == 0.8
    assert config.stress_moves == (-0.1, -0.05, 0.05, 0.1)
    assert "nautilus_trader" not in sys.modules


def test_monitor_config_uses_tws_account_environment_fallback(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "monitor.toml"
    config_path.write_text("[ib]\naccount_id = ''\n", encoding="utf-8")
    monkeypatch.setenv("TWS_ACCOUNT", "DU123456")

    config = load_monitor_config(config_path)

    assert config.account_id == "DU123456"


def test_monitor_config_rejects_wrong_field_types(tmp_path) -> None:
    config_path = tmp_path / "monitor.toml"
    config_path.write_text('[ib]\nport = "7497"\n', encoding="utf-8")

    with pytest.raises(TypeError, match="port must be an integer"):
        load_monitor_config(config_path)


@pytest.mark.parametrize(
    "table, invalid_value",
    [
        ("ib", "port = 0"),
        ("ib", "port = 65536"),
        ("monitor", "metrics_port = 0"),
        ("monitor", "interval_secs = 0"),
        ("monitor", "long_option_alert_delta = 1.1"),
        ("monitor", "stress_moves = [-1.0]"),
        ("monitor", "stress_moves = [nan]"),
    ],
)
def test_monitor_config_rejects_invalid_values(
    tmp_path, monkeypatch, table: str, invalid_value: str
) -> None:
    config_path = tmp_path / "monitor.toml"
    config_path.write_text(f"[{table}]\n{invalid_value}\n", encoding="utf-8")
    monkeypatch.delenv("TWS_ACCOUNT", raising=False)

    with pytest.raises(ValueError):
        load_monitor_config(config_path)
