"""Configuration for the standalone, read-only portfolio monitor."""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonitorConfig:
    host: str
    port: int
    client_id: int
    account_id: str | None
    market_data_type: str
    interval_secs: int
    metrics_addr: str
    metrics_port: int
    long_option_alert_delta: float
    stress_moves: tuple[float, ...]


def _table(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _integer(table: dict[str, object], key: str, default: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(table: dict[str, object], key: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _string(table: dict[str, object], key: str, default: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    if not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def load_monitor_config(path: str | Path) -> MonitorConfig:
    """Load `[ib]` and `[monitor]` settings without importing NautilusTrader."""
    with Path(path).open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    ib = _table(raw_config, "ib")
    monitor = _table(raw_config, "monitor")
    host = _string(ib, "host", "127.0.0.1")
    port = _integer(ib, "port", 7497)
    client_id = _integer(ib, "client_id", 11)
    account = ib.get("account_id")
    if account is not None and not isinstance(account, str):
        raise TypeError("account_id must be a string")
    account_id = account or os.environ.get("TWS_ACCOUNT") or None
    market_data_type = _string(ib, "market_data_type", "REALTIME").upper()
    if market_data_type not in {
        "REALTIME",
        "DELAYED",
        "DELAYED_FROZEN",
        "FROZEN",
    }:
        raise ValueError(f"unsupported market_data_type: {market_data_type}")

    interval_secs = _integer(monitor, "interval_secs", 60)
    metrics_addr = _string(monitor, "metrics_addr", "0.0.0.0")
    metrics_port = _integer(monitor, "metrics_port", 9464)
    alert_delta = _number(monitor, "long_option_alert_delta", 0.8)
    raw_stress_moves = monitor.get("stress_moves", (-0.1, -0.05, 0.05, 0.1))
    if not isinstance(raw_stress_moves, (list, tuple)):
        raise TypeError("stress_moves must be an array of finite numbers")
    stress_moves = tuple(
        _number({"move": value}, "move", 0.0) for value in raw_stress_moves
    )

    if not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    if client_id < 0:
        raise ValueError("client_id must not be negative")
    if not 1 <= metrics_port <= 65535:
        raise ValueError("metrics_port must be in 1..65535")
    if interval_secs <= 0:
        raise ValueError("interval_secs must be positive")
    if not 0.0 <= alert_delta <= 1.0:
        raise ValueError("long_option_alert_delta must be in 0..1")
    if any(move <= -1.0 for move in stress_moves):
        raise ValueError("stress_moves must be greater than -1")

    return MonitorConfig(
        host=host,
        port=port,
        client_id=client_id,
        account_id=account_id,
        market_data_type=market_data_type,
        interval_secs=interval_secs,
        metrics_addr=metrics_addr,
        metrics_port=metrics_port,
        long_option_alert_delta=alert_delta,
        stress_moves=stress_moves,
    )
