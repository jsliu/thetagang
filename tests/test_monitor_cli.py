from __future__ import annotations

import sys

from click.testing import CliRunner

from thetagang.monitor_cli import cli


def test_monitor_cli_help_does_not_import_nautilus() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--config" in result.output
    assert "nautilus_trader" not in sys.modules


def test_documented_config_option_is_available_without_starting_monitor(
    monkeypatch,
) -> None:
    from thetagang import monitor_cli

    def fail_if_started(_config: str) -> None:
        raise AssertionError("CLI help must not load configuration or start IBKR")

    monkeypatch.setattr(monitor_cli, "load_monitor_config", fail_if_started)

    result = CliRunner().invoke(cli, ["--config", "thetagang.nt.toml", "--help"])

    assert result.exit_code == 0
    assert "--config PATH" in result.output
