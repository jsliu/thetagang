"""`thetagang-nt -c thetagang.nt.toml [--role all|monitor|trade]`: run the Nautilus TradingNode.
`thetagang-nt -c thetagang.nt.toml pretrade [TRADE ...]`: preview trades, send on confirmation."""

from __future__ import annotations

import click
import msgspec
from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.live.node import TradingNode
from prometheus_client import start_http_server

from thetagang_nt.config import ROLES, load, node_config, split
from thetagang_nt.ib_extra import EXTRA_CLIENT_OFFSET, AccountExtras
from thetagang_nt.monitor import PortfolioMonitor


def build(path: str, role: str = "all", adhoc: tuple[str, ...] = ()) -> TradingNode:
    app = load(path)
    monitor, strategy, client_id = split(app, role)
    if role == "pretrade":  # a terminal: errors only, so the report stays readable
        app = msgspec.structs.replace(
            app, node=msgspec.structs.replace(app.node, log_level="ERROR")
        )
    node = TradingNode(node_config(app, client_id))
    extras = None
    if monitor.publish:  # the read-only side connection, only where metrics are served
        extras = AccountExtras(
            app.ib.host, app.ib.port, client_id + EXTRA_CLIENT_OFFSET, app.ib.account_id
        )
    node.trader.add_actor(PortfolioMonitor(monitor, rules=app.strategy, extras=extras))
    if role == "pretrade":
        from thetagang_nt.pretrade import PreTrade, PreTradeConfig

        config = PreTradeConfig(
            **msgspec.structs.asdict(strategy),
            adhoc=adhoc,
            stress_moves=app.monitor.stress_moves,
        )
        node.trader.add_strategy(PreTrade(config))
        start_http_server(strategy.pretrade_metrics_port, monitor.metrics_addr)
    elif strategy is not None:
        from thetagang_nt.strategy import Discretionary

        node.trader.add_strategy(Discretionary(strategy))
    if role == "trade":  # the strategy's own metrics; the monitor publishes none here
        start_http_server(strategy.metrics_port, monitor.metrics_addr)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()
    return node


def run(node: TradingNode) -> None:
    try:
        node.run()
    finally:
        node.dispose()


@click.group(invoke_without_command=True)
@click.option("-c", "--config", "path", default="thetagang.nt.toml", show_default=True)
@click.option(
    "--role",
    type=click.Choice(ROLES),
    default="all",
    show_default=True,
    help="monitor: the portfolio only, never trades; trade: the strategy only; "
    "all: both in one process",
)
@click.pass_context
def cli(ctx: click.Context, path: str, role: str) -> None:
    ctx.obj = path
    if ctx.invoked_subcommand is None:
        run(build(path, role))


@cli.command()
@click.argument("trades", nargs=-1)
@click.pass_obj
def pretrade(path: str, trades: tuple[str, ...]) -> None:
    """Show the portfolio's risk now and after TRADES, and send them if you say y.

    Without TRADES: the [[strategy.trades]], sized to bring SPX delta to 0.
    Each TRADE: "buy 100 MS", "sell 2 AA 20261218 40P", or a vertical as one
    combo order, "buy 1 AA 20261218 40P/23P" (buy the 40 put, sell the 23)."""
    from thetagang_nt.rules import parse_trade

    for t in trades:  # fail on a typo before connecting to IB
        try:
            parse_trade(t)
        except ValueError as e:
            raise click.BadParameter(str(e)) from None
    run(build(path, "pretrade", trades))


if __name__ == "__main__":
    cli()
