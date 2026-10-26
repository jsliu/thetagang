"""`thetagang-nt -c thetagang.nt.toml [--role all|monitor|trade]`: run the Nautilus TradingNode."""

from __future__ import annotations

import click
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


def build(path: str, role: str = "all") -> TradingNode:
    app = load(path)
    monitor, strategy, client_id = split(app, role)
    node = TradingNode(node_config(app, client_id))
    extras = None
    if monitor.publish:  # the read-only side connection, only where metrics are served
        extras = AccountExtras(
            app.ib.host, app.ib.port, client_id + EXTRA_CLIENT_OFFSET, app.ib.account_id
        )
    node.trader.add_actor(PortfolioMonitor(monitor, rules=app.strategy, extras=extras))
    if strategy is not None:
        from thetagang_nt.strategy import Discretionary

        node.trader.add_strategy(Discretionary(strategy))
    if role == "trade":  # the strategy's own metrics; the monitor publishes none here
        start_http_server(strategy.metrics_port, monitor.metrics_addr)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()
    return node


@click.command()
@click.option("-c", "--config", "path", default="thetagang.nt.toml", show_default=True)
@click.option(
    "--role",
    type=click.Choice(ROLES),
    default="all",
    show_default=True,
    help="monitor: the portfolio only, never trades; trade: the strategy only; "
    "all: both in one process",
)
def cli(path: str, role: str) -> None:
    node = build(path, role)
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    cli()
