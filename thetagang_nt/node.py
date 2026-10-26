"""`thetagang-nt -c thetagang.nt.toml`: run the Nautilus TradingNode."""

from __future__ import annotations

import click
from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.live.node import TradingNode

from thetagang_nt.config import load, node_config
from thetagang_nt.monitor import PortfolioMonitor


def build(path: str) -> TradingNode:
    app = load(path)
    node = TradingNode(node_config(app))
    node.trader.add_actor(PortfolioMonitor(app.monitor))
    if app.strategy is not None:
        from thetagang_nt.strategy import Discretionary

        node.trader.add_strategy(Discretionary(app.strategy))
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()
    return node


@click.command()
@click.option("-c", "--config", "path", default="thetagang.nt.toml", show_default=True)
def cli(path: str) -> None:
    node = build(path)
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    cli()
