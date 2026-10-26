"""Command line entry point for standalone portfolio monitoring."""

from __future__ import annotations

import asyncio
import logging

import click

from thetagang.monitor_config import load_monitor_config


@click.command()
@click.option(
    "-c",
    "--config",
    default="thetagang.nt.toml",
    show_default=True,
    type=click.Path(exists=True, readable=True),
)
def cli(config: str) -> None:
    """Serve read-only portfolio metrics without running either trading bot."""
    logging.basicConfig(level=logging.INFO)
    settings = load_monitor_config(config)
    from thetagang.monitor import run_monitor

    asyncio.run(run_monitor(settings))
