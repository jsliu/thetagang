"""Await Nautilus requests from a coroutine.

Actor.request_* take a completion callback; these wrap it in a future so a run
can read top to bottom (load chain -> quote it -> pick -> order). Must be called
from the TradingNode's event loop, i.e. from actor handlers or tasks they spawn."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterable
from datetime import datetime
from typing import Any

from nautilus_trader.common.actor import Actor
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import IndexInstrument

from thetagang_nt.portfolio import VENUE, price

REQUEST_TIMEOUT_SECS = 60
_loop: asyncio.AbstractEventLoop | None = None  # the node's, seen in on_start


def spawn(actor: Actor, coro: Coroutine[Any, Any, Any], name: str) -> None:
    """Run `coro` on the node loop; log (never swallow) a failure. Clock timer
    callbacks fire outside the loop's context, so they schedule thread-safely."""
    global _loop

    def start() -> None:
        task = _loop.create_task(coro, name=name)
        task.add_done_callback(
            lambda t: t.cancelled()
            or t.exception() is None
            or actor.log.exception(f"{name} failed", t.exception())
        )

    try:
        _loop = asyncio.get_running_loop()
        start()
    except RuntimeError:
        _loop.call_soon_threadsafe(start)


async def _request(send: Callable[[Callable[[Any], None]], Any]) -> None:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[None] = loop.create_future()

    def finished(_request_id: Any) -> None:
        loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(None))

    send(finished)
    await asyncio.wait_for(fut, REQUEST_TIMEOUT_SECS)


async def load_contracts(actor: Actor, contracts: list[dict[str, Any]]) -> None:
    """Load IB contracts (and option chains, with build_options_chain) into the cache."""
    await _request(
        lambda cb: actor.request_instruments(
            VENUE, params={"ib_contracts": contracts}, callback=cb
        )
    )


async def load_instrument(actor: Actor, instrument_id: InstrumentId) -> None:
    """Load one instrument, e.g. an option spread from new_generic_spread_id."""
    if actor.cache.instrument(instrument_id) is None:
        await _request(lambda cb: actor.request_instrument(instrument_id, callback=cb))


async def request_bars(actor: Actor, bar_type: BarType, start: datetime) -> None:
    """Historical bars; they arrive through the actor's on_historical_data."""
    await _request(lambda cb: actor.request_bars(bar_type, start, callback=cb))


def subscribe(actor: Actor, instrument_id: InstrumentId) -> None:
    if isinstance(actor.cache.instrument(instrument_id), IndexInstrument):
        actor.subscribe_index_prices(instrument_id)
    else:
        actor.subscribe_quote_ticks(instrument_id)


def unsubscribe(actor: Actor, instrument_id: InstrumentId) -> None:
    if isinstance(actor.cache.instrument(instrument_id), IndexInstrument):
        actor.unsubscribe_index_prices(instrument_id)
    else:
        actor.unsubscribe_quote_ticks(instrument_id)


async def wait_for_prices(
    actor: Actor, ids: Iterable[InstrumentId], timeout: float
) -> dict[InstrumentId, float | None]:
    """Subscribe and wait until every instrument has a price (or the timeout)."""
    ids = list(ids)
    for iid in ids:
        if actor.cache.instrument(iid) is not None:
            subscribe(actor, iid)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if all(price(actor.cache, iid) is not None for iid in ids):
            break
        await asyncio.sleep(0.25)
    return {iid: price(actor.cache, iid) for iid in ids}
