from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from ib_async import AccountValue, Contract, PortfolioItem, Stock, Ticker
from ib_async.objects import OptionComputation

from thetagang.monitor_config import load_monitor_config
from thetagang.monitor_metrics import MonitorSnapshot


class FakeIB:
    def __init__(self) -> None:
        self.account = "DU123456"
        self.stock = Stock("AAA", "SMART", "USD")
        self.stock.conId = 101
        self.stock.localSymbol = "AAA"
        self.option = Contract(
            secType="OPT",
            conId=102,
            symbol="AAA",
            lastTradeDateOrContractMonth="20261125",
            strike=60.0,
            right="C",
            multiplier="100",
            exchange="SMART",
            currency="USD",
            localSymbol="AAA 261125C00060000",
        )
        self.items = [
            PortfolioItem(self.stock, 10, 50, 500, 40, 100, 0, self.account),
            PortfolioItem(self.option, -1, 3, -300, 500, 200, 0, self.account),
        ]
        self.account_value_rows = [
            AccountValue(self.account, "NetLiquidation", "10000", "USD", ""),
            AccountValue(self.account, "AvailableFunds", "8000", "USD", ""),
        ]
        self.price_overrides: dict[int, float] = {}
        self.greeks_available = True
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.history_symbols: list[str] = []
        self.order_api_calls: list[str] = []
        self.connect_calls: list[tuple[str, int, int, bool, str]] = []
        self.market_data_types: list[int] = []
        self.connected = False
        self.disconnected = False

    def isConnected(self) -> bool:
        return self.connected

    async def connectAsync(
        self,
        host: str,
        port: int,
        *,
        clientId: int,
        readonly: bool,
        account: str,
    ) -> None:
        self.connect_calls.append((host, port, clientId, readonly, account))
        self.connected = True

    def reqMarketDataType(self, market_data_type: int) -> None:
        self.market_data_types.append(market_data_type)

    def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return [self.account]

    def accountValues(self, account: str = "") -> list[AccountValue]:
        self.calls.append(("accountValues", (account,)))
        return [
            row
            for row in self.account_value_rows
            if not account or row.account == account
        ]

    def portfolio(self, account: str = "") -> list[PortfolioItem]:
        self.calls.append(("portfolio", (account,)))
        return [item for item in self.items if not account or item.account == account]

    async def qualifyContractsAsync(self, *contracts: Contract) -> list[Contract]:
        self.calls.append(("qualifyContractsAsync", contracts))
        next_id = {"SPX": 201, "VIX": 202, "VIXMO": 203, "AAA": 204}
        for contract in contracts:
            if contract.conId == 0:
                contract.conId = next_id[contract.symbol]
        return list(contracts)

    async def reqTickersAsync(self, *contracts: Contract) -> list[Ticker]:
        self.calls.append(("reqTickersAsync", contracts))
        prices = {
            101: 50.0,
            102: 3.0,
            201: 5000.0,
            202: 20.0,
            203: 100.0,
            204: 50.0,
        }
        tickers = []
        for contract in contracts:
            price = self.price_overrides.get(contract.conId, prices[contract.conId])
            ticker = Ticker(contract=contract)
            ticker.bid = price - 0.01
            ticker.bidSize = 1.0
            ticker.ask = price + 0.01
            ticker.askSize = 1.0
            ticker.last = price
            if contract.conId == self.option.conId and self.greeks_available:
                ticker.modelGreeks = OptionComputation(
                    tickAttrib=0,
                    impliedVol=0.2,
                    delta=-0.5,
                    gamma=0.01,
                    theta=-0.02,
                    vega=0.1,
                    undPrice=50.0,
                )
            tickers.append(ticker)
        return tickers

    async def reqHistoricalDataAsync(
        self,
        contract: Contract,
        endDateTime,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int = 1,
        keepUpToDate: bool = False,
        chartOptions: Any = None,
        timeout: float = 60,
    ) -> list[SimpleNamespace]:
        self.calls.append(
            ("reqHistoricalDataAsync", (contract, durationStr, barSizeSetting))
        )
        self.history_symbols.append(contract.symbol)
        start = date(2025, 1, 1)
        return [
            SimpleNamespace(
                date=start + timedelta(days=index),
                close=(
                    100.0 * math.exp(index * 0.01 + index**2 * 0.0001)
                    if contract.symbol == "SPX"
                    else 50.0 * math.exp(2 * (index * 0.01 + index**2 * 0.0001))
                ),
            )
            for index in range(25)
        ]

    def openTrades(self) -> list[object]:
        self.calls.append(("openTrades", ()))
        return []

    def placeOrder(self, *args, **kwargs):
        self.order_api_calls.append("placeOrder")
        raise AssertionError("read-only monitor called placeOrder")

    def cancelOrder(self, *args, **kwargs):
        self.order_api_calls.append("cancelOrder")
        raise AssertionError("read-only monitor called cancelOrder")

    def reqGlobalCancel(self, *args, **kwargs):
        self.order_api_calls.append("reqGlobalCancel")
        raise AssertionError("read-only monitor called reqGlobalCancel")


@pytest.mark.asyncio
async def test_read_snapshot_maps_broker_data_and_does_not_submit_orders() -> None:
    from thetagang.monitor import read_snapshot

    config = replace(load_monitor_config("thetagang.nt.toml"), account_id="DU123456")
    ib = FakeIB()

    snapshot = await read_snapshot(ib, config)

    assert isinstance(snapshot, MonitorSnapshot)
    assert snapshot.accounts[0].account == "DU123456"
    assert snapshot.accounts[0].net_liquidation == 10_000
    assert snapshot.accounts[0].available_funds == 8_000
    stock, option = snapshot.positions
    assert (stock.kind, stock.quantity, stock.average_price, stock.market_price) == (
        "stock",
        10,
        40,
        50,
    )
    assert option.kind == "call"
    assert option.quantity == -1
    assert option.average_price == 5
    assert option.market_price == 3
    assert option.expiry == date(2026, 11, 25)
    assert option.strike == 60
    assert option.delta == -0.5
    assert option.beta == pytest.approx(2.0)
    assert snapshot.index_prices == {"spx": 5000, "vix": 20, "vixmo": 100}
    assert snapshot.open_orders == 0
    assert snapshot.timestamp_seconds <= datetime.now(UTC).timestamp()
    assert any(call[0] == "reqHistoricalDataAsync" for call in ib.calls)
    assert ib.order_api_calls == []


@pytest.mark.asyncio
async def test_read_snapshot_does_not_invent_missing_marks_or_greeks() -> None:
    from thetagang.monitor import read_snapshot

    ib = FakeIB()
    ib.price_overrides[102] = math.nan
    ib.greeks_available = False

    snapshot = await read_snapshot(ib, load_monitor_config("thetagang.nt.toml"))

    option = snapshot.positions[1]
    assert option.market_price is None
    assert option.delta is None
    assert option.implied_volatility is None


@pytest.mark.asyncio
async def test_read_snapshot_rejects_incomplete_requested_account() -> None:
    from thetagang.monitor import read_snapshot

    ib = FakeIB()
    ib.account_value_rows = ib.account_value_rows[:1]
    config = replace(load_monitor_config("thetagang.nt.toml"), account_id=ib.account)

    with pytest.raises(RuntimeError, match="Incomplete account snapshot"):
        await read_snapshot(ib, config)


@pytest.mark.asyncio
async def test_read_snapshot_rejects_when_no_accounts_are_available(
    monkeypatch,
) -> None:
    from thetagang.monitor import read_snapshot

    ib = FakeIB()
    monkeypatch.setattr(ib, "managedAccounts", list)

    with pytest.raises(RuntimeError, match="No IBKR accounts available"):
        await read_snapshot(ib, load_monitor_config("thetagang.nt.toml"))


@pytest.mark.asyncio
async def test_historical_data_is_fetched_once_per_symbol_per_day() -> None:
    from thetagang.monitor import _DailyHistoryCache, read_snapshot

    ib = FakeIB()
    config = replace(load_monitor_config("thetagang.nt.toml"), account_id=ib.account)
    history_cache = _DailyHistoryCache()

    await read_snapshot(ib, config, history_cache=history_cache)
    await read_snapshot(ib, config, history_cache=history_cache)

    assert len(ib.history_symbols) == 2
    assert set(ib.history_symbols) == {"SPX", "AAA"}


@pytest.mark.asyncio
async def test_runtime_connects_readonly_and_keeps_last_complete_snapshot(
    monkeypatch, caplog
) -> None:
    from thetagang import monitor

    config = replace(
        load_monitor_config("thetagang.nt.toml"),
        host="127.0.0.2",
        port=4002,
        client_id=37,
        account_id="DU654321",
        market_data_type="DELAYED_FROZEN",
        metrics_addr="127.0.0.3",
        metrics_port=9999,
    )
    ib = FakeIB()
    registry = _Registry()
    server = _HTTPServer()
    thread = _Thread()
    server_calls = []
    monkeypatch.setattr(monitor, "IB", lambda: ib)
    monkeypatch.setattr(monitor, "REGISTRY", registry)
    monkeypatch.setattr(
        monitor,
        "start_http_server",
        lambda port, addr: server_calls.append((port, addr)) or (server, thread),
    )
    completed_snapshot = MonitorSnapshot(
        accounts=(),
        positions=(),
        index_prices={},
        open_orders=0,
        timestamp_seconds=1234.0,
    )
    refresh_count = 0

    async def read_once(_ib, _config, *, history_cache=None):
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 1:
            return completed_snapshot
        raise RuntimeError("temporary snapshot failure")

    monkeypatch.setattr(monitor, "read_snapshot", read_once)
    real_sleep = asyncio.sleep
    sleep_count = 0
    sleep_delays = []

    async def cancel_after_failed_refresh(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        sleep_delays.append(delay)
        if sleep_count == 2:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(monitor.asyncio, "sleep", cancel_after_failed_refresh)
    caplog.set_level(logging.ERROR)

    with pytest.raises(asyncio.CancelledError):
        await monitor.run_monitor(config)

    assert server_calls == [(9999, "127.0.0.3")]
    assert sleep_delays == [config.interval_secs, config.interval_secs]
    assert ib.connect_calls == [
        (
            config.host,
            config.port,
            config.client_id,
            True,
            "DU654321",
        )
    ]
    assert ib.market_data_types == [4]
    assert ib.disconnected
    assert server.shutdown_called
    assert server.close_called
    assert thread.joined
    assert registry.unregistered
    assert refresh_count == 2
    assert "temporary snapshot failure" in caplog.text
    assert registry.collector is not None
    latest = {
        sample.name: sample.value
        for family in registry.collector.collect()
        for sample in family.samples
    }
    assert latest["thetagang_last_update_timestamp_seconds"] == 1234


class _Registry:
    def __init__(self) -> None:
        self.collector = None
        self.unregistered = False

    def register(self, collector) -> None:
        self.collector = collector

    def unregister(self, collector) -> None:
        assert collector is self.collector
        self.unregistered = True


class _HTTPServer:
    shutdown_called = False
    close_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True

    def server_close(self) -> None:
        self.close_called = True


class _Thread:
    joined = False

    def join(self, timeout: float | None = None) -> None:
        self.joined = True
