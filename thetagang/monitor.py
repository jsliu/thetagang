"""Standalone, read-only IBKR portfolio monitor."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Any, Protocol, TypeGuard

from ib_async import IB, Contract, Index, Stock, Ticker
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily

from thetagang.accounting import AccountingError, AccountMetric, select_account_value
from thetagang.monitor_config import MonitorConfig
from thetagang.monitor_metrics import (
    AccountSnapshot,
    MonitorSnapshot,
    PositionSnapshot,
    build_metric_families,
)

logger = logging.getLogger(__name__)
INDEX_SYMBOLS = {"SPX", "VIX", "VIXMO"}
MARKET_DATA_TYPES = {
    "REALTIME": 1,
    "FROZEN": 2,
    "DELAYED": 3,
    "DELAYED_FROZEN": 4,
}


class _SnapshotReader(Protocol):
    def managedAccounts(self) -> list[str]: ...

    def accountValues(self, account: str = "") -> list[Any]: ...

    def portfolio(self, account: str = "") -> list[Any]: ...

    async def qualifyContractsAsync(self, *contracts: Contract) -> Any: ...

    async def reqTickersAsync(self, *contracts: Contract) -> list[Ticker]: ...

    async def reqHistoricalDataAsync(
        self,
        contract: Contract,
        endDateTime: Any,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int = ...,
        keepUpToDate: bool = ...,
        chartOptions: Any = ...,
        timeout: float = 60,
    ) -> Any: ...

    def openTrades(self) -> list[Any]: ...


class _MetricSnapshot:
    """Serve the last complete metric set, never a partially built refresh."""

    def __init__(self) -> None:
        self.families: list[GaugeMetricFamily] = []

    def collect(self) -> list[GaugeMetricFamily]:
        return list(self.families)

    def replace(self, families: list[GaugeMetricFamily]) -> None:
        self.families = families


def _finite(value: Any) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _positive_price(ticker: Ticker | None) -> float | None:
    if ticker is None:
        return None
    price = ticker.marketPrice()
    return float(price) if _finite(price) and price > 0 else None


def _underlying_symbol(contract: Contract) -> str:
    if contract.secType in {"IND", "OPT"} and contract.symbol in INDEX_SYMBOLS:
        return f"^{contract.symbol}"
    return contract.symbol


def _underlying_contract(symbol: str) -> Contract:
    if symbol.startswith("^"):
        return Index(symbol[1:], "CBOE", "USD")
    return Stock(symbol, "SMART", "USD")


def _expiration(value: str) -> date | None:
    value = value.strip()
    try:
        if len(value) >= 8:
            return datetime.strptime(value[:8], "%Y%m%d").replace(tzinfo=UTC).date()
        if len(value) == 6:
            year, month = int(value[:4]), int(value[4:])
            next_month = date(year + month // 12, month % 12 + 1, 1)
            return next_month.fromordinal(next_month.toordinal() - 1)
    except ValueError:
        return None
    return None


def _bar_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(value[:10], fmt).replace(tzinfo=UTC).date()
            except ValueError:
                continue
    return None


def _compute_beta(
    closes: dict[date, float], index_closes: dict[date, float]
) -> float | None:
    days = sorted(
        day
        for day in closes.keys() & index_closes.keys()
        if closes[day] > 0 and index_closes[day] > 0
    )
    if len(days) < 20:
        return None
    asset_returns = [
        math.log(closes[second] / closes[first]) for first, second in pairwise(days)
    ]
    index_returns = [
        math.log(index_closes[second] / index_closes[first])
        for first, second in pairwise(days)
    ]
    asset_mean = sum(asset_returns) / len(asset_returns)
    index_mean = sum(index_returns) / len(index_returns)
    covariance = sum(
        (asset - asset_mean) * (index - index_mean)
        for asset, index in zip(asset_returns, index_returns)
    )
    variance = sum((index - index_mean) ** 2 for index in index_returns)
    if variance == 0 or not math.isfinite(variance):
        return None
    beta = covariance / variance
    return beta if math.isfinite(beta) else None


async def _historical_closes(
    ib: _SnapshotReader, contract: Contract
) -> dict[date, float]:
    bars = await ib.reqHistoricalDataAsync(
        contract,
        "",
        "1 Y",
        "1 day",
        "TRADES",
        True,
        timeout=10,
    )
    closes: dict[date, float] = {}
    for bar in bars:
        day = _bar_date(bar.date)
        close = getattr(bar, "close", None)
        if day is not None and _finite(close) and close > 0:
            closes[day] = float(close)
    return closes


class _DailyHistoryCache:
    def __init__(self) -> None:
        self._refreshed_on: dict[str, date] = {}
        self._closes: dict[str, dict[date, float]] = {}

    async def get(
        self, ib: _SnapshotReader, symbol: str, contract: Contract
    ) -> dict[date, float]:
        today = datetime.now(UTC).date()
        if self._refreshed_on.get(symbol) != today:
            self._refreshed_on[symbol] = today
            try:
                self._closes[symbol] = await _historical_closes(ib, contract)
            except (TimeoutError, ValueError) as exc:
                logger.warning("No daily history for %s: %s", symbol, exc)
        return self._closes.get(symbol, {})


def _option_greeks(ticker: Ticker | None) -> tuple[float, ...] | None:
    if ticker is None or ticker.modelGreeks is None:
        return None
    model = ticker.modelGreeks
    values = (
        model.delta,
        model.gamma,
        model.theta,
        model.vega,
        model.undPrice,
        model.impliedVol,
    )
    delta, gamma, theta, vega, underlying, volatility = values
    if not (
        _finite(delta)
        and _finite(gamma)
        and _finite(theta)
        and _finite(vega)
        and _finite(underlying)
        and _finite(volatility)
    ):
        return None
    if underlying <= 0 or volatility <= 0:
        return None
    return (
        float(delta),
        float(gamma),
        float(theta),
        float(vega),
        float(underlying),
        float(volatility),
    )


async def read_snapshot(
    ib: _SnapshotReader,
    config: MonitorConfig,
    *,
    history_cache: _DailyHistoryCache | None = None,
) -> MonitorSnapshot:
    """Read one complete broker snapshot; contains no order-management calls."""
    account_names = (
        [config.account_id] if config.account_id else list(ib.managedAccounts())
    )
    if not account_names:
        raise RuntimeError("No IBKR accounts available for monitor snapshot")
    accounts: list[AccountSnapshot] = []
    for account in account_names:
        values = ib.accountValues(account)
        try:
            net_liquidation = select_account_value(
                values, AccountMetric.NET_LIQUIDATION
            )
            available_funds = select_account_value(values, "AvailableFunds")
        except AccountingError as exc:
            raise RuntimeError(
                f"Incomplete account snapshot for {account}: {exc}"
            ) from exc
        accounts.append(AccountSnapshot(account, net_liquidation, available_funds))

    portfolio = ib.portfolio(config.account_id or "")
    position_items = [
        item
        for item in portfolio
        if item.contract.secType in {"STK", "OPT"}
        and (not config.account_id or item.account == config.account_id)
        and _finite(item.position)
        and item.position != 0
    ]
    underlying_contracts: dict[str, Contract] = {}
    quote_contracts: dict[int, Contract] = {}
    for item in position_items:
        contract = item.contract
        if contract.conId > 0:
            quote_contracts[contract.conId] = contract
        symbol = _underlying_symbol(contract)
        if contract.secType == "STK":
            underlying_contracts[symbol] = contract
        elif symbol not in underlying_contracts:
            underlying_contracts[symbol] = _underlying_contract(symbol)

    index_contracts: dict[str, Contract] = {
        f"^{symbol}": Index(symbol, "CBOE", "USD") for symbol in sorted(INDEX_SYMBOLS)
    }
    qualified_result = await ib.qualifyContractsAsync(
        *[
            contract
            for contract in (
                [item.contract for item in position_items]
                + list(underlying_contracts.values())
                + list(index_contracts.values())
            )
            if contract.conId <= 0
        ]
    )
    if isinstance(qualified_result, Contract):
        qualified = [qualified_result]
    else:
        qualified = [
            contract
            for contract in (qualified_result or [])
            if isinstance(contract, Contract)
        ]
    for contract in qualified:
        if contract.conId > 0:
            quote_contracts[contract.conId] = contract
            symbol = _underlying_symbol(contract)
            if symbol in underlying_contracts or contract.secType == "IND":
                underlying_contracts[symbol] = contract
            if contract.secType == "IND":
                index_contracts[symbol] = contract

    tickers = (
        await ib.reqTickersAsync(*quote_contracts.values()) if quote_contracts else []
    )
    ticker_by_id = {
        ticker.contract.conId: ticker
        for ticker in tickers
        if ticker.contract is not None and ticker.contract.conId > 0
    }
    index_prices = {
        name[1:].lower(): value
        for name, contract in index_contracts.items()
        if (value := _positive_price(ticker_by_id.get(contract.conId))) is not None
    }

    history_contracts = {
        "^SPX": index_contracts.get("^SPX", Index("SPX", "CBOE", "USD"))
    }
    history_contracts.update(
        {
            symbol: contract
            for symbol, contract in underlying_contracts.items()
            if not symbol.startswith("^")
        }
    )
    history_cache = history_cache or _DailyHistoryCache()
    historical: dict[str, dict[date, float]] = {}
    for symbol, contract in history_contracts.items():
        historical[symbol] = await history_cache.get(ib, symbol, contract)

    index_closes = historical.get("^SPX", {})
    betas: dict[str, float] = {}
    for symbol in underlying_contracts:
        if symbol.startswith("^"):
            continue
        beta = _compute_beta(historical.get(symbol, {}), index_closes)
        if beta is None:
            logger.warning("%s: beta defaulted to 1.0 (no usable history)", symbol)
            beta = 1.0
        betas[symbol] = beta

    positions: list[PositionSnapshot] = []
    for item in position_items:
        contract = item.contract
        symbol = _underlying_symbol(contract)
        is_option = contract.secType == "OPT"
        multiplier = 1.0
        if is_option:
            try:
                multiplier = float(contract.multiplier or "100")
            except ValueError:
                multiplier = 100.0
        instrument = contract.localSymbol or str(contract.conId)
        ticker = ticker_by_id.get(contract.conId)
        market_price = _positive_price(ticker)
        greeks = _option_greeks(ticker) if is_option else None
        underlying_ticker = ticker_by_id.get(
            underlying_contracts.get(symbol, Contract()).conId
        )
        underlying_price = (
            greeks[4]
            if greeks is not None
            else _positive_price(underlying_ticker)
            if is_option
            else None
        )
        average_cost = item.averageCost
        if is_option and multiplier > 0:
            average_cost /= multiplier
        expiry = (
            _expiration(contract.lastTradeDateOrContractMonth) if is_option else None
        )
        positions.append(
            PositionSnapshot(
                instrument=instrument,
                symbol=symbol,
                kind=(
                    "call"
                    if is_option and contract.right == "C"
                    else "put"
                    if is_option and contract.right == "P"
                    else "stock"
                ),
                quantity=float(item.position),
                average_price=float(average_cost),
                multiplier=multiplier,
                market_price=market_price,
                expiry=expiry,
                strike=float(contract.strike) if is_option else None,
                delta=greeks[0] if greeks is not None else None,
                gamma=greeks[1] if greeks is not None else None,
                theta=greeks[2] if greeks is not None else None,
                vega=greeks[3] if greeks is not None else None,
                underlying_price=underlying_price,
                implied_volatility=greeks[5] if greeks is not None else None,
                beta=1.0 if symbol.startswith("^") else betas.get(symbol, 1.0),
            )
        )

    return MonitorSnapshot(
        accounts=tuple(accounts),
        positions=tuple(positions),
        index_prices=index_prices,
        open_orders=len(ib.openTrades()),
        timestamp_seconds=time.time(),
    )


async def run_monitor(config: MonitorConfig) -> None:
    """Publish snapshots and reconnect on broker failures without submitting orders."""
    ib = IB()
    collector = _MetricSnapshot()
    history_cache = _DailyHistoryCache()
    server = None
    thread = None
    registered = False
    try:
        server, thread = start_http_server(config.metrics_port, config.metrics_addr)
        REGISTRY.register(collector)
        registered = True
        logger.info(
            "Portfolio metrics on http://%s:%s/metrics",
            config.metrics_addr,
            config.metrics_port,
        )
        while True:
            try:
                if not ib.isConnected():
                    await ib.connectAsync(
                        config.host,
                        config.port,
                        clientId=config.client_id,
                        readonly=True,
                        account=config.account_id or "",
                    )
                    ib.reqMarketDataType(MARKET_DATA_TYPES[config.market_data_type])
                snapshot = await read_snapshot(ib, config, history_cache=history_cache)
                collector.replace(build_metric_families(snapshot, config))
            except Exception:
                logger.exception("Portfolio monitor refresh failed")
            await asyncio.sleep(config.interval_secs)
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        finally:
            if registered:
                REGISTRY.unregister(collector)
            if server is not None:
                try:
                    server.shutdown()
                finally:
                    server.server_close()
            if thread is not None:
                thread.join(timeout=2)
