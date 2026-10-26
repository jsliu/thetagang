"""Pure snapshot-to-Prometheus calculations for the standalone monitor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TypeGuard

from prometheus_client.core import GaugeMetricFamily

from thetagang.monitor_config import MonitorConfig

RATE = 0.0425
VIX_ROOT = "^VIX"


@dataclass(frozen=True)
class AccountSnapshot:
    account: str
    net_liquidation: float
    available_funds: float


@dataclass(frozen=True)
class PositionSnapshot:
    instrument: str
    symbol: str
    kind: str
    quantity: float
    average_price: float
    multiplier: float
    market_price: float | None
    expiry: date | None
    strike: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    underlying_price: float | None
    implied_volatility: float | None
    beta: float


@dataclass(frozen=True)
class MonitorSnapshot:
    accounts: tuple[AccountSnapshot, ...]
    positions: tuple[PositionSnapshot, ...]
    index_prices: Mapping[str, float]
    open_orders: int
    timestamp_seconds: float


@dataclass
class _SymbolRisk:
    beta: float
    delta_shares: float = 0.0
    plain_dollars: float = 0.0
    beta_dollars: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


def _finite(value: float | None) -> TypeGuard[float]:
    return value is not None and math.isfinite(value)


def _normal_cdf(value: float) -> float:
    return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0


def _option_price(
    spot: float, strike: float, years: float, volatility: float, is_call: bool
) -> float:
    if spot <= 0 or strike <= 0 or volatility <= 0:
        return 0.0
    root_time = math.sqrt(years)
    d1 = (math.log(spot / strike) + (0.5 * volatility**2) * years) / (
        volatility * root_time
    )
    d2 = d1 - volatility * root_time
    discount = math.exp(-RATE * years)
    if is_call:
        return spot * discount * _normal_cdf(d1) - strike * discount * _normal_cdf(d2)
    return strike * discount * _normal_cdf(-d2) - spot * discount * _normal_cdf(-d1)


def _metric(
    families: dict[str, GaugeMetricFamily],
    name: str,
    help_text: str,
    value: float,
    **labels: str,
) -> None:
    family = families.get(name)
    if family is None:
        family = families[name] = GaugeMetricFamily(
            f"thetagang_{name}", help_text, labels=list(labels)
        )
    family.add_metric(list(labels.values()), float(value))


def _greek_values(position: PositionSnapshot) -> tuple[float, ...] | None:
    delta = position.delta
    gamma = position.gamma
    theta = position.theta
    vega = position.vega
    underlying = position.underlying_price
    volatility = position.implied_volatility
    if not (
        _finite(delta)
        and _finite(gamma)
        and _finite(theta)
        and _finite(vega)
        and _finite(underlying)
        and _finite(volatility)
    ):
        return None
    return delta, gamma, theta, vega, underlying, volatility


def _valid_greeks(position: PositionSnapshot) -> bool:
    return _greek_values(position) is not None


def _stress_pnl(
    positions: tuple[PositionSnapshot, ...],
    symbol_risk: dict[str, _SymbolRisk],
    move: float,
    use_beta: bool,
    today: date,
) -> float:
    total = 0.0
    for position in positions:
        if position.symbol == VIX_ROOT:
            continue
        risk = symbol_risk.get(position.symbol)
        if risk is None:
            continue
        beta_move = move * (risk.beta if use_beta else 1.0)
        shocked_move = max(beta_move, -0.99)
        if position.kind == "stock":
            if _finite(position.market_price) and position.market_price > 0:
                total += position.quantity * position.market_price * shocked_move
            continue
        greeks = _greek_values(position)
        if greeks is None or position.expiry is None:
            continue
        _, _, _, _, spot, volatility = greeks
        strike = position.strike
        if strike is None or strike <= 0:
            continue
        years = max((position.expiry - today).days, 1) / 365.25
        base = _option_price(spot, strike, years, volatility, position.kind == "call")
        shocked = _option_price(
            spot * (1 + shocked_move),
            strike,
            years,
            volatility,
            position.kind == "call",
        )
        total += position.quantity * position.multiplier * (shocked - base)
    return total


def build_metric_families(
    snapshot: MonitorSnapshot, config: MonitorConfig
) -> list[GaugeMetricFamily]:
    """Build a complete metric snapshot without mutating the published state."""
    families: dict[str, GaugeMetricFamily] = {}
    prices = {name.lower(): value for name, value in snapshot.index_prices.items()}
    for name in ("spx", "vix", "vixmo"):
        value = prices.get(name)
        if _finite(value) and value > 0:
            _metric(families, f"{name}_price", f"{name.upper()} index level", value)

    for account in snapshot.accounts:
        if not (
            account.account
            and _finite(account.net_liquidation)
            and _finite(account.available_funds)
            and account.net_liquidation > 0
        ):
            continue
        _metric(
            families,
            "net_liquidation",
            "Net liquidation (USD)",
            account.net_liquidation,
            account=account.account,
        )
        _metric(
            families,
            "available_funds",
            "Available funds (USD)",
            account.available_funds,
            account=account.account,
        )
        _metric(
            families,
            "margin_used",
            "Net liq - available funds",
            account.net_liquidation - account.available_funds,
            account=account.account,
        )

    unrealized = 0.0
    missing_greeks = 0
    long_option_alerts = 0
    symbol_risk: dict[str, _SymbolRisk] = {}
    spx = prices.get("spx")
    for position in snapshot.positions:
        if (
            not position.instrument
            or not position.symbol
            or not _finite(position.quantity)
            or position.quantity == 0
            or not _finite(position.average_price)
            or not _finite(position.multiplier)
            or position.multiplier <= 0
        ):
            continue
        labels = {
            "instrument": position.instrument,
            "symbol": position.symbol,
            "kind": position.kind,
        }
        _metric(
            families,
            "position_quantity",
            "Signed contracts or shares",
            position.quantity,
            **labels,
        )
        _metric(
            families,
            "position_avg_price",
            "Cost basis per share",
            position.average_price,
            **labels,
        )

        market_price = position.market_price
        if not _finite(market_price) or market_price <= 0:
            if (
                position.kind not in {"call", "put"}
                or position.symbol != VIX_ROOT
                and not _valid_greeks(position)
            ):
                missing_greeks += 1
            continue

        size = position.quantity * position.multiplier
        pnl = (market_price - position.average_price) * size
        unrealized += pnl
        _metric(families, "position_price", "Mid per share", market_price, **labels)
        _metric(
            families,
            "position_unrealized_pnl",
            "Unrealized P&L (USD)",
            pnl,
            **labels,
        )
        _metric(
            families,
            "position_market_value",
            "Market value (USD)",
            market_price * size,
            **labels,
        )

        if position.kind == "stock":
            if not _finite(position.beta):
                missing_greeks += 1
                continue
            risk = symbol_risk.setdefault(
                position.symbol, _SymbolRisk(beta=position.beta)
            )
            risk.delta_shares += position.quantity
            risk.plain_dollars += position.quantity * market_price
            risk.beta_dollars += position.quantity * market_price * position.beta
            continue

        if position.expiry is None:
            continue
        today = datetime.fromtimestamp(snapshot.timestamp_seconds, UTC).date()
        dte = (position.expiry - today).days
        _metric(families, "position_dte", "Days to expiry", dte, **labels)
        if position.quantity < 0:
            ratio = (
                (position.average_price - market_price) / position.average_price
                if position.average_price > 0
                else None
            )
        else:
            ratio = (
                (market_price - position.average_price) / position.average_price
                if position.average_price > 0
                else None
            )
        if ratio is not None:
            _metric(
                families,
                "position_pnl_ratio",
                "P&L as a share of premium",
                ratio,
                **labels,
            )

        if position.symbol == VIX_ROOT:
            continue
        greeks = _greek_values(position)
        if not _finite(position.beta) or greeks is None:
            missing_greeks += 1
            continue
        risk = symbol_risk.setdefault(position.symbol, _SymbolRisk(beta=position.beta))
        (
            delta_per_share,
            gamma_per_share,
            theta_per_share,
            vega_per_share,
            underlying,
            vol,
        ) = greeks
        delta = delta_per_share * size
        gamma = gamma_per_share * size
        theta = theta_per_share * size
        vega = vega_per_share * size
        risk.delta_shares += delta
        risk.gamma += gamma
        risk.theta += theta
        risk.vega += vega
        risk.plain_dollars += delta * underlying
        risk.beta_dollars += delta * underlying * position.beta
        symbol_risk[position.symbol] = risk
        _metric(
            families,
            "position_delta",
            "Option delta per share",
            delta_per_share,
            instrument=position.instrument,
        )
        _metric(
            families,
            "position_iv",
            "Implied volatility",
            vol,
            instrument=position.instrument,
        )
        if (
            position.quantity > 0
            and abs(delta_per_share) >= config.long_option_alert_delta
        ):
            long_option_alerts += 1

    _metric(
        families,
        "unrealized_pnl",
        "Unrealized P&L of priced positions",
        unrealized,
    )
    if spx and _finite(spx) and spx > 0:
        delta_dollars = sum(risk.plain_dollars for risk in symbol_risk.values())
        beta_dollars = sum(risk.beta_dollars for risk in symbol_risk.values())
        gamma_spx = 0.0
        for position in snapshot.positions:
            if position.symbol == VIX_ROOT or position.kind == "stock":
                continue
            greeks = _greek_values(position)
            if greeks is None or not _finite(position.beta):
                continue
            gamma_spx += (
                greeks[1]
                * position.quantity
                * position.multiplier
                * (position.beta * greeks[4] / spx) ** 2
            )
        theta = sum(risk.theta for risk in symbol_risk.values())
        vega = sum(risk.vega for risk in symbol_risk.values())
        _metric(
            families,
            "spx_delta",
            "SPX beta-weighted delta (SPX units)",
            beta_dollars / spx,
        )
        _metric(
            families,
            "beta_delta_dollars",
            "Beta-weighted delta $",
            beta_dollars,
        )
        _metric(
            families,
            "delta_dollars",
            "Delta $ without beta",
            delta_dollars,
        )
        _metric(
            families,
            "gamma_spx",
            "SPX delta change per SPX point",
            gamma_spx,
        )
        _metric(families, "theta", "Theta ($/day)", theta)
        _metric(families, "vega", "Vega ($ per vol point)", vega)
        _metric(
            families,
            "options_missing_greeks",
            "Positions without greeks",
            missing_greeks,
        )
        for symbol, risk in symbol_risk.items():
            _metric(families, "symbol_beta", "1y beta vs SPX", risk.beta, symbol=symbol)
            _metric(
                families,
                "symbol_delta_dollars",
                "Delta $ without beta",
                risk.plain_dollars,
                symbol=symbol,
            )
            _metric(
                families,
                "symbol_beta_delta_dollars",
                "Beta delta $",
                risk.beta_dollars,
                symbol=symbol,
            )
            _metric(
                families,
                "symbol_delta_shares",
                "Share-equivalent delta",
                risk.delta_shares,
                symbol=symbol,
            )
            _metric(
                families,
                "symbol_gamma",
                "Share delta per $1 move",
                risk.gamma,
                symbol=symbol,
            )
            _metric(
                families,
                "symbol_theta",
                "Theta ($/day)",
                risk.theta,
                symbol=symbol,
            )
            _metric(
                families,
                "symbol_vega",
                "Vega ($ per vol point)",
                risk.vega,
                symbol=symbol,
            )
        _metric(
            families,
            "long_option_alerts",
            f"Long options with |delta| >= {config.long_option_alert_delta}",
            long_option_alerts,
        )
        today = datetime.fromtimestamp(snapshot.timestamp_seconds, UTC).date()
        for move in config.stress_moves:
            for basis, use_beta in (("beta", True), ("one", False)):
                _metric(
                    families,
                    "stress_pnl",
                    "P&L if SPX moves (options repriced at current IV)",
                    _stress_pnl(snapshot.positions, symbol_risk, move, use_beta, today),
                    move=f"{move:+.0%}",
                    betas=basis,
                )
    _metric(families, "orders_open", "Open orders", snapshot.open_orders)
    _metric(
        families,
        "last_update_timestamp_seconds",
        "Last refresh",
        snapshot.timestamp_seconds,
    )
    return list(families.values())
