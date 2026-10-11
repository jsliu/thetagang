"""Portfolio risk math for manage-only mode: SPX beta-weighted greeks, stress P&L,
delta by symbol, and the SPX delta = 0 proposals. Pure functions, no IB calls."""

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from ib_async import PortfolioItem, util
from rich.table import Table

from thetagang.fmt import dfmt, ffmt, pfmt
from thetagang.options import option_dte


def compute_beta(
    closes: dict[Any, float], index_closes: dict[Any, float]
) -> float | None:
    """Beta of daily log returns vs the index over shared dates; None if there
    are fewer than 20 shared dates (callers fall back to 1.0)."""
    dates = sorted(set(closes) & set(index_closes))
    if len(dates) < 20:
        return None
    a = np.diff(np.log([closes[d] for d in dates]))
    b = np.diff(np.log([index_closes[d] for d in dates]))
    return float(np.cov(a, b)[0][1] / np.var(b, ddof=1))


SCENARIO_MOVES = [-0.10, -0.05, 0.05, 0.10]


def bs_price(spot: float, strike: float, years: float, vol: float, right: str) -> float:
    """Black-Scholes price per share, zero rate and dividends.
    ponytail: European, r=0; fine for stress deltas, not for pricing deep ITM American puts."""
    sd = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + sd * sd / 2) / sd
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    call = spot * cdf(d1) - strike * cdf(d1 - sd)
    return call if right.upper().startswith("C") else call - spot + strike


def zero_delta_targets(
    delta_dollars: dict[str, float], betas: dict[str, float]
) -> dict[str, float]:
    """Target delta $ (no beta) per symbol so that both totals are zero:
    sum(delta $) = 0 and sum(beta x delta $) = 0.

    Smallest proportional change: minimize sum(change^2 / |delta $|), i.e. weighted
    least squares. With only the first condition this is "every symbol changes by
    the same percentage of its size". Symbols with zero delta $ never change."""
    symbols = list(delta_dollars)
    if not symbols:
        return {}
    d = np.array([delta_dollars[s] for s in symbols], dtype=float)
    b = np.array([betas[s] for s in symbols], dtype=float)
    w = np.abs(d)
    a = np.vstack([np.ones_like(d), b])  # constraint rows
    residual = -(a @ d)  # how far each total is from zero
    # change = W A^T (A W A^T)^+ residual; pinv also covers all-equal betas
    change = w * (a.T @ (np.linalg.pinv((a * w) @ a.T) @ residual))
    return {s: float(x) for s, x in zip(symbols, d + change)}


def side_weight(beta_dollars: float, gross: float) -> str:
    """Positive weight (share of gross |beta delta $|) plus the market side."""
    if abs(beta_dollars) < 0.5 or not gross:
        return "0.0%"
    side = "long" if beta_dollars > 0 else "short"
    return f"{pfmt(abs(beta_dollars) / gross, 1)} {side}"


class Adjuster(NamedTuple):
    """One position the zero-delta proposal may resize: a single option, or a whole
    credit vertical. Greeks are per unit: +1 = buy 1 contract, or open 1 more spread."""

    exposure: float  # |share delta| of what is held, used to pick the main position
    name: str
    is_spread: bool
    delta: float  # share delta per unit
    gamma: float  # SPX gamma per unit
    theta: float
    vega: float
    held: float  # units held; closing is capped here so positions never flip
    close_sign: int  # direction of units that close: -1 sell/close, +1 buy back


def size_adjustment(adjuster: Adjuster, shares: float) -> tuple[int, str]:
    """Whole units that best add `shares` of delta, without flipping the position."""
    units = round(shares / adjuster.delta)
    if units * adjuster.close_sign > adjuster.held:
        units = int(adjuster.close_sign * adjuster.held)
    if not units:
        return 0, "0"
    if adjuster.is_spread:
        verb = "open" if units > 0 else "close"
        return units, f"{verb} {abs(units)} spread {adjuster.name}"
    return units, f"{units:+d} {adjuster.name}"


def stock_trade(shares: float, held: float) -> int:
    """Whole shares to trade toward `shares`, closing at most what is held."""
    traded = round(shares)
    if traded * held < 0 and abs(traded) > abs(held):
        return -round(held)
    return traded


def per_contract(
    item: PortfolioItem, g: Any, beta: float, spx_price: float
) -> tuple[float, float, float, float]:
    """One contract held long: share delta, SPX gamma, theta, vega."""
    m = float(item.contract.multiplier or 100)
    spx = (beta * g.undPrice / spx_price) ** 2
    return (g.delta * m, g.gamma * m * spx, g.theta * m, g.vega * m)


def spread_adjuster(
    short: PortfolioItem,
    long: PortfolioItem,
    short_unit: tuple[float, float, float, float],
    long_unit: tuple[float, float, float, float],
) -> Adjuster | None:
    """A credit vertical resized as whole spreads; None below 1 share of delta."""
    delta, gamma, theta, vega = (lv - sv for sv, lv in zip(short_unit, long_unit))
    if abs(delta) < 1:
        return None
    held = abs(short.position)
    c = short.contract
    name = (
        f"{c.symbol} {c.lastTradeDateOrContractMonth} "
        f"{c.strike:g}/{long.contract.strike:g}{c.right}"
    )
    return Adjuster(abs(delta * held), name, True, delta, gamma, theta, vega, held, -1)


def single_adjuster(
    item: PortfolioItem, unit: tuple[float, float, float, float]
) -> Adjuster | None:
    """One option resized a contract at a time; None below 0.01 delta."""
    delta, gamma, theta, vega = unit
    if abs(delta) < 1:
        return None
    held = abs(item.position)
    close_sign = 1 if item.position < 0 else -1
    return Adjuster(
        abs(delta * held),
        item.contract.localSymbol,
        False,
        delta,
        gamma,
        theta,
        vega,
        held,
        close_sign,
    )


def scenario_pnl(legs: list[tuple[Any, ...]], spx_move: float, use_beta: bool) -> float:
    """Estimated P&L if SPX moves by spx_move and each underlying by beta * spx_move."""
    total = 0.0
    for leg in legs:
        beta, size, spot = leg[0], leg[1], leg[2]
        # a stock can't fall more than 100% (thin-history betas can imply that)
        move = max(spx_move * (beta if use_beta else 1.0), -0.99)
        if len(leg) == 3:  # stock
            total += size * spot * move
            continue
        strike, years, vol, right, delta, gamma = leg[3:]
        d_spot = spot * move
        if vol and not util.isNan(vol) and vol > 0:
            new = bs_price(spot + d_spot, strike, years, vol, right)
            total += size * (new - bs_price(spot, strike, years, vol, right))
        else:  # no implied vol: delta-gamma approximation
            total += size * (delta * d_spot + 0.5 * gamma * d_spot**2)
    return total


@dataclass
class Totals:
    """Portfolio-wide greeks in SPX terms."""

    delta_dollars: float = 0.0  # beta delta $
    plain_delta_dollars: float = 0.0  # delta $ without beta
    gamma: float = 0.0  # SPX delta change per 1-point SPX move
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class Book:
    """Per-position data the risk tables are built from."""

    totals: Totals
    legs: list[tuple[Any, ...]]  # scenario_pnl inputs
    by_symbol: dict[str, list[float]]  # symbol -> [delta $, beta delta $]
    prices: dict[str, float]  # symbol -> underlying price
    option_items: dict[str, list[PortfolioItem]]  # all option positions, by symbol
    missing: int  # options without greeks


def valid_greeks(g: Any) -> bool:
    """IB model greeks with every field the risk math needs."""
    if g is None:
        return False
    values = (g.delta, g.gamma, g.theta, g.vega, g.undPrice)
    return all(v is not None and not util.isNan(v) for v in values)


def contract_greeks(
    items: list[PortfolioItem], greeks: dict[int, Any]
) -> dict[int, tuple[float, float, float, float]]:
    """Per contract, as IB quotes them (by conId): delta, gamma, theta, vega per
    share. Stock is delta 1; options without greeks from IB are left out."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for i in items:
        if i.contract.secType == "STK":
            out[i.contract.conId] = (1.0, 0.0, 0.0, 0.0)
            continue
        g = greeks.get(i.contract.conId)
        if g is not None and valid_greeks(g):
            out[i.contract.conId] = (g.delta, g.gamma, g.theta, g.vega)
    return out


def position_greeks(
    items: list[PortfolioItem], greeks: dict[int, Any]
) -> dict[int, tuple[float, float, float, float]]:
    """Per position (by conId): delta in shares, gamma in shares per $1 move of the
    underlying, theta in $/day, vega in $ per vol point. Stock is delta only;
    options without greeks from IB are left out."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for i in items:
        if i.contract.secType == "STK":
            out[i.contract.conId] = (float(i.position), 0.0, 0.0, 0.0)
            continue
        g = greeks.get(i.contract.conId)
        if g is None or not valid_greeks(g):
            continue
        size = i.position * float(i.contract.multiplier or 100)
        out[i.contract.conId] = (
            g.delta * size,
            g.gamma * size,
            g.theta * size,
            g.vega * size,
        )
    return out


def symbol_greeks(
    items: list[PortfolioItem],
    by_position: dict[int, tuple[float, float, float, float]],
) -> dict[str, tuple[float, float, float, float] | None]:
    """position_greeks summed per symbol; None if any option in it has no greeks."""
    out: dict[str, tuple[float, float, float, float] | None] = {}
    for i in items:
        symbol = i.contract.symbol
        g = by_position.get(i.contract.conId)
        current = out.get(symbol, (0.0, 0.0, 0.0, 0.0))
        if g is None or current is None:
            out[symbol] = None
            continue
        d, ga, t, v = current
        out[symbol] = (d + g[0], ga + g[1], t + g[2], v + g[3])
    return out


def aggregate_book(
    items: list[PortfolioItem],
    betas: dict[str, float],
    greeks: dict[int, Any],
    spx_price: float,
) -> Book:
    book = Book(Totals(), [], {}, {}, {}, 0)
    t = book.totals
    for i in items:
        symbol = i.contract.symbol
        beta = betas[symbol]
        row = book.by_symbol.setdefault(symbol, [0.0, 0.0])
        if i.contract.secType == "STK":
            exposure = i.position * i.marketPrice
            book.legs.append((beta, i.position, i.marketPrice))
            book.prices[symbol] = i.marketPrice
        else:
            # every option is listed, so spreads pair even if a leg has no greeks
            book.option_items.setdefault(symbol, []).append(i)
            g = greeks.get(i.contract.conId)
            if g is None or not valid_greeks(g):
                book.missing += 1
                continue
            size = i.position * float(i.contract.multiplier or 100)
            exposure = g.delta * size * g.undPrice
            book.prices.setdefault(symbol, g.undPrice)
            t.gamma += g.gamma * size * (beta * g.undPrice / spx_price) ** 2
            t.theta += g.theta * size
            t.vega += g.vega * size
            years = max(option_dte(i.contract.lastTradeDateOrContractMonth), 1) / 365
            book.legs.append(
                (
                    beta,
                    size,
                    g.undPrice,
                    i.contract.strike,
                    years,
                    g.impliedVol,
                    i.contract.right,
                    g.delta,
                    g.gamma,
                )
            )
        t.plain_delta_dollars += exposure
        t.delta_dollars += exposure * beta
        row[0] += exposure
        row[1] += exposure * beta
    # symbols whose only positions lacked greeks have no exposure row
    book.by_symbol = {s: r for s, r in book.by_symbol.items() if s in book.prices}
    return book


def no_noise_dollars(x: float) -> str:
    return dfmt(x if abs(x) >= 0.5 else 0.0, 0)  # no "$-0" from float noise


def greeks_table(book: Book, after: Totals, spx_price: float) -> Table:
    now = book.totals
    table = Table(title="Portfolio greeks (SPX beta-weighted)")
    table.add_column("Item")
    table.add_column("Now", justify="right")
    table.add_column("After trades", justify="right")
    table.add_row("SPX price", dfmt(spx_price), "")
    rows = [
        (
            "SPX delta",
            now.delta_dollars / spx_price,
            after.delta_dollars / spx_price,
            ffmt,
            2,
        ),
        ("Beta delta $", now.delta_dollars, after.delta_dollars, dfmt, 0),
        (
            "Delta $ (no beta)",
            now.plain_delta_dollars,
            after.plain_delta_dollars,
            dfmt,
            0,
        ),
        ("Gamma (SPX delta / pt)", now.gamma, after.gamma, ffmt, 4),
        ("Theta ($/day)", now.theta, after.theta, dfmt, 0),
        ("Vega ($/vol pt)", now.vega, after.vega, dfmt, 0),
    ]
    for name, now_value, after_value, fmt, digits in rows:
        # round away float noise so "after" shows 0 instead of -0
        after_value = 0.0 if abs(after_value) < 0.5 * 10**-digits else after_value
        table.add_row(name, fmt(now_value, digits), fmt(after_value, digits))
    if book.missing:
        table.add_section()
        table.add_row("[yellow]Options without greeks", str(book.missing), "")
    return table


def stress_table(legs: list[tuple[Any, ...]]) -> Table:
    scenarios = Table(
        title="Stress P&L (options repriced at current IV)",
        caption=(
            "How to read it:\n"
            "- Down loss much bigger than up gain, or -10% loss more than 2x the "
            "-5% loss: the book is short gamma and big moves hurt even when delta is ~0.\n"
            "- Big gap between the two columns: the hedge depends on beta "
            "estimates (correlation risk)."
        ),
        caption_justify="left",
    )
    scenarios.add_column("SPX move")
    scenarios.add_column("Each stock moves beta x", justify="right")
    scenarios.add_column("All betas = 1", justify="right")
    for move in SCENARIO_MOVES:
        scenarios.add_row(
            pfmt(move, 0),
            dfmt(scenario_pnl(legs, move, use_beta=True), 0),
            dfmt(scenario_pnl(legs, move, use_beta=False), 0),
        )
    return scenarios


def breakdown_table(book: Book, betas: dict[str, float]) -> Table:
    breakdown = Table(
        title="Delta by symbol (largest beta effect first)",
        caption=(
            "Beta effect = beta delta $ - delta $. Its total drives the gap "
            "between the stress columns: a -10% SPX move changes P&L by about "
            "-10% of the beta effect beyond the all-betas-1 column."
        ),
        caption_justify="left",
    )
    breakdown.add_column("Symbol")
    breakdown.add_column("Beta", justify="right")
    breakdown.add_column("Delta $ (no beta)", justify="right")
    breakdown.add_column("Beta delta $", justify="right")
    breakdown.add_column("Beta effect", justify="right")
    for symbol, (plain, weighted) in sorted(
        book.by_symbol.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])
    ):
        breakdown.add_row(
            symbol,
            ffmt(betas[symbol], 2),
            dfmt(plain, 0),
            dfmt(weighted, 0),
            no_noise_dollars(weighted - plain),
        )
    t = book.totals
    breakdown.add_section()
    breakdown.add_row(
        "Total",
        "",
        dfmt(t.plain_delta_dollars, 0),
        dfmt(t.delta_dollars, 0),
        no_noise_dollars(t.delta_dollars - t.plain_delta_dollars),
    )
    return breakdown
