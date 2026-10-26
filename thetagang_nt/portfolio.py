"""Nautilus cache -> thetagang view: prices, net legs, greeks and the SPX book.

Shared by the monitor (metrics) and the strategy (sizing). Every instrument lives
on one SMART venue (symbol_to_mic_venue={"": "SMART"}) so an option's underlying
resolves as "<symbol>.SMART" for Nautilus' GreeksCalculator."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from nautilus_trader.cache.cache import Cache
from nautilus_trader.core.nautilus_pyo3 import (
    black_scholes_greeks,
    imply_vol_and_greeks,
)
from nautilus_trader.model.enums import OptionKind, PriceType
from nautilus_trader.model.greeks import GreeksCalculator
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.instruments import Equity, OptionContract

from thetagang_nt.rules import Leg

VENUE = Venue("SMART")
SPX = InstrumentId.from_str("^SPX.SMART")
VIX = InstrumentId.from_str("^VIX.SMART")
VIXMO = InstrumentId.from_str("^VIXMO.SMART")
INDEX_CONTRACTS = [
    {"secType": "IND", "symbol": s, "exchange": "CBOE"} for s in ("SPX", "VIX", "VIXMO")
]
VIX_ROOT = "^VIX"  # the underlying Nautilus gives VIX options
RATE = 0.0425  # GreeksCalculator's default flat rate


def is_index(symbol: str) -> bool:
    """Nautilus names index underlyings with a caret: ^SPX, ^VIX, ^NDX."""
    return symbol.startswith("^")


def underlying_id(symbol: str) -> InstrumentId:
    return InstrumentId.from_str(f"{symbol}.SMART")


def stock_contract(symbol: str) -> dict[str, str]:
    return {"secType": "STK", "symbol": symbol, "exchange": "SMART", "currency": "USD"}


def price(cache: Cache, instrument_id: InstrumentId) -> float | None:
    """Mid of a two-sided quote, else the index price, else the last trade.
    IB sends -1 for a missing side, so anything <= 0 is no price."""
    q = cache.quote_tick(instrument_id)
    if q is not None:
        b, a = q.bid_price.as_double(), q.ask_price.as_double()
        if 0 < b <= a:
            return (b + a) / 2
    ip = cache.index_price(instrument_id)
    if ip is not None and ip.value.as_double() > 0:
        return ip.value.as_double()
    last = cache.price(instrument_id, PriceType.LAST)
    if last is not None and last.as_double() > 0:
        return last.as_double()
    return None


def bid(cache: Cache, instrument_id: InstrumentId) -> float | None:
    q = cache.quote_tick(instrument_id)
    if q is not None and q.bid_price.as_double() > 0:
        return q.bid_price.as_double()
    return None


def held_legs(cache: Cache) -> list[Leg]:
    """Net position per instrument (all strategies, EXTERNAL included).
    ponytail: one instrument split across strategies keeps the average price of
    the side that is net held; exact enough for P&L-of-premium rules."""
    by_id = defaultdict(list)
    for p in cache.positions_open(venue=VENUE):
        by_id[p.instrument_id].append(p)
    legs: list[Leg] = []
    for iid, positions in by_id.items():
        qty = sum(p.signed_qty for p in positions)
        ins = cache.instrument(iid)
        if abs(qty) < 1e-9 or ins is None:
            continue
        held = [p for p in positions if (p.signed_qty > 0) == (qty > 0)]
        avg = sum(p.avg_px_open * abs(p.signed_qty) for p in held) / sum(
            abs(p.signed_qty) for p in held
        )
        if isinstance(ins, OptionContract):
            right = "C" if ins.option_kind == OptionKind.CALL else "P"
            symbol, strike = ins.underlying, float(ins.strike_price)
            expiry = ins.expiration_utc.date()
        elif isinstance(ins, Equity):
            right, symbol, strike, expiry = "", iid.symbol.value, 0.0, None
        else:
            continue  # CFDs, futures, ...: not managed
        legs.append(
            Leg(
                iid,
                symbol,
                right,
                strike,
                expiry,
                qty,
                avg,
                price(cache, iid),
                float(ins.multiplier),
            )
        )
    return legs


# ------------------------------------------------------------------ greeks


@dataclass(frozen=True)
class Greeks:
    """Per share of one contract (multiply by qty x multiplier for a position)."""

    delta: float
    gamma: float
    theta: float  # $/day
    vega: float  # $ per vol point
    underlying: float
    vol: float
    price: float


def _ok(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def years_to(ins: OptionContract, today: date) -> float:
    return max((ins.expiration_utc.date() - today).days, 1) / 365.25


def index_forward(cache: Cache, ins: OptionContract) -> float | None:
    """VIX options price off the VIX future of their expiry: F = C - P + K at the
    same strike (r ~ 0 over a month)."""
    for other in cache.instruments(VENUE, underlying=ins.underlying):
        if (
            isinstance(other, OptionContract)
            and other.expiration_ns == ins.expiration_ns
            and other.strike_price == ins.strike_price
            and other.option_kind != ins.option_kind
        ):
            mine, theirs = price(cache, ins.id), price(cache, other.id)
            if mine is None or theirs is None:
                return None
            is_call = ins.option_kind == OptionKind.CALL
            call, put = (mine, theirs) if is_call else (theirs, mine)
            return call - put + float(ins.strike_price)
    return None


def option_greeks(
    cache: Cache,
    calc: GreeksCalculator,
    ins: OptionContract,
    today: date,
    forward: float | None = None,
) -> Greeks | None:
    """Nautilus GreeksCalculator greeks implied from the option's mid. VIX options
    price off their expiry's VIX future, not the index: Black-76 (carry 0) on the
    parity forward (or `forward`), through the same Rust pricer."""
    if ins.underlying != VIX_ROOT:
        g = calc.instrument_greeks(ins.id)
        if g is None or not _ok(g.delta, g.gamma, g.theta, g.vega):
            return None
        return Greeks(
            g.delta, g.gamma, g.theta, g.vega, g.underlying_price, g.vol, g.price
        )
    opt, und = price(cache, ins.id), forward or index_forward(cache, ins)
    if opt is None or und is None:
        return None
    is_call = ins.option_kind == OptionKind.CALL
    strike = float(ins.strike_price)
    g = imply_vol_and_greeks(und, RATE, 0.0, is_call, strike, years_to(ins, today), opt)
    if not _ok(g.delta, g.gamma, g.theta, g.vega):
        return None
    return Greeks(g.delta, g.gamma, g.theta, g.vega, und, g.vol, g.price)


# ------------------------------------------------------------------- book


@dataclass
class SymbolRisk:
    beta: float = 1.0
    plain_dollars: float = 0.0  # delta $ without beta
    beta_dollars: float = 0.0
    delta_shares: float = 0.0
    gamma: float = 0.0  # share delta change per $1 underlying move
    theta: float = 0.0
    vega: float = 0.0
    price: float = 0.0


@dataclass
class Book:
    """The account in SPX terms. VIX options are left out: they move against SPX
    and price off VIX futures, so delta x beta isn't SPX exposure."""

    spx: float
    beta_dollars: float = 0.0
    plain_dollars: float = 0.0
    gamma: float = 0.0  # SPX delta change per SPX point
    theta: float = 0.0
    vega: float = 0.0
    by_symbol: dict[str, SymbolRisk] = field(default_factory=dict)
    greeks: dict[InstrumentId, Greeks] = field(default_factory=dict)
    missing: list[InstrumentId] = field(default_factory=list)

    @property
    def spx_delta(self) -> float:
        return self.beta_dollars / self.spx if self.spx else 0.0


def build_book(
    cache: Cache,
    calc: GreeksCalculator,
    legs: list[Leg],
    betas: dict[str, float],
    spx: float,
    today: date,
) -> Book:
    book = Book(spx)
    for leg in legs:
        if leg.symbol == VIX_ROOT:
            continue
        beta = 1.0 if is_index(leg.symbol) else betas.get(leg.symbol, 1.0)
        row = book.by_symbol.setdefault(leg.symbol, SymbolRisk(beta=beta))
        if not leg.is_option:
            if leg.mid is None:
                book.missing.append(leg.instrument_id)
                continue
            delta, gamma, theta, vega, und = leg.qty, 0.0, 0.0, 0.0, leg.mid
        else:
            g = option_greeks(cache, calc, cache.instrument(leg.instrument_id), today)
            if g is None:
                book.missing.append(leg.instrument_id)
                continue
            book.greeks[leg.instrument_id] = g
            size = leg.qty * leg.multiplier
            delta, gamma, und = g.delta * size, g.gamma * size, g.underlying
            theta, vega = g.theta * size, g.vega * size
        row.price = und
        row.delta_shares += delta
        row.gamma += gamma
        row.theta += theta
        row.vega += vega
        row.plain_dollars += delta * und
        row.beta_dollars += delta * und * beta
        book.plain_dollars += delta * und
        book.beta_dollars += delta * und * beta
        book.gamma += gamma * (beta * und / spx) ** 2 if spx else 0.0
        book.theta += theta
        book.vega += vega
    return book


def stress_pnl(
    cache: Cache, legs: list[Leg], book: Book, move: float, use_beta: bool, today: date
) -> float:
    """P&L if SPX moves by `move` and each underlying by beta x move (or 1 x move),
    options repriced with Nautilus' Black-Scholes at their current implied vol."""
    total = 0.0
    for leg in legs:
        row = book.by_symbol.get(leg.symbol)
        if row is None:
            continue
        m = max(move * (row.beta if use_beta else 1.0), -0.99)
        if not leg.is_option:
            if leg.mid is not None:
                total += leg.qty * leg.mid * m
            continue
        g = book.greeks.get(leg.instrument_id)
        if g is None or not _ok(g.vol) or g.vol <= 0:
            continue
        shocked = black_scholes_greeks(
            g.underlying * (1 + m),
            RATE,
            RATE,
            g.vol,
            leg.right == "C",
            leg.strike,
            years_to(cache.instrument(leg.instrument_id), today),
        )
        total += leg.qty * leg.multiplier * (shocked.price - g.price)
    return total
