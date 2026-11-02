"""Pure decision rules ported from thetagang's discretionary strategy.

No Nautilus objects here: the strategy and monitor turn positions into `Leg`s and
ask these functions what to do, so every rule is unit-testable on plain data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from collections import defaultdict
from typing import Any

import numpy as np

# right, sign of the main leg, (strike offset direction x width, sign) of the 2nd leg
STRUCTURES: dict[str, tuple[str, int, tuple[int, int] | None]] = {
    "long_call": ("C", +1, None),
    "short_call": ("C", -1, None),
    "long_put": ("P", +1, None),
    "short_put": ("P", -1, None),
    "put_credit_spread": ("P", -1, (-1, +1)),  # short put, long put below
    "call_credit_spread": ("C", -1, (+1, +1)),  # short call, long call above
    "put_debit_spread": ("P", +1, (-1, -1)),  # long put, short put below
    "call_debit_spread": ("C", +1, (+1, -1)),  # long call, short call above
}


@dataclass(frozen=True)
class Leg:
    """One net position: shares (right == "") or an option, per-share prices."""

    instrument_id: Any
    symbol: str  # underlying symbol
    right: str  # "C", "P" or "" for stock
    strike: float
    expiry: date | None
    qty: float  # signed: contracts or shares
    avg_px: float  # cost basis per share (premium for options)
    mid: float | None  # current price per share, None without a quote
    multiplier: float = 100.0

    @property
    def is_option(self) -> bool:
        return bool(self.right)

    def dte(self, today: date) -> int:
        return (self.expiry - today).days if self.expiry else 0


# ------------------------------------------------------------------ pairing


def is_debit_vertical(short: Leg, long: Leg) -> bool:
    """The long leg is closer to the money than the short (paid a debit)."""
    if short.right == "P":
        return long.strike > short.strike
    return long.strike < short.strike


def find_verticals(options: list[Leg], *, debit: bool) -> list[tuple[Leg, Leg]]:
    """(short, long) verticals: same right, expiry and quantity."""
    longs = [p for p in options if p.qty > 0]
    pairs: list[tuple[Leg, Leg]] = []
    for short in (p for p in options if p.qty < 0):
        candidates = [
            p
            for p in longs
            if p.right == short.right
            and p.expiry == short.expiry
            and math.isclose(p.qty, -short.qty)
            and p.strike != short.strike
            and is_debit_vertical(short, p) == debit
        ]
        if candidates:
            long = min(candidates, key=lambda p: abs(p.strike - short.strike))
            longs.remove(long)
            pairs.append((short, long))
    return pairs


def short_call_shares(options: list[Leg]) -> float:
    return sum(-p.qty * p.multiplier for p in options if p.right == "C" and p.qty < 0)


def split_option_legs(
    legs: list[Leg], groups: dict[str, tuple[str, ...]] | None = None
) -> tuple[list[tuple[Leg, Leg]], list[Leg], set[str]]:
    """One symbol's legs as (verticals, options not in one, rights whose leftovers
    mix shorts and longs = calendar/diagonal legs). Credit verticals pair first;
    short calls covered by shares are never paired or treated as diagonal. Legs
    of a by-hand structure (hand_structures) are in none of them."""
    held = {id(p) for s in hand_structures(legs, groups) for p in s.legs}
    return _split([p for p in legs if id(p) not in held])


def _split(legs: list[Leg]) -> tuple[list[tuple[Leg, Leg]], list[Leg], set[str]]:
    options = [p for p in legs if p.is_option]
    shares = sum(p.qty for p in legs if not p.is_option)
    pairs = find_verticals(options, debit=False)
    credit_legs = {id(p) for pair in pairs for p in pair}
    leftover = [p for p in options if id(p) not in credit_legs]
    if shares >= short_call_shares(leftover) > 0:
        leftover = [p for p in leftover if p.right != "C"]
    pairs += find_verticals(leftover, debit=True)
    paired = {id(p) for pair in pairs for p in pair}
    unpaired = [p for p in options if id(p) not in paired]
    mixed = {
        right
        for right in {p.right for p in unpaired}
        if any(p.qty < 0 and p.right == right for p in unpaired)
        and any(p.qty > 0 and p.right == right for p in unpaired)
    }
    if "C" in mixed and shares >= short_call_shares(unpaired):
        mixed.discard("C")
    return pairs, unpaired, mixed


# ------------------------------------------------------------- structures
# One symbol's option legs as the positions they form. Recognisers run in order,
# each taking its legs out of the pool; add a structure with one more recogniser.
# Phase 1: the bot manages verticals and single legs only; the others are shown
# as one combined position and left to you ("by hand").

BY_HAND = {"long butterfly", "short butterfly", "long straddle", "long strangle"}


@dataclass(frozen=True)
class Structure:
    kind: str  # long call, credit spread, long butterfly, calendar, group, ...
    legs: tuple[Leg, ...]
    name: str = ""  # a manual group's name


def _groups(options: list[Leg], groups: dict[str, tuple[str, ...]]) -> list[Structure]:
    """[[strategy.structures]]: legs you grouped by hand, by contract label; a
    group whose legs aren't all held is ignored."""
    by_label = {option_label(p): p for p in options}
    return [
        Structure("group", tuple(by_label[x] for x in labels), name)
        for name, labels in groups.items()
        if labels and all(x in by_label for x in labels)
    ]


def _butterflies(options: list[Leg]) -> list[Structure]:
    """Same right and expiry: +n / -2n / +n (long) or -n / +2n / -n (short) at
    three strikes, the body between the wings."""
    found: list[Structure] = []
    used: set[int] = set()
    by_key: dict[tuple[str, date | None], list[Leg]] = defaultdict(list)
    for p in options:
        by_key[(p.right, p.expiry)].append(p)
    for same in by_key.values():
        for body in sorted(same, key=lambda p: p.strike):
            wing = -body.qty / 2
            if id(body) in used or not wing:
                continue
            free = [p for p in same if id(p) not in used and math.isclose(p.qty, wing)]
            lower = [p for p in free if p.strike < body.strike]
            upper = [p for p in free if p.strike > body.strike]
            if lower and upper:
                lo = max(lower, key=lambda p: p.strike)
                hi = min(upper, key=lambda p: p.strike)
                used |= {id(lo), id(body), id(hi)}
                kind = "long butterfly" if wing > 0 else "short butterfly"
                found.append(Structure(kind, (lo, body, hi)))
    return found


def _straddles(options: list[Leg]) -> list[Structure]:
    """A long put and a long call, same expiry and size: same strike = straddle."""
    found: list[Structure] = []
    calls = [p for p in options if p.right == "C" and p.qty > 0]
    for put in (p for p in options if p.right == "P" and p.qty > 0):
        match = [
            c for c in calls if c.expiry == put.expiry and math.isclose(c.qty, put.qty)
        ]
        if match:
            call = min(match, key=lambda c: abs(c.strike - put.strike))
            calls.remove(call)
            same = math.isclose(call.strike, put.strike)
            found.append(
                Structure("long straddle" if same else "long strangle", (put, call))
            )
    return found


def outside_groups(
    legs: list[Leg], groups: dict[str, tuple[str, ...]] | None
) -> list[Leg]:
    """`legs` without the manual groups' legs (the trade plan never resizes them)."""
    held = {id(p) for s in _groups(legs, groups or {}) for p in s.legs}
    return [p for p in legs if id(p) not in held]


def _take(pool: list[Leg], found: list[Structure]) -> list[Structure]:
    used = {id(p) for s in found for p in s.legs}
    pool[:] = [p for p in pool if id(p) not in used]
    return found


def hand_structures(
    legs: list[Leg], groups: dict[str, tuple[str, ...]] | None = None
) -> list[Structure]:
    """The structures the bot leaves to you: manual groups first, then
    butterflies, then long straddles/strangles of longs no vertical or
    calendar/diagonal holds."""
    pool = [p for p in legs if p.is_option]
    out = _take(pool, _groups(pool, groups or {}))
    out += _take(pool, _butterflies(pool))
    stock = [p for p in legs if not p.is_option]
    _, unpaired, mixed = _split(pool + stock)
    out += _take(pool, _straddles([p for p in unpaired if p.right not in mixed]))
    return out


def find_structures(
    legs: list[Leg], groups: dict[str, tuple[str, ...]] | None = None
) -> list[Structure]:
    """Every option leg of one symbol in exactly one structure: the by-hand ones,
    verticals, calendars/diagonals (a short and a long of one right and size at
    two expiries), then single legs."""
    out = hand_structures(legs, groups)
    pairs, unpaired, mixed = split_option_legs(legs, groups)
    for short, long in pairs:
        kind = "debit spread" if is_debit_vertical(short, long) else "credit spread"
        out.append(Structure(kind, (long, short)))
    pool = list(unpaired)
    for short in [p for p in unpaired if p.qty < 0 and p.right in mixed]:
        match = [
            p
            for p in pool
            if p.qty > 0
            and p.right == short.right
            and p.expiry != short.expiry
            and math.isclose(p.qty, -short.qty)
        ]
        if match and short in pool:
            long = min(match, key=lambda p: abs(p.strike - short.strike))
            same = math.isclose(long.strike, short.strike)
            out.append(Structure("calendar" if same else "diagonal", (long, short)))
            pool.remove(long)
            pool.remove(short)
    names = {"C": "call", "P": "put"}
    for p in pool:
        side = "long" if p.qty > 0 else "short"
        out.append(Structure(f"{side} {names[p.right]}", (p,)))
    return out


def structure_label(s: Structure) -> str:
    """ "AA 20261218 +1x35P -2x40P +1x45P"; legs at several expiries carry theirs."""
    legs = sorted(s.legs, key=lambda p: (p.expiry or date.min, p.strike, p.right))
    expiries = {p.expiry for p in legs}
    symbol = legs[0].symbol.lstrip("^")

    def leg(p: Leg) -> str:
        at = "" if len(expiries) == 1 else f"{p.expiry:%Y%m%d} "
        return f"{at}{p.qty:+g}x{p.strike:g}{p.right}"

    head = f"{symbol} {legs[0].expiry:%Y%m%d}" if len(expiries) == 1 else symbol
    return " ".join([head] + [leg(p) for p in legs])


@dataclass(frozen=True)
class StructureFigures:
    """One structure's combined figures, in $ for the whole position."""

    qty: float  # structures held: the smallest leg size
    dte: int  # to the nearest expiry
    cost: float  # paid (+) or received (-)
    value: float | None  # now, at mid
    pnl: float | None
    delta: float | None  # shares
    gamma: float | None  # shares per $1 of the underlying
    theta: float | None  # $ per day
    vega: float | None  # $ per vol point
    max_profit: float | None  # at expiry; inf: unlimited; None: several expiries
    max_loss: float | None  # negative; -inf: unlimited
    breakevens: tuple[float, ...] = ()  # underlying prices at expiry


def _intrinsic(right: str, strike: float, price: float) -> float:
    return max(price - strike, 0.0) if right == "C" else max(strike - price, 0.0)


def structure_figures(
    s: Structure, greeks: dict[Any, Any], today: date
) -> StructureFigures:
    """`greeks`: instrument id -> per-share greeks (delta, gamma, theta, vega).
    Max profit/loss and breakevens come from the payoff at expiry, so only for
    legs sharing one expiry."""
    legs = s.legs
    size = [p.qty * p.multiplier for p in legs]
    cost = sum(n * p.avg_px for n, p in zip(size, legs))
    value = (
        None
        if any(p.mid is None for p in legs)
        else sum(n * p.mid for n, p in zip(size, legs))  # type: ignore[operator]
    )
    g = [greeks.get(p.instrument_id) for p in legs]

    def total(attr: str) -> float | None:
        if any(x is None for x in g):
            return None
        return sum(n * getattr(x, attr) for n, x in zip(size, g))

    hi: float | None = None
    lo: float | None = None
    evens: list[float] = []
    if len({p.expiry for p in legs}) == 1:
        points = [0.0] + sorted({p.strike for p in legs})

        def pnl_at(price: float) -> float:
            return (
                sum(
                    n * _intrinsic(p.right, p.strike, price) for n, p in zip(size, legs)
                )
                - cost
            )

        at = [pnl_at(x) for x in points]
        slope = sum(
            n for n, p in zip(size, legs) if p.right == "C"
        )  # past the top strike
        hi, lo = max(at), min(at)
        if slope > 0:
            hi = math.inf
        elif slope < 0:
            lo = -math.inf
        for (x0, y0), (x1, y1) in zip(zip(points, at), zip(points[1:], at[1:])):
            if y0 == 0 and x0 > 0:
                evens.append(x0)
            elif y0 * y1 < 0:
                evens.append(x0 + (x1 - x0) * y0 / (y0 - y1))
        if at[-1] == 0:
            evens.append(points[-1])
        elif slope and at[-1] * slope < 0:
            evens.append(points[-1] - at[-1] / slope)
    return StructureFigures(
        qty=min(abs(p.qty) for p in legs),
        dte=min(p.dte(today) for p in legs),
        cost=cost,
        value=value,
        pnl=None if value is None else value - cost,
        delta=total("delta"),
        gamma=total("gamma"),
        theta=total("theta"),
        vega=total("vega"),
        max_profit=hi,
        max_loss=lo,
        breakevens=tuple(evens),
    )


# ------------------------------------------------------------------- P&L


def short_pnl(leg: Leg) -> float | None:
    """Share of the premium captured on a short option; None without a quote."""
    if leg.mid is None or leg.avg_px <= 0:
        return None
    return (leg.avg_px - leg.mid) / leg.avg_px


def credit_pnl(short: Leg, long: Leg) -> float | None:
    """Share of the net credit captured; None without a credit or quotes."""
    credit = short.avg_px - long.avg_px
    if credit <= 0 or short.mid is None or long.mid is None:
        return None
    return (credit - (short.mid - long.mid)) / credit


def debit_pnl(short: Leg, long: Leg) -> float | None:
    """P&L as a share of the net debit paid; None without a debit or quotes."""
    debit = long.avg_px - short.avg_px
    if debit <= 0 or short.mid is None or long.mid is None:
        return None
    return ((long.mid - short.mid) - debit) / debit


def debit_take_profit_value(short: Leg, long: Leg, fraction: float) -> float:
    """Spread value that makes `fraction` of the max profit (width - debit)."""
    width = abs(long.strike - short.strike)
    debit = long.avg_px - short.avg_px
    if not 0 < debit < width:
        return fraction * width
    return debit + fraction * (width - debit)


# ----------------------------------------------------------- roll rules


@dataclass(frozen=True)
class RollWhen:
    pnl: float = 0.9
    dte: int = 10
    min_pnl: float = 0.0
    max_dte: int | None = None
    close_at_pnl: float = 1.0
    close_if_unable_to_roll: bool = False
    itm: dict[str, bool] = field(default_factory=lambda: {"C": True, "P": True})
    always_when_itm: dict[str, bool] = field(
        default_factory=lambda: {"C": False, "P": False}
    )
    credit_only: dict[str, bool] = field(
        default_factory=lambda: {"C": False, "P": False}
    )


def should_roll(rw: RollWhen, right: str, itm: bool, dte: int, pnl: float) -> bool:
    """[credit.roll_when]: always_when_itm, itm, max_dte, then dte/min_pnl or pnl."""
    if itm and rw.always_when_itm.get(right, False):
        return True
    if itm and not rw.itm.get(right, True):
        return False
    if rw.max_dte and dte > rw.max_dte:
        return False
    return (dte <= rw.dte and pnl >= rw.min_pnl) or pnl >= rw.pnl


def should_close(rw: RollWhen, pnl: float) -> bool:
    return bool(rw.close_at_pnl) and pnl > rw.close_at_pnl


def close_if_unable_to_roll(rw: RollWhen, dte: int, pnl: float) -> bool:
    return bool(rw.close_if_unable_to_roll and rw.max_dte and dte <= rw.max_dte) and (
        pnl > 0
    )


def is_itm(right: str, strike: float, underlying: float) -> bool:
    return underlying > strike if right == "C" else underlying < strike


def roll_strike_limit(
    right: str, strike: float, captured: float, itm: bool, configured: float | None
) -> float:
    """The new strike may move against you by at most the premium captured per
    share; ITM only rolls to the same or a safer strike. Puts: upper bound,
    calls: lower bound. A configured strike_limit still applies."""
    if right == "P":
        limit = max(strike, strike + captured)
        if itm:
            limit = min(limit, strike)
        if configured:
            limit = min(limit, configured)
    else:
        limit = min(strike, strike - captured)
        if itm:
            limit = max(limit, strike)
        if configured:
            limit = max(limit, configured)
    return round(limit, 2)


# ------------------------------------------------------- chain selection


@dataclass(frozen=True)
class Candidate:
    instrument_id: Any
    strike: float
    expiry: date
    dte: int
    delta: float | None  # per share, signed
    price: float | None  # mid


def valid_strike(
    right: str, strike: float, strike_limit: float | None, underlying: float
) -> bool:
    if right == "P":
        return strike <= (strike_limit or underlying * 1.05)
    return strike >= (strike_limit or underlying * 0.95)


def scan_strikes(strikes: list[float], underlying: float, count: int) -> list[float]:
    """The `count` strikes to quote for a delta pick: the 4 x count strikes nearest
    the money, thinned evenly, so the scan spans deltas on both sides of the
    money however fine the strike grid (SPY: $1 apart), sorted by strike."""
    near = sorted(strikes, key=lambda k: abs(k - underlying))[: 4 * count]
    near.sort()
    step = max(1, math.ceil(len(near) / count))
    return near[::step]


def pick_contract(
    candidates: list[Candidate],
    right: str,
    underlying: float,
    target_delta: float,
    min_price: float,
) -> Candidate | None:
    """Nearest expiry, then the largest |delta| at or below the target; puts may
    not cost more than the stock. If no delta fits and min_price is set, fall back
    to the smallest |delta| above the target (as the old scanner did)."""

    def price_ok(c: Candidate) -> bool:
        return (
            c.price is not None
            and c.price > min_price
            and (right == "C" or c.strike <= c.price + underlying)
        )

    priced = [c for c in candidates if price_ok(c) and c.delta is not None]
    fits = [c for c in priced if abs(c.delta) <= target_delta]
    if fits:
        return min(fits, key=lambda c: (c.dte, -abs(c.delta)))
    if math.isclose(min_price, 0.0) or not priced:
        return None
    return min(priced, key=lambda c: (c.dte, abs(c.delta)))


# ------------------------------------------------------------- sizing


@dataclass
class Plan:
    """One configured trade, priced: units sized to bring SPX delta to 0."""

    weight: float
    max_contracts: int | None
    unit_spx: float = 0.0  # SPX delta added per unit
    priced: bool = False
    units: int = 0
    note: str = ""


def size_plans(plans: list[Plan], need: float) -> None:
    """Split `need` (SPX delta) across priced plans by weight, whole units, never
    the wrong direction, capped by max_contracts."""
    usable = [p for p in plans if p.priced and p.unit_spx and p.unit_spx * need > 0]
    for p in plans:
        if p.priced and p.unit_spx and p not in usable:
            p.note = p.note or "would move SPX delta the wrong way"
    total = sum(p.weight for p in usable)
    for p in usable:
        p.units = max(round(need * p.weight / total / p.unit_spx), 0)
        if p.max_contracts is not None and p.units > p.max_contracts:
            p.units = p.max_contracts
            p.note = p.note or f"capped at max_contracts={p.max_contracts}"
        if p.units == 0:
            p.note = p.note or "less than one contract needed"


# ------------------------------------------------------------- VIX hedge


def vix_hedge_weight(allocations: list[Any], vixmo: float) -> float:
    """First allocation with lower_bound <= VIXMO < upper_bound (unset = open)."""
    for a in allocations:
        if (a.lower_bound is None or a.lower_bound <= vixmo) and (
            a.upper_bound is None or vixmo < a.upper_bound
        ):
            return a.weight
    return 0.0


# ----------------------------------------------------------------- risk


def compute_beta(
    closes: dict[Any, float], index_closes: dict[Any, float]
) -> float | None:
    """Beta of daily log returns vs the index; None below 20 shared dates."""
    dates = sorted(set(closes) & set(index_closes))
    if len(dates) < 20:
        return None
    a = np.diff(np.log([closes[d] for d in dates]))
    b = np.diff(np.log([index_closes[d] for d in dates]))
    return float(np.cov(a, b)[0][1] / np.var(b, ddof=1))


def realized_move(closes: dict[Any, float], days: int = 21) -> float | None:
    """The typical (one standard deviation) daily move, a fraction, over the last
    `days` daily log returns; None with fewer closes."""
    recent = [closes[d] for d in sorted(closes)][-(days + 1) :]
    if len(recent) < days + 1:
        return None
    return float(np.std(np.diff(np.log(recent)), ddof=1))


TYPICAL_SPX_DAY = 0.01  # a typical daily SPX move
DRIFT_BANDS = 2.0  # beyond this many bands: too far from neutral
NEUTRALITY = ("neutral", "drifting", "off neutral")


def delta_band_dollars(
    theta: float, net_liq: float, theta_days: float, nav_pct: float
) -> float:
    """|Beta delta $| that still counts as SPX delta neutral: a typical 1% SPX
    day may cost at most `theta_days` days of theta, capped at `nav_pct` of net
    liq. Without theta (no options, or greeks missing) only the cap applies."""
    cap = nav_pct * net_liq
    if not theta:
        return cap
    return min(theta_days * abs(theta) / TYPICAL_SPX_DAY, cap)


def expected_move(vix: float | None) -> float:
    """The expected (one standard deviation) daily SPX move, a fraction: VIX
    spread over 252 trading days; TYPICAL_SPX_DAY without a VIX price."""
    return vix / 100 / 252**0.5 if vix and vix > 0 else TYPICAL_SPX_DAY


def gamma_pnl(gamma: float, spx: float, move: float = TYPICAL_SPX_DAY) -> float:
    """P&L from gamma alone when SPX moves `move` either way: 1/2 x gamma x
    (move x SPX)^2, gamma in SPX delta per SPX point. Negative when short gamma:
    what staying delta neutral costs on such a day."""
    return 0.5 * gamma * (move * spx) ** 2


def breakeven_move(theta: float, gamma: float, spx: float) -> tuple[str, float] | None:
    """The daily SPX move (a fraction) where gamma P&L offsets a day's theta:
    ("max", m) when short gamma and earning theta (bigger days lose money),
    ("min", m) when long gamma and paying theta (smaller days lose money);
    None when theta and gamma don't offset (every day wins, or every day loses)."""
    if spx <= 0 or not theta or not gamma or (theta > 0) == (gamma > 0):
        return None
    return ("max" if theta > 0 else "min"), (2 * abs(theta) / abs(gamma)) ** 0.5 / spx


def neutrality(beta_dollars: float, band: float) -> str:
    """'neutral' within the band, 'drifting' up to DRIFT_BANDS bands, else
    'off neutral'."""
    if abs(beta_dollars) <= band:
        return NEUTRALITY[0]
    return NEUTRALITY[1] if abs(beta_dollars) <= DRIFT_BANDS * band else NEUTRALITY[2]


# ------------------------------------------- SPX delta = 0 proposals (display)


def zero_delta_targets(
    delta_dollars: dict[str, float], betas: dict[str, float]
) -> dict[str, float]:
    """Target delta $ (no beta) per symbol so that both totals are zero:
    sum(delta $) = 0 and sum(beta x delta $) = 0. Smallest proportional change
    (weighted least squares); symbols with zero delta $ never change."""
    symbols = list(delta_dollars)
    if not symbols:
        return {}
    d = np.array([delta_dollars[s] for s in symbols], dtype=float)
    b = np.array([betas[s] for s in symbols], dtype=float)
    w = np.abs(d)
    a = np.vstack([np.ones_like(d), b])
    residual = -(a @ d)
    change = w * (a.T @ (np.linalg.pinv((a * w) @ a.T) @ residual))
    return {s: float(x) for s, x in zip(symbols, d + change)}


def side_weight(beta_dollars: float, gross: float) -> tuple[float, str]:
    """Share of gross |beta delta $| and the market side ("" when ~0)."""
    if abs(beta_dollars) < 0.5 or not gross:
        return 0.0, ""
    return abs(beta_dollars) / gross, "long" if beta_dollars > 0 else "short"


# per contract: share delta, SPX gamma, theta, vega, volga
Unit = tuple[float, float, float, float, float]


def volga(
    vega: float, spot: float, strike: float, years: float, vol: float, carry: float
) -> float:
    """How much vega ($ per vol point) changes per vol point: vega x d1 x d2 / sigma
    (Black-Scholes with cost of carry `carry`); 0 without usable inputs."""
    values = (vega, spot, strike, years, vol)
    if not all(math.isfinite(x) for x in values) or min(spot, strike, years, vol) <= 0:
        return 0.0
    sd = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (carry + vol * vol / 2) * years) / sd
    return vega * d1 * (d1 - sd) / vol / 100


@dataclass(frozen=True)
class Adjuster:
    """A position the proposal may resize: one option, or a whole vertical.
    Greeks per unit: +1 = buy one contract / open one more spread."""

    exposure: float
    name: str
    is_spread: bool
    delta: float
    gamma: float
    theta: float
    vega: float
    held: float
    close_sign: int  # direction of units that close: -1 sell/close, +1 buy back
    legs: tuple[Leg, ...] = ()
    volga: float = 0.0


def leg_name(leg: Leg) -> str:
    iid = leg.instrument_id
    return iid.symbol.value if hasattr(iid, "symbol") else str(iid)


def option_label(leg: Leg) -> str:
    """The old bot's contract label: "EWQ 20261120 43C"; the symbol for stock."""
    symbol = leg.symbol.lstrip("^")
    if not leg.is_option:
        return symbol
    expiry = leg.expiry.strftime("%Y%m%d") if leg.expiry else ""
    return f"{symbol} {expiry} {leg.strike:g}{leg.right}"


TRADE_FORMAT = "buy|sell QTY SYMBOL [YYYYMMDD STRIKE(C|P)[/STRIKE(C|P)]]"


@dataclass(frozen=True)
class TradeSpec:
    """An ad-hoc trade typed for pre-trade: QTY shares, or QTY units of an option
    or a vertical (legs: (right, strike, ratio); the first leg on the trade's side)."""

    text: str
    side: int  # +1 buy, -1 sell
    qty: int
    symbol: str
    expiry: date | None = None
    legs: tuple[tuple[str, float, int], ...] = ()


def yyyymmdd(text: str) -> date:
    if len(text) != 8 or not text.isdigit():
        raise ValueError(text)
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def parse_trade(text: str) -> TradeSpec:
    """'buy 100 MS', 'sell 2 AA 20261218 40P', 'buy 1 AA 20261218 40P/23P' (buy
    the 40 put, sell the 23 put: one combo); '40/23P' shares the right."""
    words = text.split()
    bad = ValueError(f"{text!r}: expected {TRADE_FORMAT!r}")
    if len(words) not in (3, 5) or words[0].lower() not in ("buy", "sell"):
        raise bad
    try:
        qty = int(words[1])
        expiry = None if len(words) == 3 else yyyymmdd(words[3])
    except ValueError:
        raise bad from None
    if qty < 1:
        raise bad
    side, symbol = (1 if words[0].lower() == "buy" else -1), words[2].upper()
    if expiry is None:
        return TradeSpec(text, side, qty, symbol)
    parts = words[4].upper().split("/")
    if len(parts) > 2 or parts[-1][-1:] not in ("C", "P"):
        raise bad
    legs = []
    for ratio, part in zip((1, -1), parts, strict=False):
        right = part[-1] if part[-1:] in ("C", "P") else parts[-1][-1]
        try:
            strike = float(part.rstrip("CP"))
        except ValueError:
            raise bad from None
        legs.append((right, strike, ratio))
    return TradeSpec(text, side, qty, symbol, expiry, tuple(legs))


def position_rows(legs: list[Leg]) -> list[tuple[Leg, int, int]]:
    """Legs in the old bot's Portfolio positions order (by symbol; stock first,
    then options by expiry, right and strike), each with its row number and a
    0/1 stripe that alternates per symbol, for shading the groups."""
    ordered = sorted(
        legs,
        key=lambda x: (x.symbol, x.is_option, x.expiry or date.min, x.right, x.strike),
    )
    rows, group, last = [], -1, None
    for row, leg in enumerate(ordered):
        if leg.symbol != last:
            group, last = group + 1, leg.symbol
        rows.append((leg, row, group % 2))
    return rows


def spread_name(short: Leg, long: Leg) -> str:
    """The old bot's vertical label: "EWQ 20261120 43/45C"."""
    low, high = sorted((short.strike, long.strike))
    expiry = short.expiry.strftime("%Y%m%d") if short.expiry else ""
    return f"{short.symbol.lstrip('^')} {expiry} {low:g}/{high:g}{short.right}"


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


def option_adjusters(legs: list[Leg], unit: dict[Any, Unit]) -> list[Adjuster]:
    """Verticals resized as whole spreads (both legs need greeks), other options
    one contract at a time; calendar/diagonal legs and |delta| < 0.01 left alone."""
    pairs, unpaired, mixed = split_option_legs(legs)
    found: list[Adjuster] = []
    for short, long in pairs:
        if short.instrument_id not in unit or long.instrument_id not in unit:
            continue
        su, lu = unit[short.instrument_id], unit[long.instrument_id]
        delta, gamma, theta, vega, volga_ = (lv - sv for sv, lv in zip(su, lu))
        if abs(delta) >= 1:
            held = abs(short.qty)
            found.append(
                Adjuster(
                    abs(delta * held),
                    spread_name(short, long),
                    True,
                    delta,
                    gamma,
                    theta,
                    vega,
                    held,
                    -1,
                    (short, long),
                    volga_,
                )
            )
    for leg in unpaired:
        if leg.right in mixed or leg.instrument_id not in unit:
            continue
        delta, gamma, theta, vega, volga_ = unit[leg.instrument_id]
        if abs(delta) >= 1:
            held = abs(leg.qty)
            close_sign = 1 if leg.qty < 0 else -1
            found.append(
                Adjuster(
                    abs(delta * held),
                    leg_name(leg),
                    False,
                    delta,
                    gamma,
                    theta,
                    vega,
                    held,
                    close_sign,
                    (leg,),
                    volga_,
                )
            )
    return found


@dataclass(frozen=True)
class Resize:
    """The trade toward a symbol's SPX delta = 0 target (the positions summary's
    "Net target contracts"): `units` of `combo` (signed legs of one unit:
    shares, one option, or a whole vertical), with the share delta and the
    gamma/theta/vega/volga it adds."""

    symbol: str
    units: int  # >0 buy / open more spreads, <0 sell / close spreads
    text: str
    combo: tuple[tuple[Any, int], ...] = ()  # (instrument, +1/-1) per unit
    delta: float = 0.0
    greeks: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    adds_short: bool = False  # sells more of a short option / opens credit spreads


def symbol_resize(
    symbol: str, legs: list[Leg], shares: float, unit: dict[Any, Unit]
) -> Resize:
    """The trade toward `shares` of delta in one symbol. Resizes the position with
    the most delta; stock-only symbols trade shares, never selling shares that
    cover short calls; never flips a position."""
    adjusters = option_adjusters(legs, unit)
    if adjusters:
        main = max(adjusters, key=lambda a: a.exposure)
        units, text = size_adjustment(main, shares)
        if main.is_spread:  # one more spread: buy the long leg, sell the short
            short, long = main.legs
            combo = ((short.instrument_id, -1), (long.instrument_id, 1))
            adds_short = units > 0 and not is_debit_vertical(short, long)
        else:
            combo = ((main.legs[0].instrument_id, 1),)
            adds_short = units < 0 and main.legs[0].qty < 0
        if units < 0:  # sell / close: every leg the other way
            combo = tuple((iid, -r) for iid, r in combo)
        greeks = (main.gamma, main.theta, main.vega, main.volga)
        return Resize(
            symbol,
            units,
            text,
            combo,
            units * main.delta,
            tuple(units * g for g in greeks),
            adds_short,
        )
    stock = [leg for leg in legs if not leg.is_option]
    if stock:
        traded = stock_trade(shares, stock[0].qty)
        covering = sum(
            -leg.qty * leg.multiplier
            for leg in legs
            if leg.right == "C" and leg.qty < 0
        )
        traded = max(traded, -int(max(stock[0].qty - covering, 0)))
        text = f"{traded:+d} shares" if traded else "0"
        sign = 1 if traded >= 0 else -1
        return Resize(symbol, traded, text, ((stock[0].instrument_id, sign),), traded)
    _, _, mixed = split_option_legs(legs)
    reason = "none (calendar/diagonal legs)" if mixed else "none (delta < 0.01)"
    return Resize(symbol, 0, reason)


def symbol_adjustment(
    legs: list[Leg], shares: float, unit: dict[Any, Unit]
) -> tuple[float, str, tuple[float, float, float, float]]:
    """symbol_resize as the positions summary shows it: (share delta traded,
    description, gamma/theta/vega/volga added)."""
    r = symbol_resize(legs[0].symbol if legs else "", legs, shares, unit)
    return r.delta, r.text, r.greeks


def blocking_missing(missing: list[tuple[Any, str]], exempt: set[str]) -> list[Any]:
    """The positions without greeks that stop the trade plan: (position,
    symbol) pairs, except in the symbols exempted by ignore_missing_greeks."""
    return [position for position, symbol in missing if symbol not in exempt]


def target_resizes(
    legs: list[Leg], book: Any, unit: dict[Any, Unit], skip: set[str]
) -> list[Resize]:
    """The positions summary's SPX delta = 0 targets as trades, one per symbol
    (symbols in `skip`, e.g. no_trading, get none)."""
    plain_now = {s: r.plain_dollars for s, r in book.by_symbol.items()}
    betas = {s: r.beta for s, r in book.by_symbol.items()}
    plain_target = zero_delta_targets(plain_now, betas)
    out = []
    for s, r in book.by_symbol.items():
        if s in skip or r.price <= 1e-9:
            continue
        shares = (plain_target[s] - plain_now[s]) / r.price
        out.append(symbol_resize(s, [x for x in legs if x.symbol == s], shares, unit))
    return out


def weighted_strike(legs: list[Leg], right: str, short: bool) -> float | None:
    """Contract-weighted average strike of the short or long options of a right."""
    chosen = [leg for leg in legs if leg.right == right and (leg.qty < 0) == short]
    total = sum(abs(leg.qty) for leg in chosen)
    return sum(leg.strike * abs(leg.qty) for leg in chosen) / total if total else None


# ------------------------------------------------- what the rules would do


@dataclass(frozen=True)
class Action:
    """One position or vertical, what the strategy's rules would do with it
    this run, and the price that takes profit (per share)."""

    kind: str  # short, long, credit spread, debit spread, calendar/diagonal
    name: str
    symbol: str
    right: str
    qty: float
    dte: int
    pnl: float | None  # share of the premium (credit / debit)
    take_profit: float | None
    take_profit_side: str  # "buy <=" (shorts, credit) or "sell >=" (debit)
    action: str
    breakeven: float | None = None  # underlying price at expiry with no P&L
    pnl_dollars: float | None = None  # unrealized P&L, USD
    legs: tuple[tuple[str, int], ...] = ()  # a spread's (leg name, +1 long / -1 short)


def dollar_pnl(*legs: Leg) -> float | None:
    """Unrealized P&L in USD of these legs together; None if any has no quote."""
    if any(leg.mid is None for leg in legs):
        return None
    return sum((leg.mid - leg.avg_px) * leg.qty * leg.multiplier for leg in legs)


def long_pnl(leg: Leg) -> float | None:
    """A long option's P&L as a share of the premium paid; None without a quote."""
    if leg.mid is None or not leg.avg_px:
        return None
    return (leg.mid - leg.avg_px) / leg.avg_px


def long_take_profit_value(leg: Leg, gain: float) -> float | None:
    """Option price that takes profit on a long: the premium paid x (1 + gain)."""
    return leg.avg_px * (1 + gain) if leg.avg_px > 0 else None


def breakeven(right: str, strike: float, premium: float) -> float:
    """Underlying price at expiry where one option leg makes nothing, long or
    short: the strike plus (call) or minus (put) the premium per share."""
    return strike + premium if right == "C" else strike - premium


def vertical_breakeven(short: Leg, long: Leg) -> float:
    """A vertical's breakeven at expiry: the lower strike plus the net premium
    (calls), the higher strike minus it (puts), debit or credit alike."""
    net = abs(short.avg_px - long.avg_px)
    low, high = sorted((short.strike, long.strike))
    return low + net if short.right == "C" else high - net


def position_actions(
    legs: list[Leg],
    und: float | None,
    rw: RollWhen,
    debit_take_profit: float,
    today: date,
    no_trading: bool = False,
    debit_roll_dte: int = 10,
    long_take_profit_gain: float = 1.0,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> list[Action]:
    """The roll/close/keep decision for every option leg or vertical of one
    symbol, by the same rules the strategy trades with. Standalone longs follow
    the debit spread rules: take profit, else roll once DTE <= debit_roll_dte. A roll can still fail
    when no contract fits; the strategy then closes or leaves it, as logged."""
    out: list[Action] = []
    symbol = legs[0].symbol if legs else ""
    index = symbol.startswith("^")
    tp_frac = rw.close_at_pnl if rw.close_at_pnl < 1 else None

    def decide(right: str, strike: float, dte: int, pnl: float | None) -> str:
        if no_trading:
            return "none (no_trading)"
        if index:
            return "by hand (index option)"
        if pnl is None:
            return "by hand (no credit / no quote)"
        itm = und is not None and is_itm(right, strike, und)
        if should_roll(rw, right, itm, dte, pnl):
            return "roll"
        if should_close(rw, pnl):
            return "close"
        return "keep"

    for s in hand_structures(legs, groups):  # each leg's own figures
        for leg in s.legs:
            out.append(
                Action(
                    s.kind,
                    option_label(leg),
                    symbol,
                    leg.right,
                    leg.qty,
                    leg.dte(today),
                    short_pnl(leg) if leg.qty < 0 else long_pnl(leg),
                    None,
                    "",
                    f"by hand ({s.name or s.kind})",
                    breakeven(leg.right, leg.strike, leg.avg_px),
                    dollar_pnl(leg),
                )
            )
    pairs, unpaired, mixed = split_option_legs(legs, groups)
    for short, long in pairs:
        dte = short.dte(today)
        if is_debit_vertical(short, long):
            pnl = debit_pnl(short, long)
            target = debit_take_profit_value(short, long, debit_take_profit)
            value = (
                None if short.mid is None or long.mid is None else long.mid - short.mid
            )
            if no_trading or index:
                action = "none (no_trading)" if no_trading else "by hand (index option)"
            elif value is None:
                action = "by hand (no quote)"
            elif value >= target:
                action = "close (take-profit)"
            elif dte <= debit_roll_dte:
                action = "roll"
            else:
                action = "keep"
            out.append(
                Action(
                    "debit spread",
                    spread_name(short, long),
                    symbol,
                    short.right,
                    abs(long.qty),
                    dte,
                    pnl,
                    target,
                    "sell >=",
                    action,
                    vertical_breakeven(short, long),
                    dollar_pnl(short, long),
                    ((option_label(long), 1), (option_label(short), -1)),
                )
            )
        else:
            pnl = credit_pnl(short, long)
            credit = short.avg_px - long.avg_px
            tp = credit * (1 - tp_frac) if tp_frac is not None and credit > 0 else None
            out.append(
                Action(
                    "credit spread",
                    spread_name(short, long),
                    symbol,
                    short.right,
                    abs(short.qty),
                    dte,
                    pnl,
                    tp,
                    "buy <=",
                    decide(short.right, short.strike, dte, pnl),
                    vertical_breakeven(short, long),
                    dollar_pnl(short, long),
                    ((option_label(long), 1), (option_label(short), -1)),
                )
            )
    for leg in unpaired:
        dte = leg.dte(today)
        if leg.right in mixed:  # this leg's own figures: the pair has no single one
            out.append(
                Action(
                    "calendar/diagonal",
                    option_label(leg),
                    symbol,
                    leg.right,
                    leg.qty,
                    dte,
                    short_pnl(leg) if leg.qty < 0 else long_pnl(leg),
                    None,
                    "",
                    "by hand (calendar/diagonal)",
                    breakeven(leg.right, leg.strike, leg.avg_px),
                    dollar_pnl(leg),
                )
            )
        elif leg.qty < 0:
            pnl = short_pnl(leg)
            tp = (
                leg.avg_px * (1 - tp_frac)
                if tp_frac is not None and leg.avg_px > 0
                else None
            )
            out.append(
                Action(
                    "short",
                    option_label(leg),
                    symbol,
                    leg.right,
                    leg.qty,
                    dte,
                    pnl,
                    tp,
                    "buy <=",
                    decide(leg.right, leg.strike, dte, pnl),
                    breakeven(leg.right, leg.strike, leg.avg_px),
                    dollar_pnl(leg),
                )
            )
        else:
            pnl = long_pnl(leg)
            tp = long_take_profit_value(leg, long_take_profit_gain)
            if no_trading or index:
                action = "none (no_trading)" if no_trading else "by hand (index option)"
            elif leg.mid is None or tp is None:
                action = "by hand (no cost / no quote)"
            elif leg.mid >= tp:
                action = "close (take-profit)"
            elif dte <= debit_roll_dte:
                action = "roll"
            else:
                action = "keep"
            out.append(
                Action(
                    "long",
                    option_label(leg),
                    symbol,
                    leg.right,
                    leg.qty,
                    dte,
                    pnl,
                    tp,
                    "sell >=",
                    action,
                    breakeven(leg.right, leg.strike, leg.avg_px),
                    dollar_pnl(leg),
                )
            )
    return out


def vix_hedge_action(
    held_calls: float, vix: float | None, vixmo: float | None, cfg: Any
) -> tuple[str, float]:
    """(what the VIX call hedge would do, the allocation weight at VIXMO)."""
    weight = vix_hedge_weight(cfg.allocation, vixmo) if vixmo is not None else 0.0
    if not cfg.enabled:
        return "disabled", weight
    threshold = cfg.close_hedges_when_vix_exceeds
    above = bool(threshold) and vix is not None and vix > threshold
    if held_calls > 0:
        return ("sell all (VIX above threshold)" if above else "keep"), weight
    if above or vix is None:
        return "none (VIX above threshold)" if above else "none (no VIX price)", weight
    return (
        f"buy calls ({weight:.1%} of net liq)" if weight > 0 else "none (weight 0)"
    ), weight
