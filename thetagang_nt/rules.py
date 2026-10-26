"""Pure decision rules ported from thetagang's discretionary strategy.

No Nautilus objects here: the strategy and monitor turn positions into `Leg`s and
ask these functions what to do, so every rule is unit-testable on plain data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
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
    legs: list[Leg],
) -> tuple[list[tuple[Leg, Leg]], list[Leg], set[str]]:
    """One symbol's legs as (verticals, options not in one, rights whose leftovers
    mix shorts and longs = calendar/diagonal legs). Credit verticals pair first;
    short calls covered by shares are never paired or treated as diagonal."""
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
    """[roll_when]: always_when_itm, itm, max_dte, then dte/min_pnl or pnl."""
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
