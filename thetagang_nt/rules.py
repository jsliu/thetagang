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


def breakeven_move(theta: float, gamma: float, spx: float) -> float | None:
    """The daily SPX move (a fraction) where gamma costs exactly a day's theta.
    Only meaningful when earning theta and short gamma; else None."""
    if theta <= 0 or gamma >= 0 or spx <= 0:
        return None
    return (2 * theta / -gamma) ** 0.5 / spx


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


def symbol_adjustment(
    legs: list[Leg], shares: float, unit: dict[Any, Unit]
) -> tuple[float, str, tuple[float, float, float, float]]:
    """The trade toward `shares` of delta in one symbol: (share delta traded,
    description, gamma/theta/vega/volga added). Resizes the position with the most
    delta; stock-only symbols trade shares, never selling shares that cover
    short calls; never flips a position."""
    adjusters = option_adjusters(legs, unit)
    if adjusters:
        main = max(adjusters, key=lambda a: a.exposure)
        units, text = size_adjustment(main, shares)
        return (
            units * main.delta,
            text,
            (
                units * main.gamma,
                units * main.theta,
                units * main.vega,
                units * main.volga,
            ),
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
        return traded, text, (0.0, 0.0, 0.0, 0.0)
    _, _, mixed = split_option_legs(legs)
    reason = "none (calendar/diagonal legs)" if mixed else "none (delta < 0.01)"
    return 0.0, reason, (0.0, 0.0, 0.0, 0.0)


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


def position_actions(
    legs: list[Leg],
    und: float | None,
    rw: RollWhen,
    debit_take_profit: float,
    alert_delta: float,
    deltas: dict[Any, float],
    today: date,
    no_trading: bool = False,
) -> list[Action]:
    """The roll/close/keep decision for every option leg or vertical of one
    symbol, by the same rules the strategy trades with. A roll can still fail
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

    pairs, unpaired, mixed = split_option_legs(legs)
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
            elif dte <= rw.dte:
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
                )
            )
    for leg in unpaired:
        dte = leg.dte(today)
        if leg.right in mixed:
            out.append(
                Action(
                    "calendar/diagonal",
                    option_label(leg),
                    symbol,
                    leg.right,
                    leg.qty,
                    dte,
                    None,
                    None,
                    "",
                    "by hand (calendar/diagonal)",
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
                )
            )
        else:
            delta = deltas.get(leg.instrument_id)
            pnl = (
                (leg.mid - leg.avg_px) / leg.avg_px
                if leg.mid is not None and leg.avg_px
                else None
            )
            alert = delta is not None and abs(delta) >= alert_delta
            out.append(
                Action(
                    "long",
                    option_label(leg),
                    symbol,
                    leg.right,
                    leg.qty,
                    dte,
                    pnl,
                    None,
                    "",
                    f"roll or close by hand (|delta| >= {alert_delta:g})"
                    if alert
                    else "hold",
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
