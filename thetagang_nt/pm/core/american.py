"""American option prices with discrete dividends (vault: option_repricing).

Closed-form approximations, because the monitor prices thousands of scenarios a
cycle on its event loop; a binomial tree in the tests measures their error.
Dividends are escrowed: the stochastic part of the price is S minus the present
value of the dividends inside the option's life.

- Call: Black's pseudo-American, the better of holding to expiry and exercising
  just before each ex-date. With no dividend inside its life, a call is never
  exercised early, so it is the European price.
- Put: Barone-Adesi and Whaley's quadratic approximation on the escrowed price.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

Dividends = tuple[tuple[float, float], ...]  # (years from now, $ per share), ascending
_ROOT_2PI = math.sqrt(2 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _ROOT_2PI


def european(
    und: float, strike: float, years: float, vol: float, call: bool, rate: float
) -> float:
    """Black-Scholes, no dividends (they are escrowed out of `und`)."""
    if years <= 0 or vol <= 0 or und <= 0:
        return max(und - strike, 0.0) if call else max(strike - und, 0.0)
    sd = vol * math.sqrt(years)
    d1 = (math.log(und / strike) + (rate + 0.5 * vol * vol) * years) / sd
    d2 = d1 - sd
    disc = strike * math.exp(-rate * years)
    return (
        und * _cdf(d1) - disc * _cdf(d2) if call else disc * _cdf(-d2) - und * _cdf(-d1)
    )


def _baw_put(und: float, strike: float, years: float, vol: float, rate: float) -> float:
    """Barone-Adesi and Whaley (1987) with cost of carry = rate."""
    euro = european(und, strike, years, vol, False, rate)
    if rate <= 0:
        return euro
    var = vol * vol
    m = 2 * rate / var
    k = 1 - math.exp(-rate * years)
    n1 = m - 1  # N - 1, with N = 2b / var = m
    q1 = (-n1 - math.sqrt(n1 * n1 + 4 * m / k)) / 2
    q_inf = (-n1 - math.sqrt(n1 * n1 + 4 * m)) / 2
    sd = vol * math.sqrt(years)
    s_inf = strike / (1 - 1 / q_inf)
    guess = s_inf + (strike - s_inf) * math.exp(
        (rate * years - 2 * sd) * strike / (strike - s_inf)
    )
    s = guess
    for _ in range(50):
        d1 = (math.log(s / strike) + (rate + 0.5 * var) * years) / sd
        rhs = european(s, strike, years, vol, False, rate) - (1 - _cdf(-d1)) * s / q1
        slope = -_cdf(-d1) * (1 - 1 / q1) - (1 + _pdf(-d1) / sd) / q1
        if abs(strike - s - rhs) / strike < 1e-7:
            break
        s = max((strike - rhs + slope * s) / (1 + slope), 1e-6)
    if und <= s:
        return strike - und
    d1 = (math.log(s / strike) + (rate + 0.5 * var) * years) / sd
    a = -(s / q1) * (1 - _cdf(-d1))
    return max(euro + a * (und / s) ** q1, strike - und)


def american(
    und: float,
    strike: float,
    years: float,
    vol: float,
    call: bool,
    rate: float,
    dividends: Dividends = (),
) -> float:
    """Price per share; intrinsic at or after expiry. `dividends`: those still
    to come, in years from now; any outside the option's life are ignored."""
    intrinsic = max(und - strike, 0.0) if call else max(strike - und, 0.0)
    if years <= 0 or vol <= 0 or und <= 0:
        return intrinsic
    inside = [(t, d) for t, d in dividends if 0 < t < years]
    pv_all = sum(d * math.exp(-rate * t) for t, d in inside)
    escrowed = und - pv_all
    if escrowed <= 0:
        return intrinsic
    if not call:
        value = _baw_put(escrowed, strike, years, vol, rate)
        if value > strike - escrowed + 1e-12:
            return max(value, intrinsic)
        # past the boundary, where the escrowed price overstates exercising now:
        # exercise now at the real price, or just after an ex-date
        best, paid = intrinsic, 0.0
        for t, d in inside:
            paid += d * math.exp(-rate * t)
            best = max(best, strike * math.exp(-rate * t) - und + paid)
        return best
    best = european(escrowed, strike, years, vol, True, rate)
    paid = 0.0  # PV of the dividends before the candidate exercise date
    for t, d in inside:
        # exercise just before this ex-date: earlier dividends gone, not this one
        best = max(best, european(und - paid, strike, t, vol, True, rate))
        paid += d * math.exp(-rate * t)
    return max(best, intrinsic)


def implied_vol(
    price: float,
    und: float,
    strike: float,
    years: float,
    call: bool,
    rate: float,
    dividends: Dividends = (),
) -> float | None:
    """The vol at which american() matches the market price, by bisection; None
    if no vol does (a price at or below intrinsic, or expired)."""
    if price <= 0 or years <= 0:
        return None
    lo, hi = 0.01, 8.0
    if american(und, strike, years, lo, call, rate, dividends) >= price:
        return None
    if american(und, strike, years, hi, call, rate, dividends) <= price:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if american(und, strike, years, mid, call, rate, dividends) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return (lo + hi) / 2


def schedule(
    next_date: date | None,
    next_amount: float | None,
    next_12m: float | None,
    today: date,
    years: int = 3,
) -> tuple[tuple[date, float], ...]:
    """The dividends to come. IB gives one dated dividend and a 12-month total;
    later ones repeat it, as many a year as the total implies (an estimate)."""
    if not next_date or not next_amount or next_amount <= 0 or next_date < today:
        return ()
    per_year = max(1, round((next_12m or next_amount) / next_amount))
    step = timedelta(days=365 / per_year)
    out, when = [], next_date
    end = today + timedelta(days=365 * years)
    while when <= end:
        out.append((when, next_amount))
        when += step
    return tuple(out)
