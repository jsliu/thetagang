"""The joint price and IV probability model (vault: expected_pnl, M8): each
underlying's own history of h-day price and IV moves, from the days whose IV
stood in the same third of its trailing year as today's. Every sample is equally
likely and repriced directly, so there are no grid artefacts.

No directional view on price: returns are demeaned, then scaled to the
realised-vol forecast, keeping only their shape and their link to IV. The one
view is IV mean reversion, a hypothesis on thin evidence (about a year of
weekly data), so the expected IV change is shrunk. Reported, not a signal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class IvLimits:
    """Register IDs in brackets; all hypotheses, starting values chosen here."""

    window: int = 252  # [M8] trading days today's IV is ranked against
    min_samples: int = 30  # [M8] fewer in today's third: use all of history
    shrink: float = 0.5  # [M8] share of the expected IV change that is kept
    term_days: int = 30  # [M3] a leg moves by sqrt(this / DTE) of it, at most all


@dataclass(frozen=True)
class IvHistory:
    closes: tuple[float, ...]
    ivs: tuple[float, ...]  # same days, annualised


@dataclass(frozen=True)
class IvModel:
    samples: tuple[tuple[float, float], ...]  # (simple return, IV change)
    rank: float | None  # today's IV: the share of its trailing year at or below it
    conditioned: bool  # samples from today's third, not all of history
    view: float  # the expected IV change kept, annualised (0.02 = 2 points)


def _rank(ivs: tuple[float, ...], t: int, window: int) -> float:
    """Trailing only: no day sees its own future."""
    return sum(x <= ivs[t] for x in ivs[t - window : t]) / window


def _third(rank: float) -> int:
    return 0 if rank < 1 / 3 else 2 if rank > 2 / 3 else 1


def build(
    history: IvHistory, sigma_f: float | None, h: int, lim: IvLimits = IvLimits()
) -> IvModel | None:
    """None without a realised-vol forecast, or with too little usable history."""
    px, iv = history.closes, history.ivs
    n = len(iv)
    if not sigma_f or n != len(px) or n - h < lim.min_samples:
        return None
    if min(px) <= 0 or min(iv) <= 0:
        return None
    rank = _rank(iv, n - 1, lim.window) if n > lim.window else None
    starts = list(range(n - h))
    conditioned = False
    if rank is not None:
        today = _third(rank)
        same = [
            t
            for t in range(lim.window, n - h)
            if _third(_rank(iv, t, lim.window)) == today
        ]
        if len(same) >= lim.min_samples:
            starts, conditioned = same, True
    returns = [math.log(px[t + h] / px[t]) for t in starts]
    changes = [iv[t + h] - iv[t] for t in starts]
    sd = pstdev(returns)
    if sd <= 0:
        return None
    mean_return, mean_change = fmean(returns), fmean(changes)
    scale = sigma_f * math.sqrt(h / 252) / sd
    drop = (1 - lim.shrink) * mean_change
    samples = tuple(
        (math.exp((r - mean_return) * scale) - 1, c - drop)
        for r, c in zip(returns, changes, strict=True)
    )
    return IvModel(samples, rank, conditioned, lim.shrink * mean_change)
