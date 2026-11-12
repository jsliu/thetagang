"""Stable strategy IDs (vault: data_model, ibkr_integration).

IB doesn't know strategies, and a structure's legs change when it's rolled, so
its name changes too. An ID is kept by matching today's structures to known
strategies of the same kind: first by a shared contract (an unchanged leg), then
by underlying (a roll replaced every leg); otherwise a new ID like
"AA-debit-spread-1". A changed kind (legs regrouped, a short leg expired) is a
new strategy.
The journal (step 2) stores `known` between runs."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Seen:
    symbol: str
    kind: str
    contracts: frozenset  # the legs' contract ids


def new_id(seen: Seen, taken: set[str]) -> str:
    base = f"{seen.symbol.lstrip('^')}-{re.sub(r'[^a-z0-9]+', '-', seen.kind.lower()).strip('-')}"
    n = 1
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def auto_ids(current: list[Seen], known: dict[str, Seen]) -> list[str]:
    """An ID for each of `current`, in order. Each known ID goes to at most
    one structure; deterministic for the same inputs."""
    ids: list[str | None] = [None] * len(current)
    free = dict(known)
    # 1. an unchanged leg: the known strategy of that kind sharing the most contracts
    for i, s in enumerate(current):
        best = max(
            (
                k
                for k, v in free.items()
                if (v.symbol, v.kind) == (s.symbol, s.kind)
                and v.contracts & s.contracts
            ),
            key=lambda k: (len(free[k].contracts & s.contracts), k),
            default=None,
        )
        if best is not None:
            ids[i] = best
            del free[best]
    # 2. rolled: same underlying and kind, every leg replaced
    for i, s in enumerate(current):
        if ids[i] is None:
            match = sorted(
                k for k, v in free.items() if (v.symbol, v.kind) == (s.symbol, s.kind)
            )
            if match:
                ids[i] = match[0]
                del free[match[0]]
    # 3. new
    taken = set(known) | {x for x in ids if x}
    for i, s in enumerate(current):
        if ids[i] is None:
            ids[i] = new_id(s, taken)
            taken.add(ids[i])
    return [x for x in ids if x is not None]
