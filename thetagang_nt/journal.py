"""The Portfolio Manager journal: Postgres tables of strategies, their legs, a
daily snapshot per strategy and (step 3) every decision. The monitor records
each cycle; Grafana reads it. A database problem is logged and skipped: the
dashboard never depends on it.

Strategy IDs come from pm.strategy_id, matched against the strategies the
journal knows, so an ID survives a roll. Positions opened before the journal
existed get today as an estimated entry date and their IB cost as net debit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import psycopg

from thetagang_nt.pm.assess import Assessment
from thetagang_nt.pm.strategy_id import Seen

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id text PRIMARY KEY,
    underlying text NOT NULL,
    kind text NOT NULL,
    first_seen date NOT NULL,
    entry_date date NOT NULL,
    entry_estimated boolean NOT NULL,      -- opened before the journal existed
    net_debit numeric NOT NULL,            -- $, paid (+) or received (-), at entry
    initial_dte integer,                   -- days to the earliest expiry at entry
    status text NOT NULL DEFAULT 'open',   -- open | closed
    closed_on date,
    thesis text NOT NULL DEFAULT '',       -- your inputs, from [[pm.strategies]]
    invalidation_below numeric,
    invalidation_above numeric,
    target numeric,
    max_loss numeric,
    thesis_valid boolean,
    inputs_updated timestamptz
);
CREATE TABLE IF NOT EXISTS strategy_legs (
    strategy_id text NOT NULL REFERENCES strategies (id),
    contract text NOT NULL,
    label text NOT NULL,
    qty numeric NOT NULL,                  -- signed
    multiplier numeric NOT NULL,
    entry_price numeric NOT NULL,          -- per share
    opened_on date NOT NULL,
    closed_on date,
    PRIMARY KEY (strategy_id, contract, opened_on)
);
CREATE TABLE IF NOT EXISTS snapshots (
    strategy_id text NOT NULL REFERENCES strategies (id),
    day date NOT NULL,
    at timestamptz NOT NULL,               -- the last update that day
    spot numeric,
    v_mid numeric,
    v_exec numeric,
    pnl_mid numeric,
    pnl_exec numeric,
    delta numeric, gamma numeric, theta numeric, vega numeric,
    class text,
    gt_5 numeric,
    sigma_be_5 numeric,
    remaining_rr numeric,
    dte_front integer,
    flags text NOT NULL DEFAULT '',
    PRIMARY KEY (strategy_id, day)
);
CREATE TABLE IF NOT EXISTS decisions (
    id bigserial PRIMARY KEY,
    at timestamptz NOT NULL,
    strategy_id text NOT NULL REFERENCES strategies (id),
    outcome text NOT NULL,                 -- HOLD | REVIEW | CLOSE review | ...
    signals jsonb NOT NULL,
    bot_action text,
    your_action text
);
"""

RECENT_DAYS = (
    3  # a strategy closed this recently can still be matched (a roll in two steps)
)


@dataclass(frozen=True)
class LegRow:
    contract: str
    label: str
    qty: float
    multiplier: float
    entry_price: float


@dataclass(frozen=True)
class StrategyRow:
    """One strategy as the monitor sees it this cycle."""

    id: str
    symbol: str
    kind: str
    legs: tuple[LegRow, ...]
    net_debit: float  # $, today's cost basis of the legs
    spot: float | None
    a: Assessment


class Journal:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: psycopg.Connection | None = None

    def _db(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=5)
            self._conn.execute(SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def known(self, today: date) -> dict[str, Seen]:
        """Open strategies, and those closed in the last few days, with their
        open (or last) legs: what pm.strategy_id matches today's against."""
        rows = (
            self._db()
            .execute(
                """
            SELECT s.id, s.underlying, s.kind, array_agg(l.contract)
            FROM strategies s JOIN strategy_legs l ON l.strategy_id = s.id
            WHERE (s.status = 'open' AND l.closed_on IS NULL)
               OR (s.status = 'closed' AND s.closed_on >= %s AND l.closed_on = s.closed_on)
            GROUP BY s.id, s.underlying, s.kind
            """,
                (today - timedelta(days=RECENT_DAYS),),
            )
            .fetchall()
        )
        return {r[0]: Seen(r[1], r[2], frozenset(r[3])) for r in rows}

    def record(
        self,
        rows: list[StrategyRow],
        inputs: dict[str, Any],
        today: date,
        now: datetime,
    ) -> None:
        """This cycle's strategies: open new ones and legs, close what's gone,
        copy your inputs, and update today's snapshot. One transaction."""
        conn = self._db()
        with conn.transaction():
            seen = {r.id for r in rows}
            for r in rows:
                conn.execute(
                    """
                    INSERT INTO strategies (id, underlying, kind, first_seen, entry_date,
                        entry_estimated, net_debit, initial_dte)
                    VALUES (%s, %s, %s, %s, %s, true, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET status = 'open', closed_on = NULL
                    """,
                    (r.id, r.symbol, r.kind, today, today, r.net_debit, r.a.dte_front),
                )
                open_now = {
                    c
                    for (c,) in conn.execute(
                        "SELECT contract FROM strategy_legs"
                        " WHERE strategy_id = %s AND closed_on IS NULL",
                        (r.id,),
                    )
                }
                for leg in r.legs:
                    if leg.contract not in open_now:
                        conn.execute(
                            """
                            INSERT INTO strategy_legs (strategy_id, contract, label, qty,
                                multiplier, entry_price, opened_on)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (strategy_id, contract, opened_on)
                            DO UPDATE SET closed_on = NULL, qty = EXCLUDED.qty
                            """,
                            (
                                r.id,
                                leg.contract,
                                leg.label,
                                leg.qty,
                                leg.multiplier,
                                leg.entry_price,
                                today,
                            ),
                        )
                gone = open_now - {leg.contract for leg in r.legs}
                if gone:
                    conn.execute(
                        "UPDATE strategy_legs SET closed_on = %s"
                        " WHERE strategy_id = %s AND closed_on IS NULL AND contract = ANY(%s)",
                        (today, r.id, list(gone)),
                    )
                self._snapshot(conn, r, today, now)
            # strategies no longer held
            conn.execute(
                "UPDATE strategy_legs SET closed_on = %s WHERE closed_on IS NULL"
                " AND strategy_id IN (SELECT id FROM strategies"
                "  WHERE status = 'open' AND NOT (id = ANY(%s)))",
                (today, list(seen)),
            )
            conn.execute(
                "UPDATE strategies SET status = 'closed', closed_on = %s"
                " WHERE status = 'open' AND NOT (id = ANY(%s))",
                (today, list(seen)),
            )
            for sid, x in inputs.items():
                conn.execute(
                    """
                    UPDATE strategies SET thesis = %s, invalidation_below = %s,
                        invalidation_above = %s, target = %s, max_loss = %s,
                        thesis_valid = %s, inputs_updated = %s
                    WHERE id = %s
                    """,
                    (
                        x.thesis,
                        x.invalidation_below,
                        x.invalidation_above,
                        x.target,
                        x.max_loss,
                        x.thesis_valid,
                        now,
                        sid,
                    ),
                )

    @staticmethod
    def _snapshot(
        conn: psycopg.Connection, r: StrategyRow, today: date, now: datetime
    ) -> None:
        a, net = r.a, r.a.net or {}
        pnl = lambda v: None if v is None else v - r.net_debit  # noqa: E731
        conn.execute(
            """
            INSERT INTO snapshots (strategy_id, day, at, spot, v_mid, v_exec, pnl_mid,
                pnl_exec, delta, gamma, theta, vega, class, gt_5, sigma_be_5,
                remaining_rr, dte_front, flags)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (strategy_id, day) DO UPDATE SET
                at = EXCLUDED.at, spot = EXCLUDED.spot, v_mid = EXCLUDED.v_mid,
                v_exec = EXCLUDED.v_exec, pnl_mid = EXCLUDED.pnl_mid,
                pnl_exec = EXCLUDED.pnl_exec, delta = EXCLUDED.delta,
                gamma = EXCLUDED.gamma, theta = EXCLUDED.theta, vega = EXCLUDED.vega,
                class = EXCLUDED.class, gt_5 = EXCLUDED.gt_5,
                sigma_be_5 = EXCLUDED.sigma_be_5, remaining_rr = EXCLUDED.remaining_rr,
                dte_front = EXCLUDED.dte_front, flags = EXCLUDED.flags
            """,
            (
                r.id,
                today,
                now,
                r.spot,
                a.v_mid,
                a.v_exec,
                pnl(a.v_mid),
                pnl(a.v_exec),
                net.get("delta"),
                net.get("gamma"),
                net.get("theta"),
                net.get("vega"),
                a.cls,
                a.eff.gt_h.get(5),
                a.eff.sigma_be.get(5),
                a.remaining_rr,
                a.dte_front,
                json.dumps(list(a.flags)) if a.flags else "",
            ),
        )
