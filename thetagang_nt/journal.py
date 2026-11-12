"""The Portfolio Manager journal: Postgres tables of strategies, their legs, a
daily snapshot per strategy and (step 3) every decision. The monitor records
each cycle; Grafana reads it. A database problem is logged and skipped: the
dashboard never depends on it.

Strategy IDs come from pm.core.strategy_id, matched against the strategies the
journal knows, so an ID survives a roll. Positions opened before the journal
existed get today as an estimated entry date and their IB cost as net debit."""

from __future__ import annotations

import json
from dataclasses import astuple, dataclass, fields
from datetime import date, datetime, timedelta
from typing import Any

import psycopg
from psycopg import sql

from thetagang_nt.pm.core.assess import Assessment
from thetagang_nt.pm.core.strategy_id import Seen

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
CREATE TABLE IF NOT EXISTS iv_history (    -- for the IV-change distribution (M8)
    contract text NOT NULL,
    day date NOT NULL,
    at timestamptz NOT NULL,               -- the last good quote that day
    symbol text NOT NULL,
    expiry date NOT NULL,
    strike numeric NOT NULL,
    option_right text NOT NULL,            -- C | P
    iv numeric NOT NULL,                   -- annualised, 0.30 = 30%
    underlying numeric NOT NULL,           -- spot, or the VIX option's future
    PRIMARY KEY (contract, day)
);
-- the forward test (historical_replay): what the model said each day, kept
-- before the outcome is known; forecast_results compares it with what happened
CREATE TABLE IF NOT EXISTS forecasts (
    strategy_id text NOT NULL,
    day date NOT NULL,
    at timestamptz NOT NULL,               -- the last good quotes that day
    legs_key text NOT NULL,                -- a roll or resize starts a new series
    horizon integer NOT NULL,              -- trading days of the expectation
    outcome text NOT NULL,
    view text NOT NULL,                    -- HOLD | CLOSE | ROLL | - (IV fixed)
    view_iv text NOT NULL,                 -- the same under the IV history model
    v_mid numeric NOT NULL,
    v_exec numeric NOT NULL,
    v_model numeric,
    e_pnl numeric, es numeric,
    e_pnl_iv numeric, es_iv numeric,
    u_hold numeric, u_hold_iv numeric, u_close numeric,
    wcl_5 numeric, wcl_10 numeric, stress_loss numeric, loss_budget numeric,
    iv_rank numeric, iv_view numeric, sticky_gap numeric,
    american boolean NOT NULL,
    PRIMARY KEY (strategy_id, day)
);
CREATE TABLE IF NOT EXISTS leg_marks (     -- every held leg, wide or not
    contract text NOT NULL,
    day date NOT NULL,
    at timestamptz NOT NULL,
    strategy_id text NOT NULL,
    qty numeric NOT NULL,
    bid numeric,                           -- 0: no bid
    ask numeric,
    underlying numeric,
    iv numeric,
    PRIMARY KEY (contract, day)
);
CREATE TABLE IF NOT EXISTS roll_forecasts (
    strategy_id text NOT NULL,
    candidate text NOT NULL,
    day date NOT NULL,
    at timestamptz NOT NULL,               -- when it was quoted
    net_open numeric NOT NULL,
    open_cost numeric NOT NULL,
    wcl_5 numeric, wcl_10 numeric, stress_loss numeric,
    e_pnl numeric, es numeric,
    u_new numeric, u_new_iv numeric,
    fails text NOT NULL,
    legs jsonb NOT NULL,                   -- contract, qty, entry price per leg
    PRIMARY KEY (strategy_id, candidate, day)
);
-- a forecast and its result h recorded days later, on unchanged legs. A strategy
-- that closes or expires before then has no later row, so it is not in here.
CREATE OR REPLACE VIEW forecast_results AS
SELECT strategy_id, day, later_day, horizon, outcome, view, view_iv, v_mid, v_exec,
       e_pnl, es, e_pnl_iv, es_iv, wcl_5, stress_loss,
       later_mid - v_mid AS realised_mid,
       later_exec - v_exec AS realised_exec,
       v_mid - later_mid >= COALESCE(wcl_5, 0) AS beyond_wcl
FROM (
    SELECT f.*,
           lead(day, 5) OVER w AS later_day,
           lead(v_mid, 5) OVER w AS later_mid,
           lead(v_exec, 5) OVER w AS later_exec
    FROM forecasts f
    WHERE horizon = 5
    WINDOW w AS (PARTITION BY strategy_id, legs_key ORDER BY day)
) x
WHERE later_day IS NOT NULL AND later_day - day <= 10;
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


@dataclass(frozen=True)
class IvRow:
    """One held option's implied vol this cycle."""

    contract: str
    symbol: str
    expiry: date
    strike: float
    right: str
    iv: float
    underlying: float


@dataclass(frozen=True)
class ForecastRow:
    """Column for column, the forecasts table, without day and at."""

    strategy_id: str
    legs_key: str
    horizon: int
    outcome: str
    view: str
    view_iv: str
    v_mid: float
    v_exec: float
    v_model: float | None
    e_pnl: float | None
    es: float | None
    e_pnl_iv: float | None
    es_iv: float | None
    u_hold: float | None
    u_hold_iv: float | None
    u_close: float | None
    wcl_5: float | None
    wcl_10: float | None
    stress_loss: float | None
    loss_budget: float | None
    iv_rank: float | None
    iv_view: float | None
    sticky_gap: float | None
    american: bool


@dataclass(frozen=True)
class LegMark:
    contract: str
    strategy_id: str
    qty: float
    bid: float | None
    ask: float | None
    underlying: float | None
    iv: float | None


@dataclass(frozen=True)
class RollForecast:
    strategy_id: str
    candidate: str
    at: datetime
    net_open: float
    open_cost: float
    wcl_5: float | None
    wcl_10: float | None
    stress_loss: float | None
    e_pnl: float | None
    es: float | None
    u_new: float | None
    u_new_iv: float | None
    fails: str
    legs: list[dict[str, Any]]


def closed_leg_pnl(
    qty: float,
    entry: float,
    multiplier: float,
    expiry: date,
    closed_on: date,
    bid: float | None,
    ask: float | None,
) -> float | None:
    """$ realised on an option leg no longer held. Gone after its expiry, it
    settled at 0: expired, or assigned/exercised, the shares then carrying the
    strike. Gone earlier, it was closed at its last mark's mid; None without one.
    ponytail: the last mark's mid (minutes old), not the fill; a partial close
    or a contract traded again after closing isn't told apart."""
    if closed_on > expiry:
        exit_px = 0.0
    else:
        quotes = [x for x in (bid, ask) if x]
        if not quotes:
            return None
        exit_px = sum(quotes) / len(quotes)
    return (exit_px - entry) * qty * multiplier


class Journal:
    def __init__(self, dsn: str, grafana_password: str | None = None) -> None:
        self._dsn = dsn
        self._grafana_password = grafana_password
        self._conn: psycopg.Connection | None = None

    def _db(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=5)
            self._conn.execute(SCHEMA)
            if self._grafana_password:
                self._grant_grafana(self._conn, self._grafana_password)
        return self._conn

    @staticmethod
    def _grant_grafana(conn: psycopg.Connection, password: str) -> None:
        """Grafana's login: grafana_ro, which can read the journal and nothing else."""
        conn.execute(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana_ro')"
            " THEN CREATE ROLE grafana_ro LOGIN; END IF; END $$"
        )
        conn.execute(
            sql.SQL("ALTER ROLE grafana_ro WITH PASSWORD {}").format(
                sql.Literal(password)
            )
        )
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO grafana_ro").format(
                sql.Identifier(conn.info.dbname)
            )
        )
        conn.execute("GRANT USAGE ON SCHEMA public TO grafana_ro")
        conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_ro")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def known(self, today: date) -> dict[str, Seen]:
        """Open strategies, and those closed in the last few days, with their
        open (or last) legs: what pm.core.strategy_id matches today's against."""
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

    def kept_premium(self) -> dict[str, float]:
        """Per underlying, $ realised on its option legs no longer held by any
        strategy (a regrouped leg is still held): what lowers a stock's
        breakeven. Only what the journal saw."""
        rows = self._db().execute(
            """
            WITH legs AS (
                SELECT s.underlying, l.contract, l.label, l.multiplier,
                    (array_agg(l.qty ORDER BY l.opened_on))[1] AS qty,
                    (array_agg(l.entry_price ORDER BY l.opened_on))[1] AS entry,
                    max(l.closed_on) AS closed_on
                FROM strategy_legs l JOIN strategies s ON s.id = l.strategy_id
                GROUP BY s.underlying, l.contract, l.label, l.multiplier
                HAVING bool_and(l.closed_on IS NOT NULL)
            )
            SELECT legs.underlying, legs.label, legs.qty, legs.entry,
                legs.multiplier, legs.closed_on, m.bid, m.ask
            FROM legs LEFT JOIN LATERAL (
                SELECT bid, ask FROM leg_marks
                WHERE contract = legs.contract AND day <= legs.closed_on
                ORDER BY day DESC LIMIT 1
            ) m ON true
            """
        )
        out: dict[str, float] = {}
        for und, label, qty, entry, mult, closed_on, bid, ask in rows:
            expiry = date.fromisoformat(label.split()[1])  # "MS 20261106 197.5C"
            pnl = closed_leg_pnl(
                float(qty),
                float(entry),
                float(mult),
                expiry,
                closed_on,
                None if bid is None else float(bid),
                None if ask is None else float(ask),
            )
            if pnl is not None:
                out[und] = out.get(und, 0.0) + pnl
        return out

    def entry(self, ids: list[str]) -> dict[str, tuple[float, int | None]]:
        """Each known strategy's (net debit, initial DTE) at entry."""
        rows = self._db().execute(
            "SELECT id, net_debit, initial_dte FROM strategies WHERE id = ANY(%s)",
            (ids,),
        )
        return {r[0]: (float(r[1]), r[2]) for r in rows}

    def decide(
        self, rows: list[tuple[str, str, list[dict], str]], now: datetime
    ) -> None:
        """The decision log: (strategy, outcome, signals, the bot's action), one
        row each time a strategy's outcome or signals change."""
        with self._db().transaction():
            for sid, out, found, bot in rows:
                self._db().execute(
                    "INSERT INTO decisions (at, strategy_id, outcome, signals, bot_action)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (now, sid, out, json.dumps(found), bot),
                )

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

    def record_forecasts(
        self,
        forecasts: list[ForecastRow],
        marks: list[LegMark],
        rolls: list[RollForecast],
        today: date,
        now: datetime,
    ) -> None:
        """The forward test, overwritten through the day: the last good values
        stand for the day, so nothing is changed once the day is over."""
        conn = self._db()
        names = [f.name for f in fields(ForecastRow)]
        insert = sql.SQL(
            "INSERT INTO forecasts ({}, day, at) VALUES ({}, %s, %s)"
            " ON CONFLICT (strategy_id, day) DO UPDATE SET {}, at = EXCLUDED.at"
        ).format(
            sql.SQL(", ").join(map(sql.Identifier, names)),
            sql.SQL(", ").join(sql.Placeholder() * len(names)),
            sql.SQL(", ").join(
                sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(n))
                for n in names
                if n != "strategy_id"
            ),
        )
        with conn.transaction():
            for r in forecasts:
                conn.execute(insert, (*astuple(r), today, now))
            for m in marks:
                conn.execute(
                    """
                    INSERT INTO leg_marks (contract, day, at, strategy_id, qty, bid,
                        ask, underlying, iv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (contract, day) DO UPDATE SET at = EXCLUDED.at,
                        strategy_id = EXCLUDED.strategy_id, qty = EXCLUDED.qty,
                        bid = EXCLUDED.bid, ask = EXCLUDED.ask,
                        underlying = EXCLUDED.underlying, iv = EXCLUDED.iv
                    """,
                    (
                        m.contract,
                        today,
                        now,
                        m.strategy_id,
                        m.qty,
                        m.bid,
                        m.ask,
                        m.underlying,
                        m.iv,
                    ),
                )
            for c in rolls:
                conn.execute(
                    """
                    INSERT INTO roll_forecasts (strategy_id, candidate, day, at,
                        net_open, open_cost, wcl_5, wcl_10, stress_loss, e_pnl, es,
                        u_new, u_new_iv, fails, legs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_id, candidate, day) DO NOTHING
                    """,
                    (
                        c.strategy_id,
                        c.candidate,
                        today,
                        c.at,
                        c.net_open,
                        c.open_cost,
                        c.wcl_5,
                        c.wcl_10,
                        c.stress_loss,
                        c.e_pnl,
                        c.es,
                        c.u_new,
                        c.u_new_iv,
                        c.fails,
                        json.dumps(c.legs),
                    ),
                )

    def record_iv(self, rows: list[IvRow], today: date, now: datetime) -> None:
        """Today's implied vol per held option, overwritten through the day: one
        row per contract per day, the last good quote."""
        conn = self._db()
        with conn.transaction():
            for r in rows:
                conn.execute(
                    """
                    INSERT INTO iv_history (contract, day, at, symbol, expiry, strike,
                        option_right, iv, underlying)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (contract, day) DO UPDATE SET at = EXCLUDED.at,
                        iv = EXCLUDED.iv, underlying = EXCLUDED.underlying
                    """,
                    (
                        r.contract,
                        today,
                        now,
                        r.symbol,
                        r.expiry,
                        r.strike,
                        r.right,
                        r.iv,
                        r.underlying,
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
