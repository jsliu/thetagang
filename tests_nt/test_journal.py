"""thetagang_nt.journal against a real Postgres (a throwaway test database).
Skipped unless JOURNAL_TEST_DSN is set, e.g. from deploy/:
  docker compose exec journal-db createdb -U thetagang thetagang_test"""

import os
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("psycopg")
DSN = os.environ.get("JOURNAL_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="JOURNAL_TEST_DSN not set")

from thetagang_nt.journal import Journal, LegRow, StrategyRow  # noqa: E402
from thetagang_nt.pm.core.assess import Assessment  # noqa: E402
from thetagang_nt.pm.core.efficiency import Efficiency  # noqa: E402
from thetagang_nt.pm.core.strategy_id import Seen, auto_ids  # noqa: E402

DAY1, DAY2 = date(2026, 11, 2), date(2026, 11, 3)


def assessment(v_exec=180.0):
    return Assessment(
        200.0,
        v_exec,
        (),
        "long convexity",
        {"delta": 10.0, "gamma": 2.0, "theta": -3.0, "vega": 5.0},
        Efficiency(gt_h={5: 0.9}, sigma_be={5: 0.3}),
        1.8,
        30,
    )


def row(sid, symbol, kind, contracts, v_exec=180.0):
    legs = tuple(LegRow(c, f"{symbol} {c}", 1.0, 100.0, 2.0) for c in contracts)
    return StrategyRow(sid, symbol, kind, legs, 200.0, 41.0, assessment(v_exec))


def cycle(journal, day, structures, inputs=None):
    """What the monitor does: match IDs against the journal, then record."""
    seen = [Seen(sym, kind, frozenset(cs)) for sym, kind, cs, _ in structures]
    ids = auto_ids(seen, journal.known(day))
    rows = [row(i, sym, kind, cs, v) for i, (sym, kind, cs, v) in zip(ids, structures)]
    journal.record(
        rows, inputs or {}, day, datetime.combine(day, datetime.min.time(), UTC)
    )
    return ids


@pytest.fixture
def journal():
    j = Journal(DSN)
    db = j._db()
    db.execute(
        "TRUNCATE decisions, snapshots, strategy_legs, strategies, iv_history, forecasts, leg_marks, roll_forecasts"
    )
    yield j
    j.close()


def test_strategies_open_roll_close_and_keep_their_ids(journal):
    db = journal._db()
    ids = cycle(
        journal,
        DAY1,
        [
            ("AA", "debit spread", ["c1", "c2"], 180.0),
            ("GDX", "long put", ["p1"], 50.0),
        ],
    )
    assert ids == ["AA-debit-spread-1", "GDX-long-put-1"]
    assert (
        db.execute(
            "SELECT count(*) FROM strategy_legs WHERE closed_on IS NULL"
        ).fetchone()[0]
        == 3
    )

    # day 2: AA rolled (both legs replaced), GDX closed, your thesis for AA
    inputs = {
        "AA-debit-spread-1": SimpleNamespace(
            thesis="aluminium tariffs",
            invalidation_below=35.0,
            invalidation_above=None,
            target=300.0,
            max_loss=200.0,
            thesis_valid=True,
        )
    }
    ids = cycle(journal, DAY2, [("AA", "debit spread", ["c3", "c4"], 220.0)], inputs)
    assert ids == ["AA-debit-spread-1"]  # the roll kept its ID
    legs = dict(
        db.execute(
            "SELECT contract, closed_on FROM strategy_legs"
            " WHERE strategy_id = 'AA-debit-spread-1'"
        ).fetchall()
    )
    assert legs == {"c1": DAY2, "c2": DAY2, "c3": None, "c4": None}
    status = dict(db.execute("SELECT id, status FROM strategies").fetchall())
    assert status == {"AA-debit-spread-1": "open", "GDX-long-put-1": "closed"}
    thesis, budget = db.execute(
        "SELECT thesis, max_loss FROM strategies WHERE id = 'AA-debit-spread-1'"
    ).fetchone()
    assert thesis == "aluminium tariffs" and float(budget) == 200.0
    # one snapshot per strategy per day, the latest values
    snaps = db.execute(
        "SELECT day, v_exec, pnl_exec, class FROM snapshots"
        " WHERE strategy_id = 'AA-debit-spread-1' ORDER BY day"
    ).fetchall()
    assert [(d, float(v), float(p), c) for d, v, p, c in snaps] == [
        (DAY1, 180.0, -20.0, "long convexity"),
        (DAY2, 220.0, 20.0, "long convexity"),
    ]
    cycle(journal, DAY2, [("AA", "debit spread", ["c3", "c4"], 250.0)])
    (v,) = db.execute(
        "SELECT v_exec FROM snapshots WHERE strategy_id = 'AA-debit-spread-1'"
        " AND day = %s",
        (DAY2,),
    ).fetchone()
    assert float(v) == 250.0  # overwritten, not a second row


def test_iv_history_keeps_one_row_per_contract_per_day_the_latest(journal):
    from thetagang_nt.journal import IvRow

    db = journal._db()
    at = datetime.combine(DAY1, datetime.min.time(), UTC)

    def iv(value):
        return IvRow("c1", "AA", date(2026, 12, 18), 40.0, "C", value, 41.0)

    journal.record_iv([iv(0.30)], DAY1, at)
    journal.record_iv([iv(0.32)], DAY1, at)  # later that day: overwritten
    journal.record_iv([iv(0.35)], DAY2, at)
    rows = db.execute("SELECT day, iv FROM iv_history ORDER BY day").fetchall()
    assert [(d, float(v)) for d, v in rows] == [(DAY1, 0.32), (DAY2, 0.35)]


def forecast(
    sid="AA-debit-spread-1", legs="c1:1|c2:-1", v_mid=200.0, v_exec=180.0, view="CLOSE"
):
    from thetagang_nt.journal import ForecastRow

    return ForecastRow(
        sid,
        legs,
        5,
        "REVIEW",
        view,
        "HOLD",
        v_mid,
        v_exec,
        201.0,
        3.0,
        40.0,
        2.0,
        45.0,
        -30.0,
        -32.0,
        -20.0,
        60.0,
        90.0,
        120.0,
        2000.0,
        0.8,
        -1.0,
        12.0,
        True,
    )


def days_after(start, n):
    """n weekdays from start, as the market would see them."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_forecasts_overwrite_within_a_day_and_results_wait_five_days(journal):
    db = journal._db()
    days = days_after(DAY1, 7)
    at = datetime.combine(DAY1, datetime.min.time(), UTC)
    journal.record_forecasts([forecast(v_mid=150.0)], [], [], days[0], at)
    journal.record_forecasts([forecast(v_mid=200.0)], [], [], days[0], at)
    assert db.execute("SELECT count(*) FROM forecasts").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM forecast_results").fetchone()[0] == 0
    for i, d in enumerate(days[1:], start=1):
        row = forecast(v_mid=200.0 + 10 * i, v_exec=180.0 + 10 * i)
        journal.record_forecasts([row], [], [], d, at)
    rows = db.execute(
        "SELECT day, later_day, realised_mid, realised_exec, e_pnl, view"
        " FROM forecast_results ORDER BY day"
    ).fetchall()
    assert len(rows) == 2  # days 1 and 2 of 7 have a result five days on
    day, later, mid, exe, e, view = rows[0]
    assert (day, later) == (days[0], days[5])
    assert float(mid) == 50.0 and float(exe) == 50.0
    assert float(e) == 3.0 and view == "CLOSE"


def test_a_roll_starts_a_new_forecast_series(journal):
    db = journal._db()
    days = days_after(DAY1, 6)
    at = datetime.combine(DAY1, datetime.min.time(), UTC)
    for i, d in enumerate(days):
        legs = "c1:1|c2:-1" if i < 3 else "c3:1|c4:-1"
        journal.record_forecasts([forecast(legs=legs)], [], [], d, at)
    assert db.execute("SELECT count(*) FROM forecast_results").fetchone()[0] == 0


def test_beyond_wcl_marks_a_loss_past_the_grids_worst(journal):
    db = journal._db()
    days = days_after(DAY1, 6)
    at = datetime.combine(DAY1, datetime.min.time(), UTC)
    for i, d in enumerate(days):
        mid = 200.0 if i == 0 else 100.0  # lost 100, past the grid's 60
        journal.record_forecasts([forecast(v_mid=mid)], [], [], d, at)
    assert db.execute("SELECT beyond_wcl FROM forecast_results").fetchone()[0] is True


def test_leg_marks_and_roll_forecasts_are_kept(journal):
    from thetagang_nt.journal import LegMark, RollForecast

    db = journal._db()
    at = datetime.combine(DAY1, datetime.min.time(), UTC)
    mark = LegMark("c1", "AA-debit-spread-1", 1.0, 0.0, 0.05, 41.0, None)  # no bid
    roll = RollForecast(
        "AA-debit-spread-1",
        "AA 20270115 +1x40C",
        at,
        250.0,
        12.0,
        80.0,
        110.0,
        130.0,
        4.0,
        50.0,
        -25.0,
        -20.0,
        "-",
        [{"contract": "c9", "qty": 1, "price": 2.5}],
    )
    journal.record_forecasts([], [mark], [roll], DAY1, at)
    journal.record_forecasts([], [], [roll], DAY1, at)  # the same scan, next cycle
    assert db.execute("SELECT count(*) FROM roll_forecasts").fetchone()[0] == 1
    bid, ask = db.execute("SELECT bid, ask FROM leg_marks").fetchone()
    assert float(bid) == 0.0 and float(ask) == 0.05
    legs = db.execute("SELECT legs FROM roll_forecasts").fetchone()[0]
    assert legs == [{"contract": "c9", "qty": 1, "price": 2.5}]
