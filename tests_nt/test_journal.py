"""thetagang_nt.journal against a real Postgres (a throwaway test database).
Skipped unless JOURNAL_TEST_DSN is set, e.g. from deploy/:
  docker compose exec journal-db createdb -U thetagang thetagang_test"""

import os
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("psycopg")
DSN = os.environ.get("JOURNAL_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="JOURNAL_TEST_DSN not set")

from thetagang_nt.journal import Journal, LegRow, StrategyRow  # noqa: E402
from thetagang_nt.pm.assess import Assessment  # noqa: E402
from thetagang_nt.pm.efficiency import Efficiency  # noqa: E402
from thetagang_nt.pm.strategy_id import Seen, auto_ids  # noqa: E402

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
    db.execute("TRUNCATE decisions, snapshots, strategy_legs, strategies")
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
