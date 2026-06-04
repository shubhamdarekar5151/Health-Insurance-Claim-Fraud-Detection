"""SQLite helpers for the streaming fraud-detection demo.

Two tables share a single SQLite database:

* ``claims_pending``  — claims emitted by the producer that have not yet
                        been scored. The consumer drains this table.
* ``claims_scored``   — scored claims (with risk_score + prediction).
                        The dashboard reads from here.

SQLite is opened in WAL mode so the dashboard can read while the
consumer writes. Each pending->scored move runs inside a single
transaction; if the consumer crashes mid-batch, the pending row stays
put and gets retried on the next poll.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable

# Path to the shared SQLite database. Overridable via the CLAIMS_DB
# environment variable so the producer, consumer, and dashboard can be
# pointed at a common location (e.g. a mounted volume under Docker).
DEFAULT_DB = os.environ.get("CLAIMS_DB", "claims.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS claims_pending (
    claim_id      TEXT    PRIMARY KEY,
    features_json TEXT    NOT NULL,
    display_json  TEXT    NOT NULL,
    true_label    INTEGER NOT NULL,
    arrived_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS claims_scored (
    claim_id          TEXT    PRIMARY KEY,
    provider          TEXT    NOT NULL,
    bene_id           TEXT    NOT NULL,
    claim_amount      REAL    NOT NULL,
    is_inpatient      INTEGER NOT NULL,
    claim_start_dt    TEXT    NOT NULL,
    arrived_at        REAL    NOT NULL,
    scored_at         REAL    NOT NULL,
    risk_score        REAL    NOT NULL,
    prediction        INTEGER NOT NULL,
    true_label        INTEGER NOT NULL,
    top_features_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_scored_arrived
    ON claims_scored(arrived_at);
CREATE INDEX IF NOT EXISTS idx_scored_provider
    ON claims_scored(provider);
CREATE INDEX IF NOT EXISTS idx_scored_risk
    ON claims_scored(risk_score);
"""


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Open a SQLite connection in WAL mode for concurrent reads."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def insert_pending(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    features: dict,
    display: dict,
    true_label: int,
) -> None:
    """Insert a single pending claim. Idempotent on claim_id."""
    conn.execute(
        """
        INSERT OR REPLACE INTO claims_pending
            (claim_id, features_json, display_json,
             true_label, arrived_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            claim_id,
            json.dumps(features),
            json.dumps(display),
            int(true_label),
            time.time(),
        ),
    )


def fetch_pending_batch(
    conn: sqlite3.Connection, limit: int = 200,
) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM claims_pending ORDER BY arrived_at ASC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


def move_pending_to_scored(
    conn: sqlite3.Connection,
    scored_rows: Iterable[dict],
    claim_ids: Iterable[str],
) -> None:
    """Insert scored rows + delete pending rows in a single transaction."""
    rows = list(scored_rows)
    ids = list(claim_ids)
    conn.execute("BEGIN")
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO claims_scored (
                claim_id, provider, bene_id, claim_amount,
                is_inpatient, claim_start_dt, arrived_at, scored_at,
                risk_score, prediction, true_label, top_features_json
            ) VALUES (
                :claim_id, :provider, :bene_id, :claim_amount,
                :is_inpatient, :claim_start_dt, :arrived_at, :scored_at,
                :risk_score, :prediction, :true_label, :top_features_json
            )
            """,
            rows,
        )
        conn.executemany(
            "DELETE FROM claims_pending WHERE claim_id = ?",
            [(i,) for i in ids],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def counts(conn: sqlite3.Connection) -> dict:
    pend = conn.execute(
        "SELECT COUNT(*) AS n FROM claims_pending"
    ).fetchone()["n"]
    scored = conn.execute(
        "SELECT COUNT(*) AS n FROM claims_scored"
    ).fetchone()["n"]
    return {"pending": pend, "scored": scored}
