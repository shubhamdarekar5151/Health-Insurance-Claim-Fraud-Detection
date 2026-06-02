"""Stream simulator: replay test claims into the SQLite pending table.

Reads ``data/processed/claim_stream.parquet`` (held-out claims, sorted by
their original ClaimStartDt) and inserts one row at a time into
``claims_pending`` at the configured rate.

Usage:
    python -m src.stream.producer                   # 1 claim/sec
    python -m src.stream.producer --rate 5          # 5 claims/sec
    python -m src.stream.producer --rate 0.5 \\
        --limit 100                                 # 100 claims, 2s apart
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from src import db

PROCESSED = Path("data/processed")

DISPLAY_COLS = [
    "ClaimID", "Provider", "BeneID",
    "InscClaimAmtReimbursed", "is_inpatient", "ClaimStartDt",
]


def load_stream():
    meta = json.loads((PROCESSED / "feature_columns.json").read_text())
    feature_cols = meta["feature_columns"]
    stream = pd.read_parquet(PROCESSED / "claim_stream.parquet")
    return stream, feature_cols


_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[producer] caught signal, stopping after current claim...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rate", type=float, default=1.0,
        help="claims per second (default: 1.0)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="stop after N claims (default: stream all)",
    )
    ap.add_argument("--db", default=db.DEFAULT_DB)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"[producer] loading claim stream...")
    stream, feature_cols = load_stream()
    n_total = len(stream) if args.limit is None else min(
        args.limit, len(stream),
    )
    print(f"[producer] {n_total:,} claims to emit "
          f"at {args.rate} claims/sec")

    conn = db.connect(args.db)
    db.init_db(conn)

    sleep_s = 1.0 / args.rate
    t0 = time.time()
    sent = 0

    for i in range(n_total):
        if _stop:
            break
        row = stream.iloc[i]
        features = {c: _jsonable(row[c]) for c in feature_cols}
        display = {c: _jsonable(row[c]) for c in DISPLAY_COLS}

        db.insert_pending(
            conn,
            claim_id=str(row["ClaimID"]),
            features=features,
            display=display,
            true_label=int(row["label"]),
        )
        sent += 1
        if sent % 50 == 0 or sent == n_total:
            elapsed = time.time() - t0
            rate = sent / elapsed if elapsed > 0 else 0.0
            print(f"[producer] sent {sent:,}/{n_total:,} "
                  f"({rate:.1f}/s avg)")
        time.sleep(sleep_s)

    print(f"[producer] done. emitted {sent:,} claims "
          f"in {time.time() - t0:.1f}s")


def _jsonable(v):
    """Coerce numpy / pandas scalars to plain JSON-safe types."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float) and (v != v):  # NaN
        return None
    return v


if __name__ == "__main__":
    sys.exit(main())
