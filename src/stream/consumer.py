"""Scoring service: polls pending claims, scores with XGBoost, writes
results to ``claims_scored``.

Loads the trained model once at startup. Each poll cycle:
    1. SELECT a batch of pending claims (default up to 200).
    2. Build a feature DataFrame and call predict_proba.
    3. INSERT the scored rows and DELETE the pending rows in one
       transaction.

Usage:
    python -m src.stream.consumer                 # threshold 0.5
    python -m src.stream.consumer --threshold 0.3
    python -m src.stream.consumer --poll-ms 250
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from src import db

PROCESSED = Path("data/processed")
MODELS = Path("models")

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[consumer] caught signal, stopping after current batch...")


def score_batch(model, feature_cols, rows):
    """Score a list of sqlite3.Row pending records.

    Returns a list of dicts ready for ``claims_scored`` insertion.
    """
    feats = [json.loads(r["features_json"]) for r in rows]
    disp = [json.loads(r["display_json"]) for r in rows]
    X = pd.DataFrame(feats)[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    now = time.time()
    out = []
    for r, d, p in zip(rows, disp, proba):
        out.append({
            "claim_id":         r["claim_id"],
            "provider":         str(d["Provider"]),
            "bene_id":          str(d["BeneID"]),
            "claim_amount":     float(d["InscClaimAmtReimbursed"]),
            "is_inpatient":     int(d["is_inpatient"]),
            "claim_start_dt":   str(d["ClaimStartDt"]),
            "arrived_at":       float(r["arrived_at"]),
            "scored_at":        now,
            "risk_score":       float(p),
            # The dashboard re-thresholds on the fly; this is just the
            # default prediction at the consumer's startup threshold.
            "prediction":       int(p >= THRESHOLD),
            "true_label":       int(r["true_label"]),
            "top_features_json": None,  # filled in by SHAP on Day 6
        })
    return out


# Set by main() so score_batch doesn't need an extra arg.
THRESHOLD = 0.5


def main():
    global THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument(
        "--threshold", type=float, default=0.5,
        help="risk threshold for the prediction column (default: 0.5)",
    )
    ap.add_argument(
        "--poll-ms", type=int, default=500,
        help="polling interval in ms (default: 500)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=200,
    )
    args = ap.parse_args()
    THRESHOLD = args.threshold
    poll_s = args.poll_ms / 1000.0

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"[consumer] loading model + feature list...")
    model = joblib.load(MODELS / "xgb_fraud.pkl")
    meta = json.loads((PROCESSED / "feature_columns.json").read_text())
    feature_cols = meta["feature_columns"]
    print(f"[consumer] model loaded "
          f"({len(feature_cols)} features, "
          f"threshold={THRESHOLD}, poll={args.poll_ms}ms)")

    conn = db.connect(args.db)
    db.init_db(conn)

    total_scored = 0
    total_flagged = 0
    t0 = time.time()

    print(f"[consumer] starting poll loop...")
    while not _stop:
        rows = db.fetch_pending_batch(conn, limit=args.batch_size)
        if not rows:
            time.sleep(poll_s)
            continue
        scored = score_batch(model, feature_cols, rows)
        db.move_pending_to_scored(
            conn, scored, [r["claim_id"] for r in scored],
        )
        total_scored += len(scored)
        total_flagged += sum(r["prediction"] for r in scored)
        elapsed = time.time() - t0
        rate = total_scored / elapsed if elapsed > 0 else 0.0
        print(f"[consumer] +{len(scored):3d} scored  "
              f"(total: {total_scored:,}, "
              f"flagged: {total_flagged:,} = "
              f"{total_flagged / max(total_scored, 1):.1%}, "
              f"{rate:.1f}/s avg)")

    print(f"[consumer] done. scored {total_scored:,} claims "
          f"({total_flagged:,} flagged) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
