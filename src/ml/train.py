"""XGBoost trainer for CMS provider fraud detection.

Pipeline:
  1. Load processed per-claim training rows from features.py.
  2. Group-aware split by Provider (80% train / 20% validation) so that no
     provider appears on both sides (mirrors the Day-2 train/test split logic).
  3. Optuna study (default 20 trials) maximising PR-AUC on the validation set
     with early stopping inside each trial.
  4. Retrain the best configuration on the (sub-)train fold with early
     stopping on the validation fold; save the resulting booster.

Imbalance handling: XGBoost's `scale_pos_weight = N_neg / N_pos` (~1.78 at
claim level, since 36% of training claims are from fraud providers). SMOTE
was considered but rejected: it interpolates between rows, which is
unprincipled for categorical features (State, Race, target-encoded ICD codes)
and offers little benefit at moderate imbalance.

Usage:
    python -m src.ml.train                 # 20 trials, ~3-8 minutes
    python -m src.ml.train --n-trials 5    # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VAL_SIZE = 0.20


def load_train_data():
    meta = json.loads((PROCESSED / "feature_columns.json").read_text())
    feat = meta["feature_columns"]
    df = pd.read_parquet(PROCESSED / "claims_train.parquet")
    return df[feat], df["label"].values, df["Provider"].values, feat


def split_by_provider(X, y, groups):
    gss = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(gss.split(X, y, groups))
    return (
        X.iloc[train_idx], X.iloc[val_idx],
        y[train_idx], y[val_idx],
        groups[train_idx], groups[val_idx],
    )


def build_objective(X_tr, y_tr, X_vl, y_vl, scale_pos_weight):
    def objective(trial):
        sf = trial.suggest_float
        si = trial.suggest_int
        params = {
            "n_estimators": 800,
            "max_depth": si("max_depth", 4, 10),
            "learning_rate": sf("learning_rate", 0.02, 0.3, log=True),
            "min_child_weight": si("min_child_weight", 1, 10),
            "subsample": sf("subsample", 0.6, 1.0),
            "colsample_bytree": sf("colsample_bytree", 0.6, 1.0),
            "gamma": sf("gamma", 0.0, 5.0),
            "reg_lambda": sf("reg_lambda", 0.5, 5.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "early_stopping_rounds": 30,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        proba = model.predict_proba(X_vl)[:, 1]
        return float(average_precision_score(y_vl, proba))

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading processed training data...")
    X, y, groups, feat = load_train_data()
    print(f"  X: {X.shape}  positive rate: {y.mean():.4f}"
          f"  features: {len(feat)}")

    print("Splitting train/val by Provider (group-aware)...")
    X_tr, X_vl, y_tr, y_vl, g_tr, g_vl = split_by_provider(X, y, groups)
    print(f"  train: {X_tr.shape}  positive rate: {y_tr.mean():.4f}"
          f"  providers: {len(np.unique(g_tr))}")
    print(f"  val:   {X_vl.shape}  positive rate: {y_vl.mean():.4f}"
          f"  providers: {len(np.unique(g_vl))}")

    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"  scale_pos_weight = {spw:.4f}")

    print(f"Running Optuna study ({args.n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_start = time.time()

    def cb(study, trial):
        elapsed = time.time() - trial_start
        print(f"  trial {trial.number+1:>2}/{args.n_trials}  "
              f"PR-AUC={trial.value:.4f}  "
              f"best={study.best_value:.4f}  "
              f"(+{elapsed:5.1f}s)")

    study.optimize(
        build_objective(X_tr, y_tr, X_vl, y_vl, spw),
        n_trials=args.n_trials,
        callbacks=[cb],
        show_progress_bar=False,
    )
    print(f"Best val PR-AUC: {study.best_value:.4f}")
    print(f"Best params:     {study.best_params}")

    print("Retraining best configuration with early stopping...")
    best_params = {
        **study.best_params,
        "n_estimators": 1200,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": spw,
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = XGBClassifier(**best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    val_proba = model.predict_proba(X_vl)[:, 1]
    val_pr_auc = float(average_precision_score(y_vl, val_proba))
    val_roc_auc = float(roc_auc_score(y_vl, val_proba))
    print(f"  best_iteration: {model.best_iteration}")
    print(f"  val PR-AUC:  {val_pr_auc:.4f}")
    print(f"  val ROC-AUC: {val_roc_auc:.4f}")

    print("Saving model + training summary...")
    joblib.dump(model, MODELS / "xgb_fraud.pkl")
    (MODELS / "training_summary.json").write_text(json.dumps({
        "best_params": study.best_params,
        "scale_pos_weight": spw,
        "best_iteration": int(model.best_iteration),
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "n_train_claims": int(len(X_tr)),
        "n_val_claims": int(len(X_vl)),
        "n_train_providers": int(len(np.unique(g_tr))),
        "n_val_providers": int(len(np.unique(g_vl))),
        "feature_count": len(feat),
        "n_optuna_trials": args.n_trials,
        "study_best_value": float(study.best_value),
        "elapsed_seconds": float(time.time() - t0),
    }, indent=2))
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
