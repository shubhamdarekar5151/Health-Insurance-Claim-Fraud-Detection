"""Evaluate the trained XGBoost model on the held-out test set.

Computes the full metric panel — Precision, Recall, F1, ROC-AUC, PR-AUC —
plus confusion matrices at thresholds 0.3 / 0.5 / 0.7. Saves a metrics
JSON and PNG plots used by the report and the dashboard's Model tab.

Usage:
    python -m src.ml.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROCESSED = Path("data/processed")
MODELS = Path("models")
PLOTS = Path("docs/screenshots")
PLOTS.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.3, 0.5, 0.7]


def evaluate_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "threshold": thr,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": cm.tolist(),
    }


def plot_roc(y, proba, roc_auc, path):
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#3b82f6",
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (held-out test set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_pr(y, proba, pr_auc, path):
    precision, recall, _ = precision_recall_curve(y, proba)
    baseline = y.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, color="#ef4444",
            label=f"AP = {pr_auc:.4f}")
    ax.axhline(baseline, ls="--", color="gray", lw=1,
               label=f"baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (held-out test set)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_confusion(cm, thr, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: non-fraud", "Pred: fraud"],
        yticklabels=["True: non-fraud", "True: fraud"], ax=ax,
    )
    ax.set_title(f"Confusion matrix (threshold = {thr})")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_importance(model, feat, path, top_n=20):
    imp = pd.Series(model.feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    imp.plot.barh(ax=ax, color="#3b82f6")
    ax.set_title(f"Top {top_n} feature importances (gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return imp


def main():
    print("Loading model + test data...")
    model = joblib.load(MODELS / "xgb_fraud.pkl")
    meta = json.loads((PROCESSED / "feature_columns.json").read_text())
    feat = meta["feature_columns"]
    test = pd.read_parquet(PROCESSED / "claims_test.parquet")
    X_test = test[feat]
    y_test = test["label"].values
    print(f"  test rows: {len(test):,}  "
          f"positive rate: {y_test.mean():.4f}")

    print("Scoring test set...")
    proba = model.predict_proba(X_test)[:, 1]

    print("Computing aggregate metrics...")
    pr_auc = float(average_precision_score(y_test, proba))
    roc_auc = float(roc_auc_score(y_test, proba))
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")

    print("Computing threshold-sweep metrics...")
    thr_results = [
        evaluate_at_threshold(y_test, proba, thr) for thr in THRESHOLDS
    ]
    for r in thr_results:
        print(f"  thr={r['threshold']}  "
              f"P={r['precision']:.4f}  "
              f"R={r['recall']:.4f}  "
              f"F1={r['f1']:.4f}  "
              f"acc={r['accuracy']:.4f}  "
              f"CM={r['confusion_matrix']}")

    print("Generating plots...")
    plot_roc(y_test, proba, roc_auc, PLOTS / "roc_curve.png")
    plot_pr(y_test, proba, pr_auc, PLOTS / "pr_curve.png")
    for r in thr_results:
        path = PLOTS / f"confusion_matrix_{int(r['threshold']*10):02d}.png"
        plot_confusion(r["confusion_matrix"], r["threshold"], path)
    imp = plot_importance(
        model, feat, PLOTS / "feature_importance.png", top_n=20,
    )
    print("  wrote plots:")
    for p in sorted(PLOTS.iterdir()):
        if p.suffix == ".png":
            print(f"    {p}")

    print("Writing metrics.json...")
    (MODELS / "metrics.json").write_text(json.dumps({
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "n_test_claims": int(len(test)),
        "test_positive_rate": float(y_test.mean()),
        "by_threshold": thr_results,
        "top_feature_importance": {
            k: float(v) for k, v in imp.iloc[::-1].head(20).items()
        },
    }, indent=2))

    # Capstone targets
    f1_05 = next(r for r in thr_results if r["threshold"] == 0.5)["f1"]
    print()
    print("=== Target check ===")
    ok_pr = pr_auc >= 0.90
    ok_f1 = f1_05 >= 0.85
    print(f"  PR-AUC >= 0.90 : {pr_auc:.4f}  "
          f"{'OK' if ok_pr else 'BELOW'}")
    print(f"  F1@0.5 >= 0.85 : {f1_05:.4f}  "
          f"{'OK' if ok_f1 else 'BELOW'}")


if __name__ == "__main__":
    main()
