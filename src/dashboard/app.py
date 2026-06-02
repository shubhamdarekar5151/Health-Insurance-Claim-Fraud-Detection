"""Streamlit dashboard for the live fraud-detection demo.

Reads scored claims from ``claims.db`` (populated by
``src.stream.consumer``) and re-runs every 2 seconds via
``streamlit-autorefresh``.

Tabs
----
Live Monitor      KPIs, live alerts, four Plotly panels (refreshing).
Model Performance Static training/test metrics + saved plot images.

Run
---
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

DB_PATH = "claims.db"
MODELS = Path("models")
PLOTS = Path("docs/screenshots")
REFRESH_MS = 2000

st.set_page_config(
    page_title="Live Fraud Detection",
    layout="wide",
)


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_scored(conn, since_min):
    sql = "SELECT * FROM claims_scored"
    params: tuple = ()
    if since_min is not None:
        cutoff = time.time() - since_min * 60
        sql += " WHERE scored_at >= ?"
        params = (cutoff,)
    sql += " ORDER BY scored_at DESC"
    return pd.read_sql_query(sql, conn, params=params)


def load_training_metrics():
    metrics = json.loads((MODELS / "metrics.json").read_text())
    summary = json.loads((MODELS / "training_summary.json").read_text())
    return metrics, summary


def table_exists(conn, name):
    cur = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.title("Filters")
    threshold = st.slider(
        "Risk threshold", 0.0, 1.0, 0.5, 0.05,
        help="Risk scores at or above this are flagged.",
    )
    window_label = st.selectbox(
        "Time window",
        ["Last 1 min", "Last 5 min", "Last 15 min", "All time"],
        index=1,
    )
    provider_query = st.text_input(
        "Provider contains", value="",
    ).strip().upper()
    only_flagged = st.checkbox(
        "Show only flagged in alerts", value=True,
    )
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)
    if auto_refresh:
        st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

window_min_map = {
    "Last 1 min": 1, "Last 5 min": 5,
    "Last 15 min": 15, "All time": None,
}
since_min = window_min_map[window_label]


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab_live, tab_model = st.tabs(["Live Monitor", "Model Performance"])


# ===== Live Monitor =================================================

with tab_live:
    st.title("Live Fraud Detection — CMS Provider Claims")

    if not Path(DB_PATH).exists():
        st.warning(
            f"No `{DB_PATH}` found. Start the producer and consumer "
            "first:\n\n"
            "```\n"
            "python -m src.stream.producer --rate 1 &\n"
            "python -m src.stream.consumer &\n"
            "```"
        )
        st.stop()

    conn = get_conn()
    if not table_exists(conn, "claims_scored"):
        st.info("Database exists but no scored claims yet. Waiting...")
        st.stop()

    df_window = load_scored(conn, since_min)
    df_all = load_scored(conn, None)

    if provider_query:
        df_window = df_window[
            df_window["provider"].str.contains(provider_query, na=False)
        ]

    df_window["pred_at_thr"] = (
        df_window["risk_score"] >= threshold
    ).astype(int)
    df_all["pred_at_thr"] = (
        df_all["risk_score"] >= threshold
    ).astype(int)

    # ---- KPIs (driven by the global, cumulative view) --------------
    total = len(df_all)
    flagged = int(df_all["pred_at_thr"].sum())
    flag_rate = flagged / max(total, 1)
    last_5min = df_all[df_all["scored_at"] >= time.time() - 300]
    avg_risk_5min = (
        float(last_5min["risk_score"].mean())
        if len(last_5min) else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims scored", f"{total:,}")
    c2.metric(f"Flagged @ {threshold:.2f}", f"{flagged:,}")
    c3.metric("Flag rate", f"{flag_rate:.1%}")
    c4.metric("Avg risk (last 5 min)", f"{avg_risk_5min:.3f}")

    if total == 0:
        st.info("Waiting for the first scored claim...")
        st.stop()

    st.divider()

    # ---- Alerts table ----------------------------------------------
    st.subheader(f"Live alerts ({window_label.lower()})")
    alerts = df_window.copy()
    if only_flagged:
        alerts = alerts[alerts["pred_at_thr"] == 1]
    alerts = alerts.head(20)

    if len(alerts):
        view = alerts[[
            "claim_id", "provider", "bene_id", "claim_amount",
            "is_inpatient", "risk_score", "true_label", "scored_at",
        ]].copy()
        view["risk_score"] = view["risk_score"].round(3)
        view["scored_at"] = pd.to_datetime(
            view["scored_at"], unit="s",
        ).dt.strftime("%H:%M:%S")
        view["is_inpatient"] = view["is_inpatient"].map(
            {0: "OP", 1: "IP"}
        )
        st.dataframe(
            view, hide_index=True, use_container_width=True,
        )

        with st.expander("Inspect a claim"):
            pick = st.selectbox(
                "Claim ID", options=alerts["claim_id"].tolist(),
            )
            row = alerts[alerts["claim_id"] == pick].iloc[0].to_dict()
            top_feat = row.pop("top_features_json", None)
            st.json(row)
            if top_feat:
                st.subheader("Top contributing features (SHAP)")
                st.json(json.loads(top_feat))
            else:
                st.caption(
                    "Top features (SHAP) not yet computed for this "
                    "claim — added in the Day 6 explainability pass."
                )
    else:
        st.info("No flagged claims in the current window.")

    st.divider()

    # ---- Charts: 2x2 ------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk score distribution")
        fig = px.histogram(
            df_window, x="risk_score", nbins=40,
            color_discrete_sequence=["#3b82f6"],
        )
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="red",
            annotation_text="threshold",
        )
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="risk score", yaxis_title="claims",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud rate over time (1-min buckets)")
        if len(df_window):
            tmp = df_window.copy()
            tmp["bucket"] = (
                pd.to_datetime(tmp["scored_at"], unit="s")
                .dt.floor("1min")
            )
            agg = (
                tmp.groupby("bucket")
                .agg(
                    n=("claim_id", "size"),
                    flagged=("pred_at_thr", "sum"),
                )
                .reset_index()
            )
            agg["fraud_rate"] = agg["flagged"] / agg["n"]
            fig = px.line(
                agg, x="bucket", y="fraud_rate", markers=True,
                color_discrete_sequence=["#ef4444"],
            )
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_tickformat=".0%",
                xaxis_title="time", yaxis_title="flag rate",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for the selected window.")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 10 providers by flagged claims")
        flagged_df = df_window[df_window["pred_at_thr"] == 1]
        if len(flagged_df):
            top_prov = (
                flagged_df.groupby("provider").size()
                .sort_values(ascending=False).head(10)
                .reset_index(name="flagged_claims")
            )
            fig = px.bar(
                top_prov, y="provider", x="flagged_claims",
                orientation="h",
                color_discrete_sequence=["#ef4444"],
            )
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged claims in the current window.")

    with col4:
        st.subheader(
            f"Live confusion matrix (cumulative, thr = {threshold:.2f})"
        )
        cm = pd.crosstab(
            df_all["true_label"], df_all["pred_at_thr"],
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        cm_arr = cm.values
        fig = go.Figure(go.Heatmap(
            z=cm_arr, text=cm_arr, texttemplate="%{text}",
            x=["Pred: non-fraud", "Pred: fraud"],
            y=["True: non-fraud", "True: fraud"],
            colorscale="Blues", showscale=False,
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        tp = int(cm.loc[1, 1])
        fp = int(cm.loc[0, 1])
        fn = int(cm.loc[1, 0])
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{prec:.3f}")
        m2.metric("Recall", f"{rec:.3f}")
        m3.metric("F1", f"{f1:.3f}")


# ===== Model Performance ============================================

with tab_model:
    st.title("Model Performance — Held-out Test Set")

    try:
        metrics, summary = load_training_metrics()
    except FileNotFoundError:
        st.warning(
            "Training artifacts not found. Run "
            "`python -m src.ml.train` and "
            "`python -m src.ml.evaluate` first."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Test PR-AUC",
        f"{metrics['pr_auc']:.4f}",
        delta=f"target >= 0.90",
        delta_color="off",
    )
    c2.metric("Test ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c3.metric(
        "Test claims", f"{metrics['n_test_claims']:,}",
    )
    c4.metric(
        "Test positive rate",
        f"{metrics['test_positive_rate']:.1%}",
    )

    st.subheader("Threshold sweep")
    tdf = pd.DataFrame(metrics["by_threshold"])[
        ["threshold", "precision", "recall", "f1", "accuracy"]
    ]
    st.dataframe(tdf, hide_index=True, use_container_width=True)

    st.subheader("ROC and PR curves")
    a, b = st.columns(2)
    if (PLOTS / "roc_curve.png").exists():
        a.image(str(PLOTS / "roc_curve.png"))
    if (PLOTS / "pr_curve.png").exists():
        b.image(str(PLOTS / "pr_curve.png"))

    st.subheader("Feature importance (top 20, gain)")
    if (PLOTS / "feature_importance.png").exists():
        st.image(str(PLOTS / "feature_importance.png"))

    st.subheader("Confusion matrix at threshold 0.5")
    if (PLOTS / "confusion_matrix_05.png").exists():
        st.image(str(PLOTS / "confusion_matrix_05.png"))

    st.subheader("Best hyperparameters (Optuna)")
    st.json(summary["best_params"])

    st.caption(
        f"Trained on {summary['n_train_claims']:,} claims "
        f"({summary['n_train_providers']:,} providers); "
        f"validation: {summary['n_val_claims']:,} claims "
        f"({summary['n_val_providers']:,} providers). "
        f"Best iteration: {summary['best_iteration']}. "
        f"Optuna trials: {summary['n_optuna_trials']}. "
        f"Total training time: {summary['elapsed_seconds']:.1f}s."
    )
