"""Feature engineering pipeline for CMS provider fraud detection.

Produces per-claim feature rows whose features mix three sources:
  1. Per-claim       (amount, type, durations, # diagnoses/procedures)
  2. Per-beneficiary (age, chronic conditions, alive/deceased flag)
  3. Per-provider    (claim count, total/mean/std reimbursement,
                      unique beneficiaries, % inpatient, etc.)

The label is provider-level (`PotentialFraud`) and is broadcast to every
claim of that provider. To prevent label leakage, the 80/20 train/test
split is done **by provider** (all claims from a single provider land on
the same side).

Caveat (documented in the capstone report): provider aggregates are
computed over the full claim history (incl. test-period claims). In
production these would be rolling/causal aggregates. The streaming demo
treats the per-provider profile as a static look-up at inference time.

Outputs (under data/processed/):
    claims_train.parquet      (per-claim training rows + all features)
    claims_test.parquet       (per-claim held-out rows + all features)
    claim_stream.parquet      (test rows sorted by ClaimStartDt)
    target_encoding_dx1.json  (encoding map for primary diagnosis code)
    feature_columns.json      (canonical feature list + dataset stats)

Run from the project root:
    python -m src.ml.features
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

OBSERVATION_CUTOFF = pd.Timestamp("2009-12-31")  # last date in the dataset
RANDOM_STATE = 42
TEST_SIZE = 0.20

DX_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


def load_raw():
    train = pd.read_csv(RAW / "Train.csv")
    bene = pd.read_csv(
        RAW / "Train_Beneficiarydata.csv",
        parse_dates=["DOB", "DOD"],
    )
    inp = pd.read_csv(
        RAW / "Train_Inpatientdata.csv",
        parse_dates=[
            "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt",
        ],
    )
    out = pd.read_csv(
        RAW / "Train_Outpatientdata.csv",
        parse_dates=["ClaimStartDt", "ClaimEndDt"],
    )
    return train, bene, inp, out


def build_beneficiary_features(bene: pd.DataFrame) -> pd.DataFrame:
    """Derive age, chronic-condition count, alive/deceased flag."""
    bene = bene.copy()
    age_days = (OBSERVATION_CUTOFF - bene["DOB"]).dt.days
    bene["age"] = (age_days / 365.25).astype(int)
    # Raw encoding: 1 = Yes (has condition), 2 = No.
    bene["n_chronic"] = (bene[CHRONIC_COLS] == 1).sum(axis=1)
    bene["is_dead"] = bene["DOD"].notna().astype(int)
    # RenalDiseaseIndicator is "Y"/"0" -> binarise.
    bene["renal_disease"] = (bene["RenalDiseaseIndicator"] == "Y").astype(int)
    return bene[
        ["BeneID", "age", "n_chronic", "is_dead", "renal_disease",
         "Gender", "Race", "State", "County",
         "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
         "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]
    ]


def build_per_claim_features(
    inp: pd.DataFrame, out: pd.DataFrame,
) -> pd.DataFrame:
    """Stack inpatient + outpatient claims with derived features."""
    inp = inp.copy()
    out = out.copy()

    inp["is_inpatient"] = 1
    out["is_inpatient"] = 0

    stay = (inp["DischargeDt"] - inp["AdmissionDt"]).dt.days
    inp["stay_days"] = stay.fillna(0)
    out["stay_days"] = 0

    for df in (inp, out):
        present_dx = [c for c in DX_COLS if c in df.columns]
        present_proc = [c for c in PROC_COLS if c in df.columns]
        df["n_diagnoses"] = df[present_dx].notna().sum(axis=1)
        df["n_procedures"] = df[present_proc].notna().sum(axis=1)
        df["claim_duration_days"] = (
            (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days.fillna(0)
        )
        # Raw data has sporadic NaNs in DeductibleAmtPaid; 0 = "none paid".
        df["DeductibleAmtPaid"] = df["DeductibleAmtPaid"].fillna(0)

    keep = [
        "ClaimID", "BeneID", "Provider", "ClaimStartDt", "ClaimEndDt",
        "InscClaimAmtReimbursed", "DeductibleAmtPaid", "AttendingPhysician",
        "ClmDiagnosisCode_1",
        "is_inpatient", "stay_days", "n_diagnoses", "n_procedures",
        "claim_duration_days",
    ]
    return pd.concat([inp[keep], out[keep]], ignore_index=True)


def build_provider_aggregates(
    claims_with_bene: pd.DataFrame,
) -> pd.DataFrame:
    """One row per Provider with summary stats over their claim history."""
    def _mean_when_positive(series):
        positive = series[series > 0]
        return positive.mean() if len(positive) else 0.0

    agg = claims_with_bene.groupby("Provider").agg(
        prov_n_claims=("ClaimID", "size"),
        prov_n_unique_bene=("BeneID", "nunique"),
        prov_n_unique_attending=("AttendingPhysician", "nunique"),
        prov_n_unique_dx1=("ClmDiagnosisCode_1", "nunique"),
        prov_n_unique_states=("State", "nunique"),
        prov_total_reimbursed=("InscClaimAmtReimbursed", "sum"),
        prov_mean_reimbursed=("InscClaimAmtReimbursed", "mean"),
        prov_std_reimbursed=("InscClaimAmtReimbursed", "std"),
        prov_max_reimbursed=("InscClaimAmtReimbursed", "max"),
        prov_total_deductible=("DeductibleAmtPaid", "sum"),
        prov_pct_inpatient=("is_inpatient", "mean"),
        prov_mean_stay_days_inpat=("stay_days", _mean_when_positive),
        prov_mean_n_diagnoses=("n_diagnoses", "mean"),
        prov_mean_n_procedures=("n_procedures", "mean"),
        prov_mean_claim_duration=("claim_duration_days", "mean"),
        prov_mean_patient_age=("age", "mean"),
        prov_mean_patient_n_chronic=("n_chronic", "mean"),
        prov_max_patient_n_chronic=("n_chronic", "max"),
        prov_pct_patient_dead=("is_dead", "mean"),
        prov_pct_renal=("renal_disease", "mean"),
        prov_first_claim_dt=("ClaimStartDt", "min"),
        prov_last_claim_dt=("ClaimStartDt", "max"),
    )
    agg["prov_active_period_days"] = (
        (agg["prov_last_claim_dt"] - agg["prov_first_claim_dt"]).dt.days
    )
    agg["prov_claims_per_active_day"] = (
        agg["prov_n_claims"] / agg["prov_active_period_days"].clip(lower=1)
    )
    agg["prov_bene_per_claim"] = (
        agg["prov_n_unique_bene"] / agg["prov_n_claims"]
    )
    agg["prov_std_reimbursed"] = agg["prov_std_reimbursed"].fillna(0)
    agg = agg.drop(columns=["prov_first_claim_dt", "prov_last_claim_dt"])

    # Modal state for the provider's beneficiary panel.
    modal_state = (
        claims_with_bene.groupby("Provider")["State"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else -1)
        .rename("prov_modal_state")
    )
    agg = agg.join(modal_state)
    return agg.reset_index()


def fit_target_encoding(
    codes: pd.Series, labels: pd.Series, alpha: float = 20.0,
):
    """Smoothed target encoding: (sum + alpha*global) / (count + alpha)."""
    global_mean = float(labels.mean())
    df = pd.DataFrame({"code": codes, "label": labels}).dropna(subset=["code"])
    grp = df.groupby("code")["label"].agg(["sum", "count"])
    smoothed = (grp["sum"] + alpha * global_mean) / (grp["count"] + alpha)
    return smoothed.to_dict(), global_mean


def apply_target_encoding(
    codes: pd.Series, mapping: dict, default: float,
) -> pd.Series:
    return codes.map(mapping).fillna(default).astype(float)


def main():
    t0 = time.time()
    print("Loading raw data...")
    train_labels, bene, inp, out = load_raw()
    print(f"  providers={len(train_labels):,}  beneficiaries={len(bene):,}  "
          f"inp={len(inp):,}  out={len(out):,}")

    print("Building per-beneficiary features...")
    bene_feat = build_beneficiary_features(bene)

    print("Building per-claim features...")
    claims = build_per_claim_features(inp, out)
    print(f"  claim rows={len(claims):,}")

    print("Joining beneficiary features onto claims...")
    claims = claims.merge(bene_feat, on="BeneID", how="left")

    print("Aggregating per-provider features...")
    prov_agg = build_provider_aggregates(claims)
    print(f"  providers with claims={len(prov_agg):,}")

    print("Joining provider aggregates + label onto claim rows...")
    claims = claims.merge(prov_agg, on="Provider", how="left")
    claims = claims.merge(train_labels, on="Provider", how="inner")
    claims["label"] = (claims["PotentialFraud"] == "Yes").astype(int)
    claims = claims.drop(columns=["PotentialFraud"])

    print("Stratified 80/20 split by Provider (avoids label leakage)...")
    providers = train_labels.copy()
    providers["label"] = (providers["PotentialFraud"] == "Yes").astype(int)
    train_providers, test_providers = train_test_split(
        providers["Provider"],
        test_size=TEST_SIZE,
        stratify=providers["label"],
        random_state=RANDOM_STATE,
    )
    train_set = set(train_providers)
    train_mask = claims["Provider"].isin(train_set)
    train_claims = claims[train_mask].copy()
    test_claims = claims[~train_mask].copy()
    n_tr_prov = train_claims["Provider"].nunique()
    n_ts_prov = test_claims["Provider"].nunique()
    print(f"  train: {len(train_claims):,} claims "
          f"from {n_tr_prov:,} providers "
          f"(fraud rate {train_claims['label'].mean():.4f})")
    print(f"  test:  {len(test_claims):,} claims "
          f"from {n_ts_prov:,} providers "
          f"(fraud rate {test_claims['label'].mean():.4f})")

    print("Fitting target encoding for ClmDiagnosisCode_1 (train only)...")
    enc_map, global_mean = fit_target_encoding(
        train_claims["ClmDiagnosisCode_1"], train_claims["label"], alpha=20.0
    )
    train_claims["dx1_target_enc"] = apply_target_encoding(
        train_claims["ClmDiagnosisCode_1"], enc_map, global_mean
    )
    test_claims["dx1_target_enc"] = apply_target_encoding(
        test_claims["ClmDiagnosisCode_1"], enc_map, global_mean
    )

    # Gender / Race are already integer-coded in source; cast to be sure.
    for col in ("Gender", "Race"):
        train_claims[col] = train_claims[col].astype("int8")
        test_claims[col] = test_claims[col].astype("int8")

    # Drop identifier and raw-categorical columns the model should not see.
    drop_cols = [
        "ClaimID", "BeneID", "Provider", "AttendingPhysician",
        "ClmDiagnosisCode_1",  # replaced by dx1_target_enc
        "ClaimStartDt", "ClaimEndDt",  # kept only in claim_stream for ordering
        "County",  # high-cardinality; not used in this baseline
    ]
    feature_cols = [
        c for c in train_claims.columns
        if c not in drop_cols + ["label"]
    ]

    # Final outputs: keep identifiers + dates in the streaming output so the
    # producer / dashboard can display them; the model-input matrices come from
    # `feature_cols`.
    print("Writing parquet outputs...")
    train_claims.to_parquet(PROCESSED / "claims_train.parquet", index=False)
    test_claims.to_parquet(PROCESSED / "claims_test.parquet", index=False)
    claim_stream = (
        test_claims.sort_values("ClaimStartDt").reset_index(drop=True)
    )
    claim_stream.to_parquet(
        PROCESSED / "claim_stream.parquet", index=False,
    )

    print("Writing metadata...")
    (PROCESSED / "target_encoding_dx1.json").write_text(json.dumps({
        "global_mean": global_mean,
        "alpha": 20.0,
        "mapping": {str(k): float(v) for k, v in enc_map.items()},
    }))
    (PROCESSED / "feature_columns.json").write_text(json.dumps({
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "label_column": "label",
        "stream_order_column": "ClaimStartDt",
        "n_train_claims": int(len(train_claims)),
        "n_test_claims": int(len(test_claims)),
        "n_train_providers": int(train_claims["Provider"].nunique()),
        "n_test_providers": int(test_claims["Provider"].nunique()),
        "train_fraud_rate": float(train_claims["label"].mean()),
        "test_fraud_rate": float(test_claims["label"].mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }, indent=2))

    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  features = {len(feature_cols)}")
    print(f"  feature list:")
    for c in feature_cols:
        print(f"    - {c}")


if __name__ == "__main__":
    main()
