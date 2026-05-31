# Real-Time Health Insurance Claim Fraud Detection

A capstone project that scores health-insurance claims for fraud as they arrive
on a simulated stream, and visualises results in an interactive Streamlit
dashboard.

## Architecture

```
[CMS Kaggle CSVs] --> [Stream Simulator] --> [Scoring Service] --> [SQLite] --> [Streamlit Dashboard]
   (static data)        producer.py             consumer.py          claims.db      app.py
                        (1 claim / sec          (XGBoost              (results       (live feed,
                         from test set)          predict_proba)        log table)     KPIs, SHAP)
```

Three independent Python processes communicate via a shared SQLite database, so
each piece can be developed, restarted, and demoed in isolation.

## Project Layout

```
qwerty/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Kaggle CSVs (not committed)
│   └── processed/        # Engineered features (parquet)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── ml/
│   │   ├── features.py   # Feature engineering pipeline
│   │   ├── train.py      # XGBoost training + tuning
│   │   └── evaluate.py   # P / R / F1 / ROC-AUC / PR-AUC
│   ├── stream/
│   │   ├── producer.py   # Replays test set into SQLite
│   │   └── consumer.py   # Loads model, scores claims, writes results
│   ├── dashboard/
│   │   └── app.py        # Streamlit UI
│   └── db.py             # SQLite schema + helpers
├── models/
│   └── xgb_fraud.pkl     # Trained model artifact (built on Day 3)
└── docs/
    ├── report.md
    └── screenshots/
```

## Setup

```bash
# 1. Create and activate the venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Download the Dataset

The project uses the public Kaggle dataset
**Healthcare Provider Fraud Detection Analysis** (CMS Medicare).

You need a Kaggle API token. Go to
[kaggle.com/settings/account](https://www.kaggle.com/settings/account) →
**Create New API Token**. Save the resulting `KGAT_...` string to
`~/.kaggle/access_token` with mode 600:

```bash
mkdir -p ~/.kaggle
printf '%s' 'KGAT_your_token_here' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Then download via the `kagglehub` library (already in `requirements.txt`):

```bash
KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" python -c "
import kagglehub, shutil, os
path = kagglehub.dataset_download('rohitrox/healthcare-provider-fraud-detection-analysis')
print('Downloaded to:', path)
# Symlink into data/raw/ with canonical names (strip Kaggle timestamps).
mapping = {
    'Train-1542865627584.csv':                  'Train.csv',
    'Train_Beneficiarydata-1542865627584.csv':  'Train_Beneficiarydata.csv',
    'Train_Inpatientdata-1542865627584.csv':    'Train_Inpatientdata.csv',
    'Train_Outpatientdata-1542865627584.csv':   'Train_Outpatientdata.csv',
    'Test-1542969243754.csv':                   'Test.csv',
    'Test_Beneficiarydata-1542969243754.csv':   'Test_Beneficiarydata.csv',
    'Test_Inpatientdata-1542969243754.csv':     'Test_Inpatientdata.csv',
    'Test_Outpatientdata-1542969243754.csv':    'Test_Outpatientdata.csv',
}
os.makedirs('data/raw', exist_ok=True)
for src, dst in mapping.items():
    os.symlink(os.path.join(path, src), os.path.join('data/raw', dst))
"
```

After this, `data/raw/` should contain symlinks for the 4 Train CSVs and 4
Test CSVs. Only the `Train_*` files are used; the `Test_*` files are the
unlabeled Kaggle holdout (no provider-level labels), so they cannot be used
for supervised training or evaluation.

## Run the End-to-End Demo

```bash
# 1. Train the model (one-off, ~5-10 min)
python -m src.ml.train

# 2. Sanity-check metrics
python -m src.ml.evaluate

# 3. Wipe runtime DB and start the three processes
rm -f claims.db
python -m src.stream.producer --rate 1 &
python -m src.stream.consumer &
streamlit run src/dashboard/app.py --server.headless=true
```

Open http://localhost:8501 — KPIs should refresh every ~2 s and flagged
claims should start appearing within a minute.

> **First-run note.** Without `--server.headless=true`, Streamlit prompts
> on the terminal for an email address the first time it starts; if you
> ignore the prompt the server never binds to the port and the browser
> just spins. Either pass `--server.headless=true` (as above), or run
> this once to silence the prompt permanently:
>
> ```bash
> mkdir -p ~/.streamlit
> printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
> ```

## Capstone Objectives

| # | Objective                          | Where it lives                                          |
|---|------------------------------------|---------------------------------------------------------|
| 1 | Real-time fraud detection          | `src/stream/` + `src/dashboard/app.py`                  |
| 2 | Interactive analytics dashboard    | `src/dashboard/app.py`                                  |
| 3 | Accuracy + scale optimisation      | `src/ml/train.py` (XGBoost + class imbalance + tuning)  |
| 4 | Evaluation (P / R / F1 / AUC)      | `src/ml/evaluate.py` + Model tab of the dashboard       |

Full methodology, results, and limitations are in
[`docs/report.md`](docs/report.md).
