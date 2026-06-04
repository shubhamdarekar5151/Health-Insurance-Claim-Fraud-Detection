# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------
# Shared image for the real-time fraud-detection demo.
#
# One image serves all three processes (producer, consumer, dashboard);
# docker-compose selects which one to run via the service `command`.
# ---------------------------------------------------------------------
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime that XGBoost links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Shared SQLite DB lives on a mounted volume so all services see it.
    CLAIMS_DB=/data/claims.db

WORKDIR /app

# Install dependencies first so the layer is cached across source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the pre-built artifacts the demo needs at runtime:
#   src/             producer, consumer, dashboard, ml, db helpers
#   models/          trained xgb_fraud.pkl + metrics JSON
#   data/processed/  engineered parquet streams + feature columns
#   docs/screenshots model-performance plots shown in the dashboard
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY docs/ ./docs/
COPY setup.cfg README.md ./

# Volume mount point for the shared SQLite database (and its WAL sidecars).
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8501

# Default to the dashboard; producer/consumer override this in compose.
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
