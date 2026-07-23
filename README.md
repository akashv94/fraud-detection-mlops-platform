# Fraud Detection End-to-End MLOps Capstone

A production-style portfolio project demonstrating:

- XGBoost fraud detection
- SMOTE for class imbalance
- MLflow experiment tracking
- FastAPI inference service
- pytest automated tests
- Docker + Docker Compose
- GitHub Actions CI/CD
- GitHub Container Registry (GHCR)
- Google Cloud Run deployment workflow
- Prometheus metrics
- Grafana provisioning
- Evidently data drift reports
- Streamlit demo UI
- Prediction logging

## Architecture

Dataset -> Training -> MLflow -> Model Artifact -> FastAPI -> Docker -> GitHub Actions -> GHCR -> Cloud Run

FastAPI -> Prometheus -> Grafana
Production Data -> Evidently -> Drift Report
Streamlit -> FastAPI

## 1. Setup on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Add dataset

Place the standard credit-card fraud dataset at:

```text
data/creditcard.csv
```

Expected columns:

```text
Time,V1,V2,...,V28,Amount,Class
```

`Class=1` means fraud.

## 3. Train

```bash
python -m src.models.train
```

Outputs:

- `models/fraud_model.joblib`
- MLflow experiment logs under `mlruns/`

Launch MLflow UI:

```bash
mlflow ui
```

Then visit the local URL printed by MLflow.

## 4. Run tests

```bash
pytest -v
```

## 5. Run API

```bash
uvicorn src.api.main:app --reload
```

Useful endpoints:

- `/`
- `/health`
- `/model/info`
- `/predict`
- `/predict/batch`
- `/metrics`
- `/docs`

## 6. Streamlit UI

With the API running:

```bash
streamlit run dashboard/app.py
```

## 7. Docker

```bash
docker build -t fraud-mlops-api .
docker run -p 8080:8080 fraud-mlops-api
```

## 8. Full local observability stack

The trained model must exist before starting the stack.

```bash
docker compose up --build
```

Services:

- FastAPI: port 8080
- Prometheus: port 9090
- Grafana: port 3000
- Streamlit: port 8501

Grafana default login is normally `admin` / `admin` for a fresh local container. The provisioned Prometheus data source is configured automatically.

## 9. Drift report

Create:

- `data/reference/reference.csv`
- `data/production/current.csv`

Then:

```bash
python -m src.monitoring.drift
```

Output:

```text
reports/drift/drift_report.html
```

## 10. GitHub Actions

`.github/workflows/ci-cd.yml` performs:

1. dependency installation
2. tests
3. Docker build
4. GHCR push
5. optional Google Cloud Run deployment

The Cloud Run deployment job requires these GitHub repository secrets:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_SERVICE`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

The workflow uses Google Workload Identity Federation rather than storing a service-account JSON key.

Before enabling deployment, configure the Google Cloud project, Artifact/GHCR access strategy, Cloud Run service permissions, and the GitHub OIDC identity provider.

## Notes

This repository intentionally does not contain the credit-card dataset or a trained model. Train locally after adding the dataset.

For a real production system, move prediction logs to durable storage, use a managed metrics backend compatible with your cloud architecture, add authentication, secrets management, model registry promotion, alerting, and scheduled drift jobs.
