from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY as _GLOBAL_REGISTRY,
)
from starlette.responses import Response

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_model.joblib"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))

app = FastAPI(
    title="Fraud Detection MLOps API",
    version="1.0.0",
    description="Production-style fraud detection inference service.",
)

artifact = None
model = None
FEATURES = []
THRESHOLD = 0.5
MODEL_VERSION = "unknown"

_PROM_REGISTRY = CollectorRegistry()


def _get_or_create_metric(ctor, name, *args, **kwargs):
    try:
        return ctor(name, *args, registry=_PROM_REGISTRY, **kwargs)
    except ValueError:
        # metric already registered in a registry; try to find and reuse it
        for reg in (_PROM_REGISTRY, _GLOBAL_REGISTRY):
            existing = getattr(reg, "_names_to_collectors", {}).get(name)
            if existing is not None:
                return existing
        raise


REQUEST_COUNTER = _get_or_create_metric(
    Counter, "fraud_prediction_requests_total", "Total fraud prediction requests"
)
FRAUD_COUNTER = _get_or_create_metric(
    Counter, "fraud_predictions_total", "Transactions predicted as fraud"
)
ERROR_COUNTER = _get_or_create_metric(
    Counter, "fraud_prediction_errors_total", "Prediction errors"
)
LATENCY = _get_or_create_metric(
    Histogram, "fraud_prediction_latency_seconds", "Prediction latency in seconds"
)


def load_model():
    global artifact, model, FEATURES, THRESHOLD, MODEL_VERSION
    if not MODEL_PATH.exists():
        return False
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    FEATURES = artifact["features"]
    THRESHOLD = float(artifact.get("threshold", 0.5))
    MODEL_VERSION = artifact.get("model_version", "1.0.0")
    return True


load_model()


def transaction_schema():
    fields = {
        "Time": (float, ...),
        **{f"V{i}": (float, ...) for i in range(1, 29)},
        "Amount": (float, ...),
    }
    return create_model("Transaction", **fields)


Transaction = transaction_schema()


class BatchRequest(BaseModel):
    transactions: list[Transaction]


def ensure_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Train the model and provide models/fraud_model.joblib.",
        )


def log_prediction(probability: float, prediction: int, latency_ms: float):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "prediction": prediction,
        "fraud_probability": probability,
        "latency_ms": latency_ms,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def infer(values: dict):
    ensure_model()
    df = pd.DataFrame([values])
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    df = df[FEATURES]
    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= THRESHOLD)
    return probability, prediction


@app.get("/")
def root():
    return {
        "service": "Fraud Detection MLOps API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
    }


@app.get("/model/info")
def model_info():
    ensure_model()
    return {
        "model": type(model).__name__,
        "number_of_features": len(FEATURES),
        "threshold": THRESHOLD,
        "version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    REQUEST_COUNTER.inc()
    start = time.perf_counter()
    try:
        probability, prediction = infer(transaction.model_dump())
        if prediction == 1:
            FRAUD_COUNTER.inc()
        latency = time.perf_counter() - start
        log_prediction(probability, prediction, latency * 1000)
        return {
            "prediction": prediction,
            "label": "fraud" if prediction else "legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": THRESHOLD,
            "model_version": MODEL_VERSION,
        }
    except HTTPException:
        ERROR_COUNTER.inc()
        raise
    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start)


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for item in request.transactions:
        REQUEST_COUNTER.inc()
        start = time.perf_counter()
        try:
            probability, prediction = infer(item.model_dump())
            if prediction:
                FRAUD_COUNTER.inc()
            latency = time.perf_counter() - start
            log_prediction(probability, prediction, latency * 1000)
            results.append(
                {
                    "prediction": prediction,
                    "label": "fraud" if prediction else "legitimate",
                    "fraud_probability": round(probability, 6),
                }
            )
        except Exception as exc:
            ERROR_COUNTER.inc()
            results.append({"error": str(exc)})
        finally:
            LATENCY.observe(time.perf_counter() - start)
    return {"count": len(results), "results": results}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(_PROM_REGISTRY), media_type=CONTENT_TYPE_LATEST)
