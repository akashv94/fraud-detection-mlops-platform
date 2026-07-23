import importlib
import os

os.environ["MODEL_PATH"] = "models/nonexistent-test-model.joblib"

import src.api.main as main
importlib.reload(main)

from fastapi.testclient import TestClient

client = TestClient(main.app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Fraud Detection MLOps API"


def test_health_without_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_model_info_without_model():
    response = client.get("/model/info")
    assert response.status_code == 503


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_prediction_requests_total" in response.text
