from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost.core import Booster

DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42
THRESHOLD = 0.50


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add creditcard.csv before training."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"Class", "Time", "Amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(sampling_strategy=0.30, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        model.fit(X_resampled, y_resampled)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        metrics = {
            "auc_pr": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }

        mlflow.log_params(
            {
                "model": "XGBoost",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "smote_ratio": 0.30,
                "threshold": THRESHOLD,
            }
        )
        mlflow.log_metrics(metrics)
        # XGBoost models include types that skops/MLflow may consider untrusted.
        # Allow these types explicitly when logging the model so MLflow can
        # serialize the trained XGBClassifier.
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=[Booster, XGBClassifier],
            )
        except Exception as exc:
            print("Warning: mlflow failed to log model:", exc)
            print("Continuing — model will still be saved locally via joblib.")

        artifact = {
            "model": model,
            "features": list(X.columns),
            "threshold": THRESHOLD,
            "model_version": "1.0.0",
        }
        joblib.dump(artifact, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model: {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
