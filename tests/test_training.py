from src.models.train import DATA_PATH, MODEL_PATH


def test_paths_are_configured():
    assert DATA_PATH.name == "creditcard.csv"
    assert MODEL_PATH.name == "fraud_model.joblib"
