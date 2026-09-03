import pytest
import mlflow
import dagshub
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import os


# --------------------------------
# DagsHub / MLflow configuration
# --------------------------------

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Sovith07"
repo_name = "swiggy_delivery_time"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)


# --------------------------------
# Paths
# --------------------------------

root_path = Path(__file__).parent.parent

preprocessor_path = (
    root_path / "models" / "preprocessor.joblib"
)

test_data_path = (
    root_path / "data" / "processed" / "test.csv"
)


# --------------------------------
# Load preprocessor
# --------------------------------

preprocessor = joblib.load(preprocessor_path)


# --------------------------------
# Load registered model
# --------------------------------

model_name = "delivery_time_pred_model"

model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/staging"
)


# --------------------------------
# Test model performance
# --------------------------------

def test_model_performance():

    # Load test data
    df = pd.read_csv(test_data_path)

    # Separate X and y
    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    # Preprocess features
    X_processed = preprocessor.transform(X)

    # Predict
    y_pred = model.predict(X_processed)

    # Calculate MAE
    mean_error = mean_absolute_error(y, y_pred)

    print(f"Average error: {mean_error:.4f} minutes")

    # Performance threshold
    assert mean_error <= 5, (
        f"The model does not pass the performance threshold "
        f"of 5 minutes. Actual MAE: {mean_error:.4f}"
    )

    print(
        f"The {model_name} model passed the performance test"
    )