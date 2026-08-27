import json
import logging
import os
import pickle

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("evaluate_model")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow_tracking(repo_owner: str, repo_name: str) -> None:
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")


def main():
    try:
        params = load_params()

        setup_mlflow_tracking(
            repo_owner="Sovith07",
            repo_name="swiggy_delivery_time",  # update to your actual repo name
        )
        mlflow.set_experiment("dvc-pipeline-runs")

        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv("data/processed/test_features.csv")
        target_col = "Time_taken(min)"
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}

        with mlflow.start_run():
            mlflow.log_params(params["model_training"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.debug("Model evaluation completed: %s", metrics)

    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        raise


if __name__ == "__main__":
    main()
