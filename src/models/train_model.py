import logging
import os
import pickle

import pandas as pd
import yaml
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("train_model")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def main():
    try:
        params = load_params()["model_training"]

        train_df = pd.read_csv("data/processed/train_features.csv")

        target_col = "Time_taken(min)"
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        model = LGBMRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.debug("Model training completed. Saved to models/model.pkl")

    except Exception as e:
        logger.error("Failed to complete model training: %s", e)
        raise


if __name__ == "__main__":
    main()
