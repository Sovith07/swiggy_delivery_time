import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("data_ingestion")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logger.debug("Parameters retrieved from params.yaml")
    return params


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


def save_data(train: pd.DataFrame, test: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)


def main():
    try:
        params = load_params()
        cfg = params["data_ingestion"]

        df = load_data(cfg["data_source"])

        train, test = train_test_split(
            df,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
        )

        save_data(train, test, output_dir="data/raw")
        logger.debug("Data ingestion completed. Train/test saved to data/raw")

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        raise


if __name__ == "__main__":
    main()
