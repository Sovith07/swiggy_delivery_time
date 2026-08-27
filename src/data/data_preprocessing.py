import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("data_preprocessing")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names if needed
    df.columns = [c.strip() for c in df.columns]

    # Example cleaning steps for this dataset — adjust to actual schema
    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = (
            df["Weatherconditions"].astype(str).str.replace("conditions ", "", regex=False)
        )

    if "Time_taken(min)" in df.columns:
        df["Time_taken(min)"] = (
            df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)
        )

    # Drop rows with missing target
    target_col = "Time_taken(min)"
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])

    # Replace sentinel/invalid coordinates (0.0) with NaN, then drop
    coord_cols = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for col in coord_cols:
        if col in df.columns:
            df[col] = df[col].replace(0.0, np.nan)

    df = df.dropna(subset=[c for c in coord_cols if c in df.columns])

    return df


def main():
    try:
        train = pd.read_csv("data/raw/train.csv")
        test = pd.read_csv("data/raw/test.csv")

        train_clean = clean_data(train)
        test_clean = clean_data(test)

        os.makedirs("data/interim", exist_ok=True)
        train_clean.to_csv("data/interim/train_processed.csv", index=False)
        test_clean.to_csv("data/interim/test_processed.csv", index=False)

        logger.debug("Data preprocessing completed. Output saved to data/interim")

    except Exception as e:
        logger.error("Failed to complete data preprocessing: %s", e)
        raise


if __name__ == "__main__":
    main()
