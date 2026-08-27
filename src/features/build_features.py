import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("build_features")

R = 6371  # Earth radius in km


def deg_to_rad(deg):
    return deg * (np.pi / 180)


def haversine_distance(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = (
        np.sin(d_lat / 2) ** 2
        + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_km"] = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = [
        c
        for c in [
            "Weatherconditions",
            "Road_traffic_density",
            "Type_of_order",
            "Type_of_vehicle",
            "Festival",
            "City",
        ]
        if c in df.columns
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_distance_feature(df)
    df = encode_categoricals(df)

    # Drop identifier / raw text columns not useful as model features
    drop_cols = [
        c
        for c in ["ID", "Delivery_person_ID", "Order_Date", "Time_Orderd", "Time_Order_picked"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    return df


def main():
    try:
        train = pd.read_csv("data/interim/train_processed.csv")
        test = pd.read_csv("data/interim/test_processed.csv")

        train_feat = build_features(train)
        test_feat = build_features(test)

        # Align columns between train/test (encoding may create mismatched dummy columns)
        train_feat, test_feat = train_feat.align(test_feat, join="left", axis=1, fill_value=0)

        os.makedirs("data/processed", exist_ok=True)
        train_feat.to_csv("data/processed/train_features.csv", index=False)
        test_feat.to_csv("data/processed/test_features.csv", index=False)

        logger.debug("Feature engineering completed. Output saved to data/processed")

    except Exception as e:
        logger.error("Failed to complete feature engineering: %s", e)
        raise


if __name__ == "__main__":
    main()
