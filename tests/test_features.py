import numpy as np
import pandas as pd

from src.features.build_features import build_preprocessor


def _toy_df():
    return pd.DataFrame({
        "vehicle_id": ["A", "B", "C", "D"],
        "battery_serial": ["S1", "S2", "S3", "S4"],
        "battery_health_percent": [80.0, np.nan, 60.0, 90.0],
        "vehicle_type": ["Sedan", "Truck", np.nan, "Sedan"],
        "battery_failure": [0, 1, 0, 0],
    })


def test_preprocessor_handles_missing_values():
    df = _toy_df()
    drop_cols = ["vehicle_id", "battery_serial"]
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, drop_cols, "battery_failure")

    X = preprocessor.fit_transform(df.drop(columns=drop_cols + ["battery_failure"]))

    assert not np.isnan(X).any(), "Imputer should remove all NaNs"
    assert "battery_health_percent" in numeric_cols
    assert "vehicle_type" in categorical_cols


def test_preprocessor_output_row_count_matches_input():
    df = _toy_df()
    drop_cols = ["vehicle_id", "battery_serial"]
    preprocessor, _, _ = build_preprocessor(df, drop_cols, "battery_failure")
    X = preprocessor.fit_transform(df.drop(columns=drop_cols + ["battery_failure"]))
    assert X.shape[0] == len(df)
