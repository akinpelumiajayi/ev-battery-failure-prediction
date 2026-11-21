import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_preprocessor, sanitize_feature_names

DROP = ["vehicle_id", "battery_serial"]


def _toy_df():
    return pd.DataFrame(
        {
            "vehicle_id": ["A", "B", "C", "D"],
            "battery_serial": ["S1", "S2", "S3", "S4"],
            "battery_health_percent": [80.0, np.nan, 60.0, 90.0],
            "vehicle_type": ["Sedan", "Truck", np.nan, "Sedan"],
            "battery_failure": [0, 1, 0, 0],
        }
    )


def test_preprocessor_imputes_and_splits_column_types():
    df = _toy_df()
    pre, num, cat = build_preprocessor(df, DROP, "battery_failure")
    X = pre.fit_transform(df[num + cat])
    assert not np.isnan(X).any()
    assert num == ["battery_health_percent"]
    assert cat == ["vehicle_type"]
    assert X.shape[0] == len(df)


def test_identifiers_and_target_are_never_features():
    _, num, cat = build_preprocessor(_toy_df(), DROP, "battery_failure")
    assert not {"vehicle_id", "battery_serial", "battery_failure"} & set(num + cat)


def test_unseen_category_at_inference_does_not_crash():
    df = _toy_df()
    pre, num, cat = build_preprocessor(df, DROP, "battery_failure")
    pre.fit(df[num + cat])
    new = pd.DataFrame({"battery_health_percent": [70.0], "vehicle_type": ["Spaceship"]})
    X = pre.transform(new[num + cat])
    assert X.shape == (1, 1 + 2)  # 1 numeric + 2 known categories (all-zero one-hot)


def test_preprocessor_respects_impute_strategy():
    df = _toy_df()
    pre, num, cat = build_preprocessor(df, DROP, "battery_failure", numeric_strategy="mean")
    pre.fit(df[num + cat])
    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    assert imputer.strategy == "mean"
    assert imputer.statistics_[0] == pytest.approx(np.nanmean([80, 60, 90]))


def test_sanitize_feature_names():
    assert sanitize_feature_names(["cat__vehicle_model_Model S/X", "num__a"]) == [
        "cat__vehicle_model_Model_S_X",
        "num__a",
    ]
    with pytest.raises(ValueError, match="duplicates"):
        sanitize_feature_names(["a b", "a/b"])
