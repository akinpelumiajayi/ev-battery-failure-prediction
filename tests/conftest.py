"""
Shared fixtures. Tests are hermetic: they build a small synthetic dataset
with the real raw schema and train throw-away artifacts in a temp dir, so
the suite runs in seconds and never depends on `dvc repro` having run.
"""

from __future__ import annotations

import json
import typing

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.build_features import build_preprocessor, sanitize_feature_names
from src.serving.schemas import VehicleFeatures

TARGET = "battery_failure"
DROP = ["vehicle_id", "battery_serial"]

CATEGORIES = {
    "vehicle_brand": ["Tesla", "BYD", "Ford", "Kia", "Mercedes-Benz"],
    "vehicle_model": ["Model 3", "Atto 3", "Mustang Mach-E", "EV6"],
    "vehicle_type": ["Sedan", "SUV", "Truck"],
    "battery_manufacturer": ["CATL", "Panasonic", "LG Energy Solution"],
    "battery_chemistry": ["NMC", "LFP", "NCA"],
    "drive_type": ["AWD", "RWD", "FWD"],
    "fleet_or_private": ["Fleet", "Private"],
    "terrain_type": ["Flat", "Hilly", "Desert"],
}


def _is_str_field(annotation) -> bool:
    return str in typing.get_args(annotation) or annotation is str


def make_raw_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "vehicle_id": [f"EV{i:06d}" for i in range(n)],
        "battery_serial": [f"BAT-{i:08d}" for i in range(n)],
    }
    for name, field in VehicleFeatures.model_fields.items():
        if _is_str_field(field.annotation):
            data[name] = rng.choice(CATEGORIES.get(name, ["A", "B"]), size=n)
        else:
            data[name] = rng.uniform(1, 90, size=n)
    df = pd.DataFrame(data)
    df["manufacturing_year"] = rng.integers(2015, 2025, size=n).astype(float)
    df["state_of_health"] = rng.uniform(50, 100, size=n)
    df["battery_health_percent"] = df["state_of_health"] + rng.normal(0, 1, n)
    df["battery_health_percent"] = df["battery_health_percent"].clip(0, 100)
    df["cell_voltage_avg"] = rng.uniform(3.1, 3.85, size=n)
    df["fast_charge_ratio"] = rng.uniform(0, 1, size=n)
    df["highway_driving_ratio"] = rng.uniform(0, 1, size=n)
    df["internal_resistance"] = rng.uniform(0.1, 1.1, size=n)
    logit = -2.5 + 0.08 * (75 - df["state_of_health"]) + 3 * (df["internal_resistance"] - 0.5)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # sprinkle missing values like the real data (~4%)
    for col in ("cycle_count", "pack_voltage", "vehicle_type"):
        df.loc[rng.random(n) < 0.05, col] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, raw_df):
    """Fit preprocessor + a logistic model and write them like the pipeline does."""
    root = tmp_path_factory.mktemp("artifacts")
    artifacts_dir, models_dir = root / "artifacts", root / "models"
    artifacts_dir.mkdir()
    models_dir.mkdir()

    pre, num_cols, cat_cols = build_preprocessor(raw_df, DROP, TARGET)
    raw_cols = num_cols + cat_cols
    X = pre.fit_transform(raw_df[raw_cols])
    names = sanitize_feature_names(pre.get_feature_names_out())
    X_df = pd.DataFrame(X, columns=names)
    y = raw_df[TARGET]
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_df, y)

    joblib.dump(pre, artifacts_dir / "preprocessor.joblib")
    joblib.dump(
        {
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "drop_cols": DROP,
            "feature_names": names,
            "feature_means": X.mean(axis=0).tolist(),
        },
        artifacts_dir / "feature_metadata.joblib",
    )
    joblib.dump(model, models_dir / "model.joblib")
    (models_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression",
                "decision_threshold": 0.4,
                "risk_band_medium_ratio": 0.5,
                "git_sha": "abcdef1234567890",
            }
        )
    )
    (models_dir / "test_metrics.json").write_text(json.dumps({"test_roc_auc": 0.9, "test_pr_auc": 0.6}))
    return {"artifacts_dir": artifacts_dir, "models_dir": models_dir, "X": X_df, "y": y}
