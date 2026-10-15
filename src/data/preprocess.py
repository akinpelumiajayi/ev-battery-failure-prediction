"""
Preprocessing stage.

Splits the validated raw data into train/val/test and persists them to
data/processed/. Kept deliberately "dumb" (no imputation/encoding here) —
that logic lives in src/features/build_features.py as a sklearn Pipeline
so it is fit ONLY on the training fold and reused identically at serving
time (prevents train/serve skew and target leakage).
"""
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/ev_battery_data.csv")
OUT_DIR = Path("data/processed")
REFERENCE_DIR = Path("data/reference")  # snapshot used later for drift monitoring


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def preprocess():
    params = load_params()
    target = params["target_column"]
    split_cfg = params["split"]

    df = pd.read_csv(RAW_PATH)
    logger.info("Loaded %d rows", len(df))

    # First split off test, then split remaining into train/val
    train_val, test = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=df[target],
    )
    val_ratio = split_cfg["val_size"] / (1 - split_cfg["test_size"])
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=split_cfg["random_state"],
        stratify=train_val[target],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)

    # Reference snapshot for monitoring/drift_check.py — represents the
    # distribution the model was trained on, to compare live traffic against.
    train.drop(columns=[target]).to_parquet(REFERENCE_DIR / "reference.parquet", index=False)

    logger.info(
        "Split -> train=%d val=%d test=%d (positive rates: %.3f / %.3f / %.3f)",
        len(train), len(val), len(test),
        train[target].mean(), val[target].mean(), test[target].mean(),
    )


if __name__ == "__main__":
    preprocess()
