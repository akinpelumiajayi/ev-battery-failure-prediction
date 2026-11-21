"""
Preprocessing stage: stratified train / validation / test split.

Deliberately does no imputation or encoding — that lives in
`src/features/build_features.py` so it is fit on the training fold only and
reused unchanged at serving time (no target leakage, no train/serve skew).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED_DIR, PROJECT_ROOT, REFERENCE_DIR, get_logger, load_params

logger = get_logger(__name__)


def split(df: pd.DataFrame, target: str, test_size: float, val_size: float, seed: int):
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=seed, stratify=train_val[target]
    )
    return train, val, test


def preprocess() -> None:
    params = load_params()
    target = params["target_column"]
    cfg = params["split"]

    df = pd.read_csv(PROJECT_ROOT / params["data"]["raw_path"])
    train, val, test = split(df, target, cfg["test_size"], cfg["val_size"], cfg["random_state"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    # Snapshot of the training distribution — the baseline that
    # src/monitoring/drift_check.py compares live / incoming data against.
    train.drop(columns=[target]).to_parquet(REFERENCE_DIR / "reference.parquet", index=False)

    logger.info(
        "Split -> train=%d val=%d test=%d (positive rate %.3f / %.3f / %.3f)",
        len(train),
        len(val),
        len(test),
        train[target].mean(),
        val[target].mean(),
        test[target].mean(),
    )


if __name__ == "__main__":
    preprocess()
