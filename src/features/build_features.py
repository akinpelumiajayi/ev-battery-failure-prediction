"""
Feature engineering stage.

Builds one sklearn ColumnTransformer (imputation + scaling + one-hot), fits
it on the training fold ONLY, applies it to all three folds, and persists
the fitted object to `artifacts/preprocessor.joblib`. The serving layer loads
that exact object, so the API never re-implements feature logic — the
single most common cause of train/serve skew.
"""

from __future__ import annotations

import re

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import ARTIFACTS_DIR, PROCESSED_DIR, get_logger, load_params

logger = get_logger(__name__)


def sanitize_feature_names(names) -> list[str]:
    """One-hot names embed raw category values (e.g. 'Model S/X'); tree libraries
    like LightGBM reject JSON-special characters, so normalise to [A-Za-z0-9_]."""
    clean = [re.sub(r"[^0-9A-Za-z_]+", "_", str(n)).strip("_") for n in names]
    if len(set(clean)) != len(clean):
        raise ValueError("Feature-name sanitisation produced duplicates")
    return clean


def build_preprocessor(
    df: pd.DataFrame,
    drop_cols: list[str],
    target_col: str,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in [*drop_cols, target_col]]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=numeric_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def build_features() -> None:
    params = load_params()
    target = params["target_column"]
    fcfg = params["features"]
    drop_cols = fcfg["drop_columns"]

    folds = {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in ("train", "val", "test")}

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        folds["train"],
        drop_cols,
        target,
        numeric_strategy=fcfg["numeric_impute_strategy"],
        categorical_strategy=fcfg["categorical_impute_strategy"],
    )
    raw_cols = numeric_cols + categorical_cols
    logger.info(
        "Fitting preprocessor on %d numeric + %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )
    preprocessor.fit(folds["train"][raw_cols])
    feature_names = sanitize_feature_names(preprocessor.get_feature_names_out())
    # Mean of each transformed feature on the train fold: the SHAP baseline the
    # API uses to explain linear models (contribution = coef * (x - mean)).
    train_means = preprocessor.transform(folds["train"][raw_cols]).mean(axis=0)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "drop_cols": drop_cols,
            "feature_names": feature_names,
            "feature_means": [float(m) for m in train_means],
        },
        ARTIFACTS_DIR / "feature_metadata.joblib",
    )

    for name, df in folds.items():
        X = preprocessor.transform(df[raw_cols])
        out = pd.DataFrame(X, columns=feature_names)
        out[target] = df[target].to_numpy()
        out.to_parquet(PROCESSED_DIR / f"{name}_features.parquet", index=False)
        logger.info("Saved %s_features.parquet with shape %s", name, X.shape)


if __name__ == "__main__":
    build_features()
