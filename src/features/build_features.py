"""
Feature engineering stage.

Builds a single sklearn Pipeline (imputation + scaling + one-hot encoding)
fit ONLY on the training fold, then applies it to train/val/test and
persists the fitted transformer to artifacts/preprocessor.joblib so the
exact same transformation is reused by the FastAPI serving layer at
inference time. This is the piece that prevents train/serve skew.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def build_preprocessor(df: pd.DataFrame, drop_cols, target_col):
    feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    numeric_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols


def build_features():
    params = load_params()
    target = params["target_column"]
    drop_cols = params["preprocess"]["drop_columns"]

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train, drop_cols, target)
    logger.info("Fitting preprocessor on %d numeric + %d categorical columns",
                len(numeric_cols), len(categorical_cols))

    X_train = preprocessor.fit_transform(train.drop(columns=drop_cols + [target]))
    X_val = preprocessor.transform(val.drop(columns=drop_cols + [target]))
    X_test = preprocessor.transform(test.drop(columns=drop_cols + [target]))

    feature_names = preprocessor.get_feature_names_out()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(
        {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
         "drop_cols": drop_cols, "feature_names": list(feature_names)},
        ARTIFACTS_DIR / "feature_metadata.joblib",
    )

    for name, X, y in [
        ("train", X_train, train[target].values),
        ("val", X_val, val[target].values),
        ("test", X_test, test[target].values),
    ]:
        pd.DataFrame(X, columns=feature_names).assign(**{target: y}).to_parquet(
            PROCESSED_DIR / f"{name}_features.parquet", index=False
        )
        logger.info("Saved %s_features.parquet with shape %s", name, X.shape)


if __name__ == "__main__":
    build_features()
