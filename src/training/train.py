"""
Training stage.

Trains the classifier on the engineered features, logs params/metrics/model
to MLflow (local file-based tracking store — swap tracking_uri for a remote
MLflow server in production), and registers the model version. Also saves
a plain joblib copy to models/model.joblib for the serving layer, so the
API doesn't need to talk to the MLflow registry to boot.
"""
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_model(cfg):
    model_type = cfg["model_type"]
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"], max_depth=min(cfg["max_depth"], 5),
            random_state=cfg["random_state"],
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            class_weight=cfg["class_weight"], max_iter=1000, random_state=cfg["random_state"]
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train():
    params = load_params()
    target = params["target_column"]
    train_cfg = params["train"]
    mlflow_cfg = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val_features.parquet")

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]

    with mlflow.start_run(run_name=f"{train_cfg['model_type']}-training") as run:
        mlflow.log_params(train_cfg)

        model = get_model(train_cfg)
        logger.info("Training %s on %d samples / %d features",
                    train_cfg["model_type"], *X_train.shape)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= params["evaluation"]["decision_threshold"]).astype(int)

        metrics = {
            "val_roc_auc": roc_auc_score(y_val, val_proba),
            "val_pr_auc": average_precision_score(y_val, val_proba),
            "val_f1": f1_score(y_val, val_pred),
            "val_precision": precision_score(y_val, val_pred, zero_division=0),
            "val_recall": recall_score(y_val, val_pred),
        }
        mlflow.log_metrics(metrics)
        logger.info("Validation metrics: %s", json.dumps(metrics, indent=2))

        mlflow.sklearn.log_model(
            model, artifact_path="model",
            registered_model_name=mlflow_cfg["registered_model_name"],
        )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "model.joblib")
        with open(MODELS_DIR / "train_metrics.json", "w") as f:
            json.dump({"run_id": run.info.run_id, **metrics}, f, indent=2)

        logger.info("Run ID: %s — model saved to models/model.joblib", run.info.run_id)


if __name__ == "__main__":
    train()
