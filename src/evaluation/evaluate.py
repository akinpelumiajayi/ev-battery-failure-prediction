"""
Evaluation stage.

Scores the trained model on the untouched TEST fold and writes a
metrics.json report. Also enforces a minimum-quality gate: if ROC-AUC
falls below a threshold, this script exits non-zero, which the CI
pipeline (.github/workflows/ci.yml) uses to block promotion of a
regressed model.
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MIN_ROC_AUC = 0.75  # quality gate — tune to your risk tolerance


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def evaluate():
    params = load_params()
    target = params["target_column"]
    threshold = params["evaluation"]["decision_threshold"]

    model = joblib.load(MODELS_DIR / "model.joblib")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),
        "test_f1": f1_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred),
    }
    cm = confusion_matrix(y_test, pred).tolist()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "test_metrics.json", "w") as f:
        json.dump({**metrics, "confusion_matrix": cm}, f, indent=2)

    logger.info("Test metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Confusion matrix [[tn, fp], [fn, tp]]: %s", cm)

    if metrics["test_roc_auc"] < MIN_ROC_AUC:
        logger.error(
            "QUALITY GATE FAILED: test ROC-AUC %.4f < required %.4f — blocking promotion",
            metrics["test_roc_auc"], MIN_ROC_AUC,
        )
        sys.exit(1)

    logger.info("Quality gate passed (ROC-AUC %.4f >= %.4f)", metrics["test_roc_auc"], MIN_ROC_AUC)


if __name__ == "__main__":
    evaluate()
