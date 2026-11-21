"""
Training stage: model comparison -> Bayesian tuning -> cost-based threshold.

1. Every candidate in `params.yaml -> train.candidates` is scored with
   stratified k-fold CV on the train fold (PR-AUC), then refit and checked on
   the validation fold. CV, not the single ~200-positive validation fold,
   drives selection, so a lucky split can't pick the winner. Each candidate
   is a nested MLflow run, so the comparison is one click away in the UI.
2. Optuna (TPE) tunes the configured model family on the *same* CV folds;
   the tuned model competes with the untuned candidates on equal terms.
3. The decision threshold of the winner is chosen on the validation fold to
   minimise expected business cost (missed failure >> extra inspection).
4. The winner is logged + registered in the MLflow Model Registry with the
   `challenger` alias, and exported to `models/` for the serving image.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import mlflow
from src.common import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT, get_logger, load_params, write_json
from src.training.metrics import classification_metrics, optimal_threshold
from src.training.models import build_model

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — git resolved from PATH
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(out).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def suggest_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SEARCH_SPACES = {"lightgbm": suggest_lightgbm, "xgboost": suggest_xgboost}


def cv_pr_auc(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> float:
    return float(cross_val_score(model, X, y, cv=cv, scoring="average_precision").mean())


def tune(model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int, cv: StratifiedKFold, seed: int):
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for '{model_name}'")

    def objective(trial: optuna.Trial) -> float:
        model = build_model(model_name, SEARCH_SPACES[model_name](trial), seed)
        return cv_pr_auc(model, X, y, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    if model_name == "lightgbm":
        best["subsample_freq"] = 1
    return best, float(study.best_value), study


def train() -> None:
    params = load_params()
    target = params["target_column"]
    tcfg = params["train"]
    ecfg = params["evaluation"]
    mcfg = params["mlflow"]
    seed = tcfg["random_state"]
    selection_metric = tcfg["selection_metric"]

    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])

    train_df = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val_features.parquet")
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]

    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=seed)
    results: dict[str, dict[str, Any]] = {}

    with mlflow.start_run(run_name="model-selection") as parent:
        mlflow.set_tags({"git_sha": _git_sha(), "stage": "train"})

        # ---- 1. candidate comparison ------------------------------------
        for name, hp in tcfg["candidates"].items():
            with mlflow.start_run(run_name=name, nested=True):
                cv_score = cv_pr_auc(build_model(name, hp, seed), X_train, y_train, cv)
                model = build_model(name, hp, seed).fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                metrics = {"cv_pr_auc": cv_score, **classification_metrics(y_val, proba, 0.5, prefix="val_")}
                mlflow.log_params({"model": name, **hp})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                results[name] = {"model": model, "params": hp, "proba": proba, "metrics": metrics}
                logger.info(
                    "%-22s CV PR-AUC=%.4f | val PR-AUC=%.4f ROC-AUC=%.4f",
                    name,
                    cv_score,
                    metrics["val_pr_auc"],
                    metrics["val_roc_auc"],
                )

        # ---- 2. Optuna tuning ---------------------------------------------
        tuning = tcfg["tuning"]
        if tuning["enabled"]:
            family = tuning["model"]
            with mlflow.start_run(run_name=f"{family}-tuned", nested=True):
                logger.info(
                    "Tuning %s with Optuna (%d trials, %d-fold CV)...",
                    family,
                    tuning["n_trials"],
                    tcfg["cv_folds"],
                )
                best_hp, cv_score, study = tune(family, X_train, y_train, tuning["n_trials"], cv, seed)
                model = build_model(family, best_hp, seed).fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                metrics = {"cv_pr_auc": cv_score, **classification_metrics(y_val, proba, 0.5, prefix="val_")}
                mlflow.log_params({"model": f"{family}-tuned", **best_hp})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                for t in study.trials:
                    mlflow.log_metric("trial_cv_pr_auc", t.value or 0.0, step=t.number)
                results[f"{family}_tuned"] = {
                    "model": model,
                    "params": best_hp,
                    "proba": proba,
                    "metrics": metrics,
                }
                logger.info(
                    "%-22s CV PR-AUC=%.4f | val PR-AUC=%.4f ROC-AUC=%.4f",
                    f"{family}_tuned",
                    cv_score,
                    metrics["val_pr_auc"],
                    metrics["val_roc_auc"],
                )

        # ---- 3. select winner + cost-optimal threshold ---------------------
        winner = max(results, key=lambda k: results[k]["metrics"][selection_metric])
        best = results[winner]
        threshold, val_cost = optimal_threshold(
            y_val, best["proba"], ecfg["cost_false_negative"], ecfg["cost_false_positive"]
        )
        val_metrics = classification_metrics(y_val, best["proba"], threshold, prefix="val_")
        val_metrics["val_expected_cost"] = val_cost
        val_metrics["cv_pr_auc"] = best["metrics"]["cv_pr_auc"]
        logger.info(
            "Winner: %s (%s=%.4f), cost-optimal threshold=%.3f",
            winner,
            selection_metric,
            best["metrics"][selection_metric],
            threshold,
        )

        mlflow.log_params({"winner": winner, "decision_threshold": threshold})
        mlflow.log_metrics({k: v for k, v in val_metrics.items() if isinstance(v, float)})

        # ---- 4. log, register, export -------------------------------------
        signature = infer_signature(X_val.head(50), best["model"].predict_proba(X_val.head(50)))
        info = mlflow.sklearn.log_model(
            best["model"],
            artifact_path="model",
            signature=signature,
            registered_model_name=mcfg["registered_model_name"],
        )
        version = getattr(info, "registered_model_version", None)
        if version is not None:
            MlflowClient().set_registered_model_alias(mcfg["registered_model_name"], "challenger", version)

        comparison = {
            name: {
                "cv_pr_auc": r["metrics"]["cv_pr_auc"],
                "val_pr_auc": r["metrics"]["val_pr_auc"],
                "val_roc_auc": r["metrics"]["val_roc_auc"],
                "val_brier": r["metrics"]["val_brier"],
            }
            for name, r in sorted(results.items(), key=lambda kv: -kv[1]["metrics"][selection_metric])
        }
        mlflow.log_dict(comparison, "model_comparison.json")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["model"], MODELS_DIR / "model.joblib")
        write_json(MODELS_DIR / "model_comparison.json", comparison)
        write_json(MODELS_DIR / "train_metrics.json", {"model": winner, **val_metrics})
        write_json(
            MODELS_DIR / "model_metadata.json",
            {
                "model_name": winner,
                "model_class": type(best["model"]).__name__,
                "hyperparameters": json.loads(json.dumps(best["params"], default=str)),
                "decision_threshold": threshold,
                "risk_band_medium_ratio": ecfg["risk_band_medium_ratio"],
                "cost_false_negative": ecfg["cost_false_negative"],
                "cost_false_positive": ecfg["cost_false_positive"],
                "mlflow_run_id": parent.info.run_id,
                "registry_version": str(version) if version is not None else None,
                "git_sha": _git_sha(),
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "python": platform.python_version(),
                "n_train": len(X_train),
                "n_features": X_train.shape[1],
                "val_positive_rate": float(np.mean(y_val)),
            },
        )
        logger.info("Saved models/model.joblib (run %s, registry v%s)", parent.info.run_id, version)

    # Sanity: the exported model must reproduce the score we selected on.
    reloaded = joblib.load(MODELS_DIR / "model.joblib")
    check = average_precision_score(y_val, reloaded.predict_proba(X_val)[:, 1])
    if abs(check - best["metrics"]["val_pr_auc"]) > 1e-9:
        raise RuntimeError("Exported model does not reproduce the selected validation score")


if __name__ == "__main__":
    train()
