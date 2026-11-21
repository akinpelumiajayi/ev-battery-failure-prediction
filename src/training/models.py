"""Model factory: one place that maps a config name to an estimator."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "lightgbm", "xgboost")


def build_model(name: str, hyperparams: dict[str, Any], seed: int) -> ClassifierMixin:
    hp = dict(hyperparams)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **hp)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **hp)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **hp)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="aucpr", tree_method="hist", **hp)
    raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")
