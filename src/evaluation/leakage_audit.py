"""
Target-leakage audit.

A ROC-AUC near 0.98 deserves suspicion. Several columns in this dataset are
*derived scores* (thermal_runaway_risk, aging_score, ...) whose provenance is
unknown — if any were computed from the outcome, the model would look great
offline and fail in production. This stage checks two things:

1. Single-feature AUC: any lone column with AUC > 0.95 is flagged as a
   likely leak (a genuine early-warning signal is rarely that clean).
2. Ablation: retrain with every derived score removed. If performance
   collapses, the model depends on features we cannot vouch for.

Writes reports/leakage_audit.json (tracked as a DVC metric).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import PROCESSED_DIR, REPORTS_DIR, get_logger, load_params, write_json

logger = get_logger(__name__)

DERIVED_SCORES = [
    "thermal_runaway_risk",
    "thermal_health_score",
    "predicted_remaining_life_cycles",
    "battery_stress_index",
    "aging_score",
    "charging_quality_score",
    "driving_stress_score",
    "maintenance_score",
]
LEAK_AUC_THRESHOLD = 0.95


def single_feature_auc(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}
    for col in df.select_dtypes("number").columns:
        mask = df[col].notna()
        if y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], df.loc[mask, col])
        scores[col] = max(auc, 1 - auc)  # direction-agnostic
    return pd.Series(scores).sort_values(ascending=False)


def fit_score(X_tr, y_tr, X_va, y_va, seed: int) -> tuple[float, float]:
    model = LGBMClassifier(n_estimators=400, learning_rate=0.03, verbose=-1, random_state=seed)
    model.fit(X_tr, y_tr)
    p = np.asarray(model.predict_proba(X_va))[:, 1]
    return float(roc_auc_score(y_va, p)), float(average_precision_score(y_va, p))


def audit() -> None:
    params = load_params()
    target = params["target_column"]
    drop = params["features"]["drop_columns"]
    seed = params["train"]["random_state"]

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet").drop(columns=drop)
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet").drop(columns=drop)
    y_tr, y_va = train.pop(target), val.pop(target)
    for col in train.select_dtypes("object").columns:
        cats = pd.CategoricalDtype(sorted(train[col].dropna().unique()))
        train[col], val[col] = train[col].astype(cats), val[col].astype(cats)

    auc = single_feature_auc(train, y_tr)
    suspicious = auc[auc > LEAK_AUC_THRESHOLD]

    full_auc, full_ap = fit_score(train, y_tr, val, y_va, seed)
    keep = [c for c in train.columns if c not in DERIVED_SCORES]
    abl_auc, abl_ap = fit_score(train[keep], y_tr, val[keep], y_va, seed)

    report = {
        "max_single_feature_auc": float(auc.max()),
        "n_suspicious_features": len(suspicious),
        "suspicious_features": {str(k): float(v) for k, v in suspicious.items()},
        "top_single_feature_auc": {str(k): round(float(v), 4) for k, v in auc.head(10).items()},
        "ablation_full_roc_auc": full_auc,
        "ablation_full_pr_auc": full_ap,
        "ablation_no_derived_roc_auc": abl_auc,
        "ablation_no_derived_pr_auc": abl_ap,
        "ablation_pr_auc_drop": full_ap - abl_ap,
        "derived_features_removed": DERIVED_SCORES,
    }
    write_json(REPORTS_DIR / "leakage_audit.json", report)

    logger.info("Top single-feature AUC: %s = %.3f", auc.idxmax(), auc.max())
    logger.info(
        "Ablation PR-AUC: full=%.4f, without derived scores=%.4f (drop %.4f)",
        full_ap,
        abl_ap,
        full_ap - abl_ap,
    )
    if len(suspicious):
        logger.warning(
            "Possible leakage — features with AUC > %.2f: %s", LEAK_AUC_THRESHOLD, list(suspicious.index)
        )


if __name__ == "__main__":
    audit()
