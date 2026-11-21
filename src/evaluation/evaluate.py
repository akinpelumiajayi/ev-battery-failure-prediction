"""
Evaluation stage — scores the untouched TEST fold and enforces quality gates.

Outputs
  models/test_metrics.json     flat metrics tracked by `dvc metrics`
  reports/figures/*.png        ROC/PR, calibration, cost curve, confusion
                               matrix and SHAP feature importance
  reports/feature_importance.json

Exits non-zero if ROC-AUC or PR-AUC fall below the gates in params.yaml, so
CI cannot build or deploy a regressed model. On success the registry version
is promoted from the `challenger` alias to `champion`.
"""

from __future__ import annotations

import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.common import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    get_logger,
    load_params,
    read_json,
    write_json,
)
from src.training.metrics import classification_metrics, cost_curve, expected_cost

logger = get_logger(__name__)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def raw_feature_name(transformed: str, categorical_cols: list[str]) -> str:
    """Map 'num__odometer_km' / 'cat__vehicle_brand_Tesla' back to the raw column."""
    kind, _, rest = transformed.partition("__")
    if kind == "cat":
        for col in sorted(categorical_cols, key=len, reverse=True):
            if rest.startswith(col + "_") or rest == col:
                return col
    return rest


def shap_importance(model, X: pd.DataFrame, categorical_cols: list[str]) -> pd.Series:
    """Mean |SHAP| per raw feature (one-hot columns summed back together)."""
    import shap

    sample = X.sample(min(len(X), 1500), random_state=0)
    if hasattr(model, "coef_"):  # linear model: exact, fast linear SHAP
        values = shap.LinearExplainer(model, shap.sample(X, 500, random_state=0)).shap_values(sample)
    else:
        try:
            values = shap.TreeExplainer(model).shap_values(sample)
        except Exception:  # noqa: BLE001 — unknown model type: model-agnostic fallback
            background = shap.sample(X, 100, random_state=0)
            sample = sample.iloc[:300]
            explainer = shap.Explainer(lambda d: model.predict_proba(d)[:, 1], background)
            values = explainer(sample, silent=True).values
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    per_col = pd.Series(np.abs(values).mean(axis=0), index=sample.columns)
    grouped = per_col.groupby(lambda c: raw_feature_name(c, categorical_cols)).sum()
    return grouped.sort_values(ascending=False)


def plot_all(y, proba, threshold, cfg, importance: pd.Series, metrics: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=axes[0], name="model")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set_title(f"ROC (AUC {metrics['test_roc_auc']:.3f})")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[1], name="model")
    axes[1].axhline(np.mean(y), ls="--", color="grey", lw=1, label="no-skill")
    axes[1].set_title(f"Precision-Recall (AP {metrics['test_pr_auc']:.3f})")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png")
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=f"model (Brier {metrics['test_brier']:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure rate", title="Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png")
    plt.close(fig)

    thresholds, costs = cost_curve(y, proba, cfg["cost_false_negative"], cfg["cost_false_positive"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thresholds, costs, lw=2)
    ax.axvline(threshold, color="tab:red", ls="--", label=f"chosen threshold {threshold:.2f}")
    ax.axvline(0.5, color="grey", ls=":", label="default 0.5")
    ax.set(
        xlabel="Decision threshold",
        ylabel="Expected cost per vehicle",
        title=f"Cost curve (FN={cfg['cost_false_negative']:g}, FP={cfg['cost_false_positive']:g})",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y,
        (proba >= threshold).astype(int),
        display_labels=["healthy", "failure"],
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close(fig)

    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index.astype(str), top.to_numpy(dtype=float), color="tab:blue")
    ax.set(xlabel="mean |SHAP value|", title="Top 15 drivers of failure risk")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance.png")
    plt.close(fig)


def promote_to_champion(params: dict, version: str | None) -> None:
    if not version:
        return
    try:
        mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
        MlflowClient().set_registered_model_alias(
            params["mlflow"]["registered_model_name"], "champion", version
        )
        logger.info("Registry: version %s promoted to @champion", version)
    except Exception as exc:  # noqa: BLE001 — registry is optional for the gate
        logger.warning("Could not set champion alias: %s", exc)


def evaluate() -> None:
    params = load_params()
    target = params["target_column"]
    cfg = params["evaluation"]

    model = joblib.load(MODELS_DIR / "model.joblib")
    meta = read_json(MODELS_DIR / "model_metadata.json")
    feature_meta = joblib.load(ARTIFACTS_DIR / "feature_metadata.joblib")
    threshold = float(meta["decision_threshold"])

    test_df = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")
    X_test, y_test = test_df.drop(columns=[target]), test_df[target].to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, proba, threshold, prefix="test_")
    c_fn, c_fp = cfg["cost_false_negative"], cfg["cost_false_positive"]
    metrics["test_threshold"] = threshold
    metrics["test_cost_per_vehicle"] = expected_cost(y_test, proba, threshold, c_fn, c_fp)
    metrics["test_cost_per_vehicle_at_0_5"] = expected_cost(y_test, proba, 0.5, c_fn, c_fp)
    metrics["test_cost_inspect_none"] = float(c_fn * np.mean(y_test))
    metrics["test_cost_saving_vs_0_5_pct"] = float(
        100 * (1 - metrics["test_cost_per_vehicle"] / metrics["test_cost_per_vehicle_at_0_5"])
    )

    importance = shap_importance(model, X_test, feature_meta["categorical_cols"])
    write_json(
        REPORTS_DIR / "feature_importance.json", {str(k): float(v) for k, v in importance.head(25).items()}
    )
    plot_all(y_test, proba, threshold, cfg, importance, metrics)

    write_json(MODELS_DIR / "test_metrics.json", {"model": meta["model_name"], **metrics})
    logger.info(
        "Test metrics @%.3f: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f recall=%.4f precision=%.4f",
        threshold,
        metrics["test_roc_auc"],
        metrics["test_pr_auc"],
        metrics["test_f1"],
        metrics["test_recall"],
        metrics["test_precision"],
    )
    logger.info(
        "Expected cost/vehicle: %.4f (vs %.4f at 0.5 -> %.1f%% saving)",
        metrics["test_cost_per_vehicle"],
        metrics["test_cost_per_vehicle_at_0_5"],
        metrics["test_cost_saving_vs_0_5_pct"],
    )

    failures = []
    if metrics["test_roc_auc"] < cfg["min_roc_auc"]:
        failures.append(f"ROC-AUC {metrics['test_roc_auc']:.4f} < {cfg['min_roc_auc']}")
    if metrics["test_pr_auc"] < cfg["min_pr_auc"]:
        failures.append(f"PR-AUC {metrics['test_pr_auc']:.4f} < {cfg['min_pr_auc']}")
    if failures:
        logger.error("QUALITY GATE FAILED — blocking promotion: %s", "; ".join(failures))
        sys.exit(1)

    logger.info(
        "Quality gates passed (ROC-AUC >= %.2f, PR-AUC >= %.2f)", cfg["min_roc_auc"], cfg["min_pr_auc"]
    )
    promote_to_champion(params, meta.get("registry_version"))


if __name__ == "__main__":
    evaluate()
