"""
Drift monitoring stage.

Compares a new batch of incoming data against the reference distribution
(the training set, snapshotted in data/reference/reference.parquet) using
Population Stability Index (PSI) for numeric columns. This is intentionally
dependency-light (no Evidently/whylogs required) so it runs anywhere the
training image runs — swap in a heavier drift library if you need richer
reports.

Usage:
    python src/monitoring/drift_check.py --current data/processed/test.parquet

In production this would run on a schedule (e.g. a Kubernetes CronJob) or
be triggered by the drift-check CI job, reading recent inference logs
instead of a static file. A PSI above the configured threshold should
trigger an alert and/or kick off the retraining pipeline (see README's
"closing the loop" section).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_PATH = Path("data/reference/reference.parquet")
REPORT_PATH = Path("artifacts/drift_report.json")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Standard PSI: bins the reference distribution into deciles, then
    compares the current distribution against those same bin edges."""
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0  # not enough spread to bin meaningfully

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(ref_counts.sum(), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(cur_counts.sum(), 1))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_drift_check(current_path: Path, threshold: float = 0.2) -> dict:
    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_parquet(current_path)

    numeric_cols = reference.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [c for c in numeric_cols if c in current.columns]

    psi_scores = {
        col: population_stability_index(reference[col], current[col]) for col in numeric_cols
    }
    drifted = {col: score for col, score in psi_scores.items() if score >= threshold}

    report = {
        "threshold": threshold,
        "n_features_checked": len(psi_scores),
        "n_features_drifted": len(drifted),
        "drifted_features": drifted,
        "all_scores": psi_scores,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    if drifted:
        logger.warning(
            "DRIFT DETECTED in %d/%d features (PSI >= %.2f): %s",
            len(drifted), len(psi_scores), threshold, list(drifted.keys()),
        )
    else:
        logger.info("No significant drift detected across %d features (threshold=%.2f)",
                    len(psi_scores), threshold)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True,
                         help="Parquet/CSV file with the recent batch to check for drift")
    args = parser.parse_args()

    params = load_params()
    threshold = params["monitoring"]["drift_psi_threshold"]
    run_drift_check(args.current, threshold)
