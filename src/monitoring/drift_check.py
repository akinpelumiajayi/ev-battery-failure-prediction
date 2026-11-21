"""
Data-drift detection with the Population Stability Index (PSI).

Compares an incoming batch against the training reference snapshot
(`data/reference/reference.parquet`):

* numeric features  — PSI over the reference deciles
* categorical       — PSI over category frequencies (unseen categories
                      get their own bucket)

A feature has drifted when PSI >= `monitoring.drift_psi_threshold`
(0.1 = moderate, 0.2 = significant, by industry convention). The batch as a
whole is flagged — the retraining trigger — when either

* the share of drifted features exceeds `monitoring.drift_share_threshold`
  (broad, gradual population shift), or
* any single feature exceeds `monitoring.drift_critical_psi` (one feature
  changed beyond recognition: a broken upstream pipeline, a unit change, or
  a new market — never something to wait out).

Dependency-light on purpose (numpy/pandas only) so it runs anywhere the
training image runs.

    python -m src.monitoring.drift_check --current data/processed/test.parquet
    python -m src.monitoring.drift_check --current data/drift/simulated.parquet --fail-on-drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ARTIFACTS_DIR, REFERENCE_DIR, get_logger, load_params, write_json

logger = get_logger(__name__)

REFERENCE_PATH = REFERENCE_DIR / "reference.parquet"
DEFAULT_REPORT = ARTIFACTS_DIR / "drift_report.json"
EPS = 1e-4


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # (near-)constant feature: not enough spread to bin
    # Open-ended outer bins so values outside the training range still count.
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    return _psi(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.fillna("__missing__").astype(str)
    current = current.fillna("__missing__").astype(str)
    if reference.empty or current.empty:
        return 0.0
    cats = sorted(set(reference.unique()) | set(current.unique()))
    ref_pct = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_pct, cur_pct)


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float,
    share_threshold: float,
    critical_psi: float = float("inf"),
    ignore: tuple[str, ...] = ("vehicle_id", "battery_serial", "battery_failure"),
) -> dict:
    cols = [c for c in reference.columns if c in current.columns and c not in ignore]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(reference[c])]
    categorical = [c for c in cols if c not in numeric]

    scores = {c: psi_numeric(reference[c], current[c]) for c in numeric}
    scores |= {c: psi_categorical(reference[c], current[c]) for c in categorical}
    scores = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
    drifted = {c: round(s, 4) for c, s in scores.items() if s >= psi_threshold}
    max_psi = max(scores.values(), default=0.0)
    share = len(drifted) / max(len(scores), 1)

    return {
        "psi_threshold": psi_threshold,
        "share_threshold": share_threshold,
        "critical_psi": critical_psi,
        "n_rows_current": len(current),
        "n_features_checked": len(scores),
        "n_features_drifted": len(drifted),
        "share_drifted": round(share, 4),
        "drift_detected": share > share_threshold or max_psi >= critical_psi,
        "critical_features": [c for c, s in scores.items() if s >= critical_psi],
        "max_psi": round(max_psi, 4),
        "drifted_features": drifted,
        "all_scores": {c: round(s, 5) for c, s in scores.items()},
    }


def run_drift_check(current_path: Path, report_path: Path = DEFAULT_REPORT) -> dict:
    params = load_params()["monitoring"]
    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_csv(current_path) if current_path.suffix == ".csv" else pd.read_parquet(current_path)

    report = compute_drift(
        reference,
        current,
        params["drift_psi_threshold"],
        params["drift_share_threshold"],
        params["drift_critical_psi"],
    )
    report["current_path"] = current_path.as_posix()
    write_json(report_path, report)

    if report["drift_detected"]:
        logger.warning(
            "DRIFT DETECTED: %d/%d features (%.0f%%) over PSI %.2f — top: %s",
            report["n_features_drifted"],
            report["n_features_checked"],
            100 * report["share_drifted"],
            report["psi_threshold"],
            list(report["drifted_features"])[:5],
        )
    else:
        logger.info(
            "No significant drift: %d/%d features over PSI %.2f (max PSI %.3f)",
            report["n_features_drifted"],
            report["n_features_checked"],
            report["psi_threshold"],
            report["max_psi"],
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSI drift check against the training reference")
    parser.add_argument("--current", type=Path, required=True, help="parquet/csv batch to check")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fail-on-drift", action="store_true", help="exit 2 when drift is detected")
    args = parser.parse_args()

    result = run_drift_check(args.current, args.report)
    if args.fail_on_drift and result["drift_detected"]:
        sys.exit(2)
