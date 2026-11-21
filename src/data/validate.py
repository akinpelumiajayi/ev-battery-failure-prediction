"""
Data validation stage — the pipeline's data contract.

Schema and business-rule checks run on the raw CSV before anything else.
If upstream data changes shape (wrong dtypes, out-of-range sensor readings,
missing columns, duplicate IDs, implausible class balance) the pipeline fails
here instead of quietly producing a bad model downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.common import ARTIFACTS_DIR, PROJECT_ROOT, get_logger, load_params

logger = get_logger(__name__)

REPORT_PATH = ARTIFACTS_DIR / "validation_report.txt"

CATEGORICAL_COLUMNS = [
    "vehicle_brand",
    "vehicle_model",
    "vehicle_type",
    "battery_manufacturer",
    "battery_chemistry",
    "drive_type",
    "fleet_or_private",
    "terrain_type",
]

# Columns that matter most are range-checked; the rest are only required to
# exist with the right dtype. `nullable=True` reflects the ~4% missingness in
# the sensor columns, which the training-time imputer handles.
schema = DataFrameSchema(
    {
        "vehicle_id": Column(str, nullable=False, unique=True),
        "battery_failure": Column(int, Check.isin([0, 1]), nullable=False),
        "manufacturing_year": Column(float, Check.in_range(1990, 2030), nullable=True, coerce=True),
        "vehicle_age_years": Column(float, Check.ge(0), nullable=True),
        "battery_health_percent": Column(float, Check.in_range(0, 100), nullable=True),
        "state_of_charge": Column(float, Check.in_range(0, 100), nullable=True),
        "state_of_health": Column(float, Check.in_range(0, 100), nullable=True),
        "depth_of_discharge": Column(float, Check.in_range(0, 100), nullable=True),
        "cell_voltage_avg": Column(float, Check.in_range(0, 5), nullable=True),
        "pack_voltage": Column(float, Check.gt(0), nullable=True),
        "fast_charge_ratio": Column(float, Check.in_range(0, 1), nullable=True),
        "odometer_km": Column(float, Check.ge(0), nullable=True),
        **{c: Column(str, nullable=True) for c in CATEGORICAL_COLUMNS},
    },
    strict=False,  # the remaining ~50 numeric features pass through untouched
)

MIN_ROWS = 1_000
POSITIVE_RATE_RANGE = (0.01, 0.5)


def validate(path: Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Validating schema on %d rows x %d columns", *df.shape)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        REPORT_PATH.write_text(err.failure_cases.to_string(), encoding="utf-8")
        logger.error("Schema validation FAILED — see %s", REPORT_PATH)
        raise

    problems = []
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (< {MIN_ROWS})")
    positive_rate = float(df["battery_failure"].mean())
    lo, hi = POSITIVE_RATE_RANGE
    if not lo < positive_rate < hi:
        problems.append(f"positive rate {positive_rate:.3f} outside ({lo}, {hi})")
    fully_null = df.columns[df.isna().all()].tolist()
    if fully_null:
        problems.append(f"columns entirely null: {fully_null}")
    if problems:
        REPORT_PATH.write_text("Validation FAILED\n" + "\n".join(problems), encoding="utf-8")
        raise ValueError("; ".join(problems))

    REPORT_PATH.write_text(
        "Validation PASSED\n"
        f"rows={len(df)}\ncols={df.shape[1]}\n"
        f"positive_rate={positive_rate:.4f}\n"
        f"missing_cell_rate={df.isna().mean().mean():.4f}\n",
        encoding="utf-8",
    )
    logger.info("Validation passed (positive rate %.2f%%)", positive_rate * 100)
    return df


if __name__ == "__main__":
    params = load_params()
    try:
        validate(PROJECT_ROOT / params["data"]["raw_path"])
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard, exit non-zero
        logger.error("Validation stage failed: %s", exc)
        sys.exit(1)
