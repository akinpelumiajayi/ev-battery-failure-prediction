"""
Data validation stage.

Runs schema + sanity checks on the raw CSV before anything else in the
pipeline is allowed to run. This is the "data contract" for the project:
if the raw data drifts in structure (wrong dtypes, out-of-range values,
new/missing columns) the pipeline fails loudly here instead of silently
producing a bad model downstream.
"""
import sys
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/ev_battery_data.csv")
REPORT_PATH = Path("artifacts/validation_report.txt")

# Core schema: we only hard-enforce the columns that matter for modeling
# and leave nullable=True on columns that are known to have missing values
# (confirmed during EDA — ~5% missingness spread across most numeric cols).
schema = DataFrameSchema(
    {
        "vehicle_id": Column(str, nullable=False, unique=True),
        "battery_failure": Column(int, Check.isin([0, 1]), nullable=False),
        "manufacturing_year": Column(float, Check.in_range(1990, 2026), nullable=True),
        "vehicle_age_years": Column(float, Check.ge(0), nullable=True),
        "battery_health_percent": Column(float, Check.in_range(0, 100), nullable=True),
        "state_of_charge": Column(float, Check.in_range(0, 100), nullable=True),
        "state_of_health": Column(float, Check.in_range(0, 100), nullable=True),
        "cell_voltage_avg": Column(float, Check.gt(0), nullable=True),
        "pack_voltage": Column(float, Check.gt(0), nullable=True),
    },
    strict=False,   # allow the other ~60 feature columns through untouched
    coerce=False,
)


def validate(path: Path = RAW_PATH) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)

    logger.info("Validating schema on %d rows / %d columns", *df.shape)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(err.failure_cases.to_string())
        logger.error("Schema validation FAILED — see %s", REPORT_PATH)
        raise

    # Business-rule checks that are cheap but catch real data-quality issues.
    n_dupe_ids = df["vehicle_id"].duplicated().sum()
    assert n_dupe_ids == 0, f"Found {n_dupe_ids} duplicate vehicle_id rows"

    missing_target = df["battery_failure"].isna().sum()
    assert missing_target == 0, "Target column must never be null"

    class_balance = df["battery_failure"].mean()
    logger.info("Positive class rate: %.2f%%", class_balance * 100)
    assert 0.01 < class_balance < 0.5, "Target class balance looks implausible — check upstream data"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        f"Validation PASSED\nrows={len(df)}\ncols={df.shape[1]}\n"
        f"positive_rate={class_balance:.4f}\nduplicate_ids={n_dupe_ids}\n"
    )
    logger.info("Validation passed. Report written to %s", REPORT_PATH)
    return df


if __name__ == "__main__":
    try:
        validate()
    except Exception as e:
        logger.error("Validation stage failed: %s", e)
        sys.exit(1)
