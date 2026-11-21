import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.data import validate as validate_module
from src.data.validate import schema


def _base_row(**overrides):
    row = {
        "vehicle_id": "EV999999",
        "battery_failure": 0,
        "manufacturing_year": 2020.0,
        "vehicle_age_years": 5.0,
        "battery_health_percent": 80.0,
        "state_of_charge": 60.0,
        "state_of_health": 80.0,
        "depth_of_discharge": 40.0,
        "cell_voltage_avg": 3.7,
        "pack_voltage": 370.0,
        "fast_charge_ratio": 0.3,
        "odometer_km": 50_000.0,
        **{c: "x" for c in validate_module.CATEGORICAL_COLUMNS},
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    schema.validate(pd.DataFrame([_base_row()]), lazy=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"battery_failure": 2},  # target must be binary
        {"state_of_health": 150.0},  # percentage out of range
        {"pack_voltage": -10.0},  # physically impossible
        {"cell_voltage_avg": 9.0},  # Li-ion cell can't sit at 9 V
        {"fast_charge_ratio": 1.5},  # ratio > 1
    ],
)
def test_invalid_values_fail(overrides):
    with pytest.raises(SchemaErrors):
        schema.validate(pd.DataFrame([_base_row(**overrides)]), lazy=True)


def test_duplicate_vehicle_ids_fail():
    df = pd.DataFrame([_base_row(), _base_row()])
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)


def test_missing_required_column_fails():
    df = pd.DataFrame([_base_row()]).drop(columns=["state_of_health"])
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)


def test_validate_rejects_implausible_class_balance(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_module, "REPORT_PATH", tmp_path / "report.txt")
    monkeypatch.setattr(validate_module, "MIN_ROWS", 10)
    rows = [_base_row(vehicle_id=f"EV{i}", battery_failure=1) for i in range(20)]  # 100% positive
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="positive rate"):
        validate_module.validate(path)
    assert "FAILED" in (tmp_path / "report.txt").read_text()


def test_validate_passes_and_writes_report(tmp_path, monkeypatch, raw_df):
    monkeypatch.setattr(validate_module, "MIN_ROWS", 100)
    monkeypatch.setattr(validate_module, "REPORT_PATH", tmp_path / "report.txt")
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path, index=False)
    validate_module.validate(path)
    assert (tmp_path / "report.txt").read_text().startswith("Validation PASSED")
