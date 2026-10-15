import pandas as pd
import pytest

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
        "cell_voltage_avg": 3.7,
        "pack_voltage": 370.0,
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    df = pd.DataFrame([_base_row()])
    schema.validate(df, lazy=True)  # should not raise


def test_invalid_target_value_fails():
    df = pd.DataFrame([_base_row(battery_failure=2)])  # only 0/1 allowed
    with pytest.raises(Exception):
        schema.validate(df, lazy=True)


def test_out_of_range_soh_fails():
    df = pd.DataFrame([_base_row(state_of_health=150.0)])  # must be 0-100
    with pytest.raises(Exception):
        schema.validate(df, lazy=True)


def test_negative_pack_voltage_fails():
    df = pd.DataFrame([_base_row(pack_voltage=-10.0)])
    with pytest.raises(Exception):
        schema.validate(df, lazy=True)
