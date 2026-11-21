"""
Generate a realistically drifted batch to demo the monitoring loop.

Holding out a test split from the same distribution can never trigger the
drift alarm, so this script builds production-like scenarios from the test
fold:

* ``heatwave``  — fleet moves to a hotter climate: ambient and cell
                  temperatures rise, cooling degrades, more fast charging
* ``aging``     — the fleet ages two years: more cycles, odometer and
                  capacity loss, lower state of health
* ``new_market``— a new region: different brand / chemistry mix and an
                  unseen manufacturer

    python -m src.monitoring.simulate_drift --scenario heatwave
    python -m src.monitoring.drift_check --current data/drift/heatwave.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import DATA_DIR, PROCESSED_DIR, get_logger

logger = get_logger(__name__)

DRIFT_DIR = DATA_DIR / "drift"


def _shift(df: pd.DataFrame, col: str, add: float = 0.0, mul: float = 1.0, clip=None) -> None:
    if col in df:
        df[col] = df[col] * mul + add
        if clip is not None:
            df[col] = df[col].clip(*clip)


def heatwave(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    for col in ("average_ambient_temperature", "maximum_temperature", "minimum_temperature"):
        _shift(df, col, add=9.0)
    _shift(df, "cell_temperature_avg", add=7.0)
    _shift(df, "cell_temperature_max", add=10.0)
    _shift(df, "temperature_variance", mul=1.6)
    _shift(df, "cooling_system_health", mul=0.8)
    _shift(df, "humidity", add=-15.0, clip=(0, 100))
    _shift(df, "fast_charge_ratio", add=0.2, clip=(0, 1))
    _shift(df, "thermal_runaway_risk", mul=1.5)
    df["terrain_type"] = np.where(rng.random(len(df)) < 0.5, "Desert", df["terrain_type"])
    return df


def aging(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    _shift(df, "vehicle_age_years", add=2.0)
    _shift(df, "manufacturing_year", add=-2.0)
    _shift(df, "odometer_km", mul=1.5)
    _shift(df, "cycle_count", mul=1.6)
    _shift(df, "capacity_loss_percent", mul=1.5)
    _shift(df, "internal_resistance", mul=1.3)
    for col in ("state_of_health", "battery_health_percent"):
        _shift(df, col, add=-9.0, clip=(0, 100))
    _shift(df, "aging_score", mul=1.4)
    _shift(df, "predicted_remaining_life_cycles", mul=0.7)
    return df


def new_market(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df["vehicle_brand"] = rng.choice(["BYD", "NIO", "Xpeng", "Tesla"], size=n, p=[0.4, 0.25, 0.2, 0.15])
    df["battery_chemistry"] = rng.choice(["LFP", "NMC"], size=n, p=[0.8, 0.2])
    df["battery_manufacturer"] = rng.choice(["CATL", "BYD", "NewCellCo"], size=n, p=[0.5, 0.3, 0.2])
    df["fleet_or_private"] = rng.choice(["Fleet", "Private"], size=n, p=[0.7, 0.3])
    _shift(df, "daily_distance", mul=1.4)
    _shift(df, "city_driving_ratio", add=0.15, clip=(0, 1))
    return df


SCENARIOS = {"heatwave": heatwave, "aging": aging, "new_market": new_market}


def simulate(scenario: str, source: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(source).copy()
    df = SCENARIOS[scenario](df, rng)
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    out = DRIFT_DIR / f"{scenario}.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote drifted batch '%s' (%d rows) to %s", scenario, len(df), out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="heatwave")
    parser.add_argument("--source", type=Path, default=PROCESSED_DIR / "test.parquet")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    simulate(args.scenario, args.source, args.seed)
