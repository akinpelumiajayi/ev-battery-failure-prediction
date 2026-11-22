import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_check import compute_drift, psi_categorical, psi_numeric
from src.monitoring.simulate_drift import SCENARIOS

rng = np.random.default_rng(0)


def test_psi_identical_distribution_is_near_zero():
    s = pd.Series(rng.normal(size=5000))
    assert psi_numeric(s, s.sample(frac=1, random_state=1)) < 1e-6


def test_psi_same_distribution_new_sample_is_small():
    assert psi_numeric(pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(size=5000))) < 0.05


def test_psi_detects_mean_shift():
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(1.0, 1, 5000))) > 0.2


def test_psi_counts_values_outside_reference_range():
    ref = pd.Series(rng.uniform(0, 1, 5000))
    cur = pd.Series(rng.uniform(5, 6, 5000))  # entirely beyond the training max
    assert psi_numeric(ref, cur) > 1.0


def test_psi_constant_feature_is_zero():
    assert psi_numeric(pd.Series([1.0] * 100), pd.Series([1.0] * 100)) == 0.0


def test_categorical_psi_detects_new_category_mix():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi_categorical(ref, ref.sample(frac=1, random_state=0)) < 1e-6
    assert psi_categorical(ref, pd.Series(["a"] * 100 + ["c"] * 900)) > 1.0


def test_compute_drift_no_drift_on_same_distribution(raw_df):
    half = len(raw_df) // 2
    report = compute_drift(raw_df.iloc[:half], raw_df.iloc[half:], psi_threshold=0.2, share_threshold=0.15)
    assert report["drift_detected"] is False
    assert "vehicle_id" not in report["all_scores"]
    assert "battery_failure" not in report["all_scores"]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_simulated_scenarios_trigger_drift(raw_df, scenario):
    drifted = SCENARIOS[scenario](raw_df.copy(), np.random.default_rng(1))
    report = compute_drift(raw_df, drifted, psi_threshold=0.2, share_threshold=0.10, critical_psi=1.0)
    assert report["n_features_drifted"] >= 3
    assert report["drift_detected"] is True


def test_single_critical_feature_triggers_drift(raw_df):
    broken = raw_df.copy()
    broken["pack_voltage"] = broken["pack_voltage"] * 1000  # e.g. upstream switched V -> mV
    report = compute_drift(raw_df, broken, psi_threshold=0.2, share_threshold=0.5, critical_psi=1.0)
    assert report["n_features_drifted"] == 1
    assert report["drift_detected"] is True
    assert report["critical_features"] == ["pack_voltage"]
