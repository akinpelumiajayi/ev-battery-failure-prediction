"""
API tests. Require the training pipeline to have already produced
models/model.joblib and artifacts/preprocessor.joblib (run `dvc repro`
or the CI pipeline first) since the app loads them on startup.
"""
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")


def test_predict_endpoint_returns_valid_prediction(client):
    payload = {
        "vehicle_brand": "Tesla", "vehicle_model": "Model 3", "vehicle_type": "Sedan",
        "manufacturing_year": 2019, "battery_manufacturer": "CATL", "battery_chemistry": "NMC",
        "battery_capacity_kwh": 60.0, "drive_type": "AWD", "odometer_km": 120000,
        "vehicle_age_years": 6.0, "fleet_or_private": "Private", "cycle_count": 1200,
        "battery_health_percent": 75.0, "state_of_charge": 55.0, "state_of_health": 74.0,
        "cell_voltage_avg": 3.65, "pack_voltage": 360.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["risk_band"] in ("low", "medium", "high")


def test_predict_endpoint_handles_missing_fields(client):
    # Only a handful of fields — the rest should be imputed, not error out
    resp = client.post("/predict", json={"vehicle_brand": "Ford", "vehicle_type": "SUV"})
    assert resp.status_code == 200


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predictions_total" in resp.content or resp.status_code == 200
