import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.serving import app as app_module
from src.serving.schemas import VehicleFeatures

EXAMPLE = VehicleFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(trained_artifacts):
    mp = pytest.MonkeyPatch()
    mp.setattr(app_module, "ARTIFACTS_DIR", trained_artifacts["artifacts_dir"])
    mp.setattr(app_module, "MODELS_DIR", trained_artifacts["models_dir"])
    mp.delenv("MODEL_VERSION", raising=False)
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


def test_health_ok(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "abcdef12"}


def test_model_info_exposes_threshold_and_bands(client):
    body = client.get("/model").json()
    assert body["model_name"] == "logistic_regression"
    assert body["decision_threshold"] == 0.4
    assert body["risk_bands"] == {"medium_from": 0.2, "high_from": 0.4}
    assert body["test_metrics"]["test_roc_auc"] == 0.9


def test_predict_uses_tuned_threshold(client):
    body = client.post("/predict", json=EXAMPLE).json()
    p = body["failure_probability"]
    assert 0 <= p <= 1
    assert body["decision_threshold"] == 0.4
    assert body["prediction"] == int(p >= 0.4)
    expected_band = "high" if p >= 0.4 else "medium" if p >= 0.2 else "low"
    assert body["risk_band"] == expected_band


def test_predict_tolerates_missing_and_unknown_values(client):
    resp = client.post("/predict", json={"vehicle_brand": "NeverSeenBrand", "unexpected_field": 1})
    assert resp.status_code == 200


def test_predict_rejects_out_of_range_input(client):
    resp = client.post("/predict", json={"state_of_health": 150})
    assert resp.status_code == 422


def test_batch_endpoint(client):
    resp = client.post("/predict/batch", json={"vehicles": [EXAMPLE, {}, {"state_of_health": 99}]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 3
    assert body["n_flagged"] == sum(p["prediction"] for p in body["predictions"])


def test_batch_rejects_empty(client):
    assert client.post("/predict/batch", json={"vehicles": []}).status_code == 422


def test_explain_returns_ranked_raw_feature_factors(client):
    import math

    body = client.post("/explain", params={"top_k": 25}, json=EXAMPLE).json()
    assert body["top_factors"], "expected at least one factor"
    assert {"feature", "value", "contribution"} <= set(body["top_factors"][0])
    # raw feature names (one-hot columns are aggregated back)
    assert all("__" not in f["feature"] for f in body["top_factors"])
    # sorted by absolute impact
    impacts = [abs(f["contribution"]) for f in body["top_factors"]]
    assert impacts == sorted(impacts, reverse=True)
    p = body["failure_probability"]
    assert 0 <= p <= 1
    assert math.isfinite(body["base_value"])


@pytest.mark.parametrize("model_cls", [LGBMClassifier, XGBClassifier])
def test_explain_tree_models(client, trained_artifacts, model_cls):
    kwargs = {"n_estimators": 20, "verbose": -1} if model_cls is LGBMClassifier else {"n_estimators": 20}
    tree = model_cls(**kwargs).fit(trained_artifacts["X"], trained_artifacts["y"])
    original = app_module._state["model"]
    app_module._state["model"] = tree
    try:
        resp = client.post("/explain", json=EXAMPLE)
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["top_factors"]) == 8
    finally:
        app_module._state["model"] = original


def test_metrics_endpoint_exposes_prometheus_series(client):
    client.post("/predict", json=EXAMPLE)
    text = client.get("/metrics").text
    assert "predictions_total" in text
    assert "prediction_latency_seconds_bucket" in text
    assert "model_decision_threshold 0.4" in text


def test_degraded_mode_returns_503(tmp_path):
    mp = pytest.MonkeyPatch()
    mp.setattr(app_module, "ARTIFACTS_DIR", tmp_path)
    mp.setattr(app_module, "MODELS_DIR", tmp_path)
    try:
        with TestClient(app_module.app) as c:
            assert c.get("/health").json()["status"] == "degraded"
            assert c.post("/predict", json={}).status_code == 503
    finally:
        mp.undo()
