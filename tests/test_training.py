import numpy as np
import pytest

from src.evaluation.evaluate import raw_feature_name
from src.evaluation.promotion_gate import compare
from src.training.metrics import classification_metrics, expected_cost, optimal_threshold
from src.training.models import SUPPORTED_MODELS, build_model


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_supported_model_trains(name, trained_artifacts):
    X, y = trained_artifacts["X"], trained_artifacts["y"]
    hp = {"n_estimators": 10} if name in ("random_forest", "lightgbm", "xgboost") else {"max_iter": 500}
    proba = build_model(name, hp, seed=0).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("deep_magic", {}, seed=0)


def test_expected_cost_counts_errors():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])  # one FN, one FP at threshold 0.5
    assert expected_cost(y, p, 0.5, cost_fn=10, cost_fp=1) == pytest.approx((10 + 1) / 4)


def test_optimal_threshold_moves_down_when_misses_are_expensive():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    p = np.clip(0.25 * y + rng.normal(0.3, 0.12, 4000), 0, 1)
    t_cheap, _ = optimal_threshold(y, p, cost_fn=1, cost_fp=1)
    t_costly, _ = optimal_threshold(y, p, cost_fn=20, cost_fp=1)
    assert t_costly < t_cheap


def test_classification_metrics_confusion_counts():
    m = classification_metrics([1, 0, 1, 0], [0.9, 0.2, 0.3, 0.7], threshold=0.5, prefix="x_")
    assert (m["x_tp"], m["x_fn"], m["x_fp"], m["x_tn"]) == (1, 1, 1, 1)
    assert m["x_recall"] == 0.5


def test_raw_feature_name_maps_one_hot_back():
    cats = ["vehicle_model", "vehicle_type"]
    assert raw_feature_name("cat__vehicle_type_SUV", cats) == "vehicle_type"
    assert raw_feature_name("cat__vehicle_model_Model_3", cats) == "vehicle_model"
    assert raw_feature_name("num__odometer_km", cats) == "odometer_km"


def test_promotion_gate_blocks_regression():
    champion = {"model": "a", "test_pr_auc": 0.80, "test_roc_auc": 0.98}
    ok, _ = compare({"model": "b", "test_pr_auc": 0.79, "test_roc_auc": 0.98}, champion, 0.02)
    assert ok
    ok, report = compare({"model": "b", "test_pr_auc": 0.70, "test_roc_auc": 0.97}, champion, 0.02)
    assert not ok
    assert "Gate failed" in report


def test_promotion_gate_passes_without_champion():
    ok, report = compare({"model": "b", "test_pr_auc": 0.5}, None, 0.02)
    assert ok
    assert "no champion" in report
