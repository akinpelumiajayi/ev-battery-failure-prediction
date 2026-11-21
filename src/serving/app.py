"""
FastAPI inference service.

Loads the *same* fitted preprocessor and model produced by the DVC pipeline
(no feature logic is re-implemented here) plus `model_metadata.json`, which
carries the cost-optimised decision threshold — so serving behaviour is
defined by training, not by constants in this file.

Endpoints
  GET  /health          liveness / readiness probe
  GET  /model           model card: version, threshold, test metrics
  POST /predict         single-vehicle prediction
  POST /predict/batch   up to 1000 vehicles in one call
  POST /explain         prediction + top SHAP risk factors
  GET  /metrics         Prometheus scrape endpoint

    uvicorn src.serving.app:app --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.serving.schemas import (
    BatchRequest,
    BatchResponse,
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskBand,
    VehicleFeatures,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ev-battery-api")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

# ---- Prometheus metrics ---------------------------------------------------
# One uvicorn worker per container: the prometheus_client registry is
# per-process, so multiple workers would each report partial counters.
# Kubernetes (HPA) scales pods instead.
PREDICTIONS = Counter("predictions_total", "Predictions served", ["risk_band", "endpoint"])
PREDICTION_ERRORS = Counter("prediction_errors_total", "Failed prediction requests", ["endpoint"])
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model inference latency",
    ["endpoint"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PROBA = Histogram(
    "predicted_failure_probability",
    "Distribution of predicted failure probabilities",
    buckets=[i / 10 for i in range(11)],
)
MISSING_FIELDS = Histogram(
    "request_missing_feature_ratio",
    "Share of input features missing per request",
    buckets=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)
MODEL_INFO = Gauge("model_info", "Currently loaded model", ["model_name", "model_version"])
THRESHOLD = Gauge("model_decision_threshold", "Decision threshold in use")

_state: dict[str, Any] = {"model": None}


def load_artifacts() -> None:
    try:
        model = joblib.load(MODELS_DIR / "model.joblib")
        preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
        feature_meta = joblib.load(ARTIFACTS_DIR / "feature_metadata.joblib")
    except FileNotFoundError as exc:
        logger.error("Model artifacts missing (%s) — run `dvc repro` first. Serving in degraded mode.", exc)
        _state["model"] = None
        return

    meta_path = MODELS_DIR / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics_path = MODELS_DIR / "test_metrics.json"
    test_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    threshold = float(meta.get("decision_threshold", 0.5))
    # MODEL_VERSION is stamped into the image by CI (git SHA); fall back to
    # the training commit recorded in the model metadata.
    env_version = os.environ.get("MODEL_VERSION", "")
    trained_version = meta.get("git_sha") or meta.get("mlflow_run_id") or "unknown"
    version = (env_version if env_version not in ("", "unknown") else trained_version)[:8]

    _state.update(
        model=model,
        preprocessor=preprocessor,
        feature_meta=feature_meta,
        raw_cols=feature_meta["numeric_cols"] + feature_meta["categorical_cols"],
        meta=meta,
        test_metrics={k: v for k, v in test_metrics.items() if isinstance(v, float)},
        threshold=threshold,
        medium_cutoff=threshold * float(meta.get("risk_band_medium_ratio", 0.5)),
        version=version,
    )
    MODEL_INFO.labels(model_name=meta.get("model_name", "unknown"), model_version=version).set(1)
    THRESHOLD.set(threshold)
    logger.info("Loaded %s (version %s, threshold %.3f)", meta.get("model_name"), version, threshold)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="EV Battery Failure Prediction API",
    description=(
        "Predicts the probability that an EV battery pack fails, from BMS telemetry, "
        "charging behaviour and driving patterns. Decisions use a cost-optimised "
        "threshold (a missed failure is weighted 10x an unnecessary inspection)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _require_model() -> None:
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run the training pipeline.")


def _risk_band(p: float) -> RiskBand:
    if p >= _state["threshold"]:
        return "high"
    if p >= _state["medium_cutoff"]:
        return "medium"
    return "low"


def _to_frame(vehicles: list[VehicleFeatures]) -> pd.DataFrame:
    raw = pd.DataFrame([v.model_dump() for v in vehicles])
    raw = raw.reindex(columns=_state["raw_cols"])  # adds missing cols as NaN, drops extras
    for col in _state["feature_meta"]["categorical_cols"]:
        raw[col] = raw[col].astype(object).where(raw[col].notna(), np.nan)
    for v in vehicles:
        MISSING_FIELDS.observe(1 - len(v.model_dump(exclude_none=True)) / len(_state["raw_cols"]))
    return raw


def _transform(raw: pd.DataFrame) -> pd.DataFrame:
    X = _state["preprocessor"].transform(raw)
    return pd.DataFrame(X, columns=_state["feature_meta"]["feature_names"])


def _predict(
    vehicles: list[VehicleFeatures], endpoint: str
) -> tuple[list[PredictionResponse], pd.DataFrame, pd.DataFrame]:
    _require_model()
    start = time.perf_counter()
    try:
        raw = _to_frame(vehicles)
        X = _transform(raw)
        proba = _state["model"].predict_proba(X)[:, 1]
    except Exception:
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc()
        logger.exception("Prediction failed")
        raise
    LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)

    out = []
    for p in proba:
        p = float(p)
        band = _risk_band(p)
        PREDICTIONS.labels(risk_band=band, endpoint=endpoint).inc()
        PROBA.observe(p)
        out.append(
            PredictionResponse(
                prediction=int(p >= _state["threshold"]),
                failure_probability=round(p, 4),
                risk_band=band,
                decision_threshold=_state["threshold"],
                model_version=_state["version"],
            )
        )
    return out, raw, X


def _contributions(X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Per-feature SHAP values (log-odds) for one row, computed natively."""
    model = _state["model"]
    name = type(model).__name__
    if name == "LGBMClassifier":
        contrib = model.predict(X, pred_contrib=True)[0]
        return contrib[:-1], float(contrib[-1])
    if name == "XGBClassifier":
        import xgboost as xgb

        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
        return contrib[:-1], float(contrib[-1])
    if hasattr(model, "coef_"):  # linear: exact SHAP under feature independence
        means = np.asarray(_state["feature_meta"]["feature_means"])
        coef = model.coef_[0]
        return coef * (X.to_numpy()[0] - means), float(model.intercept_[0] + coef @ means)
    raise HTTPException(status_code=501, detail=f"Explanations not supported for {name}")


def _raw_name(transformed: str) -> str:
    kind, _, rest = transformed.partition("__")
    if kind == "cat":
        for col in sorted(_state["feature_meta"]["categorical_cols"], key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
    return rest


# ---- routes -----------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "ev-battery-failure-prediction", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _state.get("model") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=_state.get("version") if loaded else None,
    )


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    _require_model()
    meta = _state["meta"]
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        model_version=_state["version"],
        decision_threshold=_state["threshold"],
        risk_bands={"medium_from": round(_state["medium_cutoff"], 4), "high_from": _state["threshold"]},
        trained_at=meta.get("trained_at"),
        git_sha=meta.get("git_sha"),
        test_metrics=_state["test_metrics"],
        n_features=len(_state["raw_cols"]),
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: VehicleFeatures) -> PredictionResponse:
    preds, _, _ = _predict([features], "predict")
    return preds[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    preds, _, _ = _predict(request.vehicles, "batch")
    return BatchResponse(predictions=preds, n_flagged=sum(p.prediction for p in preds))


@app.post("/explain", response_model=ExplanationResponse)
def explain(features: VehicleFeatures, top_k: int = 8) -> ExplanationResponse:
    preds, raw, X = _predict([features], "explain")
    contrib, base = _contributions(X)

    grouped: dict[str, float] = {}
    for fname, value in zip(X.columns, contrib, strict=True):
        key = _raw_name(fname)
        grouped[key] = grouped.get(key, 0.0) + float(value)
    top = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[: max(1, min(top_k, 25))]

    row = raw.iloc[0]
    factors = []
    for feat, value in top:
        rv = row.get(feat)
        rv = None if rv is None or (isinstance(rv, float) and np.isnan(rv)) else rv
        if isinstance(rv, np.generic):
            rv = rv.item()
        factors.append(FeatureContribution(feature=feat, value=rv, contribution=round(value, 4)))

    return ExplanationResponse(**preds[0].model_dump(), base_value=round(base, 4), top_factors=factors)
