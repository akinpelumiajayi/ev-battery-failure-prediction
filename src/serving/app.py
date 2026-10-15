"""
FastAPI serving layer.

Loads the fitted preprocessor + trained model artifacts (same objects
produced by src/features/build_features.py and src/training/train.py)
and exposes:
  GET  /health        - liveness/readiness probe for Kubernetes
  POST /predict        - single-vehicle failure prediction
  GET  /metrics        - Prometheus scrape endpoint

Run locally:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.serving.schemas import HealthResponse, PredictionResponse, VehicleFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="EV Battery Failure Prediction API",
    description="Predicts probability of EV battery failure from telemetry/usage features.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Prometheus metrics -----------------------------------------------
PREDICTION_COUNTER = Counter(
    "predictions_total", "Total number of predictions served", ["risk_band"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time spent computing a prediction"
)
FAILURE_PROBA_HIST = Histogram(
    "predicted_failure_probability", "Distribution of predicted failure probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

_state = {"model": None, "preprocessor": None, "metadata": None, "model_version": "unknown"}


def load_artifacts():
    try:
        _state["model"] = joblib.load(MODELS_DIR / "model.joblib")
        _state["preprocessor"] = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
        _state["metadata"] = joblib.load(ARTIFACTS_DIR / "feature_metadata.joblib")
        metrics_path = MODELS_DIR / "train_metrics.json"
        if metrics_path.exists():
            _state["model_version"] = json.loads(metrics_path.read_text()).get("run_id", "unknown")[:8]
        logger.info("Model artifacts loaded. version=%s", _state["model_version"])
    except FileNotFoundError as e:
        logger.error("Could not load model artifacts: %s. Run the training pipeline first.", e)


def risk_band(p: float) -> str:
    if p < 0.3:
        return "low"
    if p < 0.6:
        return "medium"
    return "high"


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = _state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=_state["model_version"] if loaded else None,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: VehicleFeatures):
    if _state["model"] is None or _state["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline.")

    start = time.time()
    raw = pd.DataFrame([features.model_dump()])

    # Ensure every column the preprocessor expects is present (fills with NaN
    # -> imputer handles it), and drops anything extra the client sent.
    expected_cols = _state["metadata"]["numeric_cols"] + _state["metadata"]["categorical_cols"]
    for col in expected_cols:
        if col not in raw.columns:
            raw[col] = None
    raw = raw[expected_cols]

    X = _state["preprocessor"].transform(raw)
    proba = float(_state["model"].predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    band = risk_band(proba)

    PREDICTION_LATENCY.observe(time.time() - start)
    PREDICTION_COUNTER.labels(risk_band=band).inc()
    FAILURE_PROBA_HIST.observe(proba)

    return PredictionResponse(
        prediction=pred,
        failure_probability=round(proba, 4),
        risk_band=band,
        model_version=_state["model_version"],
    )
