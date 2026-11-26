"""Inference logging: the API's write path and the drift check's read path."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.monitoring import inference_logs
from src.monitoring.drift_check import compute_drift
from src.serving import app as app_module
from src.serving import inference_log
from src.serving.inference_log import InferenceLogger, partition_key
from src.serving.schemas import VehicleFeatures

EXAMPLE = VehicleFeatures.model_config["json_schema_extra"]["example"]


def _records(n: int, version: str = "v1") -> list[dict]:
    return [
        {"model_version": version, "state_of_health": 90.0 - i, "vehicle_brand": "Tesla", "risk_band": "low"}
        for i in range(n)
    ]


def _payload(df: pd.DataFrame) -> list[dict]:
    """Rows as a client would send them: missing readings as null, not NaN, and
    ratios inside the API contract (the synthetic fixture lets a few exceed 1)."""
    df = df.copy()
    ratios = [c for c in df.columns if c.endswith("_ratio")]
    df[ratios] = df[ratios].clip(0, 1)
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get() if labels else metric._value.get()


class FakeS3:
    """Just enough of the boto3 S3 client for the writer and the reader."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, _name):
        store = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [k for (b, k) in store if b == Bucket and k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in sorted(keys)]}

        return Paginator()


# ---- writer -----------------------------------------------------------------
def test_partition_key_layout():
    key = partition_key("abc123", datetime(2026, 11, 26, 13, 5, 9, tzinfo=UTC))
    assert key.startswith("date=2026-11-26/model=abc123/130509-")
    assert key.endswith(".parquet")


def test_local_roundtrip(tmp_path):
    log = InferenceLogger(str(tmp_path), batch_size=1000)
    log.log(_records(3, "v1"))
    log.log(_records(2, "v2"))
    assert log.flush() == 5

    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
    assert len(files) == 2  # one file per model version
    assert {f.split("/")[1] for f in files} == {"model=v1", "model=v2"}

    back = inference_logs.load_inference_logs(str(tmp_path), window_days=1)
    assert len(back) == 5
    assert set(back["model_version"]) == {"v1", "v2"}
    assert len(inference_logs.load_inference_logs(str(tmp_path), 1, model_version="v2")) == 2


def test_flush_on_empty_buffer_writes_nothing(tmp_path):
    assert InferenceLogger(str(tmp_path)).flush() == 0
    assert not list(tmp_path.rglob("*.parquet"))


def test_sample_rate_zero_logs_nothing(tmp_path):
    log = InferenceLogger(str(tmp_path), sample_rate=0.0)
    log.log(_records(50))
    assert log.flush() == 0


def test_full_buffer_sheds_load_instead_of_growing(tmp_path):
    before = _counter(inference_log.RECORDS_DROPPED, reason="buffer_full")
    log = InferenceLogger(str(tmp_path), batch_size=1000, max_buffer=10)
    log.log(_records(8))
    log.log(_records(5))  # only 2 fit
    assert _counter(inference_log.RECORDS_DROPPED, reason="buffer_full") - before == 3
    assert log.flush() == 10


def test_write_failure_is_counted_not_raised(tmp_path):
    class BrokenS3:
        def put_object(self, **_):
            raise ConnectionError("storage down")

    before = _counter(inference_log.RECORDS_DROPPED, reason="write_error")
    log = InferenceLogger("s3://bucket/prefix", s3_client=BrokenS3())
    log.log(_records(4))
    assert log.flush() == 0  # no exception
    assert _counter(inference_log.RECORDS_DROPPED, reason="write_error") - before == 4


def test_s3_roundtrip_with_prefix():
    s3 = FakeS3()
    log = InferenceLogger("s3://logs/ev-battery/prod", s3_client=s3)
    log.log(_records(6))
    assert log.flush() == 6
    ((bucket, key),) = s3.objects
    assert bucket == "logs"
    assert key.startswith("ev-battery/prod/date=")

    back = inference_logs.load_inference_logs("s3://logs/ev-battery/prod", 1, s3_client=s3)
    assert len(back) == 6


def test_background_thread_flushes_on_batch_size_and_stop(tmp_path):
    log = InferenceLogger(str(tmp_path), batch_size=5, flush_seconds=3600)
    log.start()
    log.log(_records(5))  # hits the batch size: wakes the writer
    log.log(_records(2))
    log.stop()  # flushes the remaining 2
    assert len(inference_logs.load_inference_logs(str(tmp_path), 1)) == 7


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("INFERENCE_LOG_URI", raising=False)
    assert inference_log.from_env() is None
    monkeypatch.setenv("INFERENCE_LOG_URI", str(tmp_path))
    monkeypatch.setenv("INFERENCE_LOG_SAMPLE_RATE", "0.25")
    log = inference_log.from_env()
    assert log is not None
    assert log.sample_rate == 0.25


# ---- reader CLI -------------------------------------------------------------
def test_cli_skips_when_too_little_traffic(tmp_path):
    log = InferenceLogger(str(tmp_path / "logs"))
    log.log(_records(3))
    log.flush()
    out = tmp_path / "batch.parquet"
    rc = inference_logs.main(["--uri", str(tmp_path / "logs"), "--min-rows", "10", "--out", str(out)])
    assert rc == inference_logs.INSUFFICIENT_DATA
    assert not out.exists()


def test_cli_writes_batch(tmp_path):
    log = InferenceLogger(str(tmp_path / "logs"))
    log.log(_records(12))
    log.flush()
    out = tmp_path / "batch.parquet"
    rc = inference_logs.main(["--uri", str(tmp_path / "logs"), "--min-rows", "10", "--out", str(out)])
    assert rc == 0
    assert len(pd.read_parquet(out)) == 12


# ---- end to end through the API ---------------------------------------------
@pytest.fixture
def logging_client(trained_artifacts, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ARTIFACTS_DIR", trained_artifacts["artifacts_dir"])
    monkeypatch.setattr(app_module, "MODELS_DIR", trained_artifacts["models_dir"])
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.setenv("INFERENCE_LOG_URI", str(tmp_path / "logs"))
    monkeypatch.setenv("INFERENCE_LOG_FLUSH_SECONDS", "3600")  # only the shutdown flush
    return tmp_path / "logs"


def test_api_logs_every_prediction_and_flushes_on_shutdown(logging_client, raw_df):
    with TestClient(app_module.app) as c:
        single = c.post("/predict", json=EXAMPLE).json()
        rows = raw_df.drop(columns=["vehicle_id", "battery_serial", "battery_failure"]).head(20)
        batch = c.post("/predict/batch", json={"vehicles": _payload(rows)}).json()
    app_module._state["inference_log"] = None  # don't leak into other tests

    logs = inference_logs.load_inference_logs(str(logging_client), 1)
    assert len(logs) == 21
    assert set(logs["endpoint"]) == {"predict", "batch"}
    assert set(logs["model_version"]) == {"abcdef12"}
    first = logs[logs["endpoint"] == "predict"].iloc[0]
    assert first["failure_probability"] == single["failure_probability"]
    assert first["risk_band"] == single["risk_band"]
    assert first["state_of_health"] == EXAMPLE["state_of_health"]
    assert logs[logs["endpoint"] == "batch"]["request_id"].nunique() == 1  # one request, 20 rows
    assert sorted(logs.loc[logs["endpoint"] == "batch", "failure_probability"]) == sorted(
        p["failure_probability"] for p in batch["predictions"]
    )


def test_logged_traffic_feeds_the_drift_check(logging_client, raw_df):
    """The logged columns line up with the training reference, so PSI runs on them."""
    rows = raw_df.drop(columns=["vehicle_id", "battery_serial", "battery_failure"]).head(200)
    with TestClient(app_module.app) as c:
        c.post("/predict/batch", json={"vehicles": _payload(rows)})
    app_module._state["inference_log"] = None

    logs = inference_logs.load_inference_logs(str(logging_client), 1)
    report = compute_drift(raw_df, logs, psi_threshold=0.2, share_threshold=0.1)
    assert report["n_features_checked"] > 50  # model inputs, not the log's own metadata columns
    assert report["n_rows_current"] == 200
