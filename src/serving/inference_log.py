"""
Inference logging: every prediction the API serves, written to object storage
so the drift check can monitor real traffic instead of a simulated batch.

    request -> API -> response (unchanged latency)
                 \\-> in-memory buffer --(every N rows or T seconds)--> Parquet file
                      <uri>/date=YYYY-MM-DD/model=<version>/<HHMMSS>-<id>.parquet

One record per prediction: timestamp, request id, endpoint, model version, the
raw input features exactly as the model saw them (after schema validation,
before imputation), the probability, the risk band and the decision.
The request schema only accepts telemetry (no vehicle or owner identifiers),
so the log holds no personal data.

Configured by environment variables; logging is off unless a URI is set:

    INFERENCE_LOG_URI            s3://bucket/prefix   (any S3-compatible store; set
                                                       AWS_ENDPOINT_URL for R2/B2/MinIO)
                                 /path/to/dir          (local disk, e.g. docker-compose)
    INFERENCE_LOG_SAMPLE_RATE    fraction of predictions to keep (default 1.0)
    INFERENCE_LOG_BATCH_SIZE     flush after this many records (default 500)
    INFERENCE_LOG_FLUSH_SECONDS  ... or after this long (default 60)

Logging must never break serving: the request path only appends to a bounded
buffer, writes happen on a background thread, and a failed write drops that
batch and counts it in `inference_log_dropped_total`.
"""

from __future__ import annotations

import io
import logging
import os
import random
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from prometheus_client import Counter

logger = logging.getLogger("ev-battery-api.inference-log")

RECORDS_WRITTEN = Counter("inference_log_records_total", "Prediction records written to the inference log")
RECORDS_DROPPED = Counter(
    "inference_log_dropped_total", "Prediction records not written to the inference log", ["reason"]
)


def partition_key(model_version: str, when: datetime) -> str:
    """Key of one batch, relative to the log root. src/monitoring/inference_logs.py reads this layout."""
    return f"date={when:%Y-%m-%d}/model={model_version}/{when:%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"


class InferenceLogger:
    def __init__(
        self,
        uri: str,
        sample_rate: float = 1.0,
        batch_size: int = 500,
        flush_seconds: float = 60.0,
        max_buffer: int = 50_000,
        s3_client: Any = None,
    ) -> None:
        self.uri = uri
        self.sample_rate = min(max(sample_rate, 0.0), 1.0)
        self.batch_size = max(1, batch_size)
        self.flush_seconds = flush_seconds
        self.max_buffer = max_buffer
        self._s3 = s3_client
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if uri.startswith("s3://"):
            bucket, _, prefix = uri.removeprefix("s3://").partition("/")
            self._bucket: str | None = bucket
            self._prefix = prefix.strip("/")
            self._root: Path | None = None
        else:
            self._bucket = None
            self._prefix = ""
            self._root = Path(uri.removeprefix("file://"))

    # ---- request path: must stay cheap -------------------------------------------
    def log(self, records: Iterable[dict[str, Any]]) -> None:
        if self.sample_rate < 1.0:
            records = [r for r in records if random.random() < self.sample_rate]  # noqa: S311 (sampling, not crypto)
        else:
            records = list(records)
        if not records:
            return
        with self._lock:
            room = self.max_buffer - len(self._buffer)
            if room < len(records):  # writer is failing or far behind: shed load
                RECORDS_DROPPED.labels(reason="buffer_full").inc(len(records) - max(room, 0))
                records = records[: max(room, 0)]
            self._buffer.extend(records)
            full = len(self._buffer) >= self.batch_size
        if full:
            self._wake.set()

    # ---- background writer --------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="inference-log", daemon=True)
        self._thread.start()
        logger.info(
            "Inference logging to %s (sample rate %.2f, batch %d, every %.0fs)",
            self.uri,
            self.sample_rate,
            self.batch_size,
            self.flush_seconds,
        )

    def stop(self) -> None:
        """Flush what is buffered and stop the writer (called on shutdown)."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self.flush_seconds)
            self._wake.clear()
            self.flush()

    def flush(self) -> int:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return 0
        try:
            frame = pd.DataFrame.from_records(batch)
            now = datetime.now(UTC)
            for version, part in frame.groupby("model_version", sort=False):
                self._write(part, partition_key(str(version), now))
        except Exception:  # never let logging take the API down
            RECORDS_DROPPED.labels(reason="write_error").inc(len(batch))
            logger.exception("Inference log write failed; dropped %d records", len(batch))
            return 0
        RECORDS_WRITTEN.inc(len(batch))
        return len(batch)

    def _write(self, frame: pd.DataFrame, key: str) -> None:
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        if self._root is not None:
            path = self._root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.getvalue())
            return
        if self._s3 is None:
            import boto3  # lazy: only needed when logging to S3

            # Honours AWS_ENDPOINT_URL, so R2 / B2 / MinIO work without code changes.
            self._s3 = boto3.client("s3")
        full_key = f"{self._prefix}/{key}" if self._prefix else key
        self._s3.put_object(Bucket=self._bucket, Key=full_key, Body=buf.getvalue())


def from_env() -> InferenceLogger | None:
    uri = os.environ.get("INFERENCE_LOG_URI", "").strip()
    if not uri:
        return None
    return InferenceLogger(
        uri,
        sample_rate=float(os.environ.get("INFERENCE_LOG_SAMPLE_RATE", "1.0")),
        batch_size=int(os.environ.get("INFERENCE_LOG_BATCH_SIZE", "500")),
        flush_seconds=float(os.environ.get("INFERENCE_LOG_FLUSH_SECONDS", "60")),
    )
