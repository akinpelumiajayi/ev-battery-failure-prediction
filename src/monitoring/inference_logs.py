"""
Read the API's inference log (see src/serving/inference_log.py) back into one
batch for the drift check, so monitoring runs on real production traffic.

    python -m src.monitoring.inference_logs --out data/drift/production.parquet
    python -m src.monitoring.drift_check --current data/drift/production.parquet

Reads the last `monitoring.production_window_days` days, optionally for one
model version. If there are fewer than `monitoring.min_production_rows` rows
it exits with code 3 and writes nothing: a handful of requests is not a
population, and a PSI computed on one would raise false alarms.

The log location comes from --uri or INFERENCE_LOG_URI (s3://bucket/prefix or
a local directory). S3 credentials and AWS_ENDPOINT_URL come from the usual
AWS environment variables.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import get_logger, load_params

logger = get_logger(__name__)

INSUFFICIENT_DATA = 3


def _days(window_days: int, today: date) -> list[str]:
    return [f"{today - timedelta(days=i):%Y-%m-%d}" for i in range(window_days)]


def _local_files(root: Path, day: str, model_version: str | None) -> list[Path]:
    model = f"model={model_version}" if model_version else "model=*"
    return sorted(root.glob(f"date={day}/{model}/*.parquet"))


def _s3_frames(
    uri: str, days: list[str], model_version: str | None, client: Any = None
) -> list[pd.DataFrame]:
    if client is None:
        import boto3

        client = boto3.client("s3")
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    prefix = prefix.strip("/")
    frames = []
    paginator = client.get_paginator("list_objects_v2")
    for day in days:
        day_prefix = f"{prefix}/date={day}/" if prefix else f"date={day}/"
        if model_version:
            day_prefix += f"model={model_version}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=day_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    frames.append(pd.read_parquet(io.BytesIO(body)))
    return frames


def load_inference_logs(
    uri: str,
    window_days: int = 7,
    model_version: str | None = None,
    today: date | None = None,
    s3_client: Any = None,
) -> pd.DataFrame:
    days = _days(window_days, today or datetime.now(UTC).date())
    if uri.startswith("s3://"):
        frames = _s3_frames(uri, days, model_version, s3_client)
    else:
        root = Path(uri.removeprefix("file://"))
        frames = [pd.read_parquet(f) for day in days for f in _local_files(root, day, model_version)]
    if not frames:
        return pd.DataFrame()
    logs = pd.concat(frames, ignore_index=True)
    logger.info(
        "Read %d logged predictions from %d files (%s, last %d days)",
        len(logs),
        len(frames),
        uri,
        window_days,
    )
    return logs


def main(argv: list[str] | None = None) -> int:
    params = load_params()["monitoring"]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--uri", default=os.environ.get("INFERENCE_LOG_URI", ""))
    parser.add_argument("--days", type=int, default=params["production_window_days"])
    parser.add_argument("--model-version", default=None, help="only this model version (default: all)")
    parser.add_argument("--min-rows", type=int, default=params["min_production_rows"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.uri:
        logger.error("No inference log location: pass --uri or set INFERENCE_LOG_URI")
        return 2
    logs = load_inference_logs(args.uri, args.days, args.model_version)
    if len(logs) < args.min_rows:
        logger.warning(
            "Only %d logged predictions in the last %d days (need %d): not enough to check drift",
            len(logs),
            args.days,
            args.min_rows,
        )
        return INSUFFICIENT_DATA

    args.out.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(args.out, index=False)
    versions = (
        ", ".join(sorted(logs["model_version"].astype(str).unique())) if "model_version" in logs else "?"
    )
    logger.info("Wrote %d production rows (model versions: %s) to %s", len(logs), versions, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
