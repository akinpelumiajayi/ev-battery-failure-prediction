"""
Fetch the raw dataset for a fresh clone or a CI runner.

The DVC pointer (`data/raw/ev_battery_data.csv.dvc`) is the source of truth
for *which* version of the data the pipeline expects. This script downloads
the public mirror listed in `params.yaml -> data.remote_url` and refuses to
continue unless the md5 matches the pointer, so CI can never silently train
on the wrong data.

    python -m src.data.fetch_data           # no-op if the correct file exists
    python -m src.data.fetch_data --force   # re-download
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

from src.common import PROJECT_ROOT, get_logger, load_params

logger = get_logger(__name__)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 — integrity check, not security
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(raw_path: Path) -> str:
    pointer = raw_path.with_name(raw_path.name + ".dvc")
    meta = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    return str(meta["outs"][0]["md5"])


def fetch(force: bool = False) -> Path:
    params = load_params()
    raw_path = PROJECT_ROOT / params["data"]["raw_path"]
    url = params["data"]["remote_url"]
    want = expected_md5(raw_path)

    if raw_path.exists() and not force:
        if md5sum(raw_path) == want:
            logger.info("%s already present and verified (md5 %s)", raw_path.name, want)
            return raw_path
        logger.warning("%s exists but md5 differs from DVC pointer — re-downloading", raw_path)

    logger.info("Downloading %s", url)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        gz_path = Path(tmp) / "data.csv.gz"
        urllib.request.urlretrieve(url, gz_path)  # noqa: S310 — fixed https URL from params
        out_path = Path(tmp) / raw_path.name
        with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        got = md5sum(out_path)
        if got != want:
            raise RuntimeError(f"Checksum mismatch: expected {want}, downloaded {got}")
        shutil.move(out_path, raw_path)

    logger.info("Saved %s (md5 verified %s)", raw_path, want)
    return raw_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        logger.error("Data fetch failed: %s", exc)
        sys.exit(1)
