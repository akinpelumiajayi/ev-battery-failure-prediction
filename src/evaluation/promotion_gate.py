"""
Champion / challenger gate.

Compares the freshly trained model's test metrics (the challenger) with the
metrics of the model currently on `main` (the champion) and fails if the
challenger regresses PR-AUC by more than `evaluation.max_pr_auc_regression`.
Writes a Markdown comparison table (posted as a PR comment by CI).

    python -m src.evaluation.promotion_gate --baseline champion_metrics.json --summary out.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common import MODELS_DIR, get_logger, load_params, read_json

logger = get_logger(__name__)

ROWS = [
    ("test_roc_auc", "ROC-AUC", True),
    ("test_pr_auc", "PR-AUC", True),
    ("test_recall", "Recall", True),
    ("test_precision", "Precision", True),
    ("test_f1", "F1", True),
    ("test_brier", "Brier score", False),
    ("test_cost_per_vehicle", "Expected cost / vehicle", False),
    ("test_threshold", "Decision threshold", None),
]


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, int | float) else "—"


def compare(challenger: dict, champion: dict | None, max_regression: float) -> tuple[bool, str]:
    lines = [
        "### Model report",
        "",
        f"Challenger: **{challenger.get('model', '?')}**"
        + (
            f" · Champion (main): **{champion.get('model', '?')}**"
            if champion
            else " · no champion on main yet"
        ),
        "",
        "| Metric | Champion (main) | Challenger (this run) | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, label, higher_is_better in ROWS:
        new = challenger.get(key)
        old = champion.get(key) if champion else None
        delta = ""
        if isinstance(new, int | float) and isinstance(old, int | float):
            d = new - old
            mark = ""
            if higher_is_better is not None and abs(d) > 1e-4:
                mark = " ✅" if (d > 0) == higher_is_better else " ⚠️"
            delta = f"{d:+.4f}{mark}"
        lines.append(f"| {label} | {_fmt(old)} | {_fmt(new)} | {delta} |")

    passed = True
    if champion and isinstance(champion.get("test_pr_auc"), int | float):
        drop = champion["test_pr_auc"] - challenger["test_pr_auc"]
        if drop > max_regression:
            passed = False
            lines += ["", f"❌ **Gate failed**: PR-AUC regressed by {drop:.4f} (> {max_regression})."]
    if passed:
        lines += ["", f"✅ **Gate passed** (max allowed PR-AUC regression {max_regression})."]
    return passed, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, help="champion test_metrics.json (from main)")
    parser.add_argument("--summary", type=Path, help="write the Markdown report here")
    args = parser.parse_args()

    params = load_params()
    challenger = read_json(MODELS_DIR / "test_metrics.json")
    champion = None
    if args.baseline and args.baseline.exists() and args.baseline.stat().st_size > 0:
        champion = read_json(args.baseline)

    passed, report = compare(challenger, champion, params["evaluation"]["max_pr_auc_regression"])
    print(report)
    if args.summary:
        args.summary.write_text(report, encoding="utf-8")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
