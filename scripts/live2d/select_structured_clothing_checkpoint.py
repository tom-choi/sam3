"""Pick the best checkpoint from structured Live2D clothing metrics.

Input is the CSV produced by ``evaluate_structured_clothing.py``.  The score is
the mean of clothes / upper clothes / lower clothes best IoU, because this
matches the current Live2D deployment rule:

    clothes = upper clothes UNION lower clothes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_epoch(label: str) -> int | None:
    match = re.search(r"_e(\d+)$", label)
    return int(match.group(1)) if match else None


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def score_row(row: dict[str, str], weights: dict[str, float]) -> float:
    return (
        weights["clothes"] * float_value(row, "clothes_best_iou")
        + weights["upper"] * float_value(row, "upper_best_iou")
        + weights["lower"] * float_value(row, "lower_best_iou")
    ) / sum(weights.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--candidate-prefix", default="clothes_alias_v2")
    parser.add_argument("--baseline-label", default="base_sam3")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--clothes-weight", type=float, default=1.0)
    parser.add_argument("--upper-weight", type=float, default=1.0)
    parser.add_argument("--lower-weight", type=float, default=1.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.metrics_csv.open("r", encoding="utf-8-sig")))
    weights = {
        "clothes": args.clothes_weight,
        "upper": args.upper_weight,
        "lower": args.lower_weight,
    }
    for row in rows:
        row["epoch"] = parse_epoch(row.get("label", ""))
        row["structured_score"] = score_row(row, weights)

    baseline = next((row for row in rows if row.get("label") == args.baseline_label), None)
    baseline_score = score_row(baseline, weights) if baseline else None
    candidates = [
        row
        for row in rows
        if row.get("label", "").startswith(args.candidate_prefix)
        and row.get("epoch") is not None
    ]
    candidates.sort(key=lambda row: row["epoch"])

    best = None
    best_index = -1
    for idx, row in enumerate(candidates):
        if best is None or row["structured_score"] > best["structured_score"] + args.min_delta:
            best = row
            best_index = idx

    latest_index = len(candidates) - 1
    epochs_since_best = latest_index - best_index if best is not None else None
    should_stop = (
        best is not None
        and epochs_since_best is not None
        and epochs_since_best >= args.patience
    )

    payload = {
        "metrics_csv": str(args.metrics_csv),
        "candidate_prefix": args.candidate_prefix,
        "baseline_label": args.baseline_label,
        "weights": weights,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "baseline": baseline,
        "baseline_score": baseline_score,
        "best": best,
        "best_delta_vs_baseline": (
            best["structured_score"] - baseline_score
            if best is not None and baseline_score is not None
            else None
        ),
        "latest": candidates[-1] if candidates else None,
        "epochs_since_best": epochs_since_best,
        "should_stop": should_stop,
        "candidates": candidates,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
