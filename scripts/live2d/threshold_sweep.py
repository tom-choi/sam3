"""Sweep prediction score thresholds against Live2D COCO ground truth.

This script reuses the lightweight GT evaluator and adds the bits we need for
taxonomy-alignment experiments: optional category-id remapping and per-class
mask NMS after remapping. That lets old 24-class predictions be compared against
the hair-merged 20-class taxonomy without treating side/back hair as unrelated
classes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluate_ground_truth import (
    MaskItem,
    evaluate_threshold,
    load_gt,
    load_predictions,
    mask_iou,
)


OLD24_TO_HAIR_MERGED20 = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 8,
    10: 8,
    11: 8,
    12: 8,
    13: 9,
    14: 10,
    15: 11,
    16: 12,
    17: 13,
    18: 14,
    19: 15,
    20: 16,
    21: 17,
    22: 18,
    23: 19,
    24: 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep score thresholds for Live2D COCO/RLE predictions."
    )
    parser.add_argument("--gt", required=True, help="Ground-truth COCO annotation file.")
    parser.add_argument("--pred", required=True, help="Prediction JSON file.")
    parser.add_argument(
        "--output-dir",
        default="runs/live2d_threshold_sweep",
        help="Directory to write CSV/JSON sweep results.",
    )
    parser.add_argument("--name", default=None, help="Output prefix.")
    parser.add_argument(
        "--image-file",
        default=None,
        help="Optional single image filename to evaluate.",
    )
    parser.add_argument(
        "--score-thresholds",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ],
        help="Prediction score thresholds to sweep.",
    )
    parser.add_argument(
        "--iou-thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="IoU thresholds for class-aware greedy matching.",
    )
    parser.add_argument(
        "--category-map",
        default=None,
        help=(
            "Category-id remap. Use 'old24_to_hair_merged20', a JSON file, "
            "or comma-separated pairs like '8=8,9=8,13=9'."
        ),
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="Optional same-image/same-category mask NMS IoU threshold. 0 disables NMS.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Drop predictions smaller than this image-area ratio.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=1.0,
        help="Drop predictions larger than this image-area ratio.",
    )
    return parser.parse_args()


def parse_category_map(value: str | None) -> dict[int, int] | None:
    if value in (None, ""):
        return None
    if value == "old24_to_hair_merged20":
        return OLD24_TO_HAIR_MERGED20

    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in data.items()}

    mapping: dict[int, int] = {}
    for pair in value.split(","):
        old, new = pair.split("=", 1)
        mapping[int(old)] = int(new)
    return mapping


def remap_categories(
    pred_items: list[MaskItem],
    category_map: dict[int, int] | None,
) -> list[MaskItem]:
    if not category_map:
        return pred_items
    out = []
    for item in pred_items:
        new_category_id = category_map.get(item.category_id)
        if new_category_id is None:
            continue
        out.append(replace(item, category_id=new_category_id))
    return out


def filter_predictions(
    pred_items: list[MaskItem],
    images: dict[int, dict[str, Any]],
    *,
    score_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[MaskItem]:
    out = []
    for item in pred_items:
        if item.score < score_threshold:
            continue
        image = images[item.image_id]
        image_area = max(1, int(image["height"]) * int(image["width"]))
        ratio = item.area / image_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        out.append(item)
    return out


def mask_nms(pred_items: list[MaskItem], nms_iou: float) -> list[MaskItem]:
    if nms_iou <= 0:
        return pred_items

    grouped: dict[tuple[int, int], list[MaskItem]] = {}
    for item in pred_items:
        grouped.setdefault((item.image_id, item.category_id), []).append(item)

    kept_all: list[MaskItem] = []
    for items in grouped.values():
        kept: list[MaskItem] = []
        for item in sorted(items, key=lambda pred: pred.score, reverse=True):
            if all(mask_iou(item.mask, kept_item.mask) <= nms_iou for kept_item in kept):
                kept.append(item)
        kept_all.extend(kept)
    return kept_all


def compact_summary(
    name: str,
    score_threshold: float,
    iou_threshold: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    summary = metrics["summary"]
    return {
        "name": name,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "gt_count": summary["gt_count"],
        "pred_count": summary["pred_count"],
        "tp": summary["tp"],
        "fp": summary["fp"],
        "fn": summary["fn"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "mean_matched_iou": summary["mean_matched_iou"],
        "mean_gt_best_iou": summary["mean_gt_best_iou"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    name = args.name or pred_path.stem.replace(".coco", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, categories, images, filename_to_id, gt_items = load_gt(gt_path, args.image_file)
    allowed_image_ids = {item.image_id for item in gt_items}
    pred_items = load_predictions(pred_path, images, filename_to_id, allowed_image_ids)
    pred_items = remap_categories(pred_items, parse_category_map(args.category_map))

    rows: list[dict[str, Any]] = []
    full: dict[str, Any] = {
        "name": name,
        "gt": str(gt_path),
        "pred": str(pred_path),
        "image_file": args.image_file,
        "category_map": args.category_map,
        "nms_iou": args.nms_iou,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "score_thresholds": args.score_thresholds,
        "iou_thresholds": args.iou_thresholds,
        "rows": [],
    }
    for score_threshold in args.score_thresholds:
        filtered = filter_predictions(
            pred_items,
            images,
            score_threshold=score_threshold,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )
        filtered = mask_nms(filtered, args.nms_iou)
        for iou_threshold in args.iou_thresholds:
            metrics = evaluate_threshold(gt_items, filtered, categories, iou_threshold)
            row = compact_summary(name, score_threshold, iou_threshold, metrics)
            rows.append(row)
            full["rows"].append(row)

    csv_path = output_dir / f"{name}_threshold_sweep.csv"
    json_path = output_dir / f"{name}_threshold_sweep.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")

    best_by_iou = {}
    for iou_threshold in args.iou_thresholds:
        candidates = [row for row in rows if row["iou_threshold"] == iou_threshold]
        best_by_iou[str(iou_threshold)] = max(
            candidates,
            key=lambda row: (row["f1"], row["precision"], row["recall"]),
        )

    print(
        json.dumps(
            {
                "name": name,
                "csv": str(csv_path),
                "json": str(json_path),
                "best_by_iou": best_by_iou,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
