"""Optimize per-category score thresholds for Live2D part predictions.

The global threshold sweep is useful, but the current Live2D val set has very
different behavior by class: face/hair are already reliable, while accessories
and several small parts produce many false positives. This script chooses a
score threshold per category on a validation split, then evaluates the filtered
predictions as one combined set.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from evaluate_ground_truth import (
    MaskItem,
    evaluate_threshold,
    load_gt,
    load_predictions,
    mask_to_bbox,
)
from threshold_sweep import (
    filter_predictions,
    mask_nms,
    parse_category_map,
    remap_categories,
)


DEFAULT_SCORE_THRESHOLDS = [
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
    1.01,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose per-category score thresholds against Live2D GT."
    )
    parser.add_argument("--gt", required=True, help="Ground-truth COCO annotation file.")
    parser.add_argument("--pred", required=True, help="Prediction JSON file.")
    parser.add_argument(
        "--output-dir",
        default="runs/live2d_threshold_sweep",
        help="Directory to write threshold tables and filtered predictions.",
    )
    parser.add_argument("--name", default=None, help="Output prefix.")
    parser.add_argument("--image-file", default=None, help="Optional single image filename.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to optimize per-category F1.",
    )
    parser.add_argument(
        "--score-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_SCORE_THRESHOLDS,
        help="Candidate score thresholds. Include >1.0 to allow dropping a class.",
    )
    parser.add_argument(
        "--category-map",
        default=None,
        help="Optional category-id remap, same format as threshold_sweep.py.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="Same-image/same-category mask NMS IoU threshold.",
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.7,
        help="Fallback threshold for categories not present in the threshold table.",
    )
    return parser.parse_args()


def category_items(items: list[MaskItem], category_id: int) -> list[MaskItem]:
    return [item for item in items if item.category_id == category_id]


def summary_row(
    *,
    category_id: int,
    category_name: str,
    score_threshold: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    summary = metrics["summary"]
    return {
        "category_id": category_id,
        "category_name": category_name,
        "score_threshold": score_threshold,
        "gt": summary["gt_count"],
        "pred": summary["pred_count"],
        "tp": summary["tp"],
        "fp": summary["fp"],
        "fn": summary["fn"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "mean_matched_iou": summary["mean_matched_iou"],
        "mean_gt_best_iou": summary["mean_gt_best_iou"],
    }


def choose_threshold_for_category(
    *,
    category_id: int,
    category_name: str,
    gt_items: list[MaskItem],
    pred_items: list[MaskItem],
    images: dict[int, dict[str, Any]],
    score_thresholds: list[float],
    iou_threshold: float,
    nms_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, Any]:
    gt_cat = category_items(gt_items, category_id)
    pred_cat = category_items(pred_items, category_id)
    rows = []
    for score_threshold in score_thresholds:
        filtered = filter_predictions(
            pred_cat,
            images,
            score_threshold=score_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        filtered = mask_nms(filtered, nms_iou)
        metrics = evaluate_threshold(
            gt_cat, filtered, {category_id: category_name}, iou_threshold
        )
        rows.append(
            summary_row(
                category_id=category_id,
                category_name=category_name,
                score_threshold=score_threshold,
                metrics=metrics,
            )
        )

    return max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            -row["pred"],
            row["score_threshold"],
        ),
    )


def apply_category_thresholds(
    pred_items: list[MaskItem],
    thresholds_by_id: dict[int, float],
    default_threshold: float,
) -> list[MaskItem]:
    return [
        item
        for item in pred_items
        if item.score >= thresholds_by_id.get(item.category_id, default_threshold)
    ]


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def prediction_to_coco(
    pred_items: list[MaskItem],
    images: dict[int, dict[str, Any]],
    categories: dict[int, str],
) -> dict[str, Any]:
    annotations = []
    for ann_id, item in enumerate(pred_items, start=1):
        annotations.append(
            {
                "id": ann_id,
                "image_id": item.image_id,
                "category_id": item.category_id,
                "segmentation": encode_rle(item.mask),
                "bbox": mask_to_bbox(item.mask),
                "area": int(item.area),
                "score": float(item.score),
                "iscrowd": 0,
            }
        )
    return {
        "images": [images[image_id] for image_id in sorted(images)],
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "live2d_part"}
            for category_id, name in sorted(categories.items())
        ],
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or pred_path.stem.replace(".coco", "")

    _, categories, images, filename_to_id, gt_items = load_gt(gt_path, args.image_file)
    allowed_image_ids = {item.image_id for item in gt_items}
    pred_items = load_predictions(pred_path, images, filename_to_id, allowed_image_ids)
    pred_items = remap_categories(pred_items, parse_category_map(args.category_map))

    category_rows = [
        choose_threshold_for_category(
            category_id=category_id,
            category_name=category_name,
            gt_items=gt_items,
            pred_items=pred_items,
            images=images,
            score_thresholds=args.score_thresholds,
            iou_threshold=args.iou_threshold,
            nms_iou=args.nms_iou,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )
        for category_id, category_name in sorted(categories.items())
    ]

    thresholds_by_id = {
        int(row["category_id"]): float(row["score_threshold"]) for row in category_rows
    }
    thresholds_by_name = {
        row["category_name"]: float(row["score_threshold"]) for row in category_rows
    }
    filtered = apply_category_thresholds(
        pred_items, thresholds_by_id, args.default_threshold
    )
    filtered = filter_predictions(
        filtered,
        images,
        score_threshold=0.0,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
    )
    filtered = mask_nms(filtered, args.nms_iou)
    final_metrics = evaluate_threshold(
        gt_items, filtered, categories, args.iou_threshold
    )

    csv_path = output_dir / f"{name}_per_class_thresholds.csv"
    json_path = output_dir / f"{name}_per_class_thresholds.json"
    filtered_path = output_dir / f"{name}_per_class_filtered_predictions.coco.json"

    write_csv(csv_path, category_rows)
    filtered_path.write_text(
        json.dumps(prediction_to_coco(filtered, images, categories), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    payload = {
        "version": 1,
        "name": name,
        "gt": str(gt_path),
        "pred": str(pred_path),
        "image_file": args.image_file,
        "iou_threshold": args.iou_threshold,
        "nms_iou": args.nms_iou,
        "default_threshold": args.default_threshold,
        "category_thresholds_by_id": {str(k): v for k, v in thresholds_by_id.items()},
        "category_thresholds_by_name": thresholds_by_name,
        "per_category": category_rows,
        "filtered_predictions": str(filtered_path),
        "filtered_metrics": final_metrics,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = final_metrics["summary"]
    print(
        json.dumps(
            {
                "name": name,
                "csv": str(csv_path),
                "json": str(json_path),
                "filtered_predictions": str(filtered_path),
                "pred_count": summary["pred_count"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
