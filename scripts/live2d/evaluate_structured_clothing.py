"""Evaluate Live2D clothing with a simple structured parent rule.

The prompt-bank scan often returns a poor direct ``clothes`` / ``outfit`` mask
and better child masks from prompts such as ``bodice`` and ``skirt``.  For the
Live2D workflow, the parent clothing mask is more naturally derived as
``upper clothes`` union ``lower clothes``.  This script measures that rule
against GT without changing the saved raw predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


CLOTHING_CLASSES = ["clothes", "upper clothes", "lower clothes"]
CHILD_PROMPT_PRIORITY = {
    "upper clothes": [
        "bodice",
        "dress top",
        "upper garment",
        "shirt",
        "blouse",
        "jacket",
        "upper outfit",
        "upper clothes",
    ],
    "lower clothes": [
        "skirt",
        "dress skirt",
        "lower garment",
        "pants",
        "shorts",
        "lower outfit",
        "lower clothes",
    ],
}


def decode_rle(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rle)
    else:
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_gt(gt_path: Path, image_file: str):
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    image = next(img for img in data["images"] if img["file_name"] == image_file)
    height, width = int(image["height"]), int(image["width"])
    cat_by_id = {int(cat["id"]): cat["name"] for cat in data["categories"]}
    gt_masks = {name: [] for name in CLOTHING_CLASSES}
    for ann in data["annotations"]:
        if int(ann["image_id"]) != int(image["id"]):
            continue
        name = cat_by_id.get(int(ann["category_id"]))
        if name in gt_masks:
            gt_masks[name].append(decode_rle(ann["segmentation"], height, width))
    return image, cat_by_id, gt_masks


def load_predictions(pred_path: Path, cat_by_id: dict[int, str], height: int, width: int):
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    anns = data["annotations"] if isinstance(data, dict) else data
    predictions = []
    for ann in anns:
        category_id = int(ann["category_id"])
        category_name = cat_by_id.get(category_id, str(category_id))
        attributes = ann.get("attributes", {})
        predictions.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "prompt": attributes.get("prompt", ""),
                "score": float(ann.get("score", 1.0)),
                "mask": decode_rle(ann["segmentation"], height, width),
            }
        )
    return predictions


def select_child_prediction(predictions, category_name: str):
    candidates = [pred for pred in predictions if pred["category_name"] == category_name]
    if not candidates:
        return None
    priority = CHILD_PROMPT_PRIORITY.get(category_name, [])
    priority_index = {prompt.casefold(): idx for idx, prompt in enumerate(priority)}

    def key(pred):
        prompt = str(pred.get("prompt", "")).casefold()
        return (
            priority_index.get(prompt, len(priority_index) + 1),
            -float(pred.get("score", 0.0)),
            int(pred["mask"].sum()),
        )

    return sorted(candidates, key=key)[0]


def build_structured_predictions(predictions, height: int, width: int):
    structured = []
    child_masks = []
    child_scores = []
    for child_name in ["upper clothes", "lower clothes"]:
        child = select_child_prediction(predictions, child_name)
        if child is None:
            continue
        structured.append(child)
        child_masks.append(child["mask"])
        child_scores.append(float(child.get("score", 1.0)))

    if child_masks:
        parent_mask = np.zeros((height, width), dtype=bool)
        for mask in child_masks:
            parent_mask |= mask
        structured.append(
            {
                "category_id": next(
                    pred["category_id"] for pred in predictions if pred["category_name"] == "clothes"
                ),
                "category_name": "clothes",
                "prompt": "structured: upper clothes + lower clothes",
                "score": float(sum(child_scores) / len(child_scores)),
                "mask": parent_mask,
            }
        )
    return structured


def evaluate_class(gt_masks, pred_masks, threshold: float = 0.5):
    pairs = []
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            pairs.append((mask_iou(pred_mask, gt_mask), pred_idx, gt_idx))
    pairs.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    matched_ious = []
    for value, pred_idx, gt_idx in pairs:
        if value < threshold or pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matched_ious.append(value)

    tp = len(matched_ious)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    best_iou = (
        sum(max([mask_iou(pred, gt) for pred in pred_masks] or [0.0]) for gt in gt_masks) / len(gt_masks)
        if gt_masks
        else 0.0
    )
    return {
        "gt": len(gt_masks),
        "pred": len(pred_masks),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_iou": best_iou,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    image, cat_by_id, gt_masks = load_gt(args.gt, args.image_file)
    height, width = int(image["height"]), int(image["width"])
    rows = []
    for label in args.labels:
        pred_path = args.pred_dir / f"{args.stem}_{label}_predictions.coco.json"
        if not pred_path.exists():
            print("missing prediction:", pred_path)
            continue
        predictions = load_predictions(pred_path, cat_by_id, height, width)
        structured = build_structured_predictions(predictions, height, width)
        by_class = {
            name: [pred["mask"] for pred in structured if pred["category_name"] == name]
            for name in CLOTHING_CLASSES
        }
        metrics = {name: evaluate_class(gt_masks[name], by_class[name]) for name in CLOTHING_CLASSES}
        rows.append(
            {
                "label": label,
                "structured_predictions": len(structured),
                "clothes_f1": metrics["clothes"]["f1"],
                "clothes_best_iou": metrics["clothes"]["best_iou"],
                "upper_f1": metrics["upper clothes"]["f1"],
                "upper_best_iou": metrics["upper clothes"]["best_iou"],
                "lower_f1": metrics["lower clothes"]["f1"],
                "lower_best_iou": metrics["lower clothes"]["best_iou"],
            }
        )

    if rows:
        base = rows[0]
        for row in rows:
            row["delta_clothes_vs_first"] = row["clothes_best_iou"] - base["clothes_best_iou"]
            row["delta_upper_vs_first"] = row["upper_best_iou"] - base["upper_best_iou"]
            row["delta_lower_vs_first"] = row["lower_best_iou"] - base["lower_best_iou"]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote:", args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
