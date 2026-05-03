"""Evaluate Live2D part predictions against COCO ground truth masks.

This is a lightweight diagnostic tool for the small Live2D experiments. It
computes class-aware greedy mask matching metrics and writes GT-vs-pred overlays
so we can inspect accuracy without relying only on the large weighted SAM3 loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


@dataclass
class MaskItem:
    image_id: int
    category_id: int
    mask: np.ndarray
    score: float
    area: int
    ann_id: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare COCO/RLE predictions with Live2D ground truth masks."
    )
    parser.add_argument(
        "--gt",
        default="dataset/live2d_parts_canonical_check/val/_annotations.coco.json",
        help="Ground-truth COCO annotation file.",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction JSON. Supports COCO result list or full COCO dict.",
    )
    parser.add_argument(
        "--images-root",
        default="dataset/live2d_parts_canonical_check/val",
        help="Directory containing validation images for overlay rendering.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/live2d_gt_eval",
        help="Directory to write metrics and overlays.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Metric file prefix. Defaults to prediction file stem.",
    )
    parser.add_argument(
        "--image-file",
        default=None,
        help="Optional single image filename to evaluate/render.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="IoU thresholds for greedy matching metrics.",
    )
    parser.add_argument(
        "--overlay-threshold",
        type=float,
        default=0.0,
        help="Only draw predictions with score >= this value.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=120,
        help="Maximum labels to draw per overlay panel.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return mask_utils.decode(rle).astype(bool)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle).astype(bool)
    raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")


def color_for_category(category_id: int) -> tuple[int, int, int]:
    rng = random.Random(category_id * 1009)
    return tuple(rng.randint(40, 235) for _ in range(3))


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_gt(gt_path: Path, image_file: str | None = None):
    coco = load_json(gt_path)
    categories = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    images = {int(img["id"]): img for img in coco["images"]}
    filename_to_id = {img["file_name"]: int(img["id"]) for img in coco["images"]}
    allowed_image_ids = set(images)
    if image_file is not None:
        if image_file not in filename_to_id:
            raise ValueError(f"Image file {image_file!r} not found in {gt_path}")
        allowed_image_ids = {filename_to_id[image_file]}

    gt_items: list[MaskItem] = []
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in allowed_image_ids:
            continue
        img = images[image_id]
        mask = decode_segmentation(ann["segmentation"], img["height"], img["width"])
        gt_items.append(
            MaskItem(
                image_id=image_id,
                category_id=int(ann["category_id"]),
                mask=mask,
                score=1.0,
                area=int(mask.sum()),
                ann_id=int(ann["id"]),
            )
        )
    return coco, categories, images, filename_to_id, gt_items


def prediction_annotations(pred_data: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(pred_data, list):
        return pred_data, []
    if isinstance(pred_data, dict):
        annotations = pred_data.get("annotations", [])
        images = pred_data.get("images", [])
        return annotations, images
    raise TypeError(f"Unsupported prediction JSON type: {type(pred_data)!r}")


def map_prediction_image_ids(
    pred_images: list[dict[str, Any]],
    gt_filename_to_id: dict[str, int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for image in pred_images:
        pred_id = int(image["id"])
        file_name = image.get("file_name")
        if file_name in gt_filename_to_id:
            mapping[pred_id] = gt_filename_to_id[file_name]
        else:
            mapping[pred_id] = pred_id
    return mapping


def load_predictions(
    pred_path: Path,
    images: dict[int, dict[str, Any]],
    gt_filename_to_id: dict[str, int],
    allowed_image_ids: set[int],
) -> list[MaskItem]:
    pred_data = load_json(pred_path)
    annotations, pred_images = prediction_annotations(pred_data)
    pred_id_to_gt_id = map_prediction_image_ids(pred_images, gt_filename_to_id)

    pred_items: list[MaskItem] = []
    for idx, ann in enumerate(annotations, start=1):
        raw_image_id = int(ann["image_id"])
        image_id = pred_id_to_gt_id.get(raw_image_id, raw_image_id)
        if image_id not in allowed_image_ids:
            continue
        if image_id not in images:
            continue
        img = images[image_id]
        mask = decode_segmentation(ann["segmentation"], img["height"], img["width"])
        pred_items.append(
            MaskItem(
                image_id=image_id,
                category_id=int(ann["category_id"]),
                mask=mask,
                score=float(ann.get("score", 1.0)),
                area=int(mask.sum()),
                ann_id=int(ann.get("id", idx)) if "id" in ann else idx,
            )
        )
    return pred_items


def group_items(items: list[MaskItem]):
    grouped: dict[tuple[int, int], list[MaskItem]] = defaultdict(list)
    for item in items:
        grouped[(item.image_id, item.category_id)].append(item)
    return grouped


def evaluate_threshold(
    gt_items: list[MaskItem],
    pred_items: list[MaskItem],
    categories: dict[int, str],
    threshold: float,
) -> dict[str, Any]:
    gt_by_key = group_items(gt_items)
    pred_by_key = group_items(pred_items)
    keys = set(gt_by_key) | set(pred_by_key)

    total = {
        "threshold": threshold,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "matched_iou_sum": 0.0,
        "matched_count": 0,
        "gt_best_iou_sum": 0.0,
        "gt_count": 0,
        "pred_count": len(pred_items),
    }
    per_category: dict[int, dict[str, Any]] = {
        cid: {
            "category_id": cid,
            "category_name": name,
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_iou_sum": 0.0,
            "matched_count": 0,
            "gt_best_iou_sum": 0.0,
        }
        for cid, name in categories.items()
    }

    for image_id, category_id in sorted(keys):
        gt_list = gt_by_key.get((image_id, category_id), [])
        pred_list = sorted(
            pred_by_key.get((image_id, category_id), []),
            key=lambda item: item.score,
            reverse=True,
        )
        cat = per_category.setdefault(
            category_id,
            {
                "category_id": category_id,
                "category_name": str(category_id),
                "gt": 0,
                "pred": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "matched_iou_sum": 0.0,
                "matched_count": 0,
                "gt_best_iou_sum": 0.0,
            },
        )
        cat["gt"] += len(gt_list)
        cat["pred"] += len(pred_list)
        total["gt_count"] += len(gt_list)

        iou_matrix = np.zeros((len(pred_list), len(gt_list)), dtype=np.float32)
        for pi, pred in enumerate(pred_list):
            for gi, gt in enumerate(gt_list):
                iou_matrix[pi, gi] = mask_iou(pred.mask, gt.mask)

        if len(gt_list) > 0 and len(pred_list) > 0:
            best_per_gt = iou_matrix.max(axis=0)
            total["gt_best_iou_sum"] += float(best_per_gt.sum())
            cat["gt_best_iou_sum"] += float(best_per_gt.sum())

        matched_gt: set[int] = set()
        for pi in range(len(pred_list)):
            best_gt = -1
            best_iou = 0.0
            for gi in range(len(gt_list)):
                if gi in matched_gt:
                    continue
                iou = float(iou_matrix[pi, gi]) if iou_matrix.size else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi
            if best_gt >= 0 and best_iou >= threshold:
                matched_gt.add(best_gt)
                total["tp"] += 1
                total["matched_iou_sum"] += best_iou
                total["matched_count"] += 1
                cat["tp"] += 1
                cat["matched_iou_sum"] += best_iou
                cat["matched_count"] += 1
            else:
                total["fp"] += 1
                cat["fp"] += 1
        fn = len(gt_list) - len(matched_gt)
        total["fn"] += fn
        cat["fn"] += fn

    def finalize(row: dict[str, Any]) -> dict[str, Any]:
        tp = row["tp"]
        fp = row["fp"]
        fn = row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row["precision"] = precision
        row["recall"] = recall
        row["f1"] = f1
        row["mean_matched_iou"] = (
            row["matched_iou_sum"] / row["matched_count"] if row["matched_count"] else 0.0
        )
        row["mean_gt_best_iou"] = (
            row["gt_best_iou_sum"] / row.get("gt", row.get("gt_count", 0))
            if row.get("gt", row.get("gt_count", 0))
            else 0.0
        )
        return row

    total = finalize(total)
    per_category_rows = [
        finalize(row)
        for row in per_category.values()
        if row["gt"] > 0 or row["pred"] > 0
    ]
    per_category_rows.sort(key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return {"summary": total, "per_category": per_category_rows}


def draw_overlay(
    image: Image.Image,
    items: list[MaskItem],
    categories: dict[int, str],
    title: str,
    score_threshold: float = 0.0,
    max_labels: int = 120,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    kept = [item for item in items if item.score >= score_threshold]
    for item in sorted(kept, key=lambda x: x.area, reverse=True):
        color = color_for_category(item.category_id)
        mask_img = Image.fromarray((item.mask * 115).astype(np.uint8), mode="L")
        fill = Image.new("RGBA", base.size, (*color, 0))
        fill.putalpha(mask_img)
        overlay = Image.alpha_composite(overlay, fill)
    out = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(out)
    for item in sorted(kept, key=lambda x: x.area, reverse=True)[:max_labels]:
        x, y, w, h = mask_to_bbox(item.mask)
        color = color_for_category(item.category_id)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        name = categories.get(item.category_id, str(item.category_id))
        suffix = "" if item.score >= 0.999 else f" {item.score:.2f}"
        draw.text(
            (x + 2, y + 2),
            f"{name}{suffix}",
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
    banner = Image.new("RGB", (out.width, 40), (25, 25, 25))
    banner_draw = ImageDraw.Draw(banner)
    banner_draw.text((10, 12), title, fill=(255, 255, 255))
    panel = Image.new("RGB", (out.width, out.height + banner.height), (255, 255, 255))
    panel.paste(banner, (0, 0))
    panel.paste(out.convert("RGB"), (0, banner.height))
    return panel


def write_overlay_grid(
    output_path: Path,
    images_root: Path,
    image_info: dict[str, Any],
    gt_items: list[MaskItem],
    pred_items: list[MaskItem],
    categories: dict[int, str],
    pred_name: str,
    score_threshold: float,
    max_labels: int,
) -> None:
    image_path = images_root / image_info["file_name"]
    if not image_path.exists():
        return
    image = Image.open(image_path).convert("RGB")
    image_id = int(image_info["id"])
    gt_for_image = [item for item in gt_items if item.image_id == image_id]
    pred_for_image = [item for item in pred_items if item.image_id == image_id]
    gt_panel = draw_overlay(
        image, gt_for_image, categories, "ground truth", 0.0, max_labels
    )
    pred_panel = draw_overlay(
        image,
        pred_for_image,
        categories,
        pred_name,
        score_threshold,
        max_labels,
    )
    width = gt_panel.width + pred_panel.width
    height = max(gt_panel.height, pred_panel.height)
    grid = Image.new("RGB", (width, height), (255, 255, 255))
    grid.paste(gt_panel, (0, 0))
    grid.paste(pred_panel, (gt_panel.width, 0))
    grid.save(output_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "category_id",
        "category_name",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "mean_gt_best_iou",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    output_dir = Path(args.output_dir)
    images_root = Path(args.images_root)
    name = args.name or pred_path.stem.replace(".coco", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, categories, images, filename_to_id, gt_items = load_gt(gt_path, args.image_file)
    allowed_image_ids = {item.image_id for item in gt_items}
    pred_items = load_predictions(pred_path, images, filename_to_id, allowed_image_ids)

    metrics_by_threshold = {
        str(threshold): evaluate_threshold(gt_items, pred_items, categories, threshold)
        for threshold in args.thresholds
    }
    result = {
        "name": name,
        "gt": str(gt_path),
        "pred": str(pred_path),
        "image_file": args.image_file,
        "evaluated_image_ids": sorted(allowed_image_ids),
        "gt_count": len(gt_items),
        "pred_count": len(pred_items),
        "metrics": metrics_by_threshold,
    }

    metrics_path = output_dir / f"{name}_gt_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    threshold_for_csv = "0.5" if "0.5" in metrics_by_threshold else str(args.thresholds[0])
    write_csv(
        output_dir / f"{name}_per_category_iou{threshold_for_csv}.csv",
        metrics_by_threshold[threshold_for_csv]["per_category"],
    )

    for image_id in sorted(allowed_image_ids):
        image_info = images[image_id]
        overlay_path = output_dir / f"{name}_{Path(image_info['file_name']).stem}_gt_vs_pred.png"
        write_overlay_grid(
            overlay_path,
            images_root,
            image_info,
            gt_items,
            pred_items,
            categories,
            name,
            args.overlay_threshold,
            args.max_labels,
        )

    summary = metrics_by_threshold[threshold_for_csv]["summary"]
    print(json.dumps({
        "name": name,
        "threshold": float(threshold_for_csv),
        "gt_count": len(gt_items),
        "pred_count": len(pred_items),
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "mean_matched_iou": summary["mean_matched_iou"],
        "mean_gt_best_iou": summary["mean_gt_best_iou"],
        "metrics_path": str(metrics_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
