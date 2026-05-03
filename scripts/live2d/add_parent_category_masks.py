"""Add parent semantic masks to a COCO dataset.

For Live2D, some fine-grained parts are hard to learn immediately. Clothing is
one of them: upper/lower clothes are sparse and often fragmented. This utility
adds a parent `clothes` category by unioning child masks such as `upper clothes`
and `lower clothes`, while keeping the original child annotations intact.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add parent category union masks to COCO splits.")
    parser.add_argument("--input-root", required=True, help="Input COCO dataset root.")
    parser.add_argument("--output-root", required=True, help="Output COCO dataset root.")
    parser.add_argument("--parent-name", default="clothes", help="New parent category name.")
    parser.add_argument(
        "--child-names",
        nargs="+",
        default=["upper clothes", "lower clothes"],
        help="Child category names to union into the parent mask.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output root first.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return mask_utils.decode(rle).astype(bool)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def copy_split_files(input_split: Path, output_split: Path) -> None:
    output_split.mkdir(parents=True, exist_ok=True)
    for path in input_split.iterdir():
        if path.name == "_annotations.coco.json":
            continue
        if path.is_file():
            shutil.copy2(path, output_split / path.name)


def add_parent_masks(
    coco: dict[str, Any],
    *,
    parent_name: str,
    child_names: list[str],
) -> tuple[dict[str, Any], int]:
    categories = [dict(cat) for cat in coco["categories"]]
    name_to_id = {cat["name"]: int(cat["id"]) for cat in categories}
    child_ids = {name_to_id[name] for name in child_names if name in name_to_id}
    if not child_ids:
        raise ValueError(f"No child categories found among: {child_names}")

    if parent_name in name_to_id:
        parent_id = name_to_id[parent_name]
    else:
        parent_id = max(int(cat["id"]) for cat in categories) + 1
        categories.append(
            {"id": parent_id, "name": parent_name, "supercategory": "live2d_part"}
        )

    images = {int(img["id"]): img for img in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    annotations = [dict(ann) for ann in coco["annotations"]]
    next_ann_id = max([int(ann["id"]) for ann in annotations] or [0]) + 1
    added = 0
    existing_parent_images = {
        int(ann["image_id"])
        for ann in annotations
        if int(ann["category_id"]) == parent_id
    }

    for image_id, anns in sorted(anns_by_image.items()):
        if image_id in existing_parent_images:
            continue
        child_anns = [ann for ann in anns if int(ann["category_id"]) in child_ids]
        if not child_anns:
            continue
        image = images[image_id]
        height, width = int(image["height"]), int(image["width"])
        union = np.zeros((height, width), dtype=bool)
        for ann in child_anns:
            union |= decode_segmentation(ann["segmentation"], height, width)
        area = int(union.sum())
        if area <= 0:
            continue
        annotations.append(
            {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": parent_id,
                "segmentation": encode_rle(union),
                "bbox": mask_to_bbox(union),
                "area": area,
                "iscrowd": 0,
                "attributes": {"parent_from": child_names},
            }
        )
        next_ann_id += 1
        added += 1

    out = dict(coco)
    out["categories"] = categories
    out["annotations"] = annotations
    out.setdefault("info", {})
    out["info"]["parent_category_masks"] = {
        "parent_name": parent_name,
        "parent_id": parent_id,
        "child_names": child_names,
        "child_ids": sorted(child_ids),
        "added_annotations": added,
    }
    return out, added


def summarize(coco: dict[str, Any]) -> dict[str, Any]:
    cat_names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    counts = Counter(int(ann["category_id"]) for ann in coco["annotations"])
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": len(coco["categories"]),
        "annotation_counts": {
            cat_names[cid]: count
            for cid, count in sorted(counts.items(), key=lambda item: (-item[1], cat_names[item[0]]))
        },
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "parent_name": args.parent_name,
        "child_names": args.child_names,
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        input_split = input_root / split
        output_split = output_root / split
        copy_split_files(input_split, output_split)
        coco = load_json(input_split / "_annotations.coco.json")
        coco, added = add_parent_masks(
            coco, parent_name=args.parent_name, child_names=args.child_names
        )
        write_json(output_split / "_annotations.coco.json", coco)
        summary["splits"][split] = {"added_parent_annotations": added, **summarize(coco)}

    write_json(output_root / "parent_category_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
