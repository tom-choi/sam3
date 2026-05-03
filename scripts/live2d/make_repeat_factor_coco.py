"""Build a repeat-factor COCO train split for small Live2D datasets.

This is a simple offline alternative to a custom PyTorch sampler. Images that
contain rare categories are duplicated in the training COCO JSON, while the
underlying PNG files are copied only once. Validation and test splits are copied
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a repeat-factor sampled COCO dataset for Live2D training."
    )
    parser.add_argument(
        "--input-root",
        default="dataset/live2d_parts_canonical_hair_merged",
        help="Input COCO dataset root with train/val/test splits.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/live2d_parts_canonical_hair_merged_rfs",
        help="Output dataset root.",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.4,
        help="Repeat images containing classes present in fewer than this fraction of train images.",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=4,
        help="Maximum integer repeat factor for one source image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output root before writing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_split_files(input_split: Path, output_split: Path) -> None:
    output_split.mkdir(parents=True, exist_ok=True)
    for path in input_split.iterdir():
        if path.name == "_annotations.coco.json":
            continue
        if path.is_file():
            shutil.copy2(path, output_split / path.name)


def category_image_frequencies(coco: dict[str, Any]) -> tuple[dict[int, float], dict[int, set[int]]]:
    image_ids = {int(image["id"]) for image in coco["images"]}
    category_to_images: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        category_to_images[int(ann["category_id"])].add(int(ann["image_id"]))
    denom = max(1, len(image_ids))
    frequencies = {
        int(category["id"]): len(category_to_images[int(category["id"])]) / denom
        for category in coco["categories"]
    }
    return frequencies, category_to_images


def repeat_for_image(
    categories_in_image: set[int],
    frequencies: dict[int, float],
    *,
    target_fraction: float,
    max_repeat: int,
) -> int:
    if not categories_in_image:
        return 1
    factor = 1.0
    for category_id in categories_in_image:
        freq = max(frequencies.get(category_id, 1.0), 1e-12)
        if freq < target_fraction:
            factor = max(factor, math.sqrt(target_fraction / freq))
    return max(1, min(max_repeat, math.ceil(factor)))


def repeat_train_coco(
    coco: dict[str, Any],
    *,
    target_fraction: float,
    max_repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images_by_id = {int(image["id"]): image for image in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cats_by_image: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        anns_by_image[image_id].append(ann)
        cats_by_image[image_id].add(int(ann["category_id"]))

    frequencies, _ = category_image_frequencies(coco)
    category_names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}

    repeated_images = []
    repeated_annotations = []
    repeat_rows = []
    next_image_id = 1
    next_ann_id = 1
    for source_image_id in sorted(images_by_id):
        image = images_by_id[source_image_id]
        category_ids = cats_by_image[source_image_id]
        repeat = repeat_for_image(
            category_ids,
            frequencies,
            target_fraction=target_fraction,
            max_repeat=max_repeat,
        )
        repeat_rows.append(
            {
                "source_image_id": source_image_id,
                "file_name": image["file_name"],
                "repeat": repeat,
                "categories": "|".join(category_names[cid] for cid in sorted(category_ids)),
            }
        )
        for _ in range(repeat):
            new_image = dict(image)
            new_image["id"] = next_image_id
            repeated_images.append(new_image)
            for ann in anns_by_image[source_image_id]:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                repeated_annotations.append(new_ann)
                next_ann_id += 1
            next_image_id += 1

    out = dict(coco)
    out["images"] = repeated_images
    out["annotations"] = repeated_annotations
    out.setdefault("info", {})
    out["info"]["repeat_factor_sampling"] = {
        "target_fraction": target_fraction,
        "max_repeat": max_repeat,
        "source_images": len(coco["images"]),
        "repeated_images": len(repeated_images),
        "source_annotations": len(coco["annotations"]),
        "repeated_annotations": len(repeated_annotations),
    }
    return out, repeat_rows


def write_repeat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source_image_id", "file_name", "repeat", "categories"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(coco: dict[str, Any]) -> dict[str, Any]:
    category_names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    counts = Counter(int(ann["category_id"]) for ann in coco["annotations"])
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": len(coco["categories"]),
        "annotation_counts": {
            category_names[category_id]: count
            for category_id, count in sorted(counts.items(), key=lambda item: (-item[1], category_names[item[0]]))
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
        "target_fraction": args.target_fraction,
        "max_repeat": args.max_repeat,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        input_split = input_root / split
        output_split = output_root / split
        copy_split_files(input_split, output_split)
        coco = load_json(input_split / "_annotations.coco.json")
        if split == "train":
            coco, repeat_rows = repeat_train_coco(
                coco,
                target_fraction=args.target_fraction,
                max_repeat=args.max_repeat,
            )
            write_repeat_csv(output_root / "train_repeat_factors.csv", repeat_rows)
        write_json(output_split / "_annotations.coco.json", coco)
        summary["splits"][split] = summarize(coco)

    write_json(output_root / "repeat_factor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
