"""Add text-prompt alias categories to a COCO training split.

SAM3 fine-tuning learns from the category name used as the query text. For
Live2D clothing, deployment prompt banks often use aliases such as `skirt`,
`bodice`, `outfit`, and `anime outfit`, while the canonical COCO categories are
`upper clothes`, `lower clothes`, and `clothes`.

This utility duplicates selected annotations under new alias category ids so the
same mask supervises multiple text prompts. It is intended for training splits;
validation/test splits can stay canonical.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ALIAS_MAP: dict[str, list[str]] = {
    "clothes": [
        "garment",
        "visible clothing",
        "clothing pieces",
        "character garment",
        "clothing",
        "costume",
        "dress",
        "uniform",
        "apparel",
    ],
    "upper clothes": [
        "bodice",
        "dress top",
        "upper garment",
        "shirt",
        "blouse",
        "jacket",
        "coat",
        "upper outfit",
    ],
    "lower clothes": [
        "skirt",
        "dress skirt",
        "lower garment",
        "pants",
        "shorts",
        "lower outfit",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate COCO annotations under alias category names.")
    parser.add_argument("--input-root", required=True, help="Input COCO dataset root.")
    parser.add_argument("--output-root", required=True, help="Output COCO dataset root.")
    parser.add_argument(
        "--alias-map",
        default=None,
        help="Optional JSON file mapping canonical category names to alias names.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Splits to augment. Other splits are copied unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output root first.")
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


def augment_alias_categories(coco: dict[str, Any], alias_map: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    categories = [dict(cat) for cat in coco["categories"]]
    name_to_id = {str(cat["name"]): int(cat["id"]) for cat in categories}
    next_cat_id = max([int(cat["id"]) for cat in categories] or [0]) + 1

    alias_id_by_source_id: dict[int, list[int]] = {}
    added_categories: list[dict[str, Any]] = []
    for source_name, aliases in alias_map.items():
        if source_name not in name_to_id:
            continue
        source_id = name_to_id[source_name]
        for alias_name in aliases:
            alias_name = str(alias_name).strip()
            if not alias_name or alias_name == source_name:
                continue
            if alias_name in name_to_id:
                alias_id = name_to_id[alias_name]
            else:
                alias_id = next_cat_id
                next_cat_id += 1
                name_to_id[alias_name] = alias_id
                category = {
                    "id": alias_id,
                    "name": alias_name,
                    "supercategory": "live2d_part_alias",
                    "alias_of": source_name,
                }
                categories.append(category)
                added_categories.append(category)
            alias_id_by_source_id.setdefault(source_id, []).append(alias_id)

    annotations = [dict(ann) for ann in coco["annotations"]]
    next_ann_id = max([int(ann["id"]) for ann in annotations] or [0]) + 1
    added_annotations = 0

    originals = list(annotations)
    for ann in originals:
        source_id = int(ann["category_id"])
        for alias_id in alias_id_by_source_id.get(source_id, []):
            alias_ann = dict(ann)
            alias_ann["id"] = next_ann_id
            next_ann_id += 1
            alias_ann["category_id"] = alias_id
            attrs = dict(alias_ann.get("attributes") or {})
            attrs["alias_of_category_id"] = source_id
            attrs["alias_of_category_name"] = next(
                cat["name"] for cat in categories if int(cat["id"]) == source_id
            )
            alias_ann["attributes"] = attrs
            annotations.append(alias_ann)
            added_annotations += 1

    out = dict(coco)
    out["categories"] = categories
    out["annotations"] = annotations
    out.setdefault("info", {})
    out["info"]["prompt_alias_categories"] = {
        "alias_map": alias_map,
        "added_categories": len(added_categories),
        "added_annotations": added_annotations,
    }

    counts = Counter(int(ann["category_id"]) for ann in annotations)
    cat_names = {int(cat["id"]): str(cat["name"]) for cat in categories}
    summary = {
        "images": len(out["images"]),
        "annotations": len(annotations),
        "categories": len(categories),
        "added_categories": len(added_categories),
        "added_annotations": added_annotations,
        "alias_categories": added_categories,
        "top_counts": {
            cat_names[cid]: count
            for cid, count in sorted(counts.items(), key=lambda item: (-item[1], cat_names[item[0]]))[:20]
        },
    }
    return out, summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    alias_map = DEFAULT_ALIAS_MAP
    if args.alias_map:
        alias_map = load_json(Path(args.alias_map))

    summaries: dict[str, Any] = {}
    augment_splits = set(args.splits)
    for input_split in sorted(path for path in input_root.iterdir() if path.is_dir()):
        output_split = output_root / input_split.name
        copy_split_files(input_split, output_split)
        ann_path = input_split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        coco = load_json(ann_path)
        if input_split.name in augment_splits:
            coco, summary = augment_alias_categories(coco, alias_map)
        else:
            summary = {
                "images": len(coco.get("images", [])),
                "annotations": len(coco.get("annotations", [])),
                "categories": len(coco.get("categories", [])),
                "added_categories": 0,
                "added_annotations": 0,
            }
        write_json(output_split / "_annotations.coco.json", coco)
        summaries[input_split.name] = summary
        print(json.dumps({"split": input_split.name, **summary}, ensure_ascii=False, indent=2))

    write_json(output_root / "prompt_alias_summary.json", summaries)
    print("wrote:", output_root)


if __name__ == "__main__":
    main()
