#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Convert layered Live2D PSD files into a COCO instance-segmentation dataset.

Typical usage:

    python scripts/live2d/psd_to_coco.py \
        --psd-root D:/datasets/live2d_psd \
        --output-root D:/datasets/live2d_parts \
        --max-files 50

The default taxonomy lives next to this script. It maps common English,
Japanese, and Chinese Live2D layer names to a 30-part MVP taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    from psd_tools import PSDImage
except ImportError as exc:  # pragma: no cover - exercised by users without deps.
    raise SystemExit(
        "Missing dependency: psd-tools. Install it with "
        '`pip install -e ".[live2d]"` or `pip install psd-tools`.'
    ) from exc

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:  # pragma: no cover - exercised by users without deps.
    raise SystemExit(
        "Missing dependency: pycocotools. Install it with "
        '`pip install -e ".[live2d]"` or `pip install pycocotools`.'
    ) from exc


DEFAULT_TAXONOMY = Path(__file__).with_name("live2d_parts_taxonomy.json")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CategoryRule:
    id: int
    name: str
    aliases: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class MatchedLayer:
    layer: Any
    category: CategoryRule
    path: tuple[str, ...]
    index: int | None = None
    override_label: str | None = None


@dataclass(frozen=True)
class LayerRecord:
    layer: Any
    path: tuple[str, ...]
    index: int


@dataclass(frozen=True)
class LayerOverride:
    index: int | None
    path: str | None
    category: CategoryRule | None
    ignore: bool
    label: str | None
    note: str | None


@dataclass(frozen=True)
class PSDOverride:
    ignore: bool
    label: str | None
    note: str | None


def normalize_text(value: str) -> str:
    """Normalize names while preserving CJK text useful for PSD layer matching."""

    return unicodedata.normalize("NFKC", value).casefold()


def compact_text(value: str) -> str:
    value = normalize_text(value)
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", value)


def slugify(value: str, fallback: str = "item") -> str:
    value = normalize_text(value)
    value = re.sub(r"[^0-9a-z]+", "_", value).strip("_")
    return value or fallback


def load_taxonomy(path: Path) -> list[CategoryRule]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rules = []
    for item in data["categories"]:
        aliases = tuple(compact_text(alias) for alias in item.get("aliases", []))
        patterns = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in item.get("patterns", [])
        )
        rules.append(
            CategoryRule(
                id=int(item["id"]),
                name=str(item["name"]),
                aliases=aliases,
                patterns=patterns,
            )
        )

    ids = [rule.id for rule in rules]
    if sorted(ids) != list(range(1, len(ids) + 1)):
        raise ValueError("Taxonomy category ids must be consecutive and 1-based.")
    return rules


def normalize_rel_path(value: str) -> str:
    return Path(value).as_posix().replace("\\", "/")


def load_layer_overrides(
    path: Path | None, categories: list[CategoryRule]
) -> tuple[dict[str, list[LayerOverride]], dict[str, PSDOverride]]:
    if path is None:
        return {}, {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    category_by_name = {normalize_text(category.name): category for category in categories}
    category_by_id = {category.id: category for category in categories}
    psd_entries = data.get("psds", data)
    if not isinstance(psd_entries, dict):
        raise ValueError("Layer overrides must be a JSON object or contain a 'psds' object.")

    layer_overrides: dict[str, list[LayerOverride]] = defaultdict(list)
    psd_overrides: dict[str, PSDOverride] = {}
    for psd_key, entries_or_config in psd_entries.items():
        psd_key = normalize_rel_path(str(psd_key))
        entries = entries_or_config
        psd_config: dict[str, Any] = {}
        if isinstance(entries_or_config, dict):
            psd_config = entries_or_config
            entries = entries_or_config.get("layers", [])
        elif isinstance(entries_or_config, list):
            for entry in entries_or_config:
                if isinstance(entry, dict) and entry.get("ignore_psd"):
                    psd_config = entry
                    break
        else:
            raise ValueError(f"Overrides for {psd_key!r} must be a list or object.")

        if psd_config:
            psd_overrides[psd_key] = PSDOverride(
                ignore=bool(psd_config.get("ignore_psd", False)),
                label=psd_config.get("label"),
                note=psd_config.get("note"),
            )

        if not isinstance(entries, list):
            raise ValueError(f"Overrides for {psd_key!r} must be a list.")

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Override entries for {psd_key!r} must be objects.")
            if entry.get("ignore_psd"):
                continue

            layer_index = entry.get("index")
            if layer_index is not None:
                layer_index = int(layer_index)
            layer_path = entry.get("path")
            if layer_path is not None:
                layer_path = str(layer_path)
            if layer_index is None and layer_path is None:
                raise ValueError(
                    f"Override for {psd_key!r} needs either 'index' or 'path'."
                )

            category = None
            category_value = entry.get("category")
            if category_value not in (None, "", "ignore"):
                if isinstance(category_value, int) or str(category_value).isdigit():
                    category = category_by_id.get(int(category_value))
                else:
                    category = category_by_name.get(normalize_text(str(category_value)))
                if category is None:
                    raise ValueError(
                        f"Unknown category {category_value!r} in override for {psd_key!r}."
                    )

            ignore = bool(entry.get("ignore", False)) or category_value == "ignore"
            if category is None and not ignore:
                raise ValueError(
                    f"Override for {psd_key!r} layer {layer_index or layer_path!r} "
                    "must set either 'category' or 'ignore': true."
                )

            layer_overrides[psd_key].append(
                LayerOverride(
                    index=layer_index,
                    path=layer_path,
                    category=category,
                    ignore=ignore,
                    label=entry.get("label"),
                    note=entry.get("note"),
                )
            )
    return dict(layer_overrides), psd_overrides


def is_macos_resource_psd(path: Path, psd_root: Path) -> bool:
    rel = path.relative_to(psd_root)
    return path.name.startswith("._") or "__MACOSX" in rel.parts


def match_category(layer_path: Iterable[str], rules: list[CategoryRule]):
    raw = " / ".join(layer_path)
    raw_norm = normalize_text(raw)
    compact = compact_text(raw)

    best_rule = None
    best_score = -1
    for rule in rules:
        for pattern in rule.patterns:
            match = pattern.search(raw_norm)
            if match:
                score = 100 + len(match.group(0))
                if score > best_score:
                    best_rule = rule
                    best_score = score
        for alias in rule.aliases:
            if alias and alias in compact:
                score = 50 + len(alias)
                if score > best_score:
                    best_rule = rule
                    best_score = score
    return best_rule


def lookup_layer_override(
    overrides: dict[str, list[LayerOverride]],
    psd_rel: str,
    record: LayerRecord,
) -> LayerOverride | None:
    entries = overrides.get(psd_rel, [])
    if not entries:
        return None

    joined = " / ".join(record.path)
    for entry in entries:
        index_matches = entry.index is None or entry.index == record.index
        path_matches = entry.path is None or entry.path == joined
        if index_matches and path_matches:
            return entry
    return None


def is_visible(layer: Any, include_hidden: bool) -> bool:
    return include_hidden or bool(getattr(layer, "visible", True))


def iter_children(layer: Any):
    try:
        yield from layer
    except TypeError:
        return


def iter_matched_layers(
    layer: Any,
    rules: list[CategoryRule],
    *,
    layer_mode: str,
    include_hidden: bool,
    parent_path: tuple[str, ...] = (),
):
    if not is_visible(layer, include_hidden):
        return

    name = str(getattr(layer, "name", "layer"))
    path = (*parent_path, name)
    is_group = bool(getattr(layer, "is_group", lambda: False)())

    if is_group:
        category = match_category(path, rules)
        if layer_mode == "auto" and category is not None:
            yield MatchedLayer(layer=layer, category=category, path=path)
            return
        if layer_mode == "groups" and category is not None:
            yield MatchedLayer(layer=layer, category=category, path=path)
        for child in iter_children(layer):
            yield from iter_matched_layers(
                child,
                rules,
                layer_mode=layer_mode,
                include_hidden=include_hidden,
                parent_path=path,
            )
        return

    if layer_mode != "groups":
        category = match_category(path, rules)
        if category is not None:
            yield MatchedLayer(layer=layer, category=category, path=path)


def iter_visible_leaf_paths(
    layer: Any,
    *,
    include_hidden: bool,
    parent_path: tuple[str, ...] = (),
):
    if not is_visible(layer, include_hidden):
        return

    name = str(getattr(layer, "name", "layer"))
    path = (*parent_path, name)
    is_group = bool(getattr(layer, "is_group", lambda: False)())
    if is_group:
        for child in iter_children(layer):
            yield from iter_visible_leaf_paths(
                child, include_hidden=include_hidden, parent_path=path
            )
        return
    yield layer, path


def collect_visible_leaf_records(
    psd: Any, *, include_hidden: bool
) -> list[LayerRecord]:
    records = []
    for child in iter_children(psd):
        for layer, layer_path in iter_visible_leaf_paths(
            child, include_hidden=include_hidden
        ):
            records.append(
                LayerRecord(layer=layer, path=layer_path, index=len(records) + 1)
            )
    return records


def composite_node(node: Any):
    try:
        return node.composite(force=True)
    except TypeError:
        return node.composite()


def paste_alpha(full: Image.Image, alpha: Image.Image, offset: tuple[int, int]) -> None:
    x, y = offset
    dst_w, dst_h = full.size
    src_w, src_h = alpha.size

    dst_left = max(0, x)
    dst_top = max(0, y)
    src_left = max(0, -x)
    src_top = max(0, -y)
    width = min(src_w - src_left, dst_w - dst_left)
    height = min(src_h - src_top, dst_h - dst_top)
    if width <= 0 or height <= 0:
        return

    crop = alpha.crop((src_left, src_top, src_left + width, src_top + height))
    full.paste(crop, (dst_left, dst_top))


def layer_to_mask(
    layer: Any,
    canvas_size: tuple[int, int],
    *,
    alpha_threshold: int,
) -> np.ndarray | None:
    image = composite_node(layer)
    if image is None:
        return None
    image = image.convert("RGBA")

    if image.size == canvas_size:
        alpha = image.getchannel("A")
    else:
        alpha = Image.new("L", canvas_size, 0)
        bbox = getattr(layer, "bbox", None)
        if hasattr(bbox, "x1") and hasattr(bbox, "y1"):
            x = int(bbox.x1)
            y = int(bbox.y1)
        elif isinstance(bbox, (tuple, list)) and len(bbox) >= 2:
            x = int(bbox[0])
            y = int(bbox[1])
        else:
            x = 0
            y = 0
        paste_alpha(alpha, image.getchannel("A"), (x, y))

    mask = np.asarray(alpha) >= alpha_threshold
    return mask if mask.any() else None


def render_psd_rgb(psd: Any, background: tuple[int, int, int]) -> Image.Image:
    image = composite_node(psd)
    if image is None:
        raise ValueError("PSD did not produce a composite image.")
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (*background, 255))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def annotation_from_mask(
    mask: np.ndarray,
    *,
    annotation_id: int,
    image_id: int,
    category: CategoryRule,
    attributes: dict[str, Any],
):
    rle = encode_binary_mask(mask)
    area = float(mask_utils.area(rle))
    if area <= 0:
        return None

    bbox = [float(v) for v in mask_utils.toBbox(rle)]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category.id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": attributes,
    }


def stable_filename(psd_path: Path, psd_root: Path) -> str:
    rel = psd_path.relative_to(psd_root).with_suffix("").as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(rel, fallback=psd_path.stem)}_{digest}.png"


def split_files(
    psd_files: list[Path],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("Split ratios must satisfy train > 0 and train + val <= 1.")

    rng = random.Random(seed)
    shuffled = psd_files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)
    elif n == 2:
        n_train, n_val = 1, 1
    else:
        n_train, n_val = n, 0

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def empty_coco(categories: list[CategoryRule]) -> dict[str, Any]:
    return {
        "info": {
            "description": "Live2D PSD part segmentation dataset generated from layered PSD files",
            "version": "mvp-1",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category.id,
                "name": category.name,
                "supercategory": "live2d_part",
            }
            for category in categories
        ],
    }


def convert_one_psd(
    psd_path: Path,
    *,
    psd_root: Path,
    split_dir: Path,
    categories: list[CategoryRule],
    image_id: int,
    next_annotation_id: int,
    args: argparse.Namespace,
    unmatched_counter: Counter[str],
    unmatched_examples: dict[str, set[str]],
):
    psd = PSDImage.open(psd_path)
    width, height = int(psd.width), int(psd.height)
    canvas_size = (width, height)
    psd_rel = normalize_rel_path(str(psd_path.relative_to(psd_root)))

    image_name = stable_filename(psd_path, psd_root)
    render_psd_rgb(psd, args.background).save(split_dir / image_name)

    leaf_records = collect_visible_leaf_records(
        psd, include_hidden=args.include_hidden
    )
    matched = []
    if args.layer_mode == "path":
        for record in leaf_records:
            override = lookup_layer_override(args.layer_overrides, psd_rel, record)
            if override is not None:
                if override.ignore:
                    continue
                if override.category is not None:
                    matched.append(
                        MatchedLayer(
                            layer=record.layer,
                            category=override.category,
                            path=record.path,
                            index=record.index,
                            override_label=override.label,
                        )
                    )
                continue

            category = match_category(record.path, categories)
            if category is not None:
                matched.append(
                    MatchedLayer(
                        layer=record.layer,
                        category=category,
                        path=record.path,
                        index=record.index,
                    )
                )
    else:
        for child in iter_children(psd):
            matched.extend(
                iter_matched_layers(
                    child,
                    categories,
                    layer_mode=args.layer_mode,
                    include_hidden=args.include_hidden,
                )
            )

        ignored_paths = set()
        matched_paths_for_overrides = {item.path for item in matched}
        for record in leaf_records:
            override = lookup_layer_override(args.layer_overrides, psd_rel, record)
            if override is None:
                continue
            if override.ignore:
                ignored_paths.add(record.path)
                continue
            if override.category is not None and record.path not in matched_paths_for_overrides:
                matched.append(
                    MatchedLayer(
                        layer=record.layer,
                        category=override.category,
                        path=record.path,
                        index=record.index,
                        override_label=override.label,
                    )
                )
        if ignored_paths:
            matched = [item for item in matched if item.path not in ignored_paths]

    matched_paths = {item.path for item in matched}
    matched_indexes = {item.index for item in matched if item.index is not None}
    for record in leaf_records:
        if record.index in matched_indexes:
            continue
        if args.layer_mode == "auto" and any(
            record.path[: len(path)] == path for path in matched_paths
        ):
            continue
        if lookup_layer_override(args.layer_overrides, psd_rel, record) is not None:
            continue
        if match_category(record.path, categories) is not None:
            continue
        mask = layer_to_mask(
            record.layer, canvas_size, alpha_threshold=args.alpha_threshold
        )
        if mask is not None and int(mask.sum()) >= args.min_area:
            name = f"{record.index:03d}: {' / '.join(record.path)}"
            unmatched_counter[name] += 1
            unmatched_examples[name].add(psd_rel)

    masks_by_category: dict[int, list[tuple[np.ndarray, MatchedLayer]]] = defaultdict(
        list
    )
    for item in matched:
        mask = layer_to_mask(
            item.layer, canvas_size, alpha_threshold=args.alpha_threshold
        )
        if mask is None or int(mask.sum()) < args.min_area:
            continue
        masks_by_category[item.category.id].append((mask, item))

    annotations = []
    category_by_id = {category.id: category for category in categories}

    if args.instance_mode == "category":
        for category_id in sorted(masks_by_category):
            merged = np.zeros((height, width), dtype=bool)
            layer_paths = []
            layer_indices = []
            manual_labels = []
            for mask, item in masks_by_category[category_id]:
                merged |= mask
                layer_paths.append(" / ".join(item.path))
                if item.index is not None:
                    layer_indices.append(item.index)
                if item.override_label:
                    manual_labels.append(item.override_label)
            attributes = {
                "source_psd": psd_rel,
                "merged_layer_paths": layer_paths,
            }
            if layer_indices:
                attributes["merged_layer_indices"] = layer_indices
            if manual_labels:
                attributes["manual_labels"] = sorted(set(manual_labels))
            ann = annotation_from_mask(
                merged,
                annotation_id=next_annotation_id,
                image_id=image_id,
                category=category_by_id[category_id],
                attributes=attributes,
            )
            if ann is not None:
                annotations.append(ann)
                next_annotation_id += 1
    else:
        for category_id in sorted(masks_by_category):
            for mask, item in masks_by_category[category_id]:
                attributes = {
                    "source_psd": psd_rel,
                    "layer_path": " / ".join(item.path),
                }
                if item.index is not None:
                    attributes["layer_index"] = item.index
                if item.override_label:
                    attributes["manual_label"] = item.override_label
                ann = annotation_from_mask(
                    mask,
                    annotation_id=next_annotation_id,
                    image_id=image_id,
                    category=category_by_id[category_id],
                    attributes=attributes,
                )
                if ann is not None:
                    annotations.append(ann)
                    next_annotation_id += 1

    if len(annotations) < args.min_annotations:
        (split_dir / image_name).unlink(missing_ok=True)
        return None, [], next_annotation_id

    image_record = {
        "id": image_id,
        "file_name": image_name,
        "width": width,
        "height": height,
        "source_psd": psd_rel,
    }
    return image_record, annotations, next_annotation_id


def parse_background(value: str) -> tuple[int, int, int]:
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "transparent": (255, 255, 255),
    }
    if value in named:
        return named[value]
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
    raise argparse.ArgumentTypeError(
        "Background must be white, black, gray, transparent, or #RRGGBB."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Live2D PSD layers into COCO segmentation annotations."
    )
    parser.add_argument("--psd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--layer-overrides",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with manual per-PSD layer overrides. Entries can "
            "force a category or mark meaningless/non-training layers as ignored."
        ),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--alpha-threshold", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--min-annotations", type=int, default=1)
    parser.add_argument(
        "--layer-mode",
        choices=("path", "auto", "groups"),
        default="path",
        help=(
            "path: match visible leaf layers using their full PSD path; "
            "auto: use a matching group as a part and skip children; "
            "groups: only use matching groups."
        ),
    )
    parser.add_argument(
        "--instance-mode",
        choices=("category", "layer"),
        default="category",
        help="category merges all matched layers for each part; layer keeps them separate.",
    )
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument(
        "--background",
        type=parse_background,
        default=(255, 255, 255),
        help="RGB matte for transparent PSD areas: white, black, gray, transparent, or #RRGGBB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output-root before writing the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    psd_root = args.psd_root.resolve()
    output_root = args.output_root.resolve()
    categories = load_taxonomy(args.taxonomy)
    args.layer_overrides, psd_overrides = load_layer_overrides(
        args.layer_overrides, categories
    )

    if not psd_root.exists():
        raise SystemExit(f"PSD root does not exist: {psd_root}")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ignored_psds = []
    psd_files = []
    for psd_path in sorted(psd_root.rglob("*.psd")):
        rel = normalize_rel_path(str(psd_path.relative_to(psd_root)))
        if is_macos_resource_psd(psd_path, psd_root):
            ignored_psds.append(
                {
                    "source_psd": rel,
                    "label": "macos_resource",
                    "note": "Skipped macOS resource-fork sidecar file.",
                }
            )
            continue
        psd_override = psd_overrides.get(rel)
        if psd_override is not None and psd_override.ignore:
            ignored_psds.append(
                {
                    "source_psd": rel,
                    "label": psd_override.label,
                    "note": psd_override.note,
                }
            )
            continue
        psd_files.append(psd_path)
    if args.max_files is not None:
        psd_files = psd_files[: args.max_files]
    if not psd_files:
        raise SystemExit(f"No PSD files found under {psd_root}")

    splits = split_files(
        psd_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    unmatched_counter: Counter[str] = Counter()
    unmatched_examples: dict[str, set[str]] = defaultdict(set)
    total_images = 0
    total_annotations = 0
    skipped = 0

    for split in SPLITS:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        coco = empty_coco(categories)
        next_annotation_id = 1

        for image_id, psd_path in enumerate(splits[split], start=1):
            try:
                image_record, annotations, next_annotation_id = convert_one_psd(
                    psd_path,
                    psd_root=psd_root,
                    split_dir=split_dir,
                    categories=categories,
                    image_id=image_id,
                    next_annotation_id=next_annotation_id,
                    args=args,
                    unmatched_counter=unmatched_counter,
                    unmatched_examples=unmatched_examples,
                )
            except Exception as exc:
                skipped += 1
                rel = normalize_rel_path(str(psd_path.relative_to(psd_root)))
                ignored_psds.append(
                    {
                        "source_psd": rel,
                        "label": "conversion_error",
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"warning: skipped {rel}: {type(exc).__name__}: {exc}")
                continue
            if image_record is None:
                skipped += 1
                continue
            coco["images"].append(image_record)
            coco["annotations"].extend(annotations)

        with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

        total_images += len(coco["images"])
        total_annotations += len(coco["annotations"])
        print(
            f"{split}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations"
        )

    unmatched_report = [
        {
            "layer_path": name,
            "count": count,
            "examples": sorted(unmatched_examples[name])[:5],
        }
        for name, count in unmatched_counter.most_common()
    ]
    with (output_root / "unmatched_layers.json").open("w", encoding="utf-8") as f:
        json.dump(unmatched_report, f, indent=2, ensure_ascii=False)

    with (output_root / "ignored_psds.json").open("w", encoding="utf-8") as f:
        json.dump(ignored_psds, f, indent=2, ensure_ascii=False)

    print(
        f"done: {total_images} images, {total_annotations} annotations, "
        f"{skipped} skipped PSDs, {len(ignored_psds)} ignored PSDs"
    )
    print(f"unmatched layer report: {output_root / 'unmatched_layers.json'}")
    print(f"ignored PSD report: {output_root / 'ignored_psds.json'}")


if __name__ == "__main__":
    main()
