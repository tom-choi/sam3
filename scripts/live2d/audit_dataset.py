#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Create readable audit contact sheets for Live2D PSD and COCO outputs."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


FONT_CANDIDATES = (
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
    "C:/Windows/Fonts/NotoSansJP-VF.ttf",
    "C:/Windows/Fonts/NotoSansTC-VF.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)

OVERLAY_COLORS = np.array(
    [
        [230, 57, 70],
        [29, 185, 84],
        [69, 123, 157],
        [255, 183, 3],
        [131, 56, 236],
        [251, 86, 7],
        [0, 180, 216],
        [255, 0, 110],
        [42, 157, 143],
        [233, 196, 106],
        [141, 153, 174],
        [6, 214, 160],
        [239, 71, 111],
        [17, 138, 178],
        [255, 209, 102],
    ],
    dtype=np.float32,
) / 255.0


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_chars: int,
    max_lines: int = 2,
    line_height: int | None = None,
) -> None:
    lines = []
    for raw_line in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(raw_line, width=max_chars) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."

    x, y = xy
    line_height = line_height or int(getattr(font, "size", 14) * 1.25)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def count_layers(psd: Any) -> dict[str, int]:
    counts = {"layers": 0, "leaf": 0, "groups": 0}

    def walk(layer: Any) -> None:
        counts["layers"] += 1
        try:
            is_group = layer.is_group()
        except Exception:
            is_group = False
        if is_group:
            counts["groups"] += 1
            for child in layer:
                walk(child)
        else:
            counts["leaf"] += 1

    for child in psd:
        walk(child)
    return counts


def composite_rgb(psd: Any) -> Image.Image | None:
    image = psd.composite(force=True)
    if image is None:
        return None
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def build_psd_contact_sheet(
    psd_root: Path,
    output_root: Path,
    *,
    columns: int,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    thumb_dir = output_root / "composites"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for index, psd_path in enumerate(sorted(psd_root.rglob("*.psd")), start=1):
        rel = psd_path.relative_to(psd_root).as_posix()
        item: dict[str, Any] = {
            "index": index,
            "path": rel,
            "size_mb": round(psd_path.stat().st_size / 1024 / 1024, 2),
        }
        try:
            psd = PSDImage.open(psd_path)
            item["width"] = int(psd.width)
            item["height"] = int(psd.height)
            item.update(count_layers(psd))
            image = composite_rgb(psd)
            if image is not None:
                thumb = image.copy()
                thumb.thumbnail((320, 420), Image.Resampling.LANCZOS)
                thumb_path = thumb_dir / f"{index:03d}.jpg"
                thumb.save(thumb_path, quality=88)
                item["thumb"] = thumb_path.relative_to(output_root).as_posix()
            item["status"] = "ok"
        except Exception as exc:
            item["status"] = "error"
            item["error"] = f"{type(exc).__name__}: {exc}"
        summary.append(item)

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cell_w, cell_h = 440, 540
    rows = (len(summary) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(17)
    small = load_font(12)

    for item in summary:
        index = int(item["index"])
        x = ((index - 1) % columns) * cell_w
        y = ((index - 1) // columns) * cell_h
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(180, 180, 180))
        draw.text(
            (x + 8, y + 8),
            f"{index:02d} {item['status']} {item.get('width', '?')}x{item.get('height', '?')}",
            fill=(0, 0, 0),
            font=font,
        )
        draw_wrapped(
            draw,
            (x + 8, y + 34),
            item["path"],
            font=small,
            fill=(0, 0, 0),
            max_chars=54,
            max_lines=2,
            line_height=17,
        )
        if "error" in item:
            draw_wrapped(
                draw,
                (x + 8, y + 82),
                item["error"],
                font=small,
                fill=(180, 0, 0),
                max_chars=54,
                max_lines=3,
                line_height=17,
            )
            continue
        if "thumb" in item:
            thumb = Image.open(output_root / item["thumb"]).convert("RGB")
            sheet.paste(
                thumb,
                (x + (cell_w - thumb.width) // 2, y + 96 + (cell_h - 132 - thumb.height) // 2),
            )
        draw.text(
            (x + 8, y + cell_h - 24),
            f"layers {item.get('layers')} leaf {item.get('leaf')} {item['size_mb']}MB",
            fill=(0, 0, 0),
            font=small,
        )

    sheet.save(output_root / "psd_contact_sheet.jpg", quality=90)
    return summary


def coco_overlay_items(coco_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        ann_path = coco_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[int(ann["image_id"])].append(ann)
        categories = {int(c["id"]): c["name"] for c in data["categories"]}

        for image in data["images"]:
            image_path = coco_root / split / image["file_name"]
            rgb = Image.open(image_path).convert("RGB")
            overlay = np.asarray(rgb).astype(np.float32) / 255.0
            anns = anns_by_image.get(int(image["id"]), [])
            for index, ann in enumerate(anns):
                rle = ann["segmentation"]
                mask = mask_utils.decode(
                    {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
                ).astype(bool)
                color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
                overlay[mask] = overlay[mask] * 0.45 + color * 0.55

            image_overlay = Image.fromarray(
                np.clip(overlay * 255, 0, 255).astype(np.uint8)
            )
            image_overlay.thumbnail((360, 420), Image.Resampling.LANCZOS)
            labels = sorted({categories[int(ann["category_id"])] for ann in anns})
            items.append(
                {
                    "split": split,
                    "source": image.get("source_psd", image["file_name"]),
                    "annotations": len(anns),
                    "labels": labels,
                    "image": image_overlay,
                }
            )
    return items


def build_overlay_sheet(
    coco_root: Path,
    output_root: Path,
    *,
    columns: int,
) -> None:
    items = coco_overlay_items(coco_root)
    if not items:
        return

    cell_w, cell_h = 440, 560
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(17)
    small = load_font(12)

    for index, item in enumerate(items, start=1):
        x = ((index - 1) % columns) * cell_w
        y = ((index - 1) // columns) * cell_h
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(180, 180, 180))
        draw.text(
            (x + 8, y + 8),
            f"{index:02d} {item['split']} {item['annotations']} masks",
            fill=(0, 0, 0),
            font=font,
        )
        draw_wrapped(
            draw,
            (x + 8, y + 34),
            item["source"],
            font=small,
            fill=(0, 0, 0),
            max_chars=54,
            max_lines=2,
            line_height=17,
        )
        image = item["image"]
        sheet.paste(
            image,
            (x + (cell_w - image.width) // 2, y + 92 + (cell_h - 142 - image.height) // 2),
        )
        draw_wrapped(
            draw,
            (x + 8, y + cell_h - 42),
            ", ".join(item["labels"]),
            font=small,
            fill=(0, 0, 0),
            max_chars=68,
            max_lines=2,
            line_height=17,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    sheet.save(output_root / "converted_overlay_sheet.jpg", quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate readable Live2D PSD and COCO audit contact sheets."
    )
    parser.add_argument("--psd-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_psd_contact_sheet(args.psd_root, args.output_root, columns=args.columns)
    if args.coco_root is not None:
        build_overlay_sheet(args.coco_root, args.output_root, columns=args.columns)
    print(f"audit output: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
