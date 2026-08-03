#!/usr/bin/env python3
"""Refresh row crops from cached YOLO labels without rerunning detection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from scripts.bootstrap_real_annotations import save_row_crops


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="backend/real_annotations")
    parser.add_argument("--only", help="逗号分隔的 annotation_id")
    parser.add_argument("--split", help="刷新清单中的指定 split")
    args = parser.parse_args()

    root = (ROOT / args.annotations).resolve()
    manifest_path = root / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if not args.only and not args.split:
        parser.error("至少提供 --only 或 --split")
    selected = ({item.strip() for item in args.only.split(",") if item.strip()}
                if args.only else set())
    if args.split:
        selected.update(item["annotation_id"] for item in manifest
                        if item["split"] == args.split and item["kind"] == "jianpu")
    found = set()

    for item in manifest:
        annotation_id = item["annotation_id"]
        if annotation_id not in selected:
            continue
        found.add(annotation_id)
        with Image.open(root / item["image"]) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        detections = []
        for line in (root / "labels" / f"{annotation_id}.txt").read_text().splitlines():
            class_id, cx, cy, box_width, box_height = map(float, line.split())
            detections.append((
                int(class_id), cx * width, cy * height,
                box_width * width, box_height * height, 1.0,
            ))
        review_path = root / item["review"]
        review = json.loads(review_path.read_text())
        review["rows"] = save_row_crops(
            image, detections, root / "rows" / annotation_id, annotation_id)
        review["row_crop_policy"] = "hybrid_projection_detector_v2"
        atomic_json(review_path, review)
        item["rows"] = len(review["rows"])
        print(f"{annotation_id}: rows={item['rows']}")

    missing = selected - found
    if missing:
        raise SystemExit(f"unknown annotation ids: {sorted(missing)}")
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
