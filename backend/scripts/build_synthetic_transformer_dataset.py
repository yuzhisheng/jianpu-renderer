#!/usr/bin/env python3
"""Convert renderer-generated pages/boxes into exact row-sequence labels."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from model.jianpu_event_vocabulary import (
    events_to_ctc_tokens, events_to_targets, tokens_to_events,
)

CLASS_TOKEN = {
    0: "P1", 1: "P2", 2: "P3", 3: "P4", 4: "P5", 5: "P6", 6: "P7",
    7: "R0", 8: "-", 32: "B|", 33: "B||", 34: "B|]", 35: "B|:", 36: "B:|",
}


def cluster_rows(boxes):
    anchors = [box for box in boxes if box[0] <= 7]
    rows = []
    for box in sorted(anchors, key=lambda item: item[2]):
        tolerance = max(10.0, box[4] * 0.9)
        target = next((row for row in rows
                       if abs(box[2] - sum(item[2] for item in row) / len(row)) <= tolerance), None)
        if target is None:
            rows.append([box])
        else:
            target.append(box)
    return sorted(rows, key=lambda row: sum(item[2] for item in row) / len(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="public/training")
    parser.add_argument("--output", default="backend/jianpu_synthetic_transformer_dataset")
    parser.add_argument("--limit-pages", type=int, default=0)
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    output = (ROOT / args.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
    records = {"train": [], "val": []}
    pages = sorted(source.glob("score_*.png"))
    if args.limit_pages:
        pages = pages[:args.limit_pages]
    for page_index, image_path in enumerate(pages, 1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        boxes = []
        for line in image_path.with_suffix(".txt").read_text().splitlines():
            class_id, cx, cy, box_width, box_height = map(float, line.split())
            class_id = int(class_id)
            if class_id in CLASS_TOKEN:
                boxes.append((class_id, cx * width, cy * height, box_width * width, box_height * height))
        row_groups = cluster_rows(boxes)
        centers = [sum(item[2] for item in row) / len(row) for row in row_groups]
        split = "val" if (source / "images" / "val" / image_path.name).exists() else "train"
        for row_index, (row, center) in enumerate(zip(row_groups, centers), 1):
            upper_mid = 0 if row_index == 1 else int((centers[row_index - 2] + center) / 2)
            lower_mid = height if row_index == len(centers) else int((center + centers[row_index]) / 2)
            anchor_height = max(item[4] for item in row)
            top = max(upper_mid, int(center - anchor_height * 3.2))
            bottom = min(lower_mid, int(center + anchor_height * 4.8))
            if bottom - top < 16:
                continue
            selected = [box for box in boxes if top <= box[2] <= bottom]
            tokens = [CLASS_TOKEN[box[0]] for box in sorted(selected, key=lambda item: item[1])]
            events = tokens_to_events(tokens)
            if len(events) < 3:
                continue
            name = f"{image_path.stem}_row_{row_index:02d}.png"
            target_path = output / "images" / split / name
            image.crop((0, top, width, bottom)).save(target_path, optimize=True)
            records[split].append({
                "id": f"synthetic:{image_path.stem}:row_{row_index:02d}",
                "image": os.path.relpath(target_path, output),
                "label_grade": "gold_renderer",
                "tokens": tokens,
                "ctc_tokens": events_to_ctc_tokens(events),
                "targets": events_to_targets(events, skeleton_label=True),
            })
        if page_index % 250 == 0:
            print(f"pages={page_index} rows={sum(map(len, records.values()))}")
    for split, items in records.items():
        (output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n"
        )
    stats = {
        "pages": len(pages), "splits": {key: len(value) for key, value in records.items()},
        "events": {key: sum(len(item["ctc_tokens"]) for item in value)
                   for key, value in records.items()},
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
