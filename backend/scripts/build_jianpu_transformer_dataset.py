#!/usr/bin/env python3
"""Build traceable row-image → multi-branch event manifests."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from model.jianpu_event_vocabulary import (
    events_to_ctc_tokens, events_to_targets, tokens_to_events,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="backend/real_annotations")
    parser.add_argument("--output", default="backend/jianpu_transformer_dataset")
    parser.add_argument("--min-confidence", type=float, default=0.9)
    args = parser.parse_args()
    annotation_root = (ROOT / args.annotations).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    pages = [json.loads(line) for line in (annotation_root / "manifest.jsonl").read_text().splitlines()]
    train_pages = sorted(
        page["annotation_id"] for page in pages
        if page["split"] == "train_silver" and page["kind"] == "jianpu"
    )
    validation_pages = set(train_pages[::5])
    records = {"train": [], "val": [], "test": []}
    rejected = Counter()

    for page in pages:
        if page["kind"] != "jianpu" or page["split"] == "excluded":
            continue
        annotation_id = page["annotation_id"]
        label_path = annotation_root / "local_vlm_pitch_reviews" / f"{annotation_id}.json"
        if not label_path.exists():
            rejected["missing_label"] += 1
            continue
        label = json.loads(label_path.read_text())
        split = "test" if page["split"] == "test" else (
            "val" if annotation_id in validation_pages else "train"
        )
        for row in label.get("rows", []):
            voices = row.get("voices") or []
            tokens = voices[0].get("tokens", []) if len(voices) == 1 else []
            reason = None
            if row.get("content_type") != "score":
                reason = "not_score"
            elif len(voices) != 1:
                reason = "not_single_voice"
            elif float(row.get("confidence", 0)) < args.min_confidence:
                reason = "low_confidence"
            elif row.get("uncertainties") or "?" in tokens:
                reason = "uncertain"
            events = tokens_to_events(tokens)
            if not reason and len(events) < 3:
                reason = "too_short"
            image_path = annotation_root / row.get("image", "")
            if not reason and not image_path.exists():
                reason = "missing_image"
            if reason:
                rejected[reason] += 1
                continue
            records[split].append({
                "id": f"{annotation_id}:row_{int(row['source_row']):02d}",
                "annotation_id": annotation_id,
                "source_row": row["source_row"],
                "image": os.path.relpath(image_path, output),
                "label_grade": "silver_local_vlm_pitch_skeleton",
                "tokens": tokens,
                "ctc_tokens": events_to_ctc_tokens(events),
                "targets": events_to_targets(events, skeleton_label=True),
            })

    for split, items in records.items():
        (output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in items)
            + ("\n" if items else "")
        )
    stats = {
        "splits": {name: len(items) for name, items in records.items()},
        "events": {name: sum(len(item["targets"]["kind"]) - 2 for item in items)
                   for name, items in records.items()},
        "rejected": dict(rejected),
        "validation_policy": "every fifth sorted train_silver page",
        "license_note": "HOMR source is reference-only; these manifests contain project-owned labels.",
    }
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
