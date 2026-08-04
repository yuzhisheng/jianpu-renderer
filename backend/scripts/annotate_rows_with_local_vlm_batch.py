#!/usr/bin/env python3
"""Batched/resumable variant of local VLM pitch-skeleton annotation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from mlx_vlm import batch_generate, generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from annotate_rows_with_local_vlm import DEFAULT_MODEL, PROMPT, parse_json, save


ROOT = Path(__file__).resolve().parents[2]


def fixed_canvas(path: Path, width: int = 1600, height: int = 640):
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def repair(payload: dict):
    for old in payload["rows"]:
        if "error" not in old or not old.get("raw_response"):
            continue
        try:
            parsed = parse_json(old["raw_response"])
        except Exception:
            continue
        old.pop("error", None)
        old.update(parsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="backend/real_annotations")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--only", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    root = (ROOT / args.annotations).resolve()
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    pages = [item for item in manifest
             if item["split"] == args.split and item["kind"] == "jianpu"]
    if args.only:
        pages = [item for item in pages if item["annotation_id"] == args.only]
    output = root / "local_vlm_pitch_reviews"
    output.mkdir(parents=True, exist_ok=True)

    states = {}
    tasks = []
    for page in pages:
        annotation_id = page["annotation_id"]
        target = output / f"{annotation_id}.json"
        if target.exists():
            payload = json.loads(target.read_text())
        else:
            payload = {
                "annotation_id": annotation_id,
                "model": args.model,
                "label_grade": "silver_local_vlm_pitch_skeleton",
                "status": "in_progress",
                "rows": [],
            }
        repair(payload)
        save(target, payload)
        states[annotation_id] = (target, payload)
        completed = {row["source_row"] for row in payload["rows"] if "error" not in row}
        review = json.loads((root / page["review"]).read_text())
        for row in review["rows"]:
            if row["row"] not in completed:
                tasks.append((annotation_id, row, root / row["image"]))

    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        print("nothing to annotate")
        return

    print(f"loading {args.model}; pending={len(tasks)}, batch={args.batch_size}", flush=True)
    model, processor = load(args.model)
    config = load_config(args.model)
    prompt = apply_chat_template(
        processor, config, PROMPT, num_images=1, thinking_mode="disabled")

    for start in range(0, len(tasks), args.batch_size):
        chunk = tasks[start:start + args.batch_size]
        images = [fixed_canvas(item[2]) for item in chunk]
        started = time.perf_counter()
        try:
            response = batch_generate(
                model, processor, images=images, prompts=[prompt] * len(images),
                max_tokens=args.max_tokens, temperature=0.0, verbose=False,
                group_by_shape=True,
            )
            texts = response.texts
        except Exception as batch_error:
            print(f"batch fallback: {batch_error}", flush=True)
            texts = []
            for image in images:
                try:
                    texts.append(generate(
                        model, processor, prompt, image=image,
                        max_tokens=args.max_tokens, temperature=0.0,
                        verbose=False,
                    ).text)
                except Exception as exc:
                    texts.append(json.dumps({"_generation_error": str(exc)}))
        elapsed = time.perf_counter() - started

        for (annotation_id, row, _), text in zip(chunk, texts):
            target, payload = states[annotation_id]
            try:
                parsed = parse_json(text)
                record = {
                    "source_row": row["row"], "image": row["image"], **parsed,
                    "batch_seconds_per_row": round(elapsed / len(chunk), 2),
                    "raw_response": text,
                }
                label = parsed["content_type"]
            except Exception as exc:
                record = {
                    "source_row": row["row"], "image": row["image"],
                    "error": str(exc),
                    "batch_seconds_per_row": round(elapsed / len(chunk), 2),
                    "raw_response": text,
                }
                label = "error"
            payload["rows"] = [item for item in payload["rows"]
                               if item["source_row"] != row["row"]]
            payload["rows"].append(record)
            payload["rows"].sort(key=lambda item: item["source_row"])
            save(target, payload)
            print(f"[{start + 1:03d}/{len(tasks)}] {annotation_id} "
                  f"row={row['row']} {label}", flush=True)

    for annotation_id, (target, payload) in states.items():
        page = next(item for item in pages if item["annotation_id"] == annotation_id)
        review = json.loads((root / page["review"]).read_text())
        completed = {row["source_row"] for row in payload["rows"] if "error" not in row}
        if all(row["row"] in completed for row in review["rows"]):
            payload["status"] = "complete"
            save(target, payload)
    print(f"finished rows={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
