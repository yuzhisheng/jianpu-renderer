#!/usr/bin/env python3
"""Evaluate the complete image -> Score JSON recognizer.

Detector mAP does not measure whether octave dots and duration lines were
attached to the correct note.  This script reports transcription metrics on a
held-out image manifest and can be used on synthetic or manually-labelled real
test sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from assembler import Assembler
from detector import YoloDetector


def flatten_notes(score: dict) -> List[dict]:
    return [note for measure in score.get("measures", []) for note in measure.get("notes", [])]


def pitch_key(note: dict) -> Tuple[Any, ...]:
    if note.get("type") == "dash":
        return ("dash",)
    return (note.get("pitch"), note.get("octave", 0))


def full_key(note: dict) -> Tuple[Any, ...]:
    if note.get("type") == "dash":
        return ("dash", round(float(note.get("duration", 0.5)), 3))
    return (
        note.get("pitch"), note.get("octave", 0),
        round(float(note.get("duration", 1.0)), 3),
        note.get("accidental"), note.get("dot", 0),
    )


def edit_distance(left: Sequence, right: Sequence) -> int:
    previous = list(range(len(right) + 1))
    for i, lhs in enumerate(left, 1):
        current = [i]
        for j, rhs in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (lhs != rhs),
            ))
        previous = current
    return previous[-1]


def similarity(left: Sequence, right: Sequence) -> float:
    denominator = max(len(left), len(right), 1)
    return 1.0 - edit_distance(left, right) / denominator


def measure_keys(score: dict) -> List[Tuple]:
    result = []
    for measure in score.get("measures", []):
        notes = tuple(full_key(note) for note in measure.get("notes", []))
        # The renderer treats an omitted barline as a visible single barline;
        # normalize that equivalent representation before scoring.
        result.append((notes, measure.get("barline", "single")))
    return result


def load_manifest(path: Path, limit: int) -> List[Path]:
    images = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return images[:limit] if limit else images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="backend/weights/best.pt")
    parser.add_argument("--manifest", default="public/training/val.txt")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full manifest")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true",
                        help="只输出汇总指标，适合阈值回归")
    parser.add_argument("--ground-truth-boxes", action="store_true",
                        help="使用 YOLO 真值框，测量拼装器/标注的理论上限")
    args = parser.parse_args()

    manifest = (ROOT / args.manifest).resolve()
    weights = (ROOT / args.weights).resolve()
    images = load_manifest(manifest, args.limit)
    detector = None if args.ground_truth_boxes else YoloDetector(str(weights), device=args.device)
    assembler = Assembler(use_transformer=False)

    pitch_scores, note_scores, measure_scores = [], [], []
    page_exact = 0
    gt_total = pred_total = 0
    failures = []

    for index, image_path in enumerate(images, 1):
        ground_truth = json.loads(image_path.with_suffix(".json").read_text())
        with Image.open(image_path) as image:
            width, height = image.size
            if detector is not None:
                detections, width, height = detector.detect(
                    image, conf_threshold=args.conf, imgsz=args.imgsz,
                )
            else:
                detections = []
                for line in image_path.with_suffix(".txt").read_text().splitlines():
                    if not line.strip():
                        continue
                    class_id, cx, cy, box_width, box_height = map(float, line.split()[:5])
                    detections.append((
                        int(class_id), cx * width, cy * height,
                        box_width * width, box_height * height, 1.0,
                    ))
        predicted = assembler.assemble_from_dets(detections, width, height)["score"]

        gt_notes, pred_notes = flatten_notes(ground_truth), flatten_notes(predicted)
        gt_pitch = [pitch_key(n) for n in gt_notes]
        pred_pitch = [pitch_key(n) for n in pred_notes]
        gt_full = [full_key(n) for n in gt_notes]
        pred_full = [full_key(n) for n in pred_notes]
        gt_measures, pred_measures = measure_keys(ground_truth), measure_keys(predicted)

        pitch_scores.append(similarity(gt_pitch, pred_pitch))
        note_scores.append(similarity(gt_full, pred_full))
        measure_scores.append(similarity(gt_measures, pred_measures))
        page_exact += gt_measures == pred_measures
        gt_total += len(gt_notes)
        pred_total += len(pred_notes)
        if gt_measures != pred_measures:
            failures.append((note_scores[-1], image_path.name))
        if not args.quiet:
            print(f"[{index:>4}/{len(images)}] {image_path.name}: "
                  f"pitch={pitch_scores[-1]:.3f} note={note_scores[-1]:.3f} "
                  f"measure={measure_scores[-1]:.3f}")

    count = max(len(images), 1)
    print("\n=== Full-pipeline transcription metrics ===")
    print(f"pages:                 {len(images)}")
    print(f"pitch edit similarity: {sum(pitch_scores) / count:.4f}")
    print(f"note edit similarity:  {sum(note_scores) / count:.4f}")
    print(f"measure similarity:    {sum(measure_scores) / count:.4f}")
    print(f"page exact match:      {page_exact / count:.4f}")
    count_similarity = 1.0 - abs(pred_total - gt_total) / max(pred_total, gt_total, 1)
    print(f"note count similarity: {count_similarity:.4f} ({pred_total}/{gt_total})")
    if failures:
        print("worst pages:            " + ", ".join(name for _, name in sorted(failures)[:10]))


if __name__ == "__main__":
    main()
