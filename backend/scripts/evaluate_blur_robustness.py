#!/usr/bin/env python3
"""Measure full-pipeline recognition accuracy under controlled Gaussian blur."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from assembler import Assembler
from detector import YoloDetector
from scripts.evaluate_recognizer import (
    flatten_notes, full_key, measure_keys, pitch_key, similarity,
)


def parse_sigmas(value: str) -> list[float]:
    sigmas = [float(item) for item in value.split(",")]
    if not sigmas or any(item < 0 for item in sigmas):
        raise argparse.ArgumentTypeError("sigmas 必须是逗号分隔的非负数")
    return sigmas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="backend/weights/best.pt")
    parser.add_argument("--manifest", default="public/training/val.txt")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sigmas", type=parse_sigmas, default=parse_sigmas("0,0.5,1.0,1.5,2.0"))
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--preview-dir", default="backend/eval_outputs/blur_robustness")
    args = parser.parse_args()

    manifest = (ROOT / args.manifest).resolve()
    images = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    selected = rng.sample(images, min(args.samples, len(images)))
    detector = YoloDetector(str((ROOT / args.weights).resolve()), device=args.device)
    assembler = Assembler(use_transformer=False)
    preview_dir = (ROOT / args.preview_dir).resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"random holdout pages: {len(selected)}, seed={args.seed}")
    print("sigma\tpitch\tnote\tmeasure\tpage_exact\tcount_similarity")
    for sigma in args.sigmas:
        pitch_scores, note_scores, measure_scores = [], [], []
        page_exact = gt_total = pred_total = 0
        for index, image_path in enumerate(selected):
            ground_truth = json.loads(image_path.with_suffix(".json").read_text())
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                blurred = source if sigma == 0 else source.filter(ImageFilter.GaussianBlur(radius=sigma))
                if index == 0:
                    blurred.save(preview_dir / f"sample_sigma_{sigma:.1f}.png", optimize=True)
                detections, width, height = detector.detect(
                    blurred, conf_threshold=args.conf, imgsz=args.imgsz,
                )
            predicted = assembler.assemble_from_dets(detections, width, height)["score"]
            gt_notes, pred_notes = flatten_notes(ground_truth), flatten_notes(predicted)
            gt_pitch = [pitch_key(note) for note in gt_notes]
            pred_pitch = [pitch_key(note) for note in pred_notes]
            gt_full = [full_key(note) for note in gt_notes]
            pred_full = [full_key(note) for note in pred_notes]
            gt_measures, pred_measures = measure_keys(ground_truth), measure_keys(predicted)
            pitch_scores.append(similarity(gt_pitch, pred_pitch))
            note_scores.append(similarity(gt_full, pred_full))
            measure_scores.append(similarity(gt_measures, pred_measures))
            page_exact += gt_measures == pred_measures
            gt_total += len(gt_notes)
            pred_total += len(pred_notes)

        count = max(len(selected), 1)
        print(
            f"{sigma:.1f}\t{sum(pitch_scores) / count:.4f}\t"
            f"{sum(note_scores) / count:.4f}\t{sum(measure_scores) / count:.4f}\t"
            f"{page_exact / count:.4f}\t"
            f"{1.0 - abs(pred_total - gt_total) / max(pred_total, gt_total, 1):.4f}"
        )
    print(f"previews: {preview_dir}")


if __name__ == "__main__":
    main()
