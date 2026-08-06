#!/usr/bin/env python3
"""Run unlabeled real images and report recognition health indicators."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from assembler import Assembler
from detector import CLASS_NAMES, YoloDetector


EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def has_image_signature(path: Path) -> bool:
    """Reject HTML/error pages before Ultralytics can auto-install codecs."""
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    return (
        header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith(b"\xff\xd8\xff")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
    )


def count_rows(detections) -> int:
    anchors = sorted((d for d in detections if 0 <= d[0] <= 8), key=lambda d: d[2])
    if not anchors:
        return 0
    tolerance = max(10.0, min(28.0, float(np.median([d[4] for d in anchors])) * 1.35))
    centers = []
    for detection in anchors:
        if not centers or abs(detection[2] - float(np.median(centers[-1]))) > tolerance:
            centers.append([detection[2]])
        else:
            centers[-1].append(detection[2])
    return len(centers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--weights", default="backend/weights/best.pt")
    parser.add_argument("--output", default="backend/eval_outputs/real_image_diagnostics.csv")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()

    files = []
    for value in args.inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(item for item in path.iterdir() if item.suffix.lower() in EXTENSIONS)
        elif path.suffix.lower() in EXTENSIONS:
            files.append(path)
    files = sorted(set(item.resolve() for item in files))
    detector = YoloDetector(str((ROOT / args.weights).resolve()), device=args.device)
    assembler = Assembler(use_transformer=False)
    records = []
    skipped = []

    for index, path in enumerate(files, 1):
        started = time.perf_counter()
        if not has_image_signature(path):
            error = "file signature is not PNG/JPEG/GIF/WEBP"
            skipped.append({"file": str(path), "error": error})
            print(f"[{index:02d}/{len(files)}] {path.name}: skipped ({error})", flush=True)
            continue
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            detections, width, height = detector.detect(
                image, conf_threshold=args.conf, imgsz=args.imgsz,
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped.append({"file": str(path), "error": str(exc)})
            print(f"[{index:02d}/{len(files)}] {path.name}: skipped ({exc})", flush=True)
            continue
        except Exception as exc:
            skipped.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index:02d}/{len(files)}] {path.name}: error ({exc})", flush=True)
            continue
        score = assembler.assemble_from_dets(detections, width, height)["score"]
        counts = Counter(CLASS_NAMES[d[0]] for d in detections)
        pitches = [counts[f"pitch_{pitch}"] for pitch in range(1, 8)]
        notes = sum(pitches) + counts["rest"]
        bars = sum(counts[name] for name in (
            "bar_single", "bar_double", "bar_end", "bar_repeat_start", "bar_repeat_end",
        ))
        ties = counts["tie"] + counts["slur"]
        diversity = sum(value > 0 for value in pitches)
        dominance = max(pitches, default=0) / max(notes, 1)
        flags = []
        if notes < 5:
            flags.append("few_notes")
        if notes >= 15 and diversity < 3:
            flags.append("low_pitch_diversity")
        if notes >= 15 and dominance > 0.60:
            flags.append("dominant_pitch")
        if notes >= 10 and bars == 0:
            flags.append("no_barlines")
        if ties > notes * 0.6:
            flags.append("excess_curves")
        record = {
            "file": str(path), "width": width, "height": height,
            "detections": len(detections), "rows": count_rows(detections),
            "notes": notes, "measures": len(score.get("measures", [])),
            "bars": bars, "underlines": counts["underline_1"] + counts["underline_2"],
            "curves": ties, "pitch_diversity": diversity,
            "dominant_pitch_fraction": round(dominance, 3),
            "inference_ms": round((time.perf_counter() - started) * 1000),
            "adaptive_retry": detector.last_retry_used,
            "effective_conf": detector.last_effective_confidence,
            "flags": ",".join(flags),
            "pitch_counts": json.dumps(pitches),
        }
        records.append(record)
        retry_label = f", retry={record['effective_conf']:.2f}" if record["adaptive_retry"] else ""
        print(f"[{index:02d}/{len(files)}] {path.name}: notes={notes}, rows={record['rows']}, "
              f"bars={bars}{retry_label}, flags={record['flags'] or '-'}", flush=True)

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["file"])
        writer.writeheader()
        writer.writerows(records)
    flagged = sum(bool(record["flags"]) for record in records)
    print(f"images={len(records)}, flagged={flagged}, skipped={len(skipped)}, report={output}")
    for item in skipped:
        print(f"skipped: {item['file']}: {item['error']}")


if __name__ == "__main__":
    main()
