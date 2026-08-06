#!/usr/bin/env python3
"""Create a traceable real-image annotation workspace.

The detector supplies pixel-accurate box proposals. A vision-language reviewer
then corrects the row token sequences in reviews/*.json. These files are silver
labels until a second independent review agrees with the first one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from assembler import Assembler
from detector import CLASS_NAMES, YoloDetector
from model.spatial_tokens import detections_to_tokens


EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
TEST_NAS = {
    "gusu-xing_p1.jpg",
    "hongdou_jp.png",
    "tianbian_jp.png",
    "youmu-shiguang.png",
    "zhs_p2.png",
}


def weight_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def score_rows(detections):
    anchors = [item for item in detections if 0 <= item[0] <= 8]
    if not anchors:
        return []
    tolerance = max(10.0, min(28.0,
                    float(np.median([item[4] for item in anchors])) * 1.35))
    rows = []
    for anchor in sorted(anchors, key=lambda item: item[2]):
        if not rows or abs(anchor[2] - float(np.median(
                [item[2] for item in rows[-1]]))) > tolerance:
            rows.append([anchor])
        else:
            rows[-1].append(anchor)
    return [(
        float(np.median([item[2] for item in row])),
        tolerance,
        float(np.median([item[4] for item in row])),
    ) for row in rows]


def row_detections(detections, center: float, tolerance: float):
    selected = []
    for item in detections:
        class_id, _, cy, _, height, _ = item
        if 0 <= class_id <= 8:
            matches = abs(cy - center) <= tolerance
        elif class_id in {32, 33, 34, 35, 36}:
            matches = abs(cy - center) <= height / 2 + tolerance
        elif class_id in {9, 10}:
            matches = -8.0 <= cy - center <= 38.0
        else:
            matches = abs(cy - center) <= max(48.0, height / 2 + tolerance)
        if matches:
            selected.append(item)
    return selected


def layout_bands(image: Image.Image):
    """Find all horizontal content bands without trusting detector classes."""
    width, height = image.size
    normalized_width = 1600
    normalized_height = max(1, round(height * normalized_width / width))
    gray = np.asarray(image.convert("L").resize(
        (normalized_width, normalized_height), Image.Resampling.BILINEAR))
    active = np.flatnonzero((gray < 160).mean(axis=1) > 0.008)
    groups: List[List[int]] = []
    for row in active.tolist():
        if groups and row <= groups[-1][-1] + 25:
            groups[-1].append(row)
        else:
            groups.append([row])
    scale = height / normalized_height
    result = []
    for group in groups:
        if group[-1] - group[0] < 2:
            continue
        top = max(0, int((group[0] - 12) * scale))
        bottom = min(height, int((group[-1] + 13) * scale))
        if bottom - top >= 8:
            result.append((top, bottom))
    return result


def save_row_crops(image: Image.Image, detections, output: Path, annotation_id: str):
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    reviews = []
    width, height = image.size
    bands = layout_bands(image)
    detector_specs = score_rows(detections)
    if detector_specs and len(bands) <= max(3, int(len(detector_specs) * 0.6)):
        # Dense pages may have almost no whitespace between lyrics and the next
        # score row. In that case detector centers are useful only as vertical
        # navigation; crops remain full-width so missed symbols stay visible.
        bands = []
        for offset, (center, _, anchor_height) in enumerate(detector_specs):
            upper = 0 if offset == 0 else int((detector_specs[offset - 1][0] + center) / 2)
            lower = height if offset + 1 == len(detector_specs) else int(
                (center + detector_specs[offset + 1][0]) / 2)
            bands.append((
                max(upper, int(center - max(80.0, anchor_height * 3.0))),
                min(lower, int(center + max(100.0, anchor_height * 4.5))),
            ))
    elif detector_specs:
        # Projection bands occasionally join two consecutive score lines via
        # a nearby D.S./coda label or footer. Split only on large detector-row
        # gaps; closer centers may be simultaneous voices and must stay in one
        # crop for the VLM to preserve voice order.
        refined = []
        for top, bottom in bands:
            specs = sorted(
                (center, anchor_height)
                for center, _, anchor_height in detector_specs
                if top <= center <= bottom
            )
            centers = [center for center, _ in specs]
            cuts = [int((left + right) / 2)
                    for left, right in zip(centers, centers[1:])
                    if right - left >= 60.0]
            edges = [top, *cuts, bottom]
            for left, right in zip(edges, edges[1:]):
                contained = [(center, height) for center, height in specs
                             if left <= center <= right]
                if contained:
                    pad_up = max(80.0, max(height * 3.0 for _, height in contained))
                    pad_down = max(100.0, max(height * 4.5 for _, height in contained))
                    left = max(left, int(min(center for center, _ in contained) - pad_up))
                    right = min(right, int(max(center for center, _ in contained) + pad_down))
                if right - left >= 8:
                    refined.append((left, right))
        bands = refined
    for row_offset, (top, bottom) in enumerate(bands):
        index = row_offset + 1
        items = [item for item in detections
                 if top - item[4] / 2 <= item[2] <= bottom + item[4] / 2]
        left, right = 0, width
        filename = f"row_{index:02d}.png"
        image.crop((left, top, right, bottom)).save(output / filename, optimize=True)
        tokens = [token for token in detections_to_tokens(items)
                  if token not in {"<BOS>", "<EOS>", "<ROW>"}]
        reviews.append({
            "row": index,
            "image": f"rows/{annotation_id}/{filename}",
            "crop_box": [left, top, right, bottom],
            "content_type": None,
            "detector_tokens": tokens,
            "vlm_tokens": None,
            "review_confidence": None,
            "uncertainties": [],
        })
    return reviews


def save_overlay(image: Image.Image, detections, output: Path):
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for class_id, cx, cy, width, height, confidence in detections:
        x1, y1 = cx - width / 2, cy - height / 2
        x2, y2 = cx + width / 2, cy + height / 2
        color = "#00a060" if confidence >= 0.5 else "#e09000"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), CLASS_NAMES[class_id], fill=color, font=font)
    preview.thumbnail((1600, 2200), Image.Resampling.LANCZOS)
    preview.save(output, quality=88)


def yolo_lines(detections, width: int, height: int):
    return [
        f"{class_id} {cx / width:.8f} {cy / height:.8f} "
        f"{box_width / width:.8f} {box_height / height:.8f}"
        for class_id, cx, cy, box_width, box_height, _ in detections
    ]


def source_files(path: Path):
    def has_image_signature(item: Path) -> bool:
        try:
            with item.open("rb") as handle:
                header = handle.read(16)
        except OSError:
            return False
        return (
            header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"\xff\xd8\xff")
            or header.startswith((b"GIF87a", b"GIF89a"))
            or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
        )

    return sorted(
        item for item in path.iterdir()
        if item.suffix.lower() in EXTENSIONS and has_image_signature(item)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nas", default="/Volumes/jianpu/images")
    parser.add_argument("--web", default="backend/real_data/qupu123_samples")
    parser.add_argument("--output", default="backend/real_annotations")
    parser.add_argument("--weights", default="backend/weights/best.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    for directory in ("images", "labels", "predictions", "reviews", "rows", "overlays"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    weights = (ROOT / args.weights).resolve()
    detector = YoloDetector(str(weights), device=args.device)
    assembler = Assembler(use_transformer=False)
    inputs = (
        [("nas", path) for path in source_files(Path(args.nas))] +
        [("web", path) for path in source_files((ROOT / args.web).resolve())]
    )
    manifest = []

    for index, (source, path) in enumerate(inputs, 1):
        annotation_id = f"{source}_{path.stem}"
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        detections, width, height = detector.detect(image, imgsz=args.imgsz)
        staff_groups = detector._staff_line_group_count(np.asarray(image.convert("L")))
        kind = "staff" if staff_groups >= 2 else "jianpu"
        split = "excluded" if kind != "jianpu" else (
            "test" if source == "web" or path.name in TEST_NAS else "train_silver"
        )

        image_target = output / "images" / f"{annotation_id}{path.suffix.lower()}"
        # SMB files can carry macOS flags that cannot be restored on the local
        # destination. Annotation only needs the image bytes, not NAS metadata.
        shutil.copyfile(path, image_target)
        score = assembler.assemble_from_dets(detections, width, height)["score"]
        (output / "predictions" / f"{annotation_id}.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=2) + "\n")
        lines = yolo_lines(detections, width, height)
        (output / "labels" / f"{annotation_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        rows = save_row_crops(
            image, detections, output / "rows" / annotation_id, annotation_id)
        review = {
            "annotation_id": annotation_id,
            "status": "excluded_non_jianpu" if kind != "jianpu" else "pending_vlm_review",
            "label_grade": "silver_detector",
            "source_image": str(path.resolve()),
            "split": split,
            "rows": rows,
        }
        (output / "reviews" / f"{annotation_id}.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n")
        save_overlay(image, detections, output / "overlays" / f"{annotation_id}.jpg")
        record = {
            "annotation_id": annotation_id,
            "source": source,
            "source_image": str(path.resolve()),
            "image": str(image_target.relative_to(output)),
            "split": split,
            "kind": kind,
            "staff_groups": staff_groups,
            "detections": len(detections),
            "rows": len(rows),
            "adaptive_retry": detector.last_retry_used,
            "effective_conf": detector.last_effective_confidence,
            "label_grade": "silver_detector",
            "review": f"reviews/{annotation_id}.json",
        }
        manifest.append(record)
        print(f"[{index:02d}/{len(inputs)}] {annotation_id}: {kind}, {split}, "
              f"rows={len(rows)}, detections={len(detections)}")

    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n")
    metadata = {
        "label_policy": "silver labels: detector boxes plus pending independent VLM row review",
        "weights": str(weights),
        "weights_sha256": weight_digest(weights),
        "imgsz": args.imgsz,
        "total": len(manifest),
        "test": sum(item["split"] == "test" for item in manifest),
        "train_silver": sum(item["split"] == "train_silver" for item in manifest),
        "excluded": sum(item["split"] == "excluded" for item in manifest),
    }
    (output / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
