"""
YOLOv8 推理封装
- 加载 best.pt
- 单图推理
- 返回检测结果
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import io

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"


# 类别名, 与 generate_training_pngs.cjs 一致
CLASS_NAMES = [
    'pitch_1','pitch_2','pitch_3','pitch_4','pitch_5','pitch_6','pitch_7','rest',
    'dash','underline_1','underline_2','dot','upper_dot','lower_dot',
    'sharp','flat','natural','fermata','tenuto','accent',
    'boyin','chanyin','tie','slur','dayin','tuyin','dieyin','liyin','huayin','yinyin','dunyin',
    'dynamic','bar_single','bar_double','bar_end','bar_repeat_start','bar_repeat_end',
    'repeat_ending','crescendo','descrescendo','lyric','force_accent',
]
assert len(CLASS_NAMES) == 42

# class_id -> (cx, cy, w, h, conf, class_name)
Detection = Tuple[int, float, float, float, float, float]


class YoloDetector:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.weights_path = weights_path or str(WEIGHTS / "best.pt")
        self.device = device or self._best_device()
        self.model = None
        self.pitch_model = None
        self.pitch_refiner_path: Optional[str] = None
        self._loaded = False
        self.last_retry_used = False
        self.last_effective_confidence: Optional[float] = None

    @staticmethod
    def _best_device() -> str:
        """Prefer Apple Silicon acceleration, with a safe CPU fallback."""
        try:
            import torch
            return "mps" if torch.backends.mps.is_available() else "cpu"
        except (ImportError, AttributeError):
            return "cpu"

    def load(self):
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 未安装, 请: pip install -r backend/requirements.txt")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"YOLO 权重不存在: {self.weights_path}\n"
                "请先训练: python backend/scripts/train_detector.py"
            )
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)
        self._loaded = True

    def enable_pitch_refinement(self) -> bool:
        """Load the optional 8-class pitch refiner for accurate recognition.

        The production detector remains the 42-class model so ties, bars and
        ornaments are preserved.  The independently trained 8-class model is
        only used to replace pitch/rest boxes during accurate recognition;
        fast mode never pays the extra inference cost.
        """
        if self.pitch_model is not None:
            return True
        if os.environ.get("JIANPU_DISABLE_PITCH_REFINEMENT", "0") == "1":
            return False
        candidate = os.environ.get(
            "JIANPU_PITCH_REFINER_WEIGHTS",
            str(WEIGHTS / "pitch8_domain_mixed_v1.pt"),
        )
        if not os.path.exists(candidate):
            return False
        try:
            from ultralytics import YOLO
            self.pitch_model = YOLO(candidate)
            self.pitch_model.to(self.device)
            self.pitch_refiner_path = candidate
            return True
        except Exception:
            self.pitch_model = None
            self.pitch_refiner_path = None
            return False

    def detect(self, image: Image.Image, conf_threshold: float = 0.20,
               imgsz: int = 1280, tile_size: int = 1280,
               tile_overlap: float = 0.18) -> Tuple[List[Detection], int, int]:
        """
        检测图片中的符号
        返回: [(class_id, cx, cy, w, h, conf), ...]
        坐标为像素值 (cx, cy, w, h)
        """
        if not self._loaded:
            self.load()

        # Normalize phone rotation, colored paper and uneven overall contrast.
        image = ImageOps.exif_transpose(image)
        image = ImageOps.autocontrast(image.convert("L"), cutoff=0.5).convert("RGB")
        img = np.array(image)
        img_h, img_w = img.shape[:2]

        # This service intentionally handles numbered notation only. Reject a
        # clear Western staff page before digit detection; otherwise noteheads
        # and clefs can leak through as a handful of bogus numbered notes.
        self.last_retry_used = False
        self.last_effective_confidence = conf_threshold
        if self._staff_line_group_count(img[:, :, 0]) >= 2:
            return [], img_w, img_h

        # A whole phone photo shrunk to one detector input makes octave dots and
        # duration lines only a few pixels high. Run overlapping native-resolution
        # tiles for large pages, then merge duplicates in page coordinates.
        if max(img_w, img_h) > tile_size * 1.25:
            windows = self._tile_windows(img_w, img_h, tile_size, tile_overlap)
        else:
            windows = [(0, 0, img_w, img_h)]

        def predict(threshold: float) -> List[Detection]:
            found: List[Detection] = []
            for x1, y1, x2, y2 in windows:
                crop = img[y1:y2, x1:x2]
                tile_found: List[Detection] = []
                results = self.model.predict(
                    crop, conf=threshold, imgsz=imgsz, device=self.device,
                    max_det=1000, verbose=False,
                )
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        width, height = bx2 - bx1, by2 - by1
                        tile_found.append((
                            cls_id, x1 + bx1 + width / 2, y1 + by1 + height / 2,
                            width, height, conf,
                        ))
                if self.pitch_model is not None:
                    pitch_found: List[Detection] = []
                    pitch_results = self.pitch_model.predict(
                        crop, conf=threshold, imgsz=imgsz, device=self.device,
                        max_det=1000, verbose=False,
                    )
                    for result in pitch_results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                            width, height = bx2 - bx1, by2 - by1
                            pitch_found.append((
                                cls_id, x1 + bx1 + width / 2,
                                y1 + by1 + height / 2, width, height, conf,
                            ))
                    production_pitch = [item for item in tile_found if item[0] <= 7]
                    # Keep the production stream as a safety net when the
                    # auxiliary model is unexpectedly sparse on a crop.
                    if len(pitch_found) >= max(5, round(len(production_pitch) * 0.65)):
                        tile_found = [item for item in tile_found if item[0] > 7]
                        tile_found.extend(pitch_found)
                found.extend(tile_found)
            return self._filter_score_rows(self._class_aware_nms(found))

        detections = predict(conf_threshold)
        # High-resolution scans with a textured background or an unfamiliar
        # font often yield coherent but low-confidence score rows. Retry only
        # obviously sparse full pages; short excerpts keep the calibrated
        # threshold that performs best on the labelled validation set.
        if conf_threshold >= 0.15 and self._should_retry_full_page(
                detections, img_w, img_h):
            retry_confidence = max(0.08, conf_threshold * 0.5)
            retry = predict(retry_confidence)
            if self._prefer_retry(detections, retry):
                detections = retry
                self.last_retry_used = True
                self.last_effective_confidence = retry_confidence

        return detections, img_w, img_h

    @classmethod
    def is_staff_notation(cls, image: Image.Image) -> bool:
        """Return True for repeated five-line staff systems.

        A staff page should be rejected before the local jianpu VLM sees it;
        otherwise the VLM may hallucinate alternating rests and digits from
        noteheads and stems.  The check is deliberately the same conservative
        gate used by :meth:`detect`.
        """
        normalized = ImageOps.autocontrast(image.convert("L"), cutoff=0.5)
        return cls._staff_line_group_count(np.asarray(normalized)) >= 2

    @staticmethod
    def _staff_line_group_count(gray: np.ndarray) -> int:
        """Count repeated five-line Western staff systems conservatively."""
        height, width = gray.shape[:2]
        if width < 300 or height < 150:
            return 0
        target_width = 1000
        if width != target_width:
            target_height = max(1, round(height * target_width / width))
            normalized = Image.fromarray(gray).resize(
                (target_width, target_height), Image.Resampling.BILINEAR)
            gray = np.asarray(normalized)

        long_rows = np.flatnonzero((gray < 160).mean(axis=1) > 0.35)
        groups: List[List[int]] = []
        for row in long_rows.tolist():
            if groups and row <= groups[-1][-1] + 1:
                groups[-1].append(row)
            else:
                groups.append([row])
        centers = [float(np.mean(group)) for group in groups]

        count = 0
        index = 0
        while index + 4 < len(centers):
            spacing = np.diff(centers[index:index + 5])
            median = float(np.median(spacing))
            if (4.0 <= median <= 15.0 and
                    float(np.max(np.abs(spacing - median))) <= max(2.0, median * 0.35)):
                count += 1
                index += 5
            else:
                index += 1
        return count

    @staticmethod
    def _page_structure(detections: List[Detection]) -> Tuple[int, int, int]:
        notes = sum(0 <= item[0] <= 8 for item in detections)
        anchors = [item for item in detections if 0 <= item[0] <= 8]
        if not anchors:
            return notes, 0, 0
        tolerance = max(10.0, min(28.0,
                        float(np.median([item[4] for item in anchors])) * 1.35))
        rows: List[List[float]] = []
        for item in sorted(anchors, key=lambda value: value[2]):
            if not rows or abs(item[2] - float(np.median(rows[-1]))) > tolerance:
                rows.append([item[2]])
            else:
                rows[-1].append(item[2])
        structural_ids = {9, 10, 32, 33, 34, 35, 36}
        support = sum(item[0] in structural_ids for item in detections)
        return notes, len(rows), support

    @classmethod
    def _should_retry_full_page(cls, detections: List[Detection],
                                width: int, height: int) -> bool:
        notes, rows, _ = cls._page_structure(detections)
        large_scan = min(width, height) >= 1000 and max(width, height) >= 1600
        portrait_page = width >= 600 and height >= 900 and height / width >= 1.2
        if large_scan:
            return notes < 50 or rows < 6
        if portrait_page:
            return notes < 25 or rows < 4
        return False

    @classmethod
    def _prefer_retry(cls, baseline: List[Detection], retry: List[Detection]) -> bool:
        base_notes, base_rows, _ = cls._page_structure(baseline)
        notes, rows, support = cls._page_structure(retry)
        if notes < max(base_notes + 8, int(base_notes * 1.35)):
            return False
        if rows < max(base_rows, 2) or rows > 30:
            return False
        if notes / max(rows, 1) > 45:
            return False
        return support >= max(2, rows // 2)

    @staticmethod
    def _filter_score_rows(detections: List[Detection]) -> List[Detection]:
        """Reject isolated OCR-like detections outside geometrically valid systems.

        Printed titles, lyrics and page numbers can resemble 1/2/6.  A real
        numbered-notation system instead contains a horizontal run of anchors
        plus barlines or duration underlines.  The gate intentionally uses only
        geometry so it remains independent of language and typeface.
        """
        anchors = [d for d in detections if 0 <= d[0] <= 8]
        if len(anchors) < 3:
            return detections
        median_height = float(np.median([d[4] for d in anchors]))
        tolerance = max(10.0, min(28.0, median_height * 1.35))

        rows: List[List[Detection]] = []
        for anchor in sorted(anchors, key=lambda d: d[2]):
            if not rows or abs(anchor[2] - np.median([item[2] for item in rows[-1]])) > tolerance:
                rows.append([anchor])
            else:
                rows[-1].append(anchor)

        bar_ids = {32, 33, 34, 35, 36}
        underline_ids = {9, 10}
        valid_centers = []
        for row in rows:
            center = float(np.median([d[2] for d in row]))
            bars = [d for d in detections if d[0] in bar_ids and
                    abs(d[2] - center) <= max(tolerance, d[4] / 2)]
            underlines = [d for d in detections if d[0] in underline_ids and
                          -8.0 <= d[2] - center <= 35.0]
            if len(row) >= 5 and (bars or underlines):
                valid_centers.append(center)

        # Do not apply an uncertain gate to sparse excerpts or handwritten
        # fragments where no supported row can be established.
        if not valid_centers:
            return detections

        kept = []
        for detection in detections:
            class_id, _, cy, _, height, _ = detection
            if class_id == 40:  # lyric detections are not consumed by assembler
                continue
            if 0 <= class_id <= 8:
                matches = any(abs(cy - center) <= tolerance for center in valid_centers)
            elif class_id in bar_ids:
                matches = any(abs(cy - center) <= height / 2 + tolerance for center in valid_centers)
            elif class_id in underline_ids:
                matches = any(-8.0 <= cy - center <= 35.0 for center in valid_centers)
            else:
                matches = any(abs(cy - center) <= max(42.0, height / 2 + tolerance)
                              for center in valid_centers)
            if matches:
                kept.append(detection)
        return kept

    @staticmethod
    def _tile_windows(width: int, height: int, tile_size: int, overlap: float):
        stride = max(1, int(tile_size * (1.0 - overlap)))

        def starts(length):
            if length <= tile_size:
                return [0]
            values = list(range(0, length - tile_size + 1, stride))
            last = length - tile_size
            if values[-1] != last:
                values.append(last)
            return values

        return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
                for y in starts(height) for x in starts(width)]

    @staticmethod
    def _class_aware_nms(detections: List[Detection], iou_threshold: float = 0.45):
        # These labels are mutually-exclusive readings of the same glyph. At
        # lower confidence thresholds YOLO can emit (for example) pitch_3 and
        # pitch_5 at the same location. Suppressing only identical class IDs
        # duplicates the note and makes recall-oriented inference unusable.
        exclusive_groups = (
            frozenset(range(0, 9)),       # pitches, rest and dash
            frozenset((9, 10)),           # one/two duration underlines
            frozenset((14, 15, 16)),      # accidentals
            frozenset((22, 23)),          # tie/slur reading of one curve
            frozenset(range(32, 37)),     # barline variants
            frozenset((38, 39)),          # crescendo/decrescendo
        )

        def competes(left: int, right: int) -> bool:
            if left == right:
                return True
            return any(left in group and right in group for group in exclusive_groups)

        def corners(d):
            _, cx, cy, w, h, _ = d
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        def iou(a, b):
            ax1, ay1, ax2, ay2 = corners(a)
            bx1, by1, bx2, by2 = corners(b)
            intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
            return intersection / union if union > 0 else 0.0

        competing_iou_threshold = 0.75
        kept: List[Detection] = []
        for candidate in sorted(detections, key=lambda d: d[5], reverse=True):
            if all(
                not competes(candidate[0], existing[0]) or
                iou(candidate, existing) < (
                    iou_threshold if candidate[0] == existing[0]
                    else competing_iou_threshold
                )
                for existing in kept
            ):
                kept.append(candidate)
        return kept

    @staticmethod
    def class_name(cls_id: int) -> str:
        return CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
