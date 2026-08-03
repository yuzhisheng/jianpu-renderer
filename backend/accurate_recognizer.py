"""Local-VLM recognition path used by the high-accuracy web mode."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image
import cv2
import numpy as np

from assembler import parse_tokens_to_score
from scripts.bootstrap_real_annotations import layout_bands, score_rows


ROOT = Path(__file__).resolve().parent
RESULT_PREFIX = "__JIANPU_RESULT__"
BAR_CLASSES = {32: "B|", 33: "B||", 34: "B|]", 35: "B|:", 36: "B:|"}
ANCHOR_CLASSES = frozenset(range(8))
UNDERLINE_CLASSES = {9: "_", 10: "="}
NOTE_MODIFIER_CLASSES = {11: ".", 12: "^", 13: "v", 14: "#", 15: "b", 16: "n"}
SKELETON_WARNING = (
    "已融合谱行、小节线、减时线和像素复核的八度/附点；三连音需同时检出小3，"
    "圆滑线/延音线因误检率较高暂不自动输出，歌词仍建议人工复核。"
)
LOGGER = logging.getLogger("jianpu-accurate-recognizer")


class AccurateRecognizerBusyError(RuntimeError):
    """Raised when another memory-heavy VLM recognition is already running."""


class AccurateRecognizerInterruptedError(RuntimeError):
    """Raised when the local VLM subprocess is externally interrupted."""


class AccurateRecognizerTimeoutError(RuntimeError):
    """Raised when the local VLM exceeds the bounded inference time."""


class AccurateVLMRecognizer:
    def __init__(self):
        self.python = ROOT / ".venv-vlm" / "bin" / "python"
        self.script = ROOT / "scripts" / "infer_page_with_local_vlm.py"
        self.cache = ROOT / ".cache" / "huggingface"
        # MLX inference can exhaust unified memory or become dramatically slower
        # when two full-page jobs overlap. Reject duplicates instead of queuing a
        # second multi-minute subprocess behind the first one.
        self._vlm_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.python.exists() and self.script.exists() and self.cache.exists()

    @staticmethod
    def _hybrid_bands(image: Image.Image, detections: Iterable[Sequence[float]]):
        """Mirror the reviewed-dataset crop policy for online inference."""
        bands = layout_bands(image)
        specs = score_rows(detections)
        height = image.height
        if specs and len(bands) <= max(3, int(len(specs) * 0.6)):
            bands = []
            for offset, (center, _, anchor_height) in enumerate(specs):
                upper = 0 if offset == 0 else int((specs[offset - 1][0] + center) / 2)
                lower = height if offset + 1 == len(specs) else int(
                    (center + specs[offset + 1][0]) / 2)
                bands.append((
                    max(upper, int(center - max(80.0, anchor_height * 3.0))),
                    min(lower, int(center + max(100.0, anchor_height * 4.5))),
                ))
        elif specs:
            refined = []
            for top, bottom in bands:
                contained_specs = sorted(
                    (center, anchor_height) for center, _, anchor_height in specs
                    if top <= center <= bottom)
                centers = [center for center, _ in contained_specs]
                cuts = [int((left + right) / 2)
                        for left, right in zip(centers, centers[1:])
                        if right - left >= 60.0]
                edges = [top, *cuts, bottom]
                for left, right in zip(edges, edges[1:]):
                    # Keep the projection band's outer edges. Detector anchors
                    # are allowed to split a dense band, but must not trim it:
                    # on faint final systems the detector may lock onto footer
                    # digits and otherwise discard the actual music baseline.
                    if right - left >= 8:
                        refined.append((left, right))
            bands = refined
        return bands

    def _run_vlm(
        self, image: Image.Image, detections: Iterable[Sequence[float]],
    ) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("本地视觉大模型环境不完整，请检查 backend/.venv-vlm 和模型缓存")
        if not self._vlm_lock.acquire(blocking=False):
            raise AccurateRecognizerBusyError("已有一个精确识别任务正在运行")
        try:
            with tempfile.TemporaryDirectory(prefix="jianpu-vlm-") as directory:
                image_path = Path(directory) / "page.png"
                bands_path = Path(directory) / "bands.json"
                image.save(image_path, format="PNG")
                bands_path.write_text(json.dumps(
                    self._hybrid_bands(image, detections), ensure_ascii=False))
                environment = os.environ.copy()
                environment["HF_HOME"] = str(self.cache)
                environment["HF_HUB_OFFLINE"] = "1"
                try:
                    completed = subprocess.run(
                        [str(self.python), str(self.script), str(image_path),
                         "--batch-size", "4", "--max-tokens", "512",
                         "--bands-json", str(bands_path), "--skip-relations"],
                        cwd=str(ROOT.parent), env=environment, capture_output=True,
                        text=True, timeout=900, check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise AccurateRecognizerTimeoutError(
                        "本地视觉大模型推理超过 15 分钟，已停止本次任务") from exc
        finally:
            self._vlm_lock.release()
        marker = next(
            (line[len(RESULT_PREFIX):] for line in reversed(completed.stdout.splitlines())
             if line.startswith(RESULT_PREFIX)),
            None,
        )
        if completed.returncode != 0 or marker is None:
            detail = (completed.stderr or completed.stdout)[-2000:]
            LOGGER.error("本地视觉大模型推理失败（退出码 %s）:\n%s",
                         completed.returncode, detail)
            if ("KeyboardInterrupt" in detail
                    or completed.returncode in {-2, 130}):
                raise AccurateRecognizerInterruptedError(
                    "本地视觉大模型推理被服务重启或系统中断，请重新识别")
            lines = [line.strip() for line in detail.splitlines() if line.strip()]
            summary = lines[-1] if lines else f"退出码 {completed.returncode}"
            raise RuntimeError(f"本地视觉大模型推理失败: {summary}")
        return json.loads(marker)

    @staticmethod
    def _dedupe_bars(detections: Iterable[Sequence[float]], top: int, bottom: int):
        row_items = [item for item in detections if top <= float(item[2]) <= bottom]
        anchors = [item for item in row_items if int(item[0]) in ANCHOR_CLASSES
                   and float(item[5]) >= 0.10]
        if anchors:
            typical_height = median(float(item[4]) for item in anchors)
            baseline = max(
                (float(item[2]) for item in anchors),
                key=lambda value: sum(
                    abs(float(other[2]) - value) <= max(10.0, typical_height * 0.75)
                    for other in anchors),
            )
        else:
            typical_height = 20.0
            baseline = (top + bottom) / 2
        candidates = [
            item for item in detections
            if int(item[0]) in BAR_CLASSES and top <= float(item[2]) <= bottom
            # Real scanned barlines are frequently only 0.12--0.18 confident.
            # The music-baseline constraint prevents the lower threshold from
            # admitting vertical strokes in lyrics and metadata.
            and float(item[5]) >= 0.12
            and abs(float(item[2]) - baseline) <= max(50.0, typical_height * 2.0)
        ]
        selected = []
        for item in sorted(candidates, key=lambda value: float(value[5]), reverse=True):
            if all(abs(float(item[1]) - float(old[1])) > 12 for old in selected):
                selected.append(item)
        return sorted(selected, key=lambda value: float(value[1]))

    @staticmethod
    def _normalize_accidentals(tokens: List[str]):
        normalized = []
        index = 0
        while index < len(tokens):
            if (tokens[index] in {"#", "b", "n"} and index + 1 < len(tokens)
                    and tokens[index + 1].startswith("P")):
                normalized.extend((tokens[index + 1], tokens[index]))
                index += 2
            else:
                normalized.append(tokens[index])
                index += 1
        return normalized

    @staticmethod
    def _apply_vlm_modifiers(tokens: List[str], modifiers):
        if not isinstance(modifiers, list) or not modifiers:
            return tokens
        by_note = {}
        for modifier in modifiers:
            note = modifier.get("note")
            if isinstance(note, int) and note >= 0:
                by_note[note] = modifier
        output = []
        note_index = 0
        for token in tokens:
            output.append(token)
            if not (token.startswith("P") or token == "R0"):
                continue
            modifier = by_note.get(note_index, {})
            octave = modifier.get("octave", 0)
            dot = modifier.get("dot", 0)
            if isinstance(octave, int):
                output.extend("^" for _ in range(min(2, max(0, octave))))
                output.extend("v" for _ in range(min(2, max(0, -octave))))
            if isinstance(dot, int):
                output.extend("." for _ in range(min(2, max(0, dot))))
            note_index += 1
        return output

    def _merge_geometry_bars(
        self, tokens: List[str], detections: Iterable[Sequence[float]],
        crop_box: Sequence[int], page_width: int, image: Image.Image | None = None,
        token_geometry: List[Dict[str, Any]] | None = None,
    ) -> List[str]:
        top, bottom = int(crop_box[1]), int(crop_box[3])
        _, baseline, note_height, _ = self._row_geometry(detections, crop_box)
        source_note_boxes = []
        if token_geometry and len(token_geometry) == len(tokens):
            for token, geometry in zip(tokens, token_geometry):
                box = geometry.get("box") if isinstance(geometry, dict) else None
                if ((token.startswith("P") or token == "R0")
                        and isinstance(box, list) and len(box) == 4):
                    source_note_boxes.append(box)
        if len(source_note_boxes) >= 3:
            # The last crop often contains a footer with many OCR-like digits.
            # Its detector cluster can outvote the actual score baseline; glyph
            # localization was performed on the music row and is authoritative.
            baseline = median(float(box[1]) for box in source_note_boxes)
            note_height = median(float(box[3]) for box in source_note_boxes)
        pixel_bars = self._pixel_bars(
            image, crop_box, baseline, note_height) if image is not None else []
        bars = (pixel_bars if pixel_bars else [
            (BAR_CLASSES[int(item[0])], float(item[1]))
            for item in self._dedupe_bars(detections, top, bottom)
        ])
        content_with_geometry = [
            (token, token_geometry[index] if token_geometry and index < len(token_geometry) else None)
            for index, token in enumerate(tokens) if not token.startswith("B")
        ]
        content = [item[0] for item in content_with_geometry]
        if not bars:
            return self._normalize_accidentals(tokens)

        bar_center = baseline
        anchors = sorted(
            float(item[1]) for item in detections
            if 0 <= int(item[0]) <= 7 and top <= float(item[2]) <= bottom
            and abs(float(item[2]) - bar_center) <= 34 and float(item[5]) >= 0.10
        )
        if len(anchors) >= 3:
            left, right = anchors[0], anchors[-1]
        else:
            left, right = page_width * 0.04, page_width * 0.96
        span = max(1.0, right - left)
        positions = []
        for index, (_, geometry) in enumerate(content_with_geometry):
            if isinstance(geometry, dict) and isinstance(geometry.get("x"), (int, float)):
                positions.append(float(geometry["x"]))
            else:
                positions.append(left + span * (index + 0.5) / max(1, len(content)))
        insertions = []
        for bar_token, bar_x in bars:
            position = sum(token_x < bar_x for token_x in positions)
            insertions.append((position, bar_token))
        result = list(content)
        for position, token in reversed(insertions):
            result.insert(position, token)
        return self._normalize_accidentals(result)

    @staticmethod
    def _pixel_bars(
        image: Image.Image, crop_box: Sequence[int], baseline: float, note_height: float,
    ) -> List[tuple[str, float]]:
        """Read bar/repeat signs from tall ink and colon-dot sidedness.

        Detector classes are useful as a fallback, but on scans they often call
        a repeat sign a normal bar.  Tall vertical components give exact x and
        the two compact dots say whether the repeat opens or closes.
        """
        top, bottom = int(crop_box[1]), int(crop_box[3])
        gray = np.asarray(image.convert("L"))[top:bottom, :]
        binary = (gray < 150).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        lines = []
        dots = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            center_x, local_y = centroids[index]
            center_y = float(local_y + top)
            density = area / max(1, width * height)
            if (width <= 14
                    and height >= max(36.0, note_height * 1.25)
                    and height / max(1, width) >= 4.0
                    and density >= 0.70
                    and abs(center_y - baseline) <= max(50.0, note_height * 1.8)):
                lines.append((float(center_x), center_y, float(width), float(height)))
            max_dot = max(11.0, note_height * 0.50)
            if (3 <= width <= max_dot and 3 <= height <= max_dot
                    and 0.58 <= width / max(1, height) <= 1.65
                    and density >= 0.45
                    and abs(center_y - baseline) <= max(24.0, note_height * 0.95)):
                dots.append((float(center_x), center_y))
        if not lines:
            return []

        groups: List[List[Sequence[float]]] = []
        for line in sorted(lines):
            if groups and line[0] - groups[-1][-1][0] <= 16.0:
                groups[-1].append(line)
            else:
                groups.append([line])

        result = []
        for group in groups:
            left_x, right_x = group[0][0], group[-1][0]
            center_x = sum(line[0] for line in group) / len(group)

            def has_colon(side: str) -> bool:
                if side == "left":
                    candidates = [dot_y for dot_x, dot_y in dots
                                  if left_x - 32 <= dot_x <= left_x - 7]
                else:
                    candidates = [dot_y for dot_x, dot_y in dots
                                  if right_x + 7 <= dot_x <= right_x + 32]
                candidates.sort()
                return any(second - first >= max(8.0, note_height * 0.30)
                           for first, second in zip(candidates, candidates[1:]))

            left_colon, right_colon = has_colon("left"), has_colon("right")
            if left_colon and not right_colon:
                token = "B:|"
            elif right_colon and not left_colon:
                token = "B|:"
            elif len(group) >= 2:
                token = "B||"
            else:
                token = "B|"
            result.append((token, center_x))
        return result

    @staticmethod
    def _row_geometry(
        detections: Iterable[Sequence[float]], crop_box: Sequence[int],
    ):
        top, bottom = int(crop_box[1]), int(crop_box[3])
        row_detections = [item for item in detections if top <= float(item[2]) <= bottom]
        bars = [item for item in row_detections if int(item[0]) in BAR_CLASSES
                and float(item[5]) >= 0.18]
        anchors = [item for item in row_detections if int(item[0]) in ANCHOR_CLASSES
                   and float(item[5]) >= 0.10]
        if anchors:
            # Choose the densest y cluster so lyric OCR false positives do not
            # define the music baseline on a crop that contains both.
            typical_height = median(float(item[4]) for item in anchors)
            clusters: List[List[Sequence[float]]] = []
            for item in sorted(anchors, key=lambda value: float(value[2])):
                match = next((cluster for cluster in clusters
                              if abs(float(item[2]) - median(float(old[2]) for old in cluster))
                              <= max(8.0, typical_height * 0.8)), None)
                (match if match is not None else clusters.append([item]))
                if match is not None:
                    match.append(item)
            cluster = max(clusters, key=lambda value: (len(value),
                          sum(float(item[5]) for item in value)))
            baseline = median(float(item[2]) for item in cluster)
        elif bars:
            baseline = median(float(item[2]) for item in bars)
        else:
            baseline = (top + bottom) / 2
        typical_height = median(float(item[4]) for item in anchors) if anchors else 20.0
        baseline_anchors = sorted(
            (item for item in anchors
             if abs(float(item[2]) - baseline) <= max(12.0, typical_height * 0.9)),
            key=lambda value: float(value[1]),
        )
        return row_detections, baseline, typical_height, baseline_anchors

    @staticmethod
    def _component_note_boxes(
        image: Image.Image, crop_box: Sequence[int], baseline: float,
        note_height: float, expected_count: int,
    ) -> List[Sequence[float]]:
        """Recover precise digit boxes from ink when their count is unambiguous.

        The detector is useful for locating the baseline but often misses faint
        digits. Connected components recover their x coordinates, which makes
        small octave dots and underline assignment substantially more reliable.
        """
        if expected_count <= 0:
            return []
        top, bottom = int(crop_box[1]), int(crop_box[3])
        gray = np.asarray(image.convert("L"))[top:bottom, :]
        binary = (gray < 150).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        candidates = []
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            center_x, local_y = centroids[index]
            center_y = float(local_y + top)
            if not (
                max(3.0, note_height * 0.12) <= width <= note_height * 1.45
                and note_height * 0.52 <= height <= note_height * 1.65
                and 0.10 <= width / max(1, height) <= 1.45
                and area >= max(18.0, note_height * note_height * 0.08)
                and abs(center_y - baseline) <= max(10.0, note_height * 0.68)
            ):
                continue
            candidates.append((float(center_x), center_y, float(width), float(height),
                               abs(center_y - baseline)))
        candidates.sort(key=lambda value: value[0])
        if len(candidates) == expected_count:
            return [value[:4] for value in candidates]
        # One extra component is normally a time-signature digit or ornament;
        # discard the item furthest from the established music baseline.
        if len(candidates) == expected_count + 1:
            worst = max(range(len(candidates)), key=lambda index: candidates[index][4])
            return [value[:4] for index, value in enumerate(candidates) if index != worst]
        return []

    @classmethod
    def _component_note_positions(
        cls, image: Image.Image, crop_box: Sequence[int], baseline: float,
        note_height: float, expected_count: int,
    ) -> List[float]:
        return [box[0] for box in cls._component_note_boxes(
            image, crop_box, baseline, note_height, expected_count)]

    @staticmethod
    def _interpolated_note_positions(note_count: int, anchors: List[Sequence[float]], page_width: int):
        if note_count <= 0:
            return []
        if not anchors:
            return [page_width * (index + 1) / (note_count + 1) for index in range(note_count)]
        anchor_x = [float(item[1]) for item in anchors]
        if note_count == 1:
            return [median(anchor_x)]
        if len(anchor_x) == 1:
            return [anchor_x[0]] * note_count
        result = []
        for index in range(note_count):
            rank = index * (len(anchor_x) - 1) / (note_count - 1)
            left = int(rank)
            right = min(len(anchor_x) - 1, left + 1)
            fraction = rank - left
            result.append(anchor_x[left] * (1 - fraction) + anchor_x[right] * fraction)
        return result

    def _apply_note_modifiers(
        self, tokens: List[str], detections: Iterable[Sequence[float]],
        crop_box: Sequence[int], page_width: int, image: Image.Image | None = None,
        preferred_note_boxes: List[Sequence[float]] | None = None,
    ):
        row_detections, baseline, note_height, anchors = self._row_geometry(
            detections, crop_box)
        note_count = sum(token.startswith("P") or token == "R0" for token in tokens)
        note_positions = self._interpolated_note_positions(note_count, anchors, page_width)
        component_boxes = (list(preferred_note_boxes)
                           if preferred_note_boxes and len(preferred_note_boxes) == note_count
                           else [])
        if image is not None and not component_boxes:
            component_boxes = self._component_note_boxes(
                image, crop_box, baseline, note_height, note_count)
        if component_boxes:
            note_positions = [float(box[0]) for box in component_boxes]
        anchor_width = median(float(item[3]) for item in anchors) if anchors else 16.0
        rhythm_boxes = [
            item for item in row_detections
            if int(item[0]) in UNDERLINE_CLASSES
            and baseline + note_height * 0.2 <= float(item[2]) <= baseline + note_height * 1.8
            and float(item[5]) >= 0.15
        ]
        modifier_boxes = [
            item for item in row_detections
            if int(item[0]) in NOTE_MODIFIER_CLASSES and float(item[5]) >= 0.20
            # With source glyph boxes, compact pixels are more reliable than
            # detector guesses for octave/augmentation dots. Accidentals still
            # use the detector because they are not dot-like components.
            and (not component_boxes or int(item[0]) >= 14)
        ]
        visual_modifiers = self._visual_dot_modifiers(
            image, crop_box, note_positions, anchors, component_boxes
        ) if image is not None else {}
        visual_rhythm = self._visual_rhythm_depths(
            image, crop_box, component_boxes) if image is not None and component_boxes else {}

        output = []
        note_index = 0
        local_modifier_tokens = set(NOTE_MODIFIER_CLASSES.values()) | {"_", "=", "D0.125"}
        for token_position, token in enumerate(tokens):
            output.append(token)
            if not (token.startswith("P") or token == "R0"):
                continue
            x = note_positions[note_index]
            note_index += 1
            following = set()
            for later in tokens[token_position + 1:]:
                if later.startswith("P") or later == "R0" or later == "-" or later.startswith("B"):
                    break
                if later in local_modifier_tokens:
                    following.add(later)

            for modifier in visual_modifiers.get(note_index - 1, []):
                if modifier not in following:
                    output.append(modifier)
                    following.add(modifier)

            nearby = sorted(
                (item for item in modifier_boxes
                 if abs(float(item[1]) - x) <= max(anchor_width * 1.3, float(item[3]) / 2 + 4)),
                key=lambda item: (int(item[0]), -float(item[5])),
            )
            seen = set()
            for item in nearby:
                modifier = NOTE_MODIFIER_CLASSES[int(item[0])]
                if modifier not in seen and modifier not in following:
                    output.append(modifier)
                    seen.add(modifier)

            depth = visual_rhythm.get(note_index - 1, 0)
            if not component_boxes:
                for item in rhythm_boxes:
                    left = float(item[1]) - float(item[3]) / 2 - anchor_width * 0.25
                    right = float(item[1]) + float(item[3]) / 2 + anchor_width * 0.25
                    if left <= x <= right:
                        depth = max(depth, 2 if int(item[0]) == 10 else 1)
            if depth and not ({"_", "=", "D0.125"} & following):
                output.append("D0.125" if depth >= 3 else "=" if depth == 2 else "_")
        return output, note_positions, row_detections, baseline, note_height

    @staticmethod
    def _visual_dot_modifiers(
        image: Image.Image, crop_box: Sequence[int], note_positions: List[float],
        anchors: List[Sequence[float]], note_boxes: List[Sequence[float]] | None = None,
    ):
        """Find octave/augmentation dots as compact ink components near digits."""
        if not note_positions or (not anchors and not note_boxes):
            return {}
        top, bottom = int(crop_box[1]), int(crop_box[3])
        gray = np.asarray(image.convert("L"))[top:bottom, :]
        binary = (gray < 120).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        components = []
        if note_boxes:
            typical_width = median(float(item[2]) for item in note_boxes)
            typical_height = median(float(item[3]) for item in note_boxes)
        else:
            typical_width = median(float(item[3]) for item in anchors)
            typical_height = median(float(item[4]) for item in anchors)
        max_dot_size = max(7, round(typical_height * 0.48))
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if (2 <= width <= max_dot_size and 2 <= height <= max_dot_size
                    and 4 <= area
                    and 0.60 <= width / height <= 1.65
                    and area / (width * height) >= 0.42):
                components.append((float(centroids[index][0]),
                                   float(centroids[index][1] + top),
                                   width, height, int(area)))

        result = {}
        anchor_x = [float(item[1]) for item in anchors]
        for note_index, note_x in enumerate(note_positions):
            if note_boxes and note_index < len(note_boxes):
                center_x, center_y, width, height = map(float, note_boxes[note_index])
            else:
                anchor = anchors[min(range(len(anchors)), key=lambda i: abs(anchor_x[i] - note_x))]
                center_x, center_y = float(anchor[1]), float(anchor[2])
                width, height = float(anchor[3]), float(anchor[4])
            left, right = center_x - width / 2, center_x + width / 2
            upper = []
            lower = []
            augmentation = []
            for dot_x, dot_y, dot_width, dot_height, dot_area in components:
                if abs(dot_x - center_x) <= max(5.0, width * 0.34):
                    upper_gap = center_y - height / 2 - dot_y
                    strict_round_dot = (
                        abs(dot_x - center_x) <= max(4.0, width * 0.25)
                        and 0.68 <= dot_width / dot_height <= 1.48
                        and dot_area / (dot_width * dot_height) >= 0.48
                    )
                    if strict_round_dot and 3 <= upper_gap <= max(18.0, height * 1.35):
                        upper.append(dot_y)
                    lower_gap = dot_y - (center_y + height / 2)
                    if (strict_round_dot and 2 <= lower_gap
                            <= max(20.0, height * 0.90)):
                        lower.append(dot_y)
                if (right + 1 <= dot_x <= right + max(12.0, typical_width * 0.8)
                        and abs(dot_y - center_y) <= height * 0.38):
                    augmentation.append(dot_x)
            # A single printed note cannot be both above and below the central
            # octave. If scan noise produces evidence on both sides, the upper
            # candidate has the stricter arc-safe geometry and wins.
            if upper and lower:
                lower = []
            modifiers = (["^"] * min(2, len(upper))
                         + ["v"] * min(2, len(lower))
                         + ["."] * min(2, len(augmentation)))
            if modifiers:
                result[note_index] = modifiers
        return result

    @staticmethod
    def _visual_rhythm_depths(
        image: Image.Image, crop_box: Sequence[int], note_boxes: List[Sequence[float]],
    ):
        """Read one/two/three reduction lines below localized digit glyphs."""
        if not note_boxes:
            return {}
        top, bottom = int(crop_box[1]), int(crop_box[3])
        gray = np.asarray(image.convert("L"))[top:bottom, :]
        binary = (gray < 150).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        median_width = median(float(box[2]) for box in note_boxes)
        median_height = median(float(box[3]) for box in note_boxes)
        lines = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if (width >= max(10.0, median_width * 0.48)
                    and height <= max(8.0, median_height * 0.22)
                    and width / max(1, height) >= 2.2
                    and area / (width * height) >= 0.38):
                lines.append((float(x), float(x + width),
                              float(centroids[index][1] + top)))
        result = {}
        for note_index, (center_x, center_y, width, height) in enumerate(note_boxes):
            digit_bottom = float(center_y) + float(height) / 2
            levels = []
            for left, right, line_y in lines:
                gap = line_y - digit_bottom
                if (2 <= gap <= max(24.0, float(height) * 0.9)
                        and left - width * 0.2 <= center_x <= right + width * 0.2):
                    if all(abs(line_y - old) > 3 for old in levels):
                        levels.append(line_y)
            if levels:
                result[note_index] = min(3, len(levels))
        return result

    @staticmethod
    def _curve_relations(
        pitch_tokens: List[str], note_positions: List[float],
        row_detections: List[Sequence[float]], baseline: float, note_height: float,
    ):
        candidates = [
            item for item in row_detections
            if int(item[0]) in {22, 23} and float(item[5]) >= 0.35
            and baseline - note_height * 2.2 <= float(item[2]) <= baseline - note_height * 0.25
            and float(item[3]) >= note_height * 1.1
        ]
        candidates.sort(key=lambda item: float(item[5]), reverse=True)
        selected = []
        for item in candidates:
            left, right = float(item[1]) - float(item[3]) / 2, float(item[1]) + float(item[3]) / 2
            duplicate = False
            for old in selected:
                old_left = float(old[1]) - float(old[3]) / 2
                old_right = float(old[1]) + float(old[3]) / 2
                overlap = max(0.0, min(right, old_right) - max(left, old_left))
                if overlap / max(1.0, min(right - left, old_right - old_left)) >= 0.72:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(item)

        relations = []
        for item in sorted(selected, key=lambda value: float(value[1])):
            pad = note_height * 0.35
            left, right = (float(item[1]) - float(item[3]) / 2 - pad,
                           float(item[1]) + float(item[3]) / 2 + pad)
            covered = [index for index, x in enumerate(note_positions) if left <= x <= right]
            if len(covered) < 2 and len(note_positions) >= 2:
                center = float(item[1])
                nearest = sorted(range(len(note_positions)), key=lambda index: abs(note_positions[index] - center))[:2]
                covered = sorted(nearest)
            if len(covered) < 2:
                continue
            start, end = covered[0], covered[-1]
            if start == end:
                continue
            left_pitch = pitch_tokens[start]
            right_pitch = pitch_tokens[end]
            kind = "tie" if left_pitch == right_pitch else "slur"
            relation = (kind, start, end)
            if relation not in relations:
                relations.append(relation)
        return relations

    @staticmethod
    def _decorate_relations(score: Dict[str, Any], relations):
        notes = [item for measure in score.get("measures", []) for item in measure.get("notes", [])
                 if "pitch" in item]
        for relation_index, (kind, start, end) in enumerate(relations, 1):
            if not (0 <= start < len(notes) and 0 <= end < len(notes)) or start == end:
                continue
            field = {"tie": "tieId", "slur": "slurId", "triplet": "tripletId"}.get(kind)
            if field is None or field in notes[start] or field in notes[end]:
                continue
            relation_id = f"ocr-{kind}-{relation_index}"
            notes[start][field] = relation_id
            notes[end][field] = relation_id

    @staticmethod
    def _normalize_relation_types(relations):
        """Recover repeated three-note tuplets that a VLM calls short slurs.

        A single three-note slur is valid notation, so it stays a slur. Two or
        more non-overlapping, consecutive three-note arcs in one score row are
        the characteristic jianpu triplet pattern (the tiny printed 3 is often
        below the VLM's readable size).
        """
        candidates = [(start, end) for kind, start, end in relations
                      if kind == "slur" and end - start == 2]
        non_overlapping = []
        for start, end in sorted(candidates):
            if not non_overlapping or start > non_overlapping[-1][1]:
                non_overlapping.append((start, end))
        if len(non_overlapping) < 2:
            return relations
        triplet_ranges = set(non_overlapping)
        return [("triplet" if kind == "slur" and (start, end) in triplet_ranges else kind,
                 start, end) for kind, start, end in relations]

    @classmethod
    def _production_relations(cls, relations):
        normalized = cls._normalize_relation_types(relations)
        return [relation for relation in normalized if relation[0] == "triplet"]

    @staticmethod
    def _triplet_marker_visible(
        start: int, end: int, note_positions: List[float],
        row_detections: List[Sequence[float]], baseline: float, note_height: float,
    ) -> bool:
        """Require a small detected digit 3 above the claimed tuplet span."""
        if not (0 <= start < end < len(note_positions)):
            return False
        left, right = note_positions[start], note_positions[end]
        return any(
            int(item[0]) == 2 and float(item[5]) >= 0.08
            and left - note_height * 0.4 <= float(item[1]) <= right + note_height * 0.4
            and baseline - note_height * 2.5 <= float(item[2]) <= baseline - note_height * 0.35
            and float(item[4]) <= note_height * 0.9
            for item in row_detections
        )

    def recognize(self, image: Image.Image, detections: Iterable[Sequence[float]]):
        payload = self._run_vlm(image, detections)
        tokens: List[str] = []
        confidences = []
        relations = []
        note_offset = 0
        for row in payload.get("rows", []):
            if row.get("content_type") != "score":
                continue
            confidences.append(float(row.get("confidence", 0.0)))
            for voice in row.get("voices", []):
                original_tokens = list(voice.get("tokens", []))
                original_geometry = voice.get("token_geometry")
                if (isinstance(original_geometry, list)
                        and len(original_geometry) == len(original_tokens)):
                    pairs = [(token, geometry) for token, geometry
                             in zip(original_tokens, original_geometry) if token != "?"]
                    voice_tokens = [token for token, _ in pairs]
                    token_geometry = [geometry for _, geometry in pairs]
                else:
                    voice_tokens = [token for token in original_tokens if token != "?"]
                    token_geometry = None
                preferred_note_boxes = []
                if token_geometry:
                    for token, geometry in zip(voice_tokens, token_geometry):
                        box = geometry.get("box") if isinstance(geometry, dict) else None
                        if ((token.startswith("P") or token == "R0")
                                and isinstance(box, list) and len(box) == 4):
                            preferred_note_boxes.append(box)
                merged = self._merge_geometry_bars(
                    voice_tokens, detections, row["crop_box"], image.width,
                    image, token_geometry)
                decorated, positions, row_detections, baseline, note_height = self._apply_note_modifiers(
                    merged, detections, row["crop_box"], image.width, image,
                    preferred_note_boxes)
                pitch_tokens = [token for token in merged if token.startswith("P") or token == "R0"]
                vlm_relations = voice.get("relations")
                if isinstance(vlm_relations, list):
                    row_relations = []
                    for relation in vlm_relations:
                        kind = relation.get("type")
                        start, end = relation.get("start"), relation.get("end")
                        if (kind in {"tie", "slur", "triplet"}
                                and isinstance(start, int) and isinstance(end, int)
                                and 0 <= start < end < len(pitch_tokens)):
                            row_relations.append((kind, start, end))
                else:
                    row_relations = []
                # Precision-first production policy: generic VLM/YOLO arcs
                # frequently confuse tuplets, phrase arcs and ornament curves.
                # Do not emit tie/slur until an endpoint model is calibrated.
                row_relations = self._production_relations(row_relations)
                row_relations = [
                    relation for relation in row_relations
                    if self._triplet_marker_visible(
                        relation[1], relation[2], positions, row_detections,
                        baseline, note_height)
                ]
                for kind, start, end in row_relations:
                    relations.append((kind, note_offset + start, note_offset + end))
                note_offset += len(pitch_tokens)
                tokens.extend(decorated)
                if decorated and not any(token.startswith("B") for token in decorated[-2:]):
                    tokens.append("B|")
                tokens.append("<ROW>")
        if not any(token.startswith("P") or token == "R0" for token in tokens):
            raise RuntimeError("本地视觉大模型没有在图片中找到数字简谱")
        score = parse_tokens_to_score(tokens)
        self._decorate_relations(score, relations)
        score["title"] = "识别结果（音高骨架）"
        score_notes = [item for measure in score.get("measures", [])
                       for item in measure.get("notes", []) if "pitch" in item]
        symbol_summary = {
            "notes": len(score_notes),
            "eighth_notes": sum(note.get("duration") == 0.5 for note in score_notes),
            "sixteenth_notes": sum(note.get("duration") == 0.25 for note in score_notes),
            "thirty_second_notes": sum(note.get("duration") == 0.125 for note in score_notes),
            "octave_marks": sum(bool(note.get("octave")) for note in score_notes),
            "augmentation_dots": sum(bool(note.get("dot")) for note in score_notes),
            "ties": len({note["tieId"] for note in score_notes if note.get("tieId")}),
            "slurs": len({note["slurId"] for note in score_notes if note.get("slurId")}),
            "triplets": len({note["tripletId"] for note in score_notes if note.get("tripletId")}),
        }
        return {
            "score": score,
            "src_tokens": [],
            "tgt_tokens": tokens,
            "row_results": payload.get("rows", []),
            "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
            "warnings": [SKELETON_WARNING],
            "vlm_seconds": payload.get("elapsed_seconds"),
            "symbol_summary": symbol_summary,
        }
