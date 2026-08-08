"""Local-VLM recognition path used by the high-accuracy web mode."""
from __future__ import annotations

import json
import hashlib
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
    "已融合谱行、小节线、减时线、弧线端点、小房子和分组括号的像素复核；"
    "极淡或跨行的关系符号及歌词仍建议人工复核。"
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
        self.result_cache = ROOT / ".cache" / "accurate_vlm_results"
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
            bands = self._hybrid_bands(image, detections)
            cache_key = hashlib.sha256()
            cache_key.update(image.size[0].to_bytes(4, "big"))
            cache_key.update(image.size[1].to_bytes(4, "big"))
            cache_key.update(image.tobytes())
            cache_key.update(json.dumps(bands, separators=(",", ":")).encode())
            try:
                cache_key.update(str(self.script.stat().st_mtime_ns).encode())
            except OSError:
                pass
            cached_result = self.result_cache / f"{cache_key.hexdigest()}.json"
            if os.environ.get("JIANPU_VLM_CACHE", "1") != "0":
                try:
                    return json.loads(cached_result.read_text())
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    pass
            with tempfile.TemporaryDirectory(prefix="jianpu-vlm-") as directory:
                image_path = Path(directory) / "page.png"
                bands_path = Path(directory) / "bands.json"
                image.save(image_path, format="PNG")
                bands_path.write_text(json.dumps(
                    bands, ensure_ascii=False))
                environment = os.environ.copy()
                environment["HF_HOME"] = str(self.cache)
                environment["HF_HUB_OFFLINE"] = "1"
                max_tokens = max(512, int(environment.get("JIANPU_VLM_MAX_TOKENS", "768")))
                try:
                    completed = subprocess.run(
                        [str(self.python), str(self.script), str(image_path),
                         "--batch-size", "4", "--max-tokens", str(max_tokens),
                         "--bands-json", str(bands_path), "--skip-relations",
                         "--text-layers"],
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
        result = json.loads(marker)
        if os.environ.get("JIANPU_VLM_CACHE", "1") != "0":
            try:
                self.result_cache.mkdir(parents=True, exist_ok=True)
                temporary = cached_result.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False))
                temporary.replace(cached_result)
            except OSError:
                LOGGER.warning("无法写入本地视觉模型缓存: %s", cached_result)
        return result

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
                    # The upper repeat dot is often above the detected digit
                    # baseline by slightly more than one glyph height on
                    # dense scans.  Keep the x-side constraint narrow while
                    # allowing that real vertical offset; otherwise B:|/B|:
                    # collapses into an ordinary double bar.
                    and abs(center_y - baseline) <= max(28.0, note_height * 1.25)):
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

        # On wide scans the detector may lock its pitch boxes onto the first
        # reduction line (or an octave dot) instead of the printed digit. In
        # that case the baseline/height above are too small and the strict
        # pass finds nothing, even though the actual digit components are
        # cleanly separated. Recover the dominant glyph band without relying
        # on detector y coordinates. The expected note count is a strong guard
        # against lyric/title components in a full-page crop.
        visual_candidates = []
        min_height = max(12.0, note_height * 0.75)
        max_height = max(120.0, note_height * 4.0)
        min_width = max(4.0, note_height * 0.35)
        max_width = max(80.0, note_height * 2.5)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            center_x, local_y = centroids[index]
            center_y = float(local_y + top)
            density = area / max(1, width * height)
            if not (
                min_width <= width <= max_width
                and min_height <= height <= max_height
                and 0.18 <= width / max(1, height) <= 1.45
                and area >= max(20.0, note_height * note_height * 0.12)
                and density >= 0.08
            ):
                continue
            visual_candidates.append((float(center_x), center_y, float(width),
                                      float(height), float(area)))
        if len(visual_candidates) < max(3, min(expected_count, 3)):
            return []

        # Cluster by vertical band and glyph height. A row's music digits form
        # a much denser, more consistent cluster than lyrics or page chrome.
        bands: List[List[Sequence[float]]] = []
        for candidate in sorted(visual_candidates, key=lambda value: value[1]):
            matching = next((band for band in bands
                             if abs(candidate[1] - median(item[1] for item in band))
                             <= max(10.0, median(item[3] for item in band) * 0.30)
                             and abs(candidate[3] - median(item[3] for item in band))
                             <= max(10.0, median(item[3] for item in band) * 0.30)),
                            None)
            if matching is None:
                bands.append([candidate])
            else:
                matching.append(candidate)
        bands.sort(key=lambda band: (abs(len(band) - expected_count), -len(band)))
        visual_band = bands[0] if bands else []
        if len(visual_band) == expected_count:
            return [value[:4] for value in sorted(visual_band)]
        if len(visual_band) == expected_count + 1:
            median_y = median(value[1] for value in visual_band)
            median_h = median(value[3] for value in visual_band)
            worst = max(
                range(len(visual_band)),
                key=lambda index: (abs(visual_band[index][1] - median_y)
                                   + abs(visual_band[index][3] - median_h)),
            )
            return [value[:4] for index, value in enumerate(sorted(visual_band))
                    if index != worst]
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
            # Source glyph localization is also the most reliable music
            # baseline.  Detector clusters sometimes lock onto lyrics/footer
            # digits, which used to crop away volta lines above the real row.
            baseline = median(float(box[1]) for box in component_boxes)
            note_height = median(float(box[3]) for box in component_boxes)
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
    def _visual_curve_relations(
        image: Image.Image, crop_box: Sequence[int], pitch_tokens: List[str],
        note_positions: List[float], row_detections: List[Sequence[float]],
        baseline: float, note_height: float,
    ):
        """Recover printed arches from pixels and bind them to note endpoints.

        The current detector often classifies the digit itself as class 22/23.
        A real arch is instead a wide, sparse connected component whose two
        ends sit lower than its middle.  That shape test also rejects volta
        brackets and reduction lines before endpoint classification.
        """
        if len(note_positions) < 2:
            return []
        top, bottom = int(crop_box[1]), int(crop_box[3])
        band_top = max(top, int(baseline - note_height * 2.7))
        band_bottom = min(bottom, int(baseline - note_height * 0.55))
        if band_bottom - band_top < 5:
            return []
        gray = np.asarray(image.convert("L"))[band_top:band_bottom, :]
        binary = (gray < 150).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        relations = []
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            density = area / max(1, width * height)
            if (width < note_height * 1.25 or height < max(4, note_height * 0.18)
                    or density < 0.055 or density > 0.34):
                continue
            mask = labels[y:y + height, x:x + width] == component
            profile = []
            for column in range(width):
                ys = np.flatnonzero(mask[:, column])
                if ys.size:
                    profile.append(float(np.mean(ys)))
            if len(profile) < width * 0.55:
                continue
            edge_count = max(2, int(len(profile) * 0.18))
            edge_y = float(np.mean(profile[:edge_count] + profile[-edge_count:]))
            middle = len(profile) // 2
            middle_y = float(np.mean(profile[
                max(0, middle - edge_count // 2):middle + edge_count // 2 + 1]))
            if edge_y - middle_y < max(3.0, height * 0.18):
                continue

            pad = note_height * 0.32
            left, right = x - pad, x + width + pad
            covered = [index for index, position in enumerate(note_positions)
                       if left <= position <= right]
            if len(covered) < 2:
                nearest = sorted(range(len(note_positions)),
                                 key=lambda index: abs(note_positions[index] - (x + width / 2)))[:2]
                covered = sorted(nearest)
            if len(covered) < 2:
                continue
            start, end = covered[0], covered[-1]
            # Do not stretch a small ornament component to remote notes.
            if (abs(note_positions[start] - x) > note_height * 1.7
                    or abs(note_positions[end] - (x + width)) > note_height * 1.7):
                continue
            kind = ("triplet" if end - start == 2 and
                    AccurateVLMRecognizer._triplet_marker_visible(
                        start, end, note_positions, row_detections,
                        baseline, note_height)
                    else "tie" if pitch_tokens[start] == pitch_tokens[end]
                    else "slur")
            relation = (kind, start, end)
            if relation not in relations:
                relations.append(relation)
        return relations

    @staticmethod
    def _visual_parentheses(
        image: Image.Image, crop_box: Sequence[int], note_positions: List[float],
        baseline: float, note_height: float,
    ):
        """Find tall narrow '(' / ')' components and attach them to notes."""
        if not note_positions:
            return []
        top, bottom = int(crop_box[1]), int(crop_box[3])
        band_top = max(top, int(baseline - note_height * 0.9))
        band_bottom = min(bottom, int(baseline + note_height * 0.9))
        gray = np.asarray(image.convert("L"))[band_top:band_bottom, :]
        binary = (gray < 150).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        markers = []
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            density = area / max(1, width * height)
            if not (note_height * 0.20 <= width <= note_height * 0.72
                    and note_height * 1.20 <= height <= note_height * 1.85
                    and 0.10 <= density <= 0.42):
                continue
            mask = labels[y:y + height, x:x + width] == component
            means = []
            for low, high in ((0, height // 3), (height // 3, 2 * height // 3),
                              (2 * height // 3, height)):
                _, xs = np.where(mask[low:high])
                means.append(float(np.mean(xs)) if xs.size else width / 2)
            edge_mean = (means[0] + means[2]) / 2
            if abs(means[1] - edge_mean) < width * 0.13:
                continue
            center_x = x + width / 2
            if means[1] < edge_mean:
                candidates = [index for index, position in enumerate(note_positions)
                              if position > center_x]
                if not candidates:
                    continue
                note_index = min(candidates, key=lambda index: note_positions[index])
                side = "left"
            else:
                candidates = [index for index, position in enumerate(note_positions)
                              if position < center_x]
                if not candidates:
                    continue
                note_index = max(candidates, key=lambda index: note_positions[index])
                side = "right"
            # A parenthesis may sit outside an inline time signature or after
            # several augmentation dashes, so its nearest inside item is not
            # always immediately adjacent in pixels.
            if abs(note_positions[note_index] - center_x) > note_height * 3.5:
                continue
            marker = (side, note_index)
            if marker not in markers:
                markers.append(marker)
        return markers

    @staticmethod
    def _visual_boyin(
        image: Image.Image, crop_box: Sequence[int], note_positions: List[float],
        baseline: float, note_height: float,
    ):
        """Recover compact wave ornaments above individual jianpu notes.

        A bow/slur is a sparse, wide arc.  A local bo-yin glyph is much denser
        and occupies roughly one note width, so component density and height
        together provide a conservative pixel-only discriminator.
        """
        if not note_positions:
            return []
        top, bottom = int(crop_box[1]), int(crop_box[3])
        gray = np.asarray(image.convert("L"))[top:bottom, :]
        binary = (gray < 150).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        candidates = []
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            density = area / max(1, width * height)
            center_x = float(centroids[component][0])
            center_y = float(centroids[component][1] + top)
            if not (
                note_height * 0.40 <= width <= note_height * 1.25
                and note_height * 0.18 <= height <= note_height * 0.58
                and density >= 0.34
                and baseline - note_height * 1.75 <= center_y
                <= baseline - note_height * 0.32
            ):
                continue
            note_index = min(
                range(len(note_positions)),
                key=lambda index: abs(note_positions[index] - center_x),
            )
            if abs(note_positions[note_index] - center_x) > note_height * 0.70:
                continue
            candidates.append((note_index, density, area, center_x))

        # Keep one glyph per note; overlapping detector/component fragments are
        # common in scanned wave marks.
        selected = {}
        for note_index, density, area, center_x in candidates:
            old = selected.get(note_index)
            if old is None or (density, area) > (old[0], old[1]):
                selected[note_index] = (density, area, center_x)
        return sorted(selected)

    @staticmethod
    def _visual_lyric_slots(
        image: Image.Image, crop_box: Sequence[int], note_positions: List[float],
        baseline: float, note_height: float, lyric_lines, lyric_boxes=None,
    ):
        """Align VLM-transcribed lyric rows with glyph x coordinates."""
        clean_lines = []
        for line_index, line in enumerate(lyric_lines):
            if not isinstance(line, str) or not line.strip():
                continue
            text = "".join(line.split()).replace(",", "，").replace(".", "。")
            for prefix in ("D。S。", "DS。", "D。S", "DS"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            if (not text or any(marker in text for marker in
                                ("本谱", "声明", "软件制作", "中国曲谱网", "上传"))):
                continue
            source_box = (lyric_boxes[line_index]
                          if isinstance(lyric_boxes, list)
                          and line_index < len(lyric_boxes) else None)
            clean_lines.append((text, source_box))
        if not note_positions or not clean_lines:
            return []

        page_gray = np.asarray(image.convert("L"))
        regions = []
        if all(isinstance(box, list) and len(box) == 4 for _, box in clean_lines):
            for text, box in clean_lines:
                left = max(0, int(box[0]))
                top = max(0, int(box[1]))
                right = min(image.width, int(box[2]))
                bottom = min(image.height, int(box[3]))
                if right > left and bottom > top:
                    regions.append((text, (page_gray[top:bottom, left:right] < 150)
                                    .astype(np.uint8), left))
        else:
            top, bottom = int(crop_box[1]), int(crop_box[3])
            region_top = max(top, int(baseline + note_height * 1.55))
            if bottom - region_top < note_height * 0.6:
                return []
            binary = (page_gray[region_top:bottom, :] < 150).astype(np.uint8)
            row_threshold = max(6, round(image.width * 0.003))
            active_rows = np.flatnonzero(binary.sum(axis=1) >= row_threshold)
            row_groups = []
            for y in active_rows.tolist():
                if row_groups and y <= row_groups[-1][-1] + 3:
                    row_groups[-1].append(y)
                else:
                    row_groups.append([y])
            row_groups = [group for group in row_groups
                          if note_height * 0.65 <= group[-1] - group[0] + 1
                          <= note_height * 1.75
                          and group[-1] < binary.shape[0] - 3]
            for (text, _), group in zip(clean_lines, row_groups):
                low = max(0, group[0] - 2)
                high = min(binary.shape[0], group[-1] + 3)
                regions.append((text, binary[low:high], 0))
        if not regions:
            return []

        result = []
        for text, binary, x_offset in regions:
            active_columns = np.flatnonzero(binary.sum(axis=0) >= 2)
            column_groups = []
            max_internal_gap = max(4, round(note_height * 0.16))
            for x in active_columns.tolist():
                if column_groups and x <= column_groups[-1][-1] + max_internal_gap:
                    column_groups[-1].append(x)
                else:
                    column_groups.append([x])
            boxes = [(group_x[0], group_x[-1]) for group_x in column_groups
                     if group_x[-1] - group_x[0] + 1 >= note_height * 0.10]
            characters = list(text)
            # Printed verse numbers are extra leftmost components which the
            # text prompt intentionally omits.
            if len(boxes) > len(characters):
                boxes = boxes[-len(characters):]
            if not boxes or not characters:
                continue
            if len(boxes) != len(characters):
                # Preserve order under mild segmentation mismatch by sampling
                # the observed horizontal extent, rather than shifting every
                # later syllable by one note.
                centers = np.linspace(
                    (boxes[0][0] + boxes[0][1]) / 2,
                    (boxes[-1][0] + boxes[-1][1]) / 2,
                    len(characters)).tolist()
            else:
                centers = [(left + right) / 2 for left, right in boxes]
            centers = [center + x_offset for center in centers]
            slots = [""] * len(note_positions)
            previous_index = None
            for character, center_x in zip(characters, centers):
                if character in "，。！？；：、,.!?;:" and previous_index is not None:
                    slots[previous_index] += character
                    continue
                note_index = min(range(len(note_positions)),
                                 key=lambda index: abs(note_positions[index] - center_x))
                if abs(note_positions[note_index] - center_x) > note_height * 1.7:
                    continue
                slots[note_index] = (slots[note_index] + character)
                previous_index = note_index
            if any(slots):
                result.append(slots)
        return result

    @staticmethod
    def _time_signature_visible(
        image: Image.Image, crop_box: Sequence[int], note_x: float,
        baseline: float, note_height: float,
    ) -> bool:
        """Require the short fraction rule immediately before a claimed meter."""
        top, bottom = int(crop_box[1]), int(crop_box[3])
        left = max(0, int(note_x - note_height * 4.0))
        right = max(left + 1, int(note_x - note_height * 0.35))
        y0 = max(top, int(baseline - note_height * 0.35))
        y1 = min(bottom, int(baseline + note_height * 0.35))
        binary = (np.asarray(image.convert("L"))[y0:y1, left:right] < 150).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        return any(
            note_height * 0.55 <= int(stats[index][2]) <= note_height * 1.55
            and int(stats[index][3]) <= max(7, note_height * 0.22)
            and int(stats[index][2]) / max(1, int(stats[index][3])) >= 3.0
            for index in range(1, count)
        )

    @staticmethod
    def _visual_time_signatures(
        image: Image.Image, crop_box: Sequence[int], note_positions: List[float],
        baseline: float, note_height: float, claimed_signatures,
    ):
        """Locate every stacked meter fraction and bind it to the next note."""
        valid = [item for item in claimed_signatures if isinstance(item, dict)
                 and isinstance(item.get("numerator"), int)
                 and isinstance(item.get("denominator"), int)]
        if not valid or not note_positions:
            return []
        top, bottom = int(crop_box[1]), int(crop_box[3])
        y0 = max(top, int(baseline - note_height * 0.40))
        y1 = min(bottom, int(baseline + note_height * 0.40))
        binary = (np.asarray(image.convert("L"))[y0:y1, :] < 150).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        centers = []
        for index in range(1, count):
            x, _, width, height, _ = map(int, stats[index])
            if (note_height * 1.0 <= width <= note_height * 1.55
                    and height <= max(7, note_height * 0.20)
                    and width / max(1, height) >= 5.0):
                centers.append(x + width / 2)
        result = []
        for meter_index, center_x in enumerate(sorted(centers)):
            candidates = [index for index, note_x in enumerate(note_positions)
                          if note_x > center_x]
            if not candidates:
                continue
            before_note = min(candidates, key=lambda index: note_positions[index])
            if note_positions[before_note] - center_x > note_height * 3.5:
                continue
            signature = valid[min(meter_index, len(valid) - 1)]
            marker = (before_note, signature["numerator"], signature["denominator"])
            if marker not in result:
                result.append(marker)
        return result

    @staticmethod
    def _visual_repeat_endings(
        image: Image.Image, crop_box: Sequence[int], baseline: float,
        note_height: float, bars: List[tuple[str, float]],
    ):
        """Locate volta brackets by matching their horizontal span to bars."""
        if len(bars) < 2:
            return []
        top, bottom = int(crop_box[1]), int(crop_box[3])
        band_top = max(top, int(baseline - note_height * 3.0))
        band_bottom = min(bottom, int(baseline - note_height * 0.7))
        gray = np.asarray(image.convert("L"))[band_top:band_bottom, :]
        binary = (gray < 160).astype(np.uint8)
        kernel_width = max(12, int(note_height * 4.0))
        horizontal = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)))
        count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, 8)
        candidates = []
        bar_x = [float(item[1]) for item in bars]
        for component in range(1, count):
            x, y, width, height, _ = map(int, stats[component])
            if width < note_height * 4.0 or height > max(6, note_height * 0.25):
                continue
            right = x + width
            left_index = min(range(len(bar_x)), key=lambda index: abs(bar_x[index] - x))
            right_index = min(range(len(bar_x)), key=lambda index: abs(bar_x[index] - right))
            if (right_index <= left_index
                    or abs(bar_x[left_index] - x) > note_height * 0.75
                    or abs(bar_x[right_index] - right) > note_height * 0.75):
                continue
            candidates.append((x, y + band_top, width, left_index + 1))

        endings = []
        for ordinal, (x, line_y, width, measure_index) in enumerate(sorted(candidates)):
            # Count printed ending digits just inside the bracket.  The exact
            # glyph classifier is unnecessary here: ordering supplies 1/2 and
            # two visible glyphs on the second bracket represent "2.3.".
            roi_left = int(x + note_height * 0.25)
            roi_right = int(min(image.width, x + note_height * 2.8))
            roi_top = max(top, int(line_y - note_height * 0.15))
            roi_bottom = min(bottom, int(line_y + note_height * 1.25))
            roi = (np.asarray(image.convert("L"))[roi_top:roi_bottom,
                                                   roi_left:roi_right] < 150).astype(np.uint8)
            glyph_count, _, glyph_stats, _ = cv2.connectedComponentsWithStats(roi, 8)
            digits = []
            for glyph in range(1, glyph_count):
                glyph_x, _, glyph_width, glyph_height, glyph_area = map(
                    int, glyph_stats[glyph])
                density = glyph_area / max(1, glyph_width * glyph_height)
                if (note_height * 0.32 <= glyph_height <= note_height * 0.95
                        and note_height * 0.12 <= glyph_width <= note_height
                        and 0.10 <= density <= 0.75):
                    # The scanned italic ending font has stable width ratios:
                    # 1 is narrow, 2 is broad, and 3 lies between them.  This
                    # tiny OCR rule distinguishes "1.", "2.3." and later "3."
                    # without another expensive VLM request.
                    ratio = glyph_width / max(1, glyph_height)
                    value = 1 if ratio < 0.72 else 2 if ratio > 0.82 else 3
                    digits.append((glyph_x, value))
            numbers = []
            for _, value in sorted(digits):
                if value not in numbers:
                    numbers.append(value)
            if not numbers:
                numbers = [ordinal + 1]
            endings.append((measure_index, numbers))
        return endings

    @staticmethod
    def _inject_repeat_endings(tokens: List[str], endings):
        if not endings:
            return tokens
        by_measure = {index: numbers for index, numbers in endings}
        output = []
        measure_index = 0
        inserted = set()
        for token in tokens:
            if (measure_index in by_measure and measure_index not in inserted
                    and (token.startswith("P") or token in {"R0", "-"})):
                output.extend(f"R{number}" for number in by_measure[measure_index])
                inserted.add(measure_index)
            output.append(token)
            if token in BAR_CLASSES.values():
                measure_index += 1
        return output

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
    def _decorate_parentheses(score: Dict[str, Any], markers):
        notes = [item for measure in score.get("measures", [])
                 for item in measure.get("notes", [])]
        for side, note_index in markers:
            if 0 <= note_index < len(notes):
                notes[note_index]["parenthesisLeft" if side == "left"
                                  else "parenthesisRight"] = True

    @staticmethod
    def _decorate_text_layers(score: Dict[str, Any], lyric_layers, time_signatures):
        pitch_notes = [item for measure in score.get("measures", [])
                       for item in measure.get("notes", []) if "pitch" in item]
        for note_offset, lines in lyric_layers:
            if not isinstance(lines, list):
                continue
            for local_index in range(max((len(line) for line in lines
                                          if isinstance(line, list)), default=0)):
                values = []
                for line in lines:
                    if not isinstance(line, list) or local_index >= len(line):
                        continue
                    value = line[local_index]
                    if isinstance(value, str) and value.strip():
                        values.append(value.strip())
                global_index = note_offset + local_index
                if values and 0 <= global_index < len(pitch_notes):
                    pitch_notes[global_index]["lyrics"] = values

        note_to_measure = []
        for measure_index, measure in enumerate(score.get("measures", [])):
            note_to_measure.extend(
                measure_index for item in measure.get("notes", []) if "pitch" in item)
        for note_index, numerator, denominator in time_signatures:
            if (0 <= note_index < len(note_to_measure)
                    and numerator > 0 and denominator > 0):
                score["measures"][note_to_measure[note_index]]["timeSignature"] = {
                    "numerator": numerator, "denominator": denominator,
                }

    @staticmethod
    def _decorate_ornaments(score: Dict[str, Any], ornaments):
        notes = [item for measure in score.get("measures", [])
                 for item in measure.get("notes", []) if "pitch" in item]
        for note_index, ornament in ornaments:
            if not (0 <= note_index < len(notes) and isinstance(ornament, dict)):
                continue
            if ornament.get("type") == "boyin":
                technique = {"type": "boyin"}
                techniques = notes[note_index].setdefault("techniques", [])
                if technique not in techniques:
                    techniques.append(technique)
                continue
            if ornament.get("type") != "yinyin":
                continue
            grace_notes = ornament.get("grace_notes", [])
            if (not isinstance(grace_notes, list)
                    or not 1 <= len(grace_notes) <= 2
                    or not all(isinstance(value, int) and 1 <= value <= 7
                               for value in grace_notes)):
                continue
            technique = {
                "type": "yinyin", "graceNotes": grace_notes, "graceOctave": 0,
            }
            techniques = notes[note_index].setdefault("techniques", [])
            if technique not in techniques:
                techniques.append(technique)

    @staticmethod
    def _decorate_navigation(score: Dict[str, Any], navigation_marks):
        if not isinstance(navigation_marks, list):
            return
        notes_by_measure = []
        for measure_index, measure in enumerate(score.get("measures", [])):
            for item in measure.get("notes", []):
                if "pitch" in item:
                    notes_by_measure.append(measure_index)
        allowed = {"ds", "dc", "fine", "segno", "coda"}
        for entry in navigation_marks:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                    or not isinstance(entry[0], int)):
                continue
            note_index, mark = entry
            if not isinstance(mark, dict) or mark.get("type") not in allowed:
                continue
            if not notes_by_measure:
                continue
            note_index = min(max(0, int(note_index)), len(notes_by_measure) - 1)
            measure = score["measures"][notes_by_measure[note_index]]
            navigation = measure.setdefault("navigationMarks", [])
            clean = {"type": mark["type"]}
            if isinstance(mark.get("text"), str) and mark["text"].strip():
                clean["text"] = mark["text"].strip()
            if clean not in navigation:
                navigation.append(clean)

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
        # Geometry-validated ties/slurs are safe to keep.  Preserve the explicit
        # type here; changing every repeated three-note slur into a tuplet loses
        # legitimate phrase arcs when no printed 3 is present.
        result = []
        for relation in relations:
            if relation not in result:
                result.append(relation)
        return result

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
        parentheses = []
        lyric_layers = []
        time_signatures = []
        ornaments = []
        navigation_marks = []
        note_offset = 0
        item_offset = 0
        for row in payload.get("rows", []):
            if row.get("content_type") != "score":
                continue
            confidences.append(float(row.get("confidence", 0.0)))
            for voice_index, voice in enumerate(row.get("voices", [])):
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
                preferred_item_boxes = []
                if token_geometry:
                    for token, geometry in zip(voice_tokens, token_geometry):
                        box = geometry.get("box") if isinstance(geometry, dict) else None
                        if ((token.startswith("P") or token in {"R0", "-"})
                                and isinstance(box, list) and len(box) == 4):
                            preferred_item_boxes.append(box)
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
                if voice_index == 0:
                    row_navigation_marks = voice.get("navigation_marks", [])
                    if not isinstance(row_navigation_marks, list):
                        row_navigation_marks = []
                    for mark in row_navigation_marks:
                        if (isinstance(mark, dict)
                                and isinstance(mark.get("before_note"), int)
                                and 0 <= mark["before_note"] <= len(pitch_tokens)):
                            navigation_marks.append((
                                note_offset + mark["before_note"], mark))
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
                row_relations.extend(self._visual_curve_relations(
                    image, row["crop_box"], pitch_tokens, positions,
                    row_detections, baseline, note_height))
                row_relations = self._production_relations(row_relations)
                row_relations = [
                    relation for relation in row_relations
                    if relation[0] != "triplet" or self._triplet_marker_visible(
                        relation[1], relation[2], positions, row_detections,
                        baseline, note_height)
                ]
                for kind, start, end in row_relations:
                    relations.append((kind, note_offset + start, note_offset + end))
                row_items = [token for token in merged
                             if token.startswith("P") or token in {"R0", "-"}]
                if len(preferred_item_boxes) == len(row_items):
                    item_positions = [float(box[0]) for box in preferred_item_boxes]
                    item_markers = self._visual_parentheses(
                        image, row["crop_box"], item_positions, baseline, note_height)
                else:
                    pitch_to_item = [index for index, token in enumerate(row_items)
                                     if token.startswith("P") or token == "R0"]
                    item_markers = [
                        (side, pitch_to_item[index])
                        for side, index in self._visual_parentheses(
                            image, row["crop_box"], positions, baseline, note_height)
                        if index < len(pitch_to_item)
                    ]
                for side, item_index in item_markers:
                    parentheses.append((side, item_offset + item_index))
                if voice_index == 0:
                    text_layer = row.get("text_layer", {})
                    if isinstance(text_layer, dict):
                        lyric_layers.append((note_offset, self._visual_lyric_slots(
                            image, row["crop_box"], positions, baseline, note_height,
                            text_layer.get("lyric_lines", []),
                            text_layer.get("lyric_boxes", []))))
                        for before_note, numerator, denominator in self._visual_time_signatures(
                                image, row["crop_box"], positions, baseline, note_height,
                                text_layer.get("time_signatures", [])):
                            time_signatures.append((
                                note_offset + before_note, numerator, denominator))
                        for ornament in text_layer.get("ornaments", []):
                            if (isinstance(ornament, dict)
                                    and isinstance(ornament.get("note"), int)
                                    and 0 <= ornament["note"] < len(pitch_tokens)):
                                ornaments.append((note_offset + ornament["note"], ornament))
                    for boyin_note in self._visual_boyin(
                            image, row["crop_box"], positions, baseline, note_height):
                        ornaments.append((
                            note_offset + boyin_note, {"type": "boyin"}))
                note_offset += len(pitch_tokens)
                item_offset += len(row_items)
                pixel_bars = self._pixel_bars(
                    image, row["crop_box"], baseline, note_height)
                decorated = self._inject_repeat_endings(
                    decorated, self._visual_repeat_endings(
                        image, row["crop_box"], baseline, note_height, pixel_bars))
                tokens.extend(decorated)
                if decorated and not any(token.startswith("B") for token in decorated[-2:]):
                    tokens.append("B|")
                tokens.append("<ROW>")
        if not any(token.startswith("P") or token == "R0" for token in tokens):
            raise RuntimeError("本地视觉大模型没有在图片中找到数字简谱")
        score = parse_tokens_to_score(tokens)
        self._decorate_relations(score, relations)
        self._decorate_parentheses(score, parentheses)
        self._decorate_text_layers(score, lyric_layers, time_signatures)
        self._decorate_ornaments(score, ornaments)
        self._decorate_navigation(score, navigation_marks)
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            title = metadata.get("title")
            key = metadata.get("key")
            signature = metadata.get("time_signature")
            tempo = metadata.get("tempo")
            tempo_text = metadata.get("tempo_text")
            if isinstance(title, str) and title.strip():
                score["title"] = title.strip()
            if isinstance(key, str) and key.strip():
                score["key"] = key.strip()
            if (isinstance(signature, dict)
                    and isinstance(signature.get("numerator"), int)
                    and isinstance(signature.get("denominator"), int)):
                score["timeSignature"] = signature
            if isinstance(tempo, int) and tempo > 0:
                score["tempo"] = tempo
            if isinstance(tempo_text, str) and tempo_text.strip():
                score["tempoText"] = tempo_text.strip()
        score.setdefault("title", "识别结果")
        score_notes = [item for measure in score.get("measures", [])
                       for item in measure.get("notes", []) if "pitch" in item]
        score_items = [item for measure in score.get("measures", [])
                       for item in measure.get("notes", [])]
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
            "parentheses": sum(bool(item.get("parenthesisLeft"))
                               + bool(item.get("parenthesisRight")) for item in score_items),
            "repeat_endings": sum(bool(measure.get("repeatEnding"))
                                  for measure in score.get("measures", [])),
            "lyric_syllables": sum(len(note.get("lyrics", [])) for note in score_notes),
            "local_time_signatures": sum(bool(measure.get("timeSignature"))
                                         for measure in score.get("measures", [])),
            "grace_notes": sum(
                any(technique.get("type") == "yinyin"
                    for technique in note.get("techniques", []))
                for note in score_notes),
            "boyin": sum(
                any(technique.get("type") == "boyin"
                    for technique in note.get("techniques", []))
                for note in score_notes),
            "navigation_marks": sum(
                len(measure.get("navigationMarks", []))
                for measure in score.get("measures", [])),
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
