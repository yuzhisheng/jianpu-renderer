#!/usr/bin/env python3
"""Recognize a jianpu page as a conservative pitch/bar skeleton with MLX VLM.

The script is intentionally self-contained so it can run in ``.venv-vlm``.
Its stdout may contain messages from MLX; callers should parse the final line
prefixed with ``__JIANPU_RESULT__``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from mlx_vlm import batch_generate, generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from annotate_rows_with_local_vlm import (
    DEFAULT_MODEL, PROMPT, RELATION_PROMPT, parse_json,
)


RESULT_PREFIX = "__JIANPU_RESULT__"
GLYPH_PROMPT = """图中是一行从左到右等间距排列的简谱字形单元。
每个单元只允许分类为0,1,2,3,4,5,6,7、-或X；-表示增时短横线，X表示括号、小节线、装饰或非数字。
必须按可见顺序输出，不能根据旋律补写、合并或重排。严格只返回JSON：{"symbols":["5","-","X"]}。
"""

METADATA_PROMPT = """你是中文简谱页眉标注器。只读取图片中明确可见的歌曲标题、调号、初始拍号、速度和速度文字。
严格只返回JSON，不要Markdown：
{"title":"","key":"","time_signature":{"numerator":4,"denominator":4},"tempo":null,"tempo_text":""}
规则：key只写等号右边，例如1=♭E写♭E；没有把握的字段用空字符串或null；不要把词曲作者、软件声明当标题。
"""

METER_PROMPT = """你是简谱局部拍号标注器。图中主旋律共有{note_count}个大号数字音符（增时横线不计数）。
严格只返回JSON，不要Markdown：
{{"time_signatures":[{{"before_note":0,"numerator":2,"denominator":4}}]}}
规则：
1. 局部拍号只记录谱行内明确可见的上下分数，并用before_note指向它后面的第一个大号音符；没有局部拍号返回空数组。
2. 小房子编号、歌词行号和倚音小数字不是拍号。不要猜。
"""

LYRIC_LINE_PROMPT = """图中只有一条中文简谱歌词。请从左到右完整转写所有歌词文字。
严格只返回JSON，不要Markdown：{"text":"我的心爱在天边，天边有一片"}
去掉开头的歌词段落编号“1.”“2.”“3.”和字符间空格；保留逗号、句号；不要抄写谱面数字或根据歌曲常识补字。
"""

GRACE_NOTE_PROMPT = """图中是简谱倚音记号的局部放大图。只读取双横线上方明显可见的一个或两个小号数字，
按从左到右顺序返回；圆点、横线、弧线和右侧的大号主音都不要读取。
严格只返回JSON，不要Markdown：{"grace_notes":[1,2]}。数字只能是1到7；看不清返回空数组，禁止猜测。
"""


def layout_bands(image: Image.Image):
    """Return horizontal ink bands without relying on detector classes."""
    width, height = image.size
    normalized_width = 1600
    normalized_height = max(1, round(height * normalized_width / width))
    gray = np.asarray(image.convert("L").resize(
        (normalized_width, normalized_height), Image.Resampling.BILINEAR))
    active = np.flatnonzero((gray < 160).mean(axis=1) > 0.008)
    groups = []
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


def fixed_canvas(image: Image.Image, width: int = 1600, height: int = 640):
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def generate_many(model, processor, images, prompts, max_tokens):
    """Generate a list of responses without triggering MLX batch-shape bugs.

    The local MLX version bundled with this project occasionally raises an
    ``IndexError`` in ``batch_generate`` for mixed crop widths.  Sequential
    generation is the safe default; batching remains an explicit opt-in for
    environments where that MLX bug is fixed.
    """
    if os.environ.get("JIANPU_USE_MLX_BATCH", "0") == "1":
        try:
            response = batch_generate(
                model, processor, images=images, prompts=prompts,
                max_tokens=max_tokens, temperature=0.0, verbose=False,
                group_by_shape=True,
            )
            return response.texts
        except Exception:
            pass
    return [
        generate(
            model, processor, prompt, image=image,
            max_tokens=max_tokens, temperature=0.0, verbose=False,
        ).text
        for image, prompt in zip(images, prompts)
    ]


def estimate_main_digit_geometry(image: Image.Image):
    """Estimate digit count, baseline and glyph height on the music row."""
    gray = np.asarray(image.convert("L"))
    binary = (gray < 150).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        center_x, center_y = centroids[index]
        if (4 <= width <= 42 and 16 <= height <= 52
                and 0.10 <= width / max(1, height) <= 1.45
                and area >= 25 and center_y <= image.height * 0.68):
            candidates.append((float(center_x), float(center_y), width, height, area))
    if not candidates:
        return 0, image.height / 2, max(16.0, image.height * 0.22)
    typical_height = float(np.median([item[3] for item in candidates]))
    radius = max(9.0, typical_height * 0.48)
    # Music is the uppermost dense glyph baseline in a projection band; lyric
    # lines below it can contain more connected components and must not win by
    # raw density alone.
    baseline = None
    for seed in sorted({item[1] for item in candidates}):
        cluster_y = [item[1] for item in candidates if abs(item[1] - seed) <= radius]
        if len(cluster_y) >= 6:
            baseline = float(np.median(cluster_y))
            break
    if baseline is None:
        baseline = min(item[1] for item in candidates)
    row = [item for item in candidates if abs(item[1] - baseline) <= radius]
    return len(row), baseline, typical_height


def estimate_main_digit_count(image: Image.Image):
    """Estimate digit components on the uppermost dense music baseline."""
    count, baseline, _ = estimate_main_digit_geometry(image)
    return count, baseline


def count_digits_near_baseline(image: Image.Image, baseline: float):
    """Count digit glyphs in a measure using its already-known row baseline."""
    gray = np.asarray(image.convert("L"))
    binary = (gray < 150).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        _, center_y = centroids[index]
        if (4 <= width <= 42 and 16 <= height <= 52
                and 0.10 <= width / max(1, height) <= 1.45
                and area >= 25):
            candidates.append((float(center_y), float(height)))
    if not candidates:
        return 0
    typical_height = float(np.median([item[1] for item in candidates]))
    radius = max(9.0, typical_height * 0.48)
    return sum(abs(center_y - baseline) <= radius for center_y, _ in candidates)


def grace_note_candidates(page: Image.Image, crop_box, row):
    """Locate the distinctive double rule below one/two small grace digits."""
    voice = row.get("voices", [{}])[0]
    note_boxes = [geometry.get("box") for token, geometry in zip(
        voice.get("tokens", []), voice.get("token_geometry", []))
        if (token.startswith("P") or token == "R0")
        and isinstance(geometry, dict)
        and isinstance(geometry.get("box"), list)
        and len(geometry["box"]) >= 4]
    if not note_boxes:
        return []
    top, bottom = int(crop_box[1]), int(crop_box[3])
    note_height = float(np.median([box[3] for box in note_boxes]))
    baseline = float(np.median([box[1] for box in note_boxes]))
    binary = (np.asarray(page.convert("L"))[top:bottom, :] < 150).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    lines = []
    for component in range(1, count):
        x, _, width, height, _ = map(int, stats[component])
        center_y = float(centroids[component][1] + top)
        if (note_height * 0.35 <= width <= note_height * 1.7
                and height <= max(5, note_height * 0.15)
                and width / max(1, height) >= 5.0
                and baseline - note_height * 0.48 <= center_y
                <= baseline - note_height * 0.05):
            lines.append((x, x + width, center_y))

    candidates = []
    used = set()
    for first_index, first in enumerate(lines):
        for second_index, second in enumerate(lines[first_index + 1:], first_index + 1):
            if first_index in used or second_index in used:
                continue
            if (abs(first[0] - second[0]) > 2
                    or abs(first[1] - second[1]) > 2
                    or not 3 <= abs(first[2] - second[2]) <= note_height * 0.25):
                continue
            left, right = min(first[0], second[0]), max(first[1], second[1])
            center_x = (left + right) / 2
            following = [note_index for note_index, box in enumerate(note_boxes)
                         if float(box[0]) - float(box[2]) / 2 > right - 2]
            if not following:
                continue
            note_index = min(following, key=lambda value: float(note_boxes[value][0]))
            if float(note_boxes[note_index][0]) - center_x > note_height * 1.85:
                continue
            line_y = min(first[2], second[2])
            crop = [
                max(0, int(left - 5)),
                max(top, int(line_y - note_height * 1.35)),
                min(page.width, int(right + 5)),
                min(bottom, int(line_y - 2)),
            ]
            if crop[2] > crop[0] and crop[3] > crop[1]:
                candidates.append((note_index, crop))
                used.update((first_index, second_index))
    return candidates


def split_near_whitespace(image: Image.Image, baseline: float):
    """Choose a center split that avoids digits, underlines and barlines."""
    gray = np.asarray(image.convert("L"))
    half_height = max(22, round(image.height * 0.18))
    top = max(0, round(baseline - half_height))
    bottom = min(image.height, round(baseline + half_height))
    ink = (gray[top:bottom] < 170).sum(axis=0).astype(float)
    if len(ink) >= 9:
        ink = np.convolve(ink, np.ones(9), mode="same")
    left, right = round(image.width * 0.42), round(image.width * 0.58)
    center = image.width / 2
    return min(range(left, right), key=lambda x: (ink[x], abs(x - center)))


def pitch_count(parsed):
    return sum(
        token.startswith("P") or token == "R0"
        for voice in parsed.get("voices", [])
        for token in voice.get("tokens", [])
    )


def replace_pitch_stream(original, replacement):
    """Replace pitch/rest identities while preserving row punctuation tokens."""
    replacement_notes = [
        token for token in replacement if token.startswith("P") or token == "R0"
    ]


def glyph_cells(image: Image.Image, baseline: float):
    """Segment baseline glyphs while excluding dots, bars and stacked fractions."""
    gray = np.asarray(image.convert("L"))
    top, bottom = max(0, round(baseline - 18)), min(image.height, round(baseline + 19))
    active = np.flatnonzero((gray[top:bottom] < 150).sum(axis=0) >= 3)
    groups = []
    for x in active.tolist():
        if groups and x <= groups[-1][-1] + 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    result = []
    for group in groups:
        width = group[-1] - group[0] + 1
        # Dots and barlines are narrower than 12px. A stacked 2/4 or 4/4
        # fraction merges with its fraction rule and is wider than 38px.
        if 12 <= width <= 38:
            result.append((max(0, group[0] - 4), top,
                           min(image.width, group[-1] + 5), bottom))
    return result


def glyph_strip(image: Image.Image, cells, width: int = 1600, height: int = 200):
    canvas = Image.new("RGB", (width, height), "white")
    if not cells:
        return canvas
    spacing = min(64, max(36, (width - 16) // len(cells)))
    for index, box in enumerate(cells):
        glyph = image.crop(box)
        scale = min(52 / glyph.width, 70 / glyph.height)
        glyph = glyph.resize((max(1, round(glyph.width * scale)),
                              max(1, round(glyph.height * scale))),
                             Image.Resampling.LANCZOS)
        x = 8 + index * spacing + (spacing - glyph.width) // 2
        y = 60 + (70 - glyph.height) // 2
        if x + glyph.width <= width:
            canvas.paste(glyph, (x, y))
    return canvas


def glyph_cell_is_dash(image: Image.Image, box):
    gray = np.asarray(image.convert("L").crop(box))
    points = np.argwhere(gray < 150)
    if not len(points):
        return False
    height = int(points[:, 0].max() - points[:, 0].min() + 1)
    width = int(points[:, 1].max() - points[:, 1].min() + 1)
    return height <= 9 and width >= 12 and width / max(1, height) >= 2.2


def classify_score_glyphs(
    page, bands, rows, model, processor, glyph_prompt, max_tokens,
):
    """Classify localized baseline glyphs without lyrics/layout interference."""
    for index, parsed in enumerate(rows):
        if parsed.get("content_type") != "score" or not parsed.get("voices"):
            continue
        top, bottom = bands[index]
        row_image = page.crop((0, top, page.width, bottom))
        estimated, baseline = estimate_main_digit_count(row_image)
        cells = glyph_cells(row_image, baseline)
        if estimated < 4 or not cells:
            continue
        try:
            raw = generate(
                model, processor, glyph_prompt, image=glyph_strip(row_image, cells),
                max_tokens=min(max_tokens, 256), temperature=0.0, verbose=False,
            ).text
            payload = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            symbols = payload.get("symbols", [])
        except Exception:
            continue
        tokens = []
        token_geometry = []
        aligned = len(symbols) == len(cells)
        for symbol_index, symbol in enumerate(symbols):
            symbol = str(symbol).strip()
            visual_dash = (aligned and symbol_index < len(cells)
                           and glyph_cell_is_dash(row_image, cells[symbol_index]))
            token = None
            if visual_dash:
                token = "-"
            elif symbol in "1234567" and len(symbol) == 1:
                token = f"P{symbol}"
            elif symbol == "0":
                token = "R0"
            elif not aligned and symbol in {"-", "—", "–"}:
                token = "-"
            if token is None:
                continue
            tokens.append(token)
            if aligned and symbol_index < len(cells):
                left, cell_top, right, cell_bottom = cells[symbol_index]
                token_geometry.append({
                    "x": (left + right) / 2,
                    "box": [
                        (left + right) / 2,
                        top + (cell_top + cell_bottom) / 2,
                        right - left,
                        cell_bottom - cell_top,
                    ],
                })
        classified_count = sum(token.startswith("P") or token == "R0" for token in tokens)
        original_count = pitch_count(parsed)
        if (classified_count >= 4
                and (abs(classified_count - estimated) <= 1
                     or classified_count == original_count)):
            voice = parsed["voices"][0]
            voice["tokens"] = tokens
            # Keep the exact source x/box for every emitted glyph.  Rebuilding
            # these positions later by uniform interpolation shifts octave dots,
            # reduction lines and barlines onto neighbouring notes.
            if len(token_geometry) == len(tokens):
                voice["token_geometry"] = token_geometry
            voice["relations"] = []
            voice["modifiers"] = []
            parsed.setdefault("uncertainties", []).append(
                f"localized glyph classification: {original_count}->{classified_count}, "
                f"visual estimate {estimated}")
    return rows


def read_text_layers(
    page, bands, rows, model, processor, config, max_tokens, batch_size,
):
    """Read metadata and lyric alignment without perturbing pitch decoding."""
    score_indices = [index for index, row in enumerate(rows)
                     if row.get("content_type") == "score" and row.get("voices")]
    first_score_top = bands[score_indices[0]][0] if score_indices else round(page.height * 0.25)
    metadata_image = fixed_canvas(page.crop((0, 0, page.width, max(32, first_score_top))))
    metadata_prompt = apply_chat_template(
        processor, config, METADATA_PROMPT, num_images=1, thinking_mode="disabled")
    metadata = {}
    try:
        raw = generate(
            model, processor, metadata_prompt, image=metadata_image,
            max_tokens=min(max_tokens, 256), temperature=0.0, verbose=False,
        ).text
        metadata = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception as exc:
        metadata = {"uncertainty": f"metadata pass failed: {exc}"}

    lyric_prompt = apply_chat_template(
        processor, config, LYRIC_LINE_PROMPT,
        num_images=1, thinking_mode="disabled")
    grace_prompt = apply_chat_template(
        processor, config, GRACE_NOTE_PROMPT,
        num_images=1, thinking_mode="disabled")
    tasks = []
    lyric_line_counts = {index: 0 for index in score_indices}
    for index in score_indices:
        rows[index]["text_layer"] = {
            "lyric_lines": [], "lyric_boxes": [], "time_signatures": [],
            "ornaments": [],
        }

    for index in score_indices:
        note_count = pitch_count(rows[index])
        if note_count <= 0:
            continue
        top, bottom = bands[index]
        meter_prompt = apply_chat_template(
            processor, config, METER_PROMPT.format(
                note_count=note_count, last_note=note_count - 1),
            num_images=1, thinking_mode="disabled")
        tasks.append(("meter", index, 0,
                      fixed_canvas(page.crop((0, top, page.width, bottom))),
                      meter_prompt, note_count, None))
        for note_index, grace_box in grace_note_candidates(
                page, [0, top, page.width, bottom], rows[index]):
            tasks.append((
                "grace", index, note_index,
                fixed_canvas(page.crop(tuple(grace_box)), width=400, height=240),
                grace_prompt, note_count, grace_box,
            ))

        voice = rows[index].get("voices", [{}])[0]
        note_boxes = [geometry.get("box") for token, geometry in zip(
            voice.get("tokens", []), voice.get("token_geometry", []))
            if (token.startswith("P") or token == "R0")
            and isinstance(geometry, dict) and isinstance(geometry.get("box"), list)]
        row_image = page.crop((0, top, page.width, bottom))
        if note_boxes:
            baseline = float(np.median([box[1] for box in note_boxes])) - top
            note_height = float(np.median([box[3] for box in note_boxes]))
        else:
            # The pitch pass can be correct while omitting token_geometry.
            # Do not drop lyrics in that case: recover the score baseline from
            # connected components and continue with the same lyric projection.
            estimated, baseline, note_height = estimate_main_digit_geometry(row_image)
            if estimated < max(3, min(note_count, 6) * 0.45):
                continue
        # Detector/VLM row bands often end on the last lyric glyph.  Give the
        # projection one small, scale-aware safety margin so a line touching
        # the crop bottom is not rejected as an incomplete border group.  The
        # margin is much smaller than the distance to the next score baseline,
        # so it cannot absorb the following music row.
        context_bottom = min(
            page.height, bottom + max(8, round(note_height * 0.60)))
        if context_bottom > bottom:
            row_image = page.crop((0, top, page.width, context_bottom))
        gray = np.asarray(row_image.convert("L"))
        region_top = max(0, int(baseline + note_height * 1.55))
        binary = (gray[region_top:, :] < 150).astype(np.uint8)
        active = np.flatnonzero(binary.sum(axis=1) >= max(6, round(page.width * 0.003)))
        line_groups = []
        for y in active.tolist():
            if line_groups and y <= line_groups[-1][-1] + 3:
                line_groups[-1].append(y)
            else:
                line_groups.append([y])
        line_groups = [group for group in line_groups
                       if note_height * 0.65 <= group[-1] - group[0] + 1
                       <= note_height * 1.75
                       and group[-1] < binary.shape[0] - 3]
        for group in line_groups:
            y0 = max(0, region_top + group[0] - 8)
            y1 = min(row_image.height, region_top + group[-1] + 9)
            line_crop = row_image.crop((0, y0, row_image.width, y1))
            line_index = lyric_line_counts[index]
            lyric_line_counts[index] += 1
            tasks.append(("lyric", index, line_index,
                          fixed_canvas(line_crop, height=200), lyric_prompt, note_count,
                          [0, top + y0, page.width, top + y1]))

    # A page segmenter may put the lyric baseline in its own band instead of
    # keeping it inside the preceding score crop.  Such a band still belongs to
    # the closest score row above it; otherwise an entire lyric phrase silently
    # disappears from the structured score.
    for index, row in enumerate(rows):
        if row.get("content_type") != "lyrics":
            continue
        target = next((score_index for score_index in reversed(score_indices)
                       if score_index < index), None)
        if target is None:
            continue
        # Do not bridge across a later score row when malformed/reordered input
        # contains several independent page regions.
        if any(score_index for score_index in score_indices
               if target < score_index < index):
            continue
        note_count = pitch_count(rows[target])
        if note_count <= 0:
            continue
        top, bottom = bands[index]
        row_image = page.crop((0, top, page.width, bottom))
        gray = np.asarray(row_image.convert("L"))
        binary = (gray < 150).astype(np.uint8)
        active = np.flatnonzero(binary.sum(axis=1) >= max(6, round(page.width * 0.003)))
        line_groups = []
        for y in active.tolist():
            if line_groups and y <= line_groups[-1][-1] + 3:
                line_groups[-1].append(y)
            else:
                line_groups.append([y])

        voice = rows[target].get("voices", [{}])[0]
        heights = [float(geometry["box"][3]) for token, geometry in zip(
            voice.get("tokens", []), voice.get("token_geometry", []))
            if (token.startswith("P") or token == "R0")
            and isinstance(geometry, dict)
            and isinstance(geometry.get("box"), list)
            and len(geometry["box"]) >= 4]
        reference_height = float(np.median(heights)) if heights else max(16.0, bottom - top)
        line_groups = [group for group in line_groups
                       if reference_height * 0.65 <= group[-1] - group[0] + 1
                       <= reference_height * 1.75
                       and group[-1] < binary.shape[0] - 3]
        for group in line_groups:
            y0 = max(0, group[0] - 8)
            y1 = min(row_image.height, group[-1] + 9)
            line_index = lyric_line_counts[target]
            lyric_line_counts[target] += 1
            tasks.append(("lyric", target, line_index,
                          fixed_canvas(row_image.crop((0, y0, row_image.width, y1)),
                                       height=200),
                          lyric_prompt, note_count,
                          [0, top + y0, page.width, top + y1]))

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        texts = generate_many(
            model, processor,
            [item[3] for item in batch], [item[4] for item in batch],
            # Text-layer responses are tiny JSON objects.  Keeping this cap
            # below the page-level pitch budget prevents a long lyric crop
            # from spending minutes in autoregressive generation.
            max_tokens=min(max_tokens, 256),
        )
        for (kind, index, line_index, _, _, note_count, source_box), raw in zip(batch, texts):
            try:
                layer = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                if kind == "grace":
                    grace_notes = layer.get("grace_notes", [])
                    if (isinstance(grace_notes, list)
                            and 1 <= len(grace_notes) <= 2
                            and all(isinstance(value, int) and 1 <= value <= 7
                                    for value in grace_notes)):
                        rows[index]["text_layer"]["ornaments"].append({
                            "type": "yinyin", "note": line_index,
                            "grace_notes": grace_notes,
                        })
                    continue
                if kind == "lyric":
                    text = layer.get("text")
                    if isinstance(text, str) and text.strip():
                        text = "".join(text.split()).replace(",", "，").replace(".", "。")
                        for prefix in ("D。S。", "DS。", "D。S", "DS"):
                            if text.startswith(prefix):
                                text = text[len(prefix):]
                                break
                        if any(marker in text for marker in
                               ("本谱", "声明", "软件制作", "中国曲谱网", "上传")):
                            continue
                        rows[index]["text_layer"]["lyric_lines"].append(
                            (line_index, text, source_box))
                    continue
                signatures = []
                for signature in layer.get("time_signatures", []):
                    if (isinstance(signature, dict)
                            and isinstance(signature.get("before_note"), int)
                            and 0 <= signature["before_note"] < note_count
                            and isinstance(signature.get("numerator"), int)
                            and isinstance(signature.get("denominator"), int)):
                        signatures.append(signature)
                rows[index]["text_layer"]["time_signatures"] = signatures
            except Exception as exc:
                rows[index].setdefault("uncertainties", []).append(
                    f"text layer failed: {exc}")
    for index in score_indices:
        lines = rows[index]["text_layer"].get("lyric_lines", [])
        ordered = sorted(lines)
        rows[index]["text_layer"]["lyric_lines"] = [
            text for _, text, _ in ordered
        ]
        rows[index]["text_layer"]["lyric_boxes"] = [
            box for _, _, box in ordered
        ]
    return metadata, rows
    original_count = sum(
        token.startswith("P") or token == "R0" for token in original)
    if len(replacement_notes) != original_count:
        return None
    iterator = iter(replacement_notes)
    return [
        next(iterator) if token.startswith("P") or token == "R0" else token
        for token in original
    ]


def measure_boxes(image: Image.Image, baseline: float):
    """Split a score row on true, tall vertical barline components."""
    gray = np.asarray(image.convert("L"))
    binary = (gray < 150).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    bars = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        center_x, center_y = centroids[index]
        if (width <= 14 and height >= max(36, image.height * 0.18)
                and height / max(1, width) >= 4.0
                and area / (width * height) >= 0.70
                and abs(center_y - baseline) <= max(55, image.height * 0.28)):
            bars.append(float(center_x))
    grouped = []
    for value in sorted(bars):
        if grouped and value - grouped[-1][-1] <= 15:
            grouped[-1].append(value)
        else:
            grouped.append([value])
    bar_x = [sum(group) / len(group) for group in grouped]
    if len(bar_x) < 2:
        return []

    # Only keep segments that contain baseline ink; this drops whitespace
    # before the first pickup and after the final barline.
    baseline_top = max(0, round(baseline - 30))
    baseline_bottom = min(image.height, round(baseline + 30))
    column_ink = (gray[baseline_top:baseline_bottom] < 170).sum(axis=0)
    active = np.flatnonzero(column_ink > 1)
    if not len(active):
        return []
    left_edge, right_edge = int(active[0]), int(active[-1])
    boundaries = [max(0, left_edge - 24), *[round(value) for value in bar_x]]
    if right_edge > boundaries[-1] + 24:
        boundaries.append(min(image.width, right_edge + 24))
    boxes = []
    for left, right in zip(boundaries, boundaries[1:]):
        crop_left, crop_right = max(0, left - 10), min(image.width, right + 10)
        if crop_right - crop_left >= 36:
            boxes.append((crop_left, 0, crop_right, image.height))
    return boxes


def reread_score_rows_by_measure(
    page, bands, rows, model, processor, prompt, max_tokens, batch_size,
):
    """Use HOMR-style layout-first reading for every row with real barlines."""
    tasks = []
    row_meta = {}
    for index, parsed in enumerate(rows):
        if parsed.get("content_type") != "score" or not parsed.get("voices"):
            continue
        top, bottom = bands[index]
        row_image = page.crop((0, top, page.width, bottom))
        estimated, baseline = estimate_main_digit_count(row_image)
        boxes = measure_boxes(row_image, baseline)
        if estimated < 4 or len(boxes) < 2:
            continue
        row_meta[index] = {"estimated": estimated, "parts": [None] * len(boxes)}
        for part_index, box in enumerate(boxes):
            tasks.append((index, part_index, row_image.crop(box), baseline))

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        images = [fixed_canvas(item[2]) for item in batch]
        texts = generate_many(
            model, processor, images, [prompt] * len(images),
            max_tokens=min(max_tokens, 256),
        )
        for task, raw in zip(batch, texts):
            row_index, part_index, segment_image, row_baseline = task
            try:
                parsed = parse_json(raw)
                if parsed.get("content_type") != "score" or not parsed.get("voices"):
                    continue
                tokens = list(parsed["voices"][0].get("tokens", []))
                expected = count_digits_near_baseline(segment_image, row_baseline)
                excess = sum(token.startswith("P") or token == "R0" for token in tokens) - expected
                # A dash is sometimes called R0 at the right edge of a measure.
                # A real zero is a digit component and is already included in
                # the visual count, so only excess rests are converted.
                for token_index in range(len(tokens) - 1, -1, -1):
                    if excess <= 0:
                        break
                    if tokens[token_index] == "R0":
                        tokens[token_index] = "-"
                        excess -= 1
                row_meta[row_index]["parts"][part_index] = tokens
            except Exception:
                continue

    for row_index, meta in row_meta.items():
        if any(part is None for part in meta["parts"]):
            continue
        combined = []
        for part in meta["parts"]:
            combined.extend(token for token in part if not token.startswith("B"))
            combined.append("B|")
        combined_count = sum(token.startswith("P") or token == "R0" for token in combined)
        original_count = pitch_count(rows[row_index])
        estimated = meta["estimated"]
        original_tokens = rows[row_index]["voices"][0].get("tokens", [])
        original_error = abs(original_count - estimated)
        combined_error = abs(combined_count - estimated)
        replacement = None
        if combined_count == original_count and combined_error <= original_error:
            # Localized crops are much better at pitch identity, while the full
            # row is better at distinguishing augmentation dots from dashes.
            replacement = replace_pitch_stream(original_tokens, combined)
        elif combined_error < original_error and combined_error <= 1:
            replacement = combined
        if replacement is not None:
            voice = rows[row_index]["voices"][0]
            voice["tokens"] = replacement
            voice.pop("token_geometry", None)
            voice["relations"] = []
            voice["modifiers"] = []
            rows[row_index].setdefault("uncertainties", []).append(
                f"measure-localized reread: {original_count}->{combined_count}, "
                f"visual estimate {estimated}")
    return rows


def retry_underread_rows(page, bands, rows, model, processor, prompt, max_tokens):
    """Re-read only rows whose visual digit count proves a large under-read."""
    for index, parsed in enumerate(rows):
        if parsed.get("content_type") != "score" or not parsed.get("voices"):
            continue
        top, bottom = bands[index]
        row_image = page.crop((0, top, page.width, bottom))
        estimated, baseline = estimate_main_digit_count(row_image)
        original_count = pitch_count(parsed)
        if estimated < 6 or original_count >= estimated * 0.82:
            continue
        split = split_near_whitespace(row_image, baseline)
        halves = [
            fixed_canvas(row_image.crop((0, 0, split, row_image.height))),
            fixed_canvas(row_image.crop((split, 0, row_image.width, row_image.height))),
        ]
        try:
            parts = [parse_json(text) for text in generate_many(
                model, processor, halves, [prompt, prompt], max_tokens=max_tokens,
            )]
        except Exception:
            parts = []
            for half in halves:
                try:
                    raw = generate(
                        model, processor, prompt, image=half, max_tokens=max_tokens,
                        temperature=0.0, verbose=False,
                    ).text
                    parts.append(parse_json(raw))
                except Exception:
                    parts = []
                    break
        if len(parts) != 2 or any(
                part.get("content_type") != "score" or not part.get("voices")
                for part in parts):
            continue
        combined = []
        for part in parts:
            combined.extend(part["voices"][0].get("tokens", []))
        combined_count = sum(token.startswith("P") or token == "R0" for token in combined)
        if (combined_count >= original_count + 2
                and abs(combined_count - estimated) < abs(original_count - estimated)):
            parsed["voices"][0]["tokens"] = combined
            parsed["voices"][0].pop("token_geometry", None)
            # Full-row relation indices no longer match the replacement stream.
            parsed["voices"][0]["relations"] = []
            parsed["voices"][0]["modifiers"] = []
            parsed.setdefault("uncertainties", []).append(
                f"automatic horizontal reread: {original_count}->{combined_count}, "
                f"visual estimate {estimated}")
    return rows


def infer(
    image_path: Path, model_name: str, batch_size: int, max_tokens: int,
    only_band: int = 0, bands_override=None, skip_relations: bool = False,
    text_layers: bool = False,
):
    started = time.perf_counter()
    with Image.open(image_path) as opened:
        page = opened.convert("RGB")
    bands = bands_override or layout_bands(page)
    bands = [(max(0, int(top)), min(page.height, int(bottom)))
             for top, bottom in bands if int(bottom) - int(top) >= 8]
    if only_band:
        if only_band < 1 or only_band > len(bands):
            raise ValueError(f"only-band must be between 1 and {len(bands)}")
        bands = [bands[only_band - 1]]
    crops = [fixed_canvas(page.crop((0, top, page.width, bottom)))
             for top, bottom in bands]

    model, processor = load(model_name)
    config = load_config(model_name)
    prompt = apply_chat_template(
        processor, config, PROMPT, num_images=1, thinking_mode="disabled")
    relation_prompt = apply_chat_template(
        processor, config, RELATION_PROMPT, num_images=1, thinking_mode="disabled")
    glyph_prompt = apply_chat_template(
        processor, config, GLYPH_PROMPT, num_images=1, thinking_mode="disabled")

    rows = []
    for start in range(0, len(crops), batch_size):
        images = crops[start:start + batch_size]
        texts = generate_many(
            model, processor, images, [prompt] * len(images), max_tokens=max_tokens,
        )

        if skip_relations:
            relation_texts = [None] * len(images)
        else:
            relation_texts = generate_many(
                model, processor, images, [relation_prompt] * len(images),
                max_tokens=min(max_tokens, 384),
            )

        for offset, (raw, raw_relations) in enumerate(zip(texts, relation_texts)):
            index = start + offset
            top, bottom = bands[index]
            try:
                parsed = parse_json(raw)
                try:
                    if raw_relations is None:
                        raise StopIteration
                    relation_payload = json.loads(
                        raw_relations[raw_relations.index("{"):raw_relations.rindex("}") + 1])
                    relation_voices = relation_payload.get("voices", [])
                    for voice_index, voice in enumerate(parsed.get("voices", [])):
                        if voice_index >= len(relation_voices):
                            break
                        relation_voice = relation_voices[voice_index]
                        voice["relations"] = relation_voice.get("relations", [])
                        voice["modifiers"] = relation_voice.get("modifiers", [])
                except StopIteration:
                    pass
                except Exception as relation_error:
                    parsed.setdefault("uncertainties", []).append(
                        f"relation pass failed: {relation_error}")
                rows.append({
                    "row": index + 1,
                    "crop_box": [0, top, page.width, bottom],
                    **parsed,
                })
            except Exception as exc:
                rows.append({
                    "row": index + 1,
                    "crop_box": [0, top, page.width, bottom],
                    "content_type": "error",
                    "voices": [],
                    "confidence": 0.0,
                    "uncertainties": [str(exc)],
                })

    rows = classify_score_glyphs(
        page, bands, rows, model, processor, glyph_prompt, max_tokens)
    rows = retry_underread_rows(
        page, bands, rows, model, processor, prompt, max_tokens)

    metadata = {}
    if text_layers:
        metadata, rows = read_text_layers(
            page, bands, rows, model, processor, config, max_tokens, batch_size)

    return {
        "model": model_name,
        "page_size": list(page.size),
        "rows": rows,
        "metadata": metadata,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--only-band", type=int, default=0)
    parser.add_argument("--bands-json", default=None)
    parser.add_argument("--skip-relations", action="store_true")
    parser.add_argument("--text-layers", action="store_true")
    args = parser.parse_args()
    bands_override = None
    if args.bands_json:
        bands_override = json.loads(Path(args.bands_json).read_text())
    payload = infer(
        Path(args.image), args.model, args.batch_size, args.max_tokens,
        args.only_band, bands_override, args.skip_relations, args.text_layers)
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
