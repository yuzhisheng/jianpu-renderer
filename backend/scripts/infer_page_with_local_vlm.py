#!/usr/bin/env python3
"""Recognize a jianpu page as a conservative pitch/bar skeleton with MLX VLM.

The script is intentionally self-contained so it can run in ``.venv-vlm``.
Its stdout may contain messages from MLX; callers should parse the final line
prefixed with ``__JIANPU_RESULT__``.
"""
from __future__ import annotations

import argparse
import json
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


def estimate_main_digit_count(image: Image.Image):
    """Estimate digit components on the uppermost dense music baseline."""
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
        return 0, image.height / 2
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
    return len(row), baseline


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
        try:
            response = batch_generate(
                model, processor, images=images, prompts=[prompt] * len(images),
                max_tokens=min(max_tokens, 256), temperature=0.0, verbose=False,
                group_by_shape=True,
            )
            texts = response.texts
        except Exception:
            texts = [
                generate(
                    model, processor, prompt, image=image,
                    max_tokens=min(max_tokens, 256),
                    temperature=0.0, verbose=False,
                ).text
                for image in images
            ]
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
            response = batch_generate(
                model, processor, images=halves, prompts=[prompt, prompt],
                max_tokens=max_tokens, temperature=0.0, verbose=False,
                group_by_shape=True,
            )
            parts = [parse_json(text) for text in response.texts]
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
        try:
            response = batch_generate(
                model, processor, images=images, prompts=[prompt] * len(images),
                max_tokens=max_tokens, temperature=0.0, verbose=False,
                group_by_shape=True,
            )
            texts = response.texts
        except Exception:
            texts = [
                generate(
                    model, processor, prompt, image=image, max_tokens=max_tokens,
                    temperature=0.0, verbose=False,
                ).text
                for image in images
            ]

        if skip_relations:
            relation_texts = [None] * len(images)
        else:
            try:
                relation_response = batch_generate(
                    model, processor, images=images,
                    prompts=[relation_prompt] * len(images),
                    max_tokens=min(max_tokens, 384), temperature=0.0, verbose=False,
                    group_by_shape=True,
                )
                relation_texts = relation_response.texts
            except Exception:
                relation_texts = [
                    generate(
                        model, processor, relation_prompt, image=image,
                        max_tokens=min(max_tokens, 384), temperature=0.0, verbose=False,
                    ).text
                    for image in images
                ]

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

    return {
        "model": model_name,
        "page_size": list(page.size),
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--only-band", type=int, default=0)
    parser.add_argument("--bands-json", default=None)
    parser.add_argument("--skip-relations", action="store_true")
    args = parser.parse_args()
    bands_override = None
    if args.bands_json:
        bands_override = json.loads(Path(args.bands_json).read_text())
    payload = infer(
        Path(args.image), args.model, args.batch_size, args.max_tokens,
        args.only_band, bands_override, args.skip_relations)
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
