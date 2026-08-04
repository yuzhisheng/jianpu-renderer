#!/usr/bin/env python3
"""Build train-only, scan-like composite pages and Chinese-text examples.

The renderer produces short, clean pages.  Real numbered notation is commonly a
tall photocopied page with several systems, lyrics and page furniture.  This
script composes existing training-only images into that geometry while
transforming every YOLO box, and adds text pages labelled as the lyric class.
It is deterministic and safe to rerun.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


PREFIXES = ("scanpage_", "scanneg_", "scanrow_", "lyricrow_")
TEXT = "再回首云遮断归途今夜不会再有难舍的旧梦曾经与你有的梦明天面对多少伤痛迷惑"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def yolo_boxes(label_path: Path, width: int, height: int):
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()[:5]
        boxes.append((int(cls), float(cx) * width, float(cy) * height,
                      float(bw) * width, float(bh) * height))
    return boxes


def scan_finish(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.25))
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.55)))
    array = np.asarray(image, dtype=np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 4.5), array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def write_labels(path: Path, boxes, width: int, height: int):
    lines = []
    for cls, cx, cy, bw, bh in boxes:
        lines.append(f"{cls} {cx / width:.8f} {cy / height:.8f} {bw / width:.8f} {bh / height:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def composite_page(output: Path, sources: list[Path], rng: random.Random):
    width, height = 700, 1060
    page = Image.new("L", (width, height), rng.randint(246, 255))
    draw = ImageDraw.Draw(page)
    draw.text((width // 2 - 45, 25), rng.choice(("再 回 首", "练 习 曲", "简 谱")),
              fill=rng.randint(0, 30), font=font(24))
    draw.text((35, 68), rng.choice(("1=G   4/4", "1=C   2/4", "1=F   3/4")),
              fill=rng.randint(0, 35), font=font(18))

    transformed = []
    top = 95
    available = height - top - 55
    slot = available / len(sources)
    for index, source in enumerate(sources):
        with Image.open(source) as original:
            original = original.convert("L")
            max_h = int(slot - 12)
            scale = min(rng.uniform(0.74, 0.84), 650 / original.width, max_h / original.height)
            new_w, new_h = max(1, int(original.width * scale)), max(1, int(original.height * scale))
            resized = original.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x = rng.randint(20, max(20, width - new_w - 20))
            y = int(top + index * slot + rng.uniform(0, max(1, slot - new_h - 3)))
            page.paste(resized, (x, y))
            for cls, cx, cy, bw, bh in yolo_boxes(source.with_suffix(".txt"), original.width, original.height):
                transformed.append((cls, x + cx * scale, y + cy * scale, bw * scale, bh * scale))

    # Unlabelled page furniture is intentional: it teaches the detector that
    # arbitrary Chinese text away from a note is background, not pitch_6/tie.
    draw.text((560, 1015), f"· {rng.randint(10, 99)} ·", fill=rng.randint(0, 50), font=font(14))
    page = scan_finish(page, rng)
    page.save(output, optimize=True)
    write_labels(output.with_suffix(".txt"), transformed, width, height)


def negative_page(output: Path, rng: random.Random):
    width, height = 700, 1060
    page = Image.new("L", (width, height), rng.randint(245, 255))
    draw = ImageDraw.Draw(page)
    lyric_boxes = []

    def draw_lyric_row(text: str, x: int, row_y: int, size: int):
        row_font = font(size)
        cursor = x
        for character in text:
            bbox = draw.textbbox((cursor, row_y), character, font=row_font)
            draw.text((cursor, row_y), character, fill=rng.randint(0, 55), font=row_font)
            left, top, right, bottom = bbox
            lyric_boxes.append((40, (left + right) / 2, (top + bottom) / 2,
                                max(1, right - left), max(1, bottom - top)))
            cursor += max(1, right - left) + rng.randint(0, max(1, size // 5))

    draw_lyric_row(rng.choice(("歌词", "歌曲集", "目录", "再回首")), 250, 30, 28)
    y = 95
    while y < 980:
        size = rng.randint(14, 23)
        chars = "".join(rng.choice(TEXT) for _ in range(rng.randint(10, 24)))
        draw_lyric_row(chars, rng.randint(25, 80), y, size)
        y += rng.randint(45, 80)
    page = scan_finish(page, rng)
    page.save(output, optimize=True)
    write_labels(output.with_suffix(".txt"), lyric_boxes, width, height)


def scan_row(output: Path, source: Path, rng: random.Random):
    """Create a noisy horizontal system without shrinking its small symbols."""
    width = 800
    with Image.open(source) as original:
        original = original.convert("L")
        scale = min(rng.uniform(0.82, 1.0), 760 / original.width)
        new_w, new_h = int(original.width * scale), int(original.height * scale)
        height = max(220, min(420, new_h + rng.randint(20, 55)))
        page = Image.new("L", (width, height), rng.randint(246, 255))
        x = rng.randint(15, max(15, width - new_w - 15))
        y = rng.randint(5, max(5, height - new_h - 5))
        resized = original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        page.paste(resized, (x, y))
        boxes = [(cls, x + cx * scale, y + cy * scale, bw * scale, bh * scale)
                 for cls, cx, cy, bw, bh in yolo_boxes(
                     source.with_suffix(".txt"), original.width, original.height)]
    page = scan_finish(page, rng)
    page.save(output, optimize=True)
    write_labels(output.with_suffix(".txt"), boxes, width, height)


def lyric_row(output: Path, rng: random.Random):
    """Horizontal Chinese text page, explicitly labelled as lyric characters."""
    width, height = 800, 280
    page = Image.new("L", (width, height), rng.randint(246, 255))
    draw = ImageDraw.Draw(page)
    boxes = []
    for row in range(4):
        size = rng.randint(15, 22)
        row_font = font(size)
        cursor, y = rng.randint(25, 70), 25 + row * 60
        for character in "".join(rng.choice(TEXT) for _ in range(rng.randint(12, 23))):
            left, top, right, bottom = draw.textbbox((cursor, y), character, font=row_font)
            draw.text((cursor, y), character, fill=rng.randint(0, 50), font=row_font)
            boxes.append((40, (left + right) / 2, (top + bottom) / 2,
                          max(1, right - left), max(1, bottom - top)))
            cursor += max(1, right - left) + rng.randint(1, max(2, size // 3))
            if cursor > width - 30:
                break
    page = scan_finish(page, rng)
    page.save(output, optimize=True)
    write_labels(output.with_suffix(".txt"), boxes, width, height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="public/training")
    parser.add_argument("--pages", type=int, default=300)
    parser.add_argument("--negatives", type=int, default=200)
    parser.add_argument("--row-pages", type=int, default=0)
    parser.add_argument("--lyric-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    dataset = (root / args.dataset).resolve()
    manifest = dataset / "train.txt"
    base_images = [Path(line) for line in manifest.read_text().splitlines()
                   if line.strip() and not Path(line).name.startswith(PREFIXES)]
    rng = random.Random(args.seed)
    generated = []

    for index in range(args.pages):
        output = dataset / f"scanpage_{index:04d}.png"
        composite_page(output, rng.sample(base_images, rng.randint(3, 4)), rng)
        generated.append(output.resolve())
    for index in range(args.negatives):
        output = dataset / f"scanneg_{index:04d}.png"
        negative_page(output, rng)
        generated.append(output.resolve())
    for index in range(args.row_pages):
        output = dataset / f"scanrow_{index:04d}.png"
        scan_row(output, rng.choice(base_images), rng)
        generated.append(output.resolve())
    for index in range(args.lyric_rows):
        output = dataset / f"lyricrow_{index:04d}.png"
        lyric_row(output, rng)
        generated.append(output.resolve())

    manifest.write_text("\n".join(map(str, base_images + generated)) + "\n")
    print(f"scan-domain dataset: base={len(base_images)}, composite={args.pages}, "
          f"negative={args.negatives}, scan_rows={args.row_pages}, "
          f"lyric_rows={args.lyric_rows}, train={len(base_images) + len(generated)}")


if __name__ == "__main__":
    main()
