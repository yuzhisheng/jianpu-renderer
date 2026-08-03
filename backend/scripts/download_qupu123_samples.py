#!/usr/bin/env python3
"""Download a small, source-attributed qupu123 sample for research evaluation."""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image


BASE_URL = "https://www.qupu123.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (jianpu-recognizer research evaluation)"}
PAGE_RE = re.compile(r'href="(/[^"]*?/p\d+\.html)')
IMAGE_RE = re.compile(r'<div class="imageList">.*?href="([^"]+\.(?:png|jpe?g|gif))"', re.I | re.S)
TITLE_RE = re.compile(r'<h1[^>]*>(.*?)</h1>', re.I | re.S)


def clean_title(raw: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
    return re.sub(r"\s+", " ", value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="backend/real_data/qupu123_samples")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    home = session.get(BASE_URL, timeout=20)
    home.raise_for_status()
    candidates = []
    for path in PAGE_RE.findall(home.text):
        if path not in candidates and any(part in path for part in ("/jipu/", "/yuanchuang/", "/tongsu/", "/minge/", "/shaoer/")):
            candidates.append(path)

    records = []
    for page_path in candidates:
        if len(records) >= args.limit:
            break
        page_url = urljoin(BASE_URL, page_path)
        response = session.get(page_url, timeout=20)
        response.raise_for_status()
        image_match = IMAGE_RE.search(response.text)
        if not image_match:
            continue
        image_url = urljoin(BASE_URL, image_match.group(1))
        image_response = session.get(image_url, timeout=30)
        image_response.raise_for_status()
        suffix = Path(image_match.group(1)).suffix.lower()
        target = output / f"qupu123_{len(records):03d}{suffix}"
        target.write_bytes(image_response.content)
        try:
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                width, height = image.size
        except Exception:
            target.unlink(missing_ok=True)
            continue
        title_match = TITLE_RE.search(response.text)
        record = {
            "file": str(target),
            "title": clean_title(title_match.group(1)) if title_match else "",
            "source_page": page_url,
            "image_url": image_url,
            "width": width,
            "height": height,
        }
        records.append(record)
        print(f"[{len(records):02d}/{args.limit}] {record['title']} {width}x{height}")
        time.sleep(max(0.0, args.delay))

    metadata = output / "sources.jsonl"
    metadata.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")
    print(f"downloaded={len(records)}, metadata={metadata}")


if __name__ == "__main__":
    main()
