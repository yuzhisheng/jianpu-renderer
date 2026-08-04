"""Dataset and image preprocessing for the visual jianpu Transformer."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from model.jianpu_event_vocabulary import (
    BRANCHES, CTC_TOKEN_TO_ID, IGNORE_ID, PAD_ID, TOKEN_TO_ID, teacher_inputs,
)


def preprocess_row_image(
    image: Image.Image, image_height: int, max_width: int, augment: bool = False,
) -> torch.Tensor:
    image = image.convert("L")
    if augment:
        if random.random() < 0.45:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.15, 1.25)))
        if random.random() < 0.45:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.72, 1.3))
    scale = min(image_height / max(image.height, 1), max_width / max(image.width, 1))
    width = max(1, min(max_width, round(image.width * scale)))
    height = max(1, min(image_height, round(image.height * scale)))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (max_width, image_height), 255)
    canvas.paste(resized, (0, (image_height - height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32)
    # Ink is positive and empty paper is negative; this is stable for scans and
    # screenshots and avoids a dataset-wide normalization pass.
    return torch.from_numpy(((255.0 - pixels) / 127.5 - 1.0)[None, :, :])


def resized_content_width(image: Image.Image, image_height: int, max_width: int) -> int:
    scale = min(image_height / max(image.height, 1), max_width / max(image.width, 1))
    return max(1, min(max_width, round(image.width * scale)))


class JianpuRowDataset(Dataset):
    def __init__(
        self, manifest_path: str | Path, image_height: int = 128,
        max_width: int = 1536, augment: bool = False, limit: int = 0,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        rows = [json.loads(line) for line in self.manifest_path.read_text().splitlines() if line]
        self.rows = rows[:limit] if limit else rows
        self.image_height = image_height
        self.max_width = max_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = Path(row["image"])
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        with Image.open(path) as opened:
            content_width = resized_content_width(opened, self.image_height, self.max_width)
            image = preprocess_row_image(
                opened, self.image_height, self.max_width, self.augment,
            )
        targets = {branch: list(row["targets"][branch]) for branch in BRANCHES}
        ctc = [CTC_TOKEN_TO_ID[token] for token in row.get("ctc_tokens", [])]
        return {
            "image": image, "content_width": content_width,
            "targets": targets, "ctc": ctc, "id": row["id"],
        }


def collate_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    max_length = max(len(row["targets"]["kind"]) for row in rows)
    images = torch.stack([row["image"] for row in rows])
    max_ctc = max(len(row["ctc"]) for row in rows)
    decoder_inputs: Dict[str, List[List[int]]] = {branch: [] for branch in BRANCHES}
    decoder_targets: Dict[str, List[List[int]]] = {branch: [] for branch in BRANCHES}
    for row in rows:
        target = row["targets"]
        inputs = teacher_inputs(target)
        for branch in BRANCHES:
            input_pad = PAD_ID if branch == "kind" else TOKEN_TO_ID[branch]["NONE"]
            output_pad = PAD_ID if branch == "kind" else IGNORE_ID
            decoder_inputs[branch].append(
                inputs[branch][:-1] + [input_pad] * (max_length - len(target[branch]))
            )
            decoder_targets[branch].append(
                target[branch][1:] + [output_pad] * (max_length - len(target[branch]))
            )
    return {
        "images": images,
        "inputs": {branch: torch.tensor(values, dtype=torch.long)
                   for branch, values in decoder_inputs.items()},
        "targets": {branch: torch.tensor(values, dtype=torch.long)
                    for branch, values in decoder_targets.items()},
        "ctc_targets": torch.tensor([
            row["ctc"] + [0] * (max_ctc - len(row["ctc"])) for row in rows
        ], dtype=torch.long),
        "ctc_target_lengths": torch.tensor([len(row["ctc"]) for row in rows], dtype=torch.long),
        "content_widths": torch.tensor([row["content_width"] for row in rows], dtype=torch.long),
        "ids": [row["id"] for row in rows],
    }
