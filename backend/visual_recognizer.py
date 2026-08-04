"""Page-level adapter for the row-image jianpu Transformer."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image

from assembler import parse_tokens_to_score
from model.jianpu_vision_data import preprocess_row_image, resized_content_width
from model.jianpu_vision_transformer import JianpuVisionTransformer, VisionTransformerConfig

ROOT = Path(__file__).resolve().parent


def score_row_bands(detections: Sequence[tuple], image_height: int) -> List[Tuple[int, int]]:
    """Turn detector note anchors into non-overlapping full-width score bands."""
    anchors = [item for item in detections if 0 <= int(item[0]) <= 8]
    if not anchors:
        return []
    rows: List[List[tuple]] = []
    for anchor in sorted(anchors, key=lambda item: item[2]):
        tolerance = max(10.0, min(30.0, float(anchor[4]) * 1.35))
        target = next((row for row in rows if abs(
            anchor[2] - sum(item[2] for item in row) / len(row)
        ) <= tolerance), None)
        if target is None:
            rows.append([anchor])
        else:
            target.append(anchor)
    centers = [sum(item[2] for item in row) / len(row) for row in rows]
    bands = []
    for index, (row, center) in enumerate(zip(rows, centers)):
        median_height = sorted(float(item[4]) for item in row)[len(row) // 2]
        upper = 0 if index == 0 else int((centers[index - 1] + center) / 2)
        lower = image_height if index + 1 == len(rows) else int((center + centers[index + 1]) / 2)
        top = max(upper, int(center - max(56.0, median_height * 3.0)))
        bottom = min(lower, int(center + max(76.0, median_height * 4.2)))
        if bottom - top >= 16:
            bands.append((top, bottom))
    return bands


class VisualTransformerRecognizer:
    def __init__(
        self, weights_path: Optional[str] = None, device: str = "auto",
        beam_width: int = 15, token_bonus: float = 0.6,
    ):
        self.weights_path = Path(weights_path or ROOT / "weights" / "jianpu_vision_transformer_v2.pt")
        self.device = torch.device(
            "mps" if device == "auto" and torch.backends.mps.is_available()
            else "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.beam_width = beam_width
        self.token_bonus = token_bonus
        self.model: Optional[JianpuVisionTransformer] = None

    def load(self):
        if self.model is not None:
            return
        if not self.weights_path.exists():
            raise FileNotFoundError(f"视觉 Transformer 权重不存在: {self.weights_path}")
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        model = JianpuVisionTransformer(VisionTransformerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_page(self, image: Image.Image, detections: Sequence[tuple]) -> dict:
        self.load()
        bands = score_row_bands(detections, image.height)
        if not bands:
            return {"score": parse_tokens_to_score([]), "src_tokens": [], "tgt_tokens": []}
        tensors, lengths = [], []
        for top, bottom in bands:
            crop = image.crop((0, top, image.width, bottom))
            lengths.append((resized_content_width(
                crop, self.model.config.image_height, self.model.config.max_width,
            ) + 15) // 16)
            tensors.append(preprocess_row_image(
                crop, self.model.config.image_height, self.model.config.max_width,
            ))
        batch = torch.stack(tensors).to(self.device)
        row_tokens = self.model.generate_ctc(
            batch, torch.tensor(lengths, device=self.device),
            beam_width=self.beam_width, token_bonus=self.token_bonus,
        )
        tokens = []
        for index, row in enumerate(row_tokens):
            if index:
                tokens.append("<ROW>")
            tokens.extend(row)
        return {
            "score": parse_tokens_to_score(tokens),
            "src_tokens": [], "tgt_tokens": tokens,
        }
