#!/usr/bin/env python3
"""Recognize one cropped jianpu row and emit current Score JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from assembler import parse_tokens_to_score
from model.jianpu_event_vocabulary import events_to_tokens
from model.jianpu_vision_data import preprocess_row_image, resized_content_width
from model.jianpu_vision_transformer import JianpuVisionTransformer, VisionTransformerConfig
from scripts.train_jianpu_vision_transformer import choose_device


def load_recognizer(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = JianpuVisionTransformer(VisionTransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--weights", default="backend/weights/jianpu_vision_transformer.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--decoder", choices=("ctc", "autoregressive"), default="ctc")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--token-bonus", type=float, default=0.0)
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_recognizer((ROOT / args.weights).resolve(), device)
    with Image.open(args.image) as image:
        content_width = resized_content_width(
            image, model.config.image_height, model.config.max_width,
        )
        tensor = preprocess_row_image(
            image, model.config.image_height, model.config.max_width,
        )[None].to(device)
    if args.decoder == "ctc":
        length = torch.tensor([(content_width + 15) // 16], device=device)
        tokens = model.generate_ctc(
            tensor, length, beam_width=args.beam, token_bonus=args.token_bonus,
        )[0]
    else:
        events = model.generate(tensor, args.max_len)[0]
        tokens = events_to_tokens(events)
    print(json.dumps({
        "tokens": tokens,
        "score": parse_tokens_to_score(tokens),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
