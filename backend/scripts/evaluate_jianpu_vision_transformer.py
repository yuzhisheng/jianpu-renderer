#!/usr/bin/env python3
"""Evaluate free-running CTC recognition on a held-out row manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from model.jianpu_event_vocabulary import CTC_ID_TO_TOKEN
from model.jianpu_vision_data import JianpuRowDataset, collate_rows
from scripts.infer_jianpu_vision_transformer import load_recognizer
from scripts.train_jianpu_vision_transformer import choose_device


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def similarity(left, right):
    return 1.0 - edit_distance(left, right) / max(len(left), len(right), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="backend/jianpu_transformer_dataset/test.jsonl")
    parser.add_argument("--weights", default="backend/weights/jianpu_vision_transformer_v1.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--token-bonus", type=float, default=0.0)
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_recognizer((ROOT / args.weights).resolve(), device)
    dataset = JianpuRowDataset(
        (ROOT / args.data).resolve(), model.config.image_height, model.config.max_width,
    )
    loader = DataLoader(dataset, args.batch, shuffle=False, collate_fn=collate_rows)
    rows = []
    for batch in loader:
        images = batch["images"].to(device)
        lengths = ((batch["content_widths"] + 15) // 16).to(device)
        predictions = model.generate_ctc(
            images, lengths, beam_width=args.beam, token_bonus=args.token_bonus,
        )
        for index, prediction in enumerate(predictions):
            target_length = int(batch["ctc_target_lengths"][index])
            target = [CTC_ID_TO_TOKEN[int(value)]
                      for value in batch["ctc_targets"][index, :target_length]]
            target_pitch = [token for token in target if token.startswith("P") or token == "R0"]
            prediction_pitch = [token for token in prediction if token.startswith("P") or token == "R0"]
            rows.append({
                "id": batch["ids"][index], "target": target, "prediction": prediction,
                "skeleton_similarity": similarity(target, prediction),
                "pitch_similarity": similarity(target_pitch, prediction_pitch),
                "count_similarity": min(len(target_pitch), len(prediction_pitch)) / max(
                    len(target_pitch), len(prediction_pitch), 1,
                ),
            })
    summary = {
        "rows": len(rows),
        "skeleton_similarity": sum(row["skeleton_similarity"] for row in rows) / max(len(rows), 1),
        "pitch_similarity": sum(row["pitch_similarity"] for row in rows) / max(len(rows), 1),
        "count_similarity": sum(row["count_similarity"] for row in rows) / max(len(rows), 1),
        "exact_rows": sum(row["target"] == row["prediction"] for row in rows),
    }
    payload = {"summary": summary, "rows": rows}
    if args.output:
        output = (ROOT / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
