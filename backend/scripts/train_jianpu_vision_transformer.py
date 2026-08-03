#!/usr/bin/env python3
"""Train the row-image → multi-branch jianpu Transformer on CPU/MPS/CUDA."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from model.jianpu_event_vocabulary import BRANCHES, IGNORE_ID, PAD_ID
from model.jianpu_vision_data import JianpuRowDataset, collate_rows
from model.jianpu_vision_transformer import JianpuVisionTransformer, VisionTransformerConfig


BRANCH_WEIGHTS = {
    "kind": 1.0, "pitch": 1.0, "accidental": 0.25,
    "octave": 0.5, "duration": 0.75, "articulation": 0.25,
}
CTC_WEIGHT = 1.5


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def batch_loss(logits, targets):
    total = None
    details = {}
    for branch in BRANCHES:
        target = targets[branch]
        ignore = PAD_ID if branch == "kind" else IGNORE_ID
        valid = target.ne(ignore)
        if not bool(valid.any()):
            continue
        loss = F.cross_entropy(logits[branch].transpose(1, 2), target, ignore_index=ignore)
        total = loss * BRANCH_WEIGHTS[branch] if total is None else total + loss * BRANCH_WEIGHTS[branch]
        details[branch] = float(loss.detach())
    if total is None:
        raise RuntimeError("batch has no supervised targets")
    return total, details


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    branch_correct = {branch: 0 for branch in BRANCHES}
    branch_total = {branch: 0 for branch in BRANCHES}
    for batch in loader:
        images = batch["images"].to(device)
        inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
        targets = {key: value.to(device) for key, value in batch["targets"].items()}
        ctc_targets = batch["ctc_targets"].to(device)
        ctc_target_lengths = batch["ctc_target_lengths"].to(device)
        input_lengths = ((batch["content_widths"] + 15) // 16).clamp(min=1, max=images.shape[-1] // 16).to(device)
        with torch.set_grad_enabled(training):
            logits = model(images, inputs)
            loss, _ = batch_loss(logits, targets)
            ctc_logits = model.ctc_logits(images)
            # PyTorch does not yet implement CTCLoss on Apple MPS.  Keep the
            # visual network and Transformer on the GPU and move only this
            # small loss tensor to CPU; autograd carries gradients back across
            # the device copy.
            ctc_device = torch.device("cpu") if device.type == "mps" else device
            ctc_loss = F.ctc_loss(
                ctc_logits.float().to(ctc_device).log_softmax(-1).transpose(0, 1),
                ctc_targets.to(ctc_device), input_lengths.to(ctc_device),
                ctc_target_lengths.to(ctc_device), blank=0, zero_infinity=True,
            ).to(device)
            loss = loss + CTC_WEIGHT * ctc_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach())
        for branch in BRANCHES:
            ignore = PAD_ID if branch == "kind" else IGNORE_ID
            valid = targets[branch].ne(ignore)
            branch_correct[branch] += int((logits[branch].argmax(-1)[valid] == targets[branch][valid]).sum())
            branch_total[branch] += int(valid.sum())
    count = max(len(loader), 1)
    return total_loss / count, {
        branch: branch_correct[branch] / max(branch_total[branch], 1)
        for branch in BRANCHES if branch_total[branch]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="backend/jianpu_transformer_dataset")
    parser.add_argument("--train-manifest", default="train.jsonl")
    parser.add_argument("--val-manifest", default="val.jsonl")
    parser.add_argument("--output", default="backend/weights/jianpu_vision_transformer.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--init", help="initialize from a compatible checkpoint")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(17)
    if args.smoke:
        args.epochs, args.batch, args.d_model, args.layers = 1, 2, 96, 2
        args.max_width, args.limit = 512, 8
    config = VisionTransformerConfig(
        max_width=args.max_width, d_model=args.d_model,
        nhead=6 if args.d_model % 6 == 0 else 4,
        decoder_layers=args.layers, dim_feedforward=args.d_model * 4,
    )
    data = (ROOT / args.data).resolve()
    train = JianpuRowDataset(data / args.train_manifest, config.image_height, config.max_width, True, args.limit)
    val = JianpuRowDataset(data / args.val_manifest, config.image_height, config.max_width, False, args.limit)
    if not train or not val:
        raise SystemExit("empty dataset; run build_jianpu_transformer_dataset.py first")
    loaders = {
        "train": DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_rows),
        "val": DataLoader(val, args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_rows),
    }
    device = choose_device(args.device)
    model = JianpuVisionTransformer(config).to(device)
    if args.init:
        initial = torch.load((ROOT / args.init).resolve(), map_location=device, weights_only=False)
        model.load_state_dict(initial["model_state_dict"])
        print(f"initialized={args.init}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={len(train)} val={len(val)} params={model.parameter_count():,}")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_accuracy = run_epoch(model, loaders["train"], device, optimizer)
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(model, loaders["val"], device)
        scheduler.step()
        metrics = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_accuracy, "val_accuracy": val_accuracy,
            "seconds": round(time.time() - started, 2),
        }
        print(json.dumps(metrics, ensure_ascii=False))
        if val_loss < best:
            best = val_loss
            torch.save({
                "format": "jianpu-vision-transformer-v1",
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(), "metrics": metrics,
            }, output)
    print(f"saved={output} best_val_loss={best:.6f}")


if __name__ == "__main__":
    main()
