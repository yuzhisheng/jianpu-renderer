#!/usr/bin/env python3
"""
Transformer 训练脚本
- 输入: backend/weights/pairs.npz
- 输出: backend/weights/transformer.pt
"""
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from model.transformer import JianpuTransformer
from model.tokenizer import PAD_ID


class PairsDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.src = data["src"]  # (N, S)
        self.tgt = data["tgt"]  # (N, T)
        # 拆分为 input/target (decoder 输入右移一位)
        self.tgt_in = self.tgt[:, :-1]
        self.tgt_out = self.tgt[:, 1:]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.src[idx], dtype=torch.long),
            "tgt_in": torch.tensor(self.tgt_in[idx], dtype=torch.long),
            "tgt_out": torch.tensor(self.tgt_out[idx], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="backend/weights/pairs.npz")
    parser.add_argument("--output", default="backend/weights/transformer.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"❌ 训练数据不存在: {data_path}")
        print("请先运行: python backend/scripts/prepare_pairs.py")
        sys.exit(1)

    print(f"📦 加载训练数据: {data_path}")
    dataset = PairsDataset(str(data_path))
    print(f"   {len(dataset)} samples")

    # 划分 train / val (90 / 10)
    n = len(dataset)
    n_val = max(1, n // 10)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"   train: {n_train}, val: {n_val}")

    # === 模型 ===
    device = torch.device(args.device)
    model = JianpuTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {n_params:,}")

    # === 优化器 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # === 训练循环 ===
    best_val_loss = float("inf")
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            logits = model(src, tgt_in)  # (B, T, V)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                logits = model(src, tgt_in)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                )
                val_loss += loss.item()
                n_val_batches += 1

        train_loss /= max(1, n_batches)
        val_loss /= max(1, n_val_batches)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_encoder_layers": args.num_layers,
                    "num_decoder_layers": args.num_layers,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_path)
            print(f"  💾 Saved best model (val_loss={val_loss:.4f})")

    print(f"\n✅ 训练完成, 最佳 val_loss={best_val_loss:.4f}")
    print(f"   模型路径: {output_path}")


if __name__ == "__main__":
    main()
