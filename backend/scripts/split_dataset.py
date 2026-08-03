#!/usr/bin/env python3
"""Create deterministic, leakage-free YOLO train/validation manifests."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default="public/training")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()

    root = Path(args.training_dir).resolve()
    images = sorted(root.glob("score_*.png"))
    if not images:
        raise SystemExit(f"No score_*.png images found in {root}")

    cutoff = round(256 * args.val_ratio)
    train, val = [], []
    counts = {"train": Counter(), "val": Counter()}
    for image in images:
        # Match generate_training_pngs.cjs, which hashes the JSON file name.
        source_name = image.with_suffix(".json").name
        split = "val" if hashlib.sha1(source_name.encode()).digest()[0] < cutoff else "train"
        (val if split == "val" else train).append(str(image))
        label = image.with_suffix(".txt")
        if label.exists():
            for line in label.read_text().splitlines():
                if line.strip():
                    counts[split][int(line.split()[0])] += 1

    (root / "train.txt").write_text("\n".join(train) + "\n")
    (root / "val.txt").write_text("\n".join(val) + "\n")

    yaml_path = root / "data.yaml"
    names_line = next((line for line in yaml_path.read_text().splitlines() if line.startswith("names:")), None)
    if names_line is None:
        raise SystemExit(f"Missing names entry in {yaml_path}")
    yaml_path.write_text(
        f"# Leakage-free YOLO training data config\npath: {root}\n"
        f"train: train.txt\nval: val.txt\n\nnc: 42\n{names_line}\n"
    )

    print(f"train={len(train)}, val={len(val)}, overlap={len(set(train) & set(val))}")
    for split in ("train", "val"):
        missing = [class_id for class_id in range(42) if counts[split][class_id] == 0]
        print(f"{split}: {sum(counts[split].values())} boxes; missing classes={missing}")


if __name__ == "__main__":
    main()
