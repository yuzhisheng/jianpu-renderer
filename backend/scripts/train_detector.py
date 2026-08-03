#!/usr/bin/env python3
"""
YOLOv8 训练脚本
- 输入: public/training/data.yaml + PNG + YOLO txt
- 输出: backend/weights/best.pt
"""
import os
import sys
from pathlib import Path
import argparse
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def audit_split(data_path: Path):
    """Fail fast on validation leakage and classes with no examples."""
    dataset_dir = data_path.parent
    train_manifest = dataset_dir / "train.txt"
    val_manifest = dataset_dir / "val.txt"
    if not train_manifest.exists() or not val_manifest.exists():
        raise SystemExit("❌ 缺少 train.txt/val.txt，请先运行 backend/scripts/split_dataset.py")

    train = {line.strip() for line in train_manifest.read_text().splitlines() if line.strip()}
    val = {line.strip() for line in val_manifest.read_text().splitlines() if line.strip()}
    overlap = train & val
    if overlap:
        raise SystemExit(f"❌ train/val 泄漏：发现 {len(overlap)} 张重复图片")

    for split_name, images in (("train", train), ("val", val)):
        counts = Counter()
        for image in images:
            label = Path(image).with_suffix(".txt")
            if label.exists():
                for line in label.read_text().splitlines():
                    if line.strip():
                        counts[int(line.split()[0])] += 1
        missing = [class_id for class_id in range(42) if counts[class_id] == 0]
        if missing:
            raise SystemExit(f"❌ {split_name} 缺少类别：{missing}，请修复数据后再训练")
    print(f"✅ 数据审计通过：train={len(train)}, val={len(val)}, overlap=0")


def limited_validation_yaml(data_path: Path, output_dir: Path, limit: int) -> Path:
    """Build a tiny validation manifest for smoke tests without touching data."""
    dataset_dir = data_path.parent
    val_lines = [line for line in (dataset_dir / "val.txt").read_text().splitlines() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    limited_val = output_dir / "val.txt"
    limited_val.write_text("\n".join(val_lines[:limit]) + "\n")
    names_line = next(line for line in data_path.read_text().splitlines() if line.startswith("names:"))
    limited_yaml = output_dir / "data.yaml"
    limited_yaml.write_text(
        f"path: {dataset_dir}\ntrain: {dataset_dir / 'train.txt'}\n"
        f"val: {limited_val}\n\nnc: 42\n{names_line}\n"
    )
    return limited_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="public/training/data.yaml",
                        help="data.yaml 路径")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="预训练模型 (会自动下载)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="训练集使用比例；0-1，用于快速冒烟实验")
    parser.add_argument("--device", default="auto", help="auto、mps、cuda:0 或 cpu")
    parser.add_argument("--project", default="backend/weights/runs")
    parser.add_argument("--name", default="yolov8s_jianpu_1280")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-promote", action="store_true",
                        help="不把本次 best.pt 覆盖到 backend/weights/best.pt（冒烟实验用）")
    parser.add_argument("--val-limit", type=int, default=0,
                        help="仅验证前 N 张（仅用于设备冒烟测试）")
    parser.add_argument("--val-conf", type=float, default=0.05,
                        help="验证候选置信度；MPS 上避免 conf=0.001 导致 NMS 超时")
    parser.add_argument("--no-rect", action="store_true",
                        help="关闭矩形批次并恢复 shuffle；混合宽高比/域适配数据推荐")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics 未安装, 请先: pip install -r backend/requirements.txt")
        sys.exit(1)

    data_path = ROOT / args.data
    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = ROOT / project_dir
    if not data_path.exists():
        print(f"❌ data.yaml 不存在: {data_path}")
        print("请先运行 node scripts/generate_training_pngs.cjs")
        sys.exit(1)
    audit_split(data_path)
    if args.val_limit:
        data_path = limited_validation_yaml(
            data_path, project_dir / "_smoke_datasets" / args.name, args.val_limit,
        )
        print(f"🧪 冒烟验证集：{args.val_limit} 张 ({data_path})")

    print(f"📦 加载预训练模型: {args.model}")
    model = YOLO(args.model)

    print(f"🚀 开始训练 (epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch})")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
        patience=20,
        save_period=10,
        cache="disk",
        close_mosaic=10,
        degrees=2.0,
        translate=0.03,
        scale=0.20,
        shear=1.0,
        perspective=0.0005,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        rect=not args.no_rect,
        # Synthetic pages contain fewer than 300 labeled objects. Keeping this
        # bounded prevents validation NMS timeouts; production tiled inference
        # uses its own higher max_det setting in detector.py.
        max_det=300,
        conf=args.val_conf,
        cos_lr=True,
        verbose=True,
    )

    # 把 best.pt 拷贝到 weights/
    weights_dir = ROOT / "backend" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_src = Path(model.trainer.best)
    if best_src.exists() and not args.no_promote:
        import shutil
        best_dst = weights_dir / "best.pt"
        shutil.copy(best_src, best_dst)
        print(f"\n✅ 模型已拷贝: {best_dst}")
    elif best_src.exists():
        print(f"\n✅ 实验权重已保留: {best_src}（未覆盖生产权重）")
    else:
        print(f"⚠️ best.pt 不在预期位置: {best_src}")


if __name__ == "__main__":
    main()
