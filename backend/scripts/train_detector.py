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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="public/training/data.yaml",
                        help="data.yaml 路径")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="预训练模型 (会自动下载)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cuda:0 或 cpu")
    parser.add_argument("--project", default="backend/weights/runs")
    parser.add_argument("--name", default="yolov8s_jianpu")
    parser.add_argument("--exist-ok", action="store_true", default=True)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics 未安装, 请先: pip install -r backend/requirements.txt")
        sys.exit(1)

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"❌ data.yaml 不存在: {data_path}")
        print("请先运行 node scripts/generate_training_pngs.cjs")
        sys.exit(1)

    print(f"📦 加载预训练模型: {args.model}")
    model = YOLO(args.model)

    print(f"🚀 开始训练 (epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch})")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        patience=10,
        save_period=10,
        verbose=True,
    )

    # 把 best.pt 拷贝到 weights/
    weights_dir = ROOT / "backend" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_src = Path(args.project) / args.name / "weights" / "best.pt"
    if best_src.exists():
        import shutil
        best_dst = weights_dir / "best.pt"
        shutil.copy(best_src, best_dst)
        print(f"\n✅ 模型已拷贝: {best_dst}")
    else:
        print(f"⚠️ best.pt 不在预期位置: {best_src}")


if __name__ == "__main__":
    main()
