"""
YOLOv8 推理封装
- 加载 best.pt
- 单图推理
- 返回检测结果
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import io

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"


# 类别名, 与 generate_training_pngs.cjs 一致
CLASS_NAMES = [
    'pitch_1','pitch_2','pitch_3','pitch_4','pitch_5','pitch_6','pitch_7','rest',
    'dash','underline_1','underline_2','dot','upper_dot','lower_dot',
    'sharp','flat','natural','fermata','tenuto','accent',
    'boyin','chanyin','tie','slur','dayin','tuyin','dieyin','liyin','huayin','yinyin','dunyin',
    'dynamic','bar_single','bar_double','bar_end','bar_repeat_start','bar_repeat_end',
    'repeat_ending','crescendo','descrescendo','lyric','force_accent',
]
assert len(CLASS_NAMES) == 42

# class_id -> (cx, cy, w, h, conf, class_name)
Detection = Tuple[int, float, float, float, float, str]


class YoloDetector:
    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.weights_path = weights_path or str(WEIGHTS / "best.pt")
        self.device = device
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 未安装, 请: pip install -r backend/requirements.txt")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"YOLO 权重不存在: {self.weights_path}\n"
                "请先训练: python backend/scripts/train_detector.py"
            )
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)
        self._loaded = True

    def detect(self, image: Image.Image, conf_threshold: float = 0.25,
               imgsz: int = 640) -> List[Detection]:
        """
        检测图片中的符号
        返回: [(class_id, cx, cy, w, h, conf), ...]
        坐标为像素值 (cx, cy, w, h)
        """
        if not self._loaded:
            self.load()

        # 转 numpy RGB
        img = np.array(image.convert("RGB"))
        img_h, img_w = img.shape[:2]

        results = self.model.predict(
            img,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                # xywh in pixel coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
                detections.append((cls_id, cx, cy, w, h, conf))

        return detections, img_w, img_h

    @staticmethod
    def class_name(cls_id: int) -> str:
        return CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
