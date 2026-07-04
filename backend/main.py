"""
FastAPI 入口 - 简谱图片识别服务
"""
import os
import sys
import io
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detector import YoloDetector
from assembler import Assembler, TransformerAssembler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jianpu-api")

# === FastAPI app ===
app = FastAPI(
    title="简谱图片识别服务",
    description="上传简谱图片, 返回 Score JSON",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 全局模型 ===
detector: Optional[YoloDetector] = None
assembler: Optional[Assembler] = None
transformer: Optional[TransformerAssembler] = None


def get_models():
    global detector, assembler, transformer
    if detector is None:
        try:
            detector = YoloDetector()
            detector.load()
            logger.info("✅ YOLOv8 模型已加载")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ YOLOv8 未加载: {e}")
            detector = None
    if transformer is None:
        try:
            transformer = TransformerAssembler()
            transformer.load()
            logger.info("✅ Transformer 模型已加载")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Transformer 未加载: {e}")
            transformer = None
    if assembler is None:
        assembler = Assembler(detector=detector, transformer=transformer, use_transformer=transformer is not None)
    return detector, assembler


@app.get("/")
def root():
    return {
        "service": "jianpu-image-recognizer",
        "version": "1.0.0",
        "endpoints": ["/recognize (POST)", "/health (GET)"],
    }


@app.get("/health")
def health():
    det, asm = get_models()
    return {
        "status": "ok",
        "yolo_loaded": det is not None and det._loaded,
        "transformer_loaded": transformer is not None and transformer.model is not None,
    }


class RecognizeResponse(BaseModel):
    score: dict
    detections: list
    num_detections: int
    inference_ms: float
    src_tokens: list
    tgt_tokens: list


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="YOLO 置信度阈值"),
    use_transformer: bool = Query(True, description="是否使用 Transformer 拼装"),
):
    """
    上传图片, 识别为 Score JSON
    """
    t0 = time.time()

    # 读图片
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件超过 10MB")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片读取失败: {e}")

    # 加载模型
    det, asm = get_models()
    if det is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO 模型未加载, 请先训练: python backend/scripts/train_detector.py",
        )

    # YOLO 检测
    try:
        detections, img_w, img_h = det.detect(image, conf_threshold=conf)
    except Exception as e:
        logger.error(f"YOLO 检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {e}")

    # 组装 JSON
    asm_local = Assembler(detector=det, transformer=transformer, use_transformer=use_transformer)
    try:
        result = asm_local.assemble_from_dets(detections, img_w, img_h)
    except Exception as e:
        logger.error(f"拼装失败: {e}")
        raise HTTPException(status_code=500, detail=f"拼装失败: {e}")

    score = result["score"]
    elapsed_ms = (time.time() - t0) * 1000

    # 构造响应
    det_list = [
        {
            "class_id": d[0],
            "class_name": det.class_name(d[0]),
            "cx": d[1],
            "cy": d[2],
            "w": d[3],
            "h": d[4],
            "conf": d[5],
        }
        for d in detections
    ]

    return RecognizeResponse(
        score=score,
        detections=det_list,
        num_detections=len(detections),
        inference_ms=elapsed_ms,
        src_tokens=result["src_tokens"],
        tgt_tokens=result["tgt_tokens"],
    )


@app.on_event("startup")
def on_startup():
    logger.info("🚀 启动简谱识别服务...")
    try:
        get_models()
    except Exception as e:
        logger.warning(f"启动时模型加载失败 (可继续运行): {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
