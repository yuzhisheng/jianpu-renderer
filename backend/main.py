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
from pydantic import BaseModel, Field
from PIL import Image
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detector import YoloDetector
from assembler import Assembler, TransformerAssembler
from visual_recognizer import VisualTransformerRecognizer
from accurate_recognizer import (
    AccurateVLMRecognizer,
    AccurateRecognizerBusyError,
    AccurateRecognizerInterruptedError,
    AccurateRecognizerTimeoutError,
)

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
visual_recognizer: Optional[VisualTransformerRecognizer] = None
accurate_recognizer: Optional[AccurateVLMRecognizer] = None


def get_models(load_transformer: bool = False):
    global detector, assembler, transformer
    if detector is None:
        try:
            detector = YoloDetector()
            detector.load()
            logger.info("✅ YOLOv8 模型已加载")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ YOLOv8 未加载: {e}")
            detector = None
    if load_transformer and transformer is None:
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


def get_visual_recognizer():
    global visual_recognizer
    if visual_recognizer is None:
        visual_recognizer = VisualTransformerRecognizer()
        visual_recognizer.load()
        logger.info("✅ 图像到序列 Transformer 已加载")
    return visual_recognizer


def get_accurate_recognizer():
    global accurate_recognizer
    if accurate_recognizer is None:
        accurate_recognizer = AccurateVLMRecognizer()
    return accurate_recognizer


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
        "pitch_refiner_available": bool(
            det is not None and os.path.exists(
                os.environ.get(
                    "JIANPU_PITCH_REFINER_WEIGHTS",
                    str(ROOT / "weights" / "pitch8_domain_mixed_v1.pt"),
                )
            )
        ),
        "pitch_refiner_loaded": bool(
            det is not None and getattr(det, "pitch_model", None) is not None
        ),
        "transformer_loaded": transformer is not None and transformer.model is not None,
        "visual_transformer_loaded": visual_recognizer is not None and visual_recognizer.model is not None,
        "accurate_vlm_available": get_accurate_recognizer().available,
    }


class RecognizeResponse(BaseModel):
    score: dict
    detections: list
    num_detections: int
    inference_ms: float
    src_tokens: list
    tgt_tokens: list
    recognizer: str = "fast"
    confidence: Optional[float] = None
    warnings: list = Field(default_factory=list)
    row_results: list = Field(default_factory=list)
    symbol_summary: dict = Field(default_factory=dict)


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    conf: float = Query(0.20, ge=0.05, le=0.95, description="YOLO 置信度阈值"),
    use_transformer: bool = Query(False, description="实验性序列纠错；默认使用二维几何拼装"),
    visual_sequence: bool = Query(False, description="实验性行图片→CTC/Transformer 序列识别"),
    recognizer: str = Query("accurate", description="accurate=本地视觉大模型；fast=YOLO 几何拼装"),
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

    if recognizer not in {"accurate", "fast", "visual"}:
        raise HTTPException(status_code=400, detail="recognizer 必须是 accurate、fast 或 visual")
    if visual_sequence:
        recognizer = "visual"

    # 加载模型。精确模式仍以低阈值检测框辅助恢复小节线，但音高由 VLM 决定。
    det, asm = get_models(load_transformer=use_transformer)
    if det is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO 模型未加载, 请先训练: python backend/scripts/train_detector.py",
        )

    if (recognizer == "accurate"
            and callable(getattr(det, "is_staff_notation", None))
            and det.is_staff_notation(image)):
        raise HTTPException(
            status_code=422,
            detail="图片是五线谱/吉他谱或其他非数字简谱，当前仅支持数字简谱",
        )

    if recognizer == "accurate":
        enable_pitch_refinement = getattr(det, "enable_pitch_refinement", None)
        if callable(enable_pitch_refinement):
            enable_pitch_refinement()

    # YOLO 检测
    try:
        detector_conf = min(conf, 0.12) if recognizer == "accurate" else conf
        detections, img_w, img_h = await run_in_threadpool(
            det.detect, image, detector_conf)
    except Exception as e:
        logger.error(f"YOLO 检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {e}")

    # 组装 JSON
    try:
        if recognizer == "accurate":
            result = await run_in_threadpool(
                get_accurate_recognizer().recognize, image, detections)
        elif recognizer == "visual":
            result = await run_in_threadpool(
                get_visual_recognizer().predict_page, image, detections)
        else:
            asm_local = Assembler(
                detector=det, transformer=transformer, use_transformer=use_transformer,
            )
            result = await run_in_threadpool(
                asm_local.assemble_from_dets, detections, img_w, img_h)
    except AccurateRecognizerBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except AccurateRecognizerInterruptedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except AccurateRecognizerTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
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
        recognizer=recognizer,
        confidence=result.get("confidence"),
        warnings=result.get("warnings", []),
        row_results=result.get("row_results", []),
        symbol_summary=result.get("symbol_summary", {}),
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
