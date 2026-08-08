"""
FastAPI 入口 - 简谱图片识别服务
"""
import os
import sys
import io
import time
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
from recognition_history import RecognitionHistory
from pdf_utils import is_pdf, render_pdf_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jianpu-api")

# === FastAPI app ===
app = FastAPI(
    title="简谱图片识别服务",
    description="上传简谱图片或 PDF，返回 Score JSON",
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
recognition_history = RecognitionHistory()


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
        "endpoints": [
            "/recognize (POST)", "/recognition-history (GET)",
            "/recognition-history/{id} (GET)", "/health (GET)",
        ],
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
        "recognition_history_available": recognition_history.db_path.exists(),
    }


@app.get("/recognition-history")
def list_recognition_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return recent recognition attempts, newest first."""
    return {"items": recognition_history.list(limit=limit, offset=offset)}


@app.get("/recognition-history/{record_id}")
def get_recognition_history(record_id: str):
    record = recognition_history.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="识别记录不存在")
    return record


@app.get("/recognition-history/{record_id}/image")
def get_recognition_history_image(record_id: str):
    path = recognition_history.image_path(record_id)
    if path is None:
        raise HTTPException(status_code=404, detail="该记录没有保存原图")
    return FileResponse(path)


class RecognizeResponse(BaseModel):
    recognition_id: Optional[str] = None
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
    file_type: str = "image"
    page_count: int = 1
    page_results: list = Field(default_factory=list)


async def recognize_page_image(
    image: Image.Image,
    *,
    conf: float,
    use_transformer: bool,
    recognizer: str,
):
    """Run the existing single-image pipeline for one rasterized page."""
    det, _ = get_models(load_transformer=use_transformer)
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

    detector_conf = min(conf, 0.12) if recognizer == "accurate" else conf
    try:
        detections, img_w, img_h = await run_in_threadpool(
            det.detect, image, detector_conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"检测失败: {exc}") from exc

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
    except (AccurateRecognizerBusyError,
            AccurateRecognizerInterruptedError,
            AccurateRecognizerTimeoutError):
        raise
    except Exception as exc:
        raise RuntimeError(f"拼装失败: {exc}") from exc

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
    return result, det_list, image.width, image.height


def merge_page_scores(scores: list[dict]) -> dict:
    """Join page scores into one renderable score while preserving page breaks."""
    if not scores:
        return {"title": "识别结果", "measures": []}
    merged = deepcopy(scores[0])
    for page_score in scores[1:]:
        page_measures = deepcopy(page_score.get("measures", []))
        if page_measures:
            page_measures[0]["lineBreakBefore"] = True
            merged.setdefault("measures", []).extend(page_measures)
        if isinstance(merged.get("parts"), list) and isinstance(page_score.get("parts"), list):
            for part_index, part in enumerate(page_score["parts"]):
                if part_index >= len(merged["parts"]):
                    merged["parts"].append(deepcopy(part))
                else:
                    target = merged["parts"][part_index]
                    extra = deepcopy(part.get("measures", []))
                    if extra:
                        extra[0]["lineBreakBefore"] = True
                        target.setdefault("measures", []).extend(extra)
    return merged


def aggregate_symbol_summary(results: list[dict]) -> dict:
    aggregate: dict = {}
    for result in results:
        summary = result.get("symbol_summary", {})
        if not isinstance(summary, dict):
            continue
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                aggregate[key] = aggregate.get(key, 0) + value
            elif key not in aggregate:
                aggregate[key] = value
    return aggregate


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    conf: float = Query(0.20, ge=0.05, le=0.95, description="YOLO 置信度阈值"),
    use_transformer: bool = Query(False, description="实验性序列纠错；默认使用二维几何拼装"),
    visual_sequence: bool = Query(False, description="实验性行图片→CTC/Transformer 序列识别"),
    recognizer: str = Query("accurate", description="accurate=本地视觉大模型；fast=YOLO 几何拼装"),
):
    """
    上传图片或多页 PDF，识别为 Score JSON
    """
    t0 = time.time()
    history_id: Optional[str] = None

    def record_failure(detail: object) -> None:
        if history_id is None:
            return
        try:
            recognition_history.fail(
                history_id, str(detail), (time.time() - t0) * 1000)
        except Exception as history_error:
            logger.warning("识别失败记录写入失败: %s", history_error)

    # 读图片或 PDF。PDF 页面会先栅格化，再逐页复用同一识别管线。
    try:
        contents = await file.read()
        pdf_input = is_pdf(contents, file.filename)
        max_upload_size = 50 * 1024 * 1024 if pdf_input else 10 * 1024 * 1024
        try:
            history_id = recognition_history.begin(
                file.filename, contents if len(contents) <= max_upload_size else b"",
                recognizer="visual" if visual_sequence else recognizer,
                confidence=conf,
            )
        except Exception as history_error:
            logger.warning("识别开始记录写入失败: %s", history_error)
        if len(contents) > max_upload_size:
            record_failure(f"文件超过 {max_upload_size // (1024 * 1024)}MB")
            raise HTTPException(
                status_code=400,
                detail=f"文件超过 {max_upload_size // (1024 * 1024)}MB",
            )
        if pdf_input:
            pages = await run_in_threadpool(render_pdf_pages, contents)
        else:
            pages = [Image.open(io.BytesIO(contents)).convert("RGB")]
        if not pages:
            raise ValueError("文件没有可识别的页面")
        try:
            recognition_history.set_dimensions(history_id, pages[0].width, pages[0].height)
        except Exception as history_error:
            logger.warning("识别图片尺寸记录写入失败: %s", history_error)
    except HTTPException:
        raise
    except Exception as e:
        record_failure(f"文件读取或 PDF 转换失败: {e}")
        raise HTTPException(status_code=400, detail=f"文件读取或 PDF 转换失败: {e}")

    if recognizer not in {"accurate", "fast", "visual"}:
        record_failure("recognizer 必须是 accurate、fast 或 visual")
        raise HTTPException(status_code=400, detail="recognizer 必须是 accurate、fast 或 visual")
    if visual_sequence:
        recognizer = "visual"

    page_outputs = []
    try:
        for page_index, image in enumerate(pages):
            result, det_list, page_width, page_height = await recognize_page_image(
                image,
                conf=conf,
                use_transformer=use_transformer,
                recognizer=recognizer,
            )
            page_outputs.append({
                "page": page_index + 1,
                "width": page_width,
                "height": page_height,
                "result": result,
                "detections": det_list,
            })
    except HTTPException as e:
        record_failure(e.detail)
        raise
    except AccurateRecognizerBusyError as e:
        record_failure(e)
        raise HTTPException(status_code=409, detail=str(e))
    except AccurateRecognizerInterruptedError as e:
        record_failure(e)
        raise HTTPException(status_code=503, detail=str(e))
    except AccurateRecognizerTimeoutError as e:
        record_failure(e)
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.error(f"拼装失败: {e}")
        record_failure(f"拼装失败: {e}")
        raise HTTPException(status_code=500, detail=f"拼装失败: {e}")

    page_scores = [page["result"]["score"] for page in page_outputs]
    score = merge_page_scores(page_scores)
    page_count = len(page_outputs)
    page_results = []
    all_detections = []
    all_src_tokens = []
    all_tgt_tokens = []
    all_warnings = []
    confidences = []
    for page in page_outputs:
        result = page["result"]
        page_index = page["page"]
        all_detections.extend(
            [{**detection, "page": page_index} for detection in page["detections"]])
        all_src_tokens.extend(result.get("src_tokens", []))
        all_tgt_tokens.extend(result.get("tgt_tokens", []))
        all_warnings.extend(result.get("warnings", []))
        if isinstance(result.get("confidence"), (int, float)):
            confidences.append(float(result["confidence"]))
        if page_count > 1:
            page_results.append({
                "page": page_index,
                "width": page["width"],
                "height": page["height"],
                "score": result.get("score", {}),
                "row_results": result.get("row_results", []),
                "symbol_summary": result.get("symbol_summary", {}),
            })
    elapsed_ms = (time.time() - t0) * 1000

    response = RecognizeResponse(
        recognition_id=history_id,
        score=score,
        detections=all_detections,
        num_detections=len(all_detections),
        inference_ms=elapsed_ms,
        src_tokens=all_src_tokens,
        tgt_tokens=all_tgt_tokens,
        recognizer=recognizer,
        confidence=(sum(confidences) / len(confidences) if confidences else None),
        warnings=list(dict.fromkeys(all_warnings)),
        row_results=(page_outputs[0]["result"].get("row_results", [])
                     if page_count == 1 else page_results),
        symbol_summary=aggregate_symbol_summary(
            [page["result"] for page in page_outputs]),
        file_type="pdf" if pdf_input else "image",
        page_count=page_count,
        page_results=page_results,
    )
    try:
        response_payload = response.model_dump()
    except AttributeError:
        response_payload = response.dict()
    try:
        recognition_history.complete(history_id, response_payload, elapsed_ms)
    except Exception as history_error:
        logger.warning("识别结果记录写入失败: %s", history_error)
    return response


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
