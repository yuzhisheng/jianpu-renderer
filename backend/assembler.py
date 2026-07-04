"""
检测结果 → Score JSON 拼装
- 输入: YOLOv8 检测列表 (class_id, cx, cy, w, h, conf)
- 处理: 几何分组 + Transformer 预测 + 解析
- 输出: Score JSON
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.tokenizer import (
    ALL_TOKENS, TOKEN2ID, ID2TOKEN, VOCAB_SIZE,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID, ROW_ID,
    LYRIC_START, LYRIC_END, LYRIC_BASE,
    PITCH_TOKENS, REST_TOKEN, OCTAVE_TOKENS, ACCIDENTAL_TOKENS,
    DOT_TOKEN, UPPER_DOT_TOKEN, LOWER_DOT_TOKEN,
    DASH_TOKEN, UNDERLINE1_TOKEN, UNDERLINE2_TOKEN,
    DUR_TOKENS, DYN_TOKENS, FORCE_ACCENT_TOKENS, TECH_TOKENS,
    TENUTO_TOKEN, FERMATA_TOKEN, ACCENT_TOKEN,
    BAR_TOKENS, REPEAT_NUM_TOKENS, CRESC_TOKEN, DECRES_TOKEN,
)
from model.transformer import JianpuTransformer  # noqa: F401  (only used when transformer is loaded)

# === YOLO class_id → token 字符串映射 (与 prepare_pairs.py 保持一致) ===
YOLO_CLASS_TO_TOKEN = {
    0: "P1", 1: "P2", 2: "P3", 3: "P4", 4: "P5", 5: "P6", 6: "P7",
    7: "R0", 8: "-", 9: "_", 10: "=",
    11: ".", 12: "^", 13: "v",
    14: "#", 15: "b", 16: "n",
    17: "FER", 18: "TEN", 19: "ACC",
    20: "T:bo", 21: "T:ch",
    22: "TIE:?",
    23: "SLUR:?",
    24: "T:da", 25: "T:tu", 26: "T:di",
    27: "T:li:up", 28: "T:hu:up", 29: "T:yi:3", 30: "T:du",
    31: "DYN:?",
    32: "B|", 33: "B||", 34: "B|]", 35: "B|:", 36: "B:|",
    37: "R1",
    38: "CRES", 39: "DECRES",
    40: LYRIC_START,
    41: "sf",
}


def dets_to_yolo_tokens(dets, img_w: int, img_h: int) -> List[str]:
    """
    YOLO 检测结果 → token 序列 (与 prepare_pairs.py 一致)
    """
    if not dets:
        return ["<BOS>", "<EOS>"]

    boxes = []
    for cls_id, cx, cy, w, h, conf in dets:
        boxes.append({"cls": cls_id, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})

    # 按 cy 聚类分行
    boxes.sort(key=lambda b: b["cy"])
    heights = [b["h"] for b in boxes]
    median_h = sorted(heights)[len(heights) // 2]
    row_thresh = max(median_h * 1.5, 15)
    rows: List[List] = []
    for b in boxes:
        placed = False
        for row in rows:
            row_cy = sum(r["cy"] for r in row) / len(row)
            if abs(b["cy"] - row_cy) < row_thresh:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    tokens = ["<BOS>"]
    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            tokens.append("<ROW>")
        row.sort(key=lambda b: b["cx"])
        for b in row:
            tok = YOLO_CLASS_TO_TOKEN.get(b["cls"])
            if tok is None:
                continue
            if isinstance(tok, list):
                tokens.extend(tok)
            else:
                tokens.append(tok)
    tokens.append("<EOS>")
    return tokens


# === 状态机: token 流 → Score JSON ===
def parse_tokens_to_score(tokens: List[str]) -> Dict[str, Any]:
    """
    将结构化 token 流解析为 Score JSON
    简化策略: 按 token 顺序识别 Note, 然后按 B| 系列切分小节
    """
    score = {
        "title": "识别结果",
        "key": "C",
        "timeSignature": {"numerator": 4, "denominator": 4},
        "measures": [],
    }

    current_measure: Dict[str, Any] = {"notes": []}
    current_note: Optional[Dict[str, Any]] = None
    in_lyric = False
    lyric_buf: List[str] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("<BOS>", "<EOS>", "<ROW>"):
            i += 1
            continue

        if tok in BAR_TOKENS:
            if tok == "B|":
                current_measure["barline"] = "single"
            elif tok == "B||":
                current_measure["barline"] = "double"
            elif tok == "B|]":
                current_measure["barline"] = "end"
            elif tok == "B|:":
                current_measure["barline"] = "repeat-start"
            elif tok == "B:|":
                current_measure["barline"] = "repeat-end"
            # 提交小节
            if current_note is not None:
                current_measure["notes"].append(current_note)
                current_note = None
            if current_measure["notes"] or current_measure.get("barline"):
                score["measures"].append(current_measure)
            current_measure = {"notes": []}
            i += 1
            continue

        if tok in REPEAT_NUM_TOKENS:
            n = int(tok[1:])
            if "repeatEnding" not in current_measure:
                current_measure["repeatEnding"] = {"numbers": []}
            current_measure["repeatEnding"]["numbers"].append(n)
            i += 1
            continue

        if tok == CRESC_TOKEN:
            current_measure["dynamics"] = {"type": "crescendo"}
            i += 1
            continue
        if tok == DECRES_TOKEN:
            current_measure["dynamics"] = {"type": "descrescendo"}
            i += 1
            continue

        if tok == LYRIC_START:
            in_lyric = True
            lyric_buf = []
            i += 1
            continue
        if tok == LYRIC_END:
            in_lyric = False
            if current_note is not None:
                current_note["lyric"] = "".join(lyric_buf)
            lyric_buf = []
            i += 1
            continue
        if in_lyric:
            if tok.startswith(LYRIC_BASE):
                lyric_buf.append(tok[len(LYRIC_BASE):])
            i += 1
            continue

        # 音高数字
        if tok in PITCH_TOKENS:
            # 提交上一个 note
            if current_note is not None:
                current_measure["notes"].append(current_note)
            pitch = int(tok[1:])
            current_note = {"pitch": pitch, "duration": 1.0}
            i += 1
            continue

        if tok == REST_TOKEN:
            if current_note is not None:
                current_measure["notes"].append(current_note)
            current_note = {"pitch": 0, "duration": 1.0}
            i += 1
            continue

        # 八度
        if tok in OCTAVE_TOKENS:
            if current_note is not None:
                oct = 1 if tok == "O+1" else -1
                current_note["octave"] = current_note.get("octave", 0) + oct
            i += 1
            continue

        # 升降号
        if tok in ("#", "b", "n"):
            if current_note is not None:
                current_note["accidental"] = {"#": "sharp", "b": "flat", "n": "natural"}[tok]
            i += 1
            continue

        # 附点
        if tok == DOT_TOKEN:
            if current_note is not None:
                current_note["dot"] = current_note.get("dot", 0) + 1
            i += 1
            continue

        # 八度点
        if tok == UPPER_DOT_TOKEN:
            if current_note is not None:
                current_note["octave"] = current_note.get("octave", 0) + 1
            i += 1
            continue
        if tok == LOWER_DOT_TOKEN:
            if current_note is not None:
                current_note["octave"] = current_note.get("octave", 0) - 1
            i += 1
            continue

        # 增时线
        if tok == DASH_TOKEN:
            if current_note is not None:
                current_measure["notes"].append(current_note)
            current_note = None
            current_measure["notes"].append({"type": "dash", "duration": 0.5})
            i += 1
            continue

        # 减时线
        if tok == UNDERLINE1_TOKEN:
            if current_note is not None:
                current_note["duration"] = min(current_note.get("duration", 1.0), 0.5)
            i += 1
            continue
        if tok == UNDERLINE2_TOKEN:
            if current_note is not None:
                current_note["duration"] = min(current_note.get("duration", 1.0), 0.25)
            i += 1
            continue

        # 时值
        if tok in DUR_TOKENS:
            if current_note is not None:
                dur_map = {
                    "D4": 4.0, "D2": 2.0, "D1": 1.0, "D0.5": 0.5,
                    "D0.25": 0.25, "D0.125": 0.125, "D1.5": 1.5, "D0.75": 0.75,
                }
                current_note["duration"] = dur_map.get(tok, current_note.get("duration", 1.0))
            i += 1
            continue

        # 力度
        if tok in DYN_TOKENS:
            if current_note is not None:
                current_note["dynamic"] = tok
            i += 1
            continue

        # 力度突变
        if tok in FORCE_ACCENT_TOKENS:
            if current_note is not None:
                current_note["forceAccent"] = tok
            i += 1
            continue

        # 技巧
        if tok in TECH_TOKENS:
            if current_note is not None:
                tech = parse_tech_token(tok)
                if "techniques" not in current_note:
                    current_note["techniques"] = []
                current_note["techniques"].append(tech)
            i += 1
            continue

        # 标记
        if tok == TENUTO_TOKEN:
            if current_note is not None:
                current_note["tenuto"] = True
            i += 1
            continue
        if tok == FERMATA_TOKEN:
            if current_note is not None:
                current_note["fermata"] = True
            i += 1
            continue
        if tok == ACCENT_TOKEN:
            if current_note is not None:
                current_note["accent"] = True
            i += 1
            continue

        # 占位符 (TIE:?, SLUR:?, DYN:?) 跳过
        i += 1

    # 收尾
    if current_note is not None:
        current_measure["notes"].append(current_note)
    if current_measure["notes"] or current_measure.get("barline"):
        score["measures"].append(current_measure)

    return score


def parse_tech_token(tok: str) -> Dict[str, Any]:
    """T:xxx 形式 → DiziTechnique dict"""
    if tok == "T:bo":
        return {"type": "boyin"}
    if tok == "T:ch":
        return {"type": "chanyin"}
    if tok == "T:da":
        return {"type": "dayin"}
    if tok == "T:tu":
        return {"type": "tuyin"}
    if tok == "T:di":
        return {"type": "dieyin"}
    if tok.startswith("T:li:"):
        return {"type": "liyin", "liyinDirection": tok.split(":")[2]}
    if tok.startswith("T:hu:"):
        return {"type": "huayin", "slideDirection": tok.split(":")[2]}
    if tok.startswith("T:yi:"):
        pitch = int(tok.split(":")[2])
        return {"type": "yinyin", "graceNotes": [pitch], "graceOctave": 0}
    if tok == "T:du":
        return {"type": "dunyin"}
    return {"type": "boyin"}


# === Transformer 推理 ===
class TransformerAssembler:
    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.weights_path = weights_path or str(ROOT / "weights" / "transformer.pt")
        self.device = torch.device(device)
        self.model = None
        self.config = None

    def load(self):
        if self.model is not None:
            return
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Transformer 权重不存在: {self.weights_path}\n"
                "请先训练: python backend/scripts/train_transformer.py"
            )
        ckpt = torch.load(self.weights_path, map_location=self.device)
        self.config = ckpt.get("config", {})
        self.model = JianpuTransformer(
            d_model=self.config.get("d_model", 128),
            nhead=self.config.get("nhead", 4),
            num_encoder_layers=self.config.get("num_encoder_layers", 2),
            num_decoder_layers=self.config.get("num_decoder_layers", 2),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, src_tokens: List[str], max_len: int = 256) -> List[str]:
        """Transformer 推理: src tokens → tgt tokens"""
        if self.model is None:
            self.load()
        # token → ids
        src_ids = [TOKEN2ID.get(t, UNK_ID) for t in src_tokens]
        # padding / 截断
        max_src = self.model.max_len
        if len(src_ids) > max_src:
            src_ids = src_ids[:max_src - 1] + [EOS_ID]
        else:
            src_ids = src_ids + [PAD_ID] * (max_src - len(src_ids))
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(src, max_len=max_len, bos_id=BOS_ID, eos_id=EOS_ID)
        ids = out[0].tolist()
        # 截断到 EOS
        if EOS_ID in ids:
            ids = ids[:ids.index(EOS_ID) + 1]
        tokens = [ID2TOKEN.get(i, "<UNK>") for i in ids]
        return tokens


# === 顶层组装器 ===
class Assembler:
    """
    组合:
    1. YOLO 检测 → token 序列 (src)
    2. Transformer 推理 → 结构化 token 序列 (tgt)
    3. 状态机解析 → Score JSON
    """

    def __init__(self, detector=None, transformer=None, use_transformer: bool = True):
        self.detector = detector
        self.transformer = transformer
        self.use_transformer = use_transformer

    def assemble_from_dets(
        self,
        dets,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        """从 YOLO 检测结果直接组装 Score JSON"""
        # 1. token 化
        src_tokens = dets_to_yolo_tokens(dets, img_w, img_h)

        # 2. 推理
        if self.use_transformer and self.transformer is not None:
            try:
                tgt_tokens = self.transformer.predict(src_tokens)
            except Exception as e:
                print(f"⚠️ Transformer 推理失败: {e}, 降级到规则拼装")
                tgt_tokens = self._fallback_rule_assembly(dets, img_w, img_h)
        else:
            tgt_tokens = self._fallback_rule_assembly(dets, img_w, img_h)

        # 3. 解析
        score = parse_tokens_to_score(tgt_tokens)
        return {"score": score, "src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

    def _fallback_rule_assembly(self, dets, img_w: int, img_h: int) -> List[str]:
        """规则降级: 直接从 YOLO token 序列识别音高+时值"""
        src_tokens = dets_to_yolo_tokens(dets, img_w, img_h)
        # 过滤掉 <BOS>/<EOS>/<ROW> 后用 parse_tokens_to_score
        # 由于 YOLO tokens 已包含音高/八度/技巧/小节线, 可直接解析
        return src_tokens
