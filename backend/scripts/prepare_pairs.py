#!/usr/bin/env python3
"""
从 Score JSON 构造 Transformer 监督对
每个训练样本 = (YOLO token 序列, 结构化 JSON token 序列)

YOLO token 序列模拟: 模拟 YOLOv8 检测框的 token 化形式 (按 cy 分行, 行内按 cx 排序)
JSON token 序列: 从原始 Score JSON 按规则路径编码 (事实标签)

输出: pairs.npz  (src_ids, tgt_ids)
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path

from typing import Optional, Union

import numpy as np

# 引入 model
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from model.tokenizer import (
    ALL_TOKENS, TOKEN2ID, ID2TOKEN, VOCAB_SIZE,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID, ROW_ID,
    encode_score, encode_score_with_dynamics, encode_measure, encode_note,
    token_to_id, LYRIC_START, LYRIC_END, LYRIC_BASE,
)
from model.spatial_tokens import detections_to_tokens

# === 类别名 (与 generate_training_pngs.cjs 一致) ===
YOLO_CLASS_NAMES = [
    'pitch_1','pitch_2','pitch_3','pitch_4','pitch_5','pitch_6','pitch_7','rest',
    'dash','underline_1','underline_2','dot','upper_dot','lower_dot',
    'sharp','flat','natural','fermata','tenuto','accent',
    'boyin','chanyin','tie','slur','dayin','tuyin','dieyin','liyin','huayin','yinyin','dunyin',
    'dynamic','bar_single','bar_double','bar_end','bar_repeat_start','bar_repeat_end',
    'repeat_ending','crescendo','descrescendo','lyric','force_accent',
]
assert len(YOLO_CLASS_NAMES) == 42, f"YOLO classes should be 42, got {len(YOLO_CLASS_NAMES)}"

# === YOLO class id -> token 字符串 ===
# (从 YOLO 检测结果 token 化用)
YOLO_CLASS_TO_TOKEN = {
    0: "P1", 1: "P2", 2: "P3", 3: "P4", 4: "P5", 5: "P6", 6: "P7",
    7: "R0",  # rest
    8: "-",   # dash
    9: "_",   # underline_1
    10: "=",  # underline_2
    11: ".",  # dot
    12: "^",  # upper_dot
    13: "v",  # lower_dot
    14: "#", 15: "b", 16: "n",  # accidentals
    17: "FER", 18: "TEN", 19: "ACC",  # fermata, tenuto, accent
    20: "T:bo", 21: "T:ch",  # boyin, chanyin
    22: None,  # tie (跨音符) - 在 token 序列中以 TIE: 形式
    23: None,  # slur (跨音符) - 在 token 序列中以 SLUR: 形式
    24: "T:da", 25: "T:tu",  # dayin, tuyin
    26: "T:di", 27: None, 28: None, 29: None, 30: "T:du",  # dieyin, liyin(方向待补), huayin(方向待补), yinyin(具体音待补), dunyin
    31: None,  # dynamic text - 在 token 序列中以 DYN: 形式
    32: "B|", 33: "B||", 34: "B|]", 35: "B|:", 36: "B:|",  # barlines
    37: None,  # repeat ending - 在 token 序列中以 R: 形式
    38: "CRES", 39: "DECRES",  # crescendo, descrescendo
    40: LYRIC_START,  # lyric start
    41: None,  # force_accent - 在 token 序列中以 F: 形式
}


def parse_yolo_label_file(txt_path: str):
    """
    解析 YOLO 标注文件, 返回 [(class_id, cx, cy, w, h), ...]
    """
    dets = []
    if not os.path.exists(txt_path):
        return dets
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            dets.append((cls_id, cx, cy, w, h))
    return dets


def dets_to_tokens(dets, img_w: int, img_h: int, add_row_sep: bool = True):
    """
    将 YOLO 检测结果 token 化:
    1. 按 cy 聚类分行
    2. 每行按 cx 排序
    3. 每个检测框 → token
    """
    pixel_dets = [
        (cls_id, cx * img_w, cy * img_h, w * img_w, h * img_h, 1.0)
        for cls_id, cx, cy, w, h in dets
    ]
    tokens = detections_to_tokens(pixel_dets)
    return tokens if add_row_sep else [token for token in tokens if token != "<ROW>"]


def yolo_box_to_token(box) -> Union[str, list, None]:
    """单个 YOLO 检测框 → token (字符串)"""
    cls = box["cls"]
    if cls in (0, 1, 2, 3, 4, 5, 6):
        return YOLO_CLASS_TO_TOKEN[cls]  # P1~P7
    if cls == 7:
        return "R0"
    if cls == 8:
        return "-"
    if cls == 9:
        return "_"
    if cls == 10:
        return "="
    if cls in (11, 12, 13):
        return YOLO_CLASS_TO_TOKEN[cls]  # . ^ v
    if cls in (14, 15, 16):
        return YOLO_CLASS_TO_TOKEN[cls]  # # b n
    if cls in (17, 18, 19):
        return YOLO_CLASS_TO_TOKEN[cls]  # FER TEN ACC
    if cls in (20, 21):
        return YOLO_CLASS_TO_TOKEN[cls]  # T:bo T:ch
    if cls == 22:  # tie  - 用 TIE: 占位符, 实际解码靠 score 字段
        return "TIE:?"  # 占位
    if cls == 23:  # slur
        return "SLUR:?"
    if cls == 24:
        return "T:da"
    if cls == 25:
        return "T:tu"
    if cls == 26:
        return "T:di"
    if cls == 27:  # liyin 方向未知
        return "T:li:up"  # 默认
    if cls == 28:  # huayin 方向未知
        return "T:hu:up"
    if cls == 29:  # yinyin 音未知
        return "T:yi:3"
    if cls == 30:
        return "T:du"
    if cls == 31:  # dynamic 文本  - 用 DYN: 占位符
        return "DYN:?"
    if cls in (32, 33, 34, 35, 36):
        return YOLO_CLASS_TO_TOKEN[cls]  # B| B|| B|] B|: B:|
    if cls == 37:  # repeat ending
        return "R1"
    if cls in (38, 39):
        return YOLO_CLASS_TO_TOKEN[cls]  # CRES DECRES
    if cls == 40:  # lyric start marker - 我们用 LY: 开头表示
        return LYRIC_START
    if cls == 41:  # force accent
        return "sf"  # 默认
    return None


def tokenize_to_ids(tokens, max_len=512):
    """token 字符串列表 → id 列表, 截断/补齐"""
    ids = [token_to_id(t) for t in tokens]
    # 截断 (保留 EOS 在末尾)
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [EOS_ID]
    return ids


def build_pair_from_json_and_dets(score: dict, dets, img_w: int, img_h: int, max_src=400, max_tgt=400):
    """
    构建一个训练对:
    src: YOLO 检测 token 序列
    tgt: 从原始 JSON 编码的结构化 token 序列
    """
    # === src: YOLO 检测 token 化 ===
    src_tokens = dets_to_tokens(dets, img_w, img_h)
    src_ids = tokenize_to_ids(src_tokens, max_src)

    # === tgt: 原始 JSON 编码 ===
    tgt_tokens = encode_score_with_dynamics(score)
    tgt_ids = tokenize_to_ids(tgt_tokens, max_tgt)

    return src_ids, tgt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default="public/training",
                        help="包含 PNG / JSON / YOLO txt 的目录")
    parser.add_argument("--output", default="backend/weights/pairs.npz",
                        help="输出 .npz 路径")
    parser.add_argument("--max-src", type=int, default=400)
    parser.add_argument("--max-tgt", type=int, default=400)
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        print(f"❌ 目录不存在: {training_dir}")
        print("请先运行 python scripts/gen_training_data.py")
        print("然后 node scripts/generate_training_pngs.cjs")
        return

    # 收集所有训练样本
    json_files = sorted(training_dir.glob("score_*.json"))
    if not json_files:
        print(f"❌ {training_dir} 下未找到 score_*.json")
        return

    print(f"找到 {len(json_files)} 份 JSON, 开始构造训练对...")

    # 准备 PNG 尺寸缓存
    from PIL import Image

    all_src = []
    all_tgt = []
    skipped = 0

    for i, jf in enumerate(json_files):
        base = jf.stem  # score_0001
        png_path = training_dir / f"{base}.png"
        txt_path = training_dir / f"{base}.txt"

        if not png_path.exists() or not txt_path.exists():
            skipped += 1
            continue

        # 读 PNG 尺寸
        with Image.open(png_path) as img:
            img_w, img_h = img.size

        # 读 JSON
        with open(jf, 'r', encoding='utf-8') as f:
            score = json.load(f)
        # 去掉 title/key 等非序列字段
        score.pop("title", None)
        score.pop("key", None)
        score.pop("tempo", None)
        score.pop("tempoText", None)
        score.pop("introMeasureCount", None)
        # 注入默认 key/timeSignature (如果没有)
        if "key" not in score:
            score["key"] = "C"
        if "timeSignature" not in score:
            score["timeSignature"] = {"numerator": 4, "denominator": 4}

        # 读 YOLO 标注
        dets = parse_yolo_label_file(str(txt_path))

        # 构建对
        src_ids, tgt_ids = build_pair_from_json_and_dets(
            score, dets, img_w, img_h,
            max_src=args.max_src, max_tgt=args.max_tgt,
        )

        all_src.append(src_ids)
        all_tgt.append(tgt_ids)

        if (i + 1) % 200 == 0:
            print(f"  ✓ {i + 1}/{len(json_files)}")

    if not all_src:
        print("❌ 没有任何有效训练对")
        return

    # Padding
    def pad(seq, max_len):
        if len(seq) < max_len:
            return seq + [PAD_ID] * (max_len - len(seq))
        return seq[:max_len]

    src_max = max(len(s) for s in all_src)
    tgt_max = max(len(t) for t in all_tgt)
    src_max = min(src_max, args.max_src)
    tgt_max = min(tgt_max, args.max_tgt)

    print(f"src max len: {src_max}, tgt max len: {tgt_max}")
    print(f"vocab size: {VOCAB_SIZE}")

    src_arr = np.array([pad(s, src_max) for s in all_src], dtype=np.int64)
    tgt_arr = np.array([pad(t, tgt_max) for t in all_tgt], dtype=np.int64)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        src=src_arr,
        tgt=tgt_arr,
        vocab_size=VOCAB_SIZE,
        src_max_len=src_max,
        tgt_max_len=tgt_max,
    )
    print(f"\n✅ Saved {len(all_src)} pairs to {output_path}")
    print(f"   Skipped: {skipped} (no PNG or YOLO txt)")
    print(f"   src shape: {src_arr.shape}, tgt shape: {tgt_arr.shape}")


if __name__ == "__main__":
    main()
