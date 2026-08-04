#!/usr/bin/env python3
"""Batch-create pitch/bar skeleton silver labels with a local MLX VLM.

Run this script with backend/.venv-vlm. It checkpoints after every content band
and resumes safely, so a long annotation run can be interrupted.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
ALLOWED_TOKENS = {
    *(f"P{value}" for value in range(1, 8)),
    "R0", "-", "_", "=", ".", "^", "v",
    "B|", "B||", "B|:", "B:|", "#", "b", "n", "?",
}
PROMPT = """你是简谱（数字谱）音高骨架标注器。忽略歌词和标题文字，只看数字简谱。
严格返回一个 JSON 对象，不要 Markdown：
{"content_type":"score或metadata或lyrics","voices":[{"label":"main","tokens":[]}],"confidence":0到1,"uncertainties":[]}

规则：
1. 图片没有数字简谱时，content_type 填 metadata 或 lyrics，voices 为空数组。
2. 数字1-7依次写P1到P7；休止符0写R0；增时线写"-"；小节线写B|；终止双线写B||；反复线写B|:或B:|。
   特别注意：休止符0是闭合的椭圆数字，增时线-只是水平短线，两者绝对不能互换。
3. 升、降、还原号写#、b、n并紧跟对应音符。本次不要识别高低音点、附点、减时线、连音线和演奏法，它们由另一个视觉头负责。
4. 必须逐个保留图片中实际出现的数字、0和增时线，按从左到右排列，不能根据歌词补写。
5. 若同一图片带内有上下两个或更多同时声部，每个声部分别放在voices中，顺序从上到下；不要把歌词当声部。
6. 拍号（例如4/4、6/4）的分子和分母不是音符，必须忽略。
7. 不要为了补全输出而重复0或其他符号；每个声部的token数量必须与图片实际符号数量接近。
8. 小节上方反复房子的“1.”、“2.”等编号不是音符，必须忽略；明显小于主旋律数字、写在主旋律数字正上方的替代音或提示音也先忽略，只抄写同一基线上的大号主旋律数字。
9. 看不清就用?占位并写入uncertainties，不要猜。整个JSON应在200个输出token以内。
"""

RELATION_PROMPT = """你是简谱符号关系标注器。忽略标题和歌词，不要重新抄写音符，只分析主旋律数字之间的关系。
严格返回一个 JSON 对象，不要 Markdown：
{"content_type":"score或metadata或lyrics","voices":[{"label":"main","relations":[],"modifiers":[]}],"confidence":0到1,"uncertainties":[]}

规则：
1. 图片没有数字简谱时 voices 为空。
2. note 下标按主旋律的大号数字1-7和休止符0从左到右从0开始；横线、小节线、歌词、反复房子编号、小号提示音都不计数。
3. relations 本阶段只允许 triplet。必须检查弧线中央是否有很小的数字3；只有明确看到小3才写{"type":"triplet","start":0,"end":2}。禁止输出tie或slur，没有小3的弧线全部忽略。
4. modifiers 只记录直接位于数字正上方/正下方的八度圆点和数字右侧附点：{"note":0,"octave":1,"dot":0}。上方一个圆点octave=1，下方一个圆点octave=-1，两个点为±2；右侧一个附点dot=1。弧线、小号数字3和数字笔画都不是八度点。
5. 只记录清晰可见的符号，不要猜。relations 的 start/end 和 modifiers 的 note 必须是有效 note 下标。
6. 一条实际三连音弧线只能生成一条 relation；禁止生成滑动窗口或相邻组合。
"""


def parse_json(text: str):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("response has no JSON object")
    value = json.loads(match.group(0))
    if value.get("content_type") not in {"score", "metadata", "lyrics"}:
        raise ValueError("invalid content_type")
    voices = value.get("voices", [])
    if value["content_type"] == "score" and not voices:
        raise ValueError("score response has no voices")
    for voice in voices:
        normalized = []
        skip_bar_after_colon = False
        for token in voice.get("tokens", []):
            if skip_bar_after_colon and token == "|":
                skip_bar_after_colon = False
                continue
            skip_bar_after_colon = False
            if token in {
                ".", "(", ")", "_", "v", "V", "^", "B", "=", "↓",
                "T", "又", "f",
            }:
                # Pitch-skeleton pass intentionally omits rhythm dots and
                # phrase/articulation marks even when the model volunteers them.
                continue
            if token == ":":
                normalized.append("B:|")
                skip_bar_after_colon = True
                continue
            token = {
                "P0": "R0", "0": "R0", "|": "B|", "||": "B||",
                "|:": "B|:", ":|": "B:|", ":||": "B:|",
                "n-": "-", "R-": "-", "B-": "-", "-)": "-",
                "—": "-", "–": "-",
                "i": "P1", "Ri": "P1", "R1": "P1",
                "B0": "R0", "R0)": "R0",
            }.get(token, token)
            mistaken_rest_pitch = re.fullmatch(r"R([1-7])", token)
            if mistaken_rest_pitch:
                normalized.append(f"P{mistaken_rest_pitch.group(1)}")
                value.setdefault("uncertainties", []).append(
                    f"mistaken rest prefix normalized as pitch: {token}")
                continue
            if token in {"D.S.", "D.C.", "Fine", "rit.", "rit"}:
                continue
            prefixed_decorated_pitch = re.fullmatch(r"P([1-7])[.'~]", token)
            if prefixed_decorated_pitch:
                normalized.append(f"P{prefixed_decorated_pitch.group(1)}")
                continue
            slashed_pitch = re.fullmatch(r"\\([1-7])", token)
            if slashed_pitch:
                normalized.append(f"P{slashed_pitch.group(1)}")
                continue
            decorated_pitch = re.fullmatch(r"([1-7])[~]", token)
            if decorated_pitch:
                normalized.append(f"P{decorated_pitch.group(1)}")
                continue
            pitched_extension = re.fullmatch(r"P?([0-7])(-+)", token)
            if pitched_extension:
                digit = pitched_extension.group(1)
                normalized.append("R0" if digit == "0" else f"P{digit}")
                normalized.extend("-" for _ in pitched_extension.group(2))
                continue
            if re.fullmatch(r"[()\.^'0-7i]+", token):
                digits = [character for character in token if character in "01234567i"]
                if digits:
                    for digit in digits:
                        if digit == "0":
                            normalized.append("R0")
                        else:
                            normalized.append("P1" if digit == "i" else f"P{digit}")
                    if len(digits) > 1:
                        value.setdefault("uncertainties", []).append(
                            f"decorated/compacted digit run split into pitches: {token}")
                    continue
            joined_pitches = re.fullmatch(r"([1-7])\.([1-7])", token)
            if joined_pitches:
                normalized.extend((
                    f"P{joined_pitches.group(1)}",
                    f"P{joined_pitches.group(2)}",
                ))
                continue
            bracketed_pitch = re.fullmatch(r"\[([1-7])\]", token)
            if bracketed_pitch:
                normalized.append(f"P{bracketed_pitch.group(1)}")
                continue
            compacted_pitches = re.fullmatch(r"\(?([0-7i]{2,})\)?", token)
            if compacted_pitches:
                for digit in compacted_pitches.group(1):
                    if digit == "0":
                        normalized.append("R0")
                    else:
                        normalized.append("P1" if digit == "i" else f"P{digit}")
                value.setdefault("uncertainties", []).append(
                    f"compacted digit run split into pitches: {token}")
                continue
            if re.fullmatch(r"-+", token):
                normalized.extend("-" for _ in token)
                continue
            compound = token.split("-")
            if len(compound) > 1 and all(
                    part in {*map(str, range(8)), "X", "x", "?"}
                    for part in compound):
                for part in compound:
                    if part == "0":
                        normalized.append("R0")
                    elif part in {"X", "x", "?"}:
                        normalized.append("?")
                        value.setdefault("uncertainties", []).append(
                            f"non-pitched notehead preserved as ?: {part}")
                    else:
                        normalized.append(f"P{part}")
                continue
            if token in {"X", "x"}:
                normalized.append("?")
                value.setdefault("uncertainties", []).append(
                    f"non-pitched notehead preserved as ?: {token}")
                continue
            dotted_pitch = re.fullmatch(r"\.?([1-7])\.?", token)
            if dotted_pitch:
                # Some models attach an octave dot to the digit even though
                # this pass intentionally records pitch class only.
                normalized.append(f"P{dotted_pitch.group(1)}")
                continue
            if token in {str(value) for value in range(1, 8)}:
                normalized.append(f"P{token}")
                continue
            if re.fullmatch(r"\d+", token):
                # A pitch skeleton has no numeric symbols outside 0..7. Keep
                # the sequence position explicit without pretending that a
                # time-signature digit or a hallucinated glyph is a pitch.
                normalized.append("?")
                value.setdefault("uncertainties", []).append(
                    f"out-of-range numeric token preserved as ?: {token}")
                continue
            accidental = re.fullmatch(r"([#bn])([1-7])", token)
            if accidental:
                normalized.extend((f"P{accidental.group(2)}", accidental.group(1)))
                continue
            accidental = re.fullmatch(r"([1-7])([#bn])", token)
            if accidental:
                normalized.extend((f"P{accidental.group(1)}", accidental.group(2)))
                continue
            normalized.append(token)
        voice["tokens"] = normalized
        unknown = [token for token in normalized if token not in ALLOWED_TOKENS]
        if unknown:
            raise ValueError(f"unknown tokens: {unknown}")
    return value


def save(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="backend/real_annotations")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--only", default=None,
        help="只处理指定 annotation_id；多个 ID 用逗号分隔",
    )
    parser.add_argument("--limit", type=int, default=0, help="本次最多新增多少个内容带")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument(
        "--repair-only", action="store_true",
        help="只用当前解析规则恢复已有原始响应，不执行新的 VLM 推理",
    )
    args = parser.parse_args()

    root = (ROOT / args.annotations).resolve()
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    pages = [item for item in manifest
             if item["split"] == args.split and item["kind"] == "jianpu"]
    if args.only:
        selected = {item.strip() for item in args.only.split(",") if item.strip()}
        pages = [item for item in pages if item["annotation_id"] in selected]

    output = root / "local_vlm_pitch_reviews"
    output.mkdir(parents=True, exist_ok=True)
    model = processor = formatted_prompt = None
    added = 0

    for page_index, page in enumerate(pages, 1):
        annotation_id = page["annotation_id"]
        detector_review = json.loads((root / page["review"]).read_text())
        target = output / f"{annotation_id}.json"
        if target.exists():
            payload = json.loads(target.read_text())
        else:
            payload = {
                "annotation_id": annotation_id,
                "model": args.model,
                "label_grade": "silver_local_vlm_pitch_skeleton",
                "status": "in_progress",
                "rows": [],
            }
        # Parser improvements can repair a previously rejected deterministic
        # response without spending another VLM inference pass.
        for old in payload["rows"]:
            if "error" not in old or not old.get("raw_response"):
                continue
            try:
                repaired = parse_json(old["raw_response"])
            except Exception:
                continue
            old.pop("error", None)
            old.update(repaired)
        save(target, payload)
        completed = {row["source_row"] for row in payload["rows"]
                     if "error" not in row}

        if args.repair_only:
            if all(row["row"] in completed for row in detector_review["rows"]):
                payload["status"] = "complete"
            save(target, payload)
            continue

        for row in detector_review["rows"]:
            row_number = row["row"]
            if row_number in completed:
                continue
            if args.limit and added >= args.limit:
                print(f"limit reached: added={added}")
                return
            image_path = root / row["image"]
            started = time.perf_counter()
            raw_response = None
            try:
                if model is None:
                    print(f"loading {args.model}")
                    model, processor = load(args.model)
                    config = load_config(args.model)
                    formatted_prompt = apply_chat_template(
                        processor, config, PROMPT, num_images=1,
                        thinking_mode="disabled")
                result = generate(
                    model, processor, formatted_prompt, image=str(image_path),
                    max_tokens=args.max_tokens, temperature=0.0, verbose=False,
                    repetition_penalty=args.repetition_penalty,
                    repetition_context_size=64,
                )
                raw_response = result.text
                parsed = parse_json(raw_response)
                record = {
                    "source_row": row_number,
                    "image": row["image"],
                    **parsed,
                    "generation_seconds": round(time.perf_counter() - started, 2),
                    "raw_response": raw_response,
                }
            except Exception as exc:
                record = {
                    "source_row": row_number,
                    "image": row["image"],
                    "error": str(exc),
                    "generation_seconds": round(time.perf_counter() - started, 2),
                    "raw_response": raw_response,
                }
            payload["rows"] = [item for item in payload["rows"]
                               if item["source_row"] != row_number]
            payload["rows"].append(record)
            payload["rows"].sort(key=lambda item: item["source_row"])
            save(target, payload)
            added += 1
            label = record.get("content_type", "error")
            voices = len(record.get("voices", []))
            print(f"[{page_index:02d}/{len(pages)}] {annotation_id} row={row_number}: "
                  f"{label}, voices={voices}, {record['generation_seconds']:.1f}s")

        if all(row["row"] in {item["source_row"] for item in payload["rows"]
                              if "error" not in item}
               for row in detector_review["rows"]):
            payload["status"] = "complete"
            save(target, payload)

    print(f"complete: pages={len(pages)}, added_rows={added}")


if __name__ == "__main__":
    main()
