#!/usr/bin/env python3
"""随机简谱生成器 — 输出合法的 Score JSON"""
import json, random, argparse, sys
from typing import Any

NOTES = [1, 2, 3, 4, 5, 6, 7]
LYRICS_POOL = [
    ("我", "wǒ"), ("你", "nǐ"), ("心", "xīn"), ("爱", "ài"),
    ("花", "huā"), ("月", "yuè"), ("风", "fēng"), ("云", "yún"),
    ("山", "shān"), ("水", "shuǐ"), ("天", "tiān"), ("地", "dì"),
    ("人", "rén"), ("梦", "mèng"), ("光", "guāng"), ("星", "xīng"),
    ("春", "chūn"), ("秋", "qiū"), ("冬", "dōng"), ("夏", "xià"),
    ("红", "hóng"), ("白", "bái"), ("青", "qīng"), ("蓝", "lán"),
    ("美", "měi"), ("好", "hǎo"), ("甜", "tián"), ("香", "xiāng"),
    ("飞", "fēi"), ("走", "zǒu"), ("来", "lái"), ("去", "qù"),
]
SLUR_POOL = [
    ("s1", ["唉", "哟"], ["āi", "yō"], [(5,0), (3,0)]),
    ("s2", ["轻", "轻"], ["qīng", "qīng"], [(3,0), (5,0)]),
    ("s3", ["微", "风"], ["wēi", "fēng"], [(2,0), (3,0)]),
    ("s4", ["花", "开"], ["huā", "kāi"], [(1,1), (6,0)]),
    ("s5", ["流", "水"], ["liú", "shuǐ"], [(6,0), (5,0)]),
    ("s6", ["月", "光"], ["yuè", "guāng"], [(5,0), (3,0)]),
]


def make_note(pitch: int, dur: float, lyric: str = "", py: str = "", octave: int = 0, **kw) -> dict:
    n: dict[str, Any] = {"pitch": pitch, "duration": dur}
    if lyric:
        n["lyric"] = lyric
        if py:
            n["lyrics"] = [lyric, py]
    if octave != 0:
        n["octave"] = octave
    for k, v in kw.items():
        if v:
            n[k] = v
    return n


def make_dash(dur: float) -> dict:
    return {"type": "dash", "duration": dur}


def fill_measure(beats: int = 4) -> tuple[list[dict], list[dict]]:
    """返回 (整小节note列表, slur定义列表) 总=beats拍"""
    remaining = beats
    notes = []
    slurs = []
    beat_pos = 0

    while remaining > 0.001:
        # 随机选节奏型
        r = random.random()
        if remaining < 0.5:
            dur = remaining
        elif r < 0.3:
            dur = 1.0
        elif r < 0.55:
            dur = 0.5
        elif r < 0.75:
            dur = 0.25
        elif r < 0.9:
            dur = 1.5
        else:
            dur = 2.0

        dur = min(dur, remaining)
        remaining -= dur

        if dur <= 0.01:
            break

        # 随机决定是否用休止符
        if random.random() < 0.08 and dur >= 0.5:
            notes.append({"pitch": 0, "duration": dur})
            beat_pos += dur
            continue

        # 随机决定是否用增时线
        if random.random() < 0.12 and dur >= 1.0 and len(notes) > 0 and notes[-1].get("pitch") != 0:
            notes.append(make_dash(dur))
            beat_pos += dur
            continue

        pitch = random.choice(NOTES)
        octave = 0
        if random.random() < 0.2:
            octave = random.choice([-1, 1])
        elif random.random() < 0.05:
            octave = random.choice([-2, 2])

        lyric, py = random.choice(LYRICS_POOL)

        extra: dict[str, Any] = {}
        # 波音
        if random.random() < 0.08:
            extra["techniques"] = [{"type": "boyin"}]
        # 倚音
        if random.random() < 0.06:
            gp = random.choice(NOTES)
            extra["techniques"] = (extra.get("techniques") or []) + [{"type": "yinyin", "graceNotes": [gp]}]
        # 连音线（相邻两个0.5拍的音符）
        if dur == 0.5 and len(notes) > 0 and notes[-1].get("duration") == 0.5 and random.random() < 0.3:
            sid = f"s{len(slurs) + 1}"
            slurs.append({
                "id": sid,
                "lyrics": [lyric, py],
                "prev_lyric": notes[-1].get("lyric", lyric),
            })
            notes[-1]["slurId"] = sid
            extra["slurId"] = sid

        # 力度/装饰
        if random.random() < 0.06:
            extra["accent"] = True
        if random.random() < 0.04:
            extra["tenuto"] = True
        if random.random() < 0.03:
            extra["fermata"] = True
        if random.random() < 0.05 and dur >= 1.0:
            extra["dot"] = 1

        notes.append(make_note(pitch, dur, lyric, py, octave, **extra))
        beat_pos += dur

    return notes, slurs


def generate(args):
    measures = []
    total = args.measures or random.randint(12, 24)

    for _ in range(total):
        notes, _ = fill_measure(args.beats or random.choice([2, 3, 4]))
        m: dict[str, Any] = {"notes": notes}
        # 随机小节线
        if random.random() < 0.08:
            m["barline"] = random.choice(["double", "repeat-start", "repeat-end", "end"])
        if random.random() < 0.03:
            m["dynamics"] = {"type": random.choice(["crescendo", "descrescendo"])}
        measures.append(m)

    # 随机加小房子
    if len(measures) >= 3 and random.random() < 0.5:
        pos = random.randint(len(measures) - 3, len(measures) - 2)
        measures.insert(pos, dict(measures[pos - 1]) if pos > 0 else measures[0])
        measures[pos - 1]["barline"] = "repeat-start"
        measures[pos]["repeatEnding"] = {"numbers": [1]}
        measures[pos + 1]["repeatEnding"] = {"numbers": [2]}
        measures[pos + 1]["barline"] = "end"

    # 构建 score
    key = args.key or random.choice(["C", "D", "G", "F", "A"])
    tempo = args.tempo or random.randint(60, 120)
    title = args.title or f"随机简谱 #{random.randint(100, 999)}"

    score = {
        "title": title,
        "key": key,
        "timeSignature": {"numerator": args.beats or 4, "denominator": 4},
        "tempo": tempo,
        "measures": measures,
    }

    if args.intro:
        intro_count = min(2, len(measures))
        score["introMeasureCount"] = intro_count

    return score


def main():
    parser = argparse.ArgumentParser(description="随机简谱生成器")
    parser.add_argument("--measures", type=int, default=0, help="小节数(默认随机12-24)")
    parser.add_argument("--beats", type=int, default=4, help="每小节拍数(默认4)")
    parser.add_argument("--key", type=str, default="", help="调号(默认随机)")
    parser.add_argument("--tempo", type=int, default=0, help="速度BPM(默认随机60-120)")
    parser.add_argument("--title", type=str, default="", help="标题(默认自动)")
    parser.add_argument("--intro", action="store_true", help="是否生成前奏括号")
    parser.add_argument("--output", type=str, default="", help="输出文件(默认stdout)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")

    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)

    score = generate(args)
    json_str = json.dumps(score, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"已生成到 {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
