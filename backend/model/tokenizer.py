"""
简谱 Token Vocabulary 定义
负责:
1. 定义所有可能的 token 及其字符串表示
2. 提供 token ↔ id 双向映射
3. 提供 token 类型分类（用于按类型解码）
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import json
import os

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<ROW>"]  # 5 个

# === 基础 token ===

# 音符数字 1~7（pitch）和 休止符 0
PITCH_TOKENS = [f"P{i}" for i in range(1, 8)]  # 7 个
REST_TOKEN = "R0"

# 八度偏移
OCTAVE_TOKENS = ["O+1", "O+2", "O-1", "O-2"]  # 4 个

# 升降号
ACCIDENTAL_TOKENS = ["#", "b", "n"]  # sharp flat natural

# 附点 + 八度点 + 时值延长
DOT_TOKEN = "."
UPPER_DOT_TOKEN = "^"  # 高音点
LOWER_DOT_TOKEN = "v"  # 低音点
DASH_TOKEN = "-"  # 增时线
UNDERLINE1_TOKEN = "_"  # 第一条减时线
UNDERLINE2_TOKEN = "="  # 第二条减时线

# 时值（duration）
DUR_TOKENS = ["D4", "D2", "D1", "D0.5", "D0.25", "D0.125", "D1.5", "D0.75"]  # 8 个

# 力度
DYN_TOKENS = ["pp", "p", "mp", "mf", "f", "ff"]  # 6 个
FORCE_ACCENT_TOKENS = ["sf", "sfp", "fp"]  # 3 个

# 技巧
TECH_TOKENS = [
    "T:bo",        # 波音
    "T:ch",        # 颤音
    "T:da",        # 打音
    "T:tu",        # 吐音
    "T:di",        # 叠音
    "T:li:up",     # 历音上
    "T:li:down",   # 历音下
    "T:hu:up",     # 滑音上
    "T:hu:down",   # 滑音下
    "T:yi:1", "T:yi:2", "T:yi:3", "T:yi:4", "T:yi:5", "T:yi:6", "T:yi:7",  # 倚音
    "T:du",        # 顿音
]  # 17 个

# 标记
TENUTO_TOKEN = "TEN"  # 保持音
FERMATA_TOKEN = "FER"  # 延长
ACCENT_TOKEN = "ACC"  # 重音 >

# 小节线
BAR_TOKENS = ["B|", "B||", "B|]", "B|:", "B:|"]  # 5 个

# 反复跳跃编号（1~3）
REPEAT_NUM_TOKENS = ["R1", "R2", "R3"]

# 渐强/渐弱
CRESC_TOKEN = "CRES"
DECRES_TOKEN = "DECRES"

# === 构建完整 vocab ===
ALL_TOKENS: List[str] = []
ALL_TOKENS.extend(SPECIAL_TOKENS)                              # 0~4
ALL_TOKENS.extend(PITCH_TOKENS)                                # 5~11
ALL_TOKENS.append(REST_TOKEN)                                  # 12
ALL_TOKENS.extend(OCTAVE_TOKENS)                               # 13~16
ALL_TOKENS.extend(ACCIDENTAL_TOKENS)                           # 17~19
ALL_TOKENS.append(DOT_TOKEN)                                   # 20
ALL_TOKENS.append(UPPER_DOT_TOKEN)                             # 21
ALL_TOKENS.append(LOWER_DOT_TOKEN)                             # 22
ALL_TOKENS.append(DASH_TOKEN)                                  # 23
ALL_TOKENS.append(UNDERLINE1_TOKEN)                            # 24
ALL_TOKENS.append(UNDERLINE2_TOKEN)                            # 25
ALL_TOKENS.extend(DUR_TOKENS)                                  # 26~33
ALL_TOKENS.extend(DYN_TOKENS)                                  # 34~39
ALL_TOKENS.extend(FORCE_ACCENT_TOKENS)                         # 40~42
ALL_TOKENS.extend(TECH_TOKENS)                                 # 43~59
ALL_TOKENS.append(TENUTO_TOKEN)                                # 60
ALL_TOKENS.append(FERMATA_TOKEN)                               # 61
ALL_TOKENS.append(ACCENT_TOKEN)                                # 62
ALL_TOKENS.extend(BAR_TOKENS)                                  # 63~67
ALL_TOKENS.extend(REPEAT_NUM_TOKENS)                           # 68~70
ALL_TOKENS.append(CRESC_TOKEN)                                 # 71
ALL_TOKENS.append(DECRES_TOKEN)                                # 72

# === 歌词 token (按字符) ===
# 歌词按字符拆开,加入 vocab 后生成
LYRIC_BASE = "LY:"
# 中文常用字 + ASCII
LYRIC_CHARS = (
    "春夏秋冬风花雪月山水云雨星日天地人梦心光影灯火歌行路远近高深清静明亮暖凉红绿黄白青蓝"
    "高低长短快慢强弱轻重缓急明暗冷热甜苦悲喜忧愁思念家乡故土母亲父亲兄弟姊妹朋友"
    "你我他她它们的是在了不和也有这那为以于上下前后里外"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "，。！？、；："
)
for ch in LYRIC_CHARS:
    ALL_TOKENS.append(LYRIC_BASE + ch)
# 歌词边界
LYRIC_START = "<LY>"
LYRIC_END = "</LY>"
ALL_TOKENS.append(LYRIC_START)
ALL_TOKENS.append(LYRIC_END)

# === 映射表 ===
TOKEN2ID: Dict[str, int] = {t: i for i, t in enumerate(ALL_TOKENS)}
ID2TOKEN: Dict[int, str] = {i: t for t, i in TOKEN2ID.items()}

VOCAB_SIZE = len(ALL_TOKENS)

# === 常量 id ===
PAD_ID = TOKEN2ID["<PAD>"]
BOS_ID = TOKEN2ID["<BOS>"]
EOS_ID = TOKEN2ID["<EOS>"]
UNK_ID = TOKEN2ID["<UNK>"]
ROW_ID = TOKEN2ID["<ROW>"]


def token_to_id(token: str) -> int:
    return TOKEN2ID.get(token, UNK_ID)


def id_to_token(tid: int) -> str:
    return ID2TOKEN.get(tid, "<UNK>")


def get_vocab() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    return ALL_TOKENS, TOKEN2ID, ID2TOKEN


# === 从 Score JSON 编码为 token 序列 ===
# 这是一个确定性算法: 不依赖 YOLO, 直接从 JSON 走规则路径
# 用于 prepare_pairs.py 构造 ground truth
DUR_TO_TOKEN = {
    4.0: "D4",
    2.0: "D2",
    1.0: "D1",
    0.5: "D0.5",
    0.25: "D0.25",
    0.125: "D0.125",
    1.5: "D1.5",
    0.75: "D0.75",
}


def encode_note(note: dict) -> List[str]:
    """编码单个 Note 为 token 列表"""
    tokens: List[str] = []

    # === 增时线 (dash) ===
    if note.get("type") == "dash":
        tokens.append(DASH_TOKEN)
        dur = note.get("duration", 1.0)
        if dur in DUR_TO_TOKEN:
            tokens.append(DUR_TO_TOKEN[dur])
        return tokens

    # === 休止符 ===
    if note.get("pitch") == 0:
        tokens.append(REST_TOKEN)
    else:
        # === 音高数字 ===
        pitch = note["pitch"]
        if pitch in (1, 2, 3, 4, 5, 6, 7):
            tokens.append(f"P{pitch}")
        else:
            tokens.append("<UNK>")

    # === 八度偏移 ===
    octave = note.get("octave", 0)
    if octave != 0:
        if octave == 1:
            tokens.append("O+1")
        elif octave == 2:
            tokens.append("O+1")
            tokens.append("O+1")
        elif octave == -1:
            tokens.append("O-1")
        elif octave == -2:
            tokens.append("O-1")
            tokens.append("O-1")

    # === 升降号 ===
    acc = note.get("accidental")
    if acc == "sharp":
        tokens.append("#")
    elif acc == "flat":
        tokens.append("b")
    elif acc == "natural":
        tokens.append("n")

    # === 附点 ===
    dot = note.get("dot", 0)
    for _ in range(dot):
        tokens.append(DOT_TOKEN)

    # === 减时线 (用 duration 推断) ===
    dur = note.get("duration", 1.0)
    if dur <= 0.25:
        tokens.append(UNDERLINE2_TOKEN)
    elif dur <= 0.5:
        tokens.append(UNDERLINE1_TOKEN)

    # === 时值 (作为辅助信息冗余) ===
    if dur in DUR_TO_TOKEN:
        tokens.append(DUR_TO_TOKEN[dur])

    # === 竹笛技巧 ===
    for tech in note.get("techniques", []) or []:
        t = tech.get("type")
        if t == "boyin":
            tokens.append("T:bo")
        elif t == "chanyin":
            tokens.append("T:ch")
        elif t == "dayin":
            tokens.append("T:da")
        elif t == "tuyin":
            tokens.append("T:tu")
        elif t == "dieyin":
            tokens.append("T:di")
        elif t == "liyin":
            direction = tech.get("liyinDirection", "up")
            tokens.append(f"T:li:{direction}")
        elif t == "huayin":
            direction = tech.get("slideDirection", "up")
            tokens.append(f"T:hu:{direction}")
        elif t == "yinyin":
            grace = tech.get("graceNotes", [3])
            pitch = grace[0] if grace else 3
            tokens.append(f"T:yi:{pitch}")
        elif t == "dunyin":
            tokens.append("T:du")

    # === 重音 / 保持音 / 延长 ===
    if note.get("accent"):
        tokens.append(ACCENT_TOKEN)
    if note.get("tenuto"):
        tokens.append(TENUTO_TOKEN)
    if note.get("fermata"):
        tokens.append(FERMATA_TOKEN)

    # === 力度突变 ===
    fa = note.get("forceAccent")
    if fa in ("sf", "sfp", "fp"):
        tokens.append(fa)

    # === 力度 ===
    dyn = note.get("dynamic")
    if dyn in ("pp", "p", "mp", "mf", "f", "ff"):
        tokens.append(dyn)

    # === 歌词 ===
    lyric = note.get("lyric")
    if lyric:
        tokens.append(LYRIC_START)
        for ch in lyric:
            tok = LYRIC_BASE + ch
            if tok in TOKEN2ID:
                tokens.append(tok)
            else:
                tokens.append(LYRIC_BASE + "?")
        tokens.append(LYRIC_END)

    return tokens


def encode_measure(measure: dict) -> List[str]:
    """编码单个小节为 token 列表 (含小节线、渐强/渐弱等)"""
    tokens: List[str] = []
    notes = measure.get("notes", []) or []

    for note in notes:
        tokens.extend(encode_note(note))

    # === 小节线 ===
    bl = measure.get("barline")
    if bl == "single":
        tokens.append("B|")
    elif bl == "double":
        tokens.append("B||")
    elif bl == "end":
        tokens.append("B|]")
    elif bl == "repeat-start":
        tokens.append("B|:")
    elif bl == "repeat-end":
        tokens.append("B:|")

    # === 反复跳跃 ===
    re = measure.get("repeatEnding")
    if re and re.get("numbers"):
        for n in re["numbers"]:
            if n in (1, 2, 3):
                tokens.append(f"R{n}")

    return tokens


def encode_score(score: dict) -> List[str]:
    """编码整个 Score 为 token 序列 (BOS + 每小节 + EOS)"""
    tokens: List[str] = ["<BOS>"]
    for m in score.get("measures", []) or []:
        tokens.extend(encode_measure(m))
    tokens.append("<EOS>")
    return tokens


def encode_score_with_dynamics(score: dict) -> List[str]:
    """完整编码: 在每小节开始处插入渐强/渐弱标记"""
    tokens: List[str] = ["<BOS>"]
    measures = score.get("measures", []) or []
    for i, m in enumerate(measures):
        dyn = m.get("dynamics")
        if dyn:
            if dyn.get("type") == "crescendo":
                tokens.append(CRESC_TOKEN)
            elif dyn.get("type") == "descrescendo":
                tokens.append(DECRES_TOKEN)
        tokens.extend(encode_measure(m))
    tokens.append("<EOS>")
    return tokens


def save_vocab(path: str):
    """保存 vocab 到 json"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "tokens": ALL_TOKENS,
            "vocab_size": VOCAB_SIZE,
            "token2id": TOKEN2ID,
            "id2token": {str(k): v for k, v in ID2TOKEN.items()},
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 测试
    print(f"VOCAB_SIZE = {VOCAB_SIZE}")
    print(f"PAD_ID = {PAD_ID}, BOS_ID = {BOS_ID}, EOS_ID = {EOS_ID}")
    sample_score = {
        "title": "测试",
        "key": "C",
        "timeSignature": {"numerator": 4, "denominator": 4},
        "measures": [
            {
                "notes": [
                    {"pitch": 5, "duration": 0.5, "lyric": "春", "techniques": [{"type": "boyin"}]},
                    {"pitch": 3, "duration": 0.5, "lyric": "天"},
                    {"pitch": 0, "duration": 0.5},
                ],
                "barline": "single",
            }
        ],
    }
    toks = encode_score(sample_score)
    print(f"Sample token count: {len(toks)}")
    print(f"Tokens: {toks[:20]}")
    print(f"IDs: {[token_to_id(t) for t in toks[:20]]}")
    # 保存 vocab
    here = os.path.dirname(os.path.abspath(__file__))
    save_vocab(os.path.join(here, "..", "weights", "vocab.json"))
    print("Vocab saved.")
