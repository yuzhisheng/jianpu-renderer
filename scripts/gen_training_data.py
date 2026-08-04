#!/usr/bin/env python3
"""生成 2000 个涵盖所有 42 类简谱符号的训练素材 - 确保每类 >= 50 样本"""
import json, os, random
random.seed(42)

BARLINE_WEIGHTS = {
    "single": 70,   # 普通 |
    "double": 10,   # ||
    "end": 10,      # |]
}
# 专门生成 repeat 符号的乐曲
REPEAT_BARLINES = ["repeat-start", "repeat-end"]

ALL_PITCHES = [1, 2, 3, 4, 5, 6, 7]
COMMON_PITCHES = [1, 2, 3, 4, 5, 6, 7]  # 所有音高都要出现
LYRICS_POOL = "春夏秋冬风花雪月山水云雨星日天地人梦心光影灯火歌行路远近高深清静明亮暖凉红绿黄白青蓝"
LYRICS = list(LYRICS_POOL)
DYN_TEXTS = ["pp", "p", "mp", "mf", "f", "ff"]
FORCE_ACCENTS = ["sf", "sfp", "fp"]
ALL_TECHS = ["boyin", "chanyin", "dayin", "tuyin", "dieyin", "huayin", "liyin", "yinyin", "dunyin"]


def N(pitch, dur, **kw):
    n = {"pitch": pitch, "duration": dur}
    n.update(kw)
    return n

def D(dur): return {"type": "dash", "duration": dur}
def R(dur): return {"pitch": 0, "duration": dur}

def pick_duration(remaining, choices):
    """Pick a duration that is visually supported and never invent .75 tails."""
    eligible = [d for d in choices if d <= remaining + 1e-6]
    return random.choice(eligible or [remaining])

def mk_score(title, key="C", ts_num=4, ts_den=4, measures=None):
    return {"title": title, "key": key, "timeSignature": {"numerator": ts_num, "denominator": ts_den}, "measures": measures or []}

def T_bo(): return [{"type": "boyin"}]
def T_ch(): return [{"type": "chanyin"}]
def T_da(): return [{"type": "dayin"}]
def T_tu(): return [{"type": "tuyin"}]
def T_di(): return [{"type": "dieyin"}]
def T_du(): return [{"type": "dunyin"}]
def T_hu(d): return [{"type": "huayin", "slideDirection": d}]
def T_li(d): return [{"type": "liyin", "liyinDirection": d}]
def T_yi(p): return [{"type": "yinyin", "graceNotes": [p], "graceOctave": 0}]

def random_pop(items, prob):
    """以 prob 概率随机从 items 取一项"""
    if random.random() < prob:
        return random.choice(items)
    return None

def make_note(beats_left, **kwargs):
    """构造一个 note, 根据剩余拍数选时值, 强制生成指定符号"""
    # 选时值
    choices = []
    if beats_left >= 1.0: choices.extend([1.0, 1.0])
    if beats_left >= 0.5: choices.extend([0.5, 0.5])
    if beats_left >= 0.25: choices.extend([0.25, 0.25])
    if not choices:
        dur = beats_left  # 用剩余拍数
    else:
        dur = random.choice(choices)

    # 选择音高
    pitches = kwargs.get("pitches", COMMON_PITCHES)
    pitch = random.choice(pitches)

    kw = {}

    # 歌词 (70% 概率)
    if random.random() < 0.7:
        kw["lyric"] = random.choice(LYRICS)

    # 强制生成
    if kwargs.get("force_octave"):
        kw["octave"] = kwargs["force_octave"]
    elif kwargs.get("allow_octave") and random.random() < 0.15:
        kw["octave"] = random.choice([-1, 1])

    if kwargs.get("force_accidental"):
        kw["accidental"] = kwargs["force_accidental"]
    elif kwargs.get("allow_accidental") and random.random() < 0.1:
        kw["accidental"] = random.choice(["sharp", "flat", "natural"])

    if kwargs.get("force_dot"):
        kw["dot"] = 1
    elif kwargs.get("allow_dot") and random.random() < 0.2:
        kw["dot"] = 1

    if kwargs.get("force_tech"):
        t = kwargs["force_tech"]
        kw["techniques"] = tech_to_list(t)
    elif kwargs.get("allow_tech") and random.random() < 0.15:
        t = random.choice(ALL_TECHS)
        kw["techniques"] = tech_to_list(t)

    if kwargs.get("force_fermata"):
        kw["fermata"] = True
    elif kwargs.get("allow_fermata") and random.random() < 0.05:
        kw["fermata"] = True

    if kwargs.get("force_tenuto"):
        kw["tenuto"] = True
    elif kwargs.get("allow_tenuto") and random.random() < 0.05:
        kw["tenuto"] = True

    if kwargs.get("force_dynamic"):
        kw["dynamic"] = kwargs["force_dynamic"]
    elif kwargs.get("allow_dynamic") and random.random() < 0.06:
        kw["dynamic"] = random.choice(DYN_TEXTS)

    if kwargs.get("force_force_accent"):
        kw["dynamic"] = kwargs["force_force_accent"]
        kw["forceAccent"] = kwargs["force_force_accent"]
    elif kwargs.get("allow_force_accent") and random.random() < 0.03:
        fa = random.choice(FORCE_ACCENTS)
        kw["dynamic"] = fa
        kw["forceAccent"] = fa

    if kwargs.get("force_accent"):
        kw["accent"] = True
    elif kwargs.get("allow_accent") and random.random() < 0.05:
        kw["accent"] = True

    if kwargs.get("tie_id"):
        kw["tieId"] = kwargs["tie_id"]
    elif kwargs.get("allow_tie") and random.random() < 0.12:
        kw["tieId"] = f"t{random.randint(100, 999)}"

    if kwargs.get("slur_id"):
        kw["slurId"] = kwargs["slur_id"]
    elif kwargs.get("allow_slur") and random.random() < 0.1:
        kw["slurId"] = f"s{random.randint(100, 999)}"

    return N(pitch, dur, **kw), dur


def mk_repeat_score(title):
    """生成一个带反复跳跃的乐谱"""
    m1_notes = [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(8)]
    m1 = {"notes": m1_notes}
    m2_notes = [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(8)]
    m2 = {"notes": m2_notes}
    m3 = {"notes": [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(8)], "barline": "repeat-start"}
    m4 = {"notes": [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(5)], "repeatEnding": {"numbers": [1]}, "barline": "double"}
    m5 = {"notes": [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(5)], "repeatEnding": {"numbers": [2]}, "barline": "end"}
    return mk_score(f"{title}", measures=[m1, m2, m3, m4, m5])


def mk_crescendo_score(title):
    """生成带渐强渐弱的乐谱"""
    m1 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)],
          "dynamics": {"type": "crescendo", "endMeasureIndex": 1}}
    m2 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)]}
    m3 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)],
          "dynamics": {"type": "descrescendo", "endMeasureIndex": 3}}
    m4 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)]}
    return mk_score(f"{title}", measures=[m1, m2, m3, m4])


def mk_special_score(title, symbol_type, count=4):
    """专门为一个特定符号生成多个音符的乐谱"""
    scores = []
    for i in range(count):
        measures = []
        for _ in range(random.randint(2, 4)):
            notes = []
            total = 0.0
            while total < 3.9:
                dur = pick_duration(4.0 - total, [0.5, 0.5, 1.0, 1.0, 0.25])
                total += dur
                pitch = random.choice(COMMON_PITCHES)
                kw = {"lyric": random.choice(LYRICS)}
                if symbol_type == "dot":
                    kw["dot"] = 1
                elif symbol_type == "accidental":
                    kw["accidental"] = random.choice(["sharp", "flat", "natural"])
                elif symbol_type == "fermata":
                    kw["fermata"] = True
                elif symbol_type == "tenuto":
                    kw["tenuto"] = True
                elif symbol_type == "accent":
                    kw["accent"] = True
                elif symbol_type == "dynamic":
                    kw["dynamic"] = random.choice(DYN_TEXTS)
                elif symbol_type in ALL_TECHS:
                    kw["techniques"] = tech_to_list(symbol_type)
                elif symbol_type == "tie":
                    tid = f"t{random.randint(1000, 9999)}"
                    kw["tieId"] = tid
                    notes.append(N(pitch, dur, **kw))
                    # 配对
                    dur2 = random.choice([0.5, 1.0])
                    kw2 = {"tieId": tid, "lyric": random.choice(LYRICS)}
                    notes.append(N(pitch, dur2, **kw2))
                    continue
                elif symbol_type == "slur":
                    sid = f"s{random.randint(1000, 9999)}"
                    kw["slurId"] = sid
                    notes.append(N(pitch, dur, **kw))
                    dur2 = random.choice([0.5, 1.0])
                    kw2 = {"slurId": sid, "lyric": random.choice(LYRICS)}
                    notes.append(N(random.choice(COMMON_PITCHES), dur2, **kw2))
                    continue
                elif symbol_type == "force_accent":
                    fa = random.choice(FORCE_ACCENTS)
                    kw["dynamic"] = fa
                    kw["forceAccent"] = fa
                notes.append(N(pitch, dur, **kw))
            bar = random.choice(["single", "single", "end", "double"])
            measures.append({"notes": notes, "barline": bar})
        scores.append(mk_score(f"{title}_{i+1}", measures=measures))
    return scores


def mk_regular_score(title, measures_count=4):
    """常规多样化乐谱"""
    measures = []
    for mi in range(measures_count):
        notes = []
        total = 0.0
        tie_ids = {}
        slur_ids = {}
        while total < 3.9:
            # Only durations with an explicit visual representation. Long
            # values must be represented by Dash items, not hidden in spacing.
            dur = pick_duration(4.0 - total, [1.0, 0.5, 0.5, 0.25, 0.25, 1.0, 0.5])
            total += dur

            kw = {}
            if random.random() < 0.65:
                kw["lyric"] = random.choice(LYRICS)
            if random.random() < 0.1:
                kw["octave"] = random.choice([-1, 1])
            if random.random() < 0.06:
                kw["accidental"] = random.choice(["sharp", "flat", "natural"])
            if random.random() < 0.12:
                kw["dot"] = 1
            if random.random() < 0.1:
                t = random.choice(ALL_TECHS)
                kw["techniques"] = tech_to_list(t)
            if random.random() < 0.03:
                kw["fermata"] = True
            if random.random() < 0.04:
                kw["tenuto"] = True
            if random.random() < 0.03:
                kw["accent"] = True
            if random.random() < 0.04:
                kw["dynamic"] = random.choice(DYN_TEXTS)
            if random.random() < 0.02:
                fa = random.choice(FORCE_ACCENTS)
                kw["dynamic"] = fa
                kw["forceAccent"] = fa
            if random.random() < 0.06:
                kw["tieId"] = f"t{random.randint(100, 999)}"
            if random.random() < 0.05:
                kw["slurId"] = f"s{random.randint(100, 999)}"

            pitch = random.choice(COMMON_PITCHES)
            notes.append(N(pitch, dur, **kw))

        # 小节线
        bar = "single"
        if mi == measures_count - 1:
            bar = random.choice(["end", "double", "single", "single"])
        elif random.random() < 0.05:
            bar = "double"

        measures.append({"notes": notes, "barline": bar})
    return mk_score(f"{title}", measures=measures)


def tech_to_list(t):
    if t == "boyin": return T_bo()
    if t == "chanyin": return T_ch()
    if t == "dayin": return T_da()
    if t == "tuyin": return T_tu()
    if t == "dieyin": return T_di()
    if t == "dunyin": return T_du()
    if t == "huayin": return T_hu(random.choice(["up", "down"]))
    if t == "liyin": return T_li(random.choice(["up", "down"]))
    if t == "yinyin": return T_yi(random.choice(COMMON_PITCHES))
    return T_bo()


# ===== 主生成流程 =====
all_scores = []

# 1. 保留 9 个手工素材 (score_001~009)
all_scores.append(mk_score("节奏型示例", measures=[
    {"notes":[N(5,0.5,lyric="四"),N(3,0.5,lyric="分"),N(1,0.25,lyric="八"),N(2,0.25,lyric="八"),N(3,0.25,lyric="八"),N(5,0.25,lyric="八"),N(6,0.5,lyric="四"),N(5,0.5,lyric="四")]},
    {"notes":[N(1,1,dot=1,lyric="附"),N(1,0.5,lyric="点"),N(2,1,lyric="二"),N(3,0.5,lyric="拍"),D(0.5)]},
    {"notes":[R(0.5),N(5,0.5,lyric="休"),N(3,0.5,lyric="止"),N(2,1,lyric="长"),D(1)]},
    {"notes":[N(1,0.25,lyric="十"),N(2,0.25,lyric="六"),N(3,0.25,lyric="十"),N(5,0.25,lyric="六"),N(6,0.5,lyric="八"),N(5,0.5,lyric="八"),N(3,1,lyric="四"),D(0.5)]},
]))
all_scores.append(mk_score("八度与变音", measures=[
    {"notes":[N(5,0.5,octave=1,lyric="高"),N(6,0.5,octave=1,lyric="八"),N(1,0.5,octave=2,lyric="度"),N(5,0.5,lyric="中"),N(3,0.5,lyric="音"),N(1,0.5,lyric="区"),N(6,0.5),N(5,0.5)]},
    {"notes":[N(5,0.5,octave=-1,lyric="低"),N(6,0.5,octave=-1,lyric="八"),N(1,0.5,octave=-1,lyric="度"),N(2,0.5,lyric="低"),N(3,0.5,lyric="音"),N(2,0.5,lyric="区"),N(1,0.5),N(6,0.25,octave=-1),N(5,0.25,octave=-1)]},
    {"notes":[N(5,0.5,accidental="sharp",lyric="升"),N(6,0.5,accidental="sharp",lyric="号"),N(3,0.5,accidental="flat",lyric="降"),N(2,0.5,accidental="flat",lyric="号"),N(1,0.5,accidental="natural",lyric="还"),N(2,0.5,accidental="natural",lyric="原"),N(3,0.5),N(5,0.5)]},
    {"notes":[N(1,1,lyric="低"),N(5,1,octave=-1),N(6,0.5,octave=-1,lyric="高"),N(1,0.5,octave=1),N(2,0.5,octave=1),N(3,0.5,octave=1)]},
]))
all_scores.append(mk_score("波音颤音倚音", measures=[
    {"notes":[N(5,0.5,lyric="波",techniques=T_bo()),N(3,0.5,lyric="音"),N(2,0.5),N(1,0.5,lyric="波",techniques=T_bo()),N(6,0.5,octave=-1,lyric="音"),N(1,0.5,techniques=T_bo()),N(2,0.5,lyric="波"),N(3,0.25,lyric="音"),N(5,0.25,techniques=T_bo())]},
    {"notes":[N(6,0.5,lyric="颤",techniques=T_ch()),N(5,0.5,lyric="音"),N(3,0.5),N(2,0.5,lyric="颤",techniques=T_ch()),N(1,1,techniques=T_ch()),N(5,0.5,lyric="颤"),N(6,0.5,techniques=T_ch())]},
    {"notes":[N(5,0.5,lyric="倚",techniques=T_yi(3)),N(6,0.5,lyric="音"),N(5,0.5),N(3,0.5,lyric="倚",techniques=T_yi(6)),N(2,0.5,lyric="音"),N(1,0.5),N(5,0.25,techniques=T_yi(3)),N(3,0.25),N(2,0.5)]},
    {"notes":[N(1,1,lyric="波",techniques=T_bo()),N(5,0.5,lyric="颤",techniques=T_ch()),N(6,0.5,lyric="倚",techniques=T_yi(3)),N(5,0.5,lyric="波",techniques=T_bo()),N(3,0.5),N(2,0.5),N(1,0.5)]},
]))
all_scores.append(mk_score("滑音历音叠音", measures=[
    {"notes":[N(5,0.5,lyric="上",techniques=T_hu("up")),N(6,0.5,lyric="滑"),N(5,0.5),N(3,0.5,lyric="下",techniques=T_hu("down")),N(2,0.5,lyric="滑"),N(1,0.5),N(6,0.25,octave=-1,techniques=T_hu("up")),N(1,0.25),N(2,0.5)]},
    {"notes":[N(1,0.5,lyric="上",techniques=T_li("up")),N(5,0.5,lyric="历"),N(3,0.5,lyric="音"),N(2,0.5,lyric="下",techniques=T_li("down")),N(1,0.5,lyric="历"),N(6,0.5,octave=-1,lyric="音"),N(1,0.5),D(0.5)]},
    {"notes":[N(5,0.5,lyric="叠",techniques=T_di()),N(3,0.5,lyric="音"),N(2,0.5),N(1,0.5,lyric="叠",techniques=T_di()),N(6,0.5,octave=-1,lyric="音"),N(5,0.5),N(6,0.5,techniques=T_di()),N(1,0.5,octave=1)]},
    {"notes":[N(3,0.5,lyric="上",techniques=T_hu("up")),N(5,0.5,lyric="叠",techniques=T_di()),N(6,0.5,lyric="下",techniques=T_hu("down")),N(3,0.5,lyric="滑"),N(2,0.5),N(1,1,lyric="止"),D(0.5)]},
]))
all_scores.append(mk_score("打音与吐音", measures=[
    {"notes":[N(5,0.5,lyric="打",techniques=T_da()),N(3,0.5,lyric="音"),N(2,0.5),N(1,0.5,lyric="打",techniques=T_da()),N(6,0.5,octave=-1,lyric="音"),N(5,0.5),N(3,0.5,techniques=T_da()),N(2,0.5,lyric="打")]},
    {"notes":[N(1,0.5,lyric="吐",techniques=T_tu()),N(2,0.5,lyric="音"),N(3,0.5),N(5,0.5,lyric="吐",techniques=T_tu()),N(6,0.5,lyric="音"),N(5,0.5),N(6,0.5,techniques=T_tu()),N(1,0.5,octave=1)]},
    {"notes":[N(3,0.5,lyric="打",techniques=T_da()),N(5,0.5,lyric="吐",techniques=T_tu()),N(6,0.5,lyric="打",techniques=T_da()),N(3,0.5,lyric="吐",techniques=T_tu()),N(5,0.5),N(6,0.5),N(3,0.5),N(2,0.5)]},
    {"notes":[N(1,1,lyric="打",techniques=T_da()),N(5,0.5,techniques=T_tu()),N(3,0.5,lyric="吐"),N(2,1,lyric="打",techniques=T_da()),N(1,1)]},
]))
all_scores.append(mk_score("连音与圆滑线", measures=[
    {"notes":[N(5,0.5,lyric="圆",slurId="a1"),N(3,0.5,lyric="滑",slurId="a1"),N(2,0.5,lyric="圆",slurId="a2"),N(1,0.5,lyric="滑",slurId="a2"),N(6,0.5,octave=-1,lyric="圆",slurId="a3"),N(1,0.5,lyric="滑",slurId="a3"),N(2,0.5),N(3,0.5)]},
    {"notes":[N(5,1,lyric="连",tieId="t1"),N(5,0.5,tieId="t1"),N(5,0.5,lyric="音"),N(3,1,lyric="连",tieId="t2"),N(3,0.5,tieId="t2"),N(3,0.5,lyric="音")]},
    {"notes":[N(1,0.5,octave=1,lyric="月",slurId="a4"),N(6,0.5,lyric="亮",slurId="a4"),N(5,0.5,lyric="代",slurId="a5"),N(3,0.5,lyric="表",slurId="a5"),N(5,0.5,lyric="我",slurId="a6"),N(3,0.5,lyric="的",slurId="a6"),N(2,0.5,lyric="心"),N(1,0.5)]},
    {"notes":[N(5,1,lyric="跨",tieId="t3"),N(5,0.5,tieId="t3"),N(3,0.5,lyric="节"),N(2,1,lyric="大",slurId="a7"),N(1,1,lyric="圆",slurId="a7")]},
]))
all_scores.append(mk_score("反复跳跃", measures=[
    {"notes":[N(5,0.5,lyric="前"),N(3,0.5,lyric="奏"),N(2,0.5,lyric="段"),N(1,0.5,lyric="落"),N(5,0.5,lyric="~"),N(3,0.5,lyric="~"),N(2,0.5,lyric="~"),N(1,0.5,lyric="~")]},
    {"notes":[N(6,0.5,octave=-1,lyric="主"),N(1,0.5,lyric="题"),N(2,0.5,lyric="乐"),N(3,0.5,lyric="段"),N(5,0.5,lyric="~"),N(6,0.5,lyric="~"),N(5,0.5,lyric="~"),N(3,0.5,lyric="~")]},
    {"notes":[N(1,1,lyric="反"),N(2,0.5,lyric="复"),N(3,0.5,lyric="开"),N(5,0.5,lyric="始"),N(6,0.5,lyric="~"),N(5,0.5,lyric="~"),N(3,0.5,lyric="~"),N(2,0.5,lyric="~")],"barline":"repeat-start"},
    {"notes":[N(1,1,lyric="一"),N(5,0.5,lyric="房"),N(3,0.5,lyric="结"),N(2,1,lyric="尾"),N(1,0.5,lyric="~")],"repeatEnding":{"numbers":[1]},"barline":"double"},
    {"notes":[N(5,0.5,lyric="间"),N(3,0.5,lyric="奏"),N(2,0.5,lyric="过"),N(1,0.5,lyric="渡"),N(6,0.5,octave=-1,lyric="~"),N(5,0.5,lyric="~"),N(6,0.25,octave=-1),N(1,0.25,lyric="~"),N(2,0.5,lyric="~")]},
    {"notes":[N(1,1,lyric="二"),N(5,0.5,lyric="房"),N(3,0.5,lyric="终"),N(2,1,lyric="止"),N(1,1)],"repeatEnding":{"numbers":[2]},"barline":"end"},
]))
all_scores.append(mk_score("力度与变化", measures=[
    {"notes":[N(1,0.5,lyric="渐",dynamic="pp"),N(2,0.5,lyric="强"),N(3,0.5,lyric="自",dynamic="mp"),N(5,0.5,lyric="弱"),N(6,0.5,lyric="至"),N(1,0.5,octave=1,lyric="强",dynamic="f"),N(6,0.5),N(5,0.5)],"dynamics":{"type":"crescendo","endMeasureIndex":1}},
    {"notes":[N(1,0.25,octave=1,lyric="渐"),N(6,0.25,lyric="强"),N(5,0.25,lyric="至"),N(3,0.25,lyric="顶"),N(2,0.5,lyric="峰"),N(3,0.5),N(5,0.5),N(6,0.25),N(5,0.25)]},
    {"notes":[N(6,0.5,lyric="渐"),N(5,0.5,lyric="弱"),N(3,0.5,lyric="自"),N(2,0.5,lyric="强"),N(1,0.5,lyric="至"),N(6,0.5,octave=-1,lyric="弱"),N(5,0.5),N(3,0.5)],"dynamics":{"type":"descrescendo","endMeasureIndex":3}},
    {"notes":[N(2,0.5,lyric="渐"),N(1,0.5,lyric="弱"),N(6,0.5,octave=-1,lyric="至"),N(5,0.5,octave=-1,lyric="底"),N(6,0.25,octave=-1),N(1,0.25),N(2,0.5),N(3,0.5),N(1,0.5)]},
]))
all_scores.append(mk_score("综合符号示例", measures=[
    {"notes":[N(5,0.5,lyric="综"),N(3,0.25,lyric="合"),N(2,0.25),N(1,0.5,lyric="符",techniques=T_bo()),N(6,0.5,octave=-1,lyric="号"),N(1,0.5,lyric="大"),N(2,0.5,lyric="全")]},
    {"notes":[N(3,0.5,lyric="顿",techniques=T_du()),N(5,0.25,lyric="滑",techniques=T_hu("up")),N(6,0.25,lyric="音"),N(3,0.5,lyric="波",techniques=T_bo()),N(2,0.5,lyric="保",tenuto=True),N(1,0.5,lyric="延",fermata=True),N(5,0.25,accidental="sharp",techniques=T_da()),N(3,0.25),N(2,0.5)]},
    {"notes":[N(6,0.5,lyric="打",techniques=T_da()),N(1,0.5,octave=1,lyric="叠",techniques=T_di()),N(5,0.5,lyric="吐",techniques=T_tu()),N(3,0.5,lyric="倚",techniques=T_yi(2)),N(5,0.5,lyric="颤",techniques=T_ch()),N(6,0.5,lyric="顿",techniques=T_du()),N(5,0.5)]},
    {"notes":[N(1,1,octave=1,lyric="低",slurId="z1"),N(6,0.5,lyric="音",slurId="z1"),N(5,1,lyric="连",tieId="zt1"),N(5,0.5,tieId="zt1"),N(3,0.5),N(2,0.25),N(1,0.25)]},
    {"notes":[N(2,0.5,lyric="跨",slurId="z2"),N(3,0.5,lyric="节",slurId="z2"),N(5,1,lyric="长",fermata=True),N(6,0.5,octave=-1,techniques=T_du()),N(1,0.5,lyric="止"),N(2,1)],"barline":"end"},
]))

# 2. 专门生成低样本类别的乐谱 (每类 20 页)
SPECIAL_SYMBOLS = [
    ("dot", "附点"),
    ("fermata", "延长"),
    ("tenuto", "保持"),
    ("accent", "重音"),
]
for sym, cn_name in SPECIAL_SYMBOLS:
    all_scores.extend(mk_special_score(f"{cn_name}专项", sym, count=20))

# 力度专项
for _ in range(120):
    all_scores.append(mk_crescendo_score(f"渐强渐弱{len(all_scores)-200}"))

# 反复跳跃专项（提高反复线/小房子长尾类别覆盖）
for _ in range(150):
    all_scores.append(mk_repeat_score(f"反复跳跃{len(all_scores)-240}"))

# 每个技巧专项 (各 15 页, 确保均匀覆盖)
for tech in ALL_TECHS:
    all_scores.extend(mk_special_score(f"{tech}专项", tech, count=15))

# 连音线专项 (20 页)
all_scores.extend(mk_special_score("连音线", "tie", count=20))
# 圆滑线专项 (20 页)
all_scores.extend(mk_special_score("圆滑线", "slur", count=20))
# 力度突变专项 (15 页)
all_scores.extend(mk_special_score("力度突变", "force_accent", count=15))

# 休止符专项 (确保有 rest 符号)
def mk_rest_score(title, count=4):
    scores = []
    for i in range(count):
        measures = []
        for _ in range(random.randint(2, 4)):
            notes = []
            total = 0.0
            while total < 3.9:
                is_rest = random.random() < 0.25 and total + 0.5 <= 4.05
                if is_rest:
                    dur = pick_duration(4.0 - total, [0.5, 1.0])
                    total += dur
                    notes.append(R(dur))
                else:
                    dur = pick_duration(4.0 - total, [0.5, 1.0])
                    total += dur
                    notes.append(N(random.choice(COMMON_PITCHES), dur, lyric=random.choice(LYRICS)))
            bar = random.choice(["single", "single", "end", "double"])
            measures.append({"notes": notes, "barline": bar})
        scores.append(mk_score(f"{title}_{i+1}", measures=measures))
    return scores
all_scores.extend(mk_rest_score("休止符", count=100))

# 增时线专项 (确保有 dash 符号)
def mk_dash_score(title, count=4):
    scores = []
    for i in range(count):
        measures = []
        for _ in range(random.randint(2, 4)):
            notes = [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(6)]
            notes.append(D(0.5))  # 增时线
            bar = random.choice(["single", "single", "end", "double"])
            measures.append({"notes": notes, "barline": bar})
        scores.append(mk_score(f"{title}_{i+1}", measures=measures))
    return scores
all_scores.extend(mk_dash_score("增时线", count=100))

# 反复结束专项 (确保有 bar_repeat_end)
def mk_repeat_end_score(title):
    measures = []
    for _ in range(3):
        notes = [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(8)]
        measures.append({"notes": notes, "barline": "single"})
    measures.append({"notes": [N(random.choice(COMMON_PITCHES), 0.5, lyric=random.choice(LYRICS)) for _ in range(5)],
                     "barline": "repeat-end"})
    return mk_score(f"{title}", measures=measures)
for _ in range(150):
    all_scores.append(mk_repeat_end_score(f"反复结束{_}"))

# 渐弱专项 (确保有 descrescendo)
def mk_descrescendo_score(title):
    m1 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)]}
    m2 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)],
          "dynamics": {"type": "descrescendo", "endMeasureIndex": 3}}
    m3 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)]}
    m4 = {"notes": [N(p, 0.5, lyric=random.choice(LYRICS)) for p in random.choices(COMMON_PITCHES, k=8)]}
    return mk_score(f"{title}", measures=[m1, m2, m3, m4])
for _ in range(120):
    all_scores.append(mk_descrescendo_score(f"渐弱{_}"))

# 3. 常规多样化乐谱填充到 2000
while len(all_scores) < 2000:
    mc = random.randint(2, 6)
    title = f"素材{len(all_scores)+1:04d}"
    all_scores.append(mk_regular_score(title, mc))

# 截断到 2000
all_scores = all_scores[:2000]

# ===== 输出 =====
os.makedirs("public/training", exist_ok=True)
for i, s in enumerate(all_scores):
    fname = f"public/training/score_{i+1:04d}.json"
    with open(fname, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    if (i + 1) % 500 == 0:
        print(f"  ✓ {i+1} scores...")

scores_list = [{"file": f"score_{i+1:04d}.json", "title": s["title"], "measures": len(s["measures"])} for i, s in enumerate(all_scores)]
with open("public/training/index.json", "w") as f:
    json.dump(scores_list, f, ensure_ascii=False, indent=2)
print(f"\nDone! {len(all_scores)} scores → public/training/")
