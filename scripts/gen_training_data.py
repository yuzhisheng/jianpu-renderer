#!/usr/bin/env python3
"""生成 10 个涵盖所有简谱符号的训练素材"""
import json, os

def make_note(pitch, dur, **kw):
    n = {"pitch": pitch, "duration": dur}
    n.update(kw)
    return n

def make_dash(dur):
    return {"type": "dash", "duration": dur}

def make_rest(dur):
    return {"pitch": 0, "duration": dur}

all_scores = []

# ========== 1. 节奏型全覆盖 ==========
s1 = {
    "title": "节奏型示例",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="四"),make_note(3,0.5,lyric="分"),
            make_note(1,0.25,lyric="八"),make_note(2,0.25,lyric="八"),
            make_note(3,0.25,lyric="八"),make_note(5,0.25,lyric="八"),
            make_note(6,0.5,lyric="四"),make_note(5,0.5,lyric="四"),
        ]},
        {"notes": [
            make_note(1,1,dot=1,lyric="附点"),make_note(1,0.5,lyric="后"),
            make_note(2,1,lyric="二"),make_note(3,0.5,lyric="拍"),
            make_dash(0.5),
        ]},
        {"notes": [
            make_rest(0.5),
            make_note(5,0.5,lyric="休"),make_note(3,0.5,lyric="止"),
            make_note(2,2,lyric="长"),make_dash(0.5),
        ]},
        {"notes": [
            make_note(1,0.25,lyric="十"),make_note(2,0.25,lyric="六"),
            make_note(3,0.25,lyric="十"),make_note(5,0.25,lyric="六"),
            make_note(6,0.5,lyric="八"),make_note(5,0.5,lyric="八"),
            make_note(3,1,lyric="四"),make_dash(0.5),
        ]},
    ]
}
all_scores.append(s1)

# ========== 2. 高低八度 + 变音记号 ==========
s2 = {
    "title": "八度与变音",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,octave=1,lyric="高"),make_note(6,0.5,octave=1,lyric="八"),
            make_note(1,0.5,octave=2,lyric="度"),make_note(5,0.5,lyric="中"),
            make_note(3,0.5,lyric="音"),make_note(1,0.5,lyric="区"),
            make_note(6,0.5,lyric="~"),make_note(5,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(5,0.5,octave=-1,lyric="低"),make_note(6,0.5,octave=-1,lyric="八"),
            make_note(1,0.5,octave=-1,lyric="度"),make_note(2,0.5,lyric="低"),
            make_note(3,0.5,lyric="音"),make_note(2,0.5,lyric="区"),
            make_note(1,0.5,lyric="~"),make_note(6,0.25,octave=-1,lyric="~"),make_note(5,0.25,octave=-1),
        ]},
        {"notes": [
            make_note(5,0.5,accidental="sharp",lyric="升"),make_note(6,0.5,accidental="sharp",lyric="号"),
            make_note(3,0.5,accidental="flat",lyric="降"),make_note(2,0.5,accidental="flat",lyric="号"),
            make_note(1,0.5,accidental="natural",lyric="还"),make_note(2,0.5,accidental="natural",lyric="原"),
            make_note(3,0.5,lyric="~"),make_note(5,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(1,1,lyric="低"),make_note(5,1,octave=-1),
            make_note(6,0.5,octave=-1,lyric="高"),make_note(1,0.5,octave=1),
            make_note(2,0.5,octave=1,lyric="~"),make_note(3,0.5,octave=1),
        ]},
    ]
}
all_scores.append(s2)

# ========== 3. 波音 + 颤音 + 倚音 ==========
s3 = {
    "title": "波音颤音倚音",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="波",techniques=[{"type":"boyin"}]),
            make_note(3,0.5,lyric="音"),make_note(2,0.5,lyric="~"),
            make_note(1,0.5,lyric="波",techniques=[{"type":"boyin"}]),
            make_note(6,0.5,octave=-1,lyric="音"),
            make_note(1,0.5,techniques=[{"type":"boyin"}]),
            make_note(2,0.5,lyric="波"),
            make_note(3,0.25,lyric="音"),make_note(5,0.25,techniques=[{"type":"boyin"}]),
        ]},
        {"notes": [
            make_note(6,0.5,lyric="颤",techniques=[{"type":"chanyin"}]),
            make_note(5,0.5,lyric="音"),make_note(3,0.5,lyric="~"),
            make_note(2,0.5,lyric="颤",techniques=[{"type":"chanyin"}]),
            make_note(1,1,techniques=[{"type":"chanyin"}]),
            make_note(5,0.5,lyric="颤"),make_note(6,0.5,techniques=[{"type":"chanyin"}]),
        ]},
        {"notes": [
            make_note(5,0.5,lyric="倚",techniques=[{"type":"yinyin","graceNotes":[3],"graceOctave":0}]),
            make_note(6,0.5,lyric="音"),make_note(5,0.5,lyric="~"),
            make_note(3,0.5,lyric="倚",techniques=[{"type":"yinyin","graceNotes":[6],"graceOctave":0}]),
            make_note(2,0.5,lyric="音"),make_note(1,0.5,lyric="~"),
            make_note(5,0.25,techniques=[{"type":"yinyin","graceNotes":[3],"graceOctave":0}]),make_note(3,0.25),
            make_note(2,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(1,1,lyric="波",techniques=[{"type":"boyin"}]),
            make_note(5,0.5,lyric="颤",techniques=[{"type":"chanyin"}]),
            make_note(6,0.5,lyric="倚",techniques=[{"type":"yinyin","graceNotes":[3],"graceOctave":0}]),
            make_note(5,0.5,lyric="波",techniques=[{"type":"boyin"}]),
            make_note(3,0.5,lyric="~"),make_note(2,0.5,lyric="~"),
            make_note(1,0.5,lyric="~"),
        ]},
    ]
}
all_scores.append(s3)

# ========== 4. 滑音 + 历音 + 叠音 ==========
s4 = {
    "title": "滑音历音叠音",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="上",techniques=[{"type":"huayin","slideDirection":"up"}]),
            make_note(6,0.5,lyric="滑"),make_note(5,0.5,lyric="~"),
            make_note(3,0.5,lyric="下",techniques=[{"type":"huayin","slideDirection":"down"}]),
            make_note(2,0.5,lyric="滑"),make_note(1,0.5,lyric="~"),
            make_note(6,0.25,octave=-1,techniques=[{"type":"huayin","slideDirection":"up"}]),make_note(1,0.25),
            make_note(2,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(1,0.5,lyric="上",techniques=[{"type":"liyin","liyinDirection":"up"}]),
            make_note(5,0.5,lyric="历"),make_note(3,0.5,lyric="音"),
            make_note(2,0.5,lyric="下",techniques=[{"type":"liyin","liyinDirection":"down"}]),
            make_note(1,0.5,lyric="历"),make_note(6,0.5,octave=-1,lyric="音"),
            make_note(1,0.5,lyric="~"),make_dash(0.5),
        ]},
        {"notes": [
            make_note(5,0.5,lyric="叠",techniques=[{"type":"dieyin"}]),
            make_note(3,0.5,lyric="音"),make_note(2,0.5,lyric="~"),
            make_note(1,0.5,lyric="叠",techniques=[{"type":"dieyin"}]),
            make_note(6,0.5,octave=-1,lyric="音"),make_note(5,0.5,lyric="~"),
            make_note(6,0.5,techniques=[{"type":"dieyin"}]),make_note(1,0.5,octave=1),
        ]},
        {"notes": [
            make_note(3,0.5,lyric="上",techniques=[{"type":"huayin","slideDirection":"up"}]),
            make_note(5,0.5,lyric="叠",techniques=[{"type":"dieyin"}]),
            make_note(6,0.5,lyric="下",techniques=[{"type":"huayin","slideDirection":"down"}]),
            make_note(3,0.5,lyric="滑"),make_note(2,0.5,lyric="~"),
            make_note(1,1,lyric="止"),make_dash(0.5),
        ]},
    ]
}
all_scores.append(s4)

# ========== 5. 打音 + 吐音 ==========
s5 = {
    "title": "打音与吐音",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="打",techniques=[{"type":"dayin"}]),
            make_note(3,0.5,lyric="音"),make_note(2,0.5,lyric="~"),
            make_note(1,0.5,lyric="打",techniques=[{"type":"dayin"}]),
            make_note(6,0.5,octave=-1,lyric="音"),make_note(5,0.5,lyric="~"),
            make_note(3,0.5,techniques=[{"type":"dayin"}]),make_note(2,0.5,lyric="打"),
        ]},
        {"notes": [
            make_note(1,0.5,lyric="吐",techniques=[{"type":"tuyin"}]),
            make_note(2,0.5,lyric="音"),make_note(3,0.5,lyric="~"),
            make_note(5,0.5,lyric="吐",techniques=[{"type":"tuyin"}]),
            make_note(6,0.5,lyric="音"),make_note(5,0.5,lyric="~"),
            make_note(6,0.5,techniques=[{"type":"tuyin"}]),
            make_note(1,0.5,octave=1,lyric="~"),
        ]},
        {"notes": [
            make_note(3,0.5,lyric="吐",techniques=[{"type":"tuyin"}]),
            make_note(5,0.5,lyric="音"),make_note(6,0.5,lyric="~"),
            make_note(1,0.5,octave=1,techniques=[{"type":"tuyin"}]),
            make_note(6,0.5,lyric="音"),make_note(5,0.5,lyric="~"),
            make_note(3,0.5,techniques=[{"type":"tuyin"}]),make_note(2,0.5,lyric="~"),
            make_note(1,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(5,0.5,lyric="打",techniques=[{"type":"dayin"}]),
            make_note(6,0.5,lyric="吐",techniques=[{"type":"tuyin"}]),
            make_note(3,0.5,lyric="打",techniques=[{"type":"dayin"}]),
            make_note(5,0.5,lyric="吐",techniques=[{"type":"tuyin"}]),
            make_note(6,0.5,lyric="~"),make_note(5,0.5,lyric="~"),
            make_note(2,0.5,lyric="~"),make_note(1,0.5,lyric="~"),
        ]},
    ]
}
all_scores.append(s5)

# ========== 6. 连音线 + 圆滑线 ==========
s6 = {
    "title": "连音与圆滑线",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="圆",slurId="a1"),make_note(3,0.5,lyric="滑",slurId="a1"),
            make_note(2,0.5,lyric="圆",slurId="a2"),make_note(1,0.5,lyric="滑",slurId="a2"),
            make_note(6,0.5,octave=-1,lyric="圆",slurId="a3"),
            make_note(1,0.5,lyric="滑",slurId="a3"),
            make_note(2,0.5,lyric="~"),make_note(3,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(5,1,lyric="连",tieId="t1"),
            make_note(5,0.5,tieId="t1"),make_note(5,0.5,lyric="音"),
            make_note(3,1,lyric="连",tieId="t2"),
            make_note(3,0.5,tieId="t2"),make_note(3,0.5,lyric="音"),
        ]},
        {"notes": [
            make_note(1,0.5,octave=1,lyric="月",slurId="a4"),
            make_note(6,0.5,lyric="亮",slurId="a4"),
            make_note(5,0.5,lyric="代",slurId="a5"),
            make_note(3,0.5,lyric="表",slurId="a5"),
            make_note(5,0.5,lyric="我",slurId="a6"),
            make_note(3,0.5,lyric="的",slurId="a6"),
            make_note(2,0.5,lyric="心"),make_note(1,0.5),
        ]},
        {"notes": [
            make_note(5,1,lyric="跨",tieId="long_tie"),
            make_note(5,0.5,tieId="long_tie"),
            make_note(3,0.5,lyric="节"),
            make_note(2,1,lyric="大",slurId="a7"),
            make_note(1,1,lyric="圆",slurId="a7"),
        ]},
    ]
}
all_scores.append(s6)

# ========== 7. 小房子（反复跳跃） ==========
s8 = {
    "title": "反复跳跃",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="前"),make_note(3,0.5,lyric="奏"),
            make_note(2,0.5,lyric="段"),make_note(1,0.5,lyric="落"),
            make_note(5,0.5,lyric="~~"),make_note(3,0.5,lyric="~~"),
            make_note(2,0.5,lyric="~~"),make_note(1,0.5,lyric="~~"),
        ]},
        {"notes": [
            make_note(6,0.5,octave=-1,lyric="主"),make_note(1,0.5,lyric="题"),
            make_note(2,0.5,lyric="乐"),make_note(3,0.5,lyric="段"),
            make_note(5,0.5,lyric="~~"),make_note(6,0.5,lyric="~~"),
            make_note(5,0.5,lyric="~~"),make_note(3,0.5,lyric="~~"),
        ]},
        {"notes": [
            make_note(1,1,lyric="反"),make_note(2,0.5,lyric="复"),
            make_note(3,0.5,lyric="开"),make_note(5,0.5,lyric="始"),
            make_note(6,0.5,lyric="~~"),make_note(5,0.5,lyric="~~"),
            make_note(3,0.5,lyric="~~"),make_note(2,0.5,lyric="~~"),
        ], "barline": "repeat-start"},
        {"notes": [
            make_note(1,1,lyric="一"),make_note(5,0.5,lyric="房"),
            make_note(3,0.5,lyric="结"),make_note(2,1,lyric="尾"),
            make_note(1,0.5,lyric="~~"),
        ], "repeatEnding": {"numbers": [1]}, "barline": "double"},
        {"notes": [
            make_note(5,0.5,lyric="间"),make_note(3,0.5,lyric="奏"),
            make_note(2,0.5,lyric="过"),make_note(1,0.5,lyric="渡"),
            make_note(6,0.5,octave=-1,lyric="~~"),make_note(5,0.5,lyric="~~"),
            make_note(6,0.25,octave=-1),make_note(1,0.25,lyric="~~"),
            make_note(2,0.5,lyric="~~"),
        ]},
        {"notes": [
            make_note(1,1,lyric="二"),make_note(5,0.5,lyric="房"),
            make_note(3,0.5,lyric="终"),make_note(2,1,lyric="止"),
            make_note(1,1),
        ], "repeatEnding": {"numbers": [2]}, "barline": "end"},
    ]
}
all_scores.append(s8)

# ========== 9. 渐强渐弱 ==========
s9 = {
    "title": "力度变化",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(1,0.5,lyric="渐",dynamic="pp"),make_note(2,0.5,lyric="强"),
            make_note(3,0.5,lyric="自",dynamic="mp"),make_note(5,0.5,lyric="弱"),
            make_note(6,0.5,lyric="至"),make_note(1,0.5,octave=1,lyric="强",dynamic="f"),
            make_note(6,0.5,lyric="~"),make_note(5,0.5,lyric="~"),
        ], "dynamics": {"type": "crescendo", "endMeasureIndex": 1}},
        {"notes": [
            make_note(1,0.25,octave=1,lyric="渐"),make_note(6,0.25,lyric="强"),
            make_note(5,0.25,lyric="至"),make_note(3,0.25,lyric="顶"),
            make_note(2,0.5,lyric="峰"),make_note(3,0.5,lyric="~"),
            make_note(5,0.5,lyric="~"),make_note(6,0.25,lyric="~"),make_note(5,0.25,lyric="~"),
        ]},
        {"notes": [
            make_note(6,0.5,lyric="渐"),make_note(5,0.5,lyric="弱"),
            make_note(3,0.5,lyric="自"),make_note(2,0.5,lyric="强"),
            make_note(1,0.5,lyric="至"),make_note(6,0.5,octave=-1,lyric="弱"),
            make_note(5,0.5,lyric="~"),make_note(3,0.5,lyric="~"),
        ], "dynamics": {"type": "descrescendo", "endMeasureIndex": 3}},
        {"notes": [
            make_note(2,0.5,lyric="渐"),make_note(1,0.5,lyric="弱"),
            make_note(6,0.5,octave=-1,lyric="至"),make_note(5,0.5,octave=-1,lyric="底"),
            make_note(6,0.25,octave=-1,lyric="~"),make_note(1,0.25,lyric="~"),
            make_note(2,0.5,lyric="~"),make_note(3,0.5,lyric="~"),
            make_note(1,0.5,lyric="~"),
        ]},
    ]
}
all_scores.append(s9)

# ========== 10. 综合符号大全 ==========
s10 = {
    "title": "综合符号示例",
    "key": "C",
    "timeSignature": {"numerator": 4, "denominator": 4},
    "measures": [
        {"notes": [
            make_note(5,0.5,lyric="综"),
            make_note(3,0.25,lyric="合"),make_note(2,0.25),
            make_note(1,0.5,lyric="符",techniques=[{"type":"boyin"}]),
            make_note(6,0.5,octave=-1,lyric="号"),
            make_note(1,0.5,lyric="大"),make_note(2,0.5,lyric="全"),
        ]},
        {"notes": [
            make_note(3,0.5,lyric="顿",techniques=[{"type":"dunyin"}]),
            make_note(5,0.25,lyric="滑",techniques=[{"type":"huayin","slideDirection":"up"}]),
            make_note(6,0.25,lyric="音"),
            make_note(3,0.5,lyric="波",techniques=[{"type":"boyin"}]),
            make_note(2,0.5,lyric="保",tenuto=True),
            make_note(1,0.5,lyric="延",fermata=True),
            make_note(5,0.25,accidental="sharp",techniques=[{"type":"dayin"}]),
            make_note(3,0.25,lyric="~"),
            make_note(2,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(6,0.5,lyric="打",techniques=[{"type":"dayin"}]),
            make_note(1,0.5,octave=1,lyric="叠",techniques=[{"type":"dieyin"}]),
            make_note(5,0.5,lyric="吐",techniques=[{"type":"tuyin"}]),
            make_note(3,0.5,lyric="倚",techniques=[{"type":"yinyin","graceNotes":[2],"graceOctave":0}]),
            make_note(5,0.5,lyric="颤",techniques=[{"type":"chanyin"}]),
            make_note(6,0.5,lyric="顿",techniques=[{"type":"dunyin"}]),
            make_note(5,0.5,lyric="~"),
        ]},
        {"notes": [
            make_note(1,1,octave=1,lyric="低",slurId="z1"),
            make_note(6,0.5,lyric="音",slurId="z1"),
            make_note(5,1,lyric="连",tieId="zt1"),
            make_note(5,0.5,lyric="~",tieId="zt1"),
            make_note(3,0.5,lyric="~"),make_note(2,0.25,lyric="~"),make_note(1,0.25,lyric="~"),
        ]},
        {"notes": [
            make_note(2,0.5,lyric="跨",slurId="z2"),
            make_note(3,0.5,lyric="节",slurId="z2"),
            make_note(5,1,lyric="长",fermata=True),
            make_note(6,0.5,octave=-1,techniques=[{"type":"dunyin"}]),
            make_note(1,0.5,lyric="止"),make_note(2,1),
        ], "barline": "end"},
    ]
}
all_scores.append(s10)

# ========== 写出文件 ==========
os.makedirs("public/training", exist_ok=True)
for i, s in enumerate(all_scores):
    fname = f"public/training/score_{i+1:03d}.json"
    with open(fname, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    beats = sum(n.get("duration",0)*(1.5 if n.get("dot") else 1) for m in s["measures"] for n in m["notes"])
    print(f"✓ {fname} ({len(s['measures'])}小节, {beats:.1f}拍)")

# 更新 index.json
scores = [{"file": f"score_{i+1:03d}.json", "title": s["title"], "measures": len(s["measures"])} for i,s in enumerate(all_scores)]
with open("public/training/index.json", "w") as f:
    json.dump(scores, f, ensure_ascii=False, indent=2)
print(f"\nDone! {len(all_scores)} scores → public/training/")
