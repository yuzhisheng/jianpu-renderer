"""Convert symbol detections into a reading-order token stream.

Jianpu is two-dimensional: octave dots, accidentals and duration lines belong
to a nearby pitch digit.  Sorting every detected box by ``(y, x)`` destroys
that relationship.  This module builds a small notation graph with pitch/rest
symbols as roots, then serializes that graph for the existing score parser.
"""
from __future__ import annotations

from statistics import median
from typing import Dict, Iterable, List, Tuple


Detection = Tuple[int, float, float, float, float, float]

CLASS_TO_TOKEN = {
    0: "P1", 1: "P2", 2: "P3", 3: "P4", 4: "P5", 5: "P6", 6: "P7",
    7: "R0", 8: "-", 9: "_", 10: "=", 11: ".", 12: "^", 13: "v",
    14: "#", 15: "b", 16: "n", 17: "FER", 18: "TEN", 19: "ACC",
    20: "T:bo", 21: "T:ch", 22: "TIE:?", 23: "SLUR:?",
    24: "T:da", 25: "T:tu", 26: "T:di", 27: "T:li:up",
    28: "T:hu:up", 29: "T:yi:3", 30: "T:du", 31: "DYN:?",
    32: "B|", 33: "B||", 34: "B|]", 35: "B|:", 36: "B:|",
    37: "R1", 38: "CRES", 39: "DECRES", 40: "<LY>", 41: "sf",
}

ANCHOR_CLASSES = frozenset(range(8))
SEQUENTIAL_CLASSES = ANCHOR_CLASSES | frozenset({8, 32, 33, 34, 35, 36, 37, 38, 39})
# Lyrics need OCR before they can become useful tokens. Ties and slurs need
# endpoint prediction; their current placeholder tokens are intentionally kept
# out of the graph until those relationships are available.
MODIFIER_CLASSES = frozenset({9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                              20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 41})


def _box(det: Detection) -> Dict[str, float]:
    cls_id, cx, cy, width, height, conf = det
    return {"cls": int(cls_id), "cx": float(cx), "cy": float(cy),
            "w": float(width), "h": float(height), "conf": float(conf)}


def _cluster_anchor_rows(anchors: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    if not anchors:
        return []
    typical_h = median(a["h"] for a in anchors)
    threshold = max(typical_h * 0.85, 8.0)
    rows: List[List[Dict[str, float]]] = []
    for anchor in sorted(anchors, key=lambda a: a["cy"]):
        best = None
        best_distance = float("inf")
        for row in rows:
            baseline = median(item["cy"] for item in row)
            distance = abs(anchor["cy"] - baseline)
            if distance <= threshold and distance < best_distance:
                best, best_distance = row, distance
        if best is None:
            rows.append([anchor])
        else:
            best.append(anchor)
    return sorted(rows, key=lambda row: median(item["cy"] for item in row))


def _nearest_row(box: Dict[str, float], rows: List[List[Dict[str, float]]]) -> int:
    return min(range(len(rows)), key=lambda i: abs(box["cy"] - median(a["cy"] for a in rows[i])))


def _nearest_anchor(mod: Dict[str, float], anchors: List[Dict[str, float]]) -> Dict[str, float]:
    # Accidentals live left of their note, while augmentation dots live right.
    # Directional preference prevents attaching a symbol to the adjacent digit.
    if mod["cls"] in (14, 15, 16):
        directional = [a for a in anchors if a["cx"] >= mod["cx"]]
    elif mod["cls"] == 11:
        directional = [a for a in anchors if a["cx"] <= mod["cx"]]
    else:
        directional = anchors
    candidates = directional or anchors
    return min(candidates, key=lambda a: abs(a["cx"] - mod["cx"]))


def detections_to_tokens(detections: Iterable[Detection]) -> List[str]:
    """Serialize detections in musical reading order.

    The output vocabulary remains compatible with ``model.tokenizer`` and the
    existing rule/Transformer assembler.
    """
    boxes = [_box(d) for d in detections]
    anchors = [b for b in boxes if b["cls"] in ANCHOR_CLASSES]
    rows = _cluster_anchor_rows(anchors)
    if not rows:
        return ["<BOS>", "<EOS>"]

    row_items: List[List[Dict[str, float]]] = [[] for _ in rows]
    attachments: Dict[int, List[Dict[str, float]]] = {id(a): [] for a in anchors}

    for box in boxes:
        if box["cls"] in SEQUENTIAL_CLASSES:
            row_items[_nearest_row(box, rows)].append(box)
        elif box["cls"] in MODIFIER_CLASSES:
            row_index = _nearest_row(box, rows)
            row_anchors = rows[row_index]
            if box["cls"] in (9, 10):
                # A duration line commonly spans every short note in one beat.
                # It is a relation to all covered pitch roots, not a glyph that
                # belongs only to the closest digit.
                left, right = box["cx"] - box["w"] / 2, box["cx"] + box["w"] / 2
                covered = [a for a in row_anchors
                           if left - a["w"] * 0.25 <= a["cx"] <= right + a["w"] * 0.25]
                for anchor in covered or [_nearest_anchor(box, row_anchors)]:
                    attachments[id(anchor)].append(box)
            else:
                anchor = _nearest_anchor(box, row_anchors)
                attachments[id(anchor)].append(box)

    # Semantic order after a pitch is important to the state-machine parser.
    modifier_order = {12: 0, 13: 0, 14: 1, 15: 1, 16: 1, 11: 2,
                      9: 3, 10: 4, 17: 5, 18: 5, 19: 5}
    result = ["<BOS>"]
    for row_index, items in enumerate(row_items):
        if row_index:
            result.append("<ROW>")
        for item in sorted(items, key=lambda b: b["cx"]):
            token = CLASS_TO_TOKEN.get(item["cls"])
            if token:
                result.append(token)
            if item["cls"] in ANCHOR_CLASSES:
                mods = sorted(attachments[id(item)],
                              key=lambda b: (modifier_order.get(b["cls"], 6), b["cx"], b["cy"]))
                result.extend(CLASS_TO_TOKEN[m["cls"]] for m in mods if m["cls"] in CLASS_TO_TOKEN)
    result.append("<EOS>")
    return result
