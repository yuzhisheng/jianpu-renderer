"""Multi-branch vocabulary for image-to-jianpu recognition.

One decoder position represents one musical event.  The event kind drives the
sequence while orthogonal heads describe pitch, accidental, octave, duration
and articulation.  Unknown silver-label branches use ``IGNORE_ID`` in the loss
instead of teaching the model that a modifier is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping


IGNORE_ID = -100

BRANCH_TOKENS: Dict[str, List[str]] = {
    "kind": [
        "<PAD>", "<BOS>", "<EOS>", "NOTE", "REST", "EXTEND",
        "BAR", "BAR_DOUBLE", "BAR_FINAL", "BAR_REPEAT_START",
        "BAR_REPEAT_END", "UNKNOWN",
    ],
    "pitch": ["NONE", *(f"P{i}" for i in range(1, 8))],
    "accidental": ["NONE", "SHARP", "FLAT", "NATURAL"],
    "octave": ["NONE", "DOWN2", "DOWN1", "UP1", "UP2"],
    "duration": [
        "NONE", "WHOLE", "HALF", "QUARTER", "EIGHTH", "SIXTEENTH",
        "THIRTY_SECOND", "DOTTED",
    ],
    "articulation": ["NONE", "FERMATA", "TENUTO", "ACCENT", "STACCATO"],
}

BRANCHES = tuple(BRANCH_TOKENS)
TOKEN_TO_ID = {
    branch: {token: index for index, token in enumerate(tokens)}
    for branch, tokens in BRANCH_TOKENS.items()
}
ID_TO_TOKEN = {
    branch: {index: token for token, index in mapping.items()}
    for branch, mapping in TOKEN_TO_ID.items()
}

PAD_ID = TOKEN_TO_ID["kind"]["<PAD>"]
BOS_ID = TOKEN_TO_ID["kind"]["<BOS>"]
EOS_ID = TOKEN_TO_ID["kind"]["<EOS>"]
NOTE_ID = TOKEN_TO_ID["kind"]["NOTE"]

CTC_TOKENS = [
    "<BLANK>", *(f"P{i}" for i in range(1, 8)), "R0", "-",
    "B|", "B||", "B|]", "B|:", "B:|",
]
CTC_TOKEN_TO_ID = {token: index for index, token in enumerate(CTC_TOKENS)}
CTC_ID_TO_TOKEN = {index: token for token, index in CTC_TOKEN_TO_ID.items()}


@dataclass(frozen=True)
class JianpuEvent:
    kind: str
    pitch: str = "NONE"
    accidental: str = "NONE"
    octave: str = "NONE"
    duration: str = "NONE"
    articulation: str = "NONE"

    def as_ids(self) -> Dict[str, int]:
        return {
            branch: TOKEN_TO_ID[branch][getattr(self, branch)]
            for branch in BRANCHES
        }


BAR_TO_KIND = {
    "B|": "BAR",
    "B||": "BAR_DOUBLE",
    "B|]": "BAR_FINAL",
    "B|:": "BAR_REPEAT_START",
    "B:|": "BAR_REPEAT_END",
    "|": "BAR",
    "||": "BAR_DOUBLE",
}
KIND_TO_BAR = {value: key for key, value in BAR_TO_KIND.items() if key.startswith("B")}


def tokens_to_events(tokens: Iterable[str]) -> List[JianpuEvent]:
    """Convert the existing flat token format to aligned event branches.

    The local VLM labels intentionally omit octave and duration detail.  Those
    branches are therefore masked by :func:`events_to_targets`; explicit
    modifiers are still retained for future gold/synthetic labels.
    """
    events: List[JianpuEvent] = []
    pending_accidental = "NONE"
    accidental_map = {"#": "SHARP", "b": "FLAT", "n": "NATURAL"}
    for token in tokens:
        if token in accidental_map:
            value = accidental_map[token]
            if events and events[-1].kind == "NOTE" and events[-1].accidental == "NONE":
                events[-1] = JianpuEvent(**{
                    **events[-1].__dict__, "accidental": value,
                })
            else:
                pending_accidental = value
        elif token.startswith("P") and token[1:].isdigit() and 1 <= int(token[1:]) <= 7:
            events.append(JianpuEvent("NOTE", token, accidental=pending_accidental))
            pending_accidental = "NONE"
        elif token in {"R0", "P0", "0"}:
            events.append(JianpuEvent("REST"))
        elif token == "-":
            events.append(JianpuEvent("EXTEND"))
        elif token in BAR_TO_KIND:
            events.append(JianpuEvent(BAR_TO_KIND[token]))
        elif token in {"^", "O+1"} and events and events[-1].kind == "NOTE":
            events[-1] = JianpuEvent(**{**events[-1].__dict__, "octave": "UP1"})
        elif token in {"v", "O-1"} and events and events[-1].kind == "NOTE":
            events[-1] = JianpuEvent(**{**events[-1].__dict__, "octave": "DOWN1"})
        elif token in {"_", "D0.5"} and events and events[-1].kind in {"NOTE", "REST"}:
            events[-1] = JianpuEvent(**{**events[-1].__dict__, "duration": "EIGHTH"})
        elif token in {"=", "D0.25"} and events and events[-1].kind in {"NOTE", "REST"}:
            events[-1] = JianpuEvent(**{**events[-1].__dict__, "duration": "SIXTEENTH"})
        elif token == "." and events and events[-1].kind in {"NOTE", "REST"}:
            events[-1] = JianpuEvent(**{**events[-1].__dict__, "duration": "DOTTED"})
        elif token in {"FER", "TEN", "ACC"} and events and events[-1].kind == "NOTE":
            name = {"FER": "FERMATA", "TEN": "TENUTO", "ACC": "ACCENT"}[token]
            events[-1] = JianpuEvent(**{**events[-1].__dict__, "articulation": name})
    return events


def events_to_tokens(events: Iterable[JianpuEvent]) -> List[str]:
    result: List[str] = []
    accidental = {"SHARP": "#", "FLAT": "b", "NATURAL": "n"}
    octave = {"UP1": "O+1", "UP2": "O+2", "DOWN1": "O-1", "DOWN2": "O-2"}
    duration = {
        "EIGHTH": "_", "SIXTEENTH": "=", "THIRTY_SECOND": "=",
        "DOTTED": ".", "HALF": "D2", "WHOLE": "D4", "QUARTER": "D1",
    }
    articulation = {
        "FERMATA": "FER", "TENUTO": "TEN", "ACCENT": "ACC",
        "STACCATO": "T:du",
    }
    for event in events:
        if event.kind == "NOTE":
            result.append(event.pitch if event.pitch != "NONE" else "<UNK>")
            if event.accidental != "NONE":
                result.append(accidental[event.accidental])
            if event.octave != "NONE":
                result.append(octave[event.octave])
            if event.duration != "NONE":
                result.append(duration[event.duration])
            if event.articulation != "NONE":
                result.append(articulation[event.articulation])
        elif event.kind == "REST":
            result.append("R0")
        elif event.kind == "EXTEND":
            result.append("-")
        elif event.kind in KIND_TO_BAR:
            result.append(KIND_TO_BAR[event.kind])
    return result


def events_to_ctc_tokens(events: Iterable[JianpuEvent]) -> List[str]:
    """Flatten only monotonic skeleton symbols for the CTC alignment head."""
    result = []
    for event in events:
        if event.kind == "NOTE" and event.pitch in CTC_TOKEN_TO_ID:
            result.append(event.pitch)
        elif event.kind == "REST":
            result.append("R0")
        elif event.kind == "EXTEND":
            result.append("-")
        elif event.kind in KIND_TO_BAR:
            result.append(KIND_TO_BAR[event.kind])
    return result


def events_to_targets(
    events: Iterable[JianpuEvent], *, skeleton_label: bool = False,
) -> Dict[str, List[int]]:
    """Encode events with BOS/EOS and branch-aware supervision masks."""
    sequence = [JianpuEvent("<BOS>"), *events, JianpuEvent("<EOS>")]
    targets = {branch: [] for branch in BRANCHES}
    for event in sequence:
        ids = event.as_ids()
        for branch in BRANCHES:
            value = ids[branch]
            if branch != "kind" and event.kind != "NOTE":
                value = IGNORE_ID
            elif skeleton_label and branch in {"octave", "duration", "articulation"}:
                value = IGNORE_ID
            elif skeleton_label and branch == "accidental" and event.accidental == "NONE":
                value = IGNORE_ID
            targets[branch].append(value)
    return targets


def teacher_inputs(targets: Mapping[str, List[int]]) -> Dict[str, List[int]]:
    """Replace masked labels with neutral embeddings for decoder input."""
    return {
        branch: [TOKEN_TO_ID[branch]["NONE"] if value == IGNORE_ID else value
                 for value in values]
        if branch != "kind" else list(values)
        for branch, values in targets.items()
    }
