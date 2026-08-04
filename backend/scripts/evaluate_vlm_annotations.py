#!/usr/bin/env python3
"""Compare detector predictions with independently reviewed VLM silver labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from scripts.evaluate_recognizer import (
    flatten_notes,
    full_key,
    measure_keys,
    pitch_key,
    similarity,
)


SKELETON = {
    *(f"P{value}" for value in range(1, 8)),
    "R0", "-", "B|", "B||", "B|:", "B:|", "#", "b", "n",
}
PITCH_ONLY = {*(f"P{value}" for value in range(1, 8)), "R0", "#", "b", "n"}


def local_pitch_similarity(root: Path, annotation_id: str):
    hand_path = root / "vlm_reviews" / f"{annotation_id}.json"
    local_path = root / "local_vlm_pitch_reviews" / f"{annotation_id}.json"
    if not hand_path.exists() or not local_path.exists():
        return None
    hand = json.loads(hand_path.read_text())
    local = json.loads(local_path.read_text())
    local_rows = {row["source_row"]: row for row in local["rows"]
                  if row.get("content_type") == "score" and row.get("voices")}
    expected, actual = [], []
    for row in hand["rows"]:
        if row.get("content_type") != "score" or row["source_row"] not in local_rows:
            continue
        expected.extend(token for token in row["tokens"] if token in SKELETON)
        actual.extend(local_rows[row["source_row"]]["voices"][0]["tokens"])
    expected_pitch = [token for token in expected if token in PITCH_ONLY]
    actual_pitch = [token for token in actual if token in PITCH_ONLY]
    return {
        "structure": similarity(expected, actual),
        "pitch": similarity(expected_pitch, actual_pitch),
        "actual": len(actual),
        "expected": len(expected),
        "actual_pitch": len(actual_pitch),
        "expected_pitch": len(expected_pitch),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="backend/real_annotations")
    args = parser.parse_args()
    root = (ROOT / args.annotations).resolve()
    scores = sorted((root / "ground_truth_silver").glob("*.json"))
    pitch_values, note_values, measure_values, count_values = [], [], [], []

    for truth_path in scores:
        annotation_id = truth_path.stem
        prediction_path = root / "predictions" / f"{annotation_id}.json"
        if not prediction_path.exists():
            print(f"skip {annotation_id}: prediction missing")
            continue
        truth = json.loads(truth_path.read_text())
        predicted = json.loads(prediction_path.read_text())
        truth_notes = flatten_notes(truth)
        predicted_notes = flatten_notes(predicted)
        pitch = similarity(
            [pitch_key(item) for item in truth_notes],
            [pitch_key(item) for item in predicted_notes],
        )
        note = similarity(
            [full_key(item) for item in truth_notes],
            [full_key(item) for item in predicted_notes],
        )
        measure = similarity(measure_keys(truth), measure_keys(predicted))
        count = 1.0 - abs(len(truth_notes) - len(predicted_notes)) / max(
            len(truth_notes), len(predicted_notes), 1)
        pitch_values.append(pitch)
        note_values.append(note)
        measure_values.append(measure)
        count_values.append(count)
        print(f"{annotation_id}: pitch={pitch:.4f}, note={note:.4f}, "
              f"measure={measure:.4f}, count={count:.4f} "
              f"({len(predicted_notes)}/{len(truth_notes)})")
        local = local_pitch_similarity(root, annotation_id)
        if local:
            print(f"  local VLM pitch-only: {local['pitch']:.4f} "
                  f"({local['actual_pitch']}/{local['expected_pitch']} tokens)")
            print(f"  local VLM with bars/dashes: {local['structure']:.4f} "
                  f"({local['actual']}/{local['expected']} tokens)")

    count = max(len(pitch_values), 1)
    print("\n=== VLM-reviewed real pages (silver) ===")
    print(f"pages:              {len(pitch_values)}")
    print(f"pitch similarity:   {sum(pitch_values) / count:.4f}")
    print(f"note similarity:    {sum(note_values) / count:.4f}")
    print(f"measure similarity: {sum(measure_values) / count:.4f}")
    print(f"count similarity:   {sum(count_values) / count:.4f}")


if __name__ == "__main__":
    main()
