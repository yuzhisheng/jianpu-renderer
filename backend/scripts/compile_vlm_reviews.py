#!/usr/bin/env python3
"""Compile reviewed row tokens into silver Score JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from assembler import parse_tokens_to_score
from model.tokenizer import TOKEN2ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", default="backend/real_annotations/vlm_reviews")
    parser.add_argument("--output", default="backend/real_annotations/ground_truth_silver")
    args = parser.parse_args()

    reviews = (ROOT / args.reviews).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for path in sorted(reviews.glob("*.json")):
        review = json.loads(path.read_text())
        tokens = []
        score_rows = [row for row in review["rows"] if row["content_type"] == "score"]
        for index, row in enumerate(score_rows):
            row_tokens = row["tokens"]
            unknown = [token for token in row_tokens if token not in TOKEN2ID]
            if unknown:
                raise ValueError(f"{path.name} row {row['source_row']}: unknown tokens {unknown}")
            if index:
                tokens.append("<ROW>")
            tokens.extend(row_tokens)

        score = parse_tokens_to_score(tokens)
        metadata = review.get("metadata", {})
        for key in ("title", "key", "timeSignature", "tempo"):
            if key in metadata:
                score[key] = metadata[key]
        score["_annotation"] = {
            "grade": "silver_vlm_pass_1",
            "reviewer": review.get("reviewer"),
            "source_review": str(path),
            "minimum_row_confidence": min(
                (row.get("confidence", 0.0) for row in score_rows), default=0.0),
            "uncertainties": [
                item for row in score_rows for item in row.get("uncertainties", [])
            ],
        }
        target = output / f"{review['annotation_id']}.json"
        target.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n")
        record = {
            "annotation_id": review["annotation_id"],
            "score": str(target.relative_to(output.parent)),
            "score_rows": len(score_rows),
            "measures": len(score["measures"]),
            "minimum_row_confidence": score["_annotation"]["minimum_row_confidence"],
            "status": review["status"],
        }
        records.append(record)
        print(f"{review['annotation_id']}: rows={len(score_rows)}, "
              f"measures={len(score['measures'])}")

    (output.parent / "vlm_manifest.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) +
        ("\n" if records else ""))
    print(f"compiled={len(records)}, output={output}")


if __name__ == "__main__":
    main()
