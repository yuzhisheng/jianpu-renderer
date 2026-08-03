#!/usr/bin/env python3
"""Create a page-disjoint web/NAS adaptation split from the original test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPT_PAGES = {
    "nas_gusu-xing_p1", "nas_hongdou_jp",
    "web_qupu123_000", "web_qupu123_002", "web_qupu123_004",
    "web_qupu123_006", "web_qupu123_008",
}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="backend/jianpu_transformer_dataset")
    args = parser.parse_args()
    root = (ROOT / args.data).resolve()
    train, test = read(root / "train.jsonl"), read(root / "test.jsonl")
    adapt = [row for row in test if row["annotation_id"] in DEFAULT_ADAPT_PAGES]
    holdout = [row for row in test if row["annotation_id"] not in DEFAULT_ADAPT_PAGES]
    write(root / "train_domain_adapt.jsonl", [*train, *adapt])
    write(root / "test_final_holdout.jsonl", holdout)
    stats = {
        "base_train_rows": len(train), "adapt_rows": len(adapt),
        "final_holdout_rows": len(holdout),
        "adapt_pages": sorted(DEFAULT_ADAPT_PAGES),
        "final_holdout_pages": sorted({row["annotation_id"] for row in holdout}),
    }
    (root / "domain_adaptation_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
