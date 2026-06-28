#!/usr/bin/env python3
"""Export SOTA-label vs program-log-description pairs for compatibility judging."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_PREDICTIONS = ROOT / "sota_evaluation/out/unified_semantic_candidates.csv"
DEFAULT_GROUNDTRUTH = (
    ROOT / "evaluation_from_program_log/groundtruth_result/eval/program_log_groundtruth_candidates.jsonl"
)
DEFAULT_OUTPUT = ROOT / "sota_evaluation/out/program_log_semantic_compatibility_pairs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GROUNDTRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_groundtruth(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            field_id = str(row.get("field_id", ""))
            if field_id.startswith("b:"):
                rows[(row["protocol_name"], row["sample_id"], field_id)] = row
    return rows


def main() -> None:
    args = parse_args()
    truth = load_groundtruth(args.groundtruth)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for prediction in read_csv(args.predictions):
        key = (prediction["protocol_name"], prediction["sample_id"], prediction["field_id"])
        gt = truth.get(key)
        if gt is None:
            continue
        dedupe_key = (prediction["method"], prediction["variant"], *key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(
            {
                "method": prediction["method"],
                "variant": prediction["variant"],
                "protocol_name": prediction["protocol_name"],
                "sample_id": prediction["sample_id"],
                "field_id": prediction["field_id"],
                "sota_raw_label": prediction["raw_label"],
                "sota_mapped_coarse_tag": prediction["mapped_coarse_tag"],
                "program_log_description": gt.get("program_log_description", ""),
                "program_log_needs_review": gt.get("needs_review", False),
                "program_log_review_reason": gt.get("review_reason", ""),
                "judge_status": "pending",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[export] rows={len(rows)} output={args.output}")
    print("[note] pairs require a dedicated type-vs-behavior compatibility judge before scoring")


if __name__ == "__main__":
    main()
