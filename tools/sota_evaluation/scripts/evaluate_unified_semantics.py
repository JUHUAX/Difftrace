#!/usr/bin/env python3
"""Evaluate mapped SOTA coarse semantic tags against tshark semantic groundtruth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_PREDICTIONS = ROOT / "sota_evaluation/out/unified_semantic_candidates.csv"
DEFAULT_ALIGNMENT = ROOT / "sota_evaluation/out/tshark_sample_alignment.csv"
DEFAULT_GROUNDTRUTH = (
    ROOT / "evaluation_from_tshark/semantic_inference/eval/tshark_semantic_groundtruth_candidates.csv"
)
DEFAULT_OUTDIR = ROOT / "sota_evaluation/out/unified_semantics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GROUNDTRUTH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--exclude-other", action="store_true")
    parser.add_argument("--exclude-review", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth_rows(path: Path, exclude_other: bool, exclude_review: bool) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    for row in read_csv(path):
        field_id = row["field_id"]
        if not field_id.startswith("b:"):
            continue
        if exclude_other and row["semantic_group"] == "other_or_unknown":
            continue
        if exclude_review and row["needs_review"].strip().lower() in {"1", "true", "yes"}:
            continue
        result[(row["protocol_name"], row["sample_id"], field_id)] = row["semantic_group"]
    return result


def summarize(rows: list[dict[str, Any]], group: str, total_gt: int) -> dict[str, Any]:
    matched = len(rows)
    correct = sum(row["hit"] for row in rows)
    mappable = sum(row["mapped_coarse_tag"] != "other_or_unknown" for row in rows)
    return {
        "group": group,
        "groundtruth_fields": total_gt,
        "matched_fields": matched,
        "mappable_fields": mappable,
        "hit_fields": correct,
        "semantic_coverage": matched / total_gt if total_gt else 0.0,
        "mappable_coverage": mappable / total_gt if total_gt else 0.0,
        "matched_accuracy": correct / matched if matched else 0.0,
        "conditional_mapped_accuracy": correct / mappable if mappable else 0.0,
        "effective_hit_rate": correct / total_gt if total_gt else 0.0,
    }


def main() -> None:
    args = parse_args()
    alignment = {
        (row["protocol_name"], row["replay_sample_id"]): row["tshark_sample_id"]
        for row in read_csv(args.alignment)
    }
    aligned_tshark_samples = {
        (protocol, tshark_sample)
        for (protocol, _), tshark_sample in alignment.items()
    }
    truth = {
        key: value
        for key, value in truth_rows(args.groundtruth, args.exclude_other, args.exclude_review).items()
        if (key[0], key[1]) in aligned_tshark_samples
    }
    details: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in read_csv(args.predictions):
        tshark_sample = alignment.get((row["protocol_name"], row["sample_id"]))
        if not tshark_sample:
            continue
        truth_key = (row["protocol_name"], tshark_sample, row["field_id"])
        expected = truth.get(truth_key)
        if expected is None:
            continue
        dedupe_key = (row["method"], row["variant"], *truth_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        details.append(
            {
                **row,
                "tshark_sample_id": tshark_sample,
                "groundtruth_coarse_tag": expected,
                "hit": int(row["mapped_coarse_tag"] == expected),
            }
        )

    gt_by_protocol: dict[str, int] = defaultdict(int)
    for protocol, _, _ in truth:
        gt_by_protocol[protocol] += 1
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(row["method"], row["variant"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (method, variant), method_rows in sorted(grouped.items()):
        by_protocol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            by_protocol[row["protocol_name"]].append(row)
        for protocol in sorted(gt_by_protocol):
            summaries.append(
                {
                    "method": method,
                    "variant": variant,
                    **summarize(by_protocol[protocol], protocol, gt_by_protocol[protocol]),
                }
            )
        summaries.append(
            {
                "method": method,
                "variant": variant,
                **summarize(method_rows, "Overall Micro", sum(gt_by_protocol.values())),
            }
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "unified_semantic_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]) if details else [])
        if details:
            writer.writeheader()
            writer.writerows(details)
    summary_path = args.outdir / "unified_semantic_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)
    (args.outdir / "unified_semantic_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[evaluate] matched_rows={len(details)} output={args.outdir}")


if __name__ == "__main__":
    main()
