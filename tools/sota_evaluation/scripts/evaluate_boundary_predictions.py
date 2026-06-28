#!/usr/bin/env python3
"""Evaluate canonical SOTA byte-field predictions against aligned byte-field groundtruth."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_PREDICTIONS = ROOT / "sota_evaluation/out/boundary_predictions.jsonl"
DEFAULT_GROUNDTRUTH = (
    ROOT / "evaluation_from_program_log/groundtruth_result/eval/program_log_groundtruth_candidates.jsonl"
)
DEFAULT_OUTDIR = ROOT / "sota_evaluation/out/field_boundary/program_log"
Range = Tuple[int, int]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def metrics(self) -> dict[str, float | int]:
        precision = ratio(self.tp, self.tp + self.fp)
        recall = ratio(self.tp, self.tp + self.fn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "jaccard": ratio(self.tp, self.tp + self.fp + self.fn),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GROUNDTRUTH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compare(gt: set[Any], pred: set[Any]) -> Counts:
    return Counts(len(gt & pred), len(pred - gt), len(gt - pred))


def boundaries(fields: Iterable[Range]) -> set[int]:
    result: set[int] = set()
    for start, end in fields:
        result.add(start)
        result.add(end + 1)
    return result


def internal_boundaries(fields: Iterable[Range], payload_length: int) -> set[int]:
    return {value for value in boundaries(fields) if value not in {0, payload_length}}


def load_groundtruth(path: Path) -> dict[tuple[str, str], set[Range]]:
    truth: dict[tuple[str, str], set[Range]] = defaultdict(set)
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            match = re.fullmatch(r"b:(\d+):(\d+)", str(row.get("field_id", "")))
            if match:
                truth[(row["protocol_name"], row["sample_id"])].add(
                    (int(match.group(1)), int(match.group(2)))
                )
    return dict(truth)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def covered_bytes(fields: set[Range], payload_length: int) -> int:
    covered: set[int] = set()
    for start, end in fields:
        covered.update(range(max(0, start), min(payload_length - 1, end) + 1))
    return len(covered)


def packet_metrics(row: dict[str, Any], gt: set[Range]) -> dict[str, Any]:
    payload_length = int(row["payload_length"])
    pred = {(int(item[0]), int(item[1])) for item in row.get("fields", [])}
    gt_boundary = boundaries(gt)
    pred_boundary = boundaries(pred)
    gt_internal = internal_boundaries(gt, payload_length)
    pred_internal = internal_boundaries(pred, payload_length)
    exact = compare(gt, pred)
    boundary = compare(gt_boundary, pred_boundary)
    return {
        "method": row["method"],
        "variant": row["variant"],
        "protocol": row["protocol"],
        "sample_id": row["sample_id"],
        "status": row["status"],
        "payload_length": payload_length,
        "gt_fields": len(gt),
        "pred_fields": len(pred),
        "exact": exact,
        "boundary": boundary,
        "coverage_bytes": covered_bytes(pred, payload_length),
        "over_segmentation": len(pred_internal - gt_internal),
        "under_segmentation": len(gt_internal - pred_internal),
        "notes": row.get("notes", ""),
    }


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    exact = Counts()
    boundary = Counts()
    payload_bytes = coverage_bytes = 0
    over = under = evaluable = 0
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        exact.add(row["exact"])
        boundary.add(row["boundary"])
        payload_bytes += row["payload_length"]
        coverage_bytes += row["coverage_bytes"]
        over += row["over_segmentation"]
        under += row["under_segmentation"]
        status_counts[row["status"]] += 1
        if row["status"] == "ok":
            evaluable += 1
    return {
        "group": label,
        "packets": len(rows),
        "evaluable_packets": evaluable,
        "evaluable_rate": ratio(evaluable, len(rows)),
        "exact": exact.metrics(),
        "boundary": boundary.metrics(),
        "coverage": ratio(coverage_bytes, payload_bytes),
        "over_segmentation": over,
        "under_segmentation": under,
        "status_counts": dict(sorted(status_counts.items())),
    }


def flat(summary: dict[str, Any], method: str, variant: str) -> dict[str, Any]:
    return {
        "method": method,
        "variant": variant,
        "group": summary["group"],
        "packets": summary["packets"],
        "evaluable_packets": summary["evaluable_packets"],
        "evaluable_rate": summary["evaluable_rate"],
        "exact_precision": summary["exact"]["precision"],
        "exact_recall": summary["exact"]["recall"],
        "exact_f1": summary["exact"]["f1"],
        "boundary_precision": summary["boundary"]["precision"],
        "boundary_recall": summary["boundary"]["recall"],
        "boundary_f1": summary["boundary"]["f1"],
        "coverage": summary["coverage"],
        "over_segmentation": summary["over_segmentation"],
        "under_segmentation": summary["under_segmentation"],
        "status_counts": json.dumps(summary["status_counts"], sort_keys=True),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return ratio(sum(values), len(values))


def protocol_value_avg(protocol_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "group": "Protocol Value Avg",
        "packets": sum(row["packets"] for row in protocol_rows),
        "evaluable_packets": sum(row["evaluable_packets"] for row in protocol_rows),
        "evaluable_rate": mean(row["evaluable_rate"] for row in protocol_rows),
        "exact": {key: mean(row["exact"][key] for row in protocol_rows) for key in ("precision", "recall", "f1", "jaccard")},
        "boundary": {key: mean(row["boundary"][key] for row in protocol_rows) for key in ("precision", "recall", "f1", "jaccard")},
        "coverage": mean(row["coverage"] for row in protocol_rows),
        "over_segmentation": sum(row["over_segmentation"] for row in protocol_rows),
        "under_segmentation": sum(row["under_segmentation"] for row in protocol_rows),
        "status_counts": {},
    }


def macro_non_empty(protocol_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric(kind: str, name: str) -> float:
        valid = [
            row[kind][name]
            for row in protocol_rows
            if row[kind]["tp"] + row[kind]["fp"] > 0
            and row[kind]["tp"] + row[kind]["fn"] > 0
        ]
        return mean(valid)

    exact_precision = metric("exact", "precision")
    exact_recall = metric("exact", "recall")
    boundary_precision = metric("boundary", "precision")
    boundary_recall = metric("boundary", "recall")
    return {
        "group": "Macro Non-Empty",
        "packets": sum(row["packets"] for row in protocol_rows),
        "evaluable_packets": sum(row["evaluable_packets"] for row in protocol_rows),
        "evaluable_rate": mean(row["evaluable_rate"] for row in protocol_rows),
        "exact": {
            "precision": exact_precision,
            "recall": exact_recall,
            "f1": f1(exact_precision, exact_recall),
            "jaccard": metric("exact", "jaccard"),
        },
        "boundary": {
            "precision": boundary_precision,
            "recall": boundary_recall,
            "f1": f1(boundary_precision, boundary_recall),
            "jaccard": metric("boundary", "jaccard"),
        },
        "coverage": mean(row["coverage"] for row in protocol_rows),
        "over_segmentation": sum(row["over_segmentation"] for row in protocol_rows),
        "under_segmentation": sum(row["under_segmentation"] for row in protocol_rows),
        "status_counts": {},
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SOTA Field-Boundary Metrics",
        "",
        "Bit-field metrics are intentionally excluded. `end` offsets are normalized to inclusive ranges.",
        "",
        "| Method | Variant | Group | Packets | Evaluable | Exact F1 | Boundary-hit F1 | Coverage | Over-seg. | Under-seg. |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {variant} | {group} | {packets} | {evaluable_packets} | "
            "{exact_f1:.4f} | {boundary_f1:.4f} | {coverage:.4f} | "
            "{over_segmentation} | {under_segmentation} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    truth = load_groundtruth(args.groundtruth)
    prediction_rows = load_predictions(args.predictions)
    packet_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        key = (row["protocol"], row["sample_id"])
        if key in truth:
            packet_rows.append(packet_metrics(row, truth[key]))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in packet_rows:
        grouped[(row["method"], row["variant"])].append(row)

    summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for (method, variant), method_rows in sorted(grouped.items()):
        protocols: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            protocols[row["protocol"]].append(row)
        protocol_summaries = [summarize(rows, protocol) for protocol, rows in sorted(protocols.items())]
        all_summaries = protocol_summaries + [
            protocol_value_avg(protocol_summaries),
            summarize(method_rows, "Overall Micro"),
            macro_non_empty(protocol_summaries),
        ]
        for summary in all_summaries:
            summaries.append({"method": method, "variant": variant, **summary})
            flat_rows.append(flat(summary, method, variant))

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "boundary_metrics_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = args.outdir / "boundary_metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
    write_markdown(args.outdir / "boundary_metrics_summary.md", flat_rows)
    print(f"[evaluate] packets={len(packet_rows)} output={args.outdir}")


if __name__ == "__main__":
    main()
