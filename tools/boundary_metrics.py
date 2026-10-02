"""Shared V7 boundary metrics; legacy evaluators retain their old diagnostics.

Ranges accepted here are half-open. All gap positions, including endpoints,
are candidates. Perfection counts reference fields whose endpoints are both
predicted and whose interior contains no predicted boundary.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable


COUNT_KEYS = ("tp", "tn", "fp", "fn", "perfect_fields", "reference_fields")
RATE_KEYS = ("accuracy", "precision", "recall", "f1", "perfection")


def rates(counts: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(counts)
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    positions = tp + tn + fp + fn
    result.update(
        candidate_positions=positions,
        accuracy=(tp + tn) / positions if positions else None,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        perfection=(counts["perfect_fields"] / counts["reference_fields"]
                    if counts["reference_fields"] else None),
    )
    return result


def interval_metrics(
    reference: Iterable[tuple[int, int]],
    predicted: Iterable[tuple[int, int]],
    width: int,
) -> dict[str, Any]:
    reference, predicted = set(reference), set(predicted)
    if width < 0:
        raise ValueError("width must be non-negative")
    if any(not (0 <= start < end <= width) for start, end in reference | predicted):
        raise ValueError("field interval outside its byte/bit domain")
    gt = {point for interval in reference for point in interval}
    pred = {point for interval in predicted for point in interval}
    universe = set(range(width + 1))
    return rates({
        "tp": len(gt & pred),
        "tn": len(universe - (gt | pred)),
        "fp": len(pred - gt),
        "fn": len(gt - pred),
        "perfect_fields": sum(
            start in pred and end in pred and not any(start < point < end for point in pred)
            for start, end in reference
        ),
        "reference_fields": len(reference),
    })


def merge_metrics(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in COUNT_KEYS}
    for metric in metrics:
        for key in COUNT_KEYS:
            counts[key] += int(metric[key])
    return rates(counts)


def byte_metrics(reference: Iterable[tuple[int, int]], predicted: Iterable[tuple[int, int]],
                 payload_length: int) -> dict[str, Any]:
    return interval_metrics(
        {(start, end + 1) for start, end in reference},
        {(start, end + 1) for start, end in predicted},
        payload_length,
    )


def bit_metrics(reference: dict, predicted: dict) -> dict[str, Any]:
    def intervals(labels: Iterable[str]) -> set[tuple[int, int]]:
        result = set()
        for label in labels:
            match = re.fullmatch(r"\[(\d+)(?::(\d+))?\]", label.replace(" ", ""))
            if not match:
                raise ValueError(f"invalid bit label: {label!r}")
            first = int(match[1])
            second = int(match[2]) if match[2] is not None else first
            result.add((min(first, second), max(first, second) + 1))
        return result

    # Parent identity is preserved, as in the existing bit-boundary evaluator.
    # Include prediction-only parents, so their boundaries are not silently lost.
    return merge_metrics(
        interval_metrics(intervals(reference.get(parent, set())),
                         intervals(predicted.get(parent, set())),
                         8 * (parent[1] - parent[0] + 1))
        for parent in sorted(set(reference) | set(predicted))
    )


def packet_paper_metrics(reference_fields: Iterable, predicted_fields: Iterable,
                         reference_bits: dict, predicted_bits: dict,
                         payload_length: int) -> dict[str, Any]:
    return {
        "byte": byte_metrics(reference_fields, predicted_fields, payload_length),
        "bit": bit_metrics(reference_bits, predicted_bits),
    }


def merge_paper_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(items)
    return {kind: merge_metrics(item[kind] for item in items) for kind in ("byte", "bit")}


def average_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    # No-reference protocols (e.g. no bit ground truth) are not bit-table rows.
    items = [item for item in items if item["reference_fields"] > 0]
    return {
        **{key: (sum(float(item[key]) for item in items) / len(items) if items else None)
           for key in RATE_KEYS},
        "protocol_count": len(items),
    }


def attach_paper_summary(summary: dict[str, Any]) -> None:
    protocols = summary["protocols"]
    summary["overall"]["paper_metrics"] = merge_paper_metrics(
        protocol["paper_metrics"] for protocol in protocols)
    averages = {kind: average_metrics(protocol["paper_metrics"][kind] for protocol in protocols)
                for kind in ("byte", "bit")}
    summary["protocol_value_average"]["paper_metrics"] = averages
    summary["paper_metric_definitions"] = {
        "accuracy": "(TP + TN) / (TP + TN + FP + FN), over all candidate gap positions",
        "f1": "boundary precision/recall harmonic mean; not exact-field F1",
        "perfection": "fraction of reference fields with both endpoints and no extra interior boundary",
        "endpoints": "included; byte positions 0..payload_length, bit positions 0..parent_width",
        "bit_parent_domain": "union of reference and predicted parents; parent identity is retained",
        "table_aggregation": "arithmetic mean across protocols with reference fields at that granularity",
        "overall": "pooled confusion counts, reported separately from protocol averages",
    }


def write_paper_report(summary: dict[str, Any], output_dir: Path) -> None:
    rows = []
    groups = [(protocol["protocol"], protocol["paper_metrics"]) for protocol in summary["protocols"]]
    groups += [("Protocol Value Avg", summary["protocol_value_average"]["paper_metrics"]),
               ("Overall Micro", summary["overall"]["paper_metrics"])]
    for name, metrics in groups:
        for kind in ("byte", "bit"):
            rows.append({"group": name, "granularity": kind, **{key: metrics[kind][key]
                         for key in ("accuracy", "f1", "perfection")}})
    with (output_dir / "field_boundary_v7_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# V7 Field-Boundary Metrics", "",
             "Protocol Value Avg is the arithmetic mean over protocols with reference fields. "
             "Overall Micro pools counts. N/A means no applicable reference/domain.", "",
             "| Group | Granularity | Accuracy | F1-score | Perfection |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        values = ["N/A" if row[key] is None else f"{row[key]:.4f}"
                  for key in ("accuracy", "f1", "perfection")]
        lines.append(f"| {row['group']} | {row['granularity']} | " + " | ".join(values) + " |")
    (output_dir / "field_boundary_v7_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
