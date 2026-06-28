#!/usr/bin/env python3
"""Filter low-information fields from the Stage 3 all-fields dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_dataset/stage3_dataset_all_fields.csv")
DEFAULT_LABELS_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_observation/stage3_field_labels.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_filtered")
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "stage3_dataset_semantic_fields.csv"
DEFAULT_TRANSPARENT_REPORT = DEFAULT_OUTPUT_DIR / "stage3_transparent_field_report.json"
DEFAULT_LOW_VALUE_REPORT = DEFAULT_OUTPUT_DIR / "stage3_low_value_field_report.json"

KEY_COLUMNS = ["protocol_name", "sample_id", "field_id"]


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter transparent / low-information fields from Stage 3 all-fields dataset.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input stage3_dataset_all_fields.csv path.")
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV, help="Input stage3_field_labels.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for filtered dataset.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output semantic_fields CSV path.")
    parser.add_argument("--transparent-report", type=Path, default=DEFAULT_TRANSPARENT_REPORT, help="Transparent field report path.")
    parser.add_argument("--low-value-report", type=Path, default=DEFAULT_LOW_VALUE_REPORT, help="Low-value field report path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_csv == DEFAULT_OUTPUT_CSV:
        args.output_csv = args.output_dir / DEFAULT_OUTPUT_CSV.name
    if args.transparent_report == DEFAULT_TRANSPARENT_REPORT:
        args.transparent_report = args.output_dir / DEFAULT_TRANSPARENT_REPORT.name
    if args.low_value_report == DEFAULT_LOW_VALUE_REPORT:
        args.low_value_report = args.output_dir / DEFAULT_LOW_VALUE_REPORT.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def row_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row["protocol_name"]), str(row["sample_id"]), str(row["field_id"]))


def load_csv(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def build_label_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {row_key(row): row for row in rows}


def filter_rows(
    all_rows: Sequence[Dict[str, Any]],
    label_index: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    semantic_rows: List[Dict[str, Any]] = []
    transparent_rows: List[Dict[str, Any]] = []
    low_value_rows: List[Dict[str, Any]] = []

    for row in all_rows:
        labels = label_index.get(row_key(row))
        if labels is None:
            semantic_rows.append(dict(row))
            continue

        is_transparent = parse_bool(labels.get("is_transparent_candidate"))
        is_low_value = parse_bool(labels.get("is_low_value_candidate"))

        if is_transparent:
            merged = dict(row)
            merged.update(labels)
            transparent_rows.append(merged)
            continue

        merged = dict(row)
        merged.update(labels)
        semantic_rows.append(merged)
        if is_low_value:
            low_value_rows.append(merged)

    return semantic_rows, transparent_rows, low_value_rows


def protocol_counter(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(str(row["protocol_name"]) for row in rows)
    return dict(sorted(counter.items()))


def sample_examples(rows: Sequence[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    keys = [
        "protocol_name",
        "sample_id",
        "field_id",
        "field_kind",
        "constraint_count",
        "unique_metric_vectors",
        "deltaf_dispersion",
        "overall_response_score",
        "is_transparent_candidate",
        "is_low_value_candidate",
    ]
    examples: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        examples.append({k: row.get(k) for k in keys if k in row})
    return examples


def build_transparent_report(
    all_rows: Sequence[Dict[str, Any]],
    semantic_rows: Sequence[Dict[str, Any]],
    transparent_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "input_row_count": len(all_rows),
        "semantic_row_count": len(semantic_rows),
        "filtered_transparent_count": len(transparent_rows),
        "filtered_transparent_ratio": (len(transparent_rows) / len(all_rows)) if all_rows else 0.0,
        "protocol_counts": protocol_counter(transparent_rows),
        "examples": sample_examples(transparent_rows),
    }


def build_low_value_report(low_value_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "low_value_count": len(low_value_rows),
        "protocol_counts": protocol_counter(low_value_rows),
        "examples": sample_examples(low_value_rows),
    }


def write_csv(rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in fieldnames})


def main() -> int:
    args = parse_args()
    resolve_output_paths(args)

    if not args.input_csv.exists():
        warn(f"Input CSV does not exist: {args.input_csv}")
        return 1
    if not args.labels_csv.exists():
        warn(f"Labels CSV does not exist: {args.labels_csv}")
        return 1

    all_rows, all_fieldnames = load_csv(args.input_csv)
    label_rows, _ = load_csv(args.labels_csv)
    label_index = build_label_index(label_rows)

    semantic_rows, transparent_rows, low_value_rows = filter_rows(all_rows, label_index)
    write_csv(semantic_rows, all_fieldnames, args.output_csv)

    transparent_report = build_transparent_report(all_rows, semantic_rows, transparent_rows)
    low_value_report = build_low_value_report(low_value_rows)

    ensure_parent(args.transparent_report)
    ensure_parent(args.low_value_report)
    args.transparent_report.write_text(json.dumps(transparent_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.low_value_report.write_text(json.dumps(low_value_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info(f"Wrote semantic dataset to {args.output_csv}", args.quiet)
    info(f"Wrote transparent-field report to {args.transparent_report}", args.quiet)
    info(f"Wrote low-value-field report to {args.low_value_report}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
