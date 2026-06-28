#!/usr/bin/env python3
"""Observe Stage 3 all-fields dataset before transparent-field filtering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_dataset/stage3_dataset_all_fields.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_observation")
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "stage3_observation_summary.json"
DEFAULT_TABLES = DEFAULT_OUTPUT_DIR / "stage3_observation_tables.csv"
DEFAULT_LABELS = DEFAULT_OUTPUT_DIR / "stage3_field_labels.csv"
DEFAULT_GROUP_DIR = DEFAULT_OUTPUT_DIR / "stage3_group_rankings"

GROUPS = ["neighborhood", "boundary", "enum", "extreme"]
GROUP_SCORE_TYPES = [
    "mean_baseline_distance",
    "mean_pairwise_distance",
    "metric_vector_variance",
    "unique_vector_ratio",
]
ZERO_PAIRWISE_COLS = [
    "mean_pairwise_distance",
    "max_pairwise_distance",
    "metric_vector_variance",
    "loop_dispersion",
]
TRANSPARENT_LABEL_COLUMNS = [
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
GROUP_RANKING_COLUMNS = [
    "rank",
    "protocol_name",
    "sample_id",
    "field_id",
    "field_kind",
    "constraint_count",
    "unique_metric_vectors",
    "deltaf_dispersion",
    "mean_baseline_distance",
    "mean_pairwise_distance",
    "metric_vector_variance",
    "unique_vector_ratio",
]


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observe Stage 3 all-fields dataset and produce pre-filter diagnostics.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input stage3_dataset_all_fields.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Observation output directory.")
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY, help="Observation summary JSON path.")
    parser.add_argument("--output-tables", type=Path, default=DEFAULT_TABLES, help="Observation long-table CSV path.")
    parser.add_argument("--output-labels", type=Path, default=DEFAULT_LABELS, help="Per-field label CSV path.")
    parser.add_argument("--output-group-dir", type=Path, default=DEFAULT_GROUP_DIR, help="Per-group ranking output directory.")
    parser.add_argument("--zero-eps", type=float, default=1e-12, help="Numerical tolerance for zero-response checks.")
    parser.add_argument(
        "--transparent-max-unique-vectors",
        type=int,
        default=1,
        help="Maximum unique_metric_vectors allowed for a transparent candidate.",
    )
    parser.add_argument(
        "--low-value-score-quantile",
        type=float,
        default=0.25,
        help="Quantile on overall_response_score used to mark low-value candidates among eligible fields.",
    )
    parser.add_argument(
        "--low-value-dispersion-quantile",
        type=float,
        default=0.25,
        help="Quantile on deltaf_dispersion used to mark low-value candidates among eligible fields.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_summary == DEFAULT_SUMMARY:
        args.output_summary = args.output_dir / DEFAULT_SUMMARY.name
    if args.output_tables == DEFAULT_TABLES:
        args.output_tables = args.output_dir / DEFAULT_TABLES.name
    if args.output_labels == DEFAULT_LABELS:
        args.output_labels = args.output_dir / DEFAULT_LABELS.name
    if args.output_group_dir == DEFAULT_GROUP_DIR:
        args.output_group_dir = args.output_dir / DEFAULT_GROUP_DIR.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def read_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            converted = dict(row)
            converted["constraint_count"] = parse_int(row.get("constraint_count"))
            converted["unique_metric_vectors"] = parse_int(row.get("unique_metric_vectors"))
            converted["valid_mutations"] = parse_int(row.get("valid_mutations"))
            converted["mutation_count"] = parse_int(row.get("mutation_count"))
            converted["deltaf_dispersion"] = parse_float(row.get("deltaf_dispersion"))
            for group in GROUPS:
                for score_type in GROUP_SCORE_TYPES + ["max_pairwise_distance", "loop_dispersion"]:
                    col = f"{group}_{score_type}"
                    converted[col] = parse_float(row.get(col))
            rows.append(converted)
    return rows


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q = max(0.0, min(1.0, q))
    ordered = sorted(float(v) for v in values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_distribution(values: Iterable[float]) -> Dict[str, Any]:
    seq = [float(v) for v in values]
    if not seq:
        return {"count": 0}
    return {
        "count": len(seq),
        "min": min(seq),
        "p25": quantile(seq, 0.25),
        "median": quantile(seq, 0.5),
        "p75": quantile(seq, 0.75),
        "max": max(seq),
        "mean": sum(seq) / len(seq),
    }


def is_zero_response_for_group(row: Dict[str, Any], group: str, eps: float) -> bool:
    for suffix in ZERO_PAIRWISE_COLS:
        if abs(float(row[f"{group}_{suffix}"])) > eps:
            return False
    return True


def compute_overall_response_score(row: Dict[str, Any]) -> float:
    values: List[float] = []
    for group in GROUPS:
        values.extend(
            [
                float(row[f"{group}_mean_pairwise_distance"]),
                float(row[f"{group}_metric_vector_variance"]),
                float(row[f"{group}_unique_vector_ratio"]),
                float(row[f"{group}_loop_dispersion"]),
            ]
        )
    return sum(values) / len(values) if values else 0.0


def classify_rows(rows: List[Dict[str, Any]], zero_eps: float, transparent_max_unique_vectors: int, low_value_score_quantile: float, low_value_dispersion_quantile: float) -> Dict[str, Any]:
    for row in rows:
        row["overall_response_score"] = compute_overall_response_score(row)
        zero_by_group = [is_zero_response_for_group(row, group, zero_eps) for group in GROUPS]
        row["_zero_by_group"] = zero_by_group
        row["is_transparent_candidate"] = bool(
            row["constraint_count"] == 0
            and row["unique_metric_vectors"] <= transparent_max_unique_vectors
            and abs(float(row["deltaf_dispersion"])) <= zero_eps
            and all(zero_by_group)
        )

    low_value_pool = [
        row
        for row in rows
        if row["constraint_count"] == 0 and not row["is_transparent_candidate"]
    ]
    score_threshold = quantile([row["overall_response_score"] for row in low_value_pool], low_value_score_quantile)
    dispersion_threshold = quantile([float(row["deltaf_dispersion"]) for row in low_value_pool], low_value_dispersion_quantile)

    for row in rows:
        row["is_low_value_candidate"] = bool(
            row["constraint_count"] == 0
            and not row["is_transparent_candidate"]
            and float(row["overall_response_score"]) <= score_threshold
            and float(row["deltaf_dispersion"]) <= dispersion_threshold
        )

    return {
        "low_value_score_threshold": score_threshold,
        "low_value_dispersion_threshold": dispersion_threshold,
        "eligible_low_value_pool_size": len(low_value_pool),
    }


def build_summary(rows: Sequence[Dict[str, Any]], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    protocol_counts = Counter(str(row["protocol_name"]) for row in rows)
    field_kind_counts = Counter(str(row["field_kind"]) for row in rows)
    constraint_eq_0 = [row for row in rows if row["constraint_count"] == 0]
    constraint_gt_0 = [row for row in rows if row["constraint_count"] > 0]
    transparent_count = sum(1 for row in rows if row["is_transparent_candidate"])
    low_value_count = sum(1 for row in rows if row["is_low_value_candidate"])

    group_metric_distributions: Dict[str, Dict[str, Any]] = {}
    for group in GROUPS:
        group_metric_distributions[group] = {
            score_type: summarize_distribution(row[f"{group}_{score_type}"] for row in rows)
            for score_type in GROUP_SCORE_TYPES
        }

    return {
        "row_count": len(rows),
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "field_kind_counts": dict(sorted(field_kind_counts.items())),
        "constraint_split": {
            "constraint_count_eq_0": len(constraint_eq_0),
            "constraint_count_gt_0": len(constraint_gt_0),
        },
        "transparent_candidate_count": transparent_count,
        "low_value_candidate_count": low_value_count,
        "thresholds": thresholds,
        "group_metric_distributions": group_metric_distributions,
    }


def write_observation_tables(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    fieldnames = ["protocol_name", "sample_id", "field_id", "field_kind", "strategy_group", "score_type", "score"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for group in GROUPS:
                for score_type in GROUP_SCORE_TYPES:
                    writer.writerow(
                        {
                            "protocol_name": row["protocol_name"],
                            "sample_id": row["sample_id"],
                            "field_id": row["field_id"],
                            "field_kind": row["field_kind"],
                            "strategy_group": group,
                            "score_type": score_type,
                            "score": row[f"{group}_{score_type}"],
                        }
                    )


def write_field_labels(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRANSPARENT_LABEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in TRANSPARENT_LABEL_COLUMNS})


def ranking_sort_key(row: Dict[str, Any], group: str) -> tuple:
    return (
        -float(row[f"{group}_mean_baseline_distance"]),
        -float(row[f"{group}_mean_pairwise_distance"]),
        -float(row[f"{group}_metric_vector_variance"]),
        -float(row[f"{group}_unique_vector_ratio"]),
        str(row["protocol_name"]),
        str(row["sample_id"]),
        str(row["field_id"]),
    )


def write_group_rankings(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        path = output_dir / f"{group}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GROUP_RANKING_COLUMNS)
            writer.writeheader()
            ordered = sorted(rows, key=lambda row: ranking_sort_key(row, group))
            for idx, row in enumerate(ordered, start=1):
                writer.writerow(
                    {
                        "rank": idx,
                        "protocol_name": row["protocol_name"],
                        "sample_id": row["sample_id"],
                        "field_id": row["field_id"],
                        "field_kind": row["field_kind"],
                        "constraint_count": row["constraint_count"],
                        "unique_metric_vectors": row["unique_metric_vectors"],
                        "deltaf_dispersion": row["deltaf_dispersion"],
                        "mean_baseline_distance": row[f"{group}_mean_baseline_distance"],
                        "mean_pairwise_distance": row[f"{group}_mean_pairwise_distance"],
                        "metric_vector_variance": row[f"{group}_metric_vector_variance"],
                        "unique_vector_ratio": row[f"{group}_unique_vector_ratio"],
                    }
                )


def main() -> int:
    args = parse_args()
    resolve_output_paths(args)
    if not args.input_csv.exists():
        warn(f"Input CSV does not exist: {args.input_csv}")
        return 1

    rows = read_rows(args.input_csv)
    if not rows:
        warn(f"No rows found in input CSV: {args.input_csv}")
        return 1

    info(f"Loaded {len(rows)} field rows from {args.input_csv}", args.quiet)
    thresholds = classify_rows(
        rows,
        zero_eps=args.zero_eps,
        transparent_max_unique_vectors=args.transparent_max_unique_vectors,
        low_value_score_quantile=args.low_value_score_quantile,
        low_value_dispersion_quantile=args.low_value_dispersion_quantile,
    )

    summary = build_summary(rows, thresholds)
    ensure_parent(args.output_summary)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_observation_tables(rows, args.output_tables)
    write_field_labels(rows, args.output_labels)
    write_group_rankings(rows, args.output_group_dir)

    info(f"Wrote observation summary to {args.output_summary}", args.quiet)
    info(f"Wrote observation table to {args.output_tables}", args.quiet)
    info(f"Wrote field labels to {args.output_labels}", args.quiet)
    info(f"Wrote per-group rankings to {args.output_group_dir}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
