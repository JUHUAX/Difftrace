#!/usr/bin/env python3
"""Run Stage 3 dataset health checks on the semantic-fields training set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_filtered/stage3_dataset_semantic_fields.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_health")
DEFAULT_HEALTH_REPORT = DEFAULT_OUTPUT_DIR / "stage3_health_report.json"
DEFAULT_FEATURE_HEALTH = DEFAULT_OUTPUT_DIR / "stage3_feature_health.csv"
DEFAULT_PROTOCOL_DISTRIBUTION = DEFAULT_OUTPUT_DIR / "stage3_protocol_distribution.csv"
DEFAULT_MODEL_COLS_TXT = DEFAULT_OUTPUT_DIR / "stage3_model_feature_cols.txt"
DEFAULT_MODEL_COLS_JSON = DEFAULT_OUTPUT_DIR / "stage3_model_feature_cols.json"

MODEL_FEATURE_COLUMNS = [
    "relative_start",
    "field_instr_ratio",
    "compare_ratio",
    "constraint_value_diversity",
    "neighborhood_mean_baseline_distance",
    "neighborhood_mean_pairwise_distance",
    "neighborhood_max_pairwise_distance",
    "neighborhood_metric_vector_variance",
    "neighborhood_unique_vector_ratio",
    "neighborhood_loop_dispersion",
    "boundary_mean_baseline_distance",
    "boundary_mean_pairwise_distance",
    "boundary_max_pairwise_distance",
    "boundary_metric_vector_variance",
    "boundary_unique_vector_ratio",
    "boundary_loop_dispersion",
    "enum_mean_baseline_distance",
    "enum_mean_pairwise_distance",
    "enum_max_pairwise_distance",
    "enum_metric_vector_variance",
    "enum_unique_vector_ratio",
    "enum_loop_dispersion",
    "extreme_mean_baseline_distance",
    "extreme_mean_pairwise_distance",
    "extreme_max_pairwise_distance",
    "extreme_metric_vector_variance",
    "extreme_unique_vector_ratio",
    "extreme_loop_dispersion",
]


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage 3 semantic-fields dataset health.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input stage3_dataset_semantic_fields.csv path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Health-check output directory.")
    parser.add_argument("--output-health-report", type=Path, default=DEFAULT_HEALTH_REPORT, help="Health report JSON path.")
    parser.add_argument("--output-feature-health", type=Path, default=DEFAULT_FEATURE_HEALTH, help="Per-feature health CSV path.")
    parser.add_argument("--output-protocol-distribution", type=Path, default=DEFAULT_PROTOCOL_DISTRIBUTION, help="Protocol distribution CSV path.")
    parser.add_argument("--output-model-cols-txt", type=Path, default=DEFAULT_MODEL_COLS_TXT, help="Model feature columns TXT path.")
    parser.add_argument("--output-model-cols-json", type=Path, default=DEFAULT_MODEL_COLS_JSON, help="Model feature columns JSON path.")
    parser.add_argument("--correlation-threshold", type=float, default=0.98, help="Absolute correlation threshold for strong-correlation reporting.")
    parser.add_argument("--variance-eps", type=float, default=1e-12, help="Variance threshold used to identify near-constant columns.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_health_report == DEFAULT_HEALTH_REPORT:
        args.output_health_report = args.output_dir / DEFAULT_HEALTH_REPORT.name
    if args.output_feature_health == DEFAULT_FEATURE_HEALTH:
        args.output_feature_health = args.output_dir / DEFAULT_FEATURE_HEALTH.name
    if args.output_protocol_distribution == DEFAULT_PROTOCOL_DISTRIBUTION:
        args.output_protocol_distribution = args.output_dir / DEFAULT_PROTOCOL_DISTRIBUTION.name
    if args.output_model_cols_txt == DEFAULT_MODEL_COLS_TXT:
        args.output_model_cols_txt = args.output_dir / DEFAULT_MODEL_COLS_TXT.name
    if args.output_model_cols_json == DEFAULT_MODEL_COLS_JSON:
        args.output_model_cols_json = args.output_dir / DEFAULT_MODEL_COLS_JSON.name


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


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            converted = dict(row)
            for col in MODEL_FEATURE_COLUMNS:
                converted[col] = parse_float(row.get(col))
            converted["constraint_count"] = parse_int(row.get("constraint_count"))
            converted["valid_mutations"] = parse_int(row.get("valid_mutations"))
            converted["mutation_count"] = parse_int(row.get("mutation_count"))
            converted["unique_metric_vectors"] = parse_int(row.get("unique_metric_vectors"))
            converted["deltaf_dispersion"] = parse_float(row.get("deltaf_dispersion"))
            rows.append(converted)
    return rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def summarize_distribution(values: Iterable[float]) -> Dict[str, Any]:
    seq = [float(v) for v in values]
    if not seq:
        return {"count": 0}
    ordered = sorted(seq)
    def q(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac
    return {
        "count": len(seq),
        "min": ordered[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "max": ordered[-1],
        "mean": mean(seq),
    }


def build_feature_health(rows: Sequence[Dict[str, Any]], variance_eps: float) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for col in MODEL_FEATURE_COLUMNS:
        values = [float(row[col]) for row in rows]
        var = variance(values)
        nonzero_count = sum(1 for v in values if abs(v) > 0.0)
        output.append(
            {
                "feature_name": col,
                "count": len(values),
                "missing_count": 0,
                "mean": mean(values),
                "variance": var,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "nonzero_ratio": (nonzero_count / len(values)) if values else 0.0,
                "is_near_constant": var <= variance_eps,
            }
        )
    return output


def build_protocol_distribution(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    protocol_counts = Counter(str(row["protocol_name"]) for row in rows)
    return [{"protocol_name": protocol, "field_count": count} for protocol, count in sorted(protocol_counts.items())]


def build_correlation_report(rows: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    columns = {col: [float(row[col]) for row in rows] for col in MODEL_FEATURE_COLUMNS}
    strong_pairs: List[Dict[str, Any]] = []
    for i, left in enumerate(MODEL_FEATURE_COLUMNS):
        for right in MODEL_FEATURE_COLUMNS[i + 1 :]:
            corr = correlation(columns[left], columns[right])
            if abs(corr) >= threshold:
                strong_pairs.append(
                    {
                        "left_feature": left,
                        "right_feature": right,
                        "correlation": corr,
                    }
                )
    strong_pairs.sort(key=lambda item: (-abs(float(item["correlation"])), item["left_feature"], item["right_feature"]))
    return strong_pairs


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

    rows = load_rows(args.input_csv)
    if not rows:
        warn(f"No rows found in input CSV: {args.input_csv}")
        return 1

    info(f"Loaded {len(rows)} semantic field rows from {args.input_csv}", args.quiet)

    feature_health = build_feature_health(rows, variance_eps=args.variance_eps)
    protocol_distribution = build_protocol_distribution(rows)
    strong_correlations = build_correlation_report(rows, threshold=args.correlation_threshold)

    health_report = {
        "row_count": len(rows),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "near_constant_features": [row["feature_name"] for row in feature_health if row["is_near_constant"]],
        "strong_correlation_pairs": strong_correlations,
        "protocol_distribution": protocol_distribution,
        "diagnostic_distributions": {
            "mutation_count": summarize_distribution(row["mutation_count"] for row in rows),
            "valid_mutations": summarize_distribution(row["valid_mutations"] for row in rows),
            "unique_metric_vectors": summarize_distribution(row["unique_metric_vectors"] for row in rows),
            "deltaf_dispersion": summarize_distribution(row["deltaf_dispersion"] for row in rows),
            "constraint_count": summarize_distribution(row["constraint_count"] for row in rows),
        },
    }

    write_csv(
        feature_health,
        [
            "feature_name",
            "count",
            "missing_count",
            "mean",
            "variance",
            "min",
            "max",
            "nonzero_ratio",
            "is_near_constant",
        ],
        args.output_feature_health,
    )
    write_csv(protocol_distribution, ["protocol_name", "field_count"], args.output_protocol_distribution)

    ensure_parent(args.output_health_report)
    ensure_parent(args.output_model_cols_txt)
    ensure_parent(args.output_model_cols_json)
    args.output_health_report.write_text(json.dumps(health_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_model_cols_txt.write_text("\n".join(MODEL_FEATURE_COLUMNS) + "\n", encoding="utf-8")
    args.output_model_cols_json.write_text(json.dumps(MODEL_FEATURE_COLUMNS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info(f"Wrote health report to {args.output_health_report}", args.quiet)
    info(f"Wrote feature health table to {args.output_feature_health}", args.quiet)
    info(f"Wrote protocol distribution to {args.output_protocol_distribution}", args.quiet)
    info(f"Wrote model feature columns to {args.output_model_cols_txt} and {args.output_model_cols_json}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
