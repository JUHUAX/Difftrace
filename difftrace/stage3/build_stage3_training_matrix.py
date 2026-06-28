#!/usr/bin/env python3
"""Build the Stage 3 min-max scaled training matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from check_stage3_dataset import MODEL_FEATURE_COLUMNS


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_filtered/stage3_dataset_semantic_fields.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_training_matrix")
DEFAULT_OUTPUT_MATRIX_CSV = DEFAULT_OUTPUT_DIR / "stage3_training_matrix.csv"
DEFAULT_OUTPUT_TRAIN_MATRIX_CSV = DEFAULT_OUTPUT_DIR / "stage3_training_matrix_train.csv"
DEFAULT_OUTPUT_EVAL_MATRIX_CSV = DEFAULT_OUTPUT_DIR / "stage3_training_matrix_eval.csv"
DEFAULT_OUTPUT_SCALER_JSON = DEFAULT_OUTPUT_DIR / "stage3_training_scaler.json"
DEFAULT_OUTPUT_FEATURE_COLS_TXT = DEFAULT_OUTPUT_DIR / "stage3_training_feature_cols.txt"
DEFAULT_OUTPUT_FEATURE_COLS_JSON = DEFAULT_OUTPUT_DIR / "stage3_training_feature_cols.json"
DEFAULT_OUTPUT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "stage3_training_matrix_summary.json"

KEY_COLUMNS = ["protocol_name", "sample_id", "field_id"]


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build min-max scaled Stage 3 training matrix.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input semantic_fields CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--output-matrix-csv", type=Path, default=DEFAULT_OUTPUT_MATRIX_CSV, help="Scaled training matrix CSV.")
    parser.add_argument("--output-train-matrix-csv", type=Path, default=DEFAULT_OUTPUT_TRAIN_MATRIX_CSV, help="Scaled train-only matrix CSV.")
    parser.add_argument("--output-eval-matrix-csv", type=Path, default=DEFAULT_OUTPUT_EVAL_MATRIX_CSV, help="Scaled eval-only matrix CSV.")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Optional packet-level train/eval split manifest.")
    parser.add_argument("--output-scaler-json", type=Path, default=DEFAULT_OUTPUT_SCALER_JSON, help="Per-feature min/max scaler JSON.")
    parser.add_argument("--output-feature-cols-txt", type=Path, default=DEFAULT_OUTPUT_FEATURE_COLS_TXT, help="Model feature columns TXT.")
    parser.add_argument("--output-feature-cols-json", type=Path, default=DEFAULT_OUTPUT_FEATURE_COLS_JSON, help="Model feature columns JSON.")
    parser.add_argument("--output-summary-json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON, help="Training matrix summary JSON.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_matrix_csv == DEFAULT_OUTPUT_MATRIX_CSV:
        args.output_matrix_csv = args.output_dir / DEFAULT_OUTPUT_MATRIX_CSV.name
    if args.output_scaler_json == DEFAULT_OUTPUT_SCALER_JSON:
        args.output_scaler_json = args.output_dir / DEFAULT_OUTPUT_SCALER_JSON.name
    if args.output_train_matrix_csv == DEFAULT_OUTPUT_TRAIN_MATRIX_CSV:
        args.output_train_matrix_csv = args.output_dir / DEFAULT_OUTPUT_TRAIN_MATRIX_CSV.name
    if args.output_eval_matrix_csv == DEFAULT_OUTPUT_EVAL_MATRIX_CSV:
        args.output_eval_matrix_csv = args.output_dir / DEFAULT_OUTPUT_EVAL_MATRIX_CSV.name
    if args.output_feature_cols_txt == DEFAULT_OUTPUT_FEATURE_COLS_TXT:
        args.output_feature_cols_txt = args.output_dir / DEFAULT_OUTPUT_FEATURE_COLS_TXT.name
    if args.output_feature_cols_json == DEFAULT_OUTPUT_FEATURE_COLS_JSON:
        args.output_feature_cols_json = args.output_dir / DEFAULT_OUTPUT_FEATURE_COLS_JSON.name
    if args.output_summary_json == DEFAULT_OUTPUT_SUMMARY_JSON:
        args.output_summary_json = args.output_dir / DEFAULT_OUTPUT_SUMMARY_JSON.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            converted = dict(row)
            for col in MODEL_FEATURE_COLUMNS:
                converted[col] = parse_float(row.get(col))
            rows.append(converted)
    return rows


def fit_minmax(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    scaler: Dict[str, Dict[str, float]] = {}
    for col in MODEL_FEATURE_COLUMNS:
        values = [float(row[col]) for row in rows]
        lo = min(values)
        hi = max(values)
        scaler[col] = {
            "min": lo,
            "max": hi,
            "range": hi - lo,
            "is_constant": hi == lo,
        }
    return scaler


def transform_rows(rows: Sequence[Dict[str, Any]], scaler: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        scaled = {key: row.get(key) for key in KEY_COLUMNS}
        for col in MODEL_FEATURE_COLUMNS:
            spec = scaler[col]
            value = float(row[col])
            if spec["range"] <= 0.0:
                scaled[col] = 0.0
            else:
                scaled[col] = (value - spec["min"]) / spec["range"]
        out.append(scaled)
    return out


def load_split_index(path: Path) -> Dict[tuple[str, str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocols = payload.get("protocols")
    if not isinstance(protocols, dict):
        raise ValueError(f"{path} must contain a protocols object")
    index: Dict[tuple[str, str], str] = {}
    for protocol, spec in protocols.items():
        if not isinstance(spec, dict):
            continue
        for partition in ("train", "eval"):
            for sample_id in spec.get(partition, []) or []:
                key = (str(protocol), str(sample_id))
                if key in index:
                    raise ValueError(f"duplicate split assignment for {key}: {path}")
                index[key] = partition
    return index


def write_matrix(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    fieldnames = KEY_COLUMNS + MODEL_FEATURE_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in fieldnames})


def build_summary(rows: Sequence[Dict[str, Any]], scaler: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    protocol_counts: Dict[str, int] = {}
    for row in rows:
        protocol = str(row["protocol_name"])
        protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
    return {
        "row_count": len(rows),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "key_columns": KEY_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "constant_feature_count": sum(1 for spec in scaler.values() if spec["is_constant"]),
        "protocol_counts": dict(sorted(protocol_counts.items())),
    }


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

    train_rows = rows
    eval_rows: List[Dict[str, Any]] = []
    if args.split_manifest is not None:
        if not args.split_manifest.exists():
            warn(f"Split manifest does not exist: {args.split_manifest}")
            return 1
        try:
            split_index = load_split_index(args.split_manifest)
        except (ValueError, json.JSONDecodeError) as exc:
            warn(str(exc))
            return 1
        missing = [
            (str(row["protocol_name"]), str(row["sample_id"]))
            for row in rows
            if (str(row["protocol_name"]), str(row["sample_id"])) not in split_index
        ]
        if missing:
            warn(f"Split manifest does not assign {len(missing)} rows; first missing key={missing[0]}")
            return 1
        train_rows = [
            row for row in rows
            if split_index[(str(row["protocol_name"]), str(row["sample_id"]))] == "train"
        ]
        eval_rows = [
            row for row in rows
            if split_index[(str(row["protocol_name"]), str(row["sample_id"]))] == "eval"
        ]
        if not train_rows or not eval_rows:
            warn(f"Split manifest produced empty partition: train={len(train_rows)} eval={len(eval_rows)}")
            return 1

    scaler = fit_minmax(train_rows)
    scaled_rows = transform_rows(rows, scaler)
    scaled_train_rows = transform_rows(train_rows, scaler)
    scaled_eval_rows = transform_rows(eval_rows, scaler)

    write_matrix(scaled_rows, args.output_matrix_csv)
    if args.split_manifest is not None:
        write_matrix(scaled_train_rows, args.output_train_matrix_csv)
        write_matrix(scaled_eval_rows, args.output_eval_matrix_csv)
    ensure_parent(args.output_scaler_json)
    args.output_scaler_json.write_text(json.dumps(scaler, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ensure_parent(args.output_feature_cols_txt)
    args.output_feature_cols_txt.write_text("\n".join(MODEL_FEATURE_COLUMNS) + "\n", encoding="utf-8")
    ensure_parent(args.output_feature_cols_json)
    args.output_feature_cols_json.write_text(json.dumps(MODEL_FEATURE_COLUMNS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = build_summary(scaled_rows, scaler)
    summary["split_manifest"] = str(args.split_manifest) if args.split_manifest else None
    summary["scaler_fit_row_count"] = len(scaled_train_rows)
    summary["eval_row_count"] = len(scaled_eval_rows)
    ensure_parent(args.output_summary_json)
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info(f"Wrote scaled training matrix to {args.output_matrix_csv}", args.quiet)
    if args.split_manifest is not None:
        info(f"Wrote train-only matrix to {args.output_train_matrix_csv}", args.quiet)
        info(f"Wrote eval-only matrix to {args.output_eval_matrix_csv}", args.quiet)
    info(f"Wrote scaler to {args.output_scaler_json}", args.quiet)
    info(f"Wrote training matrix summary to {args.output_summary_json}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
