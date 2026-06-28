#!/usr/bin/env python3
"""Build Stage 3 field-level datasets from Stage 2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_field_training_samples import (
    V3_CONTEXT_COLS,
    V3_GROUPS,
    V3_GROUP_SUMMARY_COLS,
    collect_constraint_values,
    count_valid_mutation_runs,
    extract_baseline_run,
    find_mutation_entry,
    find_sample_dirs,
    flatten_v3_features,
    load_json,
    parse_csv_arg,
    safe_protocol_name,
    summarize_distribution,
)


DEFAULT_INPUT_ROOT = Path("/root/semvec/difftrace/out")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_dataset")
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "stage3_dataset_all_fields.csv"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "stage3_dataset_all_fields.jsonl"
DEFAULT_OUTPUT_SUMMARY = DEFAULT_OUTPUT_DIR / "stage3_dataset_summary.json"
DEFAULT_OUTPUT_SPEC = DEFAULT_OUTPUT_DIR / "stage3_feature_spec.json"

KEY_COLUMNS = ["protocol_name", "sample_id", "field_id"]
AUX_COLUMNS = [
    "mutation_count",
    "valid_mutations",
    "unique_metric_vectors",
    "deltaf_dispersion",
    "constraint_count",
    "field_kind",
]
MODEL_FEATURE_COLUMNS = V3_CONTEXT_COLS + [
    f"{group}_{feature}"
    for group in V3_GROUPS
    for feature in V3_GROUP_SUMMARY_COLS
]
OUTPUT_COLUMNS = KEY_COLUMNS + MODEL_FEATURE_COLUMNS + AUX_COLUMNS

VERBOSE = True


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    if VERBOSE:
        print(f"[INFO] {message}")


def normalize_field_id(field_report: Dict[str, Any]) -> str:
    unit = field_report.get("field_unit")
    if isinstance(unit, dict):
        kind = unit.get("kind")
        a = unit.get("a")
        b = unit.get("b")
        if kind == "bit":
            bits = unit.get("bits") or []
            if isinstance(bits, list) and bits:
                lo = int(min(bits))
                hi = int(max(bits))
                return f"bit:{int(a)}:{int(b)}:{lo}:{hi}"
        if kind == "byte":
            return f"b:{int(a)}:{int(b)}"

    kind = str(field_report.get("field_kind") or "byte")
    range_info = field_report.get("range") or {}
    a = range_info.get("a")
    b = range_info.get("b")
    if a is None or b is None:
        raw_id = field_report.get("field_id")
        return str(raw_id) if raw_id is not None else "field:unknown"
    if kind == "bit":
        raw_id = str(field_report.get("field_id") or "")
        if "[" in raw_id and ":" in raw_id and "]" in raw_id:
            bit_part = raw_id.split("[", 1)[1].split("]", 1)[0]
            hi_s, lo_s = bit_part.split(":", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            return f"bit:{int(a)}:{int(b)}:{lo}:{hi}"
        return f"bit:{int(a)}:{int(b)}:0:0"
    return f"b:{int(a)}:{int(b)}"


def extract_constraint_count(mutation_entry: Optional[Dict[str, Any]]) -> int:
    if not isinstance(mutation_entry, dict):
        return 0
    values = collect_constraint_values(mutation_entry)
    return len(set(values))


def extract_record(
    protocol_dir_name: str,
    sample_dir: Path,
    report_input: Optional[Dict[str, Any]],
    sample_json: Optional[Dict[str, Any]],
    mutations_json: Optional[Any],
    field_report: Dict[str, Any],
) -> Dict[str, Any]:
    protocol_name = safe_protocol_name(protocol_dir_name)
    sample_id = sample_dir.name

    field_report_local = dict(field_report)
    field_report_local["_sample_path"] = str(sample_dir)

    range_info = field_report.get("range") or {}
    range_start = range_info.get("a")
    range_end = range_info.get("b")

    diff = field_report.get("diff") or {}
    summary = diff.get("summary") or {}
    diagnostic = summary.get("diagnostic") or {}
    per_mutation = diff.get("per_mutation") or []
    runs = field_report.get("runs") or []
    baseline_run = extract_baseline_run(runs)

    valid_mutations = summary.get("valid_mutations")
    if valid_mutations is None:
        valid_mutations = len(per_mutation) if per_mutation else count_valid_mutation_runs(runs)

    mutation_count = len(per_mutation) if per_mutation else count_valid_mutation_runs(runs)

    mutation_entry = find_mutation_entry(
        mutations_json,
        range_start,
        range_end,
        field_id=str(field_report.get("field_id", "")),
    )

    baseline_payload_hex = None
    if isinstance(sample_json, dict):
        baseline_payload_hex = sample_json.get("payload_hex")
    if baseline_payload_hex is None and isinstance(report_input, dict):
        baseline_payload_hex = report_input.get("baseline_payload_hex")

    features = flatten_v3_features(
        per_mutation=per_mutation,
        mutation_entry=mutation_entry,
        field_report=field_report_local,
        baseline_run=baseline_run,
        baseline_payload_hex=baseline_payload_hex,
        range_start=range_start,
        range_end=range_end,
    )

    record: Dict[str, Any] = {
        "protocol_name": protocol_name,
        "sample_id": sample_id,
        "field_id": normalize_field_id(field_report),
        "mutation_count": mutation_count,
        "valid_mutations": valid_mutations,
        "unique_metric_vectors": diagnostic.get("unique_metric_vectors"),
        "deltaf_dispersion": diagnostic.get("deltaf_dispersion"),
        "constraint_count": extract_constraint_count(mutation_entry),
        "field_kind": field_report.get("field_kind", "byte"),
    }
    record.update(features)
    return record


def process_sample(sample_dir: Path, protocol_dir_name_override: Optional[str] = None) -> List[Dict[str, Any]]:
    protocol_dir_name = protocol_dir_name_override or sample_dir.parent.name
    report_json = load_json(sample_dir / "report.json")
    sample_json = load_json(sample_dir / "sample.json")
    mutations_json = load_json(sample_dir / "mutations.json")

    if not isinstance(report_json, dict):
        warn(f"Skipping sample without valid report.json: {sample_dir}")
        return []

    fields = report_json.get("fields")
    if not isinstance(fields, list):
        warn(f"Skipping sample with malformed fields list: {sample_dir}")
        return []

    report_input = report_json.get("input") if isinstance(report_json.get("input"), dict) else None
    records: List[Dict[str, Any]] = []
    for field_report in fields:
        if not isinstance(field_report, dict):
            continue
        records.append(
            extract_record(
                protocol_dir_name=protocol_dir_name,
                sample_dir=sample_dir,
                report_input=report_input,
                sample_json=sample_json if isinstance(sample_json, dict) else None,
                mutations_json=mutations_json,
                field_report=field_report,
            )
        )
    return records


def write_csv(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col) for col in OUTPUT_COLUMNS})


def write_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {col: record.get(col) for col in OUTPUT_COLUMNS}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def per_protocol_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sample_ids: Dict[str, set] = defaultdict(set)
    for record in records:
        protocol = str(record["protocol_name"])
        grouped[protocol].append(record)
        sample_ids[protocol].add(str(record["sample_id"]))

    summary: Dict[str, Any] = {}
    for protocol, rows in sorted(grouped.items()):
        summary[protocol] = {
            "field_count": len(rows),
            "sample_count": len(sample_ids[protocol]),
            "mutation_count_distribution": summarize_distribution(row.get("mutation_count") for row in rows),
            "valid_mutations_distribution": summarize_distribution(row.get("valid_mutations") for row in rows),
            "unique_metric_vectors_distribution": summarize_distribution(row.get("unique_metric_vectors") for row in rows),
            "deltaf_dispersion_distribution": summarize_distribution(row.get("deltaf_dispersion") for row in rows),
            "constraint_count_distribution": summarize_distribution(row.get("constraint_count") for row in rows),
        }
    return summary


def build_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    protocol_names = sorted({str(record["protocol_name"]) for record in records})
    sample_keys = {(str(record["protocol_name"]), str(record["sample_id"])) for record in records}
    field_kind_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        field_kind_counts[str(record.get("field_kind", "unknown"))] += 1

    return {
        "dataset_version": "all_fields",
        "row_count": len(records),
        "protocol_count": len(protocol_names),
        "sample_count": len(sample_keys),
        "field_kind_counts": dict(sorted(field_kind_counts.items())),
        "per_protocol": per_protocol_summary(records),
        "global_distributions": {
            "mutation_count": summarize_distribution(record.get("mutation_count") for record in records),
            "valid_mutations": summarize_distribution(record.get("valid_mutations") for record in records),
            "unique_metric_vectors": summarize_distribution(record.get("unique_metric_vectors") for record in records),
            "deltaf_dispersion": summarize_distribution(record.get("deltaf_dispersion") for record in records),
            "constraint_count": summarize_distribution(record.get("constraint_count") for record in records),
        },
    }


def build_feature_spec() -> Dict[str, Any]:
    return {
        "column_order_version": "stage3_v1",
        "dataset_version": "all_fields",
        "key_columns": KEY_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "auxiliary_columns": AUX_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "field_id_spec": {
            "byte": "b:<a>:<b>",
            "bit": "bit:<a>:<b>:<lo>:<hi>",
        },
        "notes": [
            "field_id is the stable machine key for Stage 3 and Stage 4 joins",
            "model_feature_columns are the default 28-D training inputs",
            "auxiliary_columns are retained for observation and later transparent-field analysis",
            "boundary_miss is intentionally excluded from the Stage 3 dataset",
        ],
    }


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_csv == DEFAULT_OUTPUT_CSV:
        args.output_csv = args.output_dir / DEFAULT_OUTPUT_CSV.name
    if args.output_jsonl == DEFAULT_OUTPUT_JSONL:
        args.output_jsonl = args.output_dir / DEFAULT_OUTPUT_JSONL.name
    if args.output_summary == DEFAULT_OUTPUT_SUMMARY:
        args.output_summary = args.output_dir / DEFAULT_OUTPUT_SUMMARY.name
    if args.output_feature_spec == DEFAULT_OUTPUT_SPEC:
        args.output_feature_spec = args.output_dir / DEFAULT_OUTPUT_SPEC.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 3 field-level dataset from Stage 2 outputs.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Stage 2 output root directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for Stage 3 dataset.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY, help="Output summary JSON path.")
    parser.add_argument("--output-feature-spec", type=Path, default=DEFAULT_OUTPUT_SPEC, help="Output feature spec JSON path.")
    parser.add_argument("--protocols", type=str, default=None, help="Comma-separated protocol filters.")
    parser.add_argument("--samples", type=str, default=None, help="Comma-separated sample directory names.")
    parser.add_argument("--sample-pattern", type=str, default="sample_*", help="Glob pattern for sample directories.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Stop after processing at most this many sample directories.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def main() -> int:
    global VERBOSE
    args = parse_args()
    VERBOSE = not args.quiet
    resolve_output_paths(args)

    protocol_filter = parse_csv_arg(args.protocols)
    sample_filter = parse_csv_arg(args.samples)
    sample_dirs = find_sample_dirs(
        args.input_root,
        protocol_filter=protocol_filter,
        sample_pattern=args.sample_pattern,
        sample_filter=sample_filter,
        limit_samples=args.limit_samples,
    )
    info(f"Discovered {len(sample_dirs)} sample directories under {args.input_root}")

    all_records: List[Dict[str, Any]] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        info(f"[{idx}/{len(sample_dirs)}] Processing {sample_dir}")
        all_records.extend(process_sample(sample_dir))

    if not all_records:
        warn("No Stage 3 field records were generated.")
        return 1

    summary = build_summary(all_records)
    feature_spec = build_feature_spec()

    write_csv(all_records, args.output_csv)
    write_jsonl(all_records, args.output_jsonl)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_feature_spec.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_feature_spec.write_text(json.dumps(feature_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info(f"Wrote {len(all_records)} rows to {args.output_csv}")
    info(f"Wrote JSONL rows to {args.output_jsonl}")
    info(f"Wrote dataset summary to {args.output_summary}")
    info(f"Wrote feature spec to {args.output_feature_spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
