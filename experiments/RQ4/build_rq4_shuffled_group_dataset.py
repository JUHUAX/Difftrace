#!/usr/bin/env python3
"""Build RQ4 shuffled-group Stage 3 datasets from per-mutation reports."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SEMVEC_ROOT = Path("/root/semvec")
DIFFTRACE_DIR = SEMVEC_ROOT / "difftrace"
STAGE3_DIR = DIFFTRACE_DIR / "stage3"
for path in (DIFFTRACE_DIR, STAGE3_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_field_training_samples import (  # noqa: E402
    V3_CONTEXT_COLS,
    V3_GROUPS,
    V3_GROUP_SUMMARY_COLS,
    collect_constraint_values,
    count_valid_mutation_runs,
    extract_baseline_run,
    find_mutation_entry,
    find_sample_dirs,
    load_json,
    metric_direction_vector,
    safe_protocol_name,
    summarize_distribution,
    summarize_v3_group,
)
from build_stage3_dataset import normalize_field_id  # noqa: E402
from filter_stage3_transparent_fields import build_label_index, filter_rows, load_csv, row_key  # noqa: E402


DEFAULT_INPUT_ROOT = DIFFTRACE_DIR / "out_rerun" / "outputs"
DEFAULT_LABELS_CSV = STAGE3_DIR / "out" / "stage3_observation" / "stage3_field_labels.csv"
DEFAULT_OUTDIR = SEMVEC_ROOT / "RQ4" / "out"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shuffled-group datasets for RQ4-A.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--protocol")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--sample-pattern", default="sample_*")
    parser.add_argument("--keep-all-fields", action="store_true", help="Do not filter transparent fields.")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def write_json(path: Path, data: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def group_vectors(per_mutation: Sequence[Dict[str, Any]]) -> dict[str, list[list[float]]]:
    grouped: dict[str, list[list[float]]] = {group: [] for group in V3_GROUPS}
    for item in per_mutation:
        if not isinstance(item, dict):
            continue
        group = str(item.get("strategy_group") or "")
        if group not in grouped:
            continue
        metrics = item.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        vector = metric_direction_vector(metrics)
        if vector is not None:
            grouped[group].append(vector)
    return grouped


def context_features(
    field_report: Dict[str, Any],
    mutation_entry: Optional[Dict[str, Any]],
    baseline_run: Optional[Dict[str, Any]],
    baseline_payload_hex: Optional[str],
    range_start: Optional[int],
    range_end: Optional[int],
) -> dict[str, float]:
    from build_field_training_samples import count_field_baseline_ops

    payload_len = len(baseline_payload_hex) // 2 if isinstance(baseline_payload_hex, str) else 0
    relative_start = (float(range_start) / max(payload_len - 1, 1)) if range_start is not None else 0.0
    parse_health = baseline_run.get("parse_health") if isinstance(baseline_run, dict) else {}
    if not isinstance(parse_health, dict):
        parse_health = {}
    total_instr = int(parse_health.get("instr_parsed") or 0)
    preprocess = baseline_run.get("preprocess") if isinstance(baseline_run, dict) else {}
    baseline_log_path = preprocess.get("path") if isinstance(preprocess, dict) else None
    if isinstance(baseline_log_path, str) and baseline_log_path and not Path(baseline_log_path).is_absolute():
        sample_path = Path(str(field_report.get("_sample_path", "")))
        if sample_path:
            baseline_log_path = str(sample_path / baseline_log_path)
    op_counts = count_field_baseline_ops(baseline_log_path, range_start, range_end)
    constraints = collect_constraint_values(mutation_entry)
    return {
        "relative_start": relative_start,
        "field_instr_ratio": float(op_counts["instr"] / max(total_instr, 1)),
        "compare_ratio": float(op_counts["compare"] / max(op_counts["useful"], 1)),
        "constraint_value_diversity": float(len(set(constraints)) / max(len(constraints), 1)),
    }


def shuffled_features(
    per_mutation: Sequence[Dict[str, Any]],
    seed: int,
    key: tuple[str, str, str],
) -> dict[str, float]:
    grouped = group_vectors(per_mutation)
    capacities = {group: len(grouped[group]) for group in V3_GROUPS}
    all_vectors = [vector for group in V3_GROUPS for vector in grouped[group]]
    rng_seed = f"{seed}|{key[0]}|{key[1]}|{key[2]}"
    rng = random.Random(rng_seed)
    rng.shuffle(all_vectors)

    cursor = 0
    features: dict[str, float] = {}
    for group in V3_GROUPS:
        count = capacities[group]
        vectors = all_vectors[cursor: cursor + count]
        cursor += count
        summary = summarize_v3_group(vectors)
        for name, value in summary.items():
            features[f"{group}_{name}"] = value
    return features


def extract_record(sample_dir: Path, protocol_name: str, field_report: Dict[str, Any], mutations_json: Any, sample_json: Any, report_input: Any, seed: int) -> Dict[str, Any]:
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

    field_id = normalize_field_id(field_report)
    key = (protocol_name, sample_dir.name, field_id)
    record: Dict[str, Any] = {
        "protocol_name": protocol_name,
        "sample_id": sample_dir.name,
        "field_id": field_id,
        "mutation_count": mutation_count,
        "valid_mutations": valid_mutations,
        "unique_metric_vectors": diagnostic.get("unique_metric_vectors"),
        "deltaf_dispersion": diagnostic.get("deltaf_dispersion"),
        "constraint_count": len(set(collect_constraint_values(mutation_entry))),
        "field_kind": field_report.get("field_kind", "byte"),
    }
    record.update(
        context_features(
            field_report=field_report_local,
            mutation_entry=mutation_entry,
            baseline_run=baseline_run,
            baseline_payload_hex=baseline_payload_hex,
            range_start=range_start,
            range_end=range_end,
        )
    )
    record.update(shuffled_features(per_mutation, seed=seed, key=key))
    return record


def process_sample(sample_dir: Path, seed: int) -> list[dict[str, Any]]:
    protocol_name = safe_protocol_name(sample_dir.parent.name)
    report_json = load_json(sample_dir / "report.json")
    sample_json = load_json(sample_dir / "sample.json")
    mutations_json = load_json(sample_dir / "mutations.json")
    if not isinstance(report_json, dict):
        return []
    fields = report_json.get("fields")
    if not isinstance(fields, list):
        return []
    report_input = report_json.get("input") if isinstance(report_json.get("input"), dict) else None
    rows = []
    for field_report in fields:
        if isinstance(field_report, dict):
            rows.append(extract_record(sample_dir, protocol_name, field_report, mutations_json, sample_json, report_input, seed))
    return rows


def filter_semantic_rows(rows: list[dict[str, Any]], labels_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not labels_csv.exists():
        return rows, {"labels_csv": str(labels_csv), "status": "missing_labels_kept_all"}
    label_rows, _ = load_csv(labels_csv)
    label_index = build_label_index(label_rows)
    semantic_rows, transparent_rows, low_value_rows = filter_rows(rows, label_index)
    return semantic_rows, {
        "labels_csv": str(labels_csv),
        "input_row_count": len(rows),
        "semantic_row_count": len(semantic_rows),
        "filtered_transparent_count": len(transparent_rows),
        "low_value_count": len(low_value_rows),
    }


def build_summary(rows: Sequence[dict[str, Any]], sample_count: int, seed: int, filter_summary: dict[str, Any]) -> dict[str, Any]:
    protocols = defaultdict(int)
    for row in rows:
        protocols[str(row["protocol_name"])] += 1
    return {
        "seed": seed,
        "row_count": len(rows),
        "sample_count": sample_count,
        "protocol_counts": dict(sorted(protocols.items())),
        "mutation_count_distribution": summarize_distribution(row.get("mutation_count") for row in rows),
        "valid_mutations_distribution": summarize_distribution(row.get("valid_mutations") for row in rows),
        "filter": filter_summary,
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    protocol_filter = {args.protocol} if args.protocol else None
    sample_dirs = find_sample_dirs(
        root=args.input_root,
        protocol_filter=protocol_filter,
        sample_pattern=args.sample_pattern,
        limit_samples=args.limit_samples or None,
    )
    if not sample_dirs:
        raise RuntimeError(f"no samples found under {args.input_root}")

    manifest = {
        "input_root": str(args.input_root),
        "labels_csv": str(args.labels_csv),
        "seeds": seeds,
        "sample_count": len(sample_dirs),
        "outputs": [],
    }
    for seed in seeds:
        print(f"[rq4-shuffle] seed={seed} samples={len(sample_dirs)}")
        rows: list[dict[str, Any]] = []
        for index, sample_dir in enumerate(sample_dirs, start=1):
            rows.extend(process_sample(sample_dir, seed=seed))
            if index % 25 == 0:
                print(f"[rq4-shuffle] seed={seed} processed_samples={index}/{len(sample_dirs)} rows={len(rows)}")

        if args.keep_all_fields:
            semantic_rows = rows
            filter_summary = {"status": "keep_all_fields", "input_row_count": len(rows), "semantic_row_count": len(rows)}
        else:
            semantic_rows, filter_summary = filter_semantic_rows(rows, args.labels_csv)

        out_dir = args.outdir / f"shuffled_seed_{seed}"
        dataset_csv = out_dir / "stage3_dataset_semantic_fields.csv"
        write_csv(semantic_rows, dataset_csv, OUTPUT_COLUMNS)
        write_json(out_dir / "stage3_dataset_summary.json", build_summary(semantic_rows, len(sample_dirs), seed, filter_summary))
        manifest["outputs"].append({"seed": seed, "dataset_csv": str(dataset_csv), "row_count": len(semantic_rows)})
        print(f"[rq4-shuffle] wrote {dataset_csv} rows={len(semantic_rows)}")

    write_json(args.outdir / "rq4_shuffled_group_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
