#!/usr/bin/env python3
"""Build fixed-length field-level training samples from difftrace outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


V3_GROUPS = ["neighborhood", "boundary", "enum", "extreme"]
V3_CONTEXT_COLS = [
    "relative_start",
    "field_instr_ratio",
    "compare_ratio",
    "constraint_value_diversity",
]
V3_GROUP_SUMMARY_COLS = [
    "mean_baseline_distance",
    "mean_pairwise_distance",
    "max_pairwise_distance",
    "metric_vector_variance",
    "unique_vector_ratio",
    "loop_dispersion",
]

DEFAULT_INPUT_ROOT = Path("/root/semvec/difftrace/out")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/field_training_samples")
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "field_training_samples.csv"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "field_training_samples.jsonl"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "field_training_samples.json"
DEFAULT_OUTPUT_SUMMARY = DEFAULT_OUTPUT_DIR / "field_training_summary.json"
DEFAULT_OUTPUT_COLUMNS_MD = DEFAULT_OUTPUT_DIR / "field_training_columns.md"
VERBOSE = True

def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    if VERBOSE:
        print(f"[INFO] {message}")


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        warn(f"Missing file: {path}")
    except json.JSONDecodeError as exc:
        warn(f"Failed to parse JSON {path}: {exc}")
    except OSError as exc:
        warn(f"Failed to read {path}: {exc}")
    return None


def parse_csv_arg(value: Optional[str]) -> Optional[set]:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def find_sample_dirs(
    root: Path,
    protocol_filter: Optional[set] = None,
    sample_pattern: str = "sample_*",
    sample_filter: Optional[set] = None,
    limit_samples: Optional[int] = None,
) -> List[Path]:
    sample_dirs: List[Path] = []
    if not root.exists():
        warn(f"Input root does not exist: {root}")
        return sample_dirs

    for protocol_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        protocol_dir_name = protocol_dir.name
        protocol_name = safe_protocol_name(protocol_dir_name)
        if protocol_filter and protocol_dir_name not in protocol_filter and protocol_name not in protocol_filter:
            continue
        for sample_dir in sorted(protocol_dir.glob(sample_pattern)):
            if not sample_dir.is_dir():
                continue
            if sample_filter and sample_dir.name not in sample_filter:
                continue
            if (sample_dir / "report.json").exists():
                sample_dirs.append(sample_dir)
                if limit_samples is not None and len(sample_dirs) >= limit_samples:
                    return sample_dirs
    return sample_dirs


def safe_protocol_name(protocol_dir_name: str) -> str:
    return protocol_dir_name[4:] if protocol_dir_name.startswith("out_") else protocol_dir_name


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def count_valid_mutation_runs(runs: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for run in runs:
        if run.get("kind") != "mutation":
            continue
        status = run.get("status")
        if status in (None, "ok"):
            count += 1
    return count


def extract_baseline_run(runs: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for run in runs:
        if run.get("kind") == "baseline":
            return run
    return None


def find_mutation_entry(
    mutations_json: Any,
    range_start: Optional[int],
    range_end: Optional[int],
    field_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(mutations_json, list):
        return None
    if field_id:
        for entry in mutations_json:
            field = entry.get("field") or {}
            if str(field.get("repr", "")) == str(field_id):
                return entry
    for entry in mutations_json:
        field = entry.get("field") or {}
        if field.get("a") == range_start and field.get("b") == range_end:
            return entry
    return None


def metric_direction_vector(metrics: Dict[str, Any]) -> Optional[List[float]]:
    values = {
        "branch_sites_delta": 1.0 - float(to_float(metrics.get("branch_sites_jaccard")) or 0.0),
        "bb_set_delta": 1.0 - float(to_float(metrics.get("bb_set_jaccard")) or 0.0),
        "cmp_site_delta": 1.0 - float(to_float(metrics.get("cmp_site_set_jaccard")) or 0.0),
        "lcp_delta": 1.0 - float(to_float(metrics.get("lcp_ratio")) or 0.0),
        "instr_delta": float(to_float(metrics.get("instr_delta_ratio")) or 0.0),
        "bb_multiset_delta": float(to_float(metrics.get("bb_multiset_l1_ratio")) or 0.0),
        "cmp_delta": float(to_float(metrics.get("cmp_delta_ratio")) or 0.0),
        "branch_flip_delta": float(to_float(metrics.get("branch_flip_ratio")) or 0.0),
        "loop_delta": float(to_float(metrics.get("loop_delta_ratio")) or 0.0),
    }
    return list(values.values())


def mean_abs_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    return float(statistics.fmean(abs(float(left[i]) - float(right[i])) for i in range(n)))


def summarize_v3_group(vectors: Sequence[List[float]]) -> Dict[str, float]:
    if not vectors:
        return {name: 0.0 for name in V3_GROUP_SUMMARY_COLS}

    baseline_distances = [float(statistics.fmean(vec)) if vec else 0.0 for vec in vectors]
    pairwise = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pairwise.append(mean_abs_distance(vectors[i], vectors[j]))
    metric_variances = []
    width = len(vectors[0]) if vectors and vectors[0] else 0
    for idx in range(width):
        vals = [float(vec[idx]) for vec in vectors if len(vec) > idx]
        metric_variances.append(float(statistics.pvariance(vals)) if len(vals) > 1 else 0.0)
    rounded = {tuple(round(float(v), 6) for v in vec) for vec in vectors}
    loop_values = [float(vec[8]) for vec in vectors if len(vec) > 8]

    return {
        "mean_baseline_distance": float(statistics.fmean(baseline_distances)) if baseline_distances else 0.0,
        "mean_pairwise_distance": float(statistics.fmean(pairwise)) if pairwise else 0.0,
        "max_pairwise_distance": float(max(pairwise)) if pairwise else 0.0,
        "metric_vector_variance": float(statistics.fmean(metric_variances)) if metric_variances else 0.0,
        "unique_vector_ratio": float(len(rounded) / max(len(vectors), 1)),
        "loop_dispersion": float(statistics.pstdev(loop_values)) if len(loop_values) > 1 else 0.0,
    }


def parse_field_indices(raw: str) -> set:
    return {int(token) for token in re.findall(r"\d+", raw or "")}


def count_field_baseline_ops(log_path: Optional[str], a: Optional[int], b: Optional[int]) -> Dict[str, int]:
    if not log_path or a is None or b is None:
        return {"instr": 0, "compare": 0, "useful": 0}
    path = Path(log_path)
    if not path.exists():
        return {"instr": 0, "compare": 0, "useful": 0}
    target = set(range(int(a), int(b) + 1))
    instr = 0
    compare = 0
    useful = 0
    useful_mnemonics = {
        "cmp", "test", "add", "sub", "imul", "mul", "idiv", "div",
        "and", "or", "xor", "shl", "shr", "sar", "sal", "rol", "ror",
    }
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            payload = line.rstrip("\n")
            if "\tInstruction\t" in payload:
                payload = payload.split("\tInstruction\t", 1)[1]
                payload = "Instruction\t" + payload
            if payload.startswith("LOOP\t"):
                payload = payload[len("LOOP\t"):]
            if not payload.startswith("Instruction\t"):
                continue
            parts = payload.split("\t")
            if len(parts) < 3:
                continue
            idx_set = parse_field_indices(parts[2])
            if not (idx_set & target):
                continue
            instr += 1
            disasm = parts[1].split(":", 1)[1].strip() if ":" in parts[1] else parts[1].strip()
            mnemonic = disasm.split(" ", 1)[0].lower() if disasm else ""
            if mnemonic in {"cmp", "test"}:
                compare += 1
            if mnemonic in useful_mnemonics:
                useful += 1
    return {"instr": instr, "compare": compare, "useful": useful}


def collect_constraint_values(mutation_entry: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(mutation_entry, dict):
        return []
    values = []
    for item in mutation_entry.get("mutations", []) or []:
        if not isinstance(item, dict):
            continue
        group = item.get("strategy_group")
        strategy = str(item.get("strategy", ""))
        if group == "enum" or strategy.startswith("constraint"):
            value = item.get("final_value_hex") or item.get("requested_value_hex")
            if isinstance(value, str) and value:
                values.append(value.lower())
    return values


def flatten_v3_features(
    per_mutation: Sequence[Dict[str, Any]],
    mutation_entry: Optional[Dict[str, Any]],
    field_report: Dict[str, Any],
    baseline_run: Optional[Dict[str, Any]],
    baseline_payload_hex: Optional[str],
    range_start: Optional[int],
    range_end: Optional[int],
) -> Dict[str, float]:
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

    features: Dict[str, float] = {
        "relative_start": relative_start,
        "field_instr_ratio": float(op_counts["instr"] / max(total_instr, 1)),
        "compare_ratio": float(op_counts["compare"] / max(op_counts["useful"], 1)),
        "constraint_value_diversity": float(len(set(constraints)) / max(len(constraints), 1)),
    }

    grouped_vectors: Dict[str, List[List[float]]] = {group: [] for group in V3_GROUPS}
    for item in per_mutation:
        if not isinstance(item, dict):
            continue
        group = str(item.get("strategy_group") or "")
        if group not in grouped_vectors:
            continue
        metrics = item.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        vector = metric_direction_vector(metrics)
        if vector is not None:
            grouped_vectors[group].append(vector)

    for group in V3_GROUPS:
        summary = summarize_v3_group(grouped_vectors[group])
        for key, value in summary.items():
            features[f"{group}_{key}"] = value
    return features


def collect_extra_diagnostics(
    field_report: Dict[str, Any],
    baseline_run: Optional[Dict[str, Any]],
    runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    parse_health = baseline_run.get("parse_health") if baseline_run else {}
    if not isinstance(parse_health, dict):
        parse_health = {}

    preprocess = baseline_run.get("preprocess") if baseline_run else {}
    if not isinstance(preprocess, dict):
        preprocess = {}

    run_status_counter = Counter(run.get("status", "unknown") for run in runs if run.get("kind") == "mutation")

    return {
        "baseline_status": baseline_run.get("status") if baseline_run else None,
        "baseline_taint_found": parse_health.get("taint_found"),
        "baseline_bb_parsed": parse_health.get("bb_parsed"),
        "baseline_branch_parsed": parse_health.get("branch_parsed"),
        "baseline_instr_parsed": parse_health.get("instr_parsed"),
        "baseline_bad_lines_dropped": parse_health.get("bad_lines_dropped"),
        "baseline_preprocess_line_count": preprocess.get("line_count"),
        "baseline_taint_thread_id": preprocess.get("taint_thread_id"),
        "mutation_status_ok": run_status_counter.get("ok", 0),
        "mutation_status_non_ok": sum(v for k, v in run_status_counter.items() if k != "ok"),
    }


def extract_field_record(
    protocol_dir_name: str,
    sample_dir: Path,
    report_input: Optional[Dict[str, Any]],
    sample_json: Optional[Dict[str, Any]],
    fields_json: Optional[Dict[str, Any]],
    mutations_json: Optional[Any],
    field_report: Dict[str, Any],
    field_index: int,
) -> Dict[str, Any]:
    protocol_name = safe_protocol_name(protocol_dir_name)
    protocol_id = protocol_dir_name
    sample_id = sample_dir.name

    field_id = field_report.get("field_id") or f"field_{field_index:03d}"
    range_info = field_report.get("range") or {}
    range_start = range_info.get("a")
    range_end = range_info.get("b")
    nbytes = field_report.get("nbytes")
    baseline_field_hex = field_report.get("baseline_field_hex")

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
    if mutation_count == 0:
        warn(f"{sample_dir} field={field_id} has zero mutation observations")

    mutation_entry = find_mutation_entry(mutations_json, range_start, range_end, field_id=str(field_id))
    mutation_candidates = mutation_entry.get("mutations") if mutation_entry else None

    payload_len = None
    field_partition_source = None
    baseline_payload_hex = None
    if isinstance(sample_json, dict):
        baseline_payload_hex = sample_json.get("payload_hex")
    if isinstance(report_input, dict):
        payload_len = report_input.get("payload_len")
        field_partition_source = report_input.get("field_partition_source")

    fields_partition_count = None
    if isinstance(fields_json, dict):
        field_items = fields_json.get("fields")
        if isinstance(field_items, list):
            fields_partition_count = len(field_items)
    if fields_partition_count is None and isinstance(report_input, dict):
        partition = report_input.get("fields_partition")
        if isinstance(partition, list):
            fields_partition_count = len(partition)

    record: Dict[str, Any] = {
        "protocol_id": protocol_id,
        "protocol_name": protocol_name,
        "sample_id": sample_id,
        "sample_path": str(sample_dir),
        "field_id": field_id,
        "field_index": field_index,
        "field_range_start": range_start,
        "field_range_end": range_end,
        "field_range_repr": f"{range_start}_{range_end}" if range_start is not None and range_end is not None else None,
        "field_kind": field_report.get("field_kind", "byte"),
        "field_unit_repr": field_report.get("field_id"),
        "nbytes": nbytes,
        "baseline_field_hex": baseline_field_hex,
        "valid_mutations": valid_mutations,
        "mutation_count": mutation_count,
        "deltaf_dispersion": diagnostic.get("deltaf_dispersion"),
        "unique_metric_vectors": diagnostic.get("unique_metric_vectors"),
        "mutations_json_count": len(mutation_candidates) if isinstance(mutation_candidates, list) else None,
        "baseline_payload_hex": baseline_payload_hex,
        "sample_proto": sample_json.get("proto") if isinstance(sample_json, dict) else None,
        "sample_mode": sample_json.get("mode") if isinstance(sample_json, dict) else None,
        "sample_seed": sample_json.get("seed") if isinstance(sample_json, dict) else None,
        "sample_index": sample_json.get("index") if isinstance(sample_json, dict) else None,
        "fields_in_sample": fields_partition_count,
        "field_partition_source": field_partition_source,
        "payload_len_bytes": len(baseline_payload_hex) // 2 if isinstance(baseline_payload_hex, str) else payload_len,
    }
    record.update(collect_extra_diagnostics(field_report, baseline_run, runs))
    record.update(flatten_v3_features(
        per_mutation=per_mutation,
        mutation_entry=mutation_entry,
        field_report=field_report,
        baseline_run=baseline_run,
        baseline_payload_hex=baseline_payload_hex,
        range_start=range_start,
        range_end=range_end,
    ))

    return record


def process_sample(
    sample_dir: Path,
    protocol_dir_name_override: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    protocol_dir_name = protocol_dir_name_override or sample_dir.parent.name
    report_json = load_json(sample_dir / "report.json")
    sample_json = load_json(sample_dir / "sample.json")
    fields_json = load_json(sample_dir / "fields.json")
    mutations_json = load_json(sample_dir / "mutations.json")

    if not isinstance(report_json, dict):
        warn(f"Skipping sample without valid report.json: {sample_dir}")
        return [], {"protocol_dir_name": protocol_dir_name, "sample_id": sample_dir.name, "field_count": 0}

    fields = report_json.get("fields")
    if not isinstance(fields, list):
        warn(f"Skipping sample with malformed fields list: {sample_dir}")
        return [], {"protocol_dir_name": protocol_dir_name, "sample_id": sample_dir.name, "field_count": 0}

    report_input = report_json.get("input") if isinstance(report_json.get("input"), dict) else None
    records: List[Dict[str, Any]] = []
    for idx, field_report in enumerate(fields):
        if not isinstance(field_report, dict):
            warn(f"Skipping malformed field entry at {sample_dir} index={idx}")
            continue
        records.append(
            extract_field_record(
                protocol_dir_name=protocol_dir_name,
                sample_dir=sample_dir,
                report_input=report_input,
                sample_json=sample_json if isinstance(sample_json, dict) else None,
                fields_json=fields_json if isinstance(fields_json, dict) else None,
                mutations_json=mutations_json,
                field_report=field_report,
                field_index=idx,
            )
        )

    return records, {
        "protocol_dir_name": protocol_dir_name,
        "sample_id": sample_dir.name,
        "field_count": len(records),
    }


def build_field_training_artifacts(
    sample_dirs: Sequence[Path],
    output_dir: Path,
    protocol_dir_name_override: Optional[str] = None,
    json_indent: Optional[int] = 2,
) -> Dict[str, Any]:
    """Build V3 field summary vectors for explicitly supplied sample dirs."""
    all_records: List[Dict[str, Any]] = []
    sample_summaries: List[Dict[str, Any]] = []

    for sample_dir in sample_dirs:
        records, sample_summary = process_sample(
            Path(sample_dir),
            protocol_dir_name_override=protocol_dir_name_override,
        )
        all_records.extend(records)
        sample_summaries.append(sample_summary)

    if not all_records:
        raise RuntimeError("no field-level records were generated")

    summary = build_summary(all_records, sample_summaries)
    output_dir = Path(output_dir)
    output_csv = output_dir / DEFAULT_OUTPUT_CSV.name
    output_jsonl = output_dir / DEFAULT_OUTPUT_JSONL.name
    output_summary = output_dir / DEFAULT_OUTPUT_SUMMARY.name
    output_columns_md = output_dir / DEFAULT_OUTPUT_COLUMNS_MD.name

    write_csv(all_records, output_csv)
    write_jsonl(all_records, output_jsonl)
    write_json(summary, output_summary, json_indent)
    write_column_docs(all_records, summary, output_columns_md)

    return {
        "record_count": len(all_records),
        "sample_count": len(sample_summaries),
        "output_dir": str(output_dir),
        "csv": str(output_csv),
        "jsonl": str(output_jsonl),
        "summary": str(output_summary),
        "columns_md": str(output_columns_md),
    }


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    fraction = pos - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def summarize_distribution(values: Iterable[Any]) -> Dict[str, Any]:
    cleaned = [to_float(value) for value in values]
    numeric = [value for value in cleaned if value is not None]
    if not numeric:
        return {"count": 0}
    return {
        "count": len(numeric),
        "mean": float(statistics.fmean(numeric)),
        "std": float(statistics.pstdev(numeric)) if len(numeric) > 1 else 0.0,
        "min": float(min(numeric)),
        "p25": percentile(numeric, 0.25),
        "median": percentile(numeric, 0.5),
        "p75": percentile(numeric, 0.75),
        "max": float(max(numeric)),
    }


def build_summary(
    records: Sequence[Dict[str, Any]],
    sample_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    protocol_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    protocol_to_samples: Dict[str, set] = defaultdict(set)
    protocol_to_sample_field_counts: Dict[str, List[int]] = defaultdict(list)

    for record in records:
        protocol = record["protocol_name"]
        protocol_to_records[protocol].append(record)
        protocol_to_samples[protocol].add(record["sample_id"])

    for sample_summary in sample_summaries:
        protocol = safe_protocol_name(sample_summary["protocol_dir_name"])
        protocol_to_sample_field_counts[protocol].append(sample_summary["field_count"])

    per_protocol: Dict[str, Any] = {}
    for protocol in sorted(protocol_to_records):
        per_protocol[protocol] = {
            "field_sample_count": len(protocol_to_records[protocol]),
            "sample_count": len(protocol_to_samples[protocol]),
            "avg_fields_per_sample": (
                float(statistics.fmean(protocol_to_sample_field_counts[protocol]))
                if protocol_to_sample_field_counts[protocol]
                else 0.0
            ),
            "mutation_count_distribution": summarize_distribution(
                record.get("mutation_count") for record in protocol_to_records[protocol]
            ),
            "valid_mutations_distribution": summarize_distribution(
                record.get("valid_mutations") for record in protocol_to_records[protocol]
            ),
            "unique_metric_vectors_distribution": summarize_distribution(
                record.get("unique_metric_vectors") for record in protocol_to_records[protocol]
            ),
        }

    return {
        "total_protocols": len(protocol_to_sample_field_counts),
        "total_samples": len(sample_summaries),
        "total_field_samples": len(records),
        "per_protocol": per_protocol,
        "mutation_count_distribution": summarize_distribution(record.get("mutation_count") for record in records),
        "valid_mutations_distribution": summarize_distribution(record.get("valid_mutations") for record in records),
        "unique_metric_vectors_distribution": summarize_distribution(
            record.get("unique_metric_vectors") for record in records
        ),
    }


def write_csv(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_jsonl(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(data: Dict[str, Any], output_path: Path, indent: Optional[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)


def write_json_array(records: Sequence[Dict[str, Any]], output_path: Path, indent: Optional[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=indent)


def describe_column(column: str) -> Dict[str, str]:
    if column in {"protocol_id", "protocol_name", "sample_id", "sample_path"}:
        mapping = {
            "protocol_id": ("协议目录原始标识，例如 out_modbus。", "来自样本所在协议目录名。"),
            "protocol_name": ("归一化后的协议名，例如 modbus、S7。", "对 protocol_id 去掉前缀 out_ 后得到。"),
            "sample_id": ("样本目录名，例如 sample_001。", "来自样本目录名。"),
            "sample_path": ("样本目录绝对路径。", "来自当前 sample 目录路径。"),
        }
        meaning, calc = mapping[column]
        return {"meaning": meaning, "calculation": calc}

    if column in {"field_id", "field_index", "field_range_start", "field_range_end", "field_range_repr", "nbytes"}:
        mapping = {
            "field_id": ("字段标识。", "优先读取 report.json 中字段的 field_id；缺失时退化为 field_{index}。"),
            "field_index": ("字段在当前样本字段列表中的顺序编号。", "遍历 report.json 的 fields 列表时的索引。"),
            "field_range_start": ("字段起始字节偏移。", "来自字段 range.a。"),
            "field_range_end": ("字段结束字节偏移。", "来自字段 range.b。"),
            "field_range_repr": ("字段范围的字符串形式，例如 10_11。", "由 field_range_start 与 field_range_end 拼接得到。"),
            "nbytes": ("字段字节长度。", "来自 report.json 中字段的 nbytes。"),
        }
        meaning, calc = mapping[column]
        return {"meaning": meaning, "calculation": calc}

    if column in {"baseline_field_hex", "baseline_payload_hex", "payload_len_bytes"}:
        mapping = {
            "baseline_field_hex": ("baseline 报文中该字段的十六进制值。", "来自 report.json 中字段的 baseline_field_hex。"),
            "baseline_payload_hex": ("baseline 整包 payload 的十六进制表示。", "来自 sample.json 的 payload_hex。"),
            "payload_len_bytes": ("baseline payload 总字节数。", "优先由 baseline_payload_hex 的长度除以 2；否则使用 report.input.payload_len。"),
        }
        meaning, calc = mapping[column]
        return {"meaning": meaning, "calculation": calc}

    if column in {"valid_mutations", "mutation_count", "mutations_json_count"}:
        mapping = {
            "valid_mutations": ("当前字段有效 mutation 数。", "优先读取 report.diff.summary.valid_mutations；缺失时回退为 per_mutation 数量或状态为 ok 的 mutation run 数。"),
            "mutation_count": ("当前字段实际用于聚合统计的 mutation 观测数。", "优先取 diff.per_mutation 的条目数；若缺失则统计 runs 中 kind=mutation 的有效运行数。"),
            "mutations_json_count": ("mutations.json 中该字段的候选 mutation 数。", "统计对应字段 mutation_entry.mutations 的列表长度。"),
        }
        meaning, calc = mapping[column]
        return {"meaning": meaning, "calculation": calc}

    if column in {"deltaf_dispersion", "unique_metric_vectors"}:
        mapping = {
            "deltaf_dispersion": ("字段多次 mutation 行为指标向量的离散度。", "来自 diff.summary.diagnostic.deltaf_dispersion；本质上是多项行为指标在 mutation 间波动强度的聚合量。"),
            "unique_metric_vectors": ("字段多次 mutation 后出现的不同指标向量个数。", "来自 diff.summary.diagnostic.unique_metric_vectors；通常先对单次 mutation 指标向量做适度 rounding 后计数。"),
        }
        meaning, calc = mapping[column]
        return {"meaning": meaning, "calculation": calc}

    if column in {
        "sample_proto", "sample_mode", "sample_seed", "sample_index",
        "fields_in_sample", "field_partition_source"
    }:
        mapping = {
            "sample_proto": ("样本传输协议类型，如 tcp/udp。", "来自 sample.json.proto。"),
            "sample_mode": ("样本来源模式，如 pcap/hex。", "来自 sample.json.mode。"),
            "sample_seed": ("样本选择或生成时的随机种子。", "来自 sample.json.seed。"),
            "sample_index": ("样本在来源数据中的索引。", "来自 sample.json.index。"),
            "fields_in_sample": ("该样本分割出的字段数量。", "优先统计 fields.json.fields 的数量；缺失时回退到 report.input.fields_partition 长度。"),
            "field_partition_source": ("字段切分来源标记。", "来自 report.input.field_partition_source。"),
        }
        meaning, calc = mapping[column]
        return {"meaning": meaning, "calculation": calc}

    if column.startswith("baseline_"):
        mapping = {
            "baseline_status": "baseline 运行状态。",
            "baseline_taint_found": "baseline 是否找到 taint 记录。",
            "baseline_bb_parsed": "baseline 解析出的基本块计数。",
            "baseline_branch_parsed": "baseline 解析出的分支计数。",
            "baseline_instr_parsed": "baseline 解析出的指令计数。",
            "baseline_bad_lines_dropped": "baseline 预处理时丢弃的异常行数。",
            "baseline_preprocess_line_count": "baseline 预处理输入行数。",
            "baseline_taint_thread_id": "baseline taint 日志中的线程 ID。",
        }
        if column in mapping:
            return {"meaning": mapping[column], "calculation": "来自 baseline run 的 parse_health / preprocess / status 字段。"}

    if column.startswith("mutation_status_"):
        return {
            "meaning": "字段级 mutation 运行状态统计。",
            "calculation": "统计 runs 中 kind=mutation 的状态计数；ok 单独计数，其余状态汇总到 non_ok。",
        }

    return {
        "meaning": "字段级样本表中的辅助列。",
        "calculation": "由 report.json / sample.json / fields.json / mutations.json 中的上下文字段直接提取或简单转换得到。",
    }


def write_column_docs(
    records: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for record in records for key in record.keys()})
    lines: List[str] = []
    lines.append("# Field Training Sample Columns")
    lines.append("")
    lines.append("本文件说明 `field_training_samples.csv / jsonl / json` 中每一列的含义，以及字段级指标的计算方式。")
    lines.append("")
    lines.append("## 文件关系")
    lines.append("")
    lines.append("- `field_training_samples.csv`：主训练样本表，一行一个字段。")
    lines.append("- `field_training_samples.jsonl`：与 CSV 行级内容一致，每行一个字段对象。")
    lines.append("- `field_training_samples.json`：如果显式开启，与 CSV/JSONL 内容一致，只是 JSON 数组格式。")
    lines.append("- `field_training_summary.json`：数据集整体统计摘要，不是逐字段样本表。")
    lines.append("")
    lines.append("说明：CSV、JSONL、JSON 三者的逐字段记录内容应当一致，差别主要在序列化格式；`summary` 则是聚合统计结果，不与前面三者逐行一致。")
    lines.append("")
    lines.append("## 字段级样本列")
    lines.append("")
    for column in columns:
        desc = describe_column(column)
        lines.append(f"### `{column}`")
        lines.append("")
        lines.append(f"- 含义：{desc['meaning']}")
        lines.append(f"- 计算方式：{desc['calculation']}")
        lines.append("")

    lines.append("## Summary 顶层 key")
    lines.append("")
    lines.append("- `total_protocols`：协议总数。")
    lines.append("- `total_samples`：sample 总数。")
    lines.append("- `total_field_samples`：字段级样本总数。")
    lines.append("- `per_protocol`：按协议分组的统计摘要。")
    lines.append("- `mutation_count_distribution`：全体字段的 mutation_count 分布。")
    lines.append("- `valid_mutations_distribution`：全体字段的 valid_mutations 分布。")
    lines.append("- `unique_metric_vectors_distribution`：全体字段的 unique_metric_vectors 分布。")
    lines.append("")
    lines.append("## V3 行为聚合特征的一般计算流程")
    lines.append("")
    lines.append("1. 对同一字段按 `strategy_group` 收集 mutation。")
    lines.append("2. 将每次 mutation 的差分指标转换为 9 维差异方向向量。")
    lines.append("3. 每个策略组聚合为 6 维摘要。")
    lines.append("4. 拼接 4 维上下文特征和 4 个策略组摘要，得到 28 维输入向量。")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate difftrace experiment outputs into field-level training samples."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for field-level sample artifacts.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path for a JSON array of all field records.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--output-columns-md",
        type=Path,
        default=DEFAULT_OUTPUT_COLUMNS_MD,
        help="Markdown file describing each field-sample column and how it is computed.",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default=None,
        help="Comma-separated protocol filters, e.g. 'modbus,S7,out_CIP'.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Comma-separated sample directory names, e.g. 'sample_001,sample_003'.",
    )
    parser.add_argument(
        "--sample-pattern",
        type=str,
        default="sample_*",
        help="Glob pattern for sample directories under each protocol directory.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Stop after processing at most this many discovered sample directories.",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable JSONL output and only write CSV + summary.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV output.",
    )
    parser.add_argument(
        "--pretty-json-indent",
        type=int,
        default=2,
        help="Indent level for JSON outputs. Use 0 for compact JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logging; warnings are still printed.",
    )
    return parser.parse_args()


def main() -> int:
    global VERBOSE
    args = parse_args()
    VERBOSE = not args.quiet
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
    sample_summaries: List[Dict[str, Any]] = []

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        info(f"[{idx}/{len(sample_dirs)}] Processing {sample_dir}")
        records, sample_summary = process_sample(sample_dir)
        all_records.extend(records)
        sample_summaries.append(sample_summary)

    if not all_records:
        warn("No field-level records were generated.")
        return 1

    summary = build_summary(all_records, sample_summaries)
    json_indent = None if args.pretty_json_indent == 0 else args.pretty_json_indent

    output_dir = args.output_dir
    default_output_map = {
        DEFAULT_OUTPUT_CSV: output_dir / DEFAULT_OUTPUT_CSV.name,
        DEFAULT_OUTPUT_JSONL: output_dir / DEFAULT_OUTPUT_JSONL.name,
        DEFAULT_OUTPUT_JSON: output_dir / DEFAULT_OUTPUT_JSON.name,
        DEFAULT_OUTPUT_SUMMARY: output_dir / DEFAULT_OUTPUT_SUMMARY.name,
        DEFAULT_OUTPUT_COLUMNS_MD: output_dir / DEFAULT_OUTPUT_COLUMNS_MD.name,
    }

    output_csv = default_output_map.get(args.output_csv, args.output_csv)
    output_jsonl = default_output_map.get(args.output_jsonl, args.output_jsonl)
    output_json = default_output_map.get(args.output_json, args.output_json) if args.output_json is not None else None
    output_summary = default_output_map.get(args.output_summary, args.output_summary)
    output_columns_md = default_output_map.get(args.output_columns_md, args.output_columns_md)

    if not args.no_csv:
        write_csv(all_records, output_csv)
    if not args.no_jsonl:
        write_jsonl(all_records, output_jsonl)
    if output_json is not None:
        write_json_array(all_records, output_json, json_indent)
    write_json(summary, output_summary, json_indent)
    write_column_docs(all_records, summary, output_columns_md)

    if not args.no_csv:
        info(f"Wrote {len(all_records)} field records to {output_csv}")
    if not args.no_jsonl:
        info(f"Wrote JSONL records to {output_jsonl}")
    if output_json is not None:
        info(f"Wrote JSON records to {output_json}")
    info(f"Wrote dataset summary to {output_summary}")
    info(f"Wrote column documentation to {output_columns_md}")
    info(
        "Summary: "
        f"{summary['total_protocols']} protocols, "
        f"{summary['total_samples']} samples, "
        f"{summary['total_field_samples']} field samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
