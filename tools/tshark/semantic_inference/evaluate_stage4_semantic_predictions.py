#!/usr/bin/env python3
"""Evaluate Stage 4 coarse semantic predictions against tshark groundtruth.

The evaluator accepts either Stage 4 field_semantic_vectors.csv or
field_semantic_profiles.jsonl as prediction input. Groundtruth is the tshark
semantic candidate/final CSV produced under semantic_inference/eval.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PREDICTIONS = Path(
    "/root/semvec/difftrace/stage4/out/stage4_field_semantic_fusion/field_semantic_fused_vectors.csv"
)
DEFAULT_GROUNDTRUTH = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_tshark/semantic_inference/eval/"
    "tshark_semantic_groundtruth_candidates.csv"
)
DEFAULT_OUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_tshark/semantic_inference/eval/stage4_metrics"
)

KEY_COLS = ("protocol_name", "sample_id", "field_id")
COARSE_TAGS = {
    "identifier",
    "length_or_count",
    "control_or_flags",
    "addressing",
    "data_value",
    "other_or_unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute top-k Stage 4 semantic metrics against tshark semantic groundtruth.",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GROUNDTRUTH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--top-k",
        default="1,2,3",
        help="Comma-separated top-k values to report, for example 1,2,3.",
    )
    parser.add_argument(
        "--groundtruth-label-col",
        default="semantic_group",
        help="Groundtruth CSV column containing the coarse semantic label.",
    )
    parser.add_argument(
        "--exclude-review",
        action="store_true",
        help="Exclude groundtruth rows where needs_review is true.",
    )
    parser.add_argument(
        "--exclude-other",
        action="store_true",
        help="Exclude groundtruth labels equal to other_or_unknown.",
    )
    return parser.parse_args()


def key_of(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(col, "")).strip() for col in KEY_COLS)  # type: ignore[return-value]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = [str(item).strip() for item in value]
    else:
        text = str(value or "").strip()
        if not text:
            return []
        raw = [part.strip() for part in text.replace(",", ";").split(";")]
    tags: list[str] = []
    for tag in raw:
        if tag in COARSE_TAGS and tag not in tags:
            tags.append(tag)
    return tags


def load_predictions_csv(path: Path) -> dict[tuple[str, str, str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY_COLS)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        predictions: dict[tuple[str, str, str], list[str]] = {}
        for row in reader:
            key = key_of(row)
            tags = split_tags(row.get("traditional_semantic_tags"))
            if not tags:
                tags = split_tags(row.get("predicted_semantic_tags"))
            if not tags and row.get("top1_semantic_tag"):
                tags = split_tags(row.get("top1_semantic_tag"))
            predictions[key] = tags
    return predictions


def load_predictions_jsonl(path: Path) -> dict[tuple[str, str, str], list[str]]:
    predictions: dict[tuple[str, str, str], list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} invalid JSON: {exc}") from exc
            key = key_of(row)
            tags = split_tags(row.get("traditional_semantic_tags"))
            if not tags:
                tags = split_tags(row.get("predicted_semantic_tags"))
            if not tags and row.get("top1_semantic_tag"):
                tags = split_tags(row.get("top1_semantic_tag"))
            predictions[key] = tags
    return predictions


def load_predictions(path: Path) -> dict[tuple[str, str, str], list[str]]:
    if path.suffix == ".jsonl":
        return load_predictions_jsonl(path)
    return load_predictions_csv(path)


def load_prediction_details(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    details: dict[tuple[str, str, str], dict[str, Any]] = {}
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no} invalid JSON: {exc}") from exc
                key = key_of(row)
                tags = split_tags(row.get("traditional_semantic_tags"))
                if not tags:
                    tags = split_tags(row.get("predicted_semantic_tags"))
                if not tags and row.get("top1_semantic_tag"):
                    tags = split_tags(row.get("top1_semantic_tag"))
                details[key] = {
                    "predicted_semantic_tags": tags,
                    "top1_semantic_tag": str(row.get("top1_semantic_tag", "") or (tags[0] if tags else "")),
                    "semantic_summary": str(
                        row.get("field_program_semantic_summary")
                        or row.get("semantic_summary")
                        or ""
                    ),
                    "dominant_axes": str(row.get("dominant_axes", "") or ""),
                    "dominant_axis_summary": str(row.get("dominant_axis_summary", "") or ""),
                    "semantic_tag_scores": row.get("semantic_tag_scores", {}),
                    "active_axis_explanations": row.get("active_axis_explanations", []),
                }
        return details

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY_COLS)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            key = key_of(row)
            tags = split_tags(row.get("traditional_semantic_tags"))
            if not tags:
                tags = split_tags(row.get("predicted_semantic_tags"))
            if not tags and row.get("top1_semantic_tag"):
                tags = split_tags(row.get("top1_semantic_tag"))
            details[key] = {
                "predicted_semantic_tags": tags,
                "top1_semantic_tag": str(row.get("top1_semantic_tag", "") or (tags[0] if tags else "")),
                "semantic_summary": str(
                    row.get("field_program_semantic_summary")
                    or row.get("semantic_summary")
                    or ""
                ),
                "dominant_axes": str(row.get("dominant_axes", "") or ""),
                "dominant_axis_summary": str(row.get("dominant_axis_summary", "") or ""),
                "semantic_tag_scores": str(row.get("semantic_tag_scores", "") or ""),
                "active_axis_explanations": "",
            }
    return details


def load_groundtruth(
    path: Path,
    label_col: str,
    exclude_review: bool,
    exclude_other: bool,
) -> dict[tuple[str, str, str], set[str]]:
    truth: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY_COLS) | {label_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            if exclude_review and parse_bool(row.get("needs_review")):
                continue
            label = str(row.get(label_col, "")).strip()
            if label not in COARSE_TAGS:
                continue
            if exclude_other and label == "other_or_unknown":
                continue
            truth[key_of(row)].add(label)
    return dict(truth)


def load_groundtruth_details(
    path: Path,
    label_col: str,
    exclude_review: bool,
    exclude_other: bool,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    details: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY_COLS) | {label_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            if exclude_review and parse_bool(row.get("needs_review")):
                continue
            label = str(row.get(label_col, "")).strip()
            if label not in COARSE_TAGS:
                continue
            if exclude_other and label == "other_or_unknown":
                continue
            key = key_of(row)
            item = details.setdefault(
                key,
                {
                    "semantic_labels": set(),
                    "semantic_groups": set(),
                    "semantic_tags": set(),
                    "needs_review": False,
                    "source_protocols": set(),
                    "source_field_names": set(),
                    "source_parent_fields": set(),
                    "source_showname": "",
                    "source_display_value": "",
                    "notes": set(),
                },
            )
            item["semantic_labels"].add(str(row.get("semantic_label", "")).strip())
            item["semantic_groups"].add(label)
            for tag in split_tags(row.get("semantic_tags")):
                item["semantic_tags"].add(tag)
            item["needs_review"] = item["needs_review"] or parse_bool(row.get("needs_review"))
            for set_key, col in (
                ("source_protocols", "source_protocol"),
                ("source_field_names", "source_field_name"),
                ("source_parent_fields", "source_parent_field"),
                ("notes", "note"),
            ):
                value = str(row.get(col, "") or "").strip()
                if value:
                    item[set_key].add(value)
            if not item["source_showname"]:
                item["source_showname"] = str(row.get("source_showname", "") or "")
            if not item["source_display_value"]:
                item["source_display_value"] = str(row.get("source_display_value", "") or "")

    normalized: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, item in details.items():
        normalized[key] = {
            "semantic_labels": sorted(item["semantic_labels"]),
            "semantic_groups": sorted(item["semantic_groups"]),
            "semantic_tags": sorted(item["semantic_tags"]),
            "needs_review": item["needs_review"],
            "source_protocols": sorted(item["source_protocols"]),
            "source_field_names": sorted(item["source_field_names"]),
            "source_parent_fields": sorted(item["source_parent_fields"]),
            "source_showname": item["source_showname"],
            "source_display_value": item["source_display_value"],
            "notes": sorted(item["notes"]),
        }
    return normalized


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def evaluate_at_k(
    predictions: dict[tuple[str, str, str], list[str]],
    truth: dict[tuple[str, str, str], set[str]],
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_keys = set(predictions)
    truth_keys = set(truth)
    matched_keys = sorted(prediction_keys & truth_keys)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    protocol_rows: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "prediction_fields": 0,
            "groundtruth_fields": 0,
            "matched_fields": 0,
            "prediction_only_fields": 0,
            "groundtruth_only_fields": 0,
            "hit_fields": 0,
        }
    )
    hit_count = 0
    predicted_label_count = 0
    true_label_count = 0

    for key in prediction_keys:
        row = protocol_rows[key[0]]
        row["prediction_fields"] += 1
        if key not in truth_keys:
            row["prediction_only_fields"] += 1
    for key in truth_keys:
        row = protocol_rows[key[0]]
        row["groundtruth_fields"] += 1
        if key not in prediction_keys:
            row["groundtruth_only_fields"] += 1

    for key in matched_keys:
        pred_tags = predictions.get(key, [])[:k]
        true_tags = truth[key]
        protocol = key[0]
        protocol_rows[protocol]["matched_fields"] += 1
        if true_tags.intersection(pred_tags):
            hit_count += 1
            protocol_rows[protocol]["hit_fields"] += 1
        predicted_label_count += len(pred_tags)
        true_label_count += len(true_tags)
        for label in true_tags:
            if label in pred_tags:
                tp[label] += 1
            else:
                fn[label] += 1
        for label in pred_tags:
            if label not in true_tags:
                fp[label] += 1

    labels = sorted(COARSE_TAGS | set(tp) | set(fp) | set(fn))
    per_label: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        label_tp = tp[label]
        label_fp = fp[label]
        label_fn = fn[label]
        precision = safe_div(label_tp, label_tp + label_fp)
        recall = safe_div(label_tp, label_tp + label_fn)
        label_f1 = f1(precision, recall)
        if label_tp or label_fp or label_fn:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(label_f1)
        per_label.append(
            {
                "k": k,
                "label": label,
                "tp": label_tp,
                "fp": label_fp,
                "fn": label_fn,
                "precision": precision,
                "recall": recall,
                "f1": label_f1,
            }
        )

    micro_precision = safe_div(sum(tp.values()), sum(tp.values()) + sum(fp.values()))
    micro_recall = safe_div(sum(tp.values()), sum(tp.values()) + sum(fn.values()))
    macro_precision = safe_div(sum(precisions), len(precisions))
    macro_recall = safe_div(sum(recalls), len(recalls))
    macro_f1 = safe_div(sum(f1s), len(f1s))

    per_protocol = []
    for protocol, counts in sorted(protocol_rows.items()):
        per_protocol.append(
            {
                "k": k,
                "protocol_name": protocol,
                "prediction_fields": counts["prediction_fields"],
                "groundtruth_fields": counts["groundtruth_fields"],
                "matched_fields": counts["matched_fields"],
                "prediction_only_fields": counts["prediction_only_fields"],
                "groundtruth_only_fields": counts["groundtruth_only_fields"],
                "hit_fields": counts["hit_fields"],
                "hit_rate": safe_div(counts["hit_fields"], counts["matched_fields"]),
                "match_rate_vs_groundtruth": safe_div(
                    counts["matched_fields"], counts["groundtruth_fields"]
                ),
                "match_rate_vs_predictions": safe_div(
                    counts["matched_fields"], counts["prediction_fields"]
                ),
            }
        )

    summary = {
        "k": k,
        "matched_fields": len(matched_keys),
        "hit_fields": hit_count,
        "topk_hit_rate": safe_div(hit_count, len(matched_keys)),
        "topk_accuracy": safe_div(hit_count, len(matched_keys)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": f1(micro_precision, micro_recall),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "predicted_label_count": predicted_label_count,
        "true_label_count": true_label_count,
    }
    return summary, per_label, per_protocol


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_overall_protocol_rows(
    per_protocol: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_by_k = {row["k"]: row for row in summaries}
    for index, row in enumerate(per_protocol):
        rows.append(row)
        next_index = index + 1
        next_k = per_protocol[next_index]["k"] if next_index < len(per_protocol) else None
        if next_k == row["k"]:
            continue
        summary = summary_by_k[row["k"]]
        rows.append(
            {
                "k": row["k"],
                "protocol_name": "overall",
                "prediction_fields": metadata["prediction_fields"],
                "groundtruth_fields": metadata["groundtruth_fields"],
                "matched_fields": metadata["matched_fields"],
                "prediction_only_fields": metadata["prediction_only_fields"],
                "groundtruth_only_fields": metadata["groundtruth_only_fields"],
                "hit_fields": summary["hit_fields"],
                "hit_rate": summary["topk_hit_rate"],
                "match_rate_vs_groundtruth": safe_div(
                    metadata["matched_fields"], metadata["groundtruth_fields"]
                ),
                "match_rate_vs_predictions": safe_div(
                    metadata["matched_fields"], metadata["prediction_fields"]
                ),
            }
        )
    return rows


def fmt_float(value: Any) -> str:
    return f"{float(value):.4f}"


def write_readable(
    path: Path,
    metadata: dict[str, Any],
    summaries: list[dict[str, Any]],
    per_protocol_with_overall: list[dict[str, Any]],
    per_label: list[dict[str, Any]],
    field_comparison_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Stage 4 TShark Semantic Metrics")
    lines.append("")
    lines.append("Stage 4 coarse semantic predictions are compared with tshark semantic groundtruth on matched fields.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- predictions: `{metadata['predictions']}`")
    lines.append(f"- groundtruth: `{metadata['groundtruth']}`")
    lines.append(f"- groundtruth label column: `{metadata['groundtruth_label_col']}`")
    lines.append(f"- exclude_review: `{metadata['exclude_review']}`")
    lines.append(f"- exclude_other: `{metadata['exclude_other']}`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(
        "| k | prediction fields | groundtruth fields | matched fields | "
        "prediction-only fields | groundtruth-only fields | hit fields | hit rate | "
        "micro F1 | macro F1 |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summaries:
        lines.append(
            "| "
            f"{row['k']} | {metadata['prediction_fields']} | {metadata['groundtruth_fields']} | "
            f"{metadata['matched_fields']} | {metadata['prediction_only_fields']} | "
            f"{metadata['groundtruth_only_fields']} | {row['hit_fields']} | "
            f"{fmt_float(row['topk_hit_rate'])} | {fmt_float(row['micro_f1'])} | "
            f"{fmt_float(row['macro_f1'])} |"
        )
    lines.append("")
    lines.append("## Protocol Metrics")
    lines.append("")
    lines.append(
        "| k | protocol | prediction fields | groundtruth fields | matched fields | "
        "prediction-only fields | groundtruth-only fields | hit fields | hit rate | "
        "match rate vs groundtruth | match rate vs predictions |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_protocol_with_overall:
        lines.append(
            "| "
            f"{row['k']} | {row['protocol_name']} | {row['prediction_fields']} | "
            f"{row['groundtruth_fields']} | {row['matched_fields']} | "
            f"{row['prediction_only_fields']} | {row['groundtruth_only_fields']} | "
            f"{row['hit_fields']} | {fmt_float(row['hit_rate'])} | "
            f"{fmt_float(row['match_rate_vs_groundtruth'])} | "
            f"{fmt_float(row['match_rate_vs_predictions'])} |"
        )
    lines.append("")
    lines.append("## Label Metrics")
    lines.append("")
    lines.append("| k | label | TP | FP | FN | precision | recall | F1 |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_label:
        lines.append(
            "| "
            f"{row['k']} | {row['label']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{fmt_float(row['precision'])} | {fmt_float(row['recall'])} | {fmt_float(row['f1'])} |"
        )
    lines.append("")
    lines.append("## Column Meaning")
    lines.append("")
    lines.append("- `k`: top-k cutoff. A field is counted as hit if any groundtruth label appears in the first k predicted labels.")
    lines.append("- `prediction fields`: number of Stage 4 fields with semantic predictions.")
    lines.append("- `groundtruth fields`: number of tshark groundtruth fields after the selected filters.")
    lines.append("- `matched fields`: fields whose `(protocol_name, sample_id, field_id)` exists in both prediction and groundtruth.")
    lines.append("- `prediction-only fields`: predicted fields without a matched tshark groundtruth field.")
    lines.append("- `groundtruth-only fields`: tshark groundtruth fields without a matched Stage 4 prediction.")
    lines.append("- `hit fields`: matched fields where the top-k predicted labels contain at least one groundtruth semantic label.")
    lines.append("- `hit rate`: `hit_fields / matched_fields`; this is the top-k semantic accuracy on matched fields.")
    lines.append("- `match rate vs groundtruth`: `matched_fields / groundtruth_fields`, measuring field coverage from the groundtruth side.")
    lines.append("- `match rate vs predictions`: `matched_fields / prediction_fields`, measuring field coverage from the prediction side.")
    lines.append("- `micro F1`: label-level micro F1 over matched fields at the given k.")
    lines.append("- `macro F1`: average label-level F1 over labels that appear in TP/FP/FN at the given k.")
    lines.append("")
    append_field_comparison_readable(lines, field_comparison_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def join_values(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, set):
        return ";".join(sorted(str(value) for value in values if str(value)))
    if isinstance(values, list):
        return ";".join(str(value) for value in values if str(value))
    if isinstance(values, dict):
        return json.dumps(values, ensure_ascii=False, sort_keys=True)
    return str(values)


def build_field_comparison_rows(
    prediction_details: dict[tuple[str, str, str], dict[str, Any]],
    groundtruth_details: dict[tuple[str, str, str], dict[str, Any]],
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(prediction_details) | set(groundtruth_details)):
        protocol_name, sample_id, field_id = key
        pred = prediction_details.get(key, {})
        truth = groundtruth_details.get(key, {})
        pred_tags = list(pred.get("predicted_semantic_tags", []))
        truth_groups = set(truth.get("semantic_groups", []))
        if key in prediction_details and key in groundtruth_details:
            match_status = "matched"
        elif key in prediction_details:
            match_status = "prediction_only"
        else:
            match_status = "groundtruth_only"

        row: dict[str, Any] = {
            "protocol_name": protocol_name,
            "sample_id": sample_id,
            "field_id": field_id,
            "match_status": match_status,
            "tshark_semantic_labels": join_values(truth.get("semantic_labels")),
            "tshark_semantic_groups": join_values(truth.get("semantic_groups")),
            "tshark_semantic_tags": join_values(truth.get("semantic_tags")),
            "tshark_needs_review": truth.get("needs_review", ""),
            "tshark_source_protocols": join_values(truth.get("source_protocols")),
            "tshark_source_field_names": join_values(truth.get("source_field_names")),
            "tshark_source_parent_fields": join_values(truth.get("source_parent_fields")),
            "tshark_source_showname": truth.get("source_showname", ""),
            "tshark_source_display_value": truth.get("source_display_value", ""),
            "difftrace_program_semantic_summary": pred.get("semantic_summary", ""),
            "difftrace_traditional_semantic_tags": join_values(pred_tags),
            "difftrace_top1_semantic_tag": pred.get("top1_semantic_tag", ""),
            "difftrace_semantic_tag_scores": join_values(pred.get("semantic_tag_scores")),
            "difftrace_dominant_axes": pred.get("dominant_axes", ""),
            "difftrace_dominant_axis_summary": pred.get("dominant_axis_summary", ""),
        }
        for k in top_k_values:
            row[f"top{k}_hit"] = bool(truth_groups.intersection(pred_tags[:k])) if match_status == "matched" else ""
        rows.append(row)
    return rows


def append_field_comparison_readable(lines: list[str], rows: list[dict[str, Any]]) -> None:
    matched = [row for row in rows if row["match_status"] == "matched"]
    prediction_only = [row for row in rows if row["match_status"] == "prediction_only"]
    groundtruth_only = [row for row in rows if row["match_status"] == "groundtruth_only"]
    lines.append("## Field-Level TShark Vs DiffTrace")
    lines.append("")
    lines.append("The CSV companion file `stage4_tshark_vs_difftrace_semantic_fields.csv` contains one row per field key.")
    lines.append("")
    lines.append("### Field-Level Summary")
    lines.append("")
    lines.append(f"- total rows: {len(rows)}")
    lines.append(f"- matched fields: {len(matched)}")
    lines.append(f"- prediction-only fields: {len(prediction_only)}")
    lines.append(f"- groundtruth-only fields: {len(groundtruth_only)}")
    lines.append("")
    lines.append("### Field-Level Columns")
    lines.append("")
    lines.append("- `tshark_semantic_labels`: fine-grained tshark-derived semantic labels.")
    lines.append("- `tshark_semantic_groups`: tshark labels projected to the coarse traditional semantic groups used by RQ2-A.")
    lines.append("- `difftrace_program_semantic_summary`: DiffTrace program-behavior semantic summary from Stage 4 latent explanations.")
    lines.append("- `difftrace_traditional_semantic_tags`: DiffTrace coarse traditional semantic tags projected from Stage 4 latent behavior.")
    lines.append("- `top{k}_hit`: whether `difftrace_traditional_semantic_tags[:k]` contains any tshark semantic group for matched fields.")
    lines.append("- `match_status`: `matched`, `prediction_only`, or `groundtruth_only` under `(protocol_name, sample_id, field_id)` alignment.")
    lines.append("")
    lines.append("### Matched Field Preview")
    lines.append("")
    lines.append("| protocol | sample | field | tshark groups | difftrace tags | top1 | program semantic summary |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in matched[:100]:
        summary = str(row["difftrace_program_semantic_summary"]).replace("|", "\\|")
        if len(summary) > 180:
            summary = summary[:177] + "..."
        lines.append(
            "| "
            f"{row['protocol_name']} | {row['sample_id']} | {row['field_id']} | "
            f"{row['tshark_semantic_groups']} | {row['difftrace_traditional_semantic_tags']} | "
            f"{row.get('top1_hit', '')} | {summary} |"
        )
    lines.append("")


def main() -> None:
    args = parse_args()
    top_k_values = [int(part) for part in args.top_k.split(",") if part.strip()]
    if not top_k_values or any(k <= 0 for k in top_k_values):
        raise SystemExit("--top-k must contain positive integers")

    predictions = load_predictions(args.predictions)
    prediction_details = load_prediction_details(args.predictions)
    truth = load_groundtruth(
        args.groundtruth,
        args.groundtruth_label_col,
        args.exclude_review,
        args.exclude_other,
    )
    groundtruth_details = load_groundtruth_details(
        args.groundtruth,
        args.groundtruth_label_col,
        args.exclude_review,
        args.exclude_other,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    all_per_label: list[dict[str, Any]] = []
    all_per_protocol: list[dict[str, Any]] = []
    for k in top_k_values:
        summary, per_label, per_protocol = evaluate_at_k(predictions, truth, k)
        all_summaries.append(summary)
        all_per_label.extend(per_label)
        all_per_protocol.extend(per_protocol)

    metadata = {
        "predictions": str(args.predictions),
        "groundtruth": str(args.groundtruth),
        "groundtruth_label_col": args.groundtruth_label_col,
        "exclude_review": args.exclude_review,
        "exclude_other": args.exclude_other,
        "prediction_fields": len(predictions),
        "groundtruth_fields": len(truth),
        "matched_fields": len(set(predictions) & set(truth)),
        "prediction_only_fields": len(set(predictions) - set(truth)),
        "groundtruth_only_fields": len(set(truth) - set(predictions)),
        "top_k": top_k_values,
    }
    summary_obj = {
        "metadata": metadata,
        "metrics": all_summaries,
    }

    summary_path = args.out_dir / "stage4_tshark_semantic_metrics.json"
    summary_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_per_protocol_with_overall = add_overall_protocol_rows(
        all_per_protocol, all_summaries, metadata
    )
    write_csv(args.out_dir / "stage4_tshark_semantic_metrics_by_label.csv", all_per_label)
    write_csv(
        args.out_dir / "stage4_tshark_semantic_metrics_by_protocol.csv",
        all_per_protocol_with_overall,
    )
    field_comparison_rows = build_field_comparison_rows(
        prediction_details, groundtruth_details, top_k_values
    )
    field_comparison_path = args.out_dir / "stage4_tshark_vs_difftrace_semantic_fields.csv"
    write_csv(field_comparison_path, field_comparison_rows)
    write_readable(
        args.out_dir / "stage4_tshark_semantic_metrics_readable.md",
        metadata,
        all_summaries,
        all_per_protocol_with_overall,
        all_per_label,
        field_comparison_rows,
    )

    print(f"[rq2a] predictions: {len(predictions)}")
    print(f"[rq2a] groundtruth fields: {len(truth)}")
    print(f"[rq2a] matched fields: {metadata['matched_fields']}")
    for row in all_summaries:
        print(
            "[rq2a] "
            f"k={row['k']} hit_rate={row['topk_hit_rate']:.6f} "
            f"micro_f1={row['micro_f1']:.6f} macro_f1={row['macro_f1']:.6f}"
        )
    print(f"[rq2a] wrote: {summary_path}")
    print(f"[rq2a] wrote: {field_comparison_path}")


if __name__ == "__main__":
    main()
