#!/usr/bin/env python3
"""Evaluate program-log field-boundary groundtruth against experiment output."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_GT_JSONL = (
    ROOT
    / "evaluation_from_program_log"
    / "groundtruth_result"
    / "eval"
    / "program_log_groundtruth_candidates.jsonl"
)
DEFAULT_REPLAY_ROOT = ROOT / "replay_manual_latest" / "outputs"
DEFAULT_OUTDIR = ROOT / "evaluation_from_program_log" / "groundtruth_result" / "eval"
DEFAULT_GT_READABLE = DEFAULT_OUTDIR / "field_boundary_groundtruth_readable.md"
DEFAULT_COMPARE_READABLE = DEFAULT_OUTDIR / "field_boundary_groundtruth_vs_experiment_readable.md"

ByteRange = Tuple[int, int]
BitLabel = str
BitfieldMap = Dict[ByteRange, Set[BitLabel]]
ByteBoundary = int
BitBoundary = Tuple[ByteRange, int]


@dataclass
class SetMetrics:
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        precision = ratio(self.tp, self.tp + self.fp)
        recall = ratio(self.tp, self.tp + self.fn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "total": self.tp + self.fp + self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "jaccard": ratio(self.tp, self.tp + self.fp + self.fn),
        }


@dataclass
class PacketGroundtruth:
    seq: int
    protocol: str
    sample_id: str
    fields: set[ByteRange]
    bitfields: BitfieldMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build readable program-log field-boundary groundtruth and compare it with experiment output."
    )
    parser.add_argument("--groundtruth-jsonl", type=Path, default=DEFAULT_GT_JSONL)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--groundtruth-md", type=Path, default=DEFAULT_GT_READABLE)
    parser.add_argument("--compare-md", type=Path, default=DEFAULT_COMPARE_READABLE)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def set_metrics(gt: set[Any], pred: set[Any]) -> SetMetrics:
    return SetMetrics(tp=len(gt & pred), fp=len(pred - gt), fn=len(gt - pred))


def add_metrics(acc: dict[str, int], metrics: dict[str, Any]) -> None:
    for key in ("tp", "fp", "fn"):
        acc[key] += int(metrics[key])


def parse_range_repr(text: str) -> ByteRange:
    text = str(text).strip()
    if "," in text:
        parts = [int(part) for part in text.split(",") if part.strip()]
        return min(parts), max(parts)
    if "-" in text:
        start, end = text.split("-", 1)
        return int(start), int(end)
    value = int(text)
    return value, value


def parse_field_id(field_id: str) -> tuple[str, ByteRange, BitLabel | None]:
    byte_match = re.fullmatch(r"b:(\d+):(\d+)", field_id)
    if byte_match:
        start, end = int(byte_match.group(1)), int(byte_match.group(2))
        return "byte", (start, end), None

    bit_match = re.fullmatch(r"bit:(\d+):(\d+):(\d+):(\d+)", field_id)
    if bit_match:
        start = int(bit_match.group(1))
        end = int(bit_match.group(2))
        low = int(bit_match.group(3))
        high = int(bit_match.group(4))
        return "bit", (start, end), format_bit_label(low, high)

    raise ValueError(
        f"unsupported field_id: {field_id!r}; expected 'b:start:end' or "
        "'bit:start:end:low:high'"
    )


def compact_for_error(value: object, max_len: int = 600) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_bit_label(low: int, high: int) -> str:
    if low > high:
        low, high = high, low
    if low == high:
        return f"[{low}]"
    return f"[{high}:{low}]"


def normalize_label(label: str) -> str:
    label = str(label).strip()
    if not (label.startswith("[") and label.endswith("]")):
        return label
    inner = label[1:-1]
    if ":" in inner:
        high, low = inner.split(":", 1)
        return format_bit_label(int(low), int(high))
    return f"[{int(inner)}]"


def bit_label_bounds(label: str) -> tuple[int, int] | None:
    label = normalize_label(label)
    inner = label[1:-1] if label.startswith("[") and label.endswith("]") else label
    if ":" in inner:
        high, low = inner.split(":", 1)
        return int(low), int(high)
    if inner.isdigit():
        bit = int(inner)
        return bit, bit
    return None


def label_sort_key(label: str) -> tuple[int, int, str]:
    bounds = bit_label_bounds(label)
    if bounds is None:
        return (9999, 9999, label)
    low, high = bounds
    return (low, high, normalize_label(label))


def range_text(byte_range: ByteRange) -> str:
    start, end = byte_range
    if start == end:
        return f"[{start}]"
    return "[" + ",".join(str(index) for index in range(start, end + 1)) + "]"


def line(items: Iterable[str]) -> str:
    values = list(items)
    return " ".join(values) if values else "-"


def ranges_line(ranges: Iterable[ByteRange]) -> str:
    return line(range_text(item) for item in sorted(ranges))


def bitfield_items(bitfields: BitfieldMap) -> list[str]:
    items: list[str] = []
    for byte_range in sorted(bitfields):
        for label in sorted(bitfields[byte_range], key=label_sort_key):
            items.append(f"{range_text(byte_range)}{normalize_label(label)}")
    return items


def bitfield_parent_items(bitfields: BitfieldMap) -> list[str]:
    return [range_text(byte_range) for byte_range in sorted(bitfields)]


def byte_boundary_text(boundary: ByteBoundary) -> str:
    return f"B{boundary}"


def bit_boundary_text(boundary: BitBoundary) -> str:
    byte_range, bit_boundary = boundary
    return f"{range_text(byte_range)}B{bit_boundary}"


def metric_short(metrics: dict[str, Any]) -> str:
    return "P={precision:.4f} R={recall:.4f} F1={f1:.4f}".format(**metrics)


def field_boundaries(ranges: Iterable[ByteRange]) -> set[ByteBoundary]:
    boundaries: set[ByteBoundary] = set()
    for start, end in ranges:
        boundaries.add(start)
        boundaries.add(end + 1)
    return boundaries


def bitfield_subfield_boundaries(bitfields: BitfieldMap) -> set[BitBoundary]:
    boundaries: set[BitBoundary] = set()
    for byte_range, labels in bitfields.items():
        for label in labels:
            bounds = bit_label_bounds(label)
            if bounds is None:
                continue
            low, high = bounds
            boundaries.add((byte_range, low))
            boundaries.add((byte_range, high + 1))
    return boundaries


def diff_bitfields(left: BitfieldMap, right: BitfieldMap) -> BitfieldMap:
    diff: BitfieldMap = {}
    for byte_range in sorted(set(left) | set(right)):
        labels = left.get(byte_range, set()) - right.get(byte_range, set())
        if labels:
            diff[byte_range] = set(labels)
    return diff


def boundary_metrics(gt: BitfieldMap, pred: BitfieldMap) -> dict[str, Any]:
    gt_ranges = set(gt)
    pred_ranges = set(pred)
    exact_gt = sum(1 for byte_range in gt_ranges if pred.get(byte_range, set()) == gt[byte_range])
    exact_pred = sum(1 for byte_range in pred_ranges if gt.get(byte_range, set()) == pred[byte_range])
    gt_pairs = {(byte_range, label) for byte_range, labels in gt.items() for label in labels}
    pred_pairs = {(byte_range, label) for byte_range, labels in pred.items() for label in labels}
    return {
        "gt_bitfield_count": len(gt_ranges),
        "pred_bitfield_count": len(pred_ranges),
        "exact_match_count": exact_gt,
        "exact_match_recall": ratio(exact_gt, len(gt_ranges)),
        "exact_match_precision": ratio(exact_pred, len(pred_ranges)),
        "subfield": set_metrics(gt_pairs, pred_pairs).to_dict(),
    }


def load_program_log_groundtruth(path: Path) -> dict[tuple[str, str], PacketGroundtruth]:
    packets: dict[tuple[str, str], PacketGroundtruth] = {}
    fields_by_packet: dict[tuple[str, str], set[ByteRange]] = defaultdict(set)
    bits_by_packet: dict[tuple[str, str], BitfieldMap] = defaultdict(lambda: defaultdict(set))
    seq_by_packet: dict[tuple[str, str], int] = {}

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line_text in enumerate(handle, start=1):
            if not line_text.strip():
                continue
            row = json.loads(line_text)
            protocol = str(row.get("protocol_name") or "")
            sample_id = str(row.get("sample_id") or "")
            field_id = str(row.get("field_id") or "")
            if not protocol or not sample_id or not field_id:
                continue
            key = (protocol, sample_id)
            try:
                seq_by_packet[key] = int(row.get("seq") or 0)
                kind, byte_range, bit_label = parse_field_id(field_id)
            except Exception as exc:
                context = {
                    "line_no": line_no,
                    "protocol_name": protocol,
                    "sample_id": sample_id,
                    "seq": row.get("seq", ""),
                    "field_id": field_id,
                    "source_log": row.get("source_log", ""),
                    "source_output": row.get("source_output", ""),
                    "program_log_description": row.get("program_log_description", ""),
                    "raw_row_preview": row,
                }
                raise ValueError(
                    f"{path}:{line_no}: {exc}\n"
                    f"invalid groundtruth row context: {compact_for_error(context)}"
                ) from exc
            if kind == "byte":
                fields_by_packet[key].add(byte_range)
            elif bit_label is not None:
                # Field-boundary evaluation has two layers:
                # byte granularity ignores bit internals and projects every bit
                # subfield to its covered byte range; bit granularity evaluates
                # only the bit subfields.
                fields_by_packet[key].add(byte_range)
                bits_by_packet[key][byte_range].add(bit_label)

    for key in sorted(set(fields_by_packet) | set(bits_by_packet), key=lambda item: (item[0], item[1])):
        protocol, sample_id = key
        packets[key] = PacketGroundtruth(
            seq=seq_by_packet.get(key, 0),
            protocol=protocol,
            sample_id=sample_id,
            fields=set(fields_by_packet.get(key, set())),
            bitfields={byte_range: set(labels) for byte_range, labels in bits_by_packet.get(key, {}).items()},
        )
    return packets


def parse_experiment_fields(path: Path) -> set[ByteRange]:
    if not path.exists():
        return set()
    obj = load_json(path)
    ranges: set[ByteRange] = set()
    for field in obj.get("fields", []):
        if "a" in field and "b" in field:
            ranges.add((int(field["a"]), int(field["b"])))
        elif "field_id" in field:
            ranges.add(parse_range_repr(field["field_id"]))
    return ranges


def parse_experiment_bitfields(path: Path) -> BitfieldMap:
    if not path.exists():
        return {}
    obj = load_json(path)
    bitfields: BitfieldMap = {}
    for field in obj.get("fields", []):
        byte_range = parse_range_repr(field["field_id"])
        labels = {
            normalize_label(sub.get("label", ""))
            for sub in field.get("subfields", [])
            if sub.get("label") is not None
        }
        if labels:
            bitfields[byte_range] = labels
    return bitfields


def payload_len_from_meta(pkt_dir: Path) -> int:
    meta_path = pkt_dir / "meta.json"
    if not meta_path.exists():
        return 0
    meta = load_json(meta_path)
    payload_hex = meta.get("payload_hex") or ""
    return len(bytes.fromhex(payload_hex)) if payload_hex else 0


def in_payload(byte_range: ByteRange, payload_len: int) -> bool:
    return payload_len <= 0 or (0 <= byte_range[0] <= byte_range[1] < payload_len)


def filter_ranges(ranges: set[ByteRange], payload_len: int) -> tuple[set[ByteRange], set[ByteRange]]:
    kept = {item for item in ranges if in_payload(item, payload_len)}
    return kept, ranges - kept


def filter_bitfields(bitfields: BitfieldMap, payload_len: int) -> tuple[BitfieldMap, BitfieldMap]:
    kept: BitfieldMap = {}
    dropped: BitfieldMap = {}
    for byte_range, labels in bitfields.items():
        if in_payload(byte_range, payload_len):
            kept[byte_range] = set(labels)
        else:
            dropped[byte_range] = set(labels)
    return kept, dropped


def compare_one_packet(gt: PacketGroundtruth, replay_root: Path) -> dict[str, Any]:
    pkt_dir = replay_root / gt.protocol / gt.sample_id
    if not pkt_dir.is_dir():
        return {
            "seq": gt.seq,
            "protocol": gt.protocol,
            "packet_dir": gt.sample_id,
            "error": f"missing replay packet directory: {pkt_dir}",
        }

    payload_len = payload_len_from_meta(pkt_dir)
    pred_fields, dropped_pred_fields = filter_ranges(
        parse_experiment_fields(pkt_dir / "fields.json"),
        payload_len,
    )
    pred_bits, dropped_pred_bits = filter_bitfields(
        parse_experiment_bitfields(pkt_dir / "bitfields.json"),
        payload_len,
    )
    gt_fields, dropped_gt_fields = filter_ranges(set(gt.fields), payload_len)
    gt_bits, dropped_gt_bits = filter_bitfields(gt.bitfields, payload_len)

    field = set_metrics(gt_fields, pred_fields).to_dict()
    gt_field_boundaries = field_boundaries(gt_fields)
    pred_field_boundaries = field_boundaries(pred_fields)
    field_boundary_hit = set_metrics(gt_field_boundaries, pred_field_boundaries).to_dict()
    bitfield = set_metrics(set(gt_bits), set(pred_bits)).to_dict()
    boundary = boundary_metrics(gt_bits, pred_bits)
    gt_bit_boundaries = bitfield_subfield_boundaries(gt_bits)
    pred_bit_boundaries = bitfield_subfield_boundaries(pred_bits)
    subfield_boundary_hit = set_metrics(gt_bit_boundaries, pred_bit_boundaries).to_dict()

    return {
        "seq": gt.seq,
        "protocol": gt.protocol,
        "packet_dir": gt.sample_id,
        "payload_len": payload_len,
        "field_boundary": field,
        "field_boundary_hit": field_boundary_hit,
        "bitfield_detection": bitfield,
        "bitfield_boundary": boundary,
        "bitfield_subfield_boundary_hit": subfield_boundary_hit,
        "gt": {
            "fields": [range_text(item) for item in sorted(gt_fields)],
            "field_boundaries": [byte_boundary_text(item) for item in sorted(gt_field_boundaries)],
            "bitfields": bitfield_parent_items(gt_bits),
            "subfields": bitfield_items(gt_bits),
            "subfield_boundaries": [bit_boundary_text(item) for item in sorted(gt_bit_boundaries)],
        },
        "experiment": {
            "fields": [range_text(item) for item in sorted(pred_fields)],
            "field_boundaries": [byte_boundary_text(item) for item in sorted(pred_field_boundaries)],
            "bitfields": bitfield_parent_items(pred_bits),
            "subfields": bitfield_items(pred_bits),
            "subfield_boundaries": [bit_boundary_text(item) for item in sorted(pred_bit_boundaries)],
        },
        "diff": {
            "fields_missing": [range_text(item) for item in sorted(gt_fields - pred_fields)],
            "fields_extra": [range_text(item) for item in sorted(pred_fields - gt_fields)],
            "field_boundaries_missing": [
                byte_boundary_text(item) for item in sorted(gt_field_boundaries - pred_field_boundaries)
            ],
            "field_boundaries_extra": [
                byte_boundary_text(item) for item in sorted(pred_field_boundaries - gt_field_boundaries)
            ],
            "bitfields_missing": [range_text(item) for item in sorted(set(gt_bits) - set(pred_bits))],
            "bitfields_extra": [range_text(item) for item in sorted(set(pred_bits) - set(gt_bits))],
            "subfields_missing": bitfield_items(diff_bitfields(gt_bits, pred_bits)),
            "subfields_extra": bitfield_items(diff_bitfields(pred_bits, gt_bits)),
            "subfield_boundaries_missing": [
                bit_boundary_text(item) for item in sorted(gt_bit_boundaries - pred_bit_boundaries)
            ],
            "subfield_boundaries_extra": [
                bit_boundary_text(item) for item in sorted(pred_bit_boundaries - gt_bit_boundaries)
            ],
        },
        "dropped_out_of_payload": {
            "gt_fields": [range_text(item) for item in sorted(dropped_gt_fields)],
            "experiment_fields": [range_text(item) for item in sorted(dropped_pred_fields)],
            "gt_bitfields": bitfield_items(dropped_gt_bits),
            "experiment_bitfields": bitfield_items(dropped_pred_bits),
        },
    }


def summarize_packets(protocol: str, packets: list[dict[str, Any]]) -> dict[str, Any]:
    field_acc = {"tp": 0, "fp": 0, "fn": 0}
    field_boundary_hit_acc = {"tp": 0, "fp": 0, "fn": 0}
    bit_acc = {"tp": 0, "fp": 0, "fn": 0}
    sub_acc = {"tp": 0, "fp": 0, "fn": 0}
    subfield_boundary_hit_acc = {"tp": 0, "fp": 0, "fn": 0}
    exact_match = 0
    gt_bitfields = 0
    pred_bitfields = 0

    for packet in packets:
        add_metrics(field_acc, packet["field_boundary"])
        add_metrics(field_boundary_hit_acc, packet["field_boundary_hit"])
        add_metrics(bit_acc, packet["bitfield_detection"])
        add_metrics(sub_acc, packet["bitfield_boundary"]["subfield"])
        add_metrics(subfield_boundary_hit_acc, packet["bitfield_subfield_boundary_hit"])
        exact_match += int(packet["bitfield_boundary"]["exact_match_count"])
        gt_bitfields += int(packet["bitfield_boundary"]["gt_bitfield_count"])
        pred_bitfields += int(packet["bitfield_boundary"]["pred_bitfield_count"])

    return {
        "protocol": protocol,
        "packet_count": len(packets),
        "field_boundary": SetMetrics(**field_acc).to_dict(),
        "field_boundary_hit": SetMetrics(**field_boundary_hit_acc).to_dict(),
        "bitfield_detection": SetMetrics(**bit_acc).to_dict(),
        "bitfield_boundary": {
            "gt_bitfield_count": gt_bitfields,
            "pred_bitfield_count": pred_bitfields,
            "exact_match_count": exact_match,
            "exact_match_recall": ratio(exact_match, gt_bitfields),
            "subfield": SetMetrics(**sub_acc).to_dict(),
        },
        "bitfield_subfield_boundary_hit": SetMetrics(**subfield_boundary_hit_acc).to_dict(),
    }


def build_groundtruth_readable(gt_jsonl: Path, packets_by_protocol: dict[str, list[PacketGroundtruth]]) -> str:
    lines = [
        "# Field Segmentation Groundtruth From Program Log",
        "",
        f"generated-at: {datetime.now().isoformat(timespec='seconds')}",
        f"groundtruth-jsonl: {gt_jsonl}",
        "",
        "说明：本文件只读取 LLM 基于 program log 生成的字段划分结果。每个包只输出总字段划分和位字段子字段划分，不展示字段语义描述。",
        "",
    ]
    for protocol in sorted(packets_by_protocol):
        lines.extend([f"## {protocol}", ""])
        for packet in sorted(packets_by_protocol[protocol], key=lambda item: (item.seq, item.sample_id)):
            lines.extend([f"### seq {packet.seq:06d} / {packet.sample_id}", ""])
            lines.append(f"fields: {ranges_line(packet.fields)}")
            lines.append(f"bitfields: {line(bitfield_parent_items(packet.bitfields))}")
            lines.append(f"subfields: {line(bitfield_items(packet.bitfields))}")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_compare_readable(summary: dict[str, Any], details: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Program-Log Field Segmentation Groundtruth Vs Experiment",
        "",
        f"generated-at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "每个包展示 program-log groundtruth 与实验字段恢复结果的字段边界差异，并附带位字段父字段和位字段子字段差异。",
        "这里只展示总字段划分，不展示字段语义描述。",
        "",
        "## Metrics Summary",
        "",
        "| Protocol | Packets | Field F1 | Field Boundary-Hit F1 | Bitfield F1 | Subfield F1 | Subfield Boundary-Hit F1 | Boundary Exact Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for proto in summary["protocols"]:
        lines.append(
            "| {protocol} | {packet_count} | {field_f1:.4f} | {field_boundary_hit_f1:.4f} | {bit_f1:.4f} | {sub_f1:.4f} | {subfield_boundary_hit_f1:.4f} | {exact:.4f} |".format(
                protocol=proto["protocol"],
                packet_count=proto["packet_count"],
                field_f1=proto["field_boundary"]["f1"],
                field_boundary_hit_f1=proto["field_boundary_hit"]["f1"],
                bit_f1=proto["bitfield_detection"]["f1"],
                sub_f1=proto["bitfield_boundary"]["subfield"]["f1"],
                subfield_boundary_hit_f1=proto["bitfield_subfield_boundary_hit"]["f1"],
                exact=proto["bitfield_boundary"]["exact_match_recall"],
            )
        )
    overall = summary["overall"]
    protocol_avg = summary.get("protocol_value_average")
    macro = summary.get("macro_non_empty")
    overall_packet_count = sum(int(proto["packet_count"]) for proto in summary["protocols"])
    if protocol_avg:
        lines.append(
            "| **Protocol Value Avg** | {packet_count} | {field_f1:.4f} | {field_boundary_hit_f1:.4f} | {bit_f1:.4f} | {sub_f1:.4f} | {subfield_boundary_hit_f1:.4f} | {exact:.4f} |".format(
                packet_count=overall_packet_count,
                field_f1=protocol_avg["field_boundary"]["f1"],
                field_boundary_hit_f1=protocol_avg["field_boundary_hit"]["f1"],
                bit_f1=protocol_avg["bitfield_detection"]["f1"],
                sub_f1=protocol_avg["bitfield_boundary"]["subfield"]["f1"],
                subfield_boundary_hit_f1=protocol_avg["bitfield_subfield_boundary_hit"]["f1"],
                exact=protocol_avg["bitfield_boundary"]["exact_match_recall"],
            )
        )
    lines.append(
        "| **Overall Micro** | {packet_count} | {field_f1:.4f} | {field_boundary_hit_f1:.4f} | {bit_f1:.4f} | {sub_f1:.4f} | {subfield_boundary_hit_f1:.4f} | {exact:.4f} |".format(
            packet_count=overall_packet_count,
            field_f1=overall["field_boundary"]["f1"],
            field_boundary_hit_f1=overall["field_boundary_hit"]["f1"],
            bit_f1=overall["bitfield_detection"]["f1"],
            sub_f1=overall["bitfield_boundary"]["subfield"]["f1"],
            subfield_boundary_hit_f1=overall["bitfield_subfield_boundary_hit"]["f1"],
            exact=overall["bitfield_boundary"]["exact_match_recall"],
        )
    )
    if macro:
        lines.append(
            "| **Macro Non-Empty** | {packet_count} | {field_f1:.4f} | {field_boundary_hit_f1:.4f} | {bit_f1:.4f} | {sub_f1:.4f} | {subfield_boundary_hit_f1:.4f} | {exact:.4f} |".format(
                packet_count=overall_packet_count,
                field_f1=macro["field_boundary"]["f1"],
                field_boundary_hit_f1=macro["field_boundary_hit"]["f1"],
                bit_f1=macro["bitfield_detection"]["f1"],
                sub_f1=macro["bitfield_boundary"]["subfield"]["f1"],
                subfield_boundary_hit_f1=macro["bitfield_subfield_boundary_hit"]["f1"],
                exact=macro["bitfield_boundary"]["exact_match_recall"],
            )
        )
    lines.extend(
        [
            "",
            "说明：`Protocol Value Avg` 是直接对上方协议表中的指标值求平均，每个指标只除以该指标有有效分母的协议数；`Overall Micro` 是先汇总所有协议的 TP/FP/FN 后再计算；`Macro Non-Empty` 是协议级宏平均，先平均非空协议的 precision/recall，再由宏平均 P/R 计算 F1。",
            "",
        ]
    )

    for protocol, packets in details.items():
        lines.extend([f"## {protocol}", ""])
        for packet in packets:
            if "error" in packet:
                lines.extend(
                    [
                        f"### seq {packet.get('seq', 0):06d} / {packet['packet_dir']}",
                        "",
                        f"error: {packet['error']}",
                        "",
                    ]
                )
                continue
            lines.extend(
                [
                    f"### seq {packet['seq']:06d} / {packet['packet_dir']}",
                    "",
                    f"payload-len: {packet['payload_len']}",
                    "packet metrics: field={field}; field-boundary-hit={field_hit}; bitfield={bit}; subfield={subfield}; subfield-boundary-hit={subfield_hit}".format(
                        field=metric_short(packet["field_boundary"]),
                        field_hit=metric_short(packet["field_boundary_hit"]),
                        bit=metric_short(packet["bitfield_detection"]),
                        subfield=metric_short(packet["bitfield_boundary"]["subfield"]),
                        subfield_hit=metric_short(packet["bitfield_subfield_boundary_hit"]),
                    ),
                    f"GT fields: {line(packet['gt']['fields'])}",
                    f"EXP fields: {line(packet['experiment']['fields'])}",
                    f"fields missing: {line(packet['diff']['fields_missing'])}",
                    f"fields extra: {line(packet['diff']['fields_extra'])}",
                    f"GT field boundaries: {line(packet['gt']['field_boundaries'])}",
                    f"EXP field boundaries: {line(packet['experiment']['field_boundaries'])}",
                    f"field boundaries missing: {line(packet['diff']['field_boundaries_missing'])}",
                    f"field boundaries extra: {line(packet['diff']['field_boundaries_extra'])}",
                    f"GT bitfields: {line(packet['gt']['bitfields'])}",
                    f"EXP bitfields: {line(packet['experiment']['bitfields'])}",
                    f"bitfields missing: {line(packet['diff']['bitfields_missing'])}",
                    f"bitfields extra: {line(packet['diff']['bitfields_extra'])}",
                    f"GT subfields: {line(packet['gt']['subfields'])}",
                    f"EXP subfields: {line(packet['experiment']['subfields'])}",
                    f"subfields missing: {line(packet['diff']['subfields_missing'])}",
                    f"subfields extra: {line(packet['diff']['subfields_extra'])}",
                    f"GT subfield boundaries: {line(packet['gt']['subfield_boundaries'])}",
                    f"EXP subfield boundaries: {line(packet['experiment']['subfield_boundaries'])}",
                    f"subfield boundaries missing: {line(packet['diff']['subfield_boundaries_missing'])}",
                    f"subfield boundaries extra: {line(packet['diff']['subfield_boundaries_extra'])}",
                    "",
                ]
            )
            dropped = packet["dropped_out_of_payload"]
            if any(dropped.values()):
                lines.extend(
                    [
                        f"dropped gt fields: {line(dropped['gt_fields'])}",
                        f"dropped exp fields: {line(dropped['experiment_fields'])}",
                        f"dropped gt bitfields: {line(dropped['gt_bitfields'])}",
                        f"dropped exp bitfields: {line(dropped['experiment_bitfields'])}",
                        "",
                    ]
                )
    return "\n".join(lines) + "\n"


def packet_score(packet: dict[str, Any]) -> float:
    return (
        packet["field_boundary_hit"]["f1"]
        + packet["field_boundary"]["f1"]
        + packet["bitfield_detection"]["f1"]
        + packet["bitfield_boundary"]["subfield"]["f1"]
        + packet["bitfield_subfield_boundary_hit"]["f1"]
    ) / 5


def build_packet_metrics(details: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for protocol, packets in details.items():
        result[protocol] = []
        for packet in packets:
            if "error" in packet:
                result[protocol].append(
                    {
                        "seq": packet.get("seq", 0),
                        "packet_dir": packet["packet_dir"],
                        "error": packet["error"],
                    }
                )
                continue
            result[protocol].append(
                {
                    "seq": packet["seq"],
                    "packet_dir": packet["packet_dir"],
                    "payload_len": packet["payload_len"],
                    "score": packet_score(packet),
                    "field_boundary": packet["field_boundary"],
                    "field_boundary_hit": packet["field_boundary_hit"],
                    "bitfield_detection": packet["bitfield_detection"],
                    "bitfield_subfield": packet["bitfield_boundary"]["subfield"],
                    "bitfield_subfield_boundary_hit": packet["bitfield_subfield_boundary_hit"],
                }
            )
    return result


def build_packet_metrics_markdown(packet_metrics: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Program-Log Per-Packet Field Boundary Metrics",
        "",
        f"generated-at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "每个数据包单独计算指标，并在每个协议内按 `Score` 升序排列。`Score` 是五个 F1 的简单平均：字段整段、字段边界命中、位字段检测、位子字段整段、位子字段边界命中。",
        "",
    ]
    for protocol, packets in packet_metrics.items():
        lines.extend([f"## {protocol}", ""])
        valid = [packet for packet in packets if "error" not in packet]
        if valid:
            best = max(valid, key=lambda item: item["score"])
            worst = min(valid, key=lambda item: item["score"])
            lines.append(
                "best: {pkt} score={score:.4f}; worst: {worst_pkt} score={worst_score:.4f}".format(
                    pkt=best["packet_dir"],
                    score=best["score"],
                    worst_pkt=worst["packet_dir"],
                    worst_score=worst["score"],
                )
            )
            lines.append("")
        lines.extend(
            [
                "| Seq | Packet | Len | Score | Field F1 | Field Boundary-Hit F1 | Bitfield F1 | Subfield F1 | Subfield Boundary-Hit F1 |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        sorted_packets = sorted(
            packets,
            key=lambda item: (
                float("inf") if "error" in item else item["score"],
                item.get("seq", 0),
                item["packet_dir"],
            ),
        )
        for packet in sorted_packets:
            if "error" in packet:
                lines.append(
                    f"| {packet.get('seq', 0)} | {packet['packet_dir']} | - | - | error: {packet['error']} | - | - | - | - |"
                )
                continue
            lines.append(
                "| {seq} | {packet} | {length} | {score:.4f} | {field:.4f} | {field_hit:.4f} | {bit:.4f} | {subfield:.4f} | {subfield_hit:.4f} |".format(
                    seq=packet["seq"],
                    packet=packet["packet_dir"],
                    length=packet["payload_len"],
                    score=packet["score"],
                    field=packet["field_boundary"]["f1"],
                    field_hit=packet["field_boundary_hit"]["f1"],
                    bit=packet["bitfield_detection"]["f1"],
                    subfield=packet["bitfield_subfield"]["f1"],
                    subfield_hit=packet["bitfield_subfield_boundary_hit"]["f1"],
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    overall = summary["overall"]
    protocol_avg = summary.get("protocol_value_average")
    macro = summary.get("macro_non_empty")
    lines = [
        "# Program-Log Field Boundary Evaluation Metrics",
        "",
        "指标均不计算 TN；负例全集不能从实验结果或 LLM groundtruth 中可靠枚举。",
        "",
        "## Protocol Value Avg",
        "",
    ]
    if protocol_avg:
        lines.extend(
            [
                "该行直接对协议表中的结果求平均；每个指标只除以该指标有有效分母的协议数。",
                "",
                f"- Field boundary: f1={protocol_avg['field_boundary']['f1']:.4f}, protocols={protocol_avg['field_boundary']['protocol_count']}",
                f"- Field boundary-hit: f1={protocol_avg['field_boundary_hit']['f1']:.4f}, protocols={protocol_avg['field_boundary_hit']['protocol_count']}",
                f"- Bitfield detection: f1={protocol_avg['bitfield_detection']['f1']:.4f}, protocols={protocol_avg['bitfield_detection']['protocol_count']}",
                f"- Bitfield boundary subfield: f1={protocol_avg['bitfield_boundary']['subfield']['f1']:.4f}, protocols={protocol_avg['bitfield_boundary']['subfield']['protocol_count']}",
                f"- Bitfield subfield boundary-hit: f1={protocol_avg['bitfield_subfield_boundary_hit']['f1']:.4f}, protocols={protocol_avg['bitfield_subfield_boundary_hit']['protocol_count']}",
                f"- Bitfield boundary exact recall: {protocol_avg['bitfield_boundary']['exact_match_recall']:.4f}, protocols={protocol_avg['bitfield_boundary']['protocol_count']}",
                "",
            ]
        )
    else:
        lines.extend(["- unavailable", ""])
    lines.extend(
        [
        "## Overall Micro",
        "",
        f"- Field boundary: precision={overall['field_boundary']['precision']:.4f}, recall={overall['field_boundary']['recall']:.4f}, f1={overall['field_boundary']['f1']:.4f}, jaccard={overall['field_boundary']['jaccard']:.4f}",
        f"- Field boundary-hit: precision={overall['field_boundary_hit']['precision']:.4f}, recall={overall['field_boundary_hit']['recall']:.4f}, f1={overall['field_boundary_hit']['f1']:.4f}, jaccard={overall['field_boundary_hit']['jaccard']:.4f}",
        f"- Bitfield detection: precision={overall['bitfield_detection']['precision']:.4f}, recall={overall['bitfield_detection']['recall']:.4f}, f1={overall['bitfield_detection']['f1']:.4f}, jaccard={overall['bitfield_detection']['jaccard']:.4f}",
        f"- Bitfield boundary subfield: precision={overall['bitfield_boundary']['subfield']['precision']:.4f}, recall={overall['bitfield_boundary']['subfield']['recall']:.4f}, f1={overall['bitfield_boundary']['subfield']['f1']:.4f}",
        f"- Bitfield subfield boundary-hit: precision={overall['bitfield_subfield_boundary_hit']['precision']:.4f}, recall={overall['bitfield_subfield_boundary_hit']['recall']:.4f}, f1={overall['bitfield_subfield_boundary_hit']['f1']:.4f}, jaccard={overall['bitfield_subfield_boundary_hit']['jaccard']:.4f}",
        f"- Bitfield boundary exact recall: {overall['bitfield_boundary']['exact_match_recall']:.4f} ({overall['bitfield_boundary']['exact_match_count']}/{overall['bitfield_boundary']['gt_bitfield_count']})",
        "",
        "## Macro Non-Empty",
        "",
        ]
    )
    if macro:
        lines.extend(
            [
                "该行是 protocol-level macro average；每个指标只平均该指标有有效分母的协议，并由宏平均 precision/recall 重新计算 F1，避免没有位字段的协议把 bitfield 指标拉成 0。",
                "",
                f"- Field boundary: f1={macro['field_boundary']['f1']:.4f}, protocols={macro['field_boundary']['protocol_count']}",
                f"- Field boundary-hit: f1={macro['field_boundary_hit']['f1']:.4f}, protocols={macro['field_boundary_hit']['protocol_count']}",
                f"- Bitfield detection: f1={macro['bitfield_detection']['f1']:.4f}, protocols={macro['bitfield_detection']['protocol_count']}",
                f"- Bitfield boundary subfield: f1={macro['bitfield_boundary']['subfield']['f1']:.4f}, protocols={macro['bitfield_boundary']['subfield']['protocol_count']}",
                f"- Bitfield subfield boundary-hit: f1={macro['bitfield_subfield_boundary_hit']['f1']:.4f}, protocols={macro['bitfield_subfield_boundary_hit']['protocol_count']}",
                f"- Bitfield boundary exact recall: {macro['bitfield_boundary']['exact_match_recall']:.4f}, protocols={macro['bitfield_boundary']['protocol_count']}",
                "",
            ]
        )
    else:
        lines.extend(["- unavailable", ""])
    lines.extend(
        [
        "## Per Protocol",
        "",
        "| Protocol | Packets | Field F1 | Field Boundary-Hit F1 | Bit F1 | Subfield F1 | Subfield Boundary-Hit F1 | Exact Boundary R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for proto in summary["protocols"]:
        lines.append(
            "| {protocol} | {packet_count} | {ff:.4f} | {fbhf:.4f} | {bf:.4f} | {sf:.4f} | {sbhf:.4f} | {er:.4f} |".format(
                protocol=proto["protocol"],
                packet_count=proto["packet_count"],
                ff=proto["field_boundary"]["f1"],
                fbhf=proto["field_boundary_hit"]["f1"],
                bf=proto["bitfield_detection"]["f1"],
                sf=proto["bitfield_boundary"]["subfield"]["f1"],
                sbhf=proto["bitfield_subfield_boundary_hit"]["f1"],
                er=proto["bitfield_boundary"]["exact_match_recall"],
            )
        )
    overall_packet_count = sum(int(proto["packet_count"]) for proto in summary["protocols"])
    if protocol_avg:
        lines.append(
            "| **Protocol Value Avg** | {packet_count} | {ff:.4f} | {fbhf:.4f} | {bf:.4f} | {sf:.4f} | {sbhf:.4f} | {er:.4f} |".format(
                packet_count=overall_packet_count,
                ff=protocol_avg["field_boundary"]["f1"],
                fbhf=protocol_avg["field_boundary_hit"]["f1"],
                bf=protocol_avg["bitfield_detection"]["f1"],
                sf=protocol_avg["bitfield_boundary"]["subfield"]["f1"],
                sbhf=protocol_avg["bitfield_subfield_boundary_hit"]["f1"],
                er=protocol_avg["bitfield_boundary"]["exact_match_recall"],
            )
        )
    lines.append(
        "| **Overall Micro** | {packet_count} | {ff:.4f} | {fbhf:.4f} | {bf:.4f} | {sf:.4f} | {sbhf:.4f} | {er:.4f} |".format(
            packet_count=overall_packet_count,
            ff=overall["field_boundary"]["f1"],
            fbhf=overall["field_boundary_hit"]["f1"],
            bf=overall["bitfield_detection"]["f1"],
            sf=overall["bitfield_boundary"]["subfield"]["f1"],
            sbhf=overall["bitfield_subfield_boundary_hit"]["f1"],
            er=overall["bitfield_boundary"]["exact_match_recall"],
        )
    )
    if macro:
        lines.append(
            "| **Macro Non-Empty** | {packet_count} | {ff:.4f} | {fbhf:.4f} | {bf:.4f} | {sf:.4f} | {sbhf:.4f} | {er:.4f} |".format(
                packet_count=overall_packet_count,
                ff=macro["field_boundary"]["f1"],
                fbhf=macro["field_boundary_hit"]["f1"],
                bf=macro["bitfield_detection"]["f1"],
                sf=macro["bitfield_boundary"]["subfield"]["f1"],
                sbhf=macro["bitfield_subfield_boundary_hit"]["f1"],
                er=macro["bitfield_boundary"]["exact_match_recall"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_has_denominator(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("tp", 0)) + int(metrics.get("fp", 0)) + int(metrics.get("fn", 0)) > 0


def average_metric_value(protocols: list[dict[str, Any]], path: tuple[str, ...], value_key: str = "f1") -> dict[str, Any]:
    values: list[float] = []
    for proto in protocols:
        metric: Any = proto
        for key in path:
            metric = metric[key]
        if metric_has_denominator(metric):
            values.append(float(metric[value_key]))
    return {
        value_key: 0.0 if not values else sum(values) / len(values),
        "protocol_count": len(values),
    }


def average_metric(protocols: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for proto in protocols:
        metric: Any = proto
        for key in path:
            metric = metric[key]
        if metric_has_denominator(metric):
            values.append(metric)
    if not values:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "average_f1": 0.0,
            "jaccard": 0.0,
            "protocol_count": 0,
        }
    precision = sum(float(item["precision"]) for item in values) / len(values)
    recall = sum(float(item["recall"]) for item in values) / len(values)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "average_f1": sum(float(item["f1"]) for item in values) / len(values),
        "jaccard": sum(float(item.get("jaccard", 0.0)) for item in values) / len(values),
        "protocol_count": len(values),
    }


def average_exact_recall(protocols: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        proto["bitfield_boundary"]["exact_match_recall"]
        for proto in protocols
        if int(proto["bitfield_boundary"]["gt_bitfield_count"]) > 0
    ]
    return {
        "exact_match_recall": 0.0 if not values else sum(float(value) for value in values) / len(values),
        "protocol_count": len(values),
    }


def build_macro_non_empty(protocols: list[dict[str, Any]]) -> dict[str, Any]:
    exact = average_exact_recall(protocols)
    return {
        "field_boundary": average_metric(protocols, ("field_boundary",)),
        "field_boundary_hit": average_metric(protocols, ("field_boundary_hit",)),
        "bitfield_detection": average_metric(protocols, ("bitfield_detection",)),
        "bitfield_boundary": {
            "exact_match_recall": exact["exact_match_recall"],
            "protocol_count": exact["protocol_count"],
            "subfield": average_metric(protocols, ("bitfield_boundary", "subfield")),
        },
        "bitfield_subfield_boundary_hit": average_metric(protocols, ("bitfield_subfield_boundary_hit",)),
    }


def build_protocol_value_average(protocols: list[dict[str, Any]]) -> dict[str, Any]:
    exact = average_exact_recall(protocols)
    return {
        "field_boundary": average_metric_value(protocols, ("field_boundary",)),
        "field_boundary_hit": average_metric_value(protocols, ("field_boundary_hit",)),
        "bitfield_detection": average_metric_value(protocols, ("bitfield_detection",)),
        "bitfield_boundary": {
            "exact_match_recall": exact["exact_match_recall"],
            "protocol_count": exact["protocol_count"],
            "subfield": average_metric_value(protocols, ("bitfield_boundary", "subfield")),
        },
        "bitfield_subfield_boundary_hit": average_metric_value(protocols, ("bitfield_subfield_boundary_hit",)),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    packets = load_program_log_groundtruth(args.groundtruth_jsonl)
    packets_by_protocol: dict[str, list[PacketGroundtruth]] = defaultdict(list)
    for packet in packets.values():
        packets_by_protocol[packet.protocol].append(packet)

    all_details: dict[str, list[dict[str, Any]]] = {}
    protocol_summaries: list[dict[str, Any]] = []
    for protocol in sorted(packets_by_protocol):
        details = [
            compare_one_packet(packet, args.replay_root)
            for packet in sorted(packets_by_protocol[protocol], key=lambda item: (item.seq, item.sample_id))
        ]
        all_details[protocol] = details
        protocol_summaries.append(
            summarize_packets(protocol, [item for item in details if "error" not in item])
        )

    overall_acc = {
        "field_boundary": {"tp": 0, "fp": 0, "fn": 0},
        "field_boundary_hit": {"tp": 0, "fp": 0, "fn": 0},
        "bitfield_detection": {"tp": 0, "fp": 0, "fn": 0},
        "subfield": {"tp": 0, "fp": 0, "fn": 0},
        "subfield_boundary_hit": {"tp": 0, "fp": 0, "fn": 0},
        "exact_match_count": 0,
        "gt_bitfield_count": 0,
        "pred_bitfield_count": 0,
    }
    for proto in protocol_summaries:
        add_metrics(overall_acc["field_boundary"], proto["field_boundary"])
        add_metrics(overall_acc["field_boundary_hit"], proto["field_boundary_hit"])
        add_metrics(overall_acc["bitfield_detection"], proto["bitfield_detection"])
        add_metrics(overall_acc["subfield"], proto["bitfield_boundary"]["subfield"])
        add_metrics(overall_acc["subfield_boundary_hit"], proto["bitfield_subfield_boundary_hit"])
        overall_acc["exact_match_count"] += int(proto["bitfield_boundary"]["exact_match_count"])
        overall_acc["gt_bitfield_count"] += int(proto["bitfield_boundary"]["gt_bitfield_count"])
        overall_acc["pred_bitfield_count"] += int(proto["bitfield_boundary"]["pred_bitfield_count"])

    summary = {
        "source_groundtruth_jsonl": str(args.groundtruth_jsonl),
        "source_replay_root": str(args.replay_root),
        "protocols": protocol_summaries,
        "protocol_value_average": build_protocol_value_average(protocol_summaries),
        "macro_non_empty": build_macro_non_empty(protocol_summaries),
        "overall": {
            "field_boundary": SetMetrics(**overall_acc["field_boundary"]).to_dict(),
            "field_boundary_hit": SetMetrics(**overall_acc["field_boundary_hit"]).to_dict(),
            "bitfield_detection": SetMetrics(**overall_acc["bitfield_detection"]).to_dict(),
            "bitfield_boundary": {
                "gt_bitfield_count": overall_acc["gt_bitfield_count"],
                "pred_bitfield_count": overall_acc["pred_bitfield_count"],
                "exact_match_count": overall_acc["exact_match_count"],
                "exact_match_recall": ratio(overall_acc["exact_match_count"], overall_acc["gt_bitfield_count"]),
                "subfield": SetMetrics(**overall_acc["subfield"]).to_dict(),
            },
            "bitfield_subfield_boundary_hit": SetMetrics(**overall_acc["subfield_boundary_hit"]).to_dict(),
        },
    }

    (args.outdir / "field_boundary_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.outdir / "field_boundary_packet_details.json").write_text(
        json.dumps(all_details, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    packet_metrics = build_packet_metrics(all_details)
    (args.outdir / "field_boundary_packet_metrics.json").write_text(
        json.dumps(packet_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.outdir / "field_boundary_packet_metrics.md").write_text(
        build_packet_metrics_markdown(packet_metrics),
        encoding="utf-8",
    )
    write_summary_markdown(summary, args.outdir / "field_boundary_metrics_summary.md")
    args.groundtruth_md.write_text(
        build_groundtruth_readable(args.groundtruth_jsonl, packets_by_protocol),
        encoding="utf-8",
    )
    args.compare_md.write_text(build_compare_readable(summary, all_details), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
