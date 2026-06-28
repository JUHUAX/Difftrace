#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_GT_ROOT = ROOT / "groundtruth_from_tshark"
DEFAULT_REPLAY_ROOT = ROOT / "replay_manual_latest" / "outputs"
DEFAULT_OUTDIR = ROOT / "evaluation_from_tshark" / "field_boundary"
DEFAULT_GT_READABLE = DEFAULT_OUTDIR / "groundtruth_readable.md"
DEFAULT_COMPARE_READABLE = DEFAULT_OUTDIR / "groundtruth_vs_experiment_readable.md"

GT_FILE_BY_PROTOCOL = {
    "bacnet": "bacnet.json",
    "cip": "cip.json",
    "iec104": "iec104.json",
    "iec61850": "mms.json",
    "modbus": "modbus.json",
    "snap7": "S7server.json",
}

# Replay pcaps and tshark groundtruth pcaps are normally the same packet stream.
# IEC61850 is the exception: replay uses a direct-MMS pcap stripped from the
# original ISO stack pcap, and the first ISO packet has no MMS payload. Therefore
# replay pkt_0000 maps to tshark groundtruth packet 2.
GT_PACKET_INDEX_OFFSET = {
    "iec61850": 1,
}

ByteRange = Tuple[int, int]
BitLabel = str
BitfieldMap = Dict[ByteRange, Set[BitLabel]]
ByteBoundary = int
BitBoundary = Tuple[ByteRange, int]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def normalize_label(label: str) -> str:
    label = str(label).strip()
    if not (label.startswith("[") and label.endswith("]")):
        return label
    inner = label[1:-1]
    if ":" in inner:
        hi, lo = inner.split(":", 1)
        return f"[{int(hi)}:{int(lo)}]"
    return f"[{int(inner)}]"


def label_sort_key(label: str) -> Tuple[int, int, str]:
    label = normalize_label(label)
    inner = label[1:-1] if label.startswith("[") and label.endswith("]") else label
    if ":" in inner:
        hi, lo = inner.split(":", 1)
        return (int(lo), int(hi), label)
    if inner.isdigit():
        bit = int(inner)
        return (bit, bit, label)
    return (9999, 9999, label)


def bit_label_bounds(label: str) -> Optional[Tuple[int, int]]:
    label = normalize_label(label)
    inner = label[1:-1] if label.startswith("[") and label.endswith("]") else label
    if ":" in inner:
        hi, lo = inner.split(":", 1)
        return int(lo), int(hi)
    if inner.isdigit():
        bit = int(inner)
        return bit, bit
    return None


def parse_range_repr(text: str) -> ByteRange:
    text = str(text).strip()
    if "," in text:
        parts = [int(x) for x in text.split(",") if x.strip()]
        return min(parts), max(parts)
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    value = int(text)
    return value, value


def range_text(byte_range: ByteRange) -> str:
    a, b = byte_range
    if a == b:
        return f"[{a}]"
    return "[" + ",".join(str(i) for i in range(a, b + 1)) + "]"


def bitfield_items(bitfields: BitfieldMap) -> List[str]:
    items: List[str] = []
    for byte_range in sorted(bitfields):
        for label in sorted(bitfields[byte_range], key=label_sort_key):
            items.append(f"{range_text(byte_range)}{label}")
    return items


def ranges_line(ranges: Iterable[ByteRange]) -> str:
    return line(range_text(item) for item in sorted(ranges))


def byte_boundary_text(boundary: ByteBoundary) -> str:
    return f"B{boundary}"


def bit_boundary_text(boundary: BitBoundary) -> str:
    byte_range, bit_boundary = boundary
    return f"{range_text(byte_range)}B{bit_boundary}"


def line(items: Iterable[str]) -> str:
    values = list(items)
    return " ".join(values) if values else "-"


def metric_short(metrics: dict) -> str:
    return "P={precision:.4f} R={recall:.4f} F1={f1:.4f}".format(**metrics)


def protocol_order(packet: PacketView) -> List[str]:
    ordered: List[str] = []
    for proto in packet.selected_protocols:
        if proto not in ordered:
            ordered.append(proto)
    for proto in sorted(set(packet.protocol_spans) | set(packet.fields_by_protocol) | set(packet.bitfields_by_protocol)):
        if proto not in ordered:
            ordered.append(proto)
    return ordered


def protocol_span_text(packet: PacketView, proto: str) -> str:
    span = packet.protocol_spans.get(proto)
    return range_text(span) if span else "-"


def format_by_protocol(values: Dict[str, List[str]]) -> str:
    parts = []
    for proto, items in values.items():
        parts.append(f"{proto}: {line(items)}")
    return " | ".join(parts) if parts else "-"


def payload_base(packet: dict) -> Optional[int]:
    protocol_offsets = [
        int(proto["offset"])
        for proto in packet.get("protocols", [])
        if proto.get("offset") is not None
    ]
    if protocol_offsets:
        return min(protocol_offsets)

    field_offsets = [
        int(field["field_offset"])
        for field in packet.get("fields", [])
        if field.get("field_offset") is not None
    ]
    if field_offsets:
        return min(field_offsets)
    return None


def is_contained(inner: ByteRange, outer: ByteRange) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def subtract_ranges(base_range: ByteRange, cutters: Iterable[ByteRange]) -> Set[ByteRange]:
    remaining = [base_range]
    for cutter in sorted(cutters):
        next_remaining: List[ByteRange] = []
        for start, end in remaining:
            if cutter[1] < start or cutter[0] > end:
                next_remaining.append((start, end))
                continue
            if start < cutter[0]:
                next_remaining.append((start, cutter[0] - 1))
            if cutter[1] < end:
                next_remaining.append((cutter[1] + 1, end))
        remaining = next_remaining
    return set(remaining)


def normalize_field_ranges(
    leaf_ranges: Set[ByteRange],
    container_ranges: Set[ByteRange],
) -> Set[ByteRange]:
    """Convert tshark's nested field tree into non-overlapping byte fields."""
    result = set(leaf_ranges)
    all_known = set(leaf_ranges) | set(container_ranges)

    for container in container_ranges:
        inner = {
            item
            for item in all_known
            if item != container and is_contained(item, container)
        }
        result.update(subtract_ranges(container, inner))

    return atomize_ranges({item for item in result if item[0] <= item[1]})


def atomize_ranges(ranges: Set[ByteRange]) -> Set[ByteRange]:
    """Split overlapping/containing ranges into a non-overlapping partition."""
    if not ranges:
        return set()
    points = sorted({point for start, end in ranges for point in (start, end + 1)})
    atoms: Set[ByteRange] = set()
    for left, right_exclusive in zip(points, points[1:]):
        atom = (left, right_exclusive - 1)
        if atom[0] > atom[1]:
            continue
        if any(is_contained(atom, byte_range) for byte_range in ranges):
            atoms.add(atom)
    return atoms


def in_protocol_span(byte_range: ByteRange, protocol: str, spans: Dict[str, ByteRange]) -> bool:
    span = spans.get(protocol)
    return span is None or is_contained(byte_range, span)


@dataclass
class PacketView:
    packet_index: int
    protocol_spans: Dict[str, ByteRange]
    selected_protocols: List[str]
    fields: Set[ByteRange]
    fields_by_protocol: Dict[str, Set[ByteRange]]
    bitfields: BitfieldMap
    bitfields_by_protocol: Dict[str, BitfieldMap]
    payload_base: Optional[int]

    @property
    def subfield_pairs(self) -> Set[Tuple[ByteRange, BitLabel]]:
        return {
            (byte_range, label)
            for byte_range, labels in self.bitfields.items()
            for label in labels
        }


def parse_tshark_packet(packet: dict) -> PacketView:
    base = payload_base(packet)
    protocol_spans: Dict[str, ByteRange] = {}
    selected_protocols = list(packet.get("selected_protocols", []))
    leaf_ranges: Set[ByteRange] = set()
    container_ranges: Set[ByteRange] = set()
    leaf_by_protocol: Dict[str, Set[ByteRange]] = defaultdict(set)
    container_by_protocol: Dict[str, Set[ByteRange]] = defaultdict(set)
    bitfields: BitfieldMap = defaultdict(set)
    bitfields_by_protocol: Dict[str, BitfieldMap] = defaultdict(lambda: defaultdict(set))

    if base is not None:
        for proto in packet.get("protocols", []):
            offset = proto.get("offset")
            length = proto.get("length")
            name = proto.get("protocol")
            if name is None or offset is None or length is None:
                continue
            start = int(offset) - base
            end = start + int(length) - 1
            if start >= 0 and end >= start:
                protocol_spans[str(name)] = (start, end)

    for field in packet.get("fields", []):
        offset = field.get("field_offset")
        length = field.get("field_length")
        field_protocol = str(field.get("protocol") or "unknown")
        if base is None or offset is None or length is None:
            continue
        start = int(offset) - base
        end = start + int(length) - 1
        if start < 0 or end < start:
            continue
        byte_range = (start, end)
        if not in_protocol_span(byte_range, field_protocol, protocol_spans):
            continue
        if field.get("has_children"):
            container_ranges.add(byte_range)
            container_by_protocol[field_protocol].add(byte_range)
        else:
            leaf_ranges.add(byte_range)
            leaf_by_protocol[field_protocol].add(byte_range)
        for label in field.get("bit_ranges") or []:
            normalized = normalize_label(label)
            bitfields[byte_range].add(normalized)
            bitfields_by_protocol[field_protocol][byte_range].add(normalized)

    ranges = normalize_field_ranges(leaf_ranges, container_ranges)
    ranges_by_protocol = {
        protocol: normalize_field_ranges(
            leaf_by_protocol.get(protocol, set()),
            container_by_protocol.get(protocol, set()),
        )
        for protocol in set(leaf_by_protocol) | set(container_by_protocol)
    }

    return PacketView(
        packet_index=int(packet["packet_index"]),
        protocol_spans=protocol_spans,
        selected_protocols=selected_protocols,
        fields=ranges,
        fields_by_protocol={key: set(value) for key, value in ranges_by_protocol.items()},
        bitfields={key: set(value) for key, value in bitfields.items()},
        bitfields_by_protocol={
            proto: {key: set(value) for key, value in mapping.items()}
            for proto, mapping in bitfields_by_protocol.items()
        },
        payload_base=base,
    )


def load_tshark_groundtruth(gt_root: Path, protocol: str) -> Dict[int, PacketView]:
    path = gt_root / GT_FILE_BY_PROTOCOL[protocol]
    obj = load_json(path)
    return {
        int(packet["packet_index"]): parse_tshark_packet(packet)
        for packet in obj.get("packets", [])
    }


def parse_experiment_fields(path: Path) -> Set[ByteRange]:
    if not path.exists():
        return set()
    obj = load_json(path)
    ranges: Set[ByteRange] = set()
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


def gt_packet_index_from_meta(protocol: str, pkt_dir: Path) -> int:
    offset = GT_PACKET_INDEX_OFFSET.get(protocol, 0)
    meta_path = pkt_dir / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        if "pcap_packet_index" in meta:
            return int(meta["pcap_packet_index"]) + 1 + offset
    return int(pkt_dir.name.removeprefix("pkt_")) + 1 + offset


def in_payload(byte_range: ByteRange, payload_len: int) -> bool:
    return payload_len <= 0 or (0 <= byte_range[0] <= byte_range[1] < payload_len)


def filter_ranges(ranges: Set[ByteRange], payload_len: int) -> Tuple[Set[ByteRange], Set[ByteRange]]:
    kept = {item for item in ranges if in_payload(item, payload_len)}
    return kept, ranges - kept


def filter_bitfields(bitfields: BitfieldMap, payload_len: int) -> Tuple[BitfieldMap, BitfieldMap]:
    kept: BitfieldMap = {}
    dropped: BitfieldMap = {}
    for byte_range, labels in bitfields.items():
        if in_payload(byte_range, payload_len):
            kept[byte_range] = set(labels)
        else:
            dropped[byte_range] = set(labels)
    return kept, dropped


def with_bitfield_parent_ranges(fields: Set[ByteRange], bitfields: BitfieldMap) -> Set[ByteRange]:
    """Byte-granularity evaluation projects bit subfields to covered byte ranges."""
    return set(fields) | set(bitfields)


@dataclass
class SetMetrics:
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
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


def set_metrics(gt: Set, pred: Set) -> SetMetrics:
    return SetMetrics(
        tp=len(gt & pred),
        fp=len(pred - gt),
        fn=len(gt - pred),
    )


def add_metrics(acc: dict, metrics: dict) -> None:
    for key in ("tp", "fp", "fn"):
        acc[key] += int(metrics[key])


def field_boundaries(ranges: Iterable[ByteRange]) -> Set[ByteBoundary]:
    boundaries: Set[ByteBoundary] = set()
    for start, end in ranges:
        boundaries.add(start)
        boundaries.add(end + 1)
    return boundaries


def bitfield_subfield_boundaries(bitfields: BitfieldMap) -> Set[BitBoundary]:
    boundaries: Set[BitBoundary] = set()
    for byte_range, labels in bitfields.items():
        for label in labels:
            bounds = bit_label_bounds(label)
            if bounds is None:
                continue
            lo, hi = bounds
            boundaries.add((byte_range, lo))
            boundaries.add((byte_range, hi + 1))
    return boundaries


def boundary_metrics(gt: BitfieldMap, pred: BitfieldMap) -> dict:
    gt_ranges = set(gt)
    pred_ranges = set(pred)
    exact_gt = sum(1 for byte_range in gt_ranges if pred.get(byte_range, set()) == gt[byte_range])
    exact_pred = sum(1 for byte_range in pred_ranges if gt.get(byte_range, set()) == pred[byte_range])
    gt_pairs = {
        (byte_range, label)
        for byte_range, labels in gt.items()
        for label in labels
    }
    pred_pairs = {
        (byte_range, label)
        for byte_range, labels in pred.items()
        for label in labels
    }
    pair = set_metrics(gt_pairs, pred_pairs).to_dict()
    return {
        "gt_bitfield_count": len(gt_ranges),
        "pred_bitfield_count": len(pred_ranges),
        "exact_match_count": exact_gt,
        "exact_match_recall": ratio(exact_gt, len(gt_ranges)),
        "exact_match_precision": ratio(exact_pred, len(pred_ranges)),
        "subfield": pair,
    }


def packet_dirs(replay_root: Path, protocol: str) -> List[Path]:
    proto_dir = replay_root / protocol
    if not proto_dir.is_dir():
        return []
    return sorted(
        [path for path in proto_dir.iterdir() if path.is_dir() and path.name.startswith("pkt_")],
        key=lambda path: path.name,
    )


def compare_one_packet(gt: PacketView, pkt_dir: Path) -> dict:
    payload_len = payload_len_from_meta(pkt_dir)
    pred_bits, dropped_pred_bits = filter_bitfields(
        parse_experiment_bitfields(pkt_dir / "bitfields.json"),
        payload_len,
    )
    gt_bits, dropped_gt_bits = filter_bitfields(gt.bitfields, payload_len)
    pred_fields, dropped_pred_fields = filter_ranges(
        with_bitfield_parent_ranges(parse_experiment_fields(pkt_dir / "fields.json"), pred_bits),
        payload_len,
    )
    gt_fields, dropped_gt_fields = filter_ranges(
        with_bitfield_parent_ranges(set(gt.fields), gt_bits),
        payload_len,
    )

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
        "packet_dir": pkt_dir.name,
        "groundtruth_packet_index": gt.packet_index,
        "payload_len": payload_len,
        "field_boundary": field,
        "field_boundary_hit": field_boundary_hit,
        "bitfield_detection": bitfield,
        "bitfield_boundary": boundary,
        "bitfield_subfield_boundary_hit": subfield_boundary_hit,
        "gt": {
            "protocol_spans": {
                proto: protocol_span_text(gt, proto)
                for proto in protocol_order(gt)
            },
            "fields_by_protocol": {
                proto: [range_text(item) for item in sorted(gt.fields_by_protocol.get(proto, set()))]
                for proto in protocol_order(gt)
            },
            "bitfields_by_protocol": {
                proto: bitfield_items(gt.bitfields_by_protocol.get(proto, {}))
                for proto in protocol_order(gt)
            },
            "fields": [range_text(item) for item in sorted(gt_fields)],
            "field_boundaries": [byte_boundary_text(item) for item in sorted(gt_field_boundaries)],
            "bitfields": bitfield_items(gt_bits),
            "subfield_boundaries": [bit_boundary_text(item) for item in sorted(gt_bit_boundaries)],
        },
        "experiment": {
            "fields": [range_text(item) for item in sorted(pred_fields)],
            "field_boundaries": [byte_boundary_text(item) for item in sorted(pred_field_boundaries)],
            "bitfields": bitfield_items(pred_bits),
            "subfield_boundaries": [bit_boundary_text(item) for item in sorted(pred_bit_boundaries)],
        },
        "diff": {
            "fields_missing": [range_text(item) for item in sorted(gt_fields - pred_fields)],
            "fields_extra": [range_text(item) for item in sorted(pred_fields - gt_fields)],
            "field_boundaries_missing": [byte_boundary_text(item) for item in sorted(gt_field_boundaries - pred_field_boundaries)],
            "field_boundaries_extra": [byte_boundary_text(item) for item in sorted(pred_field_boundaries - gt_field_boundaries)],
            "bitfields_missing": bitfield_items({key: gt_bits[key] for key in set(gt_bits) - set(pred_bits)}),
            "bitfields_extra": bitfield_items({key: pred_bits[key] for key in set(pred_bits) - set(gt_bits)}),
            "subfields_missing": bitfield_items(diff_bitfields(gt_bits, pred_bits)),
            "subfields_extra": bitfield_items(diff_bitfields(pred_bits, gt_bits)),
            "subfield_boundaries_missing": [bit_boundary_text(item) for item in sorted(gt_bit_boundaries - pred_bit_boundaries)],
            "subfield_boundaries_extra": [bit_boundary_text(item) for item in sorted(pred_bit_boundaries - gt_bit_boundaries)],
        },
        "dropped_out_of_payload": {
            "gt_fields": [range_text(item) for item in sorted(dropped_gt_fields)],
            "experiment_fields": [range_text(item) for item in sorted(dropped_pred_fields)],
            "gt_bitfields": bitfield_items(dropped_gt_bits),
            "experiment_bitfields": bitfield_items(dropped_pred_bits),
        },
    }


def diff_bitfields(left: BitfieldMap, right: BitfieldMap) -> BitfieldMap:
    diff: BitfieldMap = {}
    for byte_range in sorted(set(left) | set(right)):
        labels = left.get(byte_range, set()) - right.get(byte_range, set())
        if labels:
            diff[byte_range] = labels
    return diff


def summarize_packets(protocol: str, packets: List[dict]) -> dict:
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


def metric_has_denominator(metrics: dict) -> bool:
    return int(metrics.get("tp", 0)) + int(metrics.get("fp", 0)) + int(metrics.get("fn", 0)) > 0


def average_metric_value(protocols: List[dict], path: Tuple[str, ...], value_key: str = "f1") -> dict:
    values: List[float] = []
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


def average_metric(protocols: List[dict], path: Tuple[str, ...]) -> dict:
    values: List[dict] = []
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


def average_exact_recall(protocols: List[dict]) -> dict:
    values = [
        proto["bitfield_boundary"]["exact_match_recall"]
        for proto in protocols
        if int(proto["bitfield_boundary"]["gt_bitfield_count"]) > 0
    ]
    return {
        "exact_match_recall": 0.0 if not values else sum(float(value) for value in values) / len(values),
        "protocol_count": len(values),
    }


def build_protocol_value_average(protocols: List[dict]) -> dict:
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


def build_macro_non_empty(protocols: List[dict]) -> dict:
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


def build_groundtruth_readable(gt_root: Path, protocols: Sequence[str]) -> str:
    lines = [
        "# Field Segmentation Groundtruth From TShark",
        "",
        f"generated-at: {datetime.now().isoformat(timespec='seconds')}",
        f"groundtruth-root: {gt_root}",
        "",
        "说明：本文件只读取 tshark groundtruth JSON。每个包只输出总字段划分和位字段子字段划分，不展示协议栈信息。",
        "",
    ]
    for protocol in protocols:
        lines.extend([f"## {protocol}", ""])
        gt_packets = load_tshark_groundtruth(gt_root, protocol)
        for packet_index in sorted(gt_packets):
            packet = gt_packets[packet_index]
            fields = with_bitfield_parent_ranges(packet.fields, packet.bitfields)
            lines.extend([f"### packet {packet_index} / pkt_{packet_index - 1:04d}", ""])
            lines.append(f"fields: {ranges_line(fields)}")
            lines.append(f"bitfields: {line(bitfield_items(packet.bitfields))}")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_compare_readable(summary: dict, details: Dict[str, List[dict]]) -> str:
    protocol_rows = summary["protocols"]
    total_packets = sum(int(proto["packet_count"]) for proto in protocol_rows)
    protocol_avg = summary.get("protocol_value_average", build_protocol_value_average(protocol_rows))
    overall = summary["overall"]
    macro = summary.get("macro_non_empty", build_macro_non_empty(protocol_rows))

    lines = [
        "# Field Segmentation Groundtruth Vs Experiment",
        "",
        f"generated-at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "每个包展示 tshark groundtruth 与实验结果的字段边界差异，并附带位字段父字段和位字段子字段差异。",
        "这里只展示总字段划分，不展示协议栈信息。",
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
    lines.append(
        "| **Protocol Value Avg** | {packet_count} | {field_f1:.4f} | {field_boundary_hit_f1:.4f} | {bit_f1:.4f} | {sub_f1:.4f} | {subfield_boundary_hit_f1:.4f} | {exact:.4f} |".format(
            packet_count=total_packets,
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
            packet_count=total_packets,
            field_f1=overall["field_boundary"]["f1"],
            field_boundary_hit_f1=overall["field_boundary_hit"]["f1"],
            bit_f1=overall["bitfield_detection"]["f1"],
            sub_f1=overall["bitfield_boundary"]["subfield"]["f1"],
            subfield_boundary_hit_f1=overall["bitfield_subfield_boundary_hit"]["f1"],
            exact=overall["bitfield_boundary"]["exact_match_recall"],
        )
    )
    lines.append(
        "| **Macro Non-Empty** | {packet_count} | {field_f1:.4f} | {field_boundary_hit_f1:.4f} | {bit_f1:.4f} | {sub_f1:.4f} | {subfield_boundary_hit_f1:.4f} | {exact:.4f} |".format(
            packet_count=total_packets,
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
            "列含义：",
            "",
            "- `Protocol`：协议名。",
            "- `Packets`：该协议参与评估的数据包数量；overall 行为所有协议数据包数量总和。",
            "- `Field F1`：byte 粒度字段整段匹配 F1。只有预测字段范围与 groundtruth 字段范围完全相同才算 TP；字段拆分或合并会产生 FP/FN。",
            "- `Field Boundary-Hit F1`：byte 粒度字段边界命中 F1。把每个字段转换为起始边界和结束边界后计算 F1；它比整段匹配更宽松，用于观察边界点是否找对。",
            "- `Bitfield F1`：bit 粒度父字段检测 F1。只比较哪些 byte range 被识别为包含 bit 子字段，不比较具体 bit 子字段边界。",
            "- `Subfield F1`：bit 子字段整段匹配 F1。只有 byte range 和 bit range 都完全一致才算 TP。",
            "- `Subfield Boundary-Hit F1`：bit 子字段边界命中 F1。把每个 bit 子字段转换为 bit 起止边界后计算 F1，比 bit 子字段整段匹配更宽松。",
            "- `Boundary Exact Recall`：bit 父字段内部子字段划分完全一致的召回率，即 groundtruth 中有多少 bit 父字段的全部 bit 子字段集合被实验结果完全恢复。",
            "",
        ]
    )

    for protocol, packets in details.items():
        lines.extend([f"## {protocol}", ""])
        for packet in packets:
            if "error" in packet:
                lines.extend([
                    f"### {packet['packet_dir']} / groundtruth packet {packet['groundtruth_packet_index']}",
                    "",
                    f"error: {packet['error']}",
                    "",
                ])
                continue
            lines.extend([
                f"### {packet['packet_dir']} / groundtruth packet {packet['groundtruth_packet_index']}",
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
                f"subfields missing: {line(packet['diff']['subfields_missing'])}",
                f"subfields extra: {line(packet['diff']['subfields_extra'])}",
                f"GT subfield boundaries: {line(packet['gt']['subfield_boundaries'])}",
                f"EXP subfield boundaries: {line(packet['experiment']['subfield_boundaries'])}",
                f"subfield boundaries missing: {line(packet['diff']['subfield_boundaries_missing'])}",
                f"subfield boundaries extra: {line(packet['diff']['subfield_boundaries_extra'])}",
                "",
            ])
            dropped = packet["dropped_out_of_payload"]
            if any(dropped.values()):
                lines.extend([
                    f"dropped gt fields: {line(dropped['gt_fields'])}",
                    f"dropped exp fields: {line(dropped['experiment_fields'])}",
                    f"dropped gt bitfields: {line(dropped['gt_bitfields'])}",
                    f"dropped exp bitfields: {line(dropped['experiment_bitfields'])}",
                    "",
                ])
    return "\n".join(lines) + "\n"


def packet_score(packet: dict) -> float:
    return (
        packet["field_boundary_hit"]["f1"]
        + packet["field_boundary"]["f1"]
        + packet["bitfield_detection"]["f1"]
        + packet["bitfield_boundary"]["subfield"]["f1"]
        + packet["bitfield_subfield_boundary_hit"]["f1"]
    ) / 5


def build_packet_metrics(details: Dict[str, List[dict]]) -> dict:
    result: Dict[str, List[dict]] = {}
    for protocol, packets in details.items():
        result[protocol] = []
        for packet in packets:
            if "error" in packet:
                result[protocol].append({
                    "packet_dir": packet["packet_dir"],
                    "groundtruth_packet_index": packet["groundtruth_packet_index"],
                    "error": packet["error"],
                })
                continue
            result[protocol].append({
                "packet_dir": packet["packet_dir"],
                "groundtruth_packet_index": packet["groundtruth_packet_index"],
                "payload_len": packet["payload_len"],
                "score": packet_score(packet),
                "field_boundary": packet["field_boundary"],
                "field_boundary_hit": packet["field_boundary_hit"],
                "bitfield_detection": packet["bitfield_detection"],
                "bitfield_subfield": packet["bitfield_boundary"]["subfield"],
                "bitfield_subfield_boundary_hit": packet["bitfield_subfield_boundary_hit"],
            })
    return result


def build_packet_metrics_markdown(packet_metrics: Dict[str, List[dict]]) -> str:
    lines = [
        "# Per-Packet Evaluation Metrics",
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
        lines.extend([
            "| Packet | GT Packet | Len | Score | Field F1 | Field Boundary-Hit F1 | Bitfield F1 | Subfield F1 | Subfield Boundary-Hit F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        sorted_packets = sorted(
            packets,
            key=lambda item: (
                float("inf") if "error" in item else item["score"],
                item["packet_dir"],
            ),
        )
        for packet in sorted_packets:
            if "error" in packet:
                lines.append(
                    f"| {packet['packet_dir']} | {packet['groundtruth_packet_index']} | - | - | error: {packet['error']} | - | - | - | - |"
                )
                continue
            lines.append(
                "| {packet} | {gt} | {length} | {score:.4f} | {field:.4f} | {field_hit:.4f} | {bit:.4f} | {subfield:.4f} | {subfield_hit:.4f} |".format(
                    packet=packet["packet_dir"],
                    gt=packet["groundtruth_packet_index"],
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


def write_summary_markdown(summary: dict, path: Path) -> None:
    protocol_avg = summary.get("protocol_value_average")
    lines = [
        "# TShark Groundtruth Evaluation Metrics",
        "",
        "指标均不计算 TN；因为负例全集只能来自纯 tshark 字段全集，不能从实验结果反推。",
        "",
        "## Protocol Value Avg",
        "",
    ]
    overall = summary["overall"]
    if protocol_avg:
        lines.extend([
            "该行直接对协议表中的结果求平均；每个指标只除以该指标有有效分母的协议数。",
            "",
            f"- Field boundary: f1={protocol_avg['field_boundary']['f1']:.4f}, protocols={protocol_avg['field_boundary']['protocol_count']}",
            f"- Field boundary-hit: f1={protocol_avg['field_boundary_hit']['f1']:.4f}, protocols={protocol_avg['field_boundary_hit']['protocol_count']}",
            f"- Bitfield detection: f1={protocol_avg['bitfield_detection']['f1']:.4f}, protocols={protocol_avg['bitfield_detection']['protocol_count']}",
            f"- Bitfield boundary subfield: f1={protocol_avg['bitfield_boundary']['subfield']['f1']:.4f}, protocols={protocol_avg['bitfield_boundary']['subfield']['protocol_count']}",
            f"- Bitfield subfield boundary-hit: f1={protocol_avg['bitfield_subfield_boundary_hit']['f1']:.4f}, protocols={protocol_avg['bitfield_subfield_boundary_hit']['protocol_count']}",
            f"- Bitfield boundary exact recall: {protocol_avg['bitfield_boundary']['exact_match_recall']:.4f}, protocols={protocol_avg['bitfield_boundary']['protocol_count']}",
            "",
        ])
    else:
        lines.extend(["- unavailable", ""])
    lines.extend([
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
    ])
    macro = summary.get("macro_non_empty")
    if macro:
        lines.extend([
            "该行是 protocol-level macro average；每个指标只平均该指标有有效分母的协议，并由宏平均 precision/recall 重新计算 F1，避免没有位字段的协议把 bitfield 指标拉成 0。",
            "",
            f"- Field boundary: f1={macro['field_boundary']['f1']:.4f}, protocols={macro['field_boundary']['protocol_count']}",
            f"- Field boundary-hit: f1={macro['field_boundary_hit']['f1']:.4f}, protocols={macro['field_boundary_hit']['protocol_count']}",
            f"- Bitfield detection: f1={macro['bitfield_detection']['f1']:.4f}, protocols={macro['bitfield_detection']['protocol_count']}",
            f"- Bitfield boundary subfield: f1={macro['bitfield_boundary']['subfield']['f1']:.4f}, protocols={macro['bitfield_boundary']['subfield']['protocol_count']}",
            f"- Bitfield subfield boundary-hit: f1={macro['bitfield_subfield_boundary_hit']['f1']:.4f}, protocols={macro['bitfield_subfield_boundary_hit']['protocol_count']}",
            f"- Bitfield boundary exact recall: {macro['bitfield_boundary']['exact_match_recall']:.4f}, protocols={macro['bitfield_boundary']['protocol_count']}",
            "",
        ])
    else:
        lines.extend(["- unavailable", ""])
    lines.extend([
        "## Per Protocol",
        "",
        "| Protocol | Packets | Field F1 | Field Boundary-Hit F1 | Bit F1 | Subfield F1 | Subfield Boundary-Hit F1 | Exact Boundary R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
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
    total_packets = sum(int(proto["packet_count"]) for proto in summary["protocols"])
    if protocol_avg:
        lines.append(
            "| **Protocol Value Avg** | {packet_count} | {ff:.4f} | {fbhf:.4f} | {bf:.4f} | {sf:.4f} | {sbhf:.4f} | {er:.4f} |".format(
                packet_count=total_packets,
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
            packet_count=total_packets,
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
                packet_count=total_packets,
                ff=macro["field_boundary"]["f1"],
                fbhf=macro["field_boundary_hit"]["f1"],
                bf=macro["bitfield_detection"]["f1"],
                sf=macro["bitfield_boundary"]["subfield"]["f1"],
                sbhf=macro["bitfield_subfield_boundary_hit"]["f1"],
                er=macro["bitfield_boundary"]["exact_match_recall"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def protocols_to_run(value: Optional[str]) -> List[str]:
    if value:
        return [value]
    return list(GT_FILE_BY_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tshark groundtruth with experiment fields/bitfields")
    parser.add_argument("--gt-root", default=str(DEFAULT_GT_ROOT))
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--protocol", choices=sorted(GT_FILE_BY_PROTOCOL))
    parser.add_argument("--groundtruth-md", default=str(DEFAULT_GT_READABLE))
    parser.add_argument("--compare-md", default=str(DEFAULT_COMPARE_READABLE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root)
    replay_root = Path(args.replay_root)
    outdir = Path(args.outdir)
    protocols = protocols_to_run(args.protocol)

    all_details: Dict[str, List[dict]] = {}
    protocol_summaries: List[dict] = []

    for protocol in protocols:
        gt_packets = load_tshark_groundtruth(gt_root, protocol)
        details: List[dict] = []
        for pkt_dir in packet_dirs(replay_root, protocol):
            gt_index = gt_packet_index_from_meta(protocol, pkt_dir)
            gt = gt_packets.get(gt_index)
            if gt is None:
                details.append({
                    "packet_dir": pkt_dir.name,
                    "groundtruth_packet_index": gt_index,
                    "error": "missing groundtruth packet",
                })
                continue
            details.append(compare_one_packet(gt, pkt_dir))
        all_details[protocol] = details
        protocol_summaries.append(summarize_packets(protocol, [item for item in details if "error" not in item]))

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

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outdir / "packet_details.json").write_text(
        json.dumps(all_details, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    packet_metrics = build_packet_metrics(all_details)
    (outdir / "packet_metrics.json").write_text(
        json.dumps(packet_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outdir / "packet_metrics.md").write_text(
        build_packet_metrics_markdown(packet_metrics),
        encoding="utf-8",
    )
    write_summary_markdown(summary, outdir / "metrics_summary.md")
    Path(args.groundtruth_md).write_text(build_groundtruth_readable(gt_root, protocols), encoding="utf-8")
    Path(args.compare_md).write_text(build_compare_readable(summary, all_details), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
