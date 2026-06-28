#!/usr/bin/env python3
"""Export SOTA field-boundary predictions into one canonical JSONL format."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/root")
DEFAULT_REPLAY_ROOT = ROOT / "semvec/bitfield_groundtruth/replay_manual_latest/outputs"
DEFAULT_BINPRE_ROOT = ROOT / "SOTA/BinPRE/BinPRE_Res"
DEFAULT_BINARYINFERNO_ROOT = ROOT / "SOTA/binaryinferno/outputs/semvec"
DEFAULT_FIELDHUNTER_ROOT = ROOT / "SOTA/fieldhunter/reports"
DEFAULT_OUTPUT = (
    ROOT / "semvec/bitfield_groundtruth/sota_evaluation/out/boundary_predictions.jsonl"
)
PROTOCOLS = ("bacnet", "cip", "iec104", "iec61850", "modbus", "snap7")
ALIASES = {"eip": "cip", "s7": "snap7"}
BINPRE_DIRS = {
    "bacnet": "bacnet_50",
    "cip": "eip_50",
    "iec104": "iec104_50",
    "iec61850": "iec61850_50",
    "modbus": "modbus_50",
    "snap7": "s7_50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--binpre-root", type=Path, default=DEFAULT_BINPRE_ROOT)
    parser.add_argument("--binaryinferno-root", type=Path, default=DEFAULT_BINARYINFERNO_ROOT)
    parser.add_argument("--fieldhunter-root", type=Path, default=DEFAULT_FIELDHUNTER_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_protocol(value: str) -> str:
    value = value.strip().lower()
    return ALIASES.get(value, value)


def ranges_from_syntax(values: Any) -> list[list[int]]:
    ranges: list[list[int]] = []
    for value in values or []:
        indices = [int(item) for item in str(value).split(",") if item.strip()]
        if indices:
            ranges.append([min(indices), max(indices)])
    return normalize_ranges(ranges)


def normalize_ranges(values: Iterable[Iterable[int]]) -> list[list[int]]:
    ranges = sorted({(int(value[0]), int(value[1])) for value in values})
    return [[start, end] for start, end in ranges if start >= 0 and end >= start]


def replay_packets(replay_root: Path) -> dict[str, list[dict[str, Any]]]:
    packets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for protocol in PROTOCOLS:
        for pkt_dir in sorted((replay_root / protocol).glob("pkt_*")):
            meta = read_json(pkt_dir / "meta.json")
            packets[protocol].append(
                {
                    "protocol": protocol,
                    "sample_id": pkt_dir.name,
                    "payload_hex": str(meta["payload_hex"]).lower(),
                    "payload_length": len(bytes.fromhex(str(meta["payload_hex"]))),
                    "message_index": int(meta.get("pcap_packet_index", -1)),
                }
            )
    return packets


def row_for(packet: dict[str, Any], method: str, variant: str, fields: list[list[int]],
            status: str = "ok", notes: str = "", semantics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "method": method,
        "variant": variant,
        **packet,
        "fields": normalize_ranges(fields),
        "semantics": semantics or [],
        "status": status,
        "notes": notes,
    }


def export_difftrace(packets: dict[str, list[dict[str, Any]]], replay_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol, protocol_packets in packets.items():
        for packet in protocol_packets:
            fields_path = replay_root / protocol / packet["sample_id"] / "fields.json"
            data = read_json(fields_path)
            fields = [[item["a"], item["b"]] for item in data.get("fields", [])]
            rows.append(row_for(packet, "difftrace", "difftrace", fields))
    return rows


def export_binpre(packets: dict[str, list[dict[str, Any]]], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = ("binpre", "polyglot", "tupni", "autoformat")
    for protocol in PROTOCOLS:
        result_dir = root / BINPRE_DIRS[protocol]
        paths = sorted(result_dir.glob("*_predictions.jsonl"))
        source_rows: list[dict[str, Any]] = []
        if paths:
            with paths[0].open(encoding="utf-8") as handle:
                source_rows = [json.loads(line) for line in handle if line.strip()]
        by_tool_payload: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for source in source_rows:
            key = (str(source.get("tool", "")), str(source.get("payload_hex", "")).lower())
            by_tool_payload[key].append(source)
        for packet in packets[protocol]:
            baseline_key = ("ExeT-based", packet["payload_hex"])
            binpre_key = ("BinPRE", packet["payload_hex"])
            baseline = by_tool_payload[baseline_key].popleft() if by_tool_payload[baseline_key] else None
            binpre = by_tool_payload[binpre_key].popleft() if by_tool_payload[binpre_key] else None
            for variant in variants:
                source = binpre if variant == "binpre" else baseline
                if source is None:
                    note = "payload not found in BinPRE refinement JSONL" if variant == "binpre" else "payload not found in ExeT-based JSONL"
                    rows.append(row_for(packet, "binpre", variant, [], "no_prediction", note))
                    continue
                if variant != "binpre" and not source.get("analysis_status", {}).get("success", False):
                    error = source.get("analysis_status", {}).get("error", "BinPRE analysis failed")
                    rows.append(row_for(packet, "binpre", variant, [], "analysis_failed", str(error)))
                    continue
                if variant == "binpre":
                    refined = source.get("refined", {})
                    semantic_rows = []
                    for field in refined.get("fields", []):
                        field_key = str(field)
                        indices = [int(item) for item in field_key.split(",") if item.strip()]
                        if not indices:
                            continue
                        semantic_rows.append(
                            {
                                "start": min(indices),
                                "end": max(indices),
                                "raw_label": json.dumps(
                                    {
                                        "types": refined.get("field_types", {}).get(field_key, []),
                                        "functions": refined.get("field_functions", {}).get(field_key, []),
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
                    status = "ok" if refined.get("fields") is not None else "no_prediction"
                    note = str(source.get("refinement_status", {}).get("error", ""))
                    rows.append(row_for(packet, "binpre", variant, ranges_from_syntax(refined.get("fields")), status, note, semantic_rows))
                else:
                    syntax = source.get(variant, {}).get("syntax")
                    if syntax is None:
                        rows.append(row_for(packet, "binpre", variant, [], "no_prediction"))
                    elif not isinstance(syntax, list):
                        rows.append(row_for(packet, "binpre", variant, [], "not_exported", "rerun BinPRE to serialize this variant as field ranges"))
                    else:
                        rows.append(row_for(packet, "binpre", variant, ranges_from_syntax(syntax)))
    return rows


def parse_binaryinferno_spec(path: Path) -> list[tuple[str, int | None, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"SPECSTART\s*(.*?)\s*SPECEND", text, re.S)
    if not match:
        return []
    spec: list[tuple[str, int | None, str]] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        kind = line.split()[0]
        size_match = re.search(r"\b(\d+)V(?:_| |\()", line)
        size = int(size_match.group(1)) if size_match else None
        spec.append((kind, size, line))
    return spec


def expand_binaryinferno_spec(spec: list[tuple[str, int | None, str]], payload_length: int) -> tuple[list[list[int]], list[dict[str, Any]], str]:
    fields: list[list[int]] = []
    semantics: list[dict[str, Any]] = []
    offset = 0
    for index, (kind, size, raw) in enumerate(spec):
        if size is None or kind in {"FieldVar", "FieldRep"}:
            if offset < payload_length:
                fields.append([offset, payload_length - 1])
                semantics.append({"start": offset, "end": payload_length - 1, "raw_label": raw})
            return fields, semantics, "coarse_template_expansion"
        end = min(offset + size - 1, payload_length - 1)
        if offset <= end:
            fields.append([offset, end])
            semantics.append({"start": offset, "end": end, "raw_label": raw})
        offset += size
        if offset >= payload_length:
            break
    if offset < payload_length:
        fields.append([offset, payload_length - 1])
        semantics.append({"start": offset, "end": payload_length - 1, "raw_label": "unparsed trailing bytes"})
    return fields, semantics, "fixed_template_expansion"


def export_binaryinferno(packets: dict[str, list[dict[str, Any]]], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        path = root / protocol / "inferred_description.txt"
        spec = parse_binaryinferno_spec(path) if path.is_file() else []
        for packet in packets[protocol]:
            if not spec:
                rows.append(row_for(packet, "binaryinferno", "binaryinferno", [], "no_prediction", "SPECSTART/SPECEND block not found"))
                continue
            fields, semantics, note = expand_binaryinferno_spec(spec, packet["payload_length"])
            rows.append(row_for(packet, "binaryinferno", "binaryinferno", fields, notes=note, semantics=semantics))
    return rows


def export_fieldhunter(packets: dict[str, list[dict[str, Any]]], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        path = root / protocol / "prediction-only" / "fieldhunter_selected_fields.csv"
        by_payload: dict[str, list[dict[str, str]]] = defaultdict(list)
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for source in csv.DictReader(handle):
                    by_payload[source["message_hex"].lower()].append(source)
        for packet in packets[protocol]:
            source_rows = by_payload.get(packet["payload_hex"], [])
            fields: list[list[int]] = []
            semantics: list[dict[str, Any]] = []
            for source in source_rows:
                start = int(source["start"])
                end = int(source["end"]) - 1  # FieldHunter CSV uses [start, end).
                fields.append([start, end])
                semantics.append({"start": start, "end": end, "raw_label": source["field_type"]})
            status = "ok" if source_rows else "no_prediction"
            notes = "partial semantic-field predictions; uncovered bytes are intentionally not gap-filled"
            rows.append(row_for(packet, "fieldhunter", "fieldhunter", fields, status, notes, semantics))
    return rows


def main() -> None:
    args = parse_args()
    packets = replay_packets(args.replay_root)
    rows: list[dict[str, Any]] = []
    rows.extend(export_difftrace(packets, args.replay_root))
    rows.extend(export_binpre(packets, args.binpre_root))
    rows.extend(export_binaryinferno(packets, args.binaryinferno_root))
    rows.extend(export_fieldhunter(packets, args.fieldhunter_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[export] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
