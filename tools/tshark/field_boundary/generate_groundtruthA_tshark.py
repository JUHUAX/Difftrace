#!/usr/bin/env python3
"""
Generate structured per-packet tshark extraction results for groundtruth A.

This script uses tshark PDML output as the primary source because it preserves
field offsets, lengths, display strings, raw values, and nested dissector
structure. The output keeps every visible field exported by tshark, not only
bitfield-related entries.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_PCAPS: Dict[str, str] = {
    "S7server": "/root/semvec/snap7/S7comm_500.pcap",
    "modbus": "/root/semvec/modbus/modbus_250.pcap",
    "iec104": "/root/semvec/iec104/iec104.pcap",
    "bacnet": "/root/semvec/bacnet/pcap/bacnet_client_to_server.pcap",
    "cip": "/root/semvec/CIP/pcaps/opener_client_to_server.pcap",
    "mms": "/root/semvec/iec61850/pcap/mms_client_to_server_only.pcap",
}

SKIP_PROTOCOLS = {
    "geninfo",
    "frame",
    "sll",
    "eth",
    "ethertype",
    "ip",
    "ipv6",
    "tcp",
    "udp",
    "data",
}

PATTERN_RE = re.compile(r"^\s*([01. ][01. ]+?)\s*=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-packet tshark field data for later bitfield analysis."
    )
    parser.add_argument(
        "--output-dir",
        default="/root/semvec/bitfield_groundtruth/groundtruthA_raw",
        help="Directory for per-pcap JSON output files.",
    )
    parser.add_argument(
        "--pcap",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional extra or replacement pcap entry. Can be given multiple times.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Process only the named pcaps from the configured set.",
    )
    return parser.parse_args()


def load_pcaps(args: argparse.Namespace) -> Dict[str, str]:
    pcaps = dict(DEFAULT_PCAPS)
    if args.pcap:
        pcaps = {}
        for item in args.pcap:
            if "=" not in item:
                raise SystemExit(f"Invalid --pcap value: {item!r}; expected NAME=PATH")
            name, path = item.split("=", 1)
            pcaps[name] = path
    if args.only:
        selected = {}
        for name in args.only:
            if name not in pcaps:
                raise SystemExit(f"Unknown pcap name in --only: {name}")
            selected[name] = pcaps[name]
        pcaps = selected
    return pcaps


def run_tshark_pdml(pcap_path: str) -> str:
    cmd = ["tshark", "-r", pcap_path, "-T", "pdml"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tshark failed for {pcap_path} with code {proc.returncode}:\n{proc.stderr}"
        )
    return proc.stdout


def get_protocol_chain(packet_elem: ET.Element) -> List[str]:
    for proto in packet_elem.findall("proto"):
        if proto.attrib.get("name") != "frame":
            continue
        for field in proto.findall("field"):
            if field.attrib.get("name") == "frame.protocols":
                show = field.attrib.get("show", "")
                return [p for p in show.split(":") if p]
    return []


def choose_allowed_protocols(pcap_name: str, protocol_chain: Sequence[str]) -> List[str]:
    if pcap_name == "mms":
        return ["mms"]

    allowed: List[str] = []
    seen_transport = False
    for proto in protocol_chain:
        if proto in {"tcp", "udp"}:
            seen_transport = True
            continue
        if not seen_transport:
            continue
        if proto in SKIP_PROTOCOLS:
            continue
        allowed.append(proto)
    return allowed


def to_int(text: Optional[str]) -> Optional[int]:
    if text is None or text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def hex_width_for_size(size_bytes: Optional[int]) -> Optional[int]:
    if size_bytes is None or size_bytes <= 0:
        return None
    return size_bytes * 2


def normalize_hex(text: Optional[str], size_bytes: Optional[int]) -> Optional[str]:
    if not text:
        return None
    raw = text.strip().lower()
    if raw.startswith("0x"):
        raw = raw[2:]
    if not re.fullmatch(r"[0-9a-f]+", raw):
        return None
    width = hex_width_for_size(size_bytes)
    if width is not None:
        raw = raw.zfill(width)
    return raw


def parse_pattern_bits(showname: Optional[str], size_bytes: Optional[int]) -> Tuple[Optional[List[int]], Optional[str]]:
    if not showname or not size_bytes or size_bytes <= 0:
        return None, None
    match = PATTERN_RE.match(showname)
    if not match:
        return None, None
    pattern = match.group(1)
    compact = pattern.replace(" ", "")
    total_bits = size_bytes * 8
    if len(compact) != total_bits:
        return None, compact
    used_bits: List[int] = []
    for idx, char in enumerate(compact):
        if char != ".":
            used_bits.append(total_bits - 1 - idx)
    return used_bits or None, compact


def bits_to_mask(bits: Optional[Sequence[int]]) -> Optional[str]:
    if not bits:
        return None
    value = 0
    for bit in bits:
        value |= 1 << bit
    width = max(bits) // 8 + 1
    return f"0x{value:0{width * 2}x}"


def bits_to_ranges(bits: Optional[Sequence[int]]) -> Optional[List[str]]:
    if not bits:
        return None
    ordered = sorted(set(bits), reverse=True)
    ranges: List[str] = []
    start = ordered[0]
    prev = ordered[0]
    for bit in ordered[1:]:
        if bit == prev - 1:
            prev = bit
            continue
        ranges.append(format_range(start, prev))
        start = bit
        prev = bit
    ranges.append(format_range(start, prev))
    return ranges


def format_range(high: int, low: int) -> str:
    if high == low:
        return f"[{high}]"
    return f"[{high}:{low}]"


def infer_raw_bytes(field_elem: ET.Element, size_bytes: Optional[int]) -> Optional[str]:
    unmasked = normalize_hex(field_elem.attrib.get("unmaskedvalue"), size_bytes)
    if unmasked is not None:
        return unmasked
    return normalize_hex(field_elem.attrib.get("value"), size_bytes)


def collect_fields(
    field_elem: ET.Element,
    packet_index: int,
    protocol_name: str,
    parent_field: Optional[str],
    parent_proto_pos: Optional[int],
    results: List[dict],
) -> None:
    field_name = field_elem.attrib.get("name", "")
    if not field_name:
        for child in field_elem.findall("field"):
            collect_fields(
                child,
                packet_index,
                protocol_name,
                parent_field,
                parent_proto_pos,
                results,
            )
        return

    if field_elem.attrib.get("hide") == "yes":
        return

    field_size = to_int(field_elem.attrib.get("size"))
    field_pos = to_int(field_elem.attrib.get("pos"))
    raw_bytes = infer_raw_bytes(field_elem, field_size)
    used_bits, bit_pattern = parse_pattern_bits(field_elem.attrib.get("showname"), field_size)

    entry = {
        "packet_index": packet_index,
        "protocol": protocol_name,
        "parent_field": parent_field,
        "field_name": field_name,
        "field_offset": field_pos,
        "field_offset_relative_to_proto": (
            None if field_pos is None or parent_proto_pos is None else field_pos - parent_proto_pos
        ),
        "field_length": field_size,
        "display_value": field_elem.attrib.get("show"),
        "showname": field_elem.attrib.get("showname"),
        "raw_bytes": raw_bytes,
        "value": field_elem.attrib.get("value"),
        "unmaskedvalue": field_elem.attrib.get("unmaskedvalue"),
        "bitmask": bits_to_mask(used_bits),
        "bit_offset": used_bits,
        "bit_ranges": bits_to_ranges(used_bits),
        "bit_pattern": bit_pattern,
        "has_children": bool(field_elem.findall("field")),
    }
    results.append(entry)

    for child in field_elem.findall("field"):
        collect_fields(
            child,
            packet_index,
            protocol_name,
            field_name,
            parent_proto_pos,
            results,
        )


def extract_packet(packet_elem: ET.Element, pcap_name: str) -> dict:
    protocol_chain = get_protocol_chain(packet_elem)
    allowed_protocols = choose_allowed_protocols(pcap_name, protocol_chain)

    packet_index: Optional[int] = None
    for proto in packet_elem.findall("proto"):
        if proto.attrib.get("name") == "geninfo":
            for field in proto.findall("field"):
                if field.attrib.get("name") == "num":
                    packet_index = to_int(field.attrib.get("show"))
                    break
    if packet_index is None:
        packet_index = 0

    extracted_fields: List[dict] = []
    extracted_protocols: List[dict] = []

    for proto in packet_elem.findall("proto"):
        proto_name = proto.attrib.get("name", "")
        if proto_name not in allowed_protocols:
            continue
        proto_pos = to_int(proto.attrib.get("pos"))
        proto_size = to_int(proto.attrib.get("size"))
        extracted_protocols.append(
            {
                "protocol": proto_name,
                "showname": proto.attrib.get("showname"),
                "offset": proto_pos,
                "length": proto_size,
            }
        )
        for field in proto.findall("field"):
            collect_fields(
                field,
                packet_index,
                proto_name,
                None,
                proto_pos,
                extracted_fields,
            )

    return {
        "packet_index": packet_index,
        "protocol_chain": protocol_chain,
        "selected_protocols": [item["protocol"] for item in extracted_protocols],
        "protocols": extracted_protocols,
        "fields": extracted_fields,
    }


def process_pcap(pcap_name: str, pcap_path: str) -> dict:
    pdml_text = run_tshark_pdml(pcap_path)
    root = ET.fromstring(pdml_text)
    packets = [extract_packet(packet_elem, pcap_name) for packet_elem in root.findall("packet")]
    return {
        "pcap_name": pcap_name,
        "pcap_path": pcap_path,
        "packet_count": len(packets),
        "scope_rule": (
            "MMS only" if pcap_name == "mms" else "Protocols above TCP/UDP only"
        ),
        "packets": packets,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    pcaps = load_pcaps(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pcap_name, pcap_path in pcaps.items():
        result = process_pcap(pcap_name, pcap_path)
        output_path = output_dir / f"{pcap_name}.json"
        write_json(output_path, result)
        print(f"[ok] {pcap_name}: wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
