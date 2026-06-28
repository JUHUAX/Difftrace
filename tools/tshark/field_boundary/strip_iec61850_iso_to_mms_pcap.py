#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

from scapy.all import IP, TCP, Raw, rdpcap, wrpcap


DEFAULT_INPUT = Path("/root/semvec/bitfield_groundtruth/pcap/iec61850_client_to_server_only.original_iso_stack.pcap")
DEFAULT_OUTPUT = Path("/root/semvec/bitfield_groundtruth/pcap/iec61850_client_to_server_only.pcap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip ISO stack bytes and keep TCP payload as direct MMS bytes."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="If output exists, copy it to <output>.bak before writing.",
    )
    return parser.parse_args()


def run_pdml(path: Path) -> str:
    proc = subprocess.run(
        ["tshark", "-r", str(path), "-T", "pdml"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout


def packet_number(packet_elem: ET.Element) -> int:
    for proto in packet_elem.findall("proto"):
        if proto.attrib.get("name") != "geninfo":
            continue
        for field in proto.findall("field"):
            if field.attrib.get("name") == "num":
                return int(field.attrib["show"])
    raise ValueError("packet has no geninfo.num")


def mms_spans(path: Path) -> Dict[int, Tuple[int, int]]:
    root = ET.fromstring(run_pdml(path))
    spans: Dict[int, Tuple[int, int]] = {}
    for packet_elem in root.findall("packet"):
        number = packet_number(packet_elem)
        for proto in packet_elem.findall("proto"):
            if proto.attrib.get("name") != "mms":
                continue
            pos = int(proto.attrib["pos"])
            size = int(proto.attrib["size"])
            spans[number] = (pos, size)
            break
    return spans


def strip_pcap(input_path: Path, output_path: Path, backup_existing: bool) -> None:
    spans = mms_spans(input_path)
    packets = rdpcap(str(input_path))
    stripped = []

    for number, packet in enumerate(packets, start=1):
        if number not in spans:
            continue
        if TCP not in packet or Raw not in packet:
            continue

        mms_abs_pos, mms_size = spans[number]
        raw = bytes(packet[Raw].load)
        tcp_abs_pos = len(bytes(packet)) - len(raw)
        mms_rel_pos = mms_abs_pos - tcp_abs_pos
        if mms_rel_pos < 0 or mms_rel_pos + mms_size > len(raw):
            raise ValueError(
                f"packet {number}: MMS span pos={mms_abs_pos} size={mms_size} "
                f"does not fit TCP payload len={len(raw)} tcp_abs_pos={tcp_abs_pos}"
            )

        new_packet = packet.copy()
        new_packet[Raw].load = raw[mms_rel_pos : mms_rel_pos + mms_size]
        if IP in new_packet:
            del new_packet[IP].len
            del new_packet[IP].chksum
        del new_packet[TCP].chksum
        stripped.append(new_packet)

    if backup_existing and output_path.exists():
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        backup_path.write_bytes(output_path.read_bytes())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(output_path), stripped)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"input_packets={len(packets)}")
    print(f"mms_packets={len(spans)}")
    print(f"written_packets={len(stripped)}")


def main() -> int:
    args = parse_args()
    strip_pcap(Path(args.input), Path(args.output), args.backup_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

