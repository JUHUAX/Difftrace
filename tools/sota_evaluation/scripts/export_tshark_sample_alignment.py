#!/usr/bin/env python3
"""Export an explicit replay-packet to tshark-sample alignment table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_REPLAY_ROOT = ROOT / "replay_manual_latest/outputs"
DEFAULT_OUTPUT = ROOT / "sota_evaluation/out/tshark_sample_alignment.csv"
PROTOCOLS = ("bacnet", "cip", "iec104", "iec61850", "modbus", "snap7")
SAMPLE_OFFSET = {
    "bacnet": 1,
    "cip": 1,
    "iec104": 1,
    "iec61850": 2,
    "modbus": 1,
    "snap7": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for protocol in PROTOCOLS:
        for pkt_dir in sorted((args.replay_root / protocol).glob("pkt_*")):
            meta = json.loads((pkt_dir / "meta.json").read_text(encoding="utf-8"))
            packet_index = int(meta["pcap_packet_index"])
            tshark_index = packet_index + SAMPLE_OFFSET[protocol]
            rows.append(
                {
                    "protocol_name": protocol,
                    "replay_sample_id": pkt_dir.name,
                    "pcap_packet_index_zero_based": packet_index,
                    "tshark_sample_id": f"sample_{tshark_index:03d}",
                    "mapping_rule": f"tshark_sample_number = pcap_packet_index + {SAMPLE_OFFSET[protocol]}",
                    "payload_hex": str(meta["payload_hex"]).lower(),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[export] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
