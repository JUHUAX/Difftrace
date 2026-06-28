#!/usr/bin/env python3
"""Export tshark byte-field groundtruth aligned to replay pkt_* sample IDs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_TSHARK_ROOT = ROOT / "groundtruth_from_tshark"
DEFAULT_REPLAY_ROOT = ROOT / "replay_manual_latest/outputs"
DEFAULT_OUTPUT = ROOT / "sota_evaluation/out/tshark_boundary_groundtruth.jsonl"
REFERENCE_EVALUATOR = ROOT / "evaluation_from_tshark/field_boundary/evaluate_tshark_vs_experiment.py"
PROTOCOLS = ("bacnet", "cip", "iec104", "iec61850", "modbus", "snap7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tshark-root", type=Path, default=DEFAULT_TSHARK_ROOT)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_reference_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("tshark_boundary_reference", REFERENCE_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REFERENCE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    reference = load_reference_evaluator()
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        packets = reference.load_tshark_groundtruth(args.tshark_root, protocol)
        for pkt_dir in sorted((args.replay_root / protocol).glob("pkt_*")):
            gt_index = reference.gt_packet_index_from_meta(protocol, pkt_dir)
            packet = packets.get(gt_index)
            if packet is None:
                continue
            for start, end in sorted(packet.fields):
                rows.append(
                    {
                        "protocol_name": protocol,
                        "sample_id": pkt_dir.name,
                        "field_id": f"b:{start}:{end}",
                        "source": "tshark",
                        "tshark_packet_index": gt_index,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[export] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
