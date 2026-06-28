#!/usr/bin/env python3
"""Generate RQ3 bitfield-ablation outputs from existing replay logs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


SEMVEC_ROOT = Path("/root/semvec")
DIFFTRACE_DIR = SEMVEC_ROOT / "difftrace"
if str(DIFFTRACE_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFTRACE_DIR))

from analyze_bitfields_planA import RECOVERY_MODES, analyze_logs, load_fields  # noqa: E402


DEFAULT_REPLAY_ROOT = SEMVEC_ROOT / "bitfield_groundtruth" / "replay_manual_latest" / "outputs"
DEFAULT_OUTDIR = SEMVEC_ROOT / "RQ3" / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse replay logs to generate full and ablated bitfield-recovery outputs."
    )
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mode", nargs="+", choices=RECOVERY_MODES, default=list(RECOVERY_MODES))
    parser.add_argument("--protocol", help="Only process one protocol")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: stop after N packets per mode")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-fill-bit-gaps",
        action="store_true",
        help="Do not add synthetic uncovered bit ranges to detected bitfield parents",
    )
    return parser.parse_args()


def packet_dirs(replay_root: Path, protocol: Optional[str]) -> Iterable[Tuple[str, Path]]:
    for proto_dir in sorted(replay_root.iterdir()):
        if not proto_dir.is_dir() or (protocol and proto_dir.name != protocol):
            continue
        for pkt_dir in sorted(proto_dir.glob("pkt_*")):
            if pkt_dir.is_dir():
                yield proto_dir.name, pkt_dir


def bit_label_bounds(label: str) -> Optional[Tuple[int, int]]:
    text = str(label).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1]
    if ":" in inner:
        high, low = inner.split(":", 1)
        return int(low), int(high)
    if inner.isdigit():
        bit = int(inner)
        return bit, bit
    return None


def bit_label(low: int, high: int) -> str:
    return f"[{low}]" if low == high else f"[{high}:{low}]"


def parse_byte_range(text: str) -> Tuple[int, int]:
    values = [int(value) for value in str(text).replace("-", ",").split(",") if value.strip()]
    if not values:
        raise ValueError(f"invalid byte range: {text!r}")
    return min(values), max(values)


def fill_bit_gaps(subfields: Sequence[dict], width_bits: int) -> list[dict]:
    ranges = []
    for item in subfields:
        bounds = bit_label_bounds(item.get("label", ""))
        if bounds is None:
            continue
        low, high = bounds
        low = max(0, low)
        high = min(width_bits - 1, high)
        if low <= high:
            ranges.append((low, high, item))

    result: list[dict] = []
    cursor = 0
    for low, high, item in sorted(ranges, key=lambda value: (value[0], value[1])):
        if low > cursor:
            result.append({"label": bit_label(cursor, low - 1), "constraints": [], "synthetic_gap": True})
        if high >= cursor:
            copied = dict(item)
            copied["label"] = bit_label(max(low, cursor), high)
            result.append(copied)
            cursor = high + 1
    if cursor < width_bits:
        result.append({"label": bit_label(cursor, width_bits - 1), "constraints": [], "synthetic_gap": True})
    return result


def complete_bit_gaps(result: dict) -> int:
    synthetic_count = 0
    for field in result.get("fields", []):
        start, end = parse_byte_range(field["field_id"])
        width_bits = (end - start + 1) * 8
        field["subfields"] = fill_bit_gaps(field.get("subfields", []), width_bits)
        synthetic_count += sum(bool(item.get("synthetic_gap")) for item in field["subfields"])
    return synthetic_count


def count_result(result: dict) -> dict[str, int]:
    fields = result.get("fields", [])
    subfields = [item for field in fields for item in field.get("subfields", [])]
    synthetic = [item for item in subfields if item.get("synthetic_gap")]
    return {
        "bitfield_parents": len(fields),
        "subfields": len(subfields),
        "synthetic_gap_subfields": len(synthetic),
        "non_synthetic_subfields": len(subfields) - len(synthetic),
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_packet_inputs(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("fields.json", "meta.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)


def main() -> None:
    args = parse_args()
    if not args.replay_root.is_dir():
        raise FileNotFoundError(f"replay root not found: {args.replay_root}")

    packets = list(packet_dirs(args.replay_root, args.protocol))
    if args.limit > 0:
        packets = packets[: args.limit]
    if not packets:
        raise ValueError("no replay packet directories matched")

    manifest: dict[str, object] = {
        "source_replay_root": str(args.replay_root),
        "fill_bit_gaps": not args.no_fill_bit_gaps,
        "modes": {},
    }

    for mode in args.mode:
        stats = defaultdict(int)
        mode_root = args.outdir / mode
        print(f"[rq3] mode={mode} packets={len(packets)}")
        for index, (protocol, pkt_dir) in enumerate(packets, start=1):
            trace_path = pkt_dir / "trace.preprocessed.log"
            fields_path = pkt_dir / "fields.json"
            out_pkt_dir = mode_root / protocol / pkt_dir.name
            out_bitfields = out_pkt_dir / "bitfields.json"
            if out_bitfields.exists() and not args.overwrite:
                print(f"[rq3] skip {mode} {protocol}/{pkt_dir.name}: output exists")
                continue
            if not trace_path.exists() or not fields_path.exists():
                print(f"[rq3] skip {mode} {protocol}/{pkt_dir.name}: missing trace.preprocessed.log or fields.json")
                stats["skipped_packets"] += 1
                continue

            fields = load_fields(str(fields_path))
            result = analyze_logs(fields, [str(trace_path)], mode=mode)
            raw_counts = count_result(result)
            if not args.no_fill_bit_gaps:
                complete_bit_gaps(result)
            output_counts = count_result(result)

            copy_packet_inputs(pkt_dir, out_pkt_dir)
            write_json(out_bitfields, result)

            stats["processed_packets"] += 1
            for key, value in raw_counts.items():
                stats[f"raw_{key}"] += value
            for key, value in output_counts.items():
                stats[f"output_{key}"] += value
            print(
                f"[rq3] {mode} {index}/{len(packets)} {protocol}/{pkt_dir.name} "
                f"parents={raw_counts['bitfield_parents']} raw_subfields={raw_counts['subfields']} "
                f"output_subfields={output_counts['subfields']}"
            )
        manifest["modes"][mode] = dict(stats)

    write_json(args.outdir / "rq3_generation_manifest.json", manifest)
    print(f"[rq3] manifest={args.outdir / 'rq3_generation_manifest.json'}")


if __name__ == "__main__":
    main()
