#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze replay execution logs into field and bitfield results.

Input layout:
  <replay-root>/<protocol>/pkt_XXXX/trace.log

Output per packet:
  trace.preprocessed.log
  field_layout.txt
  fields.json
  bitfields.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
SEMVEC_ROOT = THIS_DIR.parent.parent
DIFFTRACE_DIR = SEMVEC_ROOT / "difftrace"
if str(DIFFTRACE_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFTRACE_DIR))

from common import write_json  # noqa: E402
from diff import preprocess_log  # noqa: E402
from fields import extract_fields_from_log  # noqa: E402


DEFAULT_REPLAY_ROOT = Path("/root/semvec/bitfield_groundtruth/replay_manual_latest/outputs")
DEFAULT_FIELDS_SCRIPT = DIFFTRACE_DIR / "fields.py"
DEFAULT_BITFIELD_SCRIPT = DIFFTRACE_DIR / "analyze_bitfields_planA.py"


def format_field_ranges_simple(fields: Sequence[Tuple[int, int]]) -> str:
    parts = []
    for start, end in fields:
        if start <= end:
            members = ",".join(str(i) for i in range(start, end + 1))
        else:
            members = str(start)
        parts.append(f"[{members}]")
    return "[" + ",".join(parts) + "]"


def range_repr(start: int, end: int) -> str:
    return f"{start}" if start == end else f"{start},{end}"


def parse_byte_range(text: str) -> Tuple[int, int]:
    text = str(text).strip()
    if "," in text:
        parts = [int(x) for x in text.split(",") if x.strip()]
        return min(parts), max(parts)
    if "-" in text:
        start, end = text.split("-", 1)
        return int(start), int(end)
    value = int(text)
    return value, value


def payload_len_from_meta(pkt_dir: Path) -> int:
    meta_path = pkt_dir / "meta.json"
    if not meta_path.exists():
        return 0
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    payload_hex = meta.get("payload_hex") or ""
    return len(bytes.fromhex(payload_hex)) if payload_hex else 0


def fill_byte_gaps(fields: Sequence[Tuple[int, int]], payload_len: int) -> List[Tuple[int, int, bool]]:
    """Return non-overlapping byte fields plus synthetic gaps."""
    normalized = sorted(
        {
            (max(0, int(start)), min(payload_len - 1, int(end)))
            for start, end in fields
            if payload_len <= 0 or (int(start) <= int(end) and int(end) >= 0 and int(start) < payload_len)
        }
    )
    if payload_len <= 0:
        return [(start, end, False) for start, end in normalized]

    result: List[Tuple[int, int, bool]] = []
    cursor = 0
    for start, end in normalized:
        if start > cursor:
            result.append((cursor, start - 1, True))
        if end >= cursor:
            result.append((max(start, cursor), end, False))
            cursor = end + 1
    if cursor < payload_len:
        result.append((cursor, payload_len - 1, True))
    return result


def write_field_layout(preprocessed_log_path: Path, out_path: Path, payload_len: int) -> dict:
    raw_fields = extract_fields_from_log(str(preprocessed_log_path))
    fields_with_gaps = fill_byte_gaps(raw_fields, payload_len)
    fields = [(start, end) for start, end, _ in fields_with_gaps]
    simple_text = format_field_ranges_simple(fields)
    out_path.write_text(simple_text + "\n", encoding="utf-8")
    return {
        "path": str(out_path),
        "fields": [[start, end] for start, end in fields],
        "synthetic_gaps": [[start, end] for start, end, synthetic in fields_with_gaps if synthetic],
        "simple": simple_text,
    }


def fill_fields_json(path: Path, payload_len: int) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    ranges = []
    original_by_range = {}
    for item in obj.get("fields", []):
        if "a" in item and "b" in item:
            start, end = int(item["a"]), int(item["b"])
        elif "repr" in item:
            start, end = parse_byte_range(item["repr"])
        elif "field_id" in item:
            start, end = parse_byte_range(item["field_id"])
        else:
            continue
        ranges.append((start, end))
        original_by_range[(start, end)] = item

    filled = []
    for start, end, synthetic in fill_byte_gaps(ranges, payload_len):
        if not synthetic and (start, end) in original_by_range:
            item = dict(original_by_range[(start, end)])
            item["a"] = start
            item["b"] = end
            item["repr"] = range_repr(start, end)
        else:
            item = {
                "a": start,
                "b": end,
                "repr": range_repr(start, end),
                "synthetic_gap": True,
            }
        filled.append(item)
    obj["fields"] = filled
    write_json(str(path), obj)
    return obj


def bit_label_bounds(label: str) -> Optional[Tuple[int, int]]:
    label = str(label).strip()
    if not (label.startswith("[") and label.endswith("]")):
        return None
    inner = label[1:-1]
    if ":" in inner:
        high, low = inner.split(":", 1)
        return int(low), int(high)
    if inner.isdigit():
        bit = int(inner)
        return bit, bit
    return None


def bit_label(low: int, high: int) -> str:
    return f"[{low}]" if low == high else f"[{high}:{low}]"


def fill_bit_gaps(subfields: Sequence[dict], width_bits: int) -> List[dict]:
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

    result: List[dict] = []
    cursor = 0
    for low, high, item in sorted(ranges, key=lambda value: (value[0], value[1])):
        if low > cursor:
            result.append({
                "label": bit_label(cursor, low - 1),
                "constraints": [],
                "synthetic_gap": True,
            })
        if high >= cursor:
            copied = dict(item)
            copied["label"] = bit_label(max(low, cursor), high)
            result.append(copied)
            cursor = high + 1
    if cursor < width_bits:
        result.append({
            "label": bit_label(cursor, width_bits - 1),
            "constraints": [],
            "synthetic_gap": True,
        })
    return result


def fill_bitfields_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    for field in obj.get("fields", []):
        try:
            start, end = parse_byte_range(field["field_id"])
        except Exception:
            continue
        width_bits = (end - start + 1) * 8
        if width_bits > 0:
            field["subfields"] = fill_bit_gaps(field.get("subfields", []), width_bits)
    write_json(str(path), obj)
    return obj


def packet_dirs(replay_root: Path) -> Iterable[Tuple[str, Path]]:
    for proto_dir in sorted(replay_root.iterdir()):
        if not proto_dir.is_dir():
            continue
        for pkt_dir in sorted(proto_dir.glob("pkt_*")):
            if pkt_dir.is_dir():
                yield proto_dir.name, pkt_dir


def run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def update_meta(pkt_dir: Path, updates: Dict[str, object]) -> None:
    meta_path = pkt_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    else:
        meta = {}
    meta.update(updates)
    write_json(str(meta_path), meta)


def analyze_packet(
    pkt_dir: Path,
    seen_taint_thread_ids: set,
    fields_script: Path,
    bitfield_script: Path,
    force: bool,
    skip_bitfields: bool,
    fill_gaps: bool,
) -> bool:
    raw_log = pkt_dir / "trace.log"
    if not raw_log.exists():
        print(f"skip {pkt_dir}: missing trace.log", file=sys.stderr)
        return False

    preprocessed_log = pkt_dir / "trace.preprocessed.log"
    field_layout_path = pkt_dir / "field_layout.txt"
    fields_json = pkt_dir / "fields.json"
    bitfields_json = pkt_dir / "bitfields.json"
    payload_len = payload_len_from_meta(pkt_dir)

    if force or not preprocessed_log.exists():
        preprocess = preprocess_log(str(raw_log), seen_taint_thread_ids)
    else:
        preprocess = {
            "path": str(preprocessed_log),
            "taint_found": None,
            "taint_thread_id": None,
            "line_count": None,
        }

    if force or not field_layout_path.exists():
        layout_payload_len = payload_len if fill_gaps else 0
        field_layout = write_field_layout(Path(preprocess["path"]), field_layout_path, layout_payload_len)
    else:
        field_layout = {
            "path": str(field_layout_path),
            "fields": None,
            "synthetic_gaps": None,
            "simple": field_layout_path.read_text(encoding="utf-8").strip(),
        }

    if force or not fields_json.exists():
        run_command([
            sys.executable,
            str(fields_script),
            "--log",
            str(preprocess["path"]),
            "--outdir",
            str(pkt_dir),
        ])
    if fill_gaps and fields_json.exists():
        fill_fields_json(fields_json, payload_len)

    if not skip_bitfields and (force or not bitfields_json.exists()):
        run_command([
            sys.executable,
            str(bitfield_script),
            "--log",
            str(preprocess["path"]),
            "--fields",
            str(fields_json),
            "--out",
            str(bitfields_json),
        ])
    if fill_gaps and not skip_bitfields and bitfields_json.exists():
        fill_bitfields_json(bitfields_json)

    update_meta(pkt_dir, {
        "preprocessed_log": str(preprocess["path"]),
        "preprocess": preprocess,
        "field_layout": field_layout,
    })
    print(f"[analyze] {pkt_dir} fields={field_layout['simple']}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze replay trace.log files into fields.json and bitfields.json"
    )
    parser.add_argument(
        "replay_root",
        nargs="?",
        default=str(DEFAULT_REPLAY_ROOT),
        help="Replay output root containing <protocol>/pkt_XXXX directories",
    )
    parser.add_argument("--protocol", help="Only analyze one protocol directory")
    parser.add_argument("--fields-script", default=str(DEFAULT_FIELDS_SCRIPT))
    parser.add_argument("--bitfield-script", default=str(DEFAULT_BITFIELD_SCRIPT))
    parser.add_argument("--force", action="store_true", help="Overwrite existing analysis outputs")
    parser.add_argument("--skip-bitfields", action="store_true", help="Only generate preprocessing and fields.json")
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Fill uncovered payload byte ranges and uncovered bit ranges with synthetic_gap entries",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    replay_root = Path(args.replay_root)
    fields_script = Path(args.fields_script)
    bitfield_script = Path(args.bitfield_script)

    if not replay_root.is_dir():
        raise FileNotFoundError(f"replay root not found: {replay_root}")

    seen_by_protocol: Dict[str, set] = {}
    analyzed = 0
    for protocol, pkt_dir in packet_dirs(replay_root):
        if args.protocol and protocol != args.protocol:
            continue
        seen = seen_by_protocol.setdefault(protocol, set())
        if analyze_packet(
            pkt_dir=pkt_dir,
            seen_taint_thread_ids=seen,
            fields_script=fields_script,
            bitfield_script=bitfield_script,
            force=bool(args.force),
            skip_bitfields=bool(args.skip_bitfields),
            fill_gaps=bool(args.fill_gaps),
        ):
            analyzed += 1

    print(f"[done] analyzed_packets={analyzed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
