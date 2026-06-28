#!/usr/bin/env python3
"""Collect per-log RQ2-B LLM JSON outputs into JSONL and CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple


REPO_ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_INPUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/llm_output"
)
DEFAULT_OUTPUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval"
)
DEFAULT_REPLAY_ROOT = REPO_ROOT / "replay_manual_latest" / "outputs"

ByteRange = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RQ2-B per-log LLM outputs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument(
        "--no-fill-gaps",
        action="store_true",
        help="Disable auto-filled byte and bit gap fields in collected groundtruth.",
    )
    parser.add_argument("--jsonl-name", default="program_log_groundtruth_candidates.jsonl")
    parser.add_argument("--csv-name", default="program_log_groundtruth_candidates.csv")
    parser.add_argument("--errors-name", default="program_log_groundtruth_collect_errors.csv")
    return parser.parse_args()


def parse_filename(path: Path) -> tuple[int | None, str, str]:
    match = re.match(r"^(\d+)_(.+)_(pkt_\d+)\.json$", path.name)
    if not match:
        return None, "", ""
    return int(match.group(1)), match.group(2), match.group(3)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        starts = [m.start() for m in re.finditer(r"\{", stripped)]
        for start in starts:
            candidate = stripped[start:]
            end = candidate.rfind("}")
            if end < 0:
                continue
            try:
                parsed = json.loads(candidate[: end + 1])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON must be an object")
    return parsed


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def payload_len_for(replay_root: Path, protocol: str, pkt: str) -> int:
    meta_path = replay_root / protocol / pkt / "meta.json"
    if not meta_path.exists():
        return 0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload_hex = meta.get("payload_hex") or ""
    return len(bytes.fromhex(payload_hex)) if payload_hex else 0


def normalize_field_id(field_id: Any) -> str:
    text = str(field_id or "").strip()
    byte_match = re.fullmatch(r"b:(\d+)", text)
    if byte_match:
        index = int(byte_match.group(1))
        return f"b:{index}:{index}"
    return text


def parse_field_id(field_id: str) -> tuple[str, ByteRange, tuple[int, int] | None] | None:
    byte_match = re.fullmatch(r"b:(\d+):(\d+)", field_id)
    if byte_match:
        start, end = int(byte_match.group(1)), int(byte_match.group(2))
        return "byte", (min(start, end), max(start, end)), None

    bit_match = re.fullmatch(r"bit:(\d+):(\d+):(\d+):(\d+)", field_id)
    if bit_match:
        start = int(bit_match.group(1))
        end = int(bit_match.group(2))
        low = int(bit_match.group(3))
        high = int(bit_match.group(4))
        return "bit", (min(start, end), max(start, end)), (min(low, high), max(low, high))

    return None


def field_id_for_byte_range(byte_range: ByteRange) -> str:
    return f"b:{byte_range[0]}:{byte_range[1]}"


def field_id_for_bit_range(byte_range: ByteRange, bit_range: tuple[int, int]) -> str:
    return f"bit:{byte_range[0]}:{byte_range[1]}:{bit_range[0]}:{bit_range[1]}"


def fill_uncovered_byte_gaps(ranges: set[ByteRange], payload_len: int) -> set[ByteRange]:
    if payload_len <= 0:
        return set()
    covered = [False] * payload_len
    for start, end in ranges:
        start = max(start, 0)
        end = min(end, payload_len - 1)
        if start > end:
            continue
        for index in range(start, end + 1):
            covered[index] = True

    gaps: set[ByteRange] = set()
    index = 0
    while index < payload_len:
        if covered[index]:
            index += 1
            continue
        start = index
        while index + 1 < payload_len and not covered[index + 1]:
            index += 1
        gaps.add((start, index))
        index += 1
    return gaps


def fill_uncovered_bit_gaps(bitfields: dict[ByteRange, set[tuple[int, int]]]) -> dict[ByteRange, set[tuple[int, int]]]:
    gaps: dict[ByteRange, set[tuple[int, int]]] = {}
    for byte_range, bit_ranges in bitfields.items():
        total_bits = (byte_range[1] - byte_range[0] + 1) * 8
        if total_bits <= 0:
            continue
        covered = [False] * total_bits
        for low, high in bit_ranges:
            low = max(low, 0)
            high = min(high, total_bits - 1)
            if low > high:
                continue
            for bit in range(low, high + 1):
                covered[bit] = True

        bit = 0
        while bit < total_bits:
            if covered[bit]:
                bit += 1
                continue
            low = bit
            while bit + 1 < total_bits and not covered[bit + 1]:
                bit += 1
            gaps.setdefault(byte_range, set()).add((low, bit))
            bit += 1
    return gaps


def field_rows(path: Path, obj: dict[str, Any]) -> list[dict[str, Any]]:
    seq, protocol, pkt = parse_filename(path)
    fields = obj.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("'fields' must be a list")
    rows: list[dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        rows.append(
            {
                "seq": seq if seq is not None else "",
                "protocol_name": protocol,
                "sample_id": pkt,
                "field_id": normalize_field_id(field.get("field_id", "")),
                "program_log_description": field.get("program_log_description", ""),
                "needs_review": field.get("needs_review", ""),
                "review_reason": field.get("review_reason", ""),
                "field_partition_evidence": field.get("field_partition_evidence", []),
                "observed_behaviors": field.get("observed_behaviors", []),
                "packet_summary": obj.get("packet_summary", {}),
                "source_output": str(path),
            }
        )
    return rows


def gap_row(
    seq: Any,
    protocol: str,
    pkt: str,
    field_id: str,
    source_output: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "protocol_name": protocol,
        "sample_id": pkt,
        "field_id": field_id,
        "program_log_description": f"Auto-filled {kind} gap field not explicitly described by the program-log LLM output.",
        "needs_review": True,
        "review_reason": "auto_filled_gap_field",
        "field_partition_evidence": [],
        "observed_behaviors": [],
        "packet_summary": {"auto_filled_gap_field": True},
        "source_output": source_output,
    }


def add_gap_rows(rows: list[dict[str, Any]], replay_root: Path) -> list[dict[str, Any]]:
    by_packet: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_packet[(str(row.get("protocol_name") or ""), str(row.get("sample_id") or ""))].append(row)

    output = list(rows)
    existing_ids = {
        (
            str(row.get("protocol_name") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("field_id") or ""),
        )
        for row in rows
    }
    for (protocol, pkt), packet_rows in sorted(by_packet.items()):
        if not protocol or not pkt:
            continue
        payload_len = payload_len_for(replay_root, protocol, pkt)
        if payload_len <= 0:
            continue

        covered_byte_ranges: set[ByteRange] = set()
        bitfields: dict[ByteRange, set[tuple[int, int]]] = defaultdict(set)
        seq = packet_rows[0].get("seq", "")
        source_output = packet_rows[0].get("source_output", "")

        for row in packet_rows:
            parsed = parse_field_id(str(row.get("field_id") or ""))
            if parsed is None:
                continue
            kind, byte_range, bit_range = parsed
            if byte_range[1] >= payload_len or byte_range[0] < 0:
                continue
            covered_byte_ranges.add(byte_range)
            if kind == "bit" and bit_range is not None:
                bitfields[byte_range].add(bit_range)

        for byte_range in sorted(fill_uncovered_byte_gaps(covered_byte_ranges, payload_len)):
            field_id = field_id_for_byte_range(byte_range)
            row_key = (protocol, pkt, field_id)
            if row_key in existing_ids:
                continue
            output.append(gap_row(seq, protocol, pkt, field_id, str(source_output), "byte"))
            existing_ids.add(row_key)

        for byte_range, bit_ranges in sorted(fill_uncovered_bit_gaps(bitfields).items()):
            for bit_range in sorted(bit_ranges):
                field_id = field_id_for_bit_range(byte_range, bit_range)
                row_key = (protocol, pkt, field_id)
                if row_key in existing_ids:
                    continue
                output.append(gap_row(seq, protocol, pkt, field_id, str(source_output), "bit"))
                existing_ids.add(row_key)

    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seq",
        "protocol_name",
        "sample_id",
        "field_id",
        "program_log_description",
        "needs_review",
        "review_reason",
        "field_partition_evidence",
        "observed_behaviors",
        "packet_summary",
        "source_output",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ("field_partition_evidence", "observed_behaviors", "packet_summary"):
                csv_row[key] = json_dumps(csv_row[key])
            writer.writerow(csv_row)


def write_errors(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_output", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.input_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in inputs:
        try:
            obj = extract_json_object(path.read_text(encoding="utf-8", errors="replace"))
            rows.extend(field_rows(path, obj))
            print(f"[collect] ok: {path.name}")
        except Exception as exc:
            errors.append({"source_output": str(path), "error": str(exc)})
            print(f"[collect] error: {path.name}: {exc}")

    original_row_count = len(rows)
    if not args.no_fill_gaps:
        rows = add_gap_rows(rows, args.replay_root)

    jsonl_path = args.output_dir / args.jsonl_name
    csv_path = args.output_dir / args.csv_name
    errors_path = args.output_dir / args.errors_name
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    write_errors(errors_path, errors)
    print(f"[collect] outputs: {len(inputs)}")
    print(f"[collect] field rows: {len(rows)}")
    if not args.no_fill_gaps:
        print(f"[collect] auto-filled gap rows: {len(rows) - original_row_count}")
    print(f"[collect] errors: {len(errors)}")
    print(f"[collect] wrote: {jsonl_path}")
    print(f"[collect] wrote: {csv_path}")
    print(f"[collect] wrote: {errors_path}")


if __name__ == "__main__":
    main()
