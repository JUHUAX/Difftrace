#!/usr/bin/env python3
"""Generate a packet-level 70/30 held-out split for DiffTrace."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DIFFTRACE_ROOT = Path("/root/semvec/difftrace/out_frozen/outputs")
DEFAULT_PROGRAM_LOG_JSONL = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval/"
    "program_log_groundtruth_candidates.jsonl"
)
DEFAULT_OUTPUT = Path("/root/semvec/difftrace/stage4/splits/stage4_packet_split_seed0.json")
MIN_FIELD_MATCH_RATIO = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a packet-level held-out split from frozen-baseline Stage 2 outputs.",
    )
    parser.add_argument("--difftrace-root", type=Path, default=DEFAULT_DIFFTRACE_ROOT)
    parser.add_argument("--program-log-jsonl", type=Path, default=DEFAULT_PROGRAM_LOG_JSONL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Write a best-effort manifest even when a protocol has fewer eligible packets than requested.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalize_field_id(unit: dict[str, Any]) -> str:
    kind = str(unit.get("kind") or "byte")
    start = int(unit["a"])
    end = int(unit["b"])
    if kind == "bit":
        bits = [int(bit) for bit in unit.get("bits", [])]
        if not bits:
            raise ValueError(f"bit field is missing bits: {unit}")
        return f"bit:{start}:{end}:{min(bits)}:{max(bits)}"
    return f"b:{start}:{end}"


def load_program_log_fields(path: Path) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_no}: {exc}") from exc
            protocol = str(row.get("protocol_name") or "")
            packet_id = str(row.get("sample_id") or "")
            field_id = str(row.get("field_id") or "")
            if not protocol or not packet_id or not field_id:
                raise ValueError(f"missing protocol_name/sample_id/field_id: {path}:{line_no}")
            result[(protocol, packet_id)].add(field_id)
    return result


def load_difftrace_packets(root: Path) -> dict[str, list[dict[str, Any]]]:
    packets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for protocol_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for sample_dir in sorted(protocol_dir.glob("sample_*")):
            if not sample_dir.is_dir():
                continue
            sample_json = load_json(sample_dir / "sample.json")
            fields_json = load_json(sample_dir / "fields.json")
            packet_id = str(sample_json.get("frozen_packet_id") or "")
            ordinal = int(sample_json.get("frozen_ordinal") or 0)
            if not packet_id or ordinal <= 0:
                raise ValueError(
                    f"{sample_dir}/sample.json is not a frozen-baseline sample: "
                    "missing frozen_packet_id or frozen_ordinal"
                )
            fields = {
                normalize_field_id(unit)
                for unit in fields_json.get("fields", [])
                if isinstance(unit, dict) and "a" in unit and "b" in unit
            }
            if not fields:
                raise ValueError(f"no DiffTrace fields found: {sample_dir}/fields.json")
            packets[protocol_dir.name].append(
                {
                    "sample_id": sample_dir.name,
                    "packet_id": packet_id,
                    "ordinal": ordinal,
                    "fields": fields,
                }
            )
    for protocol in packets:
        packets[protocol].sort(key=lambda item: (item["ordinal"], item["sample_id"]))
    return dict(sorted(packets.items()))


def choose_eval_count(packet_count: int, ratio: float) -> int:
    return max(1, int(packet_count * ratio + 0.5))


def build_manifest(
    difftrace_packets: dict[str, list[dict[str, Any]]],
    program_log_fields: dict[tuple[str, str], set[str]],
    eval_ratio: float,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    rng = random.Random(seed)
    protocols: dict[str, Any] = {}
    shortfalls: list[str] = []
    for protocol, packets in difftrace_packets.items():
        eligible: list[dict[str, Any]] = []
        packet_stats: list[dict[str, Any]] = []
        for packet in packets:
            gt_fields = program_log_fields.get((protocol, packet["packet_id"]), set())
            missing = sorted(packet["fields"] - gt_fields)
            matched = packet["fields"] & gt_fields
            field_match_ratio = len(matched) / len(packet["fields"])
            eligible_for_eval = field_match_ratio >= MIN_FIELD_MATCH_RATIO
            record = {
                "sample_id": packet["sample_id"],
                "packet_id": packet["packet_id"],
                "ordinal": packet["ordinal"],
                "difftrace_field_count": len(packet["fields"]),
                "program_log_field_count": len(gt_fields),
                "matched_field_count": len(matched),
                "field_match_ratio": field_match_ratio,
                "eligible_for_eval": eligible_for_eval,
                "missing_difftrace_fields": missing,
            }
            packet_stats.append(record)
            if eligible_for_eval:
                eligible.append(record)

        requested_eval_count = choose_eval_count(len(packets), eval_ratio)
        if len(eligible) < requested_eval_count:
            shortfalls.append(
                f"{protocol}: requested_eval={requested_eval_count}, eligible={len(eligible)}, "
                f"total_packets={len(packets)}"
            )
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        selected = sorted(
            shuffled[: min(requested_eval_count, len(shuffled))],
            key=lambda item: (item["ordinal"], item["sample_id"]),
        )
        eval_ids = {item["sample_id"] for item in selected}
        protocols[protocol] = {
            "train": [item["sample_id"] for item in packets if item["sample_id"] not in eval_ids],
            "eval": [item["sample_id"] for item in selected],
            "eligible_eval_candidates": [item["sample_id"] for item in eligible],
            "selection_stats": {
                "candidate_rule": (
                    "at least 70% of DiffTrace Stage 2 fields in packet exactly match "
                    "program-log groundtruth"
                ),
                "min_field_match_ratio": MIN_FIELD_MATCH_RATIO,
                "packet_count": len(packets),
                "requested_eval_packet_count": requested_eval_count,
                "eligible_candidate_count": len(eligible),
                "selected_eval_packet_count": len(selected),
                "selected_eval_difftrace_field_count": sum(item["difftrace_field_count"] for item in selected),
                "selected_eval_matched_field_count": sum(item["matched_field_count"] for item in selected),
            },
            "packet_alignment": packet_stats,
        }
    manifest = {
        "split_name": f"stage4_packet_split_seed{seed}",
        "split_unit": "packet",
        "seed": seed,
        "train_ratio": 1.0 - eval_ratio,
        "eval_ratio": eval_ratio,
        "eval_candidate_rule": (
            "at least 70% of DiffTrace Stage 2 fields in an eval packet must exactly match "
            "program-log groundtruth"
        ),
        "min_field_match_ratio": MIN_FIELD_MATCH_RATIO,
        "difftrace_root": str(DEFAULT_DIFFTRACE_ROOT),
        "program_log_jsonl": str(DEFAULT_PROGRAM_LOG_JSONL),
        "protocols": protocols,
        "shortfalls": shortfalls,
    }
    return manifest, shortfalls


def main() -> int:
    args = parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        raise SystemExit(f"--eval-ratio must be between 0 and 1, got {args.eval_ratio}")
    if not args.difftrace_root.is_dir():
        raise SystemExit(f"DiffTrace output root does not exist: {args.difftrace_root}")
    if not args.program_log_jsonl.exists():
        raise SystemExit(f"program-log JSONL does not exist: {args.program_log_jsonl}")

    gt_fields = load_program_log_fields(args.program_log_jsonl)
    packets = load_difftrace_packets(args.difftrace_root)
    manifest, shortfalls = build_manifest(packets, gt_fields, args.eval_ratio, args.seed)
    manifest["difftrace_root"] = str(args.difftrace_root)
    manifest["program_log_jsonl"] = str(args.program_log_jsonl)

    if shortfalls and not args.allow_shortfall:
        print("[split] 70/30 split is not feasible under the 70% field-match rule:", file=sys.stderr)
        for item in shortfalls:
            print(f"[split]   {item}", file=sys.stderr)
        print("[split] no manifest written; correct alignment first or pass --allow-shortfall explicitly", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[split] wrote {args.output}")
    for protocol, payload in manifest["protocols"].items():
        stats = payload["selection_stats"]
        print(
            f"[split] {protocol}: train={len(payload['train'])} eval={len(payload['eval'])} "
            f"eligible={stats['eligible_candidate_count']}/{stats['packet_count']} "
            f"matched_eval_fields={stats['selected_eval_matched_field_count']}"
        )
    if shortfalls:
        print(f"[split] warning: wrote best-effort manifest with {len(shortfalls)} protocol shortfalls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
