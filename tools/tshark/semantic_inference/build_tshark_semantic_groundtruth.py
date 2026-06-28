#!/usr/bin/env python3
"""Build rule-based tshark semantic groundtruth candidates.

The script consumes the structured tshark JSON files produced by
evaluation_from_tshark/field_boundary/generate_groundtruthA_tshark.py and emits
a reviewable CSV using the schema defined in README_EVALUATION_GROUNDTRUTH_PLAN.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path("/root/semvec/bitfield_groundtruth")
DEFAULT_INPUT_DIR = REPO_ROOT / "groundtruth_from_tshark"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_from_tshark" / "semantic_inference" / "eval"

PCAP_TO_PROTOCOL = {
    "S7server": "snap7",
    "mms": "iec61850",
}

CSV_COLUMNS = [
    "protocol_name",
    "sample_id",
    "field_id",
    "source",
    "source_protocol",
    "source_field_name",
    "source_parent_field",
    "source_showname",
    "source_display_value",
    "semantic_label",
    "semantic_group",
    "semantic_tags",
    "needs_review",
    "note",
]


@dataclass(frozen=True)
class SemanticGuess:
    label: str
    group: str
    tags: tuple[str, ...] = ()
    needs_review: bool = False
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rule-based tshark semantic groundtruth candidates."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing groundtruth_from_tshark-style JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and summary outputs.",
    )
    parser.add_argument(
        "--candidate-name",
        default="tshark_semantic_groundtruth_candidates.csv",
        help="Candidate CSV filename under --output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        default="tshark_semantic_groundtruth_summary.json",
        help="Summary JSON filename under --output-dir.",
    )
    return parser.parse_args()


def protocol_name_from_path(path: Path) -> str:
    return PCAP_TO_PROTOCOL.get(path.stem, path.stem.lower())


def sample_id(packet: dict) -> str:
    packet_index = packet.get("packet_index")
    if isinstance(packet_index, int) and packet_index > 0:
        return f"sample_{packet_index:03d}"
    return "sample_000"


def packet_payload_base(packet: dict) -> Optional[int]:
    offsets = [
        proto.get("offset")
        for proto in packet.get("protocols", [])
        if isinstance(proto.get("offset"), int)
    ]
    if not offsets:
        return None
    return min(offsets)


def field_id_for(field: dict, payload_base: int) -> Optional[str]:
    offset = field.get("field_offset")
    length = field.get("field_length")
    if not isinstance(offset, int) or not isinstance(length, int) or length <= 0:
        return None

    start = offset - payload_base
    end = start + length - 1
    if start < 0:
        return None

    bit_offsets = field.get("bit_offset")
    if isinstance(bit_offsets, list) and bit_offsets:
        bits = [b for b in bit_offsets if isinstance(b, int)]
        if bits:
            return f"bit:{start}:{end}:{min(bits)}:{max(bits)}"

    return f"b:{start}:{end}"


def text_of(field: dict) -> str:
    parts = [
        field.get("field_name"),
        field.get("parent_field"),
        field.get("showname"),
        field.get("display_value"),
    ]
    return " ".join(str(p) for p in parts if p).lower()


def name_of(field: dict) -> str:
    return str(field.get("field_name") or "").lower()


def has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def has_token(text: str, token: str) -> bool:
    return re.search(rf"(^|[._\\-\\s]){re.escape(token)}($|[._\\-\\s])", text) is not None


def is_constant_like(field: dict, text: str) -> bool:
    raw_name = name_of(field)
    show = str(field.get("showname") or "").lower()
    value = str(field.get("display_value") or "").lower()

    if has_any(raw_name, ["version", "prot_id", "protocol_id", "protid"]):
        return True
    if has_any(text, ["reserved", "padding", "shall be zero", "must be zero"]):
        return True
    if "protocol identifier" in show and value in {"0", "0x00000000"}:
        return True
    return False


def tags_for(field: dict, text: str) -> list[str]:
    tags: list[str] = []
    structural_text = " ".join(
        str(field.get(key) or "").lower()
        for key in ("field_name", "parent_field")
    )
    if is_constant_like(field, text):
        tags.append("constant_like")
    if has_any(
        structural_text,
        ["lvt", "tag", "segment", "ber.", "wrapper", "variablespecification"],
    ):
        tags.append("structure_or_encoding")
    return tags


def guess_timestamp_label(text: str) -> str:
    if "millisecond" in text or ".milliseconds" in text or "timestamp_ms" in text:
        return "timestamp_ms"
    if "minute" in text:
        return "timestamp_minute"
    if "hour" in text:
        return "timestamp_hour"
    if "day" in text or "dayof" in text:
        return "timestamp_day"
    if "month" in text:
        return "timestamp_month"
    if "year" in text:
        return "timestamp_year"
    return "value"


def rule_guess(field: dict) -> SemanticGuess:
    name = name_of(field)
    text = text_of(field)
    tags = tags_for(field, text)
    bitmask = field.get("bitmask")

    def out(label: str, group: str, *, review: bool = False, note: str = "") -> SemanticGuess:
        return SemanticGuess(label, group, tuple(tags), review, note)

    if has_any(text, ["reserved", "padding"]):
        return out("reserved", "other_or_unknown")

    if has_any(name, ["trans_id", "transaction"]):
        return out("transaction_id", "identifier")
    if has_any(name, ["prot_id", "protocol_id"]) or "protocol identifier" in text:
        return out("protocol_id", "identifier")
    if has_any(name, ["version"]):
        return out("protocol_id", "identifier", note="version treated as protocol identifier")
    if has_any(name, ["session"]):
        return out("session_id", "identifier")
    if has_any(name, ["context"]):
        return out("session_id", "identifier")
    if has_any(name, ["invoke"]):
        return out("invoke_id", "identifier")
    if has_any(name, ["pduref", "pdu_ref", "pdu-reference", "pdu.reference"]):
        return out("pdu_reference", "identifier")
    if has_token(name, "rx") or has_token(name, "tx"):
        return out("pdu_reference", "identifier", note="transport/APDU sequence counter")
    if has_any(name, ["protid"]):
        return out("protocol_id", "identifier")
    if has_any(name, ["unit_id", "unitid"]):
        return out("unit_id", "identifier")
    if has_any(name, ["domainid", "domain_id"]):
        return out("domain_id", "identifier")
    if has_any(name, ["itemid", "item_id"]):
        return out("item_id", "identifier")
    if has_any(name, ["iface", "serial_num", "vendor"]):
        return out("session_id", "identifier")

    if (
        has_any(name, ["length", ".len", "_len", "apdulen", "byte_cnt", "byte_count", "size"])
        or has_token(name, "li")
        or "length:" in text
        or "byte count" in text
    ):
        return out("length", "length_or_count")
    if has_any(name, ["count", "_cnt", ".cnt", "numix", "quantity", "qty"]) or has_any(
        text, [" count:", "quantity"]
    ):
        label = "quantity" if has_any(name, ["quantity", "qty"]) else "count"
        return out(label, "length_or_count")

    if has_any(name, ["func_code", "function", "command"]) or has_token(name, "func"):
        return out("function_code", "control_or_flags")
    if has_any(name, ["service"]):
        return out("service_code", "control_or_flags")
    if has_any(name, ["cause"]):
        return out("cause_of_transmission", "control_or_flags")
    if has_any(name, ["status", "error"]):
        return out("status", "control_or_flags")
    if has_any(name, ["rosctr", "typeid", "type_id"]) or has_token(name, "type"):
        return out("type_id", "control_or_flags")
    if has_any(name, ["start", "qos", "timeout", "control", "option", "priority", "pduflags", "flags"]) or bitmask:
        return out("flag", "control_or_flags")

    if has_any(name, ["common_address", "causetx_originator", "originator"]):
        return out("common_address", "addressing")
    if has_token(name, "oa"):
        return out("common_address", "addressing")
    if has_any(name, ["object_address", "ioa"]):
        return out("object_address", "addressing")
    if has_any(name, ["reference_num", "regnum", "reference", "addr", "address", "destref", "srcref"]):
        return out("address", "addressing")
    if has_any(name, ["class_id", ".class", "_class"]):
        return out("class_id", "addressing")
    if has_any(name, ["instance"]):
        return out("instance_id", "addressing")
    if has_any(name, ["attribute"]):
        return out("attribute_id", "addressing")
    if has_any(text, ["device instance range", "property identifier"]):
        return out("instance_id", "addressing")

    if has_any(text, ["cp56time", "timestamp", "utc_time", "time of day", "milliseconds"]):
        return out(guess_timestamp_label(text), "data_value")
    if has_any(name, ["mask"]):
        return out("mask", "data_value")
    if has_any(name, ["regval", "value", "data", "boolean", "integer", "float", "payload"]):
        label = "payload_data" if has_any(name, ["payload", "data"]) else "value"
        return out(label, "data_value")

    if "structure_or_encoding" in tags:
        return out(
            "unknown",
            "other_or_unknown",
            review=True,
            note="structure/encoding wrapper; review if it maps to a finer semantic role",
        )

    return out("unknown", "other_or_unknown", review=True, note="no rule matched")


def iter_candidate_rows(input_path: Path) -> Iterable[dict]:
    protocol_name = protocol_name_from_path(input_path)
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for packet in data.get("packets", []):
        base = packet_payload_base(packet)
        if base is None:
            continue

        sid = sample_id(packet)
        for field in packet.get("fields", []):
            fid = field_id_for(field, base)
            if fid is None:
                continue

            guess = rule_guess(field)
            note_parts = []
            if field.get("has_children"):
                note_parts.append("aggregate field with child fields")
            if guess.note:
                note_parts.append(guess.note)

            yield {
                "protocol_name": protocol_name,
                "sample_id": sid,
                "field_id": fid,
                "source": "tshark",
                "source_protocol": field.get("protocol") or "",
                "source_field_name": field.get("field_name") or "",
                "source_parent_field": field.get("parent_field") or "",
                "source_showname": field.get("showname") or "",
                "source_display_value": field.get("display_value") or "",
                "semantic_label": guess.label,
                "semantic_group": guess.group,
                "semantic_tags": ",".join(guess.tags),
                "needs_review": "true" if guess.needs_review else "false",
                "note": "; ".join(note_parts),
            }


def build_summary(rows: list[dict], skipped_files: list[str]) -> dict:
    by_protocol = Counter(row["protocol_name"] for row in rows)
    by_group = Counter(row["semantic_group"] for row in rows)
    by_label = Counter(row["semantic_label"] for row in rows)
    review_by_protocol: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["needs_review"] == "true":
            review_by_protocol[row["protocol_name"]] += 1

    return {
        "row_count": len(rows),
        "protocol_count": len(by_protocol),
        "by_protocol": dict(sorted(by_protocol.items())),
        "by_semantic_group": dict(sorted(by_group.items())),
        "top_semantic_labels": dict(by_label.most_common(30)),
        "needs_review_count": sum(1 for row in rows if row["needs_review"] == "true"),
        "needs_review_by_protocol": dict(sorted(review_by_protocol.items())),
        "skipped_files": skipped_files,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.json"))
    if not input_paths:
        raise SystemExit(f"No JSON files found under {args.input_dir}")

    rows: list[dict] = []
    skipped_files: list[str] = []
    for input_path in input_paths:
        try:
            rows.extend(iter_candidate_rows(input_path))
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            skipped_files.append(f"{input_path}: {exc}")

    rows.sort(
        key=lambda row: (
            row["protocol_name"],
            row["sample_id"],
            row["field_id"],
            row["source_field_name"],
        )
    )

    output_dir = args.output_dir
    csv_path = output_dir / args.candidate_name
    summary_path = output_dir / args.summary_name
    write_csv(csv_path, rows)
    summary = build_summary(rows, skipped_files)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[ok] wrote {len(rows)} rows to {csv_path}")
    print(f"[ok] wrote summary to {summary_path}")
    if skipped_files:
        print(f"[warn] skipped {len(skipped_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
