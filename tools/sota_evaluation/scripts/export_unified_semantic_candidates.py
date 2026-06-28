#!/usr/bin/env python3
"""Export auditable coarse-tag candidates from canonical SOTA boundary rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/root/semvec/bitfield_groundtruth/sota_evaluation")
DEFAULT_INPUT = ROOT / "out/boundary_predictions.jsonl"
DEFAULT_MAPPING = ROOT / "config/semantic_label_mapping.json"
DEFAULT_OUTPUT = ROOT / "out/unified_semantic_candidates.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def map_binaryinferno(raw_label: str, mapping: dict[str, str]) -> str:
    if raw_label.startswith("Length "):
        return mapping["Length"]
    if "Sequence" in raw_label:
        return mapping["Sequence"]
    if "Checksum" in raw_label:
        return mapping["Checksum"]
    if "Float" in raw_label:
        return mapping["Float"]
    for prefix in ("FieldVar", "FieldRep", "FieldFixed"):
        if raw_label.startswith(prefix):
            return mapping[prefix]
    return "other_or_unknown"


def mapped_tag(method: str, raw_label: str, mapping: dict[str, Any]) -> str:
    if method == "fieldhunter":
        return mapping["fieldhunter"].get(raw_label, "other_or_unknown")
    if method == "binaryinferno":
        return map_binaryinferno(raw_label, mapping["binaryinferno"])
    if method == "binpre":
        try:
            labels = json.loads(raw_label)
        except json.JSONDecodeError:
            return "other_or_unknown"
        for label in labels.get("functions", []):
            if label in mapping["binpre_functions"]:
                return mapping["binpre_functions"][label]
        for label in labels.get("types", []):
            if label in mapping["binpre_types"]:
                return mapping["binpre_types"][label]
    return "other_or_unknown"


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            for semantic in source.get("semantics", []):
                raw_label = str(semantic.get("raw_label", ""))
                start, end = int(semantic["start"]), int(semantic["end"])
                rows.append(
                    {
                        "method": source["method"],
                        "variant": source["variant"],
                        "protocol_name": source["protocol"],
                        "sample_id": source["sample_id"],
                        "field_id": f"b:{start}:{end}",
                        "raw_label": raw_label,
                        "mapped_coarse_tag": mapped_tag(source["method"], raw_label, mapping),
                        "source_status": source["status"],
                        "notes": source.get("notes", ""),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[export] rows={len(rows)} output={args.output}")
    print("[note] this is an auditable candidate inventory, not a scored semantic comparison")


if __name__ == "__main__":
    main()
