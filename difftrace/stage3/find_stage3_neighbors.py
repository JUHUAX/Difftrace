#!/usr/bin/env python3
"""Find nearest fields around a target field in Stage 3 embedding space."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


KEY_COLUMNS = ["protocol_name", "sample_id", "field_id"]
DEFAULT_EMBEDDINGS_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_ae/ae_latent8/ae_embeddings.csv")


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find nearest Stage 3 neighbors for a target field.")
    parser.add_argument("--embeddings-csv", type=Path, default=DEFAULT_EMBEDDINGS_CSV, help="Input embeddings CSV.")
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Compact target identifier in the form "
            "'protocol-sample_id-field_id', "
            "e.g. snap7-sample_001-bit:2:3:0:7."
        ),
    )
    parser.add_argument("--protocol-name", default=None, help="Target protocol name.")
    parser.add_argument("--field-id", default=None, help="Target field_id, e.g. b:10:11 or bit:2:3:0:7.")
    parser.add_argument("--sample-id", default=None, help="Optional sample_id. Required if protocol+field_id is not unique.")
    parser.add_argument("--top-k", type=int, default=50, help="Number of nearest neighbors to return.")
    parser.add_argument("--same-protocol-only", action="store_true", help="Restrict neighbors to the same protocol.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    if args.target:
        try:
            protocol_name, sample_id, field_id = parse_compact_target(args.target)
        except ValueError as exc:
            parser.error(str(exc))
        if args.protocol_name and args.protocol_name != protocol_name:
            parser.error("--target and --protocol-name conflict.")
        if args.sample_id and args.sample_id != sample_id:
            parser.error("--target and --sample-id conflict.")
        if args.field_id and args.field_id != field_id:
            parser.error("--target and --field-id conflict.")
        args.protocol_name = protocol_name
        args.sample_id = sample_id
        args.field_id = field_id

    if not args.protocol_name or not args.field_id:
        parser.error("You must provide either --target or both --protocol-name and --field-id.")

    return args


def parse_compact_target(value: str) -> tuple[str, str, str]:
    parts = value.split("-", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Invalid --target format. Expected 'protocol-sample_id-field_id', "
            "e.g. snap7-sample_001-bit:2:3:0:7."
        )
    protocol_name, sample_id, field_id = parts
    return protocol_name, sample_id, field_id


def load_rows(path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    latent_cols = [col for col in (reader.fieldnames or []) if col not in KEY_COLUMNS]
    return rows, latent_cols


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def select_target(rows: Sequence[Dict[str, Any]], protocol_name: str, field_id: str, sample_id: str | None) -> Dict[str, Any]:
    candidates = [
        row for row in rows
        if str(row["protocol_name"]) == protocol_name and str(row["field_id"]) == field_id
    ]
    if sample_id is not None:
        candidates = [row for row in candidates if str(row["sample_id"]) == sample_id]
    if not candidates:
        raise ValueError("No matching target field found.")
    if len(candidates) > 1:
        sample_ids = sorted({str(row['sample_id']) for row in candidates})
        raise ValueError(
            "Target is not unique. Please specify --sample-id. "
            f"Candidate sample_ids: {', '.join(sample_ids[:20])}"
        )
    return candidates[0]


def build_neighbor_rows(rows: Sequence[Dict[str, Any]], latent_cols: Sequence[str], target: Dict[str, Any], top_k: int, same_protocol_only: bool) -> List[Dict[str, Any]]:
    target_vec = [parse_float(target[col]) for col in latent_cols]
    target_protocol = str(target["protocol_name"])
    target_sample = str(target["sample_id"])
    target_field = str(target["field_id"])

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        if str(row["protocol_name"]) == target_protocol and str(row["sample_id"]) == target_sample and str(row["field_id"]) == target_field:
            continue
        if same_protocol_only and str(row["protocol_name"]) != target_protocol:
            continue
        vec = [parse_float(row[col]) for col in latent_cols]
        dist = euclidean_distance(target_vec, vec)
        scored.append((dist, row))
    scored.sort(key=lambda item: item[0])

    out: List[Dict[str, Any]] = []
    for rank, (dist, row) in enumerate(scored[:top_k], start=1):
        out.append(
            {
                "rank": rank,
                "distance": dist,
                "protocol_name": row["protocol_name"],
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
            }
        )
    return out


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "distance", "protocol_name", "sample_id", "field_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if not args.embeddings_csv.exists():
        warn(f"Embeddings CSV does not exist: {args.embeddings_csv}")
        return 1

    rows, latent_cols = load_rows(args.embeddings_csv)
    try:
        target = select_target(rows, args.protocol_name, args.field_id, args.sample_id)
    except ValueError as exc:
        warn(str(exc))
        return 1

    neighbors = build_neighbor_rows(
        rows=rows,
        latent_cols=latent_cols,
        target=target,
        top_k=args.top_k,
        same_protocol_only=args.same_protocol_only,
    )

    print("TARGET")
    print(f"protocol_name={target['protocol_name']}")
    print(f"sample_id={target['sample_id']}")
    print(f"field_id={target['field_id']}")
    print()
    print("NEIGHBORS")
    for row in neighbors:
        print(
            f"{row['rank']:>3}  dist={row['distance']:.6f}  "
            f"{row['protocol_name']}  {row['sample_id']}  {row['field_id']}"
        )

    if args.output_csv is not None:
        write_csv(neighbors, args.output_csv)
        print()
        print(f"Wrote neighbors to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
