#!/usr/bin/env python3
"""Compute FSIBP native Top-1/Top-2 accuracy for rows with known cluster labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/SOTA/FSIBP")
DEFAULT_PREDICTIONS = ROOT / "outputs/semvec/predictions.csv"
DEFAULT_DESCRIPTIONS = ROOT / "outputs/semvec/field-name-description"
DEFAULT_LABELS = ROOT / "data/description-label/description_label.csv"
DEFAULT_OUTDIR = Path("/root/semvec/bitfield_groundtruth/sota_evaluation/out/fsibp_native")
DEFAULT_SENTENCE_MODEL = ROOT / "all-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--field-description-dir", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--description-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--dynamic-labeling",
        action="store_true",
        help="Assign unknown descriptions to the nearest labeled description embedding.",
    )
    parser.add_argument("--sentence-model", type=Path, default=DEFAULT_SENTENCE_MODEL)
    return parser.parse_args()


def load_description_labels(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                result[row[0].strip()] = int(row[1])
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_field_descriptions(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("*_field_name_description.csv")):
        protocol = path.name[: -len("_field_name_description.csv")]
        fields: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2:
                    fields[row[0].strip()] = row[1].strip()
        result[protocol] = fields
    return result


def lookup_description(field_name: str, fields: dict[str, str]) -> str | None:
    if field_name in fields:
        return fields[field_name]
    if field_name.endswith("_raw") and field_name[:-4] in fields:
        return fields[field_name[:-4]]
    return None


def dynamic_description_labels(
    descriptions: set[str],
    known_labels: dict[str, int],
    sentence_model: Path,
) -> dict[str, int]:
    unknown = sorted(description for description in descriptions if description not in known_labels)
    if not unknown:
        return {}
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("--dynamic-labeling requires sentence-transformers in the active environment") from exc
    known = sorted(known_labels)
    model = SentenceTransformer(str(sentence_model))
    known_vectors = model.encode(known)
    unknown_vectors = model.encode(unknown)
    similarities = np.dot(unknown_vectors, known_vectors.T) / (
        np.linalg.norm(unknown_vectors, axis=1, keepdims=True)
        * np.linalg.norm(known_vectors, axis=1)
    )
    return {
        description: known_labels[known[int(np.argmax(scores))]]
        for description, scores in zip(unknown, similarities)
    }


def summary(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    labeled = [row for row in rows if row["groundtruth_cluster"] is not None]
    top1 = sum(row["top1_hit"] for row in labeled)
    top2 = sum(row["top2_hit"] for row in labeled)
    return {
        "group": group,
        "prediction_rows": len(rows),
        "labeled_rows": len(labeled),
        "native_label_coverage": len(labeled) / len(rows) if rows else 0.0,
        "top1_correct": top1,
        "top1_accuracy": top1 / len(labeled) if labeled else 0.0,
        "top2_correct": top2,
        "top2_accuracy": top2 / len(labeled) if labeled else 0.0,
    }


def main() -> None:
    args = parse_args()
    description_labels = load_description_labels(args.description_labels)
    field_descriptions = load_field_descriptions(args.field_description_dir)
    source_rows = read_csv(args.predictions)
    descriptions = {
        description
        for source in source_rows
        for description in [lookup_description(source["field_name"], field_descriptions.get(source["protocol"], {}))]
        if description is not None
    }
    dynamic_labels = (
        dynamic_description_labels(descriptions, description_labels, args.sentence_model)
        if args.dynamic_labeling
        else {}
    )
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        protocol = source["protocol"]
        description = lookup_description(source["field_name"], field_descriptions.get(protocol, {}))
        groundtruth = (
            description_labels.get(description, dynamic_labels.get(description))
            if description is not None
            else None
        )
        top1 = int(source["predicted_cluster_top1"])
        top2_text = source.get("predicted_cluster_top2", "")
        top2 = int(top2_text) if str(top2_text).strip() else None
        rows.append(
            {
                **source,
                "field_description": description or "",
                "groundtruth_cluster": groundtruth,
                "top1_hit": int(groundtruth is not None and top1 == groundtruth),
                "top2_hit": int(groundtruth is not None and groundtruth in {top1, top2}),
                "label_status": (
                    "exact_description_match"
                    if description in description_labels
                    else "dynamic_nearest_description"
                    if description in dynamic_labels
                    else "unlabeled"
                ),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["protocol"]].append(row)
    summaries = [summary(group_rows, protocol) for protocol, group_rows in sorted(grouped.items())]
    summaries.append(summary(rows, "Overall Micro"))

    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "fsibp_native_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary_path = args.outdir / "fsibp_native_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)
    (args.outdir / "fsibp_native_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[evaluate] rows={len(rows)} output={args.outdir}")
    print("[note] unlabeled rows are reported as coverage gaps; they are not silently discarded from coverage")


if __name__ == "__main__":
    main()
