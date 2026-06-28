#!/usr/bin/env python3
"""Analyze Stage 3 PCA / AE latent spaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


KEY_COLUMNS = ["protocol_name", "sample_id", "field_id"]

DEFAULT_SEMANTIC_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_filtered/stage3_dataset_semantic_fields.csv")
DEFAULT_PCA_EMBEDDINGS = Path("/root/semvec/difftrace/stage3/out/stage3_pca/stage3_pca_embeddings.csv")
DEFAULT_AE_ROOT = Path("/root/semvec/difftrace/stage3/out/stage3_ae")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_latent_analysis")

DEFAULT_OUTPUT_REPORT = DEFAULT_OUTPUT_DIR / "latent_structure_report.json"
DEFAULT_OUTPUT_REPRESENTATIVES = DEFAULT_OUTPUT_DIR / "representative_fields.csv"
DEFAULT_OUTPUT_OUTLIERS = DEFAULT_OUTPUT_DIR / "outlier_fields.csv"
DEFAULT_OUTPUT_PROTOCOL_SEPARATION = DEFAULT_OUTPUT_DIR / "space_separation_by_protocol.csv"
DEFAULT_OUTPUT_DIAGNOSTICS = DEFAULT_OUTPUT_DIR / "space_separation_by_diagnostics.csv"


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PCA / AE latent spaces for Stage 3.")
    parser.add_argument("--semantic-csv", type=Path, default=DEFAULT_SEMANTIC_CSV, help="Input semantic_fields CSV.")
    parser.add_argument("--pca-embeddings-csv", type=Path, default=DEFAULT_PCA_EMBEDDINGS, help="PCA embeddings CSV.")
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT, help="AE output root containing ae_latent*/.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT, help="Latent structure report JSON.")
    parser.add_argument("--output-representatives", type=Path, default=DEFAULT_OUTPUT_REPRESENTATIVES, help="Representative fields CSV.")
    parser.add_argument("--output-outliers", type=Path, default=DEFAULT_OUTPUT_OUTLIERS, help="Outlier fields CSV.")
    parser.add_argument("--output-protocol-separation", type=Path, default=DEFAULT_OUTPUT_PROTOCOL_SEPARATION, help="Protocol separation CSV.")
    parser.add_argument("--output-diagnostics", type=Path, default=DEFAULT_OUTPUT_DIAGNOSTICS, help="Diagnostics grouping CSV.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k representatives / outliers to export per source and scope.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_report == DEFAULT_OUTPUT_REPORT:
        args.output_report = args.output_dir / DEFAULT_OUTPUT_REPORT.name
    if args.output_representatives == DEFAULT_OUTPUT_REPRESENTATIVES:
        args.output_representatives = args.output_dir / DEFAULT_OUTPUT_REPRESENTATIVES.name
    if args.output_outliers == DEFAULT_OUTPUT_OUTLIERS:
        args.output_outliers = args.output_dir / DEFAULT_OUTPUT_OUTLIERS.name
    if args.output_protocol_separation == DEFAULT_OUTPUT_PROTOCOL_SEPARATION:
        args.output_protocol_separation = args.output_dir / DEFAULT_OUTPUT_PROTOCOL_SEPARATION.name
    if args.output_diagnostics == DEFAULT_OUTPUT_DIAGNOSTICS:
        args.output_diagnostics = args.output_dir / DEFAULT_OUTPUT_DIAGNOSTICS.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def row_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row["protocol_name"]), str(row["sample_id"]), str(row["field_id"]))


def load_semantic_rows(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for row in reader:
            converted = dict(row)
            for field in ["constraint_count", "unique_metric_vectors"]:
                converted[field] = int(float(row.get(field, 0) or 0))
            converted["deltaf_dispersion"] = parse_float(row.get("deltaf_dispersion"))
            rows[row_key(converted)] = converted
    return rows


def mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            acc[i] += float(value)
    return [value / len(vectors) for value in acc]


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def vector_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in a))


def load_embedding_rows(path: Path, semantic_index: Dict[Tuple[str, str, str], Dict[str, Any]], source_name: str) -> tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        latent_cols = [col for col in fieldnames if col not in KEY_COLUMNS]
        rows: List[Dict[str, Any]] = []
        for row in reader:
            key = row_key(row)
            semantic = semantic_index.get(key)
            if semantic is None:
                continue
            merged = dict(semantic)
            merged.update({col: row.get(col) for col in KEY_COLUMNS})
            merged["source_name"] = source_name
            merged["_latent_vector"] = [parse_float(row.get(col)) for col in latent_cols]
            rows.append(merged)
    return rows, latent_cols


def discover_sources(args: argparse.Namespace, semantic_index: Dict[Tuple[str, str, str], Dict[str, Any]], quiet: bool) -> List[tuple[str, List[Dict[str, Any]], List[str]]]:
    sources: List[tuple[str, List[Dict[str, Any]], List[str]]] = []
    if args.pca_embeddings_csv.exists():
        rows, latent_cols = load_embedding_rows(args.pca_embeddings_csv, semantic_index, "pca")
        if rows:
            sources.append(("pca", rows, latent_cols))
            info(f"Loaded PCA embeddings: {len(rows)} rows", quiet)
    if args.ae_root.exists():
        for run_dir in sorted(path for path in args.ae_root.iterdir() if path.is_dir() and path.name.startswith("ae_latent")):
            emb_path = run_dir / "ae_embeddings.csv"
            if not emb_path.exists():
                continue
            rows, latent_cols = load_embedding_rows(emb_path, semantic_index, run_dir.name)
            if rows:
                sources.append((run_dir.name, rows, latent_cols))
                info(f"Loaded {run_dir.name}: {len(rows)} rows", quiet)
    return sources


def representative_rows(rows: Sequence[Dict[str, Any]], centroid: Sequence[float], top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for row in rows:
        dist = euclidean_distance(row["_latent_vector"], centroid)
        scored.append((dist, row))
    scored.sort(key=lambda item: item[0])
    out: List[Dict[str, Any]] = []
    for rank, (dist, row) in enumerate(scored[:top_k], start=1):
        out.append(
            {
                "rank": rank,
                "protocol_name": row["protocol_name"],
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
                "field_kind": row.get("field_kind"),
                "constraint_count": row.get("constraint_count"),
                "unique_metric_vectors": row.get("unique_metric_vectors"),
                "deltaf_dispersion": row.get("deltaf_dispersion"),
                "distance_to_centroid": dist,
                "embedding_norm": vector_norm(row["_latent_vector"]),
            }
        )
    return out


def outlier_rows(rows: Sequence[Dict[str, Any]], global_centroid: Sequence[float], protocol_centroid: Sequence[float], top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for row in rows:
        dist_global = euclidean_distance(row["_latent_vector"], global_centroid)
        dist_protocol = euclidean_distance(row["_latent_vector"], protocol_centroid)
        norm = vector_norm(row["_latent_vector"])
        scored.append((dist_protocol, dist_global, norm, row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    out: List[Dict[str, Any]] = []
    for rank, (dist_protocol, dist_global, norm, row) in enumerate(scored[:top_k], start=1):
        out.append(
            {
                "rank": rank,
                "protocol_name": row["protocol_name"],
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
                "field_kind": row.get("field_kind"),
                "constraint_count": row.get("constraint_count"),
                "unique_metric_vectors": row.get("unique_metric_vectors"),
                "deltaf_dispersion": row.get("deltaf_dispersion"),
                "distance_to_protocol_centroid": dist_protocol,
                "distance_to_global_centroid": dist_global,
                "embedding_norm": norm,
            }
        )
    return out


def bucket_constraint_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 6:
        return "4-6"
    return "7+"


def bucket_unique_vectors(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    return "5+"


def summarize_group(rows: Sequence[Dict[str, Any]], centroid: Sequence[float]) -> Dict[str, float]:
    dists = [euclidean_distance(row["_latent_vector"], centroid) for row in rows]
    norms = [vector_norm(row["_latent_vector"]) for row in rows]
    return {
        "field_count": len(rows),
        "mean_distance_to_centroid": sum(dists) / len(dists) if dists else 0.0,
        "mean_embedding_norm": sum(norms) / len(norms) if norms else 0.0,
    }


def analyze_source(source_name: str, rows: Sequence[Dict[str, Any]], latent_cols: Sequence[str], top_k: int) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    vectors = [row["_latent_vector"] for row in rows]
    global_centroid = mean_vector(vectors)
    protocol_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        protocol_groups[str(row["protocol_name"])].append(row)

    report: Dict[str, Any] = {
        "source_name": source_name,
        "row_count": len(rows),
        "latent_dim": len(latent_cols),
        "protocol_counts": {proto: len(group) for proto, group in sorted(protocol_groups.items())},
        "protocol_centroids": {},
        "top_protocol_separations": [],
    }
    representatives: List[Dict[str, Any]] = []
    outliers: List[Dict[str, Any]] = []
    protocol_separation_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []

    representatives.extend(
        {
            "source_name": source_name,
            "scope": "global",
            "scope_value": "all",
            **row,
        }
        for row in representative_rows(rows, global_centroid, top_k)
    )

    for proto, group in sorted(protocol_groups.items()):
        centroid = mean_vector([row["_latent_vector"] for row in group])
        report["protocol_centroids"][proto] = {
            "field_count": len(group),
            "centroid_norm": vector_norm(centroid),
            "mean_distance_to_protocol_centroid": summarize_group(group, centroid)["mean_distance_to_centroid"],
        }
        representatives.extend(
            {
                "source_name": source_name,
                "scope": "protocol",
                "scope_value": proto,
                **row,
            }
            for row in representative_rows(group, centroid, top_k)
        )
        outliers.extend(
            {
                "source_name": source_name,
                "scope": "protocol",
                "scope_value": proto,
                **row,
            }
            for row in outlier_rows(group, global_centroid, centroid, top_k)
        )

    protocols = sorted(protocol_groups)
    pairwise = []
    for i, left in enumerate(protocols):
        left_centroid = mean_vector([row["_latent_vector"] for row in protocol_groups[left]])
        for right in protocols[i + 1 :]:
            right_centroid = mean_vector([row["_latent_vector"] for row in protocol_groups[right]])
            dist = euclidean_distance(left_centroid, right_centroid)
            protocol_separation_rows.append(
                {
                    "source_name": source_name,
                    "left_protocol": left,
                    "right_protocol": right,
                    "left_field_count": len(protocol_groups[left]),
                    "right_field_count": len(protocol_groups[right]),
                    "centroid_distance": dist,
                }
            )
            pairwise.append((dist, left, right))
    pairwise.sort(reverse=True)
    report["top_protocol_separations"] = [
        {"left_protocol": left, "right_protocol": right, "centroid_distance": dist}
        for dist, left, right in pairwise[:10]
    ]

    diag_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        diag_groups[("field_kind", str(row.get("field_kind", "unknown")))].append(row)
        diag_groups[("constraint_count_bucket", bucket_constraint_count(int(row.get("constraint_count", 0))))].append(row)
        diag_groups[("unique_metric_vectors_bucket", bucket_unique_vectors(int(row.get("unique_metric_vectors", 0))))].append(row)
    for (diag_name, diag_value), group in sorted(diag_groups.items()):
        summary = summarize_group(group, global_centroid)
        diagnostics_rows.append(
            {
                "source_name": source_name,
                "diagnostic_name": diag_name,
                "diagnostic_value": diag_value,
                **summary,
            }
        )

    return report, representatives, outliers, protocol_separation_rows, diagnostics_rows


def write_csv(rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in fieldnames})


def main() -> int:
    args = parse_args()
    resolve_output_paths(args)
    if not args.semantic_csv.exists():
        warn(f"Semantic CSV does not exist: {args.semantic_csv}")
        return 1

    semantic_index = load_semantic_rows(args.semantic_csv)
    sources = discover_sources(args, semantic_index, args.quiet)
    if not sources:
        warn("No PCA/AE embedding sources found.")
        return 1

    full_report: Dict[str, Any] = {"sources": {}}
    all_representatives: List[Dict[str, Any]] = []
    all_outliers: List[Dict[str, Any]] = []
    all_protocol_separation: List[Dict[str, Any]] = []
    all_diagnostics: List[Dict[str, Any]] = []

    for source_name, rows, latent_cols in sources:
        report, reps, outs, prot_sep, diags = analyze_source(source_name, rows, latent_cols, args.top_k)
        full_report["sources"][source_name] = report
        all_representatives.extend(reps)
        all_outliers.extend(outs)
        all_protocol_separation.extend(prot_sep)
        all_diagnostics.extend(diags)

    ensure_parent(args.output_report)
    args.output_report.write_text(json.dumps(full_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(
        all_representatives,
        [
            "source_name",
            "scope",
            "scope_value",
            "rank",
            "protocol_name",
            "sample_id",
            "field_id",
            "field_kind",
            "constraint_count",
            "unique_metric_vectors",
            "deltaf_dispersion",
            "distance_to_centroid",
            "embedding_norm",
        ],
        args.output_representatives,
    )
    write_csv(
        all_outliers,
        [
            "source_name",
            "scope",
            "scope_value",
            "rank",
            "protocol_name",
            "sample_id",
            "field_id",
            "field_kind",
            "constraint_count",
            "unique_metric_vectors",
            "deltaf_dispersion",
            "distance_to_protocol_centroid",
            "distance_to_global_centroid",
            "embedding_norm",
        ],
        args.output_outliers,
    )
    write_csv(
        all_protocol_separation,
        [
            "source_name",
            "left_protocol",
            "right_protocol",
            "left_field_count",
            "right_field_count",
            "centroid_distance",
        ],
        args.output_protocol_separation,
    )
    write_csv(
        all_diagnostics,
        [
            "source_name",
            "diagnostic_name",
            "diagnostic_value",
            "field_count",
            "mean_distance_to_centroid",
            "mean_embedding_norm",
        ],
        args.output_diagnostics,
    )

    info(f"Wrote latent structure report to {args.output_report}", args.quiet)
    info(f"Wrote representative fields to {args.output_representatives}", args.quiet)
    info(f"Wrote outlier fields to {args.output_outliers}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
