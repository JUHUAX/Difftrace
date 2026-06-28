#!/usr/bin/env python3
"""Train a PCA baseline on the Stage 3 training matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from build_stage3_training_matrix import KEY_COLUMNS
from check_stage3_dataset import MODEL_FEATURE_COLUMNS


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_training_matrix/stage3_training_matrix.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_pca")
DEFAULT_OUTPUT_EMBEDDINGS_CSV = DEFAULT_OUTPUT_DIR / "stage3_pca_embeddings.csv"
DEFAULT_OUTPUT_LOADINGS_CSV = DEFAULT_OUTPUT_DIR / "stage3_pca_loadings.csv"
DEFAULT_OUTPUT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "stage3_pca_summary.json"
DEFAULT_OUTPUT_EXPLAINED_CSV = DEFAULT_OUTPUT_DIR / "stage3_pca_explained_variance.csv"


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA baseline on Stage 3 training matrix.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input scaled training matrix CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--output-embeddings-csv", type=Path, default=DEFAULT_OUTPUT_EMBEDDINGS_CSV, help="Per-field PCA embedding CSV.")
    parser.add_argument("--output-loadings-csv", type=Path, default=DEFAULT_OUTPUT_LOADINGS_CSV, help="Feature loading CSV.")
    parser.add_argument("--output-summary-json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON, help="PCA summary JSON.")
    parser.add_argument("--output-explained-csv", type=Path, default=DEFAULT_OUTPUT_EXPLAINED_CSV, help="Explained variance CSV.")
    parser.add_argument("--components", type=int, default=8, help="Number of PCA components to export.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_embeddings_csv == DEFAULT_OUTPUT_EMBEDDINGS_CSV:
        args.output_embeddings_csv = args.output_dir / DEFAULT_OUTPUT_EMBEDDINGS_CSV.name
    if args.output_loadings_csv == DEFAULT_OUTPUT_LOADINGS_CSV:
        args.output_loadings_csv = args.output_dir / DEFAULT_OUTPUT_LOADINGS_CSV.name
    if args.output_summary_json == DEFAULT_OUTPUT_SUMMARY_JSON:
        args.output_summary_json = args.output_dir / DEFAULT_OUTPUT_SUMMARY_JSON.name
    if args.output_explained_csv == DEFAULT_OUTPUT_EXPLAINED_CSV:
        args.output_explained_csv = args.output_dir / DEFAULT_OUTPUT_EXPLAINED_CSV.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            converted = dict(row)
            for col in MODEL_FEATURE_COLUMNS:
                converted[col] = parse_float(row.get(col))
            rows.append(converted)
    return rows


def build_matrix(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in MODEL_FEATURE_COLUMNS] for row in rows], dtype=float)


def run_pca(matrix: np.ndarray, n_components: int) -> Dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(n_components, vt.shape[0], matrix.shape[0], matrix.shape[1])
    components = vt[:rank]
    embeddings = centered @ components.T
    if matrix.shape[0] > 1:
        explained_variance = (s[:rank] ** 2) / (matrix.shape[0] - 1)
        total_variance = (s ** 2).sum() / (matrix.shape[0] - 1)
    else:
        explained_variance = np.zeros(rank, dtype=float)
        total_variance = 0.0
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros(rank, dtype=float)
    return {
        "components": components,
        "embeddings": embeddings,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": np.cumsum(explained_variance_ratio),
        "rank": rank,
    }


def write_embeddings(rows: Sequence[Dict[str, Any]], embeddings: np.ndarray, path: Path) -> None:
    ensure_parent(path)
    fieldnames = KEY_COLUMNS + [f"pc{i+1}" for i in range(embeddings.shape[1])]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {col: row.get(col) for col in KEY_COLUMNS}
            for i in range(embeddings.shape[1]):
                out[f"pc{i+1}"] = float(embeddings[idx, i])
            writer.writerow(out)


def write_loadings(components: np.ndarray, path: Path) -> None:
    ensure_parent(path)
    fieldnames = ["feature_name"] + [f"pc{i+1}" for i in range(components.shape[0])]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feat_idx, feature_name in enumerate(MODEL_FEATURE_COLUMNS):
            row = {"feature_name": feature_name}
            for comp_idx in range(components.shape[0]):
                row[f"pc{comp_idx+1}"] = float(components[comp_idx, feat_idx])
            writer.writerow(row)


def write_explained(pca: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    fieldnames = ["component", "explained_variance", "explained_variance_ratio", "cumulative_variance_ratio"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(int(pca["rank"])):
            writer.writerow(
                {
                    "component": idx + 1,
                    "explained_variance": float(pca["explained_variance"][idx]),
                    "explained_variance_ratio": float(pca["explained_variance_ratio"][idx]),
                    "cumulative_variance_ratio": float(pca["cumulative_variance_ratio"][idx]),
                }
            )


def build_summary(rows: Sequence[Dict[str, Any]], pca: Dict[str, Any]) -> Dict[str, Any]:
    protocol_counts: Dict[str, int] = {}
    for row in rows:
        protocol = str(row["protocol_name"])
        protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
    return {
        "row_count": len(rows),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "component_count": int(pca["rank"]),
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "explained_variance_ratio": [float(x) for x in pca["explained_variance_ratio"]],
        "cumulative_variance_ratio": [float(x) for x in pca["cumulative_variance_ratio"]],
    }


def main() -> int:
    args = parse_args()
    resolve_output_paths(args)
    if not args.input_csv.exists():
        warn(f"Input CSV does not exist: {args.input_csv}")
        return 1

    rows = load_rows(args.input_csv)
    if not rows:
        warn(f"No rows found in input CSV: {args.input_csv}")
        return 1

    matrix = build_matrix(rows)
    pca = run_pca(matrix, args.components)
    write_embeddings(rows, pca["embeddings"], args.output_embeddings_csv)
    write_loadings(pca["components"], args.output_loadings_csv)
    write_explained(pca, args.output_explained_csv)
    summary = build_summary(rows, pca)
    ensure_parent(args.output_summary_json)
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info(f"Wrote PCA embeddings to {args.output_embeddings_csv}", args.quiet)
    info(f"Wrote PCA loadings to {args.output_loadings_csv}", args.quiet)
    info(f"Wrote PCA summary to {args.output_summary_json}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
