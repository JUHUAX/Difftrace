#!/usr/bin/env python3
"""Visualize Stage 3 PCA / AE latent spaces."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - dependency guard
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency guard
    MATPLOTLIB_IMPORT_ERROR = None


DEFAULT_PCA_EMBEDDINGS = Path("/root/semvec/difftrace/stage3/out/stage3_pca/stage3_pca_embeddings.csv")
DEFAULT_AE_EMBEDDINGS = Path("/root/semvec/difftrace/stage3/out/stage3_ae/ae_latent8/ae_embeddings.csv")
DEFAULT_AE_RECON = Path("/root/semvec/difftrace/stage3/out/stage3_ae/ae_latent8/ae_reconstruction_errors.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_latent_plots")


PALETTE = {
    "bacnet": "#0B6E4F",
    "cip": "#C84C09",
    "iec104": "#1D4E89",
    "iec61850": "#7A306C",
    "modbus": "#AA2222",
    "snap7": "#B8860B",
}


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage 3 PCA / AE latent spaces.")
    parser.add_argument("--pca-embeddings-csv", type=Path, default=DEFAULT_PCA_EMBEDDINGS, help="PCA embeddings CSV.")
    parser.add_argument("--ae-embeddings-csv", type=Path, default=DEFAULT_AE_EMBEDDINGS, help="AE embeddings CSV.")
    parser.add_argument("--ae-reconstruction-csv", type=Path, default=DEFAULT_AE_RECON, help="AE reconstruction error CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output plot directory.")
    parser.add_argument("--alpha", type=float, default=0.65, help="Scatter alpha.")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_protocol(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protocol_name"])].append(row)
    return grouped


def scatter_by_protocol(rows: Sequence[Dict[str, Any]], x_col: str, y_col: str, title: str, output_path: Path, alpha: float, point_size: float) -> None:
    grouped = group_by_protocol(rows)
    fig, ax = plt.subplots(figsize=(9, 7))
    for proto, group in sorted(grouped.items()):
        xs = [parse_float(row.get(x_col)) for row in group]
        ys = [parse_float(row.get(y_col)) for row in group]
        ax.scatter(
            xs,
            ys,
            s=point_size,
            alpha=alpha,
            label=proto,
            color=PALETTE.get(proto),
            edgecolors="none",
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.15, linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def reconstruction_histogram(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    values = [parse_float(row.get("reconstruction_mse")) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=60, color="#1D4E89", alpha=0.85)
    ax.set_xlabel("reconstruction_mse")
    ax.set_ylabel("field count")
    ax.set_title("AE Latent8 Reconstruction Error Distribution")
    ax.grid(alpha=0.15, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    if plt is None:
        warn(
            "matplotlib is required for visualization. "
            f"Import error: {MATPLOTLIB_IMPORT_ERROR!r}"
        )
        return 1

    args = parse_args()
    for path in [args.pca_embeddings_csv, args.ae_embeddings_csv, args.ae_reconstruction_csv]:
        if not path.exists():
            warn(f"Input file does not exist: {path}")
            return 1

    ensure_dir(args.output_dir)

    pca_rows = load_csv(args.pca_embeddings_csv)
    ae_rows = load_csv(args.ae_embeddings_csv)
    recon_rows = load_csv(args.ae_reconstruction_csv)

    scatter_by_protocol(
        pca_rows,
        x_col="pc1",
        y_col="pc2",
        title="Stage 3 PCA: PC1 vs PC2",
        output_path=args.output_dir / "pca_pc1_pc2_by_protocol.png",
        alpha=args.alpha,
        point_size=args.point_size,
    )
    scatter_by_protocol(
        ae_rows,
        x_col="z1",
        y_col="z2",
        title="Stage 3 AE Latent8: z1 vs z2",
        output_path=args.output_dir / "ae_latent8_z1_z2_by_protocol.png",
        alpha=args.alpha,
        point_size=args.point_size,
    )
    reconstruction_histogram(
        recon_rows,
        output_path=args.output_dir / "ae_latent8_reconstruction_error_hist.png",
    )

    info(f"Wrote plots to {args.output_dir}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
