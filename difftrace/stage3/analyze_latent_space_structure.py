#!/usr/bin/env python3
"""Analyze learned latent-space structure from existing PCA/AE outputs."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("/root/semvec/difftrace/out/unsupervised_v1")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/latent_space_analysis")
DEFAULT_BASIC_VIS_SUMMARY = Path("/root/semvec/difftrace/out/basic_visual_analysis/basic_visual_analysis_summary.json")
DEFAULT_SUBSET_SUMMARY = Path("/root/semvec/difftrace/out/basic_visual_analysis/subset_summary.csv")

EMBEDDING_META_COLS = [
    "protocol_name",
    "sample_id",
    "field_id",
    "transparent_flag",
    "boundary_miss",
    "mutation_count",
    "unique_metric_vectors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latent-space structure from trained unsupervised models.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing unsupervised_v1 outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for structure analysis outputs")
    parser.add_argument(
        "--dataset-versions",
        choices=["full", "nontransparent", "both"],
        default="both",
        help="Which dataset versions to analyze",
    )
    parser.add_argument(
        "--model-types",
        choices=["pca", "ae", "both"],
        default="both",
        help="Which model families to analyze",
    )
    parser.add_argument(
        "--latent-dims",
        type=str,
        default="2,3,4,8",
        help="Comma-separated latent dims to focus on; for PCA, leading PCs are truncated to these dims when possible",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.10,
        help="Distance threshold on z-scored latent coordinates for near-duplicate ratio",
    )
    parser.add_argument(
        "--collapsed-variance-threshold",
        type=float,
        default=1e-6,
        help="Variance threshold for marking a latent dimension as collapsed",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.98,
        help="Quantile threshold for outlier field extraction by distance to global center",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=5,
        help="How many representative or extreme fields to keep per space/group",
    )
    parser.add_argument(
        "--basic-visual-summary",
        type=Path,
        default=DEFAULT_BASIC_VIS_SUMMARY,
        help="Optional basic_visual_analysis_summary.json path",
    )
    parser.add_argument(
        "--subset-summary",
        type=Path,
        default=DEFAULT_SUBSET_SUMMARY,
        help="Optional subset_summary.csv path",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    return parser.parse_args()


def info(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def ensure_dependencies() -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    missing = []
    for name in [
        "matplotlib",
        "matplotlib.pyplot",
        "sklearn.metrics",
    ]:
        try:
            modules[name] = __import__(name, fromlist=["*"])
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required Python packages for latent-space analysis: "
            f"{missing}. Please install matplotlib and scikit-learn before running this script."
        )
    return modules


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def maybe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def parse_choice_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def selected_dataset_versions(raw: str) -> List[str]:
    return ["full", "nontransparent"] if raw == "both" else [raw]


def selected_model_types(raw: str) -> List[str]:
    return ["pca", "ae"] if raw == "both" else [raw]


def mutation_count_bucket(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bins = [-np.inf, 2, 4, 8, np.inf]
    labels = ["<=2", "3-4", "5-8", ">=9"]
    return pd.cut(vals, bins=bins, labels=labels)


def unique_metric_vectors_bucket(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = ["1", "2", "3-4", ">=5"]
    return pd.cut(vals, bins=bins, labels=labels)


def latent_cols_from_df(df: pd.DataFrame) -> List[str]:
    pca_cols = sorted([col for col in df.columns if col.startswith("pc")], key=lambda x: int(x[2:]))
    if pca_cols:
        return pca_cols
    ae_cols = sorted([col for col in df.columns if col.startswith("z")], key=lambda x: int(x[1:]))
    return ae_cols


def zscore_matrix(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def pairwise_distances_upper(X: np.ndarray) -> np.ndarray:
    if len(X) < 2:
        return np.array([], dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(len(X), k=1)
    return dists[iu]


def centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def mean_distance_to_center(points: np.ndarray, center: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(points - center, axis=1)))


def average_pairwise_distance(points: np.ndarray) -> float:
    d = pairwise_distances_upper(points)
    return float(np.mean(d)) if len(d) else 0.0


def rounded_unique_point_count(points: np.ndarray, decimals: int = 6) -> int:
    if len(points) == 0:
        return 0
    rounded = np.round(points, decimals=decimals)
    return int(np.unique(rounded, axis=0).shape[0])


@dataclass
class LatentSpace:
    dataset_version: str
    model_type: str
    latent_dim: int
    embedding_path: Path
    df: pd.DataFrame
    latent_cols: List[str]
    reconstruction_errors: Optional[pd.DataFrame]
    training_summary: Optional[Dict[str, Any]]
    pca_explained_variance: Optional[List[float]]

    @property
    def space_id(self) -> str:
        return f"{self.dataset_version}_{self.model_type}_latent{self.latent_dim}"


def load_pca_spaces(
    input_dir: Path,
    dataset_version: str,
    target_dims: Sequence[int],
    original_feature_csv: Optional[Path],
    quiet: bool,
) -> List[LatentSpace]:
    spaces: List[LatentSpace] = []
    pca_dir = input_dir / dataset_version / "pca"
    embedding_path = pca_dir / "pca_embedding.csv"
    training_summary_path = pca_dir / "training_summary.json"
    recon_path = pca_dir / "reconstruction_errors.csv"
    if not embedding_path.exists():
        warn(f"Missing PCA embedding: {embedding_path}")
        return spaces

    full_df = pd.read_csv(embedding_path)
    latent_cols = latent_cols_from_df(full_df)
    if not latent_cols:
        warn(f"No PCA latent columns found in {embedding_path}")
        return spaces

    training_summary = maybe_load_json(training_summary_path)
    recon_df = maybe_load_csv(recon_path)
    pca_dim_available = len(latent_cols)
    usable_dims = sorted({dim for dim in target_dims if dim <= pca_dim_available})
    if pca_dim_available not in usable_dims:
        usable_dims.append(pca_dim_available)

    # Attempt to recompute truncated PCA reconstruction errors if original feature matrix is available.
    pca_model_path = pca_dir / "pca_model.pkl"
    scaler_path = pca_dir / "scaler.pkl"
    original_df = None
    if original_feature_csv and original_feature_csv.exists():
        try:
            original_df = pd.read_csv(original_feature_csv)
        except Exception as exc:
            warn(f"Failed to read original feature CSV {original_feature_csv}: {exc}")
            original_df = None

    trunc_recon_by_dim: Dict[int, pd.DataFrame] = {}
    if original_df is not None and pca_model_path.exists() and scaler_path.exists():
        try:
            with pca_model_path.open("rb") as handle:
                pca_model = pickle.load(handle)
            with scaler_path.open("rb") as handle:
                scaler = pickle.load(handle)
            feature_cols = getattr(pca_model, "feature_names_in_", None)
            if feature_cols is None:
                # Fallback to scaler feature names for newer sklearn or explicit list in provenance
                feature_cols = getattr(scaler, "feature_names_in_", None)
            if feature_cols is not None:
                filtered_df = original_df.copy()
                if dataset_version == "nontransparent" and "transparent_flag" in filtered_df.columns:
                    filtered_df = filtered_df.loc[pd.to_numeric(filtered_df["transparent_flag"], errors="coerce").fillna(0).eq(0)].copy()
                X = filtered_df[list(feature_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                if np.isnan(X).any():
                    col_means = np.nanmean(X, axis=0)
                    inds = np.where(np.isnan(X))
                    X[inds] = np.take(col_means, inds[1])
                X_scaled = scaler.transform(X)
                for dim in usable_dims:
                    Z = np.asarray(full_df[[f"pc{i}" for i in range(1, dim + 1)]], dtype=float)
                    recon = pca_model.mean_.copy()
                    recon = recon + Z @ pca_model.components_[:dim]
                    errors = np.mean((X_scaled - recon) ** 2, axis=1)
                    err_df = filtered_df[[col for col in EMBEDDING_META_COLS if col in filtered_df.columns]].copy()
                    err_df["model_type"] = "pca"
                    err_df["latent_dim"] = dim
                    err_df["reconstruction_error"] = errors
                    trunc_recon_by_dim[dim] = err_df
        except Exception as exc:
            warn(f"Failed to compute truncated PCA reconstructions for {dataset_version}: {exc}")

    for dim in usable_dims:
        trunc_df = full_df[[col for col in full_df.columns if col in EMBEDDING_META_COLS] + [f"pc{i}" for i in range(1, dim + 1)]].copy()
        space = LatentSpace(
            dataset_version=dataset_version,
            model_type="pca",
            latent_dim=dim,
            embedding_path=embedding_path,
            df=trunc_df,
            latent_cols=[f"pc{i}" for i in range(1, dim + 1)],
            reconstruction_errors=trunc_recon_by_dim.get(dim, recon_df if dim == pca_dim_available else None),
            training_summary=training_summary,
            pca_explained_variance=training_summary.get("pca_explained_variance") if training_summary else None,
        )
        spaces.append(space)
    info(f"Loaded PCA spaces for {dataset_version}: dims={usable_dims}", quiet)
    return spaces


def load_ae_spaces(input_dir: Path, dataset_version: str, target_dims: Sequence[int], quiet: bool) -> List[LatentSpace]:
    spaces: List[LatentSpace] = []
    dataset_dir = input_dir / dataset_version
    for latent_dim in target_dims:
        ae_dir = dataset_dir / f"ae_latent{latent_dim}"
        embedding_path = ae_dir / f"ae_embedding_latent{latent_dim}.csv"
        if not embedding_path.exists():
            continue
        df = pd.read_csv(embedding_path)
        latent_cols = latent_cols_from_df(df)
        if not latent_cols:
            warn(f"No AE latent columns found in {embedding_path}")
            continue
        spaces.append(
            LatentSpace(
                dataset_version=dataset_version,
                model_type="ae",
                latent_dim=latent_dim,
                embedding_path=embedding_path,
                df=df,
                latent_cols=latent_cols,
                reconstruction_errors=maybe_load_csv(ae_dir / "reconstruction_errors.csv"),
                training_summary=maybe_load_json(ae_dir / "training_summary.json"),
                pca_explained_variance=None,
            )
        )
    info(f"Loaded AE spaces for {dataset_version}: {[s.latent_dim for s in spaces]}", quiet)
    return spaces


def merge_reconstruction_errors(space: LatentSpace) -> pd.DataFrame:
    df = space.df.copy()
    if space.reconstruction_errors is None or "reconstruction_error" not in space.reconstruction_errors.columns:
        return df
    merge_cols = [col for col in EMBEDDING_META_COLS if col in df.columns and col in space.reconstruction_errors.columns]
    if not merge_cols:
        df["reconstruction_error"] = space.reconstruction_errors["reconstruction_error"].values[: len(df)]
        return df
    merged = df.merge(space.reconstruction_errors[merge_cols + ["reconstruction_error"]], on=merge_cols, how="left")
    return merged


def structure_summary_row(space: LatentSpace, args: argparse.Namespace) -> Dict[str, Any]:
    df = merge_reconstruction_errors(space)
    X = df[space.latent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(X, axis=0)
    collapsed = int(np.sum(variances <= args.collapsed_variance_threshold))
    X_std = zscore_matrix(X)
    dists = pairwise_distances_upper(X_std)
    near_dup_ratio = float(np.mean(dists < args.near_duplicate_threshold)) if len(dists) else 0.0
    unique_points = rounded_unique_point_count(X_std)
    return {
        "space_id": space.space_id,
        "dataset_version": space.dataset_version,
        "model_type": space.model_type,
        "latent_dim": int(space.latent_dim),
        "sample_count": int(len(df)),
        "dimension_variances": json.dumps([float(v) for v in variances.tolist()]),
        "mean_pairwise_distance": float(np.mean(dists)) if len(dists) else 0.0,
        "median_pairwise_distance": float(np.median(dists)) if len(dists) else 0.0,
        "near_duplicate_ratio": near_dup_ratio,
        "collapsed_dimension_count": collapsed,
        "unique_point_ratio": float(unique_points / len(X_std)) if len(X_std) else 0.0,
    }


def group_separation_for_space(
    space: LatentSpace,
    label_col: str,
    metrics_module: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = merge_reconstruction_errors(space)
    working = df.copy()
    if label_col == "mutation_count_bucket":
        if "mutation_count" not in working.columns:
            return pd.DataFrame(), {
                "space_id": space.space_id,
                "dataset_version": space.dataset_version,
                "model_type": space.model_type,
                "latent_dim": space.latent_dim,
                "group_column": label_col,
                "available": False,
            }
        working[label_col] = mutation_count_bucket(working["mutation_count"])
    elif label_col == "unique_metric_vectors_bucket":
        if "unique_metric_vectors" not in working.columns:
            return pd.DataFrame(), {
                "space_id": space.space_id,
                "dataset_version": space.dataset_version,
                "model_type": space.model_type,
                "latent_dim": space.latent_dim,
                "group_column": label_col,
                "available": False,
            }
        working[label_col] = unique_metric_vectors_bucket(working["unique_metric_vectors"])
    elif label_col not in working.columns:
        return pd.DataFrame(), {
            "space_id": space.space_id,
            "dataset_version": space.dataset_version,
            "model_type": space.model_type,
            "latent_dim": space.latent_dim,
            "group_column": label_col,
            "available": False,
        }
    labels = working[label_col]
    valid_mask = labels.notna()
    working = working.loc[valid_mask].copy()
    if working.empty or working[label_col].nunique() < 1:
        return pd.DataFrame(), {
            "space_id": space.space_id,
            "dataset_version": space.dataset_version,
            "model_type": space.model_type,
            "latent_dim": space.latent_dim,
            "group_column": label_col,
            "available": False,
        }

    X = working[space.latent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_std = zscore_matrix(X)
    groups = working[label_col].astype(str)
    global_center = centroid(X_std)
    rows = []
    centroids: Dict[str, np.ndarray] = {}
    within_dispersion: Dict[str, float] = {}
    for group_value, idxs in groups.groupby(groups).groups.items():
        points = X_std[list(idxs)]
        c = centroid(points)
        centroids[group_value] = c
        within = mean_distance_to_center(points, c)
        within_dispersion[group_value] = within
        rows.append(
            {
                "space_id": space.space_id,
                "dataset_version": space.dataset_version,
                "model_type": space.model_type,
                "latent_dim": int(space.latent_dim),
                "group_column": label_col,
                "group_value": group_value,
                "sample_count": int(len(points)),
                "centroid_json": json.dumps([float(v) for v in c.tolist()]),
                "within_dispersion": float(within),
                "distance_to_global_centroid": float(np.linalg.norm(c - global_center)),
            }
        )

    centroid_points = np.stack(list(centroids.values())) if centroids else np.zeros((0, len(space.latent_cols)))
    inter_centroid_distance = average_pairwise_distance(centroid_points) if len(centroid_points) > 1 else 0.0
    overall_within = float(np.mean(list(within_dispersion.values()))) if within_dispersion else 0.0
    separability_ratio = float(inter_centroid_distance / overall_within) if overall_within > 1e-12 else float("inf") if inter_centroid_distance > 0 else 0.0

    silhouette = np.nan
    if working[label_col].nunique() >= 2 and len(working) >= 3:
        counts = working[label_col].value_counts()
        if (counts >= 2).all():
            try:
                silhouette = float(metrics_module.silhouette_score(X_std, groups.to_numpy()))
            except Exception:
                silhouette = np.nan

    summary = {
        "space_id": space.space_id,
        "dataset_version": space.dataset_version,
        "model_type": space.model_type,
        "latent_dim": int(space.latent_dim),
        "group_column": label_col,
        "available": True,
        "group_count": int(len(centroids)),
        "inter_centroid_distance": float(inter_centroid_distance),
        "within_dispersion_mean": float(overall_within),
        "separability_ratio": separability_ratio,
        "silhouette_score": silhouette,
    }
    return pd.DataFrame(rows), summary


def render_embedding_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    path: Path,
    plt: Any,
    categorical: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    if categorical:
        categories = sorted(df[color_col].astype(str).fillna("NA").unique().tolist())
        cmap = plt.matplotlib.colormaps.get_cmap("tab10")
        for idx, category in enumerate(categories):
            sub = df.loc[df[color_col].astype(str) == category]
            color = cmap(idx % cmap.N)
            ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.75, color=color, label=category)
        ax.legend(loc="best", fontsize=8, ncol=2)
    else:
        vals = pd.to_numeric(df[color_col], errors="coerce")
        sc = ax.scatter(df[x_col], df[y_col], c=vals, cmap="viridis", s=18, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=color_col)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_centroid_scatter(
    group_df: pd.DataFrame,
    title: str,
    path: Path,
    plt: Any,
) -> None:
    if group_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    centroids = group_df["centroid_json"].apply(json.loads).tolist()
    x = [c[0] for c in centroids]
    y = [c[1] if len(c) > 1 else 0.0 for c in centroids]
    sizes = np.clip(group_df["sample_count"].to_numpy(dtype=float) * 4, 50, 500)
    ax.scatter(x, y, s=sizes, alpha=0.75)
    for xi, yi, label in zip(x, y, group_df["group_value"].astype(str)):
        ax.text(xi, yi, label, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_bar_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    path: Path,
    plt: Any,
    hue_col: Optional[str] = None,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if hue_col and hue_col in df.columns:
        groups = sorted(df[hue_col].astype(str).unique().tolist())
        x_values = sorted(df[x_col].astype(str).unique().tolist())
        width = 0.8 / max(len(groups), 1)
        x_pos = np.arange(len(x_values))
        for idx, group in enumerate(groups):
            sub = df.loc[df[hue_col].astype(str) == group]
            vals = [float(sub.loc[sub[x_col].astype(str) == xv, y_col].iloc[0]) if not sub.loc[sub[x_col].astype(str) == xv].empty else np.nan for xv in x_values]
            ax.bar(x_pos + idx * width, vals, width=width, label=group)
        ax.set_xticks(x_pos + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(x_values, rotation=20)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.bar(df[x_col].astype(str), df[y_col].astype(float))
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_error_distribution(df: pd.DataFrame, title: str, path: Path, plt: Any) -> None:
    if "reconstruction_error" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pd.to_numeric(df["reconstruction_error"], errors="coerce").dropna(), bins=30, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("reconstruction_error")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_error_boxplot(df: pd.DataFrame, group_col: str, title: str, path: Path, plt: Any) -> None:
    if "reconstruction_error" not in df.columns or group_col not in df.columns:
        return
    groups = []
    labels = []
    for group_value, group in df.groupby(group_col):
        vals = pd.to_numeric(group["reconstruction_error"], errors="coerce").dropna().to_numpy()
        if len(vals):
            groups.append(vals)
            labels.append(str(group_value))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel("reconstruction_error")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def find_representative_and_outlier_fields(
    space: LatentSpace,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = merge_reconstruction_errors(space)
    X = df[space.latent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_std = zscore_matrix(X)
    global_center = centroid(X_std)
    distances = np.linalg.norm(X_std - global_center, axis=1)
    work = df.copy()
    work["space_id"] = space.space_id
    work["distance_to_global_center"] = distances
    keep_cols = [col for col in EMBEDDING_META_COLS if col in work.columns] + space.latent_cols
    if "reconstruction_error" in work.columns:
        keep_cols.append("reconstruction_error")
    keep_cols = ["space_id"] + keep_cols + ["distance_to_global_center"]

    representative_rows = []
    outlier_rows = []
    extremes_rows = []

    global_rep = work.nsmallest(args.representative_count, "distance_to_global_center")[keep_cols].copy()
    global_rep["selection_type"] = "global_center"
    representative_rows.append(global_rep)

    threshold = work["distance_to_global_center"].quantile(args.outlier_quantile)
    outliers = work.loc[work["distance_to_global_center"] >= threshold, keep_cols].copy()
    outliers["selection_type"] = "global_outlier"
    outlier_rows.append(outliers)

    if "protocol_name" in work.columns:
        for protocol, group in work.groupby("protocol_name"):
            sub = group.nsmallest(1, "distance_to_global_center")[keep_cols].copy()
            sub["selection_type"] = "protocol_center"
            sub["group_value"] = protocol
            representative_rows.append(sub)

    for group_col in ["boundary_miss", "transparent_flag"]:
        if group_col in work.columns:
            for group_value, group in work.groupby(group_col):
                if len(group) == 0:
                    continue
                local_points = X_std[group.index]
                local_center = centroid(local_points)
                local_d = np.linalg.norm(local_points - local_center, axis=1)
                local = group.copy()
                local["distance_to_group_center"] = local_d
                rep = local.nsmallest(1, "distance_to_group_center")[keep_cols + ["distance_to_group_center"]].copy()
                rep["selection_type"] = f"{group_col}_center"
                rep["group_value"] = group_value
                representative_rows.append(rep)

    if "reconstruction_error" in work.columns:
        low = work.nsmallest(args.representative_count, "reconstruction_error")[keep_cols].copy()
        low["selection_type"] = "lowest_reconstruction_error"
        high = work.nlargest(args.representative_count, "reconstruction_error")[keep_cols].copy()
        high["selection_type"] = "highest_reconstruction_error"
        extremes_rows.extend([low, high])

    rep_df = pd.concat(representative_rows, ignore_index=True) if representative_rows else pd.DataFrame()
    outlier_df = pd.concat(outlier_rows, ignore_index=True) if outlier_rows else pd.DataFrame()
    extremes_df = pd.concat(extremes_rows, ignore_index=True) if extremes_rows else pd.DataFrame()
    return rep_df, outlier_df, extremes_df


def dominant_factor_guess(signal_map: Dict[str, float]) -> str:
    clean = {k: v for k, v in signal_map.items() if v is not None and not math.isnan(v)}
    if not clean:
        return "unclear"
    ordered = sorted(clean.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] < 0.75:
        return "unclear"
    if len(ordered) == 1:
        return ordered[0][0]
    if ordered[0][1] >= ordered[1][1] * 1.2:
        return ordered[0][0]
    return "mixed"


def has_cross_protocol_mixing(protocol_signal: float, protocol_silhouette: float) -> str:
    if math.isnan(protocol_signal):
        return "unknown"
    if protocol_signal < 0.8 and (math.isnan(protocol_silhouette) or protocol_silhouette < 0.20):
        return "high"
    if protocol_signal < 1.2 and (math.isnan(protocol_silhouette) or protocol_silhouette < 0.35):
        return "medium"
    return "low"


def has_nontrivial_structure_after_filter(row: Dict[str, Any]) -> str:
    if row["dataset_version"] != "nontransparent":
        return "not_applicable"
    if row["mean_pairwise_distance"] > 1.0 and row["collapsed_dimension_count"] == 0:
        return "yes"
    if row["mean_pairwise_distance"] > 0.6:
        return "maybe"
    return "no"


def main() -> int:
    args = parse_args()
    imported = ensure_dependencies()
    plt = imported["matplotlib.pyplot"]
    metrics_module = imported["sklearn.metrics"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dims = parse_choice_list(args.latent_dims)
    provenance = maybe_load_json(args.input_dir / "provenance.json") or {}
    original_csv = None
    if isinstance(provenance, dict):
        input_csv = provenance.get("input_csv")
        if input_csv:
            original_csv = Path(str(input_csv))

    all_spaces: List[LatentSpace] = []
    for dataset_version in selected_dataset_versions(args.dataset_versions):
        if "pca" in selected_model_types(args.model_types):
            all_spaces.extend(load_pca_spaces(args.input_dir, dataset_version, target_dims, original_csv, args.quiet))
        if "ae" in selected_model_types(args.model_types):
            all_spaces.extend(load_ae_spaces(args.input_dir, dataset_version, target_dims, args.quiet))

    if not all_spaces:
        raise ValueError(f"No latent spaces discovered under {args.input_dir}")

    structure_rows: List[Dict[str, Any]] = []
    group_summary_rows: List[Dict[str, Any]] = []
    group_detail_frames: Dict[str, List[pd.DataFrame]] = {
        "protocol_name": [],
        "boundary_miss": [],
        "transparent_flag": [],
        "mutation_count_bucket": [],
        "unique_metric_vectors_bucket": [],
    }
    representative_frames = []
    outlier_frames = []
    reconstruction_extreme_frames = []
    latent_structure_report_rows = []

    for space in all_spaces:
        info(f"Analyzing {space.space_id}", args.quiet)
        structure = structure_summary_row(space, args)
        structure_rows.append(structure)

        merged_df = merge_reconstruction_errors(space)
        latent_dir = output_dir / "plots" / space.dataset_version / f"{space.model_type}_latent{space.latent_dim}"
        latent_dir.mkdir(parents=True, exist_ok=True)

        if len(space.latent_cols) >= 2:
            plot_specs = [
                ("protocol_name", True),
                ("boundary_miss", True),
                ("transparent_flag", True),
                ("reconstruction_error", False),
                ("unique_metric_vectors", False),
            ]
            for color_col, categorical in plot_specs:
                if color_col in merged_df.columns:
                    render_embedding_scatter(
                        merged_df,
                        space.latent_cols[0],
                        space.latent_cols[1],
                        color_col,
                        f"{space.space_id} by {color_col}",
                        latent_dir / f"{space.space_id}_by_{color_col}.png",
                        plt,
                        categorical=categorical,
                    )
            render_error_distribution(
                merged_df,
                f"{space.space_id} reconstruction error",
                latent_dir / f"{space.space_id}_reconstruction_error_hist.png",
                plt,
            )
            render_error_boxplot(
                merged_df,
                "protocol_name",
                f"{space.space_id} reconstruction error by protocol",
                latent_dir / f"{space.space_id}_reconstruction_error_by_protocol_boxplot.png",
                plt,
            )
            render_error_boxplot(
                merged_df,
                "boundary_miss",
                f"{space.space_id} reconstruction error by boundary_miss",
                latent_dir / f"{space.space_id}_reconstruction_error_by_boundary_miss_boxplot.png",
                plt,
            )

        signal_map = {}
        for group_col in list(group_detail_frames.keys()):
            detail_df, summary = group_separation_for_space(space, group_col, metrics_module)
            if not detail_df.empty:
                group_detail_frames[group_col].append(detail_df)
                if len(space.latent_cols) >= 2:
                    render_centroid_scatter(
                        detail_df,
                        f"{space.space_id} {group_col} centroids",
                        latent_dir / f"{space.space_id}_{group_col}_centroids.png",
                        plt,
                    )
                    render_bar_comparison(
                        detail_df[["group_value", "within_dispersion"]].copy(),
                        "group_value",
                        "within_dispersion",
                        f"{space.space_id} {group_col} within dispersion",
                        latent_dir / f"{space.space_id}_{group_col}_within_dispersion.png",
                        plt,
                    )
            group_summary_rows.append(summary)
            signal_map[group_col] = summary.get("separability_ratio") if summary.get("available") else np.nan

        rep_df, outlier_df, extremes_df = find_representative_and_outlier_fields(space, args)
        if not rep_df.empty:
            representative_frames.append(rep_df)
        if not outlier_df.empty:
            outlier_frames.append(outlier_df)
        if not extremes_df.empty:
            reconstruction_extreme_frames.append(extremes_df)

        summary_lookup = {
            row["group_column"]: row
            for row in group_summary_rows
            if row.get("space_id") == space.space_id and row.get("available")
        }
        protocol_signal = summary_lookup.get("protocol_name", {}).get("separability_ratio", np.nan)
        boundary_signal = summary_lookup.get("boundary_miss", {}).get("separability_ratio", np.nan)
        transparent_signal = summary_lookup.get("transparent_flag", {}).get("separability_ratio", np.nan)
        protocol_silhouette = summary_lookup.get("protocol_name", {}).get("silhouette_score", np.nan)
        report_row = {
            "space_id": space.space_id,
            "dataset_version": space.dataset_version,
            "model_type": space.model_type,
            "latent_dim": int(space.latent_dim),
            "sample_count": int(len(space.df)),
            "reconstruction_error_summary": reconstruction_error_stats(merged_df["reconstruction_error"].dropna().to_numpy())
            if "reconstruction_error" in merged_df.columns and merged_df["reconstruction_error"].notna().any()
            else None,
            "protocol_signal_strength": protocol_signal,
            "boundary_miss_signal_strength": boundary_signal,
            "transparent_signal_strength": transparent_signal,
            "dominant_factor_guess": dominant_factor_guess(
                {
                    "protocol": protocol_signal,
                    "boundary_miss": boundary_signal,
                    "transparent_vs_nontransparent": transparent_signal,
                }
            ),
            "has_cross_protocol_mixing": has_cross_protocol_mixing(protocol_signal, protocol_silhouette),
            "has_nontrivial_structure_after_transparent_filter": has_nontrivial_structure_after_filter(structure),
            "notes": [
                f"mean_pairwise_distance={structure['mean_pairwise_distance']:.6f}",
                f"near_duplicate_ratio={structure['near_duplicate_ratio']:.6f}",
                f"collapsed_dimension_count={structure['collapsed_dimension_count']}",
            ],
        }
        latent_structure_report_rows.append(report_row)

    latent_structure_summary_df = pd.DataFrame(structure_rows)
    write_table(latent_structure_summary_df, output_dir / "latent_structure_summary.csv")
    write_json({"rows": structure_rows}, output_dir / "latent_structure_summary.json")

    group_files = {
        "protocol_name": "group_separation_by_protocol.csv",
        "boundary_miss": "group_separation_by_boundary_miss.csv",
        "transparent_flag": "group_separation_by_transparent_flag.csv",
        "mutation_count_bucket": "group_separation_by_mutation_count_bucket.csv",
        "unique_metric_vectors_bucket": "group_separation_by_unique_metric_vectors_bucket.csv",
    }
    for group_col, frames in group_detail_frames.items():
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        write_table(df, output_dir / group_files[group_col])

    group_summary_df = pd.DataFrame(group_summary_rows)
    write_table(group_summary_df, output_dir / "group_separation_summary.csv")

    representative_df = pd.concat(representative_frames, ignore_index=True) if representative_frames else pd.DataFrame()
    outlier_df = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    reconstruction_extremes_df = (
        pd.concat(reconstruction_extreme_frames, ignore_index=True) if reconstruction_extreme_frames else pd.DataFrame()
    )
    write_table(representative_df, output_dir / "representative_fields.csv")
    write_table(outlier_df, output_dir / "outlier_fields.csv")
    write_table(reconstruction_extremes_df, output_dir / "reconstruction_error_extremes.csv")

    # full vs nontransparent
    full_vs_rows = []
    for model_type in ["pca", "ae"]:
        for latent_dim in target_dims:
            full_row = latent_structure_summary_df.loc[
                (latent_structure_summary_df["dataset_version"] == "full")
                & (latent_structure_summary_df["model_type"] == model_type)
                & (latent_structure_summary_df["latent_dim"] == latent_dim)
            ]
            non_row = latent_structure_summary_df.loc[
                (latent_structure_summary_df["dataset_version"] == "nontransparent")
                & (latent_structure_summary_df["model_type"] == model_type)
                & (latent_structure_summary_df["latent_dim"] == latent_dim)
            ]
            if full_row.empty or non_row.empty:
                continue
            full_row = full_row.iloc[0]
            non_row = non_row.iloc[0]
            def lookup_signal(dataset_version: str, group_col: str) -> float:
                matches = group_summary_df.loc[
                    (group_summary_df["dataset_version"] == dataset_version)
                    & (group_summary_df["model_type"] == model_type)
                    & (group_summary_df["latent_dim"] == latent_dim)
                    & (group_summary_df["group_column"] == group_col)
                ]
                if matches.empty:
                    return np.nan
                return float(matches.iloc[0]["separability_ratio"])

            full_vs_rows.append(
                {
                    "model_type": model_type,
                    "latent_dim": latent_dim,
                    "sample_count_full": int(full_row["sample_count"]),
                    "sample_count_nontransparent": int(non_row["sample_count"]),
                    "mean_pairwise_distance_full": float(full_row["mean_pairwise_distance"]),
                    "mean_pairwise_distance_nontransparent": float(non_row["mean_pairwise_distance"]),
                    "near_duplicate_ratio_full": float(full_row["near_duplicate_ratio"]),
                    "near_duplicate_ratio_nontransparent": float(non_row["near_duplicate_ratio"]),
                    "protocol_signal_full": lookup_signal("full", "protocol_name"),
                    "protocol_signal_nontransparent": lookup_signal("nontransparent", "protocol_name"),
                    "boundary_signal_full": lookup_signal("full", "boundary_miss"),
                    "boundary_signal_nontransparent": lookup_signal("nontransparent", "boundary_miss"),
                    "transparent_signal_full": lookup_signal("full", "transparent_flag"),
                    "transparent_signal_nontransparent": lookup_signal("nontransparent", "transparent_flag"),
                }
            )
    full_vs_df = pd.DataFrame(full_vs_rows)
    write_table(full_vs_df, output_dir / "full_vs_nontransparent_comparison.csv")
    write_json({"rows": full_vs_rows}, output_dir / "full_vs_nontransparent_summary.json")

    # pca vs ae
    pca_vs_ae_rows = []
    for dataset_version in selected_dataset_versions(args.dataset_versions):
        for latent_dim in target_dims:
            def lookup(model_type: str, col: str, group_col: Optional[str] = None) -> float:
                if group_col is None:
                    matches = latent_structure_summary_df.loc[
                        (latent_structure_summary_df["dataset_version"] == dataset_version)
                        & (latent_structure_summary_df["model_type"] == model_type)
                        & (latent_structure_summary_df["latent_dim"] == latent_dim)
                    ]
                    if matches.empty:
                        return np.nan
                    return float(matches.iloc[0][col])
                matches = group_summary_df.loc[
                    (group_summary_df["dataset_version"] == dataset_version)
                    & (group_summary_df["model_type"] == model_type)
                    & (group_summary_df["latent_dim"] == latent_dim)
                    & (group_summary_df["group_column"] == group_col)
                ]
                if matches.empty:
                    return np.nan
                return float(matches.iloc[0][col])
            pca_val = lookup("pca", "mean_pairwise_distance")
            ae_val = lookup("ae", "mean_pairwise_distance")
            if math.isnan(pca_val) or math.isnan(ae_val):
                continue
            pca_vs_ae_rows.append(
                {
                    "dataset_version": dataset_version,
                    "latent_dim": latent_dim,
                    "pca_mean_pairwise_distance": pca_val,
                    "ae_mean_pairwise_distance": ae_val,
                    "pca_near_duplicate_ratio": lookup("pca", "near_duplicate_ratio"),
                    "ae_near_duplicate_ratio": lookup("ae", "near_duplicate_ratio"),
                    "pca_protocol_signal": lookup("pca", "separability_ratio", "protocol_name"),
                    "ae_protocol_signal": lookup("ae", "separability_ratio", "protocol_name"),
                    "pca_boundary_signal": lookup("pca", "separability_ratio", "boundary_miss"),
                    "ae_boundary_signal": lookup("ae", "separability_ratio", "boundary_miss"),
                    "pca_transparent_signal": lookup("pca", "separability_ratio", "transparent_flag"),
                    "ae_transparent_signal": lookup("ae", "separability_ratio", "transparent_flag"),
                }
            )
    pca_vs_ae_df = pd.DataFrame(pca_vs_ae_rows)
    write_table(pca_vs_ae_df, output_dir / "pca_vs_ae_comparison.csv")

    # Aggregated plots
    if not group_summary_df.empty:
        for group_col in ["protocol_name", "boundary_miss", "transparent_flag"]:
            subset = group_summary_df.loc[group_summary_df["group_column"] == group_col].copy()
            if subset.empty:
                continue
            subset["space_label"] = subset["dataset_version"] + "_" + subset["model_type"] + "_d" + subset["latent_dim"].astype(str)
            render_bar_comparison(
                subset[["space_label", "separability_ratio", "dataset_version"]].copy(),
                "space_label",
                "separability_ratio",
                f"{group_col} separability across spaces",
                output_dir / f"separability_{group_col}_comparison.png",
                plt,
                hue_col="dataset_version",
            )
    if not full_vs_df.empty:
        render_bar_comparison(
            full_vs_df.assign(space=lambda d: d["model_type"] + "_d" + d["latent_dim"].astype(str))[["space", "protocol_signal_full", "protocol_signal_nontransparent"]],
            "space",
            "protocol_signal_full",
            "protocol signal full",
            output_dir / "full_vs_nontransparent_protocol_signal.png",
            plt,
        )
    if not pca_vs_ae_df.empty:
        render_bar_comparison(
            pca_vs_ae_df.assign(space=lambda d: d["dataset_version"] + "_d" + d["latent_dim"].astype(str))[["space", "pca_protocol_signal", "ae_protocol_signal"]],
            "space",
            "pca_protocol_signal",
            "pca protocol signal by dataset/dim",
            output_dir / "pca_vs_ae_protocol_signal.png",
            plt,
        )

    report = {
        "rows": latent_structure_report_rows,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "basic_visual_analysis_summary": maybe_load_json(args.basic_visual_summary),
        "subset_summary": maybe_load_csv(args.subset_summary).to_dict(orient="records")
        if maybe_load_csv(args.subset_summary) is not None
        else None,
    }
    write_json(report, output_dir / "latent_structure_report.json")

    info(f"Wrote latent-space structure analysis to {output_dir}", args.quiet)
    return 0


def reconstruction_error_stats(errors: np.ndarray) -> Dict[str, float]:
    clean = np.asarray(errors, dtype=float)
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "median": float(np.median(clean)),
        "max": float(np.max(clean)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
