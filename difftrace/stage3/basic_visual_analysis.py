#!/usr/bin/env python3
"""Basic visualization and statistical analysis for field-level features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dataset_health_check import Canvas, value_to_y, write_json, write_lines


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/out/field_training_samples.csv")
DEFAULT_FEATURE_COLS = Path("/root/semvec/difftrace/out/dataset_health_report/model_feature_cols.txt")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/basic_visual_analysis")

CORE_JACCARD_METRICS = [
    "branch_sites_jaccard",
    "bb_set_jaccard",
    "cmp_site_set_jaccard",
    "lcp_ratio",
]
CORE_DELTA_METRICS = [
    "instr_delta_ratio",
    "bb_multiset_l1_ratio",
    "cmp_delta_ratio",
    "branch_flip_ratio",
]


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def info(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def load_feature_cols(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model feature column list: {path}")
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            cols = obj.get("model_feature_cols")
            if isinstance(cols, list):
                return [str(col) for col in cols]
        if isinstance(obj, list):
            return [str(col) for col in obj]
        raise ValueError(f"Unsupported JSON structure in {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def detect_metadata_and_diagnosis_cols(df: pd.DataFrame, model_feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    diagnosis_candidates = [
        "mutation_count",
        "valid_mutations",
        "boundary_miss",
        "deltaf_dispersion",
        "unique_metric_vectors",
        "metrics_source",
        "transparent_flag",
        "low_coverage_flag",
    ]
    diagnosis_cols = [col for col in diagnosis_candidates if col in df.columns]
    metadata_cols = [col for col in df.columns if col not in set(model_feature_cols) | set(diagnosis_cols)]
    return metadata_cols, diagnosis_cols


def core_metric_stability_rule(row: pd.Series, eps: float = 1e-6) -> bool:
    for metric in CORE_JACCARD_METRICS:
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"
        std_col = f"{metric}_std"
        if min_col not in row.index or max_col not in row.index:
            return False
        min_val = row.get(min_col)
        max_val = row.get(max_col)
        std_val = row.get(std_col, 0.0)
        if pd.isna(min_val) or pd.isna(max_val):
            return False
        if float(min_val) < 0.99 or float(max_val) < 0.99 or (pd.notna(std_val) and float(std_val) > eps):
            return False
    for metric in CORE_DELTA_METRICS:
        max_col = f"{metric}_max"
        std_col = f"{metric}_std"
        if max_col not in row.index:
            return False
        max_val = row.get(max_col)
        std_val = row.get(std_col, 0.0)
        if pd.isna(max_val):
            return False
        if float(max_val) > 0.01 or (pd.notna(std_val) and float(std_val) > 0.01):
            return False
    return True


def compute_transparent_flag(df: pd.DataFrame) -> pd.Series:
    boundary_rule = (
        pd.to_numeric(df.get("boundary_miss"), errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(df.get("unique_metric_vectors"), errors="coerce").fillna(-1).eq(1)
    )
    stability_rule = df.apply(core_metric_stability_rule, axis=1)
    return (boundary_rule | stability_rule).astype(int)


def compute_low_coverage_flag(df: pd.DataFrame, threshold: int = 3) -> pd.Series:
    return pd.to_numeric(df.get("mutation_count"), errors="coerce").fillna(-1).lt(threshold).astype(int)


def build_subsets(
    df: pd.DataFrame,
    enable_transparent_filter: bool,
    enable_low_coverage_filter: bool,
    low_coverage_threshold: int,
) -> Dict[str, pd.DataFrame]:
    subsets = {"full": df.copy()}
    if enable_transparent_filter:
        subsets["nontransparent"] = df.loc[df["transparent_flag"] == 0].copy()
    if enable_low_coverage_filter:
        subsets["coverage_ge_threshold"] = df.loc[df["low_coverage_flag"] == 0].copy()
        if enable_transparent_filter:
            subsets["nontransparent_coverage_ge_threshold"] = df.loc[
                (df["transparent_flag"] == 0) & (df["low_coverage_flag"] == 0)
            ].copy()
    return subsets


def feature_basic_stats(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "feature": col,
                "count": int(series.count()),
                "mean": float(series.mean()) if not series.empty else np.nan,
                "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
                "min": float(series.min()) if not series.empty else np.nan,
                "max": float(series.max()) if not series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    q = clean.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "max": float(clean.max()),
    }


def subset_summary_table(subsets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    full_count = len(subsets.get("full", pd.DataFrame()))
    for name, subset in subsets.items():
        rows.append(
            {
                "subset": name,
                "sample_count": int(len(subset)),
                "sample_ratio_vs_full": float(len(subset) / full_count) if full_count else np.nan,
                "protocol_count": int(subset["protocol_name"].nunique()) if "protocol_name" in subset.columns else np.nan,
                "sample_id_count": int(subset[["protocol_name", "sample_id"]].drop_duplicates().shape[0])
                if {"protocol_name", "sample_id"}.issubset(subset.columns)
                else np.nan,
                "transparent_count": int((subset["transparent_flag"] == 1).sum()) if "transparent_flag" in subset.columns else np.nan,
                "low_coverage_count": int((subset["low_coverage_flag"] == 1).sum()) if "low_coverage_flag" in subset.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def standardize_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    X = df[list(feature_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds
    return X_std, {"mean": means.tolist(), "std": stds.tolist()}


def compute_pca(X: np.ndarray, n_components: int = 5) -> Dict[str, Any]:
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for PCA")
    X_centered = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = (s ** 2) / max(X.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    components = vt[:n_components]
    scores = X_centered @ components.T
    return {
        "components": components,
        "scores": scores[:, :n_components],
        "explained_variance": explained_variance[:n_components],
        "explained_variance_ratio": explained_variance_ratio[:n_components],
        "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio[:n_components]),
        "all_explained_variance_ratio": explained_variance_ratio,
    }


def pca_loadings_df(feature_cols: Sequence[str], components: np.ndarray, subset_name: str) -> pd.DataFrame:
    rows = []
    for feature_idx, feature in enumerate(feature_cols):
        row = {"subset": subset_name, "feature": feature}
        for pc_idx in range(components.shape[0]):
            row[f"PC{pc_idx + 1}"] = float(components[pc_idx, feature_idx])
            row[f"abs_PC{pc_idx + 1}"] = abs(float(components[pc_idx, feature_idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def top_loading_features(loadings_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for subset, group in loadings_df.groupby("subset"):
        for pc_col in [col for col in group.columns if col.startswith("abs_PC")]:
            pc_name = pc_col.replace("abs_", "")
            top = group.sort_values(pc_col, ascending=False).head(top_n)
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                rows.append(
                    {
                        "subset": subset,
                        "principal_component": pc_name,
                        "rank": rank,
                        "feature": row["feature"],
                        "loading": row[pc_name],
                        "abs_loading": row[pc_col],
                    }
                )
    return pd.DataFrame(rows)


def centroid_stats(proj_df: pd.DataFrame, group_col: str, subset_name: str) -> pd.DataFrame:
    if group_col not in proj_df.columns:
        return pd.DataFrame()
    rows = []
    for group_value, group in proj_df.groupby(group_col, dropna=False):
        rows.append(
            {
                "subset": subset_name,
                "group_column": group_col,
                "group_value": group_value,
                "sample_count": int(len(group)),
                "pc1_mean": float(group["PC1"].mean()),
                "pc2_mean": float(group["PC2"].mean()),
                "pc1_std": float(group["PC1"].std(ddof=0)) if len(group) > 1 else 0.0,
                "pc2_std": float(group["PC2"].std(ddof=0)) if len(group) > 1 else 0.0,
                "pc12_radius_mean": float(np.sqrt(group["PC1"] ** 2 + group["PC2"] ** 2).mean()),
            }
        )
    return pd.DataFrame(rows)


def average_pairwise_distance(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    dists = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.append(float(np.linalg.norm(points[i] - points[j])))
    return float(np.mean(dists)) if dists else 0.0


def group_structure_signal(centroid_df: pd.DataFrame) -> Dict[str, Any]:
    if centroid_df.empty:
        return {"signal_ratio": 0.0, "obvious_structure": False}
    centroids = centroid_df[["pc1_mean", "pc2_mean"]].to_numpy(dtype=float)
    spreads = np.sqrt(centroid_df["pc1_std"].to_numpy(dtype=float) ** 2 + centroid_df["pc2_std"].to_numpy(dtype=float) ** 2)
    centroid_sep = average_pairwise_distance(centroids)
    within_spread = float(np.mean(spreads)) if len(spreads) else 0.0
    ratio = float(centroid_sep / within_spread) if within_spread > 1e-12 else float("inf") if centroid_sep > 0 else 0.0
    return {
        "signal_ratio": ratio,
        "obvious_structure": bool(ratio >= 1.0),
    }


def kmeans_numpy(X: np.ndarray, n_clusters: int, random_state: int = 1337, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(X) < n_clusters:
        raise ValueError("Sample count is smaller than n_clusters")
    centers = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(n_clusters):
            members = X[labels == idx]
            if len(members) == 0:
                centers[idx] = X[rng.integers(0, len(X))]
            else:
                centers[idx] = members.mean(axis=0)
    return labels


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def palette_for_categories(categories: Sequence[Any]) -> Dict[Any, Tuple[int, int, int]]:
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    return {cat: palette[idx % len(palette)] for idx, cat in enumerate(categories)}


def draw_point(canvas: Canvas, x: int, y: int, color: Tuple[int, int, int], radius: int = 3) -> None:
    canvas.fill_rect(x - radius, y - radius, x + radius + 1, y + radius + 1, color)


def color_from_continuous(value: float, min_val: float, max_val: float) -> Tuple[int, int, int]:
    if not np.isfinite(value):
        return (180, 180, 180)
    if max_val <= min_val:
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)
    t = min(max(t, 0.0), 1.0)
    r = int(255 * t)
    g = int(80 + 120 * (1 - abs(t - 0.5) * 2))
    b = int(255 * (1 - t))
    return (r, g, b)


def render_line_chart(values: Sequence[float], title: str, y_label: str, path: Path) -> None:
    width, height = 1100, 700
    canvas = Canvas(width, height)
    left, right, top, bottom = 90, width - 40, 80, height - 110
    canvas.draw_text(20, 20, title, scale=2)
    canvas.line(left, top, left, bottom, (0, 0, 0), 2)
    canvas.line(left, bottom, right, bottom, (0, 0, 0), 2)
    canvas.draw_text(20, 130, y_label, vertical=True)
    max_val = max(values) if values else 1.0
    max_val = max(max_val, 1.0)
    min_val = 0.0
    xs = np.linspace(left, right, max(len(values), 2))
    prev = None
    for idx, value in enumerate(values):
        x = int(xs[idx])
        y = value_to_y(float(value), min_val, max_val, top, bottom)
        draw_point(canvas, x, y, (50, 100, 200), 3)
        if prev is not None:
            canvas.line(prev[0], prev[1], x, y, (50, 100, 200), 2)
        canvas.draw_text(x - 6, bottom + 10, str(idx + 1), scale=1)
        prev = (x, y)
    for tick in range(6):
        val = max_val * tick / 5.0
        y = value_to_y(val, min_val, max_val, top, bottom)
        canvas.line(left, y, right, y, (235, 235, 235), 1)
        canvas.draw_text(10, y - 4, f"{val:.2f}"[:8], scale=1)
    canvas.save_png(path)


def render_bar_chart(values: Sequence[float], labels: Sequence[str], title: str, y_label: str, path: Path) -> None:
    width, height = 1100, 700
    canvas = Canvas(width, height)
    left, right, top, bottom = 90, width - 40, 80, height - 110
    canvas.draw_text(20, 20, title, scale=2)
    canvas.line(left, top, left, bottom, (0, 0, 0), 2)
    canvas.line(left, bottom, right, bottom, (0, 0, 0), 2)
    canvas.draw_text(20, 130, y_label, vertical=True)
    max_val = max(values) if values else 1.0
    max_val = max(max_val, 1.0)
    bar_space = (right - left) / max(len(values), 1)
    bar_width = max(8, int(bar_space * 0.7))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = int(left + idx * bar_space + (bar_space - bar_width) / 2)
        x1 = x0 + bar_width
        y = value_to_y(float(value), 0.0, max_val, top, bottom)
        canvas.fill_rect(x0, y, x1, bottom, (65, 105, 225))
        canvas.draw_text(x0, bottom + 10, label[:10], scale=1)
    for tick in range(6):
        val = max_val * tick / 5.0
        y = value_to_y(val, 0.0, max_val, top, bottom)
        canvas.line(left, y, right, y, (235, 235, 235), 1)
        canvas.draw_text(10, y - 4, f"{val:.2f}"[:8], scale=1)
    canvas.save_png(path)


def render_scatter(
    proj_df: pd.DataFrame,
    color_col: str,
    title: str,
    path: Path,
    categorical: bool,
) -> None:
    width, height = 1200, 850
    canvas = Canvas(width, height)
    left, right, top, bottom = 100, width - 240, 80, height - 100
    canvas.draw_text(20, 20, title, scale=2)
    canvas.line(left, top, left, bottom, (0, 0, 0), 2)
    canvas.line(left, bottom, right, bottom, (0, 0, 0), 2)
    canvas.draw_text(right - 40, bottom + 25, "PC1", scale=2)
    canvas.draw_text(20, top + 40, "PC2", vertical=True, scale=2)

    x_vals = proj_df["PC1"].to_numpy(dtype=float)
    y_vals = proj_df["PC2"].to_numpy(dtype=float)
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def x_to_px(val: float) -> int:
        if x_max <= x_min:
            return (left + right) // 2
        return int(left + (val - x_min) / (x_max - x_min) * (right - left))

    def y_to_px(val: float) -> int:
        return value_to_y(val, y_min, y_max, top, bottom)

    if categorical:
        categories = sorted(proj_df[color_col].astype(str).fillna("NA").unique().tolist())
        palette = palette_for_categories(categories)
        for _, row in proj_df.iterrows():
            draw_point(canvas, x_to_px(float(row["PC1"])), y_to_px(float(row["PC2"])), palette[str(row[color_col])], 3)
        legend_y = 120
        for cat in categories[:20]:
            canvas.fill_rect(width - 200, legend_y, width - 184, legend_y + 16, palette[cat])
            canvas.draw_text(width - 176, legend_y + 3, str(cat)[:18], scale=1)
            legend_y += 22
    else:
        color_vals = pd.to_numeric(proj_df[color_col], errors="coerce").to_numpy(dtype=float)
        c_min = float(np.nanmin(color_vals)) if np.isfinite(np.nanmin(color_vals)) else 0.0
        c_max = float(np.nanmax(color_vals)) if np.isfinite(np.nanmax(color_vals)) else 1.0
        for _, row in proj_df.iterrows():
            val = float(row[color_col]) if pd.notna(row[color_col]) else np.nan
            color = color_from_continuous(val, c_min, c_max)
            draw_point(canvas, x_to_px(float(row["PC1"])), y_to_px(float(row["PC2"])), color, 3)
        gradient_top = 140
        for i in range(180):
            t = i / 179.0
            color = color_from_continuous(c_max - t * (c_max - c_min), c_min, c_max)
            canvas.fill_rect(width - 180, gradient_top + i, width - 150, gradient_top + i + 1, color)
        canvas.draw_text(width - 140, gradient_top - 4, f"{c_max:.2f}"[:8], scale=1)
        canvas.draw_text(width - 140, gradient_top + 170, f"{c_min:.2f}"[:8], scale=1)
        canvas.draw_text(width - 190, gradient_top - 30, color_col[:20], scale=1)

    for tick in range(5):
        xv = x_min + (x_max - x_min) * tick / 4.0
        x = x_to_px(xv)
        canvas.line(x, top, x, bottom, (240, 240, 240), 1)
        canvas.draw_text(x - 12, bottom + 10, f"{xv:.1f}"[:8], scale=1)
        yv = y_min + (y_max - y_min) * tick / 4.0
        y = y_to_px(yv)
        canvas.line(left, y, right, y, (240, 240, 240), 1)
        canvas.draw_text(10, y - 4, f"{yv:.1f}"[:8], scale=1)
    canvas.save_png(path)


def render_loading_heatmap(loadings: pd.DataFrame, subset_name: str, path: Path) -> None:
    subset = loadings.loc[loadings["subset"] == subset_name, ["feature", "PC1", "PC2", "PC3"]].copy()
    if subset.empty:
        return
    features = subset["feature"].tolist()
    matrix = subset[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    width = 420
    height = 120 + len(features) * 18
    canvas = Canvas(width, max(300, height))
    canvas.draw_text(20, 20, f"{subset_name.upper()} PCA LOADINGS", scale=2)
    left, top = 180, 80

    def color(v: float) -> Tuple[int, int, int]:
        v = max(-1.0, min(1.0, float(v)))
        if v >= 0:
            base = int(255 - 140 * v)
            return (255, base, base)
        base = int(255 - 140 * abs(v))
        return (base, base, 255)

    for i, feature in enumerate(features):
        canvas.draw_text(10, top + i * 18 + 4, feature[:24], scale=1)
        for j, pc in enumerate(["PC1", "PC2", "PC3"]):
            x0 = left + j * 70
            y0 = top + i * 18
            canvas.fill_rect(x0, y0, x0 + 60, y0 + 16, color(matrix[i, j]))
            canvas.draw_text(x0 + 8, 55, pc, scale=1)
    canvas.save_png(path)


def cluster_distribution_table(proj_df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if "cluster" not in proj_df.columns:
        return pd.DataFrame()
    rows = []
    for cluster, group in proj_df.groupby("cluster"):
        protocol_counts = group["protocol_name"].value_counts().to_dict() if "protocol_name" in group.columns else {}
        rows.append(
            {
                "subset": subset_name,
                "cluster": int(cluster),
                "sample_count": int(len(group)),
                "protocol_distribution_json": json.dumps(protocol_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def analyze_subset(
    subset_name: str,
    subset_df: pd.DataFrame,
    feature_cols: Sequence[str],
    output_dir: Path,
    kmeans_k: int,
    quiet: bool,
) -> Dict[str, Any]:
    if len(subset_df) < 3:
        warn(f"Subset {subset_name} has fewer than 3 samples; skipping PCA outputs.")
        return {"subset": subset_name, "sample_count": int(len(subset_df)), "skipped": True}

    info(f"Analyzing subset {subset_name} ({len(subset_df)} samples)", quiet)
    X_std, scaling = standardize_matrix(subset_df, feature_cols)
    pca = compute_pca(X_std, n_components=min(5, X_std.shape[1], X_std.shape[0]))
    scores = pca["scores"]
    proj_df = subset_df.copy()
    for i in range(scores.shape[1]):
        proj_df[f"PC{i + 1}"] = scores[:, i]

    if kmeans_k and len(subset_df) >= kmeans_k:
        proj_df["cluster"] = kmeans_numpy(X_std, kmeans_k)

    explained_df = pd.DataFrame(
        {
            "subset": subset_name,
            "principal_component": [f"PC{i + 1}" for i in range(len(pca["explained_variance_ratio"]))],
            "explained_variance_ratio": pca["explained_variance_ratio"],
            "explained_variance": pca["explained_variance"],
        }
    )
    cumulative_df = pd.DataFrame(
        {
            "subset": subset_name,
            "principal_component": [f"PC{i + 1}" for i in range(len(pca["cumulative_explained_variance_ratio"]))],
            "cumulative_explained_variance_ratio": pca["cumulative_explained_variance_ratio"],
        }
    )
    loadings_df = pca_loadings_df(feature_cols, pca["components"][:3], subset_name)

    centroid_protocol = centroid_stats(proj_df, "protocol_name", subset_name)
    centroid_boundary = centroid_stats(proj_df, "boundary_miss", subset_name)
    centroid_transparent = centroid_stats(proj_df, "transparent_flag", subset_name)
    centroid_cluster = centroid_stats(proj_df, "cluster", subset_name) if "cluster" in proj_df.columns else pd.DataFrame()

    render_bar_chart(
        explained_df["explained_variance_ratio"].tolist(),
        explained_df["principal_component"].tolist(),
        f"{subset_name.upper()} EXPLAINED VARIANCE",
        "RATIO",
        output_dir / f"explained_variance_bar_{subset_name}.png",
    )
    render_line_chart(
        cumulative_df["cumulative_explained_variance_ratio"].tolist(),
        f"{subset_name.upper()} CUMULATIVE EXPLAINED VARIANCE",
        "CUM RATIO",
        output_dir / f"cumulative_explained_variance_{subset_name}.png",
    )

    plot_specs = [
        ("protocol_name", "protocol", True),
        ("boundary_miss", "boundary_miss", True),
        ("mutation_count", "mutation_count", False),
        ("unique_metric_vectors", "unique_metric_vectors", False),
        ("transparent_flag", "transparent_flag", True),
    ]
    for column, file_label, categorical in plot_specs:
        if column in proj_df.columns:
            render_scatter(
                proj_df,
                color_col=column,
                title=f"{subset_name.upper()} PCA BY {column.upper()}",
                path=output_dir / f"pca_2d_{subset_name}_by_{file_label}.png",
                categorical=categorical,
            )
    if "cluster" in proj_df.columns:
        render_scatter(
            proj_df,
            color_col="cluster",
            title=f"{subset_name.upper()} PCA BY CLUSTER",
            path=output_dir / f"pca_2d_{subset_name}_by_cluster.png",
            categorical=True,
        )

    render_loading_heatmap(loadings_df, subset_name, output_dir / f"pca_loading_heatmap_{subset_name}.png")
    proj_df.to_csv(output_dir / f"pca_projection_{subset_name}.csv", index=False)

    protocol_signal = group_structure_signal(centroid_protocol)
    boundary_signal = group_structure_signal(centroid_boundary)
    transparent_signal = group_structure_signal(centroid_transparent)

    cluster_distribution = cluster_distribution_table(proj_df, subset_name)

    return {
        "subset": subset_name,
        "sample_count": int(len(subset_df)),
        "skipped": False,
        "scaling": scaling,
        "explained_df": explained_df,
        "cumulative_df": cumulative_df,
        "loadings_df": loadings_df,
        "projection_df": proj_df,
        "centroid_protocol": centroid_protocol,
        "centroid_boundary": centroid_boundary,
        "centroid_transparent": centroid_transparent,
        "centroid_cluster": centroid_cluster,
        "protocol_signal": protocol_signal,
        "boundary_signal": boundary_signal,
        "transparent_signal": transparent_signal,
        "cluster_distribution": cluster_distribution,
        "top_loading_features": top_loading_features(loadings_df, top_n=10),
        "explained_variance_top3": [float(v) for v in pca["explained_variance_ratio"][:3]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Basic visualization and statistical analysis for field-level features.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to field_training_samples.csv")
    parser.add_argument(
        "--model-feature-cols",
        type=Path,
        default=DEFAULT_FEATURE_COLS,
        help="Path to model_feature_cols.txt or model_feature_cols.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--disable-transparent-filter",
        action="store_true",
        help="Do not build the nontransparent subset",
    )
    parser.add_argument(
        "--enable-low-coverage-filter",
        action="store_true",
        help="Build additional subsets that remove low-coverage fields",
    )
    parser.add_argument("--low-coverage-threshold", type=int, default=3, help="Threshold for low_coverage_flag")
    parser.add_argument("--kmeans-k", type=int, default=4, help="KMeans cluster count; <=1 disables clustering")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")
    feature_cols = load_feature_cols(args.model_feature_cols)
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        warn(f"Missing requested model feature columns: {missing_features}")
    feature_cols = [col for col in feature_cols if col in df.columns]
    if len(feature_cols) < 2:
        raise ValueError("Need at least two model feature columns for PCA")

    numeric_candidates = set(feature_cols) | {
        "mutation_count",
        "valid_mutations",
        "boundary_miss",
        "deltaf_dispersion",
        "unique_metric_vectors",
    }
    df = to_numeric_frame(df, [col for col in numeric_candidates if col in df.columns])

    df["transparent_flag"] = compute_transparent_flag(df)
    df["low_coverage_flag"] = compute_low_coverage_flag(df, args.low_coverage_threshold)

    metadata_cols, diagnosis_cols = detect_metadata_and_diagnosis_cols(df, feature_cols)
    subsets = build_subsets(
        df,
        enable_transparent_filter=not args.disable_transparent_filter,
        enable_low_coverage_filter=args.enable_low_coverage_filter,
        low_coverage_threshold=args.low_coverage_threshold,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_summary = subset_summary_table(subsets)
    write_table(subset_summary, output_dir / "subset_summary.csv")

    full_stats = feature_basic_stats(subsets["full"], feature_cols)
    write_table(full_stats, output_dir / "feature_basic_stats_full.csv")
    if "nontransparent" in subsets:
        nontransparent_stats = feature_basic_stats(subsets["nontransparent"], feature_cols)
        write_table(nontransparent_stats, output_dir / "feature_basic_stats_nontransparent.csv")

    results = {}
    combined_explained = []
    combined_cumulative = []
    combined_loadings = []
    combined_top_features = []
    centroid_protocol_frames = []
    centroid_boundary_frames = []
    centroid_transparent_frames = []
    cluster_distribution_frames = []

    for subset_name, subset_df in subsets.items():
        result = analyze_subset(
            subset_name,
            subset_df,
            feature_cols,
            output_dir,
            kmeans_k=args.kmeans_k if args.kmeans_k > 1 else 0,
            quiet=args.quiet,
        )
        results[subset_name] = result
        if result.get("skipped"):
            continue
        combined_explained.append(result["explained_df"])
        combined_cumulative.append(result["cumulative_df"])
        combined_loadings.append(result["loadings_df"])
        combined_top_features.append(result["top_loading_features"])
        centroid_protocol_frames.append(result["centroid_protocol"])
        centroid_boundary_frames.append(result["centroid_boundary"])
        centroid_transparent_frames.append(result["centroid_transparent"])
        if not result["cluster_distribution"].empty:
            cluster_distribution_frames.append(result["cluster_distribution"])

    explained_all = pd.concat(combined_explained, ignore_index=True) if combined_explained else pd.DataFrame()
    cumulative_all = pd.concat(combined_cumulative, ignore_index=True) if combined_cumulative else pd.DataFrame()
    loadings_all = pd.concat(combined_loadings, ignore_index=True) if combined_loadings else pd.DataFrame()
    top_features_all = pd.concat(combined_top_features, ignore_index=True) if combined_top_features else pd.DataFrame()
    centroid_protocol_all = pd.concat(centroid_protocol_frames, ignore_index=True) if centroid_protocol_frames else pd.DataFrame()
    centroid_boundary_all = pd.concat(centroid_boundary_frames, ignore_index=True) if centroid_boundary_frames else pd.DataFrame()
    centroid_transparent_all = pd.concat(centroid_transparent_frames, ignore_index=True) if centroid_transparent_frames else pd.DataFrame()

    write_table(explained_all, output_dir / "explained_variance.csv")
    write_table(cumulative_all, output_dir / "cumulative_explained_variance.csv")
    write_table(loadings_all, output_dir / "pca_loadings.csv")
    write_table(top_features_all, output_dir / "pca_top_features_pc1_pc2.csv")
    write_table(centroid_protocol_all, output_dir / "pca_group_centroids_by_protocol.csv")
    write_table(centroid_boundary_all, output_dir / "pca_group_centroids_by_boundary_miss.csv")
    write_table(centroid_transparent_all, output_dir / "pca_group_centroids_by_transparent_flag.csv")
    if cluster_distribution_frames:
        write_table(pd.concat(cluster_distribution_frames, ignore_index=True), output_dir / "cluster_distribution.csv")

    # Produce default filenames requested by the task, using the full subset.
    if "full" in results and not results["full"].get("skipped"):
        full_explained = results["full"]["explained_df"]
        full_cumulative = results["full"]["cumulative_df"]
        render_bar_chart(
            full_explained["explained_variance_ratio"].tolist(),
            full_explained["principal_component"].tolist(),
            "FULL EXPLAINED VARIANCE",
            "RATIO",
            output_dir / "explained_variance_bar.png",
        )
        render_line_chart(
            full_cumulative["cumulative_explained_variance_ratio"].tolist(),
            "FULL CUMULATIVE EXPLAINED VARIANCE",
            "CUM RATIO",
            output_dir / "cumulative_explained_variance.png",
        )
        full_loadings = results["full"]["loadings_df"]
        render_loading_heatmap(full_loadings, "full", output_dir / "pca_loading_heatmap.png")

    column_groups = {
        "metadata_cols": metadata_cols,
        "diagnosis_cols": diagnosis_cols,
        "model_feature_cols": feature_cols,
    }
    write_json(column_groups, output_dir / "column_groups.json")
    write_lines(feature_cols, output_dir / "model_feature_cols.txt")

    full_result = results.get("full", {})
    nontransparent_result = results.get("nontransparent", {})
    protocol_structure = full_result.get("protocol_signal", {"signal_ratio": 0.0, "obvious_structure": False})
    transparent_structure = full_result.get("transparent_signal", {"signal_ratio": 0.0, "obvious_structure": False})
    boundary_structure = full_result.get("boundary_signal", {"signal_ratio": 0.0, "obvious_structure": False})

    summary = {
        "input_paths": {
            "field_training_samples_csv": str(args.input_csv),
            "model_feature_cols_path": str(args.model_feature_cols),
        },
        "input_sample_count": int(len(df)),
        "subset_counts": {name: int(len(subset)) for name, subset in subsets.items()},
        "column_groups": column_groups,
        "transparent_rule": {
            "rule_1": "boundary_miss == 1 and unique_metric_vectors == 1",
            "rule_2": "core jaccard/lcp metrics stay ~1 and delta metrics stay ~0 with near-zero std",
        },
        "low_coverage_rule": {
            "enabled": bool(args.enable_low_coverage_filter),
            "threshold": args.low_coverage_threshold,
        },
        "pca_overview": {
            "full_top_explained_variance_ratio": full_result.get("explained_variance_top3", []),
            "nontransparent_top_explained_variance_ratio": nontransparent_result.get("explained_variance_top3", []),
        },
        "structure_signals": {
            "protocol_signal_full": protocol_structure,
            "boundary_miss_signal_full": boundary_structure,
            "transparent_signal_full": transparent_structure,
        },
        "observations": [
            f"Full subset size: {len(subsets.get('full', []))}",
            f"Nontransparent subset size: {len(subsets.get('nontransparent', [])) if 'nontransparent' in subsets else 'disabled'}",
            f"Protocol-layering signal in PCA space: ratio={protocol_structure.get('signal_ratio', 0.0):.3f}",
            f"Transparent/nontransparent separation signal in PCA space: ratio={transparent_structure.get('signal_ratio', 0.0):.3f}",
            "Use pca_2d_full_by_protocol.png and pca_2d_full_by_transparent_flag.png together to judge whether protocol or transparent fields dominate the geometry.",
        ],
    }
    write_json(summary, output_dir / "basic_visual_analysis_summary.json")
    info(f"Wrote outputs to {output_dir}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
