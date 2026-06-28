#!/usr/bin/env python3
"""First-pass probe analysis and human-interpretation helpers for latent spaces."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("/root/semvec/difftrace/out/unsupervised_v1")
DEFAULT_FIELD_SAMPLES_CSV = Path("/root/semvec/difftrace/out/field_training_samples/field_training_samples.csv")
DEFAULT_MODEL_FEATURE_COLS = Path("/root/semvec/difftrace/out/dataset_health_report/model_feature_cols.txt")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/probe_analysis_v1")
DEFAULT_PCA_LOADINGS = Path("/root/semvec/difftrace/out/basic_visual_analysis/pca_loadings.csv")
DEFAULT_REPRESENTATIVE_FIELDS = Path("/root/semvec/difftrace/out/latent_space_analysis/representative_fields.csv")
DEFAULT_OUTLIER_FIELDS = Path("/root/semvec/difftrace/out/latent_space_analysis/outlier_fields.csv")
DEFAULT_RECON_EXTREMES = Path("/root/semvec/difftrace/out/latent_space_analysis/reconstruction_error_extremes.csv")
DEFAULT_LATENT_STRUCTURE_REPORT = Path("/root/semvec/difftrace/out/latent_space_analysis/latent_structure_report.json")
DEFAULT_FULL_VS_NONTRANSPARENT = Path("/root/semvec/difftrace/out/latent_space_analysis/full_vs_nontransparent_summary.json")

KEY_COLS = ["protocol_name", "sample_id", "field_id"]
SPACE_META_COLS = [
    "protocol_name",
    "sample_id",
    "field_id",
    "transparent_flag",
    "boundary_miss",
    "mutation_count",
    "unique_metric_vectors",
]
ROLE_FAMILIES = {
    "control_like": [
        "branch_divergence_mean",
        "bb_divergence_mean",
        "cmp_divergence_mean",
        "flip_change",
    ],
    "constraint_like": [
        "lcp_change",
        "exec_scale_change",
        "loop_change",
        "cmp_count_change",
    ],
    "integrity_like": [
        "deltaf_dispersion",
        "behavior_diversity_inverse",
        "boundary_miss",
        "low_behavior_diversity",
        "reconstruction_error",
    ],
    "transparent_like": [
        "transparent_flag",
        "low_change_proxy",
        "low_behavior_diversity",
        "boundary_miss",
    ],
}


@dataclass
class Space:
    dataset_version: str
    model_type: str
    latent_dim: int
    embedding_path: Path
    df: pd.DataFrame
    latent_cols: List[str]
    reconstruction_path: Optional[Path]

    @property
    def space_id(self) -> str:
        return f"{self.dataset_version}_{self.model_type}_latent{self.latent_dim}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe latent/PCA spaces without retraining models.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing unsupervised_v1 outputs")
    parser.add_argument("--field-samples-csv", type=Path, default=DEFAULT_FIELD_SAMPLES_CSV, help="Field-level training sample table")
    parser.add_argument("--model-feature-cols", type=Path, default=DEFAULT_MODEL_FEATURE_COLS, help="model_feature_cols.txt or .json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for probe outputs")
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
    parser.add_argument("--latent-dims", type=str, default="2,3,4,8", help="Comma-separated latent dims to include")
    parser.add_argument(
        "--focus-latent-dim",
        type=int,
        default=8,
        help="Also export a dedicated top-probe summary for this latent dimension (default: 8)",
    )
    parser.add_argument("--pca-loadings", type=Path, default=DEFAULT_PCA_LOADINGS, help="Optional PCA loadings CSV")
    parser.add_argument("--representative-fields", type=Path, default=DEFAULT_REPRESENTATIVE_FIELDS, help="Optional representative_fields.csv")
    parser.add_argument("--outlier-fields", type=Path, default=DEFAULT_OUTLIER_FIELDS, help="Optional outlier_fields.csv")
    parser.add_argument("--reconstruction-extremes", type=Path, default=DEFAULT_RECON_EXTREMES, help="Optional reconstruction_error_extremes.csv")
    parser.add_argument("--latent-structure-report", type=Path, default=DEFAULT_LATENT_STRUCTURE_REPORT, help="Optional latent_structure_report.json")
    parser.add_argument("--full-vs-nontransparent-summary", type=Path, default=DEFAULT_FULL_VS_NONTRANSPARENT, help="Optional full_vs_nontransparent_summary.json")
    parser.add_argument("--correlation-method", choices=["pearson", "spearman", "both"], default="both", help="Which correlation metrics to compute")
    parser.add_argument("--top-n-probes", type=int, default=5, help="How many top probes/features to keep per dimension")
    parser.add_argument("--generate-role-profile", action="store_true", help="Generate exploratory continuous field role profiles")
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
    for name in ["matplotlib", "matplotlib.pyplot", "scipy.stats", "sklearn.preprocessing"]:
        try:
            modules[name] = __import__(name, fromlist=["*"])
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required Python packages for probe analysis: "
            f"{missing}. Please install matplotlib, scipy, and scikit-learn before running this script."
        )
    return modules


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def maybe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def maybe_load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_choice_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def selected_dataset_versions(raw: str) -> List[str]:
    return ["full", "nontransparent"] if raw == "both" else [raw]


def selected_model_types(raw: str) -> List[str]:
    return ["pca", "ae"] if raw == "both" else [raw]


def latent_cols_from_df(df: pd.DataFrame) -> List[str]:
    pca_cols = sorted([col for col in df.columns if col.startswith("pc")], key=lambda x: int(x[2:]))
    if pca_cols:
        return pca_cols
    ae_cols = sorted([col for col in df.columns if col.startswith("z")], key=lambda x: int(x[1:]))
    return ae_cols


def read_model_feature_cols(path: Path) -> List[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        data = json.load(path.open("r", encoding="utf-8"))
        if isinstance(data, dict) and "model_feature_cols" in data:
            return [str(x) for x in data["model_feature_cols"]]
        if isinstance(data, list):
            return [str(x) for x in data]
        return []
    cols = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cols.append(line)
    return cols


def robust_minmax(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99, invert: bool = False) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    lo = float(vals.quantile(lower_q))
    hi = float(vals.quantile(upper_q))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        out = pd.Series(0.0, index=series.index)
    else:
        clipped = vals.clip(lo, hi)
        out = (clipped - lo) / (hi - lo)
    if invert:
        out = 1.0 - out
    return out.fillna(0.0)


def ensure_transparent_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "transparent_flag" in out.columns:
        out["transparent_flag"] = pd.to_numeric(out["transparent_flag"], errors="coerce").fillna(0).astype(int)
        return out

    branch_div = 1.0 - pd.to_numeric(out.get("branch_sites_jaccard_mean"), errors="coerce")
    bb_div = 1.0 - pd.to_numeric(out.get("bb_set_jaccard_mean"), errors="coerce")
    cmp_div = 1.0 - pd.to_numeric(out.get("cmp_site_set_jaccard_mean"), errors="coerce")
    lcp_change = 1.0 - pd.to_numeric(out.get("lcp_ratio_mean"), errors="coerce")
    exec_change = pd.to_numeric(out.get("instr_delta_ratio_mean"), errors="coerce")
    loop_change = pd.to_numeric(out.get("bb_multiset_l1_ratio_mean"), errors="coerce")
    cmp_count = pd.to_numeric(out.get("cmp_delta_ratio_mean"), errors="coerce")
    flip_change = pd.to_numeric(out.get("branch_flip_ratio_mean"), errors="coerce")
    probe_mat = pd.concat(
        [branch_div, bb_div, cmp_div, lcp_change, exec_change, loop_change, cmp_count, flip_change],
        axis=1,
    )
    overall_change = probe_mat.fillna(0.0).mean(axis=1)
    boundary = pd.to_numeric(out.get("boundary_miss"), errors="coerce").fillna(0).astype(int)
    uniq = pd.to_numeric(out.get("unique_metric_vectors"), errors="coerce").fillna(0)
    transparent = ((boundary > 0) & (uniq <= 1)) | (overall_change <= 0.01)
    out["transparent_flag"] = transparent.astype(int)
    return out


def build_probe_table(field_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = ensure_transparent_flag(field_df)
    out = df.copy()

    out["branch_divergence_mean"] = 1.0 - pd.to_numeric(out.get("branch_sites_jaccard_mean"), errors="coerce")
    out["bb_divergence_mean"] = 1.0 - pd.to_numeric(out.get("bb_set_jaccard_mean"), errors="coerce")
    out["cmp_divergence_mean"] = 1.0 - pd.to_numeric(out.get("cmp_site_set_jaccard_mean"), errors="coerce")
    out["lcp_change"] = 1.0 - pd.to_numeric(out.get("lcp_ratio_mean"), errors="coerce")
    out["exec_scale_change"] = pd.to_numeric(out.get("instr_delta_ratio_mean"), errors="coerce")
    out["loop_change"] = pd.to_numeric(out.get("bb_multiset_l1_ratio_mean"), errors="coerce")
    out["cmp_count_change"] = pd.to_numeric(out.get("cmp_delta_ratio_mean"), errors="coerce")
    out["flip_change"] = pd.to_numeric(out.get("branch_flip_ratio_mean"), errors="coerce")
    out["deltaf_dispersion"] = pd.to_numeric(out.get("deltaf_dispersion"), errors="coerce")
    out["unique_metric_vectors"] = pd.to_numeric(out.get("unique_metric_vectors"), errors="coerce")
    out["mutation_count"] = pd.to_numeric(out.get("mutation_count"), errors="coerce")
    out["boundary_miss"] = pd.to_numeric(out.get("boundary_miss"), errors="coerce").fillna(0).astype(int)
    out["transparent_flag"] = pd.to_numeric(out.get("transparent_flag"), errors="coerce").fillna(0).astype(int)
    out["low_mutation_coverage"] = out["mutation_count"].fillna(0).lt(3).astype(int)
    out["low_behavior_diversity"] = out["unique_metric_vectors"].fillna(0).le(1).astype(int)

    change_cols = [
        "branch_divergence_mean",
        "bb_divergence_mean",
        "cmp_divergence_mean",
        "lcp_change",
        "exec_scale_change",
        "loop_change",
        "cmp_count_change",
        "flip_change",
    ]
    out["overall_change_proxy"] = out[change_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
    out["low_change_proxy"] = robust_minmax(out["overall_change_proxy"], invert=True)
    out["behavior_diversity_inverse"] = robust_minmax(out["unique_metric_vectors"], invert=True)
    out["dispersion_inverse"] = robust_minmax(out["deltaf_dispersion"], invert=True)

    continuous_probes = [
        "branch_divergence_mean",
        "bb_divergence_mean",
        "cmp_divergence_mean",
        "lcp_change",
        "exec_scale_change",
        "loop_change",
        "cmp_count_change",
        "flip_change",
        "deltaf_dispersion",
        "unique_metric_vectors",
        "mutation_count",
        "overall_change_proxy",
        "low_change_proxy",
        "behavior_diversity_inverse",
        "dispersion_inverse",
    ]
    binary_probes = [
        "boundary_miss",
        "transparent_flag",
        "low_mutation_coverage",
        "low_behavior_diversity",
    ]
    return out, continuous_probes, binary_probes


def attach_role_profiles(probe_df: pd.DataFrame) -> pd.DataFrame:
    out = probe_df.copy()
    norm = {}
    norm["branch_divergence_mean"] = robust_minmax(out["branch_divergence_mean"])
    norm["bb_divergence_mean"] = robust_minmax(out["bb_divergence_mean"])
    norm["cmp_divergence_mean"] = robust_minmax(out["cmp_divergence_mean"])
    norm["flip_change"] = robust_minmax(out["flip_change"])
    norm["lcp_change"] = robust_minmax(out["lcp_change"])
    norm["exec_scale_change"] = robust_minmax(out["exec_scale_change"])
    norm["loop_change"] = robust_minmax(out["loop_change"])
    norm["cmp_count_change"] = robust_minmax(out["cmp_count_change"])
    norm["deltaf_dispersion_low"] = robust_minmax(out["deltaf_dispersion"], invert=True)
    norm["unique_metric_vectors_low"] = robust_minmax(out["unique_metric_vectors"], invert=True)
    norm["overall_change_low"] = robust_minmax(out["overall_change_proxy"], invert=True)

    out["control_like_score"] = pd.concat(
        [norm["branch_divergence_mean"], norm["bb_divergence_mean"], norm["cmp_divergence_mean"], norm["flip_change"]],
        axis=1,
    ).mean(axis=1)
    out["constraint_like_score"] = pd.concat(
        [norm["lcp_change"], norm["exec_scale_change"], norm["loop_change"], norm["cmp_count_change"]],
        axis=1,
    ).mean(axis=1)
    out["integrity_like_score"] = pd.concat(
        [
            out["boundary_miss"].astype(float),
            out["low_behavior_diversity"].astype(float),
            norm["deltaf_dispersion_low"],
            norm["unique_metric_vectors_low"],
        ],
        axis=1,
    ).mean(axis=1)
    out["transparent_like_score"] = pd.concat(
        [
            out["transparent_flag"].astype(float),
            out["low_behavior_diversity"].astype(float),
            out["boundary_miss"].astype(float),
            norm["overall_change_low"],
        ],
        axis=1,
    ).mean(axis=1)

    score_cols = [
        "control_like_score",
        "constraint_like_score",
        "integrity_like_score",
        "transparent_like_score",
    ]
    out["top_role_profile_hint"] = out[score_cols].idxmax(axis=1)
    return out


def load_pca_spaces(input_dir: Path, dataset_version: str, target_dims: Sequence[int], quiet: bool) -> List[Space]:
    pca_dir = input_dir / dataset_version / "pca"
    embedding_path = pca_dir / "pca_embedding.csv"
    if not embedding_path.exists():
        warn(f"Missing PCA embedding: {embedding_path}")
        return []
    df = pd.read_csv(embedding_path)
    latent_cols = latent_cols_from_df(df)
    usable_dims = [dim for dim in sorted(set(target_dims)) if dim <= len(latent_cols)]
    spaces = []
    for dim in usable_dims:
        keep = [col for col in df.columns if col in SPACE_META_COLS] + [f"pc{i}" for i in range(1, dim + 1)]
        spaces.append(
            Space(
                dataset_version=dataset_version,
                model_type="pca",
                latent_dim=dim,
                embedding_path=embedding_path,
                df=df[keep].copy(),
                latent_cols=[f"pc{i}" for i in range(1, dim + 1)],
                reconstruction_path=(pca_dir / "reconstruction_errors.csv") if dim == len(latent_cols) else None,
            )
        )
    info(f"Loaded PCA spaces for {dataset_version}: {usable_dims}", quiet)
    return spaces


def load_ae_spaces(input_dir: Path, dataset_version: str, target_dims: Sequence[int], quiet: bool) -> List[Space]:
    spaces = []
    for dim in sorted(set(target_dims)):
        ae_dir = input_dir / dataset_version / f"ae_latent{dim}"
        embedding_path = ae_dir / f"ae_embedding_latent{dim}.csv"
        if not embedding_path.exists():
            continue
        df = pd.read_csv(embedding_path)
        latent_cols = latent_cols_from_df(df)
        if not latent_cols:
            continue
        spaces.append(
            Space(
                dataset_version=dataset_version,
                model_type="ae",
                latent_dim=dim,
                embedding_path=embedding_path,
                df=df.copy(),
                latent_cols=latent_cols,
                reconstruction_path=ae_dir / "reconstruction_errors.csv",
            )
        )
    info(f"Loaded AE spaces for {dataset_version}: {[s.latent_dim for s in spaces]}", quiet)
    return spaces


def load_spaces(args: argparse.Namespace) -> List[Space]:
    spaces: List[Space] = []
    target_dims = parse_choice_list(args.latent_dims)
    for dataset_version in selected_dataset_versions(args.dataset_versions):
        if args.model_types in {"pca", "both"}:
            spaces.extend(load_pca_spaces(args.input_dir, dataset_version, target_dims, args.quiet))
        if args.model_types in {"ae", "both"}:
            spaces.extend(load_ae_spaces(args.input_dir, dataset_version, target_dims, args.quiet))
    return spaces


def merge_reconstruction_error(space_df: pd.DataFrame, reconstruction_path: Optional[Path]) -> pd.DataFrame:
    df = space_df.copy()
    if reconstruction_path is None or not reconstruction_path.exists():
        return df
    recon_df = maybe_load_csv(reconstruction_path)
    if recon_df is None or recon_df.empty or "reconstruction_error" not in recon_df.columns:
        return df
    merge_cols = [col for col in KEY_COLS if col in df.columns and col in recon_df.columns]
    if not merge_cols:
        return df
    merged = df.merge(recon_df[merge_cols + ["reconstruction_error"]], on=merge_cols, how="left")
    return merged


def space_with_probes(space: Space, probe_df: pd.DataFrame) -> pd.DataFrame:
    df = merge_reconstruction_error(space.df, space.reconstruction_path)
    merged = df.merge(probe_df, on=KEY_COLS, how="left", suffixes=("", "_probe"))

    for col in ["transparent_flag", "boundary_miss", "mutation_count", "unique_metric_vectors"]:
        probe_col = f"{col}_probe"
        if probe_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[probe_col])
            merged = merged.drop(columns=[probe_col])

    if space.dataset_version == "nontransparent" and "transparent_flag" in merged.columns:
        merged = merged.loc[pd.to_numeric(merged["transparent_flag"], errors="coerce").fillna(0).eq(0)].copy()
    return merged


def valid_numeric_pair(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    return x_num.loc[mask], y_num.loc[mask]


def compute_probe_latent_correlations(
    space: Space,
    df: pd.DataFrame,
    probes: Sequence[str],
    scipy_stats: Any,
) -> pd.DataFrame:
    rows = []
    available_probes = [probe for probe in probes if probe in df.columns]
    for latent_col in space.latent_cols:
        for probe in available_probes:
            x, y = valid_numeric_pair(df[probe], df[latent_col])
            if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
                pearson_r = np.nan
                pearson_p = np.nan
                spearman_rho = np.nan
                spearman_p = np.nan
            else:
                pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
                spearman_rho, spearman_p = scipy_stats.spearmanr(x, y)
            rows.append(
                {
                    "space_id": space.space_id,
                    "dataset_version": space.dataset_version,
                    "model_type": space.model_type,
                    "latent_dim": int(space.latent_dim),
                    "dimension": latent_col,
                    "probe": probe,
                    "pearson_r": float(pearson_r) if pd.notna(pearson_r) else np.nan,
                    "pearson_pvalue": float(pearson_p) if pd.notna(pearson_p) else np.nan,
                    "spearman_rho": float(spearman_rho) if pd.notna(spearman_rho) else np.nan,
                    "spearman_pvalue": float(spearman_p) if pd.notna(spearman_p) else np.nan,
                    "abs_pearson_r": float(abs(pearson_r)) if pd.notna(pearson_r) else np.nan,
                    "abs_spearman_rho": float(abs(spearman_rho)) if pd.notna(spearman_rho) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def top_probe_per_dimension(corr_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame()
    rows = []
    for (space_id, dim), group in corr_df.groupby(["space_id", "dimension"], sort=False):
        best = group.sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=False).head(top_n).copy()
        best["rank_within_dimension"] = range(1, len(best) + 1)
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def top_dimension_per_probe(corr_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame()
    rows = []
    for (space_id, probe), group in corr_df.groupby(["space_id", "probe"], sort=False):
        best = group.sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=False).head(top_n).copy()
        best["rank_within_probe"] = range(1, len(best) + 1)
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def focused_dimension_top_probe_table(top_probe_df: pd.DataFrame, latent_dim: int) -> pd.DataFrame:
    if top_probe_df.empty:
        return pd.DataFrame()
    subset = top_probe_df.loc[top_probe_df["latent_dim"] == latent_dim].copy()
    if subset.empty:
        return subset
    order_cols = [
        "dataset_version",
        "model_type",
        "latent_dim",
        "space_id",
        "dimension",
        "rank_within_dimension",
        "probe",
        "spearman_rho",
        "pearson_r",
        "abs_spearman_rho",
        "abs_pearson_r",
        "spearman_pvalue",
        "pearson_pvalue",
    ]
    keep = [col for col in order_cols if col in subset.columns]
    subset = subset.sort_values(
        ["dataset_version", "model_type", "dimension", "rank_within_dimension"],
        ascending=[True, True, True, True],
    )
    return subset[keep].reset_index(drop=True)


def focused_dimension_markdown(top_probe_df: pd.DataFrame, latent_dim: int, top_n: int) -> str:
    if top_probe_df.empty:
        return f"# latent{latent_dim} top probes\n\nNo rows were available for latent{latent_dim}.\n"

    lines = [f"# latent{latent_dim} top probes", ""]
    for (dataset_version, model_type, space_id), group in top_probe_df.groupby(
        ["dataset_version", "model_type", "space_id"], sort=False
    ):
        lines.append(f"## {space_id}")
        lines.append("")
        for dimension, dim_group in group.groupby("dimension", sort=False):
            lines.append(f"### {dimension}")
            for _, row in dim_group.head(top_n).iterrows():
                lines.append(
                    "- "
                    f"`{row['probe']}`: "
                    f"spearman={row['spearman_rho']:.6f}, "
                    f"pearson={row['pearson_r']:.6f}, "
                    f"|spearman|={row['abs_spearman_rho']:.6f}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def top_original_features_for_pca(
    pca_loadings_df: Optional[pd.DataFrame],
    dataset_version: str,
    dimension: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    if pca_loadings_df is None or pca_loadings_df.empty:
        return []
    subset = dataset_version
    dim_upper = dimension.upper()
    if dim_upper not in pca_loadings_df.columns:
        return []
    abs_col = f"abs_{dim_upper}"
    if abs_col not in pca_loadings_df.columns:
        return []
    group = pca_loadings_df.loc[pca_loadings_df["subset"] == subset].copy()
    if group.empty:
        return []
    group = group.sort_values(abs_col, ascending=False).head(top_n)
    rows = []
    for _, row in group.iterrows():
        rows.append(
            {
                "feature": row["feature"],
                "loading": float(row[dim_upper]),
                "abs_loading": float(row[abs_col]),
            }
        )
    return rows


def interpret_dimensions(
    corr_df: pd.DataFrame,
    pca_loadings_df: Optional[pd.DataFrame],
    top_n: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if corr_df.empty:
        return pd.DataFrame(), {"rows": []}
    rows = []
    json_rows: List[Dict[str, Any]] = []
    for (space_id, dimension), group in corr_df.groupby(["space_id", "dimension"], sort=False):
        first = group.iloc[0]
        ranked = group.sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=False)
        top_probes = []
        for _, row in ranked.head(top_n).iterrows():
            top_probes.append(
                {
                    "probe": row["probe"],
                    "spearman_rho": None if pd.isna(row["spearman_rho"]) else float(row["spearman_rho"]),
                    "pearson_r": None if pd.isna(row["pearson_r"]) else float(row["pearson_r"]),
                    "abs_spearman_rho": None if pd.isna(row["abs_spearman_rho"]) else float(row["abs_spearman_rho"]),
                }
            )

        family_scores = {}
        for family, family_probes in ROLE_FAMILIES.items():
            sub = ranked.loc[ranked["probe"].isin(family_probes)]
            family_scores[family] = float(sub["abs_spearman_rho"].mean()) if not sub.empty else 0.0
        ordered_roles = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        candidate_role = ordered_roles[0][0] if ordered_roles else "unclear"
        secondary_role = ordered_roles[1][0] if len(ordered_roles) > 1 else None

        top_features = []
        if first["model_type"] == "pca":
            top_features = top_original_features_for_pca(
                pca_loadings_df,
                dataset_version=str(first["dataset_version"]),
                dimension=str(dimension),
                top_n=top_n,
            )

        row = {
            "space_id": space_id,
            "dataset_version": first["dataset_version"],
            "model_type": first["model_type"],
            "latent_dim": int(first["latent_dim"]),
            "dimension": dimension,
            "candidate_role": candidate_role,
            "secondary_role": secondary_role,
            "candidate_role_score": ordered_roles[0][1] if ordered_roles else np.nan,
            "secondary_role_score": ordered_roles[1][1] if len(ordered_roles) > 1 else np.nan,
            "top_probe_1": top_probes[0]["probe"] if len(top_probes) > 0 else None,
            "top_probe_1_spearman": top_probes[0]["spearman_rho"] if len(top_probes) > 0 else np.nan,
            "top_probe_2": top_probes[1]["probe"] if len(top_probes) > 1 else None,
            "top_probe_2_spearman": top_probes[1]["spearman_rho"] if len(top_probes) > 1 else np.nan,
            "top_probe_3": top_probes[2]["probe"] if len(top_probes) > 2 else None,
            "top_probe_3_spearman": top_probes[2]["spearman_rho"] if len(top_probes) > 2 else np.nan,
            "supporting_probe_count": len(top_probes),
            "top_original_features_json": json.dumps(top_features, ensure_ascii=False),
        }
        for family, score in ordered_roles:
            row[f"{family}_support"] = score
        rows.append(row)
        json_rows.append(
            {
                "space_id": space_id,
                "dataset_version": first["dataset_version"],
                "model_type": first["model_type"],
                "latent_dim": int(first["latent_dim"]),
                "dimension": dimension,
                "candidate_role": candidate_role,
                "secondary_role": secondary_role,
                "family_scores": {family: float(score) for family, score in ordered_roles},
                "top_probes": top_probes,
                "top_original_features": top_features,
            }
        )
    return pd.DataFrame(rows), {"rows": json_rows}


def strongest_dimension_columns(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    latent_cols = [col for col in df.columns if col.startswith("pc") or col.startswith("z")]
    if not latent_cols or df.empty:
        return pd.Series([None] * len(df), index=df.index), pd.Series([np.nan] * len(df), index=df.index)
    numeric = df[latent_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    abs_vals = numeric.abs()
    strongest_col = abs_vals.idxmax(axis=1)
    strongest_val = pd.Series(
        [numeric.loc[idx, col] if pd.notna(col) else np.nan for idx, col in strongest_col.items()],
        index=df.index,
    )
    return strongest_col, strongest_val


def attach_probe_profiles_to_special_fields(
    source_df: Optional[pd.DataFrame],
    probe_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
) -> pd.DataFrame:
    if source_df is None or source_df.empty:
        return pd.DataFrame()
    merged = source_df.merge(
        probe_df[
            KEY_COLS
            + [
                "branch_divergence_mean",
                "bb_divergence_mean",
                "cmp_divergence_mean",
                "lcp_change",
                "exec_scale_change",
                "loop_change",
                "cmp_count_change",
                "flip_change",
                "deltaf_dispersion",
                "overall_change_proxy",
                "control_like_score",
                "constraint_like_score",
                "integrity_like_score",
                "transparent_like_score",
                "top_role_profile_hint",
            ]
        ],
        on=KEY_COLS,
        how="left",
    )
    strongest_dim, strongest_val = strongest_dimension_columns(merged)
    merged["strongest_dimension"] = strongest_dim
    merged["strongest_dimension_value"] = strongest_val
    if not interpretation_df.empty:
        interp_cols = ["space_id", "dimension", "candidate_role", "secondary_role", "candidate_role_score"]
        merged = merged.merge(
            interpretation_df[interp_cols],
            left_on=["space_id", "strongest_dimension"],
            right_on=["space_id", "dimension"],
            how="left",
        )
        if "dimension" in merged.columns:
            merged = merged.drop(columns=["dimension"])
    return merged


def render_probe_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    probe_col: str,
    path: Path,
    plt: Any,
) -> None:
    if probe_col not in df.columns or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    vals = df[probe_col]
    numeric_vals = pd.to_numeric(vals, errors="coerce")
    unique_non_na = numeric_vals.dropna().unique()
    if len(unique_non_na) <= 4 and set(np.unique(unique_non_na).tolist()).issubset({0, 1}):
        categories = sorted(vals.fillna("NA").astype(str).unique().tolist())
        cmap = plt.matplotlib.colormaps.get_cmap("tab10")
        for idx, cat in enumerate(categories):
            sub = df.loc[vals.fillna("NA").astype(str) == cat]
            ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.8, color=cmap(idx % cmap.N), label=cat)
        ax.legend(loc="best", fontsize=8)
    else:
        sc = ax.scatter(df[x_col], df[y_col], c=numeric_vals, cmap="viridis", s=18, alpha=0.82)
        fig.colorbar(sc, ax=ax, label=probe_col)
    ax.set_title(probe_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_correlation_heatmap(
    corr_df: pd.DataFrame,
    path: Path,
    plt: Any,
) -> None:
    if corr_df.empty:
        return
    pivot = corr_df.pivot(index="probe", columns="dimension", values="spearman_rho").sort_index()
    if pivot.empty:
        return
    fig_w = max(6, 1.1 * len(pivot.columns) + 2)
    fig_h = max(7, 0.35 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Probe-Latent Correlations (Spearman)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_top_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    path: Path,
    plt: Any,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = df[label_col].astype(str).tolist()
    vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).tolist()
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_marked_fields(
    base_df: pd.DataFrame,
    special_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    path: Path,
    plt: Any,
) -> None:
    if base_df.empty or special_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(base_df[x_col], base_df[y_col], s=12, alpha=0.25, color="#9aa0a6")
    ax.scatter(special_df[x_col], special_df[y_col], s=28, alpha=0.9, color="#d62728")
    for _, row in special_df.head(20).iterrows():
        ax.text(row[x_col], row[y_col], f"{row['protocol_name']}:{row['field_id']}", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def space_summary_interpretation(interpret_df: pd.DataFrame, space_id: str) -> Dict[str, Any]:
    group = interpret_df.loc[interpret_df["space_id"] == space_id].copy()
    if group.empty:
        return {}
    return {
        "space_id": space_id,
        "dimension_roles": group[["dimension", "candidate_role", "secondary_role", "candidate_role_score"]].to_dict(orient="records"),
        "role_frequency": group["candidate_role"].value_counts().to_dict(),
    }


def attach_reference_coords(role_df: pd.DataFrame, spaces: Sequence[Space]) -> pd.DataFrame:
    out = role_df.copy()
    selected_refs = [
        ("full", "pca", 2),
        ("full", "ae", 2),
        ("nontransparent", "pca", 2),
        ("nontransparent", "ae", 2),
    ]
    for dataset_version, model_type, latent_dim in selected_refs:
        target = next(
            (
                space
                for space in spaces
                if space.dataset_version == dataset_version and space.model_type == model_type and space.latent_dim == latent_dim
            ),
            None,
        )
        if target is None:
            continue
        ref = target.df[KEY_COLS + target.latent_cols[:2]].copy()
        renamed = {
            target.latent_cols[0]: f"{dataset_version}_{model_type}_dim1",
            target.latent_cols[1]: f"{dataset_version}_{model_type}_dim2",
        }
        ref = ref.rename(columns=renamed)
        out = out.merge(ref, on=KEY_COLS, how="left")
    return out


def unique_preserve_order(columns: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def main() -> int:
    args = parse_args()
    imported = ensure_dependencies()
    plt = imported["matplotlib.pyplot"]
    scipy_stats = imported["scipy.stats"]

    model_feature_cols = read_model_feature_cols(args.model_feature_cols)
    field_df = pd.read_csv(args.field_samples_csv)
    probe_df, continuous_probes, binary_probes = build_probe_table(field_df)
    if args.generate_role_profile:
        probe_df = attach_role_profiles(probe_df)

    spaces = load_spaces(args)
    if not spaces:
        warn("No embedding spaces found.")
        return 1

    pca_loadings_df = maybe_load_csv(args.pca_loadings)
    representative_df = maybe_load_csv(args.representative_fields)
    outlier_df = maybe_load_csv(args.outlier_fields)
    recon_extremes_df = maybe_load_csv(args.reconstruction_extremes)
    latent_structure_report = maybe_load_json(args.latent_structure_report)
    full_vs_nontransparent_summary = maybe_load_json(args.full_vs_nontransparent_summary)

    all_corr_rows = []
    per_space_top_dim_rows = []
    per_space_top_probe_rows = []
    interpretation_frames = []
    interpretation_json_rows = []
    embeddings_used = []

    probe_cols = continuous_probes + binary_probes
    output_plots_dir = args.output_dir / "plots"

    for space in spaces:
        merged = space_with_probes(space, probe_df)
        if merged.empty:
            warn(f"Empty merged dataframe for {space.space_id}")
            continue
        if "reconstruction_error" in merged.columns and merged["reconstruction_error"].notna().any():
            if "reconstruction_error" not in probe_cols:
                local_probe_cols = probe_cols + ["reconstruction_error"]
            else:
                local_probe_cols = probe_cols
        else:
            local_probe_cols = probe_cols

        corr_df = compute_probe_latent_correlations(space, merged, local_probe_cols, scipy_stats)
        if corr_df.empty:
            continue
        all_corr_rows.append(corr_df)

        top_probe_df = top_probe_per_dimension(corr_df, args.top_n_probes)
        top_dim_df = top_dimension_per_probe(corr_df, args.top_n_probes)
        if not top_probe_df.empty:
            per_space_top_dim_rows.append(top_probe_df)
        if not top_dim_df.empty:
            per_space_top_probe_rows.append(top_dim_df)

        interpret_df, interpret_json = interpret_dimensions(corr_df, pca_loadings_df, args.top_n_probes)
        if not interpret_df.empty:
            interpretation_frames.append(interpret_df)
            interpretation_json_rows.extend(interpret_json["rows"])

        plot_space_dir = output_plots_dir / space.space_id
        if len(space.latent_cols) >= 2:
            x_col, y_col = space.latent_cols[0], space.latent_cols[1]
            for probe in [
                "branch_divergence_mean",
                "lcp_change",
                "exec_scale_change",
                "loop_change",
                "deltaf_dispersion",
                "unique_metric_vectors",
                "reconstruction_error",
                "transparent_flag",
                "boundary_miss",
            ]:
                if probe in merged.columns:
                    render_probe_scatter(
                        merged,
                        x_col,
                        y_col,
                        probe,
                        plot_space_dir / f"{space.space_id}_by_{probe}.png",
                        plt,
                    )
            render_correlation_heatmap(corr_df, plot_space_dir / f"{space.space_id}_probe_latent_heatmap.png", plt)

            if not top_probe_df.empty:
                dim_bar = (
                    top_probe_df.loc[top_probe_df["rank_within_dimension"] == 1, ["dimension", "abs_spearman_rho"]]
                    .sort_values("dimension")
                    .copy()
                )
                render_top_bar(
                    dim_bar,
                    label_col="dimension",
                    value_col="abs_spearman_rho",
                    title=f"{space.space_id} top probe strength by dimension",
                    path=plot_space_dir / f"{space.space_id}_top_probe_per_dimension.png",
                    plt=plt,
                )
            if not top_dim_df.empty:
                probe_bar = (
                    top_dim_df.loc[top_dim_df["rank_within_probe"] == 1, ["probe", "abs_spearman_rho"]]
                    .sort_values("abs_spearman_rho", ascending=False)
                    .head(12)
                    .copy()
                )
                render_top_bar(
                    probe_bar,
                    label_col="probe",
                    value_col="abs_spearman_rho",
                    title=f"{space.space_id} top dimension per probe",
                    path=plot_space_dir / f"{space.space_id}_top_dimension_per_probe.png",
                    plt=plt,
                )

            if representative_df is not None and not representative_df.empty:
                reps = representative_df.loc[representative_df["space_id"] == space.space_id].copy()
                if not reps.empty:
                    render_marked_fields(
                        merged,
                        reps,
                        x_col,
                        y_col,
                        f"{space.space_id} representative fields",
                        plot_space_dir / f"{space.space_id}_representative_fields.png",
                        plt,
                    )
            if outlier_df is not None and not outlier_df.empty:
                outs = outlier_df.loc[outlier_df["space_id"] == space.space_id].copy()
                if not outs.empty:
                    render_marked_fields(
                        merged,
                        outs,
                        x_col,
                        y_col,
                        f"{space.space_id} outlier fields",
                        plot_space_dir / f"{space.space_id}_outlier_fields.png",
                        plt,
                    )
            if recon_extremes_df is not None and not recon_extremes_df.empty:
                ex = recon_extremes_df.loc[recon_extremes_df["space_id"] == space.space_id].copy()
                if not ex.empty:
                    render_marked_fields(
                        merged,
                        ex,
                        x_col,
                        y_col,
                        f"{space.space_id} reconstruction error extremes",
                        plot_space_dir / f"{space.space_id}_reconstruction_error_extremes.png",
                        plt,
                    )

        embeddings_used.append(
            {
                "space_id": space.space_id,
                "dataset_version": space.dataset_version,
                "model_type": space.model_type,
                "latent_dim": space.latent_dim,
                "embedding_path": str(space.embedding_path),
            }
        )

    corr_df = pd.concat(all_corr_rows, ignore_index=True) if all_corr_rows else pd.DataFrame()
    top_probe_df = pd.concat(per_space_top_dim_rows, ignore_index=True) if per_space_top_dim_rows else pd.DataFrame()
    top_dim_df = pd.concat(per_space_top_probe_rows, ignore_index=True) if per_space_top_probe_rows else pd.DataFrame()
    interpretation_df = pd.concat(interpretation_frames, ignore_index=True) if interpretation_frames else pd.DataFrame()

    write_table(corr_df, args.output_dir / "probe_latent_correlations.csv")
    write_table(top_probe_df, args.output_dir / "top_probe_per_dimension.csv")
    write_table(top_dim_df, args.output_dir / "top_dimension_per_probe.csv")
    write_table(interpretation_df, args.output_dir / "latent_dimension_interpretation.csv")
    write_json({"rows": interpretation_json_rows}, args.output_dir / "latent_dimension_interpretation.json")

    focused_top_probe_df = focused_dimension_top_probe_table(top_probe_df, args.focus_latent_dim)
    write_table(
        focused_top_probe_df,
        args.output_dir / f"latent{args.focus_latent_dim}_top_probe_per_dimension.csv",
    )
    write_text(
        focused_dimension_markdown(focused_top_probe_df, args.focus_latent_dim, args.top_n_probes),
        args.output_dir / f"latent{args.focus_latent_dim}_top_probe_per_dimension.md",
    )

    rep_profiles = attach_probe_profiles_to_special_fields(representative_df, probe_df, interpretation_df)
    outlier_profiles = attach_probe_profiles_to_special_fields(outlier_df, probe_df, interpretation_df)
    recon_profiles = attach_probe_profiles_to_special_fields(recon_extremes_df, probe_df, interpretation_df)
    write_table(rep_profiles, args.output_dir / "representative_field_probe_profiles.csv")
    write_table(outlier_profiles, args.output_dir / "outlier_field_probe_profiles.csv")
    write_table(recon_profiles, args.output_dir / "reconstruction_extreme_probe_profiles.csv")

    if args.generate_role_profile:
        role_cols = unique_preserve_order(KEY_COLS + [
            "control_like_score",
            "constraint_like_score",
            "integrity_like_score",
            "transparent_like_score",
            "top_role_profile_hint",
            "boundary_miss",
            "transparent_flag",
            "mutation_count",
            "unique_metric_vectors",
        ])
        role_df = probe_df[[col for col in role_cols if col in probe_df.columns]].copy()
        role_df = attach_reference_coords(role_df, spaces)
        write_table(role_df, args.output_dir / "field_role_profiles.csv")

    summary_rows = []
    for space_id in sorted({space.space_id for space in spaces}):
        summary_rows.append(space_summary_interpretation(interpretation_df, space_id))

    consistency_pca_vs_ae = []
    for dataset_version in selected_dataset_versions(args.dataset_versions):
        for latent_dim in parse_choice_list(args.latent_dims):
            pca_group = interpretation_df.loc[
                (interpretation_df["dataset_version"] == dataset_version)
                & (interpretation_df["model_type"] == "pca")
                & (interpretation_df["latent_dim"] == latent_dim)
            ]
            ae_group = interpretation_df.loc[
                (interpretation_df["dataset_version"] == dataset_version)
                & (interpretation_df["model_type"] == "ae")
                & (interpretation_df["latent_dim"] == latent_dim)
            ]
            if pca_group.empty or ae_group.empty:
                continue
            pca_roles = pca_group["candidate_role"].tolist()
            ae_roles = ae_group["candidate_role"].tolist()
            overlap = len(set(pca_roles) & set(ae_roles))
            consistency_pca_vs_ae.append(
                {
                    "dataset_version": dataset_version,
                    "latent_dim": latent_dim,
                    "pca_candidate_roles": pca_roles,
                    "ae_candidate_roles": ae_roles,
                    "shared_role_count": overlap,
                }
            )

    consistency_full_vs_nontransparent = []
    for model_type in selected_model_types(args.model_types):
        for latent_dim in parse_choice_list(args.latent_dims):
            full_group = interpretation_df.loc[
                (interpretation_df["dataset_version"] == "full")
                & (interpretation_df["model_type"] == model_type)
                & (interpretation_df["latent_dim"] == latent_dim)
            ]
            non_group = interpretation_df.loc[
                (interpretation_df["dataset_version"] == "nontransparent")
                & (interpretation_df["model_type"] == model_type)
                & (interpretation_df["latent_dim"] == latent_dim)
            ]
            if full_group.empty or non_group.empty:
                continue
            full_roles = full_group["candidate_role"].tolist()
            non_roles = non_group["candidate_role"].tolist()
            overlap = len(set(full_roles) & set(non_roles))
            consistency_full_vs_nontransparent.append(
                {
                    "model_type": model_type,
                    "latent_dim": latent_dim,
                    "full_candidate_roles": full_roles,
                    "nontransparent_candidate_roles": non_roles,
                    "shared_role_count": overlap,
                }
            )

    noteworthy_fields = []
    for df_name, df in [
        ("representative", rep_profiles),
        ("outlier", outlier_profiles),
        ("reconstruction_extreme", recon_profiles),
    ]:
        if df is None or df.empty:
            continue
        cols = [col for col in ["space_id", "protocol_name", "sample_id", "field_id", "top_role_profile_hint", "strongest_dimension", "candidate_role"] if col in df.columns]
        subset = df[cols].head(20).copy()
        subset["source"] = df_name
        noteworthy_fields.extend(subset.to_dict(orient="records"))

    suggested_probes = []
    if not top_probe_df.empty:
        probe_counts = top_probe_df.loc[top_probe_df["rank_within_dimension"] == 1, "probe"].value_counts()
        for probe, count in probe_counts.head(12).items():
            suggested_probes.append({"probe": probe, "top_rank_count": int(count)})

    summary = {
        "embeddings_used": embeddings_used,
        "model_feature_cols": model_feature_cols,
        "probes_used": {
            "continuous": continuous_probes,
            "binary": binary_probes,
        },
        "space_interpretations": summary_rows,
        "pca_vs_ae_interpretation_overlap": consistency_pca_vs_ae,
        "full_vs_nontransparent_interpretation_overlap": consistency_full_vs_nontransparent,
        "noteworthy_fields": noteworthy_fields,
        "suggested_probe_refinements": suggested_probes,
        "latent_structure_report_used": latent_structure_report is not None,
        "full_vs_nontransparent_summary_used": full_vs_nontransparent_summary is not None,
        "focused_latent_dim_report": {
            "latent_dim": args.focus_latent_dim,
            "csv": str(args.output_dir / f"latent{args.focus_latent_dim}_top_probe_per_dimension.csv"),
            "markdown": str(args.output_dir / f"latent{args.focus_latent_dim}_top_probe_per_dimension.md"),
            "row_count": int(len(focused_top_probe_df)),
        },
    }
    write_json(summary, args.output_dir / "probe_analysis_summary.json")

    info(f"Wrote probe analysis outputs to {args.output_dir}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
