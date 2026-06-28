#!/usr/bin/env python3
"""Inference pipeline for field-level probe/latent explanations from report_compact.json."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_REPORT = Path("/root/semvec/difftrace/out/out_bacnet/sample_001/report_compact.json")
DEFAULT_MODEL_FEATURE_COLS = Path("/root/semvec/difftrace/out/dataset_health_report/model_feature_cols.txt")
DEFAULT_UNSUP_DIR = Path("/root/semvec/difftrace/out/unsupervised_v1")
DEFAULT_REPRESENTATIVE_FIELDS = Path("/root/semvec/difftrace/out/latent_space_analysis/representative_fields.csv")
DEFAULT_INTERPRETATION = Path("/root/semvec/difftrace/out/probe_analysis_v1/latent_dimension_interpretation.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/inference_v1")

KEY_COLS = ["protocol_name", "sample_id", "field_id"]
BASE_METRICS = [
    "branch_sites_jaccard",
    "bb_set_jaccard",
    "cmp_site_set_jaccard",
    "lcp_ratio",
    "instr_delta_ratio",
    "bb_multiset_l1_ratio",
    "cmp_delta_ratio",
    "branch_flip_ratio",
]


@dataclass
class SpaceArtifacts:
    dataset_version: str
    model_type: str
    latent_dim: int
    scaler: Any
    model: Any
    latent_cols: List[str]
    interpretation_df: pd.DataFrame
    representative_df: pd.DataFrame

    @property
    def space_id(self) -> str:
        return f"{self.dataset_version}_{self.model_type}_latent{self.latent_dim}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer probe composition and latent8 positions from report_compact.json")
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT, help="Path to report_compact.json")
    parser.add_argument("--model-feature-cols", type=Path, default=DEFAULT_MODEL_FEATURE_COLS, help="Path to model_feature_cols.txt/json")
    parser.add_argument("--unsupervised-dir", type=Path, default=DEFAULT_UNSUP_DIR, help="Path to unsupervised_v1 outputs")
    parser.add_argument("--representative-fields", type=Path, default=DEFAULT_REPRESENTATIVE_FIELDS, help="Path to representative_fields.csv")
    parser.add_argument("--interpretation-csv", type=Path, default=DEFAULT_INTERPRETATION, help="Path to latent_dimension_interpretation.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for inference outputs")
    parser.add_argument("--neighbors-k", type=int, default=5, help="Top-k representative neighbors per space")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    return parser.parse_args()


def info(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ensure_dependencies() -> None:
    missing = []
    for name in ["sklearn", "torch"]:
        try:
            __import__(name)
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required Python packages for inference: "
            f"{missing}. Please install scikit-learn and torch in the runtime environment."
        )


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_model_feature_cols(path: Path) -> List[str]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "model_feature_cols" in obj:
            return [str(x) for x in obj["model_feature_cols"]]
        if isinstance(obj, list):
            return [str(x) for x in obj]
        raise ValueError(f"Unsupported JSON structure in {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def parse_context_from_report_path(path: Path) -> Tuple[str, str]:
    sample_id = path.parent.name
    protocol_dir = path.parent.parent.name
    protocol_name = protocol_dir
    if protocol_name.startswith("out_"):
        protocol_name = protocol_name[4:]
    return protocol_name, sample_id


def field_range_and_nbytes(field_id: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        left, right = field_id.split("_", 1)
        start = int(left)
        end = int(right)
        return start, end, end - start + 1
    except Exception:
        return None, None, None


def baseline_hex_to_value(hex_text: Optional[str]) -> Optional[int]:
    if not hex_text:
        return None
    hex_text = str(hex_text).strip().replace(" ", "")
    if not hex_text:
        return None
    try:
        return int(hex_text, 16)
    except Exception:
        return None


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def unique_metric_vector_count(per_mutation: Sequence[Dict[str, Any]], decimals: int = 6) -> int:
    vectors = []
    for item in per_mutation:
        metrics = item.get("metrics") or {}
        vec = []
        for metric in BASE_METRICS:
            val = metrics.get(metric)
            vec.append(None if val is None else round(float(val), decimals))
        vectors.append(tuple(vec))
    return len(set(vectors))


def deltaf_dispersion(per_mutation: Sequence[Dict[str, Any]]) -> float:
    per_metric_std = []
    for metric in BASE_METRICS:
        vals = []
        for item in per_mutation:
            metrics = item.get("metrics") or {}
            if metric in metrics and metrics[metric] is not None:
                vals.append(float(metrics[metric]))
        if vals:
            per_metric_std.append(float(np.std(np.asarray(vals, dtype=float))))
    return float(np.mean(per_metric_std)) if per_metric_std else 0.0


def infer_boundary_miss(boundary_miss: Optional[Any], uniq: int, dispersion: float, eps: float = 1e-6) -> int:
    if boundary_miss is not None:
        try:
            return int(boundary_miss)
        except Exception:
            pass
    return 1 if uniq <= 1 or dispersion <= eps else 0


def build_field_level_rows(report_compact_path: Path) -> pd.DataFrame:
    obj = json.loads(report_compact_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "fields" not in obj:
        raise ValueError(f"Unsupported report_compact.json structure: {report_compact_path}")
    protocol_name, sample_id = parse_context_from_report_path(report_compact_path)
    rows: List[Dict[str, Any]] = []
    for field in obj.get("fields", []):
        field_id = str(field.get("field_id"))
        per_mutation = field.get("per_mutation") or []
        row: Dict[str, Any] = {
            "protocol_name": protocol_name,
            "sample_id": sample_id,
            "field_id": field_id,
            "baseline_field_hex": field.get("basline_filed_hex"),
        }
        start, end, nbytes = field_range_and_nbytes(field_id)
        row["field_range_start"] = start
        row["field_range_end"] = end
        row["nbytes"] = nbytes
        row["baseline_value"] = baseline_hex_to_value(field.get("basline_filed_hex"))
        row["mutation_count"] = int(len(per_mutation))
        row["valid_mutations"] = int(len(per_mutation))

        uniq = unique_metric_vector_count(per_mutation)
        dispersion = deltaf_dispersion(per_mutation)
        row["unique_metric_vectors"] = int(uniq)
        row["deltaf_dispersion"] = float(dispersion)
        row["boundary_miss"] = infer_boundary_miss(field.get("boundary_miss"), uniq, dispersion)

        for metric in BASE_METRICS:
            vals = []
            for item in per_mutation:
                metrics = item.get("metrics") or {}
                if metric in metrics and metrics[metric] is not None:
                    vals.append(float(metrics[metric]))
            stats = aggregate_metric(vals)
            for stat_name, stat_value in stats.items():
                row[f"{metric}_{stat_name}"] = stat_value
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_transparent_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    branch_div = 1.0 - pd.to_numeric(out.get("branch_sites_jaccard_mean"), errors="coerce")
    bb_div = 1.0 - pd.to_numeric(out.get("bb_set_jaccard_mean"), errors="coerce")
    cmp_div = 1.0 - pd.to_numeric(out.get("cmp_site_set_jaccard_mean"), errors="coerce")
    lcp_change = 1.0 - pd.to_numeric(out.get("lcp_ratio_mean"), errors="coerce")
    exec_change = pd.to_numeric(out.get("instr_delta_ratio_mean"), errors="coerce")
    loop_change = pd.to_numeric(out.get("bb_multiset_l1_ratio_mean"), errors="coerce")
    cmp_count = pd.to_numeric(out.get("cmp_delta_ratio_mean"), errors="coerce")
    flip_change = pd.to_numeric(out.get("branch_flip_ratio_mean"), errors="coerce")
    probe_mat = pd.concat([branch_div, bb_div, cmp_div, lcp_change, exec_change, loop_change, cmp_count, flip_change], axis=1)
    overall_change = probe_mat.fillna(0.0).mean(axis=1)
    boundary = pd.to_numeric(out.get("boundary_miss"), errors="coerce").fillna(0).astype(int)
    uniq = pd.to_numeric(out.get("unique_metric_vectors"), errors="coerce").fillna(0)
    out["transparent_flag"] = (((boundary > 0) & (uniq <= 1)) | (overall_change <= 0.01)).astype(int)
    return out


def build_probe_table(field_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_transparent_flag(field_df).copy()
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

    norm = {
        "branch_divergence_mean": robust_minmax(out["branch_divergence_mean"]),
        "bb_divergence_mean": robust_minmax(out["bb_divergence_mean"]),
        "cmp_divergence_mean": robust_minmax(out["cmp_divergence_mean"]),
        "flip_change": robust_minmax(out["flip_change"]),
        "lcp_change": robust_minmax(out["lcp_change"]),
        "exec_scale_change": robust_minmax(out["exec_scale_change"]),
        "loop_change": robust_minmax(out["loop_change"]),
        "cmp_count_change": robust_minmax(out["cmp_count_change"]),
        "deltaf_dispersion_low": robust_minmax(out["deltaf_dispersion"], invert=True),
        "unique_metric_vectors_low": robust_minmax(out["unique_metric_vectors"], invert=True),
        "overall_change_low": robust_minmax(out["overall_change_proxy"], invert=True),
    }
    out["control_like_score"] = pd.concat(
        [norm["branch_divergence_mean"], norm["bb_divergence_mean"], norm["cmp_divergence_mean"], norm["flip_change"]],
        axis=1,
    ).mean(axis=1)
    out["constraint_like_score"] = pd.concat(
        [norm["lcp_change"], norm["exec_scale_change"], norm["loop_change"], norm["cmp_count_change"]],
        axis=1,
    ).mean(axis=1)
    out["integrity_like_score"] = pd.concat(
        [out["boundary_miss"].astype(float), out["low_behavior_diversity"].astype(float), norm["deltaf_dispersion_low"], norm["unique_metric_vectors_low"]],
        axis=1,
    ).mean(axis=1)
    out["transparent_like_score"] = pd.concat(
        [out["transparent_flag"].astype(float), out["low_behavior_diversity"].astype(float), out["boundary_miss"].astype(float), norm["overall_change_low"]],
        axis=1,
    ).mean(axis=1)
    score_cols = ["control_like_score", "constraint_like_score", "integrity_like_score", "transparent_like_score"]
    out["top_role_profile_hint"] = out[score_cols].idxmax(axis=1)
    return out


def define_autoencoder(torch: Any, input_dim: int, latent_dim: int) -> Any:
    import torch.nn as nn

    class AutoEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim),
            )

        def forward(self, x: Any) -> Tuple[Any, Any]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

    return AutoEncoder()


def load_representative_reference(path: Path, space_id: str, latent_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df["space_id"] == space_id].copy()
    latent_cols = [f"{latent_prefix}{i}" for i in range(1, 9) if f"{latent_prefix}{i}" in df.columns]
    df = df.dropna(subset=latent_cols, how="any").copy()
    if df.empty:
        return df
    grouped = (
        df.groupby(KEY_COLS + latent_cols, dropna=False)
        .agg(
            selection_types=("selection_type", lambda s: "|".join(sorted({str(x) for x in s.dropna()}))),
            group_values=("group_value", lambda s: "|".join(sorted({str(x) for x in s.dropna()}))),
            transparent_flag=("transparent_flag", "first"),
            boundary_miss=("boundary_miss", "first"),
            mutation_count=("mutation_count", "first"),
            unique_metric_vectors=("unique_metric_vectors", "first"),
        )
        .reset_index()
    )
    return grouped


def load_interpretation(path: Path, space_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["space_id"] == space_id].copy()


def load_space_artifacts(args: argparse.Namespace) -> Dict[str, SpaceArtifacts]:
    import torch

    spaces: Dict[str, SpaceArtifacts] = {}
    interpretation_csv = args.interpretation_csv
    for dataset_version, model_type, latent_prefix in [
        ("full", "pca", "pc"),
        ("full", "ae", "z"),
        ("nontransparent", "pca", "pc"),
        ("nontransparent", "ae", "z"),
    ]:
        space_id = f"{dataset_version}_{model_type}_latent8"
        if model_type == "pca":
            model_dir = args.unsupervised_dir / dataset_version / "pca"
            scaler = pickle.load((model_dir / "scaler.pkl").open("rb"))
            model = pickle.load((model_dir / "pca_model.pkl").open("rb"))
            latent_cols = [f"pc{i}" for i in range(1, 9)]
        else:
            model_dir = args.unsupervised_dir / dataset_version / "ae_latent8"
            scaler = pickle.load((model_dir / "scaler.pkl").open("rb"))
            state_dict = torch.load(model_dir / "ae_best.pt", map_location="cpu")
            model = define_autoencoder(torch, input_dim=int(getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", [])))), latent_dim=8)
            model.load_state_dict(state_dict)
            model.eval()
            latent_cols = [f"z{i}" for i in range(1, 9)]
        spaces[space_id] = SpaceArtifacts(
            dataset_version=dataset_version,
            model_type=model_type,
            latent_dim=8,
            scaler=scaler,
            model=model,
            latent_cols=latent_cols,
            interpretation_df=load_interpretation(interpretation_csv, space_id),
            representative_df=load_representative_reference(args.representative_fields, space_id, latent_prefix),
        )
    return spaces


def run_pca_inference(model: Any, scaler: Any, X: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.transform(X_scaled)[:, :8]


def run_ae_inference(model: Any, scaler: Any, X: np.ndarray) -> np.ndarray:
    import torch

    X_scaled = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        tensor = torch.tensor(X_scaled, dtype=torch.float32)
        _, z = model(tensor)
    return z.detach().cpu().numpy()


def nearest_representatives(reference_df: pd.DataFrame, latent_vec: np.ndarray, latent_cols: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
    if reference_df is None or reference_df.empty:
        return []
    coords = reference_df[list(latent_cols)].to_numpy(dtype=float)
    dists = np.linalg.norm(coords - latent_vec[None, :], axis=1)
    order = np.argsort(dists)[:top_k]
    rows = []
    for idx in order:
        row = reference_df.iloc[int(idx)]
        rows.append(
            {
                "protocol_name": row["protocol_name"],
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
                "distance": float(dists[int(idx)]),
                "selection_types": row.get("selection_types"),
                "group_values": row.get("group_values"),
                "boundary_miss": int(row["boundary_miss"]) if pd.notna(row.get("boundary_miss")) else None,
                "transparent_flag": int(row["transparent_flag"]) if pd.notna(row.get("transparent_flag")) else None,
                "mutation_count": int(row["mutation_count"]) if pd.notna(row.get("mutation_count")) else None,
                "unique_metric_vectors": int(row["unique_metric_vectors"]) if pd.notna(row.get("unique_metric_vectors")) else None,
            }
        )
    return rows


def strongest_dimensions(latent_vec: np.ndarray, interpretation_df: pd.DataFrame, latent_cols: Sequence[str], top_k: int = 3) -> List[Dict[str, Any]]:
    order = np.argsort(np.abs(latent_vec))[::-1][:top_k]
    interp_lookup = {str(row["dimension"]): row for _, row in interpretation_df.iterrows()} if interpretation_df is not None else {}
    rows = []
    for idx in order:
        dim = latent_cols[int(idx)]
        interp = interp_lookup.get(dim, {})
        rows.append(
            {
                "dimension": dim,
                "value": float(latent_vec[int(idx)]),
                "abs_value": float(abs(latent_vec[int(idx)])),
                "candidate_role": interp.get("candidate_role"),
                "secondary_role": interp.get("secondary_role"),
                "top_probe_1": interp.get("top_probe_1"),
                "top_probe_2": interp.get("top_probe_2"),
                "top_probe_3": interp.get("top_probe_3"),
            }
        )
    return rows


def feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    X = df[list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.to_numpy(dtype=float)


def infer_spaces_for_fields(df: pd.DataFrame, feature_cols: Sequence[str], spaces: Dict[str, SpaceArtifacts], top_k: int) -> List[Dict[str, Any]]:
    results = []
    X = feature_matrix(df, feature_cols)
    full_pca = run_pca_inference(spaces["full_pca_latent8"].model, spaces["full_pca_latent8"].scaler, X)
    full_ae = run_ae_inference(spaces["full_ae_latent8"].model, spaces["full_ae_latent8"].scaler, X)

    non_mask = pd.to_numeric(df["transparent_flag"], errors="coerce").fillna(0).eq(0).to_numpy()
    non_pca = np.full((len(df), 8), np.nan, dtype=float)
    non_ae = np.full((len(df), 8), np.nan, dtype=float)
    if np.any(non_mask):
        X_non = X[non_mask]
        non_pca_vals = run_pca_inference(spaces["nontransparent_pca_latent8"].model, spaces["nontransparent_pca_latent8"].scaler, X_non)
        non_ae_vals = run_ae_inference(spaces["nontransparent_ae_latent8"].model, spaces["nontransparent_ae_latent8"].scaler, X_non)
        non_pca[non_mask] = non_pca_vals
        non_ae[non_mask] = non_ae_vals

    for i, row in df.reset_index(drop=True).iterrows():
        row_result = {
            "protocol_name": row["protocol_name"],
            "sample_id": row["sample_id"],
            "field_id": row["field_id"],
            "transparent_flag": int(row["transparent_flag"]),
            "boundary_miss": int(row["boundary_miss"]),
            "probe_summary": {
                "top_role_profile_hint": row["top_role_profile_hint"],
                "control_like_score": float(row["control_like_score"]),
                "constraint_like_score": float(row["constraint_like_score"]),
                "integrity_like_score": float(row["integrity_like_score"]),
                "transparent_like_score": float(row["transparent_like_score"]),
            },
            "spaces": {},
        }
        for space_id, latent_vec in [
            ("full_pca_latent8", full_pca[i]),
            ("full_ae_latent8", full_ae[i]),
            ("nontransparent_pca_latent8", non_pca[i]),
            ("nontransparent_ae_latent8", non_ae[i]),
        ]:
            if np.isnan(latent_vec).any():
                row_result["spaces"][space_id] = {
                    "skipped": True,
                    "reason": "transparent_flag=1 so nontransparent inference was skipped",
                }
                continue
            artifacts = spaces[space_id]
            row_result["spaces"][space_id] = {
                "skipped": False,
                "latent_cols": artifacts.latent_cols,
                "latent_values": {col: float(val) for col, val in zip(artifacts.latent_cols, latent_vec.tolist())},
                "strongest_dimensions": strongest_dimensions(latent_vec, artifacts.interpretation_df, artifacts.latent_cols, top_k=3),
                "nearest_representatives": nearest_representatives(
                    artifacts.representative_df, latent_vec, artifacts.latent_cols, top_k=top_k
                ),
            }
        results.append(row_result)
    return results


def field_json_record(row: pd.Series, space_result: Dict[str, Any]) -> Dict[str, Any]:
    probe_cols = [
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
        "boundary_miss",
        "transparent_flag",
        "low_mutation_coverage",
        "low_behavior_diversity",
        "control_like_score",
        "constraint_like_score",
        "integrity_like_score",
        "transparent_like_score",
        "top_role_profile_hint",
    ]
    feature_cols = [f"{metric}_{stat}" for metric in BASE_METRICS for stat in ["mean", "std", "min", "max"]]
    return {
        "protocol_name": row["protocol_name"],
        "sample_id": row["sample_id"],
        "field_id": row["field_id"],
        "field_range_start": None if pd.isna(row.get("field_range_start")) else int(row["field_range_start"]),
        "field_range_end": None if pd.isna(row.get("field_range_end")) else int(row["field_range_end"]),
        "nbytes": None if pd.isna(row.get("nbytes")) else int(row["nbytes"]),
        "baseline_field_hex": row.get("baseline_field_hex"),
        "baseline_value": None if pd.isna(row.get("baseline_value")) else int(row["baseline_value"]),
        "model_features": {col: None if pd.isna(row.get(col)) else float(row[col]) for col in feature_cols},
        "probes": {col: (row[col].item() if hasattr(row[col], "item") else row[col]) for col in probe_cols if col in row.index},
        "inference": space_result["spaces"],
    }


def markdown_report(records: List[Dict[str, Any]], input_report: Path) -> str:
    lines = [
        "# Field Inference Report",
        "",
        f"- input_report: `{input_report}`",
        f"- field_count: `{len(records)}`",
        "",
    ]
    for rec in records:
        lines.append(f"## {rec['protocol_name']} / {rec['sample_id']} / {rec['field_id']}")
        lines.append("")
        lines.append(
            f"- boundary_miss: `{rec['probes'].get('boundary_miss')}`"
            f", transparent_flag: `{rec['probes'].get('transparent_flag')}`"
            f", mutation_count: `{rec['probes'].get('mutation_count')}`"
            f", unique_metric_vectors: `{rec['probes'].get('unique_metric_vectors')}`"
        )
        lines.append(
            f"- role_profile: `control={rec['probes'].get('control_like_score'):.4f}`"
            f", `constraint={rec['probes'].get('constraint_like_score'):.4f}`"
            f", `integrity={rec['probes'].get('integrity_like_score'):.4f}`"
            f", `transparent={rec['probes'].get('transparent_like_score'):.4f}`"
            f", top=`{rec['probes'].get('top_role_profile_hint')}`"
        )
        lines.append("")
        for space_id, payload in rec["inference"].items():
            lines.append(f"### {space_id}")
            if payload.get("skipped"):
                lines.append(f"- skipped: `{payload.get('reason')}`")
                lines.append("")
                continue
            strongest = payload.get("strongest_dimensions", [])
            if strongest:
                lines.append("- strongest_dimensions:")
                for item in strongest:
                    lines.append(
                        f"  - `{item['dimension']}` value=`{item['value']:.4f}` "
                        f"candidate_role=`{item.get('candidate_role')}` "
                        f"top_probes=`{item.get('top_probe_1')}`, `{item.get('top_probe_2')}`, `{item.get('top_probe_3')}`"
                    )
            neighbors = payload.get("nearest_representatives", [])
            if neighbors:
                lines.append("- nearest_representatives:")
                for nb in neighbors:
                    lines.append(
                        f"  - `{nb['protocol_name']}/{nb['sample_id']}/{nb['field_id']}` "
                        f"distance=`{nb['distance']:.4f}` selection_types=`{nb.get('selection_types')}`"
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    info(f"Loading input report: {args.input_report}", args.quiet)
    feature_cols = read_model_feature_cols(args.model_feature_cols)
    field_df = build_field_level_rows(args.input_report)
    probe_df = build_probe_table(field_df)

    missing = [col for col in feature_cols if col not in probe_df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns from inference table: {missing}")

    spaces = load_space_artifacts(args)
    space_results = infer_spaces_for_fields(probe_df, feature_cols, spaces, top_k=args.neighbors_k)

    records = []
    for (_, row), space_result in zip(probe_df.iterrows(), space_results):
        records.append(field_json_record(row, space_result))

    summary = {
        "input_report": str(args.input_report),
        "field_count": len(records),
        "transparent_field_count": int(pd.to_numeric(probe_df["transparent_flag"], errors="coerce").fillna(0).sum()),
        "nontransparent_field_count": int(pd.to_numeric(probe_df["transparent_flag"], errors="coerce").fillna(0).eq(0).sum()),
        "models_used": ["full_pca_latent8", "full_ae_latent8", "nontransparent_pca_latent8", "nontransparent_ae_latent8"],
        "reference_library": str(args.representative_fields),
        "fields": records,
    }

    protocol_name, sample_id = parse_context_from_report_path(args.input_report)
    out_dir = args.output_dir / protocol_name / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, out_dir / "field_inference_results.json")
    write_text(markdown_report(records, args.input_report), out_dir / "field_inference_report.md")
    info(f"Wrote inference outputs to {out_dir}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
