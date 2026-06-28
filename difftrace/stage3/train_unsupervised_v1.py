#!/usr/bin/env python3
"""Train first-pass unsupervised models on field-level behavior features."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/out/field_training_samples.csv")
DEFAULT_MODEL_FEATURE_COLS = Path("/root/semvec/difftrace/out/dataset_health_report/model_feature_cols.txt")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/unsupervised_v1")
DEFAULT_BASIC_VIS_SUMMARY = Path("/root/semvec/difftrace/out/basic_visual_analysis/basic_visual_analysis_summary.json")
DEFAULT_SUBSET_SUMMARY = Path("/root/semvec/difftrace/out/basic_visual_analysis/subset_summary.csv")

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
EMBEDDING_METADATA_COLS = [
    "protocol_name",
    "sample_id",
    "field_id",
    "transparent_flag",
    "boundary_miss",
    "mutation_count",
    "unique_metric_vectors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA and AutoEncoder baselines for field-level features.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to field_training_samples.csv")
    parser.add_argument(
        "--model-feature-cols",
        type=Path,
        default=DEFAULT_MODEL_FEATURE_COLS,
        help="Path to model_feature_cols.txt or model_feature_cols.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for full/nontransparent experiment results",
    )
    parser.add_argument(
        "--dataset-version",
        choices=["full", "nontransparent", "both"],
        default="both",
        help="Which dataset version to train on",
    )
    parser.add_argument(
        "--latent-dims",
        type=str,
        default="2,3,4,8",
        help="Comma-separated AE latent dimensions",
    )
    parser.add_argument("--pca-components", type=int, default=8, help="Number of PCA components to fit")
    parser.add_argument("--epochs", type=int, default=200, help="AE training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="AE batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AE learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AE optimizer weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--random-seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--basic-visual-summary",
        type=Path,
        default=DEFAULT_BASIC_VIS_SUMMARY,
        help="Optional basic_visual_analysis_summary.json path for provenance",
    )
    parser.add_argument(
        "--subset-summary",
        type=Path,
        default=DEFAULT_SUBSET_SUMMARY,
        help="Optional subset_summary.csv path for provenance",
    )
    parser.add_argument(
        "--recompute-transparent-flag",
        action="store_true",
        help="Ignore any existing transparent_flag column and recompute it from rules",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging")
    return parser.parse_args()


def info(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def ensure_dependencies() -> Dict[str, Any]:
    missing = []
    imported: Dict[str, Any] = {}
    for name in ["matplotlib", "matplotlib.pyplot", "sklearn", "torch"]:
        try:
            module = __import__(name, fromlist=["*"])
            imported[name] = module
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required Python packages for training: "
            f"{missing}. Please install scikit-learn, matplotlib, and torch before running this script."
        )
    return imported


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_feature_cols(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model feature list: {path}")
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


def maybe_load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    return pd.read_csv(path)


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


def ensure_transparent_flag(df: pd.DataFrame, recompute: bool) -> pd.DataFrame:
    out = df.copy()
    if recompute or "transparent_flag" not in out.columns:
        out["transparent_flag"] = compute_transparent_flag(out)
    else:
        out["transparent_flag"] = pd.to_numeric(out["transparent_flag"], errors="coerce").fillna(0).astype(int)
    return out


def choose_dataset_versions(requested: str) -> List[str]:
    if requested == "both":
        return ["full", "nontransparent"]
    return [requested]


def apply_dataset_version_filter(df: pd.DataFrame, dataset_version: str) -> pd.DataFrame:
    if dataset_version == "full":
        return df.copy()
    if dataset_version == "nontransparent":
        return df.loc[pd.to_numeric(df["transparent_flag"], errors="coerce").fillna(0).eq(0)].copy()
    raise ValueError(f"Unsupported dataset version: {dataset_version}")


def validate_feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input CSV: {missing}")
    X = df[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    if X.empty:
        raise ValueError("No feature rows available after selecting model feature columns")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Input feature matrix contains inf/-inf values")
    if X.isna().all(axis=1).any():
        bad = X.index[X.isna().all(axis=1)].tolist()[:10]
        raise ValueError(f"Some rows are entirely NaN after numeric conversion, sample indices: {bad}")
    return X


def create_val_split(n_samples: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(n_samples * val_ratio))) if n_samples >= 5 else max(1, n_samples // 5)
    val_size = min(val_size, n_samples - 1) if n_samples > 1 else 0
    if val_size == 0:
        return indices, np.array([], dtype=int)
    return indices[val_size:], indices[:val_size]


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def reconstruction_error_stats(errors: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


def build_embedding_df(
    base_df: pd.DataFrame,
    embedding: np.ndarray,
    dim_prefix: str,
    metadata_cols: Sequence[str],
) -> pd.DataFrame:
    keep_cols = [col for col in metadata_cols if col in base_df.columns]
    out = base_df[keep_cols].copy()
    for idx in range(embedding.shape[1]):
        out[f"{dim_prefix}{idx + 1}"] = embedding[:, idx]
    return out


def plot_loss_curve(history_df: pd.DataFrame, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    ax.set_title("AutoEncoder Training Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_embedding_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    path: Path,
    plt: Any,
    categorical: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        categories = sorted(df[color_col].astype(str).fillna("NA").unique().tolist())
        cmap = plt.matplotlib.colormaps.get_cmap("tab10")
        for idx, category in enumerate(categories):
            sub = df.loc[df[color_col].astype(str) == category]
            color = cmap(idx % cmap.N)
            ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.75, label=category, color=color)
        ax.legend(loc="best", fontsize=8, ncol=2)
    else:
        vals = pd.to_numeric(df[color_col], errors="coerce")
        sc = ax.scatter(df[x_col], df[y_col], c=vals, cmap="viridis", s=18, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{path.stem}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reconstruction_errors(
    df: pd.DataFrame,
    errors: np.ndarray,
    path: Path,
    model_type: str,
    latent_dim: int,
) -> pd.DataFrame:
    keep_cols = [col for col in EMBEDDING_METADATA_COLS if col in df.columns]
    out = df[keep_cols].copy()
    out["model_type"] = model_type
    out["latent_dim"] = latent_dim
    out["reconstruction_error"] = errors
    out.to_csv(path, index=False)
    return out


@dataclass
class PCAResult:
    embedding_df: pd.DataFrame
    reconstruction_errors_df: pd.DataFrame
    explained_variance_ratio: List[float]
    training_summary: Dict[str, Any]


def train_pca_baseline(
    dataset_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_dir: Path,
    dataset_version: str,
    n_components: int,
    imported: Dict[str, Any],
    quiet: bool,
) -> PCAResult:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    plt = imported["matplotlib.pyplot"]
    X_df = validate_feature_matrix(dataset_df, feature_cols)
    X = X_df.to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]), random_state=0)
    Z = pca.fit_transform(X_scaled)
    X_recon = pca.inverse_transform(Z)
    recon_errors = np.mean((X_scaled - X_recon) ** 2, axis=1)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(scaler, model_dir / "scaler.pkl")
    save_pickle(pca, model_dir / "pca_model.pkl")

    explained_df = pd.DataFrame(
        {
            "principal_component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_,
        }
    )
    explained_df.to_csv(model_dir / "explained_variance.csv", index=False)

    embedding_df = build_embedding_df(dataset_df, Z, "pc", EMBEDDING_METADATA_COLS)
    embedding_df.to_csv(model_dir / "pca_embedding.csv", index=False)
    errors_df = save_reconstruction_errors(dataset_df, recon_errors, model_dir / "reconstruction_errors.csv", "pca", pca.n_components_)

    plot_embedding_scatter(embedding_df, "pc1", "pc2", "protocol_name", model_dir / "pca_embedding_by_protocol.png", plt, categorical=True)
    if "boundary_miss" in embedding_df.columns:
        plot_embedding_scatter(embedding_df, "pc1", "pc2", "boundary_miss", model_dir / "pca_embedding_by_boundary_miss.png", plt, categorical=True)
    if "transparent_flag" in embedding_df.columns:
        plot_embedding_scatter(embedding_df, "pc1", "pc2", "transparent_flag", model_dir / "pca_embedding_by_transparent_flag.png", plt, categorical=True)

    summary = {
        "dataset_version": dataset_version,
        "model_type": "pca",
        "sample_count": int(len(dataset_df)),
        "input_dim": int(len(feature_cols)),
        "latent_dim": int(pca.n_components_),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        "reconstruction_error_stats": reconstruction_error_stats(recon_errors),
        "device": "cpu",
        "hyperparameters": {
            "n_components": int(pca.n_components_),
        },
    }
    write_json(summary, model_dir / "training_summary.json")
    info(f"PCA done for {dataset_version}: n_components={pca.n_components_}", quiet)
    return PCAResult(
        embedding_df=embedding_df,
        reconstruction_errors_df=errors_df,
        explained_variance_ratio=summary["pca_explained_variance"],
        training_summary=summary,
    )


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


def train_autoencoder(
    dataset_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_dir: Path,
    dataset_version: str,
    latent_dim: int,
    args: argparse.Namespace,
    imported: Dict[str, Any],
    quiet: bool,
) -> Dict[str, Any]:
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    plt = imported["matplotlib.pyplot"]

    X_df = validate_feature_matrix(dataset_df, feature_cols)
    X = X_df.to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    train_idx, val_idx = create_val_split(len(X_scaled), args.val_ratio, args.random_seed + latent_dim)
    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx] if len(val_idx) else X_scaled[train_idx[: max(1, len(train_idx) // 5)]]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=min(args.batch_size, len(X_train)),
        shuffle=True,
    )
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"AE device for {dataset_version}/latent{latent_dim}: {device}", quiet)

    model = define_autoencoder(torch, input_dim=len(feature_cols), latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_batch = val_tensor.to(device)
            val_recon, _ = model(val_batch)
            val_loss = float(criterion(val_recon, val_batch).detach().cpu().item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if not quiet:
            print(
                f"[AE][{dataset_version}][latent={latent_dim}] epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                info(f"Early stopping at epoch {epoch} for latent={latent_dim}", quiet)
                break

    if best_state is None:
        raise RuntimeError("AE training failed to produce a checkpoint")

    model.load_state_dict(best_state)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "ae_best.pt")
    save_pickle(scaler, model_dir / "scaler.pkl")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(model_dir / "training_history.csv", index=False)
    plot_loss_curve(history_df, model_dir / "loss_curve.png", plt)

    model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        recon_all, latent_all = model(all_tensor)
    latent_np = latent_all.detach().cpu().numpy()
    recon_np = recon_all.detach().cpu().numpy()
    recon_errors = np.mean((X_scaled - recon_np) ** 2, axis=1)

    embedding_df = build_embedding_df(dataset_df, latent_np, "z", EMBEDDING_METADATA_COLS)
    embedding_df.to_csv(model_dir / f"ae_embedding_latent{latent_dim}.csv", index=False)
    errors_df = save_reconstruction_errors(
        dataset_df,
        recon_errors,
        model_dir / "reconstruction_errors.csv",
        "ae",
        latent_dim,
    )

    if latent_np.shape[1] >= 2:
        plot_embedding_scatter(embedding_df, "z1", "z2", "protocol_name", model_dir / f"ae_latent{latent_dim}_by_protocol.png", plt, categorical=True)
        if "boundary_miss" in embedding_df.columns:
            plot_embedding_scatter(
                embedding_df,
                "z1",
                "z2",
                "boundary_miss",
                model_dir / f"ae_latent{latent_dim}_by_boundary_miss.png",
                plt,
                categorical=True,
            )
        if "transparent_flag" in embedding_df.columns:
            plot_embedding_scatter(
                embedding_df,
                "z1",
                "z2",
                "transparent_flag",
                model_dir / f"ae_latent{latent_dim}_by_transparent_flag.png",
                plt,
                categorical=True,
            )

    latent_summary = {
        "dataset_version": dataset_version,
        "model_type": "autoencoder",
        "sample_count": int(len(dataset_df)),
        "input_dim": int(len(feature_cols)),
        "latent_dim": int(latent_dim),
        "final_train_loss": float(history_df["train_loss"].iloc[-1]),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "pca_explained_variance": None,
        "reconstruction_error_stats": reconstruction_error_stats(recon_errors),
        "device": str(device),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "val_ratio": float(args.val_ratio),
        },
    }
    write_json(latent_summary, model_dir / "latent_summary.json")
    write_json(latent_summary, model_dir / "training_summary.json")
    info(f"AE done for {dataset_version}: latent_dim={latent_dim}, best_val_loss={best_val_loss:.6f}", quiet)
    return {
        "embedding_df": embedding_df,
        "reconstruction_errors_df": errors_df,
        "history_df": history_df,
        "training_summary": latent_summary,
    }


def run_dataset_version(
    dataset_version: str,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    args: argparse.Namespace,
    imported: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dataset_df = apply_dataset_version_filter(df, dataset_version)
    if dataset_df.empty:
        raise ValueError(f"Dataset version {dataset_version} is empty after filtering")

    dataset_dir = args.output_dir / dataset_version
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_summary = {
        "dataset_version": dataset_version,
        "sample_count": int(len(dataset_df)),
        "transparent_count": int(pd.to_numeric(dataset_df["transparent_flag"], errors="coerce").fillna(0).sum())
        if "transparent_flag" in dataset_df.columns
        else None,
        "boundary_miss_count": int(pd.to_numeric(dataset_df["boundary_miss"], errors="coerce").fillna(0).sum())
        if "boundary_miss" in dataset_df.columns
        else None,
    }
    write_json(metadata_summary, dataset_dir / "dataset_metadata.json")

    comparison_rows: List[Dict[str, Any]] = []
    pca_dir = dataset_dir / "pca"
    pca_result = train_pca_baseline(
        dataset_df=dataset_df,
        feature_cols=feature_cols,
        model_dir=pca_dir,
        dataset_version=dataset_version,
        n_components=args.pca_components,
        imported=imported,
        quiet=args.quiet,
    )
    comparison_rows.append(
        {
            "dataset_version": dataset_version,
            "model_type": "pca",
            "latent_dim": int(len(pca_result.explained_variance_ratio)),
            "best_val_loss": np.nan,
            "reconstruction_error_mean": pca_result.training_summary["reconstruction_error_stats"]["mean"],
            "reconstruction_error_std": pca_result.training_summary["reconstruction_error_stats"]["std"],
            "sample_count": int(len(dataset_df)),
        }
    )

    latent_dims = [int(part.strip()) for part in args.latent_dims.split(",") if part.strip()]
    for latent_dim in latent_dims:
        ae_dir = dataset_dir / f"ae_latent{latent_dim}"
        ae_result = train_autoencoder(
            dataset_df=dataset_df,
            feature_cols=feature_cols,
            model_dir=ae_dir,
            dataset_version=dataset_version,
            latent_dim=latent_dim,
            args=args,
            imported=imported,
            quiet=args.quiet,
        )
        comparison_rows.append(
            {
                "dataset_version": dataset_version,
                "model_type": "autoencoder",
                "latent_dim": int(latent_dim),
                "best_val_loss": ae_result["training_summary"]["best_val_loss"],
                "reconstruction_error_mean": ae_result["training_summary"]["reconstruction_error_stats"]["mean"],
                "reconstruction_error_std": ae_result["training_summary"]["reconstruction_error_stats"]["std"],
                "sample_count": int(len(dataset_df)),
            }
        )

    dataset_summary = {
        "dataset_version": dataset_version,
        "sample_count": int(len(dataset_df)),
        "input_dim": int(len(feature_cols)),
        "feature_cols": list(feature_cols),
        "comparison_rows": comparison_rows,
    }
    write_json(dataset_summary, dataset_dir / "training_summary.json")
    return comparison_rows


def main() -> int:
    args = parse_args()
    imported = ensure_dependencies()
    set_random_seed(args.random_seed)

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {args.input_csv}")

    feature_cols = load_feature_cols(args.model_feature_cols)
    df = ensure_transparent_flag(df, args.recompute_transparent_flag)

    numeric_cols = set(feature_cols) | {"transparent_flag", "boundary_miss", "mutation_count", "unique_metric_vectors"}
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "input_csv": str(args.input_csv),
        "model_feature_cols": str(args.model_feature_cols),
        "basic_visual_summary": maybe_load_json(args.basic_visual_summary),
        "subset_summary": maybe_load_csv(args.subset_summary).to_dict(orient="records")
        if maybe_load_csv(args.subset_summary) is not None
        else None,
    }
    write_json(provenance, args.output_dir / "provenance.json")

    all_rows: List[Dict[str, Any]] = []
    for dataset_version in choose_dataset_versions(args.dataset_version):
        info(f"Running dataset version: {dataset_version}", args.quiet)
        all_rows.extend(run_dataset_version(dataset_version, df, feature_cols, args, imported))

    comparison_df = pd.DataFrame(all_rows)
    comparison_df.to_csv(args.output_dir / "experiment_comparison.csv", index=False)
    write_json(
        {
            "dataset_version": args.dataset_version,
            "latent_dims": [int(part.strip()) for part in args.latent_dims.split(",") if part.strip()],
            "pca_components": args.pca_components,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "random_seed": args.random_seed,
            "rows": all_rows,
        },
        args.output_dir / "run_summary.json",
    )
    info(f"Wrote experiment comparison to {args.output_dir / 'experiment_comparison.csv'}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
