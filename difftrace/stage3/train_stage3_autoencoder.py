#!/usr/bin/env python3
"""Train Stage 3 autoencoders on the scaled training matrix."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from build_stage3_training_matrix import KEY_COLUMNS
from check_stage3_dataset import MODEL_FEATURE_COLUMNS


DEFAULT_INPUT_CSV = Path("/root/semvec/difftrace/stage3/out/stage3_training_matrix/stage3_training_matrix.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/stage3/out/stage3_ae")


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_hidden_dims(value: str) -> List[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("Hidden dims must contain at least one integer.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 3 autoencoders on the scaled training matrix.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input scaled training matrix CSV.")
    parser.add_argument(
        "--projection-csv",
        type=Path,
        default=None,
        help="Optional all-fields matrix to project with the train-fitted encoder. Defaults to --input-csv.",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Optional eval-only matrix. When provided, write eval_ae_embeddings.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--latent-dims", type=str, default="4,8,12", help="Comma-separated latent dimensions to train.")
    parser.add_argument("--hidden-dims", type=str, default="16,12", help="Comma-separated hidden dimensions.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs per latent dimension.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Training device: auto, cpu, cuda, cuda:0 ...")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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


class FieldMatrixDataset(Dataset[torch.Tensor]):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.data = torch.tensor(
            [[float(row[col]) for col in MODEL_FEATURE_COLUMNS] for row in rows],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        last_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(last_dim, dim), nn.ReLU()])
            last_dim = dim
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        last_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(last_dim, dim), nn.ReLU()])
            last_dim = dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


@dataclass
class TrainArtifacts:
    model: AutoEncoder
    history: List[Dict[str, float]]
    best_epoch: int
    best_val_loss: float


def split_dataset(dataset: FieldMatrixDataset, validation_ratio: float, seed: int) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor]]:
    if validation_ratio <= 0.0:
        return dataset, torch.utils.data.Subset(dataset, [])
    total = len(dataset)
    val_size = max(1, int(total * validation_ratio))
    if val_size >= total:
        val_size = max(1, total - 1)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(model: AutoEncoder, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = criterion(output, batch)
        if is_train:
            loss.backward()
            optimizer.step()
        batch_size = batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def train_autoencoder(dataset: FieldMatrixDataset, latent_dim: int, hidden_dims: Sequence[int], epochs: int, batch_size: int, learning_rate: float, weight_decay: float, validation_ratio: float, seed: int, device: torch.device, quiet: bool) -> TrainArtifacts:
    train_set, val_set = split_dataset(dataset, validation_ratio=validation_ratio, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if len(val_set) > 0 else None

    model = AutoEncoder(input_dim=len(MODEL_FEATURE_COLUMNS), hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, criterion, None, device)
        else:
            val_loss = train_loss
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if not quiet and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            info(f"latent={latent_dim} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", quiet)

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainArtifacts(model=model, history=history, best_epoch=best_epoch, best_val_loss=best_val_loss)


def encode_rows(model: AutoEncoder, dataset: FieldMatrixDataset, device: torch.device) -> tuple[List[List[float]], List[float]]:
    model.eval()
    embeddings: List[List[float]] = []
    reconstruction_errors: List[float] = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=512, shuffle=False):
            batch = batch.to(device)
            latent = model.encode(batch)
            recon = model.decode(latent)
            mse = ((recon - batch) ** 2).mean(dim=1)
            embeddings.extend(latent.cpu().tolist())
            reconstruction_errors.extend(mse.cpu().tolist())
    return embeddings, reconstruction_errors


def write_embeddings(rows: Sequence[Dict[str, Any]], embeddings: Sequence[Sequence[float]], path: Path) -> None:
    fieldnames = KEY_COLUMNS + [f"z{i+1}" for i in range(len(embeddings[0]))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, emb in zip(rows, embeddings):
            out = {col: row.get(col) for col in KEY_COLUMNS}
            for idx, value in enumerate(emb, start=1):
                out[f"z{idx}"] = float(value)
            writer.writerow(out)


def write_reconstruction_errors(rows: Sequence[Dict[str, Any]], reconstruction_errors: Sequence[float], path: Path) -> None:
    fieldnames = KEY_COLUMNS + ["reconstruction_mse"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, error in zip(rows, reconstruction_errors):
            out = {col: row.get(col) for col in KEY_COLUMNS}
            out["reconstruction_mse"] = float(error)
            writer.writerow(out)


def write_history(history: Sequence[Dict[str, float]], path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def build_summary(rows: Sequence[Dict[str, Any]], latent_dim: int, hidden_dims: Sequence[int], args: argparse.Namespace, device: torch.device, artifacts: TrainArtifacts, reconstruction_errors: Sequence[float]) -> Dict[str, Any]:
    protocol_counts: Dict[str, int] = {}
    for row in rows:
        protocol = str(row["protocol_name"])
        protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
    sorted_errors = sorted(float(x) for x in reconstruction_errors)
    median_error = sorted_errors[len(sorted_errors) // 2] if sorted_errors else 0.0
    return {
        "row_count": len(rows),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "latent_dim": latent_dim,
        "hidden_dims": list(hidden_dims),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": artifacts.best_epoch,
        "best_val_loss": artifacts.best_val_loss,
        "final_train_loss": artifacts.history[-1]["train_loss"] if artifacts.history else None,
        "final_val_loss": artifacts.history[-1]["val_loss"] if artifacts.history else None,
        "reconstruction_error": {
            "min": min(sorted_errors) if sorted_errors else 0.0,
            "median": median_error,
            "max": max(sorted_errors) if sorted_errors else 0.0,
            "mean": (sum(sorted_errors) / len(sorted_errors)) if sorted_errors else 0.0,
        },
        "protocol_counts": dict(sorted(protocol_counts.items())),
    }


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        warn(f"Input CSV does not exist: {args.input_csv}")
        return 1
    if args.projection_csv is not None and not args.projection_csv.exists():
        warn(f"Projection CSV does not exist: {args.projection_csv}")
        return 1
    if args.eval_csv is not None and not args.eval_csv.exists():
        warn(f"Eval CSV does not exist: {args.eval_csv}")
        return 1

    try:
        latent_dims = parse_int_list(args.latent_dims)
        hidden_dims = parse_hidden_dims(args.hidden_dims)
    except ValueError as exc:
        warn(str(exc))
        return 1

    if not latent_dims:
        warn("No latent dimensions provided.")
        return 1

    set_seed(args.seed)
    device = choose_device(args.device)
    rows = load_rows(args.input_csv)
    if not rows:
        warn(f"No rows found in input CSV: {args.input_csv}")
        return 1
    dataset = FieldMatrixDataset(rows)
    projection_rows = load_rows(args.projection_csv) if args.projection_csv is not None else rows
    if not projection_rows:
        warn(f"No rows found in projection CSV: {args.projection_csv}")
        return 1
    projection_dataset = FieldMatrixDataset(projection_rows)
    eval_rows = load_rows(args.eval_csv) if args.eval_csv is not None else []
    eval_dataset = FieldMatrixDataset(eval_rows) if eval_rows else None

    ensure_dir(args.output_dir)
    for latent_dim in latent_dims:
        run_dir = args.output_dir / f"ae_latent{latent_dim}"
        ensure_dir(run_dir)
        artifacts = train_autoencoder(
            dataset=dataset,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            device=device,
            quiet=args.quiet,
        )
        train_embeddings, train_reconstruction_errors = encode_rows(artifacts.model, dataset, device)
        embeddings, reconstruction_errors = encode_rows(artifacts.model, projection_dataset, device)
        write_embeddings(projection_rows, embeddings, run_dir / "ae_embeddings.csv")
        write_embeddings(rows, train_embeddings, run_dir / "train_ae_embeddings.csv")
        write_reconstruction_errors(projection_rows, reconstruction_errors, run_dir / "ae_reconstruction_errors.csv")
        write_reconstruction_errors(rows, train_reconstruction_errors, run_dir / "train_ae_reconstruction_errors.csv")
        if eval_dataset is not None:
            eval_embeddings, eval_reconstruction_errors = encode_rows(artifacts.model, eval_dataset, device)
            write_embeddings(eval_rows, eval_embeddings, run_dir / "eval_ae_embeddings.csv")
            write_reconstruction_errors(eval_rows, eval_reconstruction_errors, run_dir / "eval_ae_reconstruction_errors.csv")
        write_history(artifacts.history, run_dir / "ae_training_history.csv")
        torch.save(
            {
                "model_state_dict": artifacts.model.state_dict(),
                "latent_dim": latent_dim,
                "hidden_dims": list(hidden_dims),
                "model_feature_columns": MODEL_FEATURE_COLUMNS,
                "key_columns": KEY_COLUMNS,
            },
            run_dir / "ae_model.pt",
        )
        summary = build_summary(rows, latent_dim, hidden_dims, args, device, artifacts, train_reconstruction_errors)
        summary["projection_csv"] = str(args.projection_csv) if args.projection_csv is not None else str(args.input_csv)
        summary["projection_row_count"] = len(projection_rows)
        summary["eval_csv"] = str(args.eval_csv) if args.eval_csv is not None else None
        summary["eval_row_count"] = len(eval_rows)
        (run_dir / "ae_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        info(f"Completed AE training for latent={latent_dim}; outputs in {run_dir}", args.quiet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
