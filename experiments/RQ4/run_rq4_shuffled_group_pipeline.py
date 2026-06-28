#!/usr/bin/env python3
"""Run the downstream semantic pipeline for one RQ4 shuffled-group seed.

This script assumes build_rq4_shuffled_group_dataset.py has already produced:

    /root/semvec/RQ4/out/shuffled_seed_<seed>/stage3_dataset_semantic_fields.csv

For that seed it runs the same downstream steps as the full method, using
seed-local output directories so the normal Stage 3/4 artifacts are not
overwritten.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SEMVEC_ROOT = Path("/root/semvec")
RQ4_ROOT = SEMVEC_ROOT / "RQ4"
STAGE3_DIR = SEMVEC_ROOT / "difftrace" / "stage3"
STAGE4_DIR = SEMVEC_ROOT / "difftrace" / "stage4"
DEFAULT_OUT_ROOT = RQ4_ROOT / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ4-A shuffled-group downstream pipeline for one seed.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backend", choices=["api", "codex"], default="api")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run through field-profile construction but skip latent naming and field fusion LLM calls.",
    )
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[rq4-shuffled-pipeline] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    seed_dir = args.out_root / f"shuffled_seed_{args.seed}"
    dataset_csv = seed_dir / "stage3_dataset_semantic_fields.csv"
    if not dataset_csv.exists():
        raise SystemExit(
            f"missing shuffled dataset: {dataset_csv}\n"
            "run build_rq4_shuffled_group_dataset.py first"
        )

    matrix_dir = seed_dir / "stage3_training_matrix"
    matrix_csv = matrix_dir / "stage3_training_matrix.csv"
    ae_dir = seed_dir / "stage3_ae"
    ae_embeddings = ae_dir / f"ae_latent{args.latent_dim}" / "ae_embeddings.csv"
    latent_naming_dir = seed_dir / "stage4_latent_naming"
    evidence_json = latent_naming_dir / "z_topk_probe_evidence.json"
    axis_semantics_json = latent_naming_dir / "z_axis_semantics.json"
    field_profiles_dir = seed_dir / "stage4_field_profiles"
    field_profiles_jsonl = field_profiles_dir / "field_semantic_profiles.jsonl"
    fusion_dir = seed_dir / "stage4_field_semantic_fusion"

    py = sys.executable
    commands: list[list[str]] = [
        [
            py,
            str(STAGE3_DIR / "build_stage3_training_matrix.py"),
            "--input-csv",
            str(dataset_csv),
            "--output-dir",
            str(matrix_dir),
        ],
        [
            py,
            str(STAGE3_DIR / "train_stage3_autoencoder.py"),
            "--input-csv",
            str(matrix_csv),
            "--projection-csv",
            str(matrix_csv),
            "--output-dir",
            str(ae_dir),
            "--latent-dims",
            str(args.latent_dim),
            "--epochs",
            str(args.epochs),
        ],
        [
            py,
            str(STAGE4_DIR / "build_stage4_latent_names.py"),
            "--embeddings",
            str(ae_embeddings),
            "--training-matrix",
            str(matrix_csv),
            "--out-dir",
            str(latent_naming_dir),
        ],
    ]

    for command in commands:
        run(command, args.dry_run)

    if not args.skip_llm:
        run(
            [
                py,
                str(STAGE4_DIR / "run_stage4_llm_naming.py"),
                "--evidence",
                str(evidence_json),
                "--out",
                str(latent_naming_dir / "z_llm_prompt_responses.md"),
                "--semantics-out",
                str(axis_semantics_json),
                "--mode",
                "naming",
            ],
            args.dry_run,
        )
    elif not axis_semantics_json.exists() and not args.dry_run:
        print(
            "[rq4-shuffled-pipeline] --skip-llm enabled; stop before field profiles because "
            f"{axis_semantics_json} does not exist",
            flush=True,
        )
        return 0

    run(
        [
            py,
            str(STAGE4_DIR / "build_stage4_field_profiles.py"),
            "--embeddings",
            str(ae_embeddings),
            "--axis-semantics",
            str(axis_semantics_json),
            "--out-dir",
            str(field_profiles_dir),
        ],
        args.dry_run,
    )

    if not args.skip_llm:
        run(
            [
                py,
                str(STAGE4_DIR / "run_stage4_field_semantic_fusion.py"),
                "--input",
                str(field_profiles_jsonl),
                "--output-jsonl",
                str(fusion_dir / "field_semantic_fused_profiles.jsonl"),
                "--output-csv",
                str(fusion_dir / "field_semantic_fused_vectors.csv"),
                "--responses-md",
                str(fusion_dir / "field_semantic_fusion_prompt_responses.md"),
                "--run-log",
                str(fusion_dir / "field_semantic_fusion_run.log"),
                "--backend",
                args.backend,
                "--workers",
                str(args.workers),
            ],
            args.dry_run,
        )

    print(f"[rq4-shuffled-pipeline] done seed={args.seed} out={seed_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
