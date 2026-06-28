#!/usr/bin/env python3
"""Build field-level semantic profiles from Stage 3 AE embeddings.

The script maps each field's latent coordinates onto the named Stage 4 z-axis
semantics. It does not call an LLM; field semantics are represented as the
combination of high/low activations on the already named latent axes.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDINGS = Path("/root/semvec/difftrace/stage3/out/stage3_ae/ae_latent8/ae_embeddings.csv")
DEFAULT_AXIS_SEMANTICS = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming/z_axis_semantics.json")
DEFAULT_OUT_DIR = Path("/root/semvec/difftrace/stage4/out/stage4_field_profiles")
Z_AXES = [f"z{i}" for i in range(1, 9)]
CONFIDENCE_WEIGHTS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build field-level Stage 4 semantic profiles.")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--threshold-embeddings",
        type=Path,
        default=None,
        help="Optional train-only embeddings used to calibrate latent percentile thresholds.",
    )
    parser.add_argument("--axis-semantics", type=Path, default=DEFAULT_AXIS_SEMANTICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--high-quantile", type=float, default=0.8)
    parser.add_argument("--low-quantile", type=float, default=0.2)
    return parser.parse_args()


def validate_quantiles(low: float, high: float) -> None:
    if not 0.0 <= low <= 1.0:
        raise SystemExit(f"--low-quantile must be in [0, 1], got {low}")
    if not 0.0 <= high <= 1.0:
        raise SystemExit(f"--high-quantile must be in [0, 1], got {high}")
    if low >= high:
        raise SystemExit(f"--low-quantile must be smaller than --high-quantile, got {low} >= {high}")


def load_axis_semantics(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    axes = data.get("axes")
    if not isinstance(axes, list):
        raise SystemExit(f"{path} must contain an 'axes' list")
    by_axis: dict[str, dict[str, Any]] = {}
    for item in axes:
        if not isinstance(item, dict):
            continue
        axis = item.get("axis")
        if axis in Z_AXES:
            by_axis[axis] = item
    missing = [axis for axis in Z_AXES if axis not in by_axis]
    if missing:
        raise SystemExit(f"{path} missing axis semantics for: {', '.join(missing)}")
    return by_axis


def load_embeddings(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"protocol_name", "sample_id", "field_id", *Z_AXES}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {', '.join(sorted(missing))}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {
                "protocol_name": raw["protocol_name"],
                "sample_id": raw["sample_id"],
                "field_id": raw["field_id"],
            }
            for axis in Z_AXES:
                row[axis] = float(raw[axis])
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} contains no field rows")
    return rows


def percentile_ranks(values: list[float]) -> list[float]:
    """Return percentile ranks in [0, 1], using average rank for ties."""
    n = len(values)
    if n == 1:
        return [0.5]
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start + 1
        while end < n and indexed[end][0] == indexed[start][0]:
            end += 1
        average_position = (start + end - 1) / 2.0
        percentile = average_position / (n - 1)
        for _, original_index in indexed[start:end]:
            ranks[original_index] = percentile
        start = end
    return ranks


def percentile_ranks_against(values: list[float], reference_values: list[float]) -> list[float]:
    """Return percentile ranks calibrated against a fixed reference distribution."""
    if not reference_values:
        raise ValueError("reference_values must not be empty")
    if len(reference_values) == 1:
        return [0.5 for _ in values]
    ordered = sorted(reference_values)
    denominator = len(ordered) - 1
    ranks: list[float] = []
    for value in values:
        left = bisect.bisect_left(ordered, value)
        right = bisect.bisect_right(ordered, value)
        average_position = (left + right - 1) / 2.0 if left != right else float(left)
        ranks.append(min(1.0, max(0.0, average_position / denominator)))
    return ranks


def field_uid(row: dict[str, Any]) -> str:
    return f"{row['protocol_name']}-{row['sample_id']}-{row['field_id']}"


def activation_for(percentile: float, low_quantile: float, high_quantile: float) -> str:
    if percentile >= high_quantile:
        return "high"
    if percentile <= low_quantile:
        return "low"
    return "neutral"


def side_semantics(axis_semantics: dict[str, Any], activation: str) -> dict[str, Any]:
    if activation == "high":
        side = axis_semantics["high_value"]
    elif activation == "low":
        side = axis_semantics["low_value"]
    else:
        raise ValueError(f"neutral activation has no side semantics")
    if not isinstance(side, dict):
        raise ValueError(f"axis side semantics must be an object: {axis_semantics.get('axis')} {activation}")
    return side


def has_semantic_evidence(side: dict[str, Any]) -> bool:
    latent_name = str(side.get("latent_name", "")).strip().lower()
    return bool(latent_name) and latent_name != "insufficient evidence"


def confidence_weight(confidence: Any) -> float:
    return CONFIDENCE_WEIGHTS.get(str(confidence).strip().lower(), 0.4)


def build_profiles(
    rows: list[dict[str, Any]],
    axis_semantics: dict[str, dict[str, Any]],
    low_quantile: float,
    high_quantile: float,
    threshold_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    percentiles_by_axis: dict[str, list[float]] = {}
    for axis in Z_AXES:
        values = [row[axis] for row in rows]
        if threshold_rows is None:
            percentiles_by_axis[axis] = percentile_ranks(values)
        else:
            percentiles_by_axis[axis] = percentile_ranks_against(
                values,
                [row[axis] for row in threshold_rows],
            )

    vector_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        latent_values = {axis: row[axis] for axis in Z_AXES}
        latent_percentiles = {axis: percentiles_by_axis[axis][index] for axis in Z_AXES}
        latent_activations = {
            axis: activation_for(latent_percentiles[axis], low_quantile, high_quantile)
            for axis in Z_AXES
        }
        active_axis_explanations: list[dict[str, Any]] = []
        axis_semantic_scores: dict[str, float] = {}
        for axis in Z_AXES:
            activation = latent_activations[axis]
            if activation == "neutral":
                continue
            side = side_semantics(axis_semantics[axis], activation)
            if not has_semantic_evidence(side):
                continue
            activation_score = abs(latent_percentiles[axis] - 0.5) * confidence_weight(side.get("confidence"))
            axis_key = f"{axis}:{activation}"
            axis_semantic_scores[axis_key] = activation_score
            active_axis_explanations.append(
                {
                    "axis": axis,
                    "side": activation,
                    "value": latent_values[axis],
                    "percentile": latent_percentiles[axis],
                    "latent_name": side.get("latent_name", ""),
                    "definition": side.get("definition", ""),
                    "confidence": side.get("confidence", ""),
                    "axis_score": activation_score,
                }
            )
        active_axis_explanations.sort(key=lambda item: abs(item["percentile"] - 0.5), reverse=True)
        sorted_axis_semantic_scores = dict(
            sorted(axis_semantic_scores.items(), key=lambda item: (-item[1], item[0]))
        )
        semantic_summary = "; ".join(
            f"{item['axis']} {item['side']}: {item['latent_name']}"
            for item in active_axis_explanations
        )
        dominant_axes = ",".join(sorted_axis_semantic_scores)
        dominant_axis_summary = "; ".join(
            f"{item['axis']} {item['side']} (score={item['axis_score']:.6f}): {item['latent_name']}"
            for item in active_axis_explanations
        )

        vector_row: dict[str, Any] = {
            "protocol_name": row["protocol_name"],
            "sample_id": row["sample_id"],
            "field_id": row["field_id"],
            "field_uid": field_uid(row),
        }
        for axis in Z_AXES:
            vector_row[axis] = latent_values[axis]
        for axis in Z_AXES:
            vector_row[f"{axis}_percentile"] = latent_percentiles[axis]
        for axis in Z_AXES:
            vector_row[f"{axis}_activation"] = latent_activations[axis]
        vector_row["active_axis_count"] = len(active_axis_explanations)
        vector_row["active_axes"] = ",".join(f"{item['axis']}:{item['side']}" for item in active_axis_explanations)
        vector_row["semantic_summary"] = semantic_summary
        vector_row["dominant_axes"] = dominant_axes
        vector_row["dominant_axis_summary"] = dominant_axis_summary
        vector_row["axis_semantic_scores"] = json.dumps(sorted_axis_semantic_scores, ensure_ascii=False, sort_keys=True)
        vector_rows.append(vector_row)

        profiles.append(
            {
                "protocol_name": row["protocol_name"],
                "sample_id": row["sample_id"],
                "field_id": row["field_id"],
                "field_uid": field_uid(row),
                "latent_values": latent_values,
                "latent_percentiles": latent_percentiles,
                "latent_activations": latent_activations,
                "active_axis_explanations": active_axis_explanations,
                "semantic_summary": semantic_summary,
                "dominant_axes": dominant_axes,
                "dominant_axis_summary": dominant_axis_summary,
                "axis_semantic_scores": sorted_axis_semantic_scores,
            }
        )
    return vector_rows, profiles


def write_vectors(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "protocol_name",
        "sample_id",
        "field_id",
        "field_uid",
        *Z_AXES,
        *(f"{axis}_percentile" for axis in Z_AXES),
        *(f"{axis}_activation" for axis in Z_AXES),
        "active_axis_count",
        "active_axes",
        "semantic_summary",
        "dominant_axes",
        "dominant_axis_summary",
        "axis_semantic_scores",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_profiles(path: Path, profiles: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    validate_quantiles(args.low_quantile, args.high_quantile)
    axis_semantics = load_axis_semantics(args.axis_semantics)
    rows = load_embeddings(args.embeddings)
    threshold_rows = load_embeddings(args.threshold_embeddings) if args.threshold_embeddings else None
    vector_rows, profiles = build_profiles(
        rows,
        axis_semantics,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        threshold_rows=threshold_rows,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.out_dir / "field_semantic_vectors.csv"
    profiles_path = args.out_dir / "field_semantic_profiles.jsonl"
    write_vectors(vectors_path, vector_rows)
    write_profiles(profiles_path, profiles)
    print(f"[stage4-field] fields: {len(rows)}")
    print(
        "[stage4-field] percentile ranking: "
        + ("calibrated against train-only embeddings" if threshold_rows is not None else "per-axis")
    )
    print(f"[stage4-field] low activation threshold: percentile <= {args.low_quantile}")
    print(f"[stage4-field] high activation threshold: percentile >= {args.high_quantile}")
    print(f"[stage4-field] wrote: {vectors_path}")
    print(f"[stage4-field] wrote: {profiles_path}")


if __name__ == "__main__":
    main()
