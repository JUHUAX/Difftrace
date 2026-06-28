#!/usr/bin/env python3
"""Build Stage 4A probe-anchored latent evidence.

This script computes correlations between AE latent dimensions and the
28 predefined program-behavior probes, then exports top-k positive and
negative evidence for constrained LLM naming.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EMBEDDINGS = Path("/root/semvec/difftrace/stage3/out/stage3_ae/ae_latent8/ae_embeddings.csv")
DEFAULT_MATRIX = Path("/root/semvec/difftrace/stage3/out/stage3_training_matrix/stage3_training_matrix.csv")
DEFAULT_OUT_DIR = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming")
DEFAULT_CORRELATION_THRESHOLD = 0.30

KEY_COLS = ["protocol_name", "sample_id", "field_id"]
AXIS_RE = re.compile(r"^z\d+$")

PROBE_GROUPS = {
    "relative_start": "context",
    "field_instr_ratio": "context",
    "compare_ratio": "context",
    "constraint_value_diversity": "context",
}

PROBE_TABLE = {
    "relative_start": {
        "probe_meaning": "字段首次被程序消费的位置，表示字段影响解析流程的早晚位置。",
        "probe_high_value_meaning": "字段首次被程序消费得越晚。",
    },
    "field_instr_ratio": {
        "probe_meaning": "字段相关指令在 baseline 中的占比，表示程序对该字段投入的处理强度。",
        "probe_high_value_meaning": "baseline 中字段相关处理指令占比越高，程序对该字段投入的处理越多。",
    },
    "compare_ratio": {
        "probe_meaning": "字段相关比较行为比例，表示字段参与条件判断、约束检查或分支决策的程度。",
        "probe_high_value_meaning": "字段越多参与比较、条件判断、约束检查或分支决策。",
    },
    "constraint_value_diversity": {
        "probe_meaning": "字段约束值多样性，表示程序能区分出的候选值或约束值丰富度。",
        "probe_high_value_meaning": "程序能区分出的候选值、约束值或离散取值越丰富。",
    },
    "neighborhood_mean_baseline_distance": {
        "probe_meaning": "小幅值变化是否改变程序处理路径或处理强度。",
        "probe_high_value_meaning": "字段原值附近的小幅变化相对 baseline 造成的平均程序行为偏离越大。",
    },
    "neighborhood_mean_pairwise_distance": {
        "probe_meaning": "字段局部邻域的小幅扰动是否触发多种处理状态或执行路径。",
        "probe_high_value_meaning": "局部邻域内不同 mutation 之间的平均处理分歧越大，越可能触发多种处理状态。",
    },
    "neighborhood_max_pairwise_distance": {
        "probe_meaning": "局部小扰动中最强的一次处理分歧有多大。",
        "probe_high_value_meaning": "局部小扰动中最强的一对 mutation 处理分歧越大。",
    },
    "neighborhood_metric_vector_variance": {
        "probe_meaning": "程序对局部小扰动的响应是否稳定，还是对少数邻近值特别敏感。",
        "probe_high_value_meaning": "程序对局部邻近值的响应越不稳定，越可能只对少数邻近值高度敏感。",
    },
    "neighborhood_unique_vector_ratio": {
        "probe_meaning": "字段局部邻域是否被程序划分成多个行为等价类。",
        "probe_high_value_meaning": "局部邻域取值被程序划分出的行为等价类越多。",
    },
    "neighborhood_loop_dispersion": {
        "probe_meaning": "小幅值变化是否影响循环、批量处理或重复执行行为。",
        "probe_high_value_meaning": "小幅取值变化导致的循环、批量处理或重复执行差异越大。",
    },
    "boundary_mean_baseline_distance": {
        "probe_meaning": "边界值是否改变程序处理路径或处理强度。",
        "probe_high_value_meaning": "边界值 mutation 相对 baseline 造成的平均程序行为偏离越大。",
    },
    "boundary_mean_pairwise_distance": {
        "probe_meaning": "边界值扰动是否触发多种范围检查、错误处理或边界分支路径。",
        "probe_high_value_meaning": "不同边界值之间的平均处理分歧越大，越可能触发多种范围检查、错误处理或边界分支。",
    },
    "boundary_max_pairwise_distance": {
        "probe_meaning": "边界测试中最强的处理路径分裂有多大。",
        "probe_high_value_meaning": "边界测试中最强的一对 mutation 处理路径分裂越大。",
    },
    "boundary_metric_vector_variance": {
        "probe_meaning": "程序是否只对某些边界值高度敏感。",
        "probe_high_value_meaning": "程序对边界值响应越不稳定，越可能只对某些边界点高度敏感。",
    },
    "boundary_unique_vector_ratio": {
        "probe_meaning": "边界附近是否存在多个可区分的程序处理类别。",
        "probe_high_value_meaning": "边界附近取值被程序划分出的可区分处理类别越多。",
    },
    "boundary_loop_dispersion": {
        "probe_meaning": "边界值是否改变循环次数、批量处理规模或解析长度。",
        "probe_high_value_meaning": "边界值导致的循环次数、批量处理规模或解析长度差异越大。",
    },
    "enum_mean_baseline_distance": {
        "probe_meaning": "候选枚举值是否改变程序处理路径或处理动作。",
        "probe_high_value_meaning": "候选枚举值 mutation 相对 baseline 造成的平均程序行为偏离越大。",
    },
    "enum_mean_pairwise_distance": {
        "probe_meaning": "候选离散值扰动是否触发多种 handler、case 分支或离散处理状态。",
        "probe_high_value_meaning": "不同候选离散值之间的平均处理分歧越大，越可能触发多种 handler、case 分支或离散处理状态。",
    },
    "enum_max_pairwise_distance": {
        "probe_meaning": "枚举候选中最强的处理动作差异有多大。",
        "probe_high_value_meaning": "枚举候选中最强的一对 mutation 处理动作差异越大。",
    },
    "enum_metric_vector_variance": {
        "probe_meaning": "程序是否对某些候选值有特殊处理。",
        "probe_high_value_meaning": "程序对候选值响应越不稳定，越可能对某些候选值存在特殊处理。",
    },
    "enum_unique_vector_ratio": {
        "probe_meaning": "字段值空间是否被程序划分为多个离散语义类别。",
        "probe_high_value_meaning": "字段候选取值被程序划分出的离散语义类别越多。",
    },
    "enum_loop_dispersion": {
        "probe_meaning": "某些候选值是否改变循环、批量处理或复杂处理路径。",
        "probe_high_value_meaning": "候选值导致的循环、批量处理或复杂处理路径差异越大。",
    },
    "extreme_mean_baseline_distance": {
        "probe_meaning": "极端值是否打破正常处理流程。",
        "probe_high_value_meaning": "极端值 mutation 相对 baseline 造成的平均程序行为偏离越大，越可能打破正常处理流程。",
    },
    "extreme_mean_pairwise_distance": {
        "probe_meaning": "极端值扰动是否触发多种异常路径、拒绝路径或资源相关处理状态。",
        "probe_high_value_meaning": "不同极端值之间的平均处理分歧越大，越可能触发多种异常路径、拒绝路径或资源相关状态。",
    },
    "extreme_max_pairwise_distance": {
        "probe_meaning": "极端测试中最强的异常处理分歧有多大。",
        "probe_high_value_meaning": "极端测试中最强的一对 mutation 异常处理分歧越大。",
    },
    "extreme_metric_vector_variance": {
        "probe_meaning": "程序是否只对某些极端值爆发式响应。",
        "probe_high_value_meaning": "程序对极端值响应越不稳定，越可能只对某些极端值爆发式响应。",
    },
    "extreme_unique_vector_ratio": {
        "probe_meaning": "非典型取值是否被程序划分为多个异常处理类别。",
        "probe_high_value_meaning": "非典型取值被程序划分出的异常处理类别越多。",
    },
    "extreme_loop_dispersion": {
        "probe_meaning": "极端值是否引起循环放大、批处理规模变化或资源消耗路径。",
        "probe_high_value_meaning": "极端值导致的循环放大、批处理规模变化或资源消耗路径差异越大。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute z-probe correlations and top-k Stage 4A evidence.",
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--training-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--probe-table",
        type=Path,
        default=None,
        help="Optional Markdown probe table override. By default, the built-in Stage 4 probe dictionary is used.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
        help="Keep only evidence with abs(correlation) >= this threshold before top-k selection.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation metric used to rank positive/negative evidence.",
    )
    return parser.parse_args()


def probe_group(probe: str) -> str:
    if probe in PROBE_GROUPS:
        return PROBE_GROUPS[probe]
    for group in ("neighborhood", "boundary", "enum", "extreme"):
        if probe.startswith(group + "_"):
            return group
    return "unknown"


def parse_probe_table(path: Path) -> dict[str, dict[str, str]]:
    probes: dict[str, dict[str, str]] = {}
    table_row = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(?:.*?)\s*\|$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = table_row.match(line.strip())
        if not match:
            continue
        probe, meaning, high_value_meaning = match.groups()
        probes[probe] = {
            "probe_meaning": meaning.strip(),
            "probe_high_value_meaning": high_value_meaning.strip(),
        }
    return probes


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{source} missing required columns: {missing}")


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return math.nan
    left = pair.iloc[:, 0]
    right = pair.iloc[:, 1]
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return math.nan
    if method == "spearman":
        left = left.rank(method="average")
        right = right.rank(method="average")
    value = left.corr(right, method="pearson")
    return float(value) if pd.notna(value) else math.nan


def json_float(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def evidence_strength(count: int) -> str:
    if count >= 3:
        return "strong"
    if count == 2:
        return "medium"
    if count == 1:
        return "weak"
    return "insufficient"


def evidence_records(
    corr_df: pd.DataFrame,
    axis: str,
    direction: str,
    rank_by: str,
    top_k: int,
    correlation_threshold: float,
) -> list[dict[str, Any]]:
    metric = rank_by
    axis_df = corr_df[corr_df["axis"] == axis].copy()
    axis_df = axis_df[pd.notna(axis_df[metric])]
    axis_df = axis_df[axis_df[metric].abs() >= correlation_threshold]
    if direction == "positive":
        axis_df = axis_df[axis_df[metric] > 0].sort_values(metric, ascending=False)
    else:
        axis_df = axis_df[axis_df[metric] < 0].sort_values(metric, ascending=True)

    records: list[dict[str, Any]] = []
    for rank, row in enumerate(axis_df.head(top_k).to_dict(orient="records"), start=1):
        records.append(
            {
                "rank": rank,
                "probe": row["probe"],
                "feature_group": row["feature_group"],
                "correlation_metric": rank_by,
                "correlation": json_float(row[metric]),
                "abs_correlation": json_float(abs(row[metric])),
                "pearson": json_float(row["pearson"]),
                "spearman": json_float(row["spearman"]),
                "probe_meaning": row["probe_meaning"],
                "probe_high_value_meaning": row["probe_high_value_meaning"],
            }
        )
    return records


def write_report(
    path: Path,
    evidence: list[dict[str, Any]],
    embeddings_path: Path,
    matrix_path: Path,
    rank_by: str,
    top_k: int,
    correlation_threshold: float,
    n_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# Stage 4A Latent Probe Evidence")
    lines.append("")
    lines.append("This report summarizes the computed evidence for probe-anchored latent naming.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- embeddings: `{embeddings_path}`")
    lines.append(f"- training matrix: `{matrix_path}`")
    lines.append(f"- joined field rows: `{n_rows}`")
    lines.append(f"- ranking metric: `{rank_by}`")
    lines.append(f"- top-k per direction: `{top_k}`")
    lines.append(f"- evidence threshold: `abs({rank_by}) >= {correlation_threshold}`")
    lines.append("")
    lines.append("## Axis Evidence")
    lines.append("")
    for axis_obj in evidence:
        lines.append(f"### {axis_obj['axis']}")
        lines.append("")
        lines.append(
            "Positive evidence "
            f"(strength={axis_obj['positive_evidence_strength']}, "
            f"count={axis_obj['positive_evidence_count']}):"
        )
        for item in axis_obj["positive_evidence"]:
            corr = item["correlation"]
            lines.append(
                f"- `{item['probe']}` ({item['feature_group']}), {rank_by}={corr:.6f}: "
                f"{item['probe_meaning']} High value: {item['probe_high_value_meaning']}"
            )
        if not axis_obj["positive_evidence"]:
            lines.append("- none")
        lines.append("")
        lines.append(
            "Negative evidence "
            f"(strength={axis_obj['negative_evidence_strength']}, "
            f"count={axis_obj['negative_evidence_count']}):"
        )
        for item in axis_obj["negative_evidence"]:
            corr = item["correlation"]
            lines.append(
                f"- `{item['probe']}` ({item['feature_group']}), {rank_by}={corr:.6f}: "
                f"{item['probe_meaning']} High value: {item['probe_high_value_meaning']}"
            )
        if not axis_obj["negative_evidence"]:
            lines.append("- none")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = pd.read_csv(args.embeddings)
    matrix = pd.read_csv(args.training_matrix)
    require_columns(embeddings, KEY_COLS, args.embeddings)
    require_columns(matrix, KEY_COLS, args.training_matrix)

    axes = [c for c in embeddings.columns if AXIS_RE.match(c)]
    if not axes:
        raise SystemExit(f"{args.embeddings} has no z-axis columns")

    probe_cols = [c for c in matrix.columns if c not in KEY_COLS]
    if len(probe_cols) != 28:
        raise SystemExit(f"expected 28 probe columns in {args.training_matrix}, found {len(probe_cols)}")

    probe_table = parse_probe_table(args.probe_table) if args.probe_table else PROBE_TABLE
    missing_meanings = [p for p in probe_cols if p not in probe_table]
    if missing_meanings:
        source = str(args.probe_table) if args.probe_table else "built-in PROBE_TABLE"
        raise SystemExit(f"{source} missing probe meanings for: {missing_meanings}")

    joined = embeddings[KEY_COLS + axes].merge(matrix[KEY_COLS + probe_cols], on=KEY_COLS, how="inner")
    if joined.empty:
        raise SystemExit("no matching rows between embeddings and training matrix")

    rows: list[dict[str, Any]] = []
    for axis in axes:
        for probe in probe_cols:
            pearson = corr_pair(joined[axis], joined[probe], "pearson")
            spearman = corr_pair(joined[axis], joined[probe], "spearman")
            rows.append(
                {
                    "axis": axis,
                    "probe": probe,
                    "feature_group": probe_group(probe),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if pd.notna(pearson) else math.nan,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else math.nan,
                    "probe_meaning": probe_table[probe]["probe_meaning"],
                    "probe_high_value_meaning": probe_table[probe]["probe_high_value_meaning"],
                    "n": int(joined[[axis, probe]].dropna().shape[0]),
                }
            )

    corr_df = pd.DataFrame(rows)
    corr_path = args.out_dir / "z_probe_correlation.csv"
    corr_df.to_csv(corr_path, index=False)

    evidence: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for axis in axes:
        positive_evidence = evidence_records(
            corr_df,
            axis,
            "positive",
            args.rank_by,
            args.top_k,
            args.correlation_threshold,
        )
        negative_evidence = evidence_records(
            corr_df,
            axis,
            "negative",
            args.rank_by,
            args.top_k,
            args.correlation_threshold,
        )
        axis_obj = {
            "axis": axis,
            "rank_by": args.rank_by,
            "top_k": args.top_k,
            "correlation_threshold": args.correlation_threshold,
            "positive_evidence_strength": evidence_strength(len(positive_evidence)),
            "positive_evidence_count": len(positive_evidence),
            "negative_evidence_strength": evidence_strength(len(negative_evidence)),
            "negative_evidence_count": len(negative_evidence),
            "positive_evidence": positive_evidence,
            "negative_evidence": negative_evidence,
        }
        evidence.append(axis_obj)
        for direction, key in (("positive", "positive_evidence"), ("negative", "negative_evidence")):
            for item in axis_obj[key]:
                flat_rows.append(
                    {
                        "axis": axis,
                        "direction": direction,
                        "side_evidence_strength": axis_obj[f"{direction}_evidence_strength"],
                        "side_evidence_count": axis_obj[f"{direction}_evidence_count"],
                        "correlation_threshold": args.correlation_threshold,
                        **item,
                    }
                )

    evidence_json_path = args.out_dir / "z_topk_probe_evidence.json"
    evidence_json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(args.out_dir / "z_topk_probe_evidence.csv", index=False)

    write_report(
        args.out_dir / "latent_naming_report.md",
        evidence,
        args.embeddings,
        args.training_matrix,
        args.rank_by,
        args.top_k,
        args.correlation_threshold,
        len(joined),
    )

    print(f"[stage4a] joined rows: {len(joined)}")
    print(f"[stage4a] axes: {', '.join(axes)}")
    print(f"[stage4a] probes: {len(probe_cols)}")
    print(f"[stage4a] wrote: {corr_path}")
    print(f"[stage4a] wrote: {evidence_json_path}")


if __name__ == "__main__":
    main()
