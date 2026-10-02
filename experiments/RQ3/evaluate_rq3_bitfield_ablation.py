#!/usr/bin/env python3
"""Evaluate and summarize RQ3 bitfield-ablation outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SEMVEC_ROOT = Path("/root/semvec")
RQ3_ROOT = SEMVEC_ROOT / "RQ3"
DEFAULT_OUTDIR = RQ3_ROOT / "out"
DEFAULT_GT_JSONL = (
    SEMVEC_ROOT
    / "bitfield_groundtruth"
    / "evaluation_from_program_log"
    / "groundtruth_result"
    / "eval"
    / "program_log_groundtruth_candidates.jsonl"
)
DEFAULT_EVALUATOR = (Path(__file__).resolve().parents[2] / "tools/program_log/scripts"
                     / "evaluate_program_log_field_boundary.py")
MODES = ("operation_driven", "flat_evidence", "full")
DISPLAY_NAMES = {
    "operation_driven": "Operation-Driven Recovery",
    "flat_evidence": "Flat Evidence Aggregation",
    "full": "Full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RQ3 bitfield ablations with program-log groundtruth.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--groundtruth-jsonl", type=Path, default=DEFAULT_GT_JSONL)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--skip-evaluate", action="store_true", help="Only rebuild summary from existing metrics")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(summary: dict) -> dict[str, float]:
    overall = summary["overall"]
    paper = summary.get("protocol_value_average", {}).get("paper_metrics", {}).get("bit")
    if paper is None or paper["protocol_count"] == 0:
        raise ValueError("No V7 bit metrics: rerun the updated evaluator on reference bit fields.")
    return {
        "accuracy": paper["accuracy"],
        "f1_score": paper["f1"],
        "perfection": paper["perfection"],
        "protocol_count": paper["protocol_count"],
        "bitfield_precision": overall["bitfield_detection"]["precision"],
        "bitfield_recall": overall["bitfield_detection"]["recall"],
        "bitfield_f1": overall["bitfield_detection"]["f1"],
        "subfield_precision": overall["bitfield_boundary"]["subfield"]["precision"],
        "subfield_recall": overall["bitfield_boundary"]["subfield"]["recall"],
        "subfield_f1": overall["bitfield_boundary"]["subfield"]["f1"],
        "exact_partition_recall": overall["bitfield_boundary"]["exact_match_recall"],
    }


def fmt(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def markdown(rows: list[dict[str, Any]], manifest: dict) -> str:
    lines = [
        "# RQ3 位字段恢复消融实验汇总",
        "",
        "## 准确率对比",
        "",
        "沿用RQ1的边界指标；表中为有参考bit字段的协议的算术平均，旧指标仅保留为JSON诊断项。",
        "",
        "| 实验组 | Accuracy | F1-score | Perfection |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {fmt(row['accuracy'])} | {fmt(row['f1_score'])} | {fmt(row['perfection'])} |"
        )

    lines.extend([
        "",
        "## 原始恢复规模",
        "",
        "以下数量统计发生在评估侧 synthetic bit gap 补齐之前，可用于观察不同方法为后续 mutation 生成了多少候选单元。",
        "",
        "| 实验组 | 已处理数据包 | Bitfield 父字段数 | 原始 Subfield 数 |",
        "|---|---:|---:|---:|",
    ])
    mode_stats = manifest.get("modes", {})
    for row in rows:
        stats = mode_stats.get(row["mode"], {})
        lines.append(
            f"| {row['display_name']} | {stats.get('processed_packets', 0)} | "
            f"{stats.get('raw_bitfield_parents', 0)} | {stats.get('raw_subfields', 0)} |"
        )

    lines.extend([
        "",
        "## 说明",
        "",
        "- `Operation-Driven Recovery`：仅使用局部位操作证据恢复候选，不使用后续消费路径辅助确认。",
        "- `Flat Evidence Aggregation`：扫描完整轨迹并收集消费事件，但不再根据指令证据强弱裁决候选边界。",
        "- `Full`：位级消费追踪与指令证据裁决的完整流程。",
        "- 准确率评估复用 program-log 字段划分 groundtruth。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest_path = args.outdir / "rq3_generation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"generation manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)

    rows = []
    for mode in args.mode:
        replay_root = args.outdir / mode
        metrics_dir = args.outdir / "metrics" / mode
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / "field_boundary_metrics_summary.json"
        if not args.skip_evaluate:
            log_path = metrics_dir / "evaluate.log"
            command = [
                sys.executable,
                str(args.evaluator),
                "--groundtruth-jsonl",
                str(args.groundtruth_jsonl),
                "--replay-root",
                str(replay_root),
                "--outdir",
                str(metrics_dir),
                "--groundtruth-md",
                str(metrics_dir / "field_boundary_groundtruth_readable.md"),
                "--compare-md",
                str(metrics_dir / "field_boundary_groundtruth_vs_experiment_readable.md"),
            ]
            print(f"[rq3-eval] mode={mode}")
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        if not summary_path.exists():
            raise FileNotFoundError(f"metrics summary not found: {summary_path}")
        row = {"mode": mode, "display_name": DISPLAY_NAMES[mode]}
        row.update(metric_values(load_json(summary_path)))
        rows.append(row)

    summary_json = {"rows": rows, "generation_manifest": str(manifest_path)}
    (args.outdir / "rq3_bitfield_ablation_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.outdir / "rq3_bitfield_ablation_summary.md").write_text(
        markdown(rows, manifest),
        encoding="utf-8",
    )
    print(f"[rq3-eval] summary={args.outdir / 'rq3_bitfield_ablation_summary.md'}")


if __name__ == "__main__":
    main()
