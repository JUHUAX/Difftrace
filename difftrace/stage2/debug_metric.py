#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_metric.py - 单指标调试脚本
给定 baseline / mutation 日志路径与指标名，输出该指标值与中间计算信息。
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

from diff import (
    diff_metrics,
    lcp_length,
    parse_log_features,
    preprocess_log,
)


SUPPORTED_METRICS = (
    "branch_sites_jaccard",
    "bb_set_jaccard",
    "cmp_site_set_jaccard",
    "lcp_ratio",
    "instr_delta_ratio",
    "bb_multiset_l1_ratio",
    "cmp_delta_ratio",
    "branch_flip_ratio",
    "loop_delta_ratio",
)


def _load_features(path: str, preprocess: bool) -> Tuple[str, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"log not found: {path}")
    final_path = path
    if preprocess:
        final_path = preprocess_log(path)["path"]
    features = parse_log_features(final_path)
    return final_path, features


def _debug_intermediate(metric: str, base: Dict, other: Dict) -> Dict:
    if metric == "branch_sites_jaccard":
        a = set(base["branch_sites"])
        b = set(other["branch_sites"])
        return {
            "base_count": len(a),
            "mutation_count": len(b),
            "intersection_count": len(a & b),
            "union_count": len(a | b),
        }
    if metric == "bb_set_jaccard":
        a = set(base["bb_counts"])
        b = set(other["bb_counts"])
        return {
            "base_count": len(a),
            "mutation_count": len(b),
            "intersection_count": len(a & b),
            "union_count": len(a | b),
        }
    if metric == "cmp_site_set_jaccard":
        a = set(base["cmp_sites"])
        b = set(other["cmp_sites"])
        return {
            "base_count": len(a),
            "mutation_count": len(b),
            "intersection_count": len(a & b),
            "union_count": len(a | b),
        }
    if metric == "lcp_ratio":
        lcp = lcp_length(base["instr_addrs"], other["instr_addrs"])
        return {
            "lcp_len": lcp,
            "base_instr_len": len(base["instr_addrs"]),
            "mutation_instr_len": len(other["instr_addrs"]),
        }
    if metric == "instr_delta_ratio":
        b_len = len(base["instr_addrs"])
        o_len = len(other["instr_addrs"])
        return {
            "base_instr_len": b_len,
            "mutation_instr_len": o_len,
            "abs_diff": abs(o_len - b_len),
            "denominator": max(b_len, 1),
        }
    if metric == "bb_multiset_l1_ratio":
        keys = set(base["bb_counts"]) | set(other["bb_counts"])
        l1 = 0
        for k in keys:
            l1 += abs(base["bb_counts"].get(k, 0) - other["bb_counts"].get(k, 0))
        base_total = sum(base["bb_counts"].values())
        mut_total = sum(other["bb_counts"].values())
        return {
            "unique_bb_sites": len(keys),
            "l1": l1,
            "base_total_exec": base_total,
            "mutation_total_exec": mut_total,
            "denominator": base_total + mut_total,
        }
    if metric == "cmp_delta_ratio":
        b_count = int(base["cmp_count"])
        o_count = int(other["cmp_count"])
        return {
            "base_cmp_count": b_count,
            "mutation_cmp_count": o_count,
            "abs_diff": abs(o_count - b_count),
            "denominator": max(b_count, 1),
        }
    if metric == "branch_flip_ratio":
        base_sites = set(base["branch_sites"])
        other_sites = set(other["branch_sites"])
        common = base_sites & other_sites
        flip = 0
        for site in common:
            if base["branch_outcomes_first"].get(site) != other["branch_outcomes_first"].get(site):
                flip += 1
        return {
            "common_branch_sites": len(common),
            "flip_count": flip,
            "denominator": max(len(common), 1),
        }
    if metric == "loop_delta_ratio":
        return {
            "base_loop_instruction_count": int(base.get("loop_instruction_count", 0)),
            "mutation_loop_instruction_count": int(other.get("loop_instruction_count", 0)),
            "base_loop_density": float(base.get("loop_density", 0.0)),
            "mutation_loop_density": float(other.get("loop_density", 0.0)),
        }
    return {}


def _print_list_block(title: str, items: List[str], max_show: int) -> None:
    print(f"- {title}: {len(items)}")
    if not items:
        return
    for x in items[:max_show]:
        print(f"  {x}")
    if len(items) > max_show:
        print(f"  ... 还有 {len(items) - max_show} 条")


def _print_feature_overview(name: str, f: Dict) -> None:
    print(f"[{name}]")
    print(f"  instr_count={f.get('instr_count', 0)}")
    print(f"  bb_sites={len(f.get('bb_counts', {}))}")
    print(f"  branch_sites={len(f.get('branch_sites', []))}")
    print(f"  cmp_sites={len(f.get('cmp_sites', []))}, cmp_count={f.get('cmp_count', 0)}")
    print(f"  loop_count={f.get('loop_instruction_count', 0)}, loop_density={f.get('loop_density', 0.0)}")


def _print_metric_detail(metric: str, value: float, inter: Dict, base: Dict, other: Dict, max_show: int) -> None:
    print("\n=== 指标计算细节 ===")
    print(f"metric = {metric}")
    print(f"value  = {value}")
    print()

    if metric in ("branch_sites_jaccard", "bb_set_jaccard", "cmp_site_set_jaccard"):
        print("公式: |A∩B| / |A∪B|")
        print(f"A_count={inter['base_count']}, B_count={inter['mutation_count']}")
        print(f"intersection={inter['intersection_count']}, union={inter['union_count']}")
        if inter["union_count"] > 0:
            print(f"check={inter['intersection_count']}/{inter['union_count']}={inter['intersection_count']/inter['union_count']}")

        if metric == "branch_sites_jaccard":
            a = sorted(set(base["branch_sites"]))
            b = sorted(set(other["branch_sites"]))
        elif metric == "bb_set_jaccard":
            a = sorted(set(base["bb_counts"]))
            b = sorted(set(other["bb_counts"]))
        else:
            a = sorted(set(base["cmp_sites"]))
            b = sorted(set(other["cmp_sites"]))
        inter_set = sorted(set(a) & set(b))
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        _print_list_block("交集示例", inter_set, max_show)
        _print_list_block("仅 baseline", only_a, max_show)
        _print_list_block("仅 mutation", only_b, max_show)
        return

    if metric == "lcp_ratio":
        print("公式: LCP_len / len(S_base)")
        print(f"LCP_len={inter['lcp_len']}, base_len={inter['base_instr_len']}, mutation_len={inter['mutation_instr_len']}")
        if inter["base_instr_len"] > 0:
            print(f"check={inter['lcp_len']}/{inter['base_instr_len']}={inter['lcp_len']/inter['base_instr_len']}")
        i = inter["lcp_len"]
        if i < len(base["instr_addrs"]) and i < len(other["instr_addrs"]):
            print(f"首个分歧位置={i}")
            print(f"  baseline[{i}]={base['instr_addrs'][i]}")
            print(f"  mutation[{i}]={other['instr_addrs'][i]}")
        else:
            print("未发现分歧（至少一侧在 LCP 处结束）")
        return

    if metric == "instr_delta_ratio":
        print("公式: |I_mut-I_base| / max(I_base,1)")
        print(f"I_base={inter['base_instr_len']}, I_mut={inter['mutation_instr_len']}")
        print(f"abs_diff={inter['abs_diff']}, denominator={inter['denominator']}")
        if inter["denominator"] > 0:
            print(f"check={inter['abs_diff']}/{inter['denominator']}={inter['abs_diff']/inter['denominator']}")
        return

    if metric == "bb_multiset_l1_ratio":
        print("公式: L1 / (sum(v_base)+sum(v_mut))")
        print(f"unique_bb_sites={inter['unique_bb_sites']}")
        print(f"L1={inter['l1']}, base_total={inter['base_total_exec']}, mut_total={inter['mutation_total_exec']}, denominator={inter['denominator']}")
        if inter["denominator"] > 0:
            print(f"check={inter['l1']}/{inter['denominator']}={inter['l1']/inter['denominator']}")
        # 显示差异最大的 bb 站点
        keys = set(base["bb_counts"]) | set(other["bb_counts"])
        diffs = []
        for k in keys:
            d = abs(base["bb_counts"].get(k, 0) - other["bb_counts"].get(k, 0))
            if d > 0:
                diffs.append((d, k, base["bb_counts"].get(k, 0), other["bb_counts"].get(k, 0)))
        diffs.sort(reverse=True)
        print("top 差异 BB:")
        for d, k, b_cnt, m_cnt in diffs[:max_show]:
            print(f"  {k}: |{b_cnt}-{m_cnt}|={d}")
        if len(diffs) > max_show:
            print(f"  ... 还有 {len(diffs)-max_show} 条")
        return

    if metric == "cmp_delta_ratio":
        print("公式: |C_mut-C_base| / max(C_base,1)")
        print(f"C_base={inter['base_cmp_count']}, C_mut={inter['mutation_cmp_count']}")
        print(f"abs_diff={inter['abs_diff']}, denominator={inter['denominator']}")
        if inter["denominator"] > 0:
            print(f"check={inter['abs_diff']}/{inter['denominator']}={inter['abs_diff']/inter['denominator']}")
        return

    if metric == "branch_flip_ratio":
        print("公式: flip_count / max(|B_common|,1)")
        print(f"common_branch_sites={inter['common_branch_sites']}, flip_count={inter['flip_count']}, denominator={inter['denominator']}")
        if inter["denominator"] > 0:
            print(f"check={inter['flip_count']}/{inter['denominator']}={inter['flip_count']/inter['denominator']}")
        base_sites = set(base["branch_sites"])
        other_sites = set(other["branch_sites"])
        common = sorted(base_sites & other_sites)
        flipped = []
        for site in common:
            b = base["branch_outcomes_first"].get(site)
            m = other["branch_outcomes_first"].get(site)
            if b != m:
                flipped.append(f"{site}: {b} -> {m}")
        _print_list_block("翻转站点", flipped, max_show)
        return

    if metric == "loop_delta_ratio":
        print("公式: |loop_density_mut-loop_density_base|")
        print(
            f"base_loop_count={inter['base_loop_instruction_count']}, "
            f"mutation_loop_count={inter['mutation_loop_instruction_count']}"
        )
        print(
            f"base_loop_density={inter['base_loop_density']}, "
            f"mutation_loop_density={inter['mutation_loop_density']}"
        )
        print(f"check=abs({inter['mutation_loop_density']}-{inter['base_loop_density']})={value}")
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算单个指标并输出中间计算信息"
    )
    parser.add_argument("--baseline", required=True, help="baseline 日志路径")
    parser.add_argument("--mutation", required=True, help="mutation 日志路径")
    parser.add_argument("--metric", required=True, choices=SUPPORTED_METRICS,
                        help="指标名称")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="禁用预处理（默认先做与 diff.py 一致的预处理）")
    parser.add_argument("--max-show", type=int, default=10,
                        help="明细最多显示条目数（默认 10）")
    args = parser.parse_args()

    preprocess = not args.no_preprocess
    try:
        base_path, base_features = _load_features(args.baseline, preprocess)
        mut_path, mut_features = _load_features(args.mutation, preprocess)
        all_metrics = diff_metrics(base_features, mut_features)
        value = all_metrics[args.metric]
        inter = _debug_intermediate(args.metric, base_features, mut_features)
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    print("=" * 80)
    print("Metric Debug")
    print("=" * 80)
    print(f"baseline_log = {base_path}")
    print(f"mutation_log = {mut_path}")
    print(f"preprocess   = {preprocess}")
    print()
    _print_feature_overview("baseline", base_features)
    print()
    _print_feature_overview("mutation", mut_features)
    _print_metric_detail(args.metric, value, inter, base_features, mut_features, max(1, args.max_show))


if __name__ == "__main__":
    main()
