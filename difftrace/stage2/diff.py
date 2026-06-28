#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diff.py - 计算差分指标
比较 baseline 与 mutation 的执行差异，输出 report.json
"""

import argparse
import datetime
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

from common import (
    DEFAULT_OUTDIR,
    print_user_error,
    read_json,
    write_json,
)
from field_units import (
    field_id,
    normalize_unit,
    read_value,
    safe_field_dir,
    units_from_fields_json,
    width_bits as unit_width_bits,
)


THREAD_PREFIX_RE = re.compile(r"^THREADID\t([^\t]+)\t(.*)$")


def split_thread_prefix(line: str) -> Tuple[Optional[str], str]:
    """拆分可选的 THREADID 前缀。"""
    match = THREAD_PREFIX_RE.match(line)
    if not match:
        return None, line
    return match.group(1), match.group(2)


def split_loop_prefix(payload: str) -> Tuple[bool, str]:
    """拆分 LOOP 标记，兼容 LOOP 前缀和行内 LOOP 标记。"""
    loop_marked = False
    text = payload
    while text.startswith("LOOP\t"):
        loop_marked = True
        text = text[len("LOOP\t"):]
    if "\tLOOP" in text or text.endswith(" LOOP"):
        loop_marked = True
    return loop_marked, text


def parse_field_indices(raw: str) -> set:
    return {int(token) for token in re.findall(r"\d+", raw or "")}


def preprocess_log(log_path: str, seen_taint_thread_ids: Optional[set] = None) -> dict:
    """预处理日志：
    1) 扫描所有 Taint，统计其 THREADID 出现次数；
    2) 选择 Taint 次数最多的 THREADID；
    3) 若并列，则优先选择最后一个 Taint 的 THREADID；
    4) 若仍无可用 THREADID，则退化为从第一个 Taint 开始截断。

    输出文件与原日志同目录，命名为 `<name>.preprocessed.log`。
    """
    root, ext = os.path.splitext(log_path)
    out_path = log_path if root.endswith(".preprocessed") else root + ".preprocessed" + ext

    parsed_lines: List[Tuple[str, Optional[str], str]] = []
    taint_events: List[Tuple[int, Optional[str]]] = []
    kept_lines: List[str] = []
    taint_found = False
    first_taint_tid: Optional[str] = None
    last_taint_tid: Optional[str] = None
    taint_thread_counts: Dict[str, int] = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            line_tid, payload = split_thread_prefix(line)
            parsed_lines.append((line, line_tid, payload))

            if payload.startswith("Taint\t"):
                if not taint_found:
                    taint_found = True
                    first_taint_tid = line_tid
                taint_events.append((len(parsed_lines) - 1, line_tid))
                if line_tid is not None:
                    taint_thread_counts[line_tid] = taint_thread_counts.get(line_tid, 0) + 1
                    last_taint_tid = line_tid

    target_tid: Optional[str] = None
    if taint_found:
        # 新逻辑：优先选择“第一个未被使用过”的 Taint 线程 ID
        if seen_taint_thread_ids is not None:
            for _, tid in taint_events:
                if tid is None:
                    continue
                if tid not in seen_taint_thread_ids:
                    target_tid = tid
                    break

        if taint_thread_counts:
            if target_tid is None:
                max_count = max(taint_thread_counts.values())
                candidate_tids = {
                    tid for tid, count in taint_thread_counts.items()
                    if count == max_count
                }
                if max_count == 1:
                    target_tid = first_taint_tid
                elif last_taint_tid in candidate_tids:
                    target_tid = last_taint_tid
                elif first_taint_tid in candidate_tids:
                    target_tid = first_taint_tid
                else:
                    target_tid = next(iter(candidate_tids))

        if target_tid is None:
            saw_taint = False
            for line, _, payload in parsed_lines:
                if not saw_taint and payload.startswith("Taint\t"):
                    saw_taint = True
                if saw_taint:
                    kept_lines.append(line)
        else:
            if seen_taint_thread_ids is not None:
                seen_taint_thread_ids.add(target_tid)
            start_index = next(
                index for index, tid in taint_events if tid == target_tid
            )
            for index, (line, line_tid, _) in enumerate(parsed_lines):
                if index >= start_index and line_tid == target_tid:
                    kept_lines.append(line)

    if out_path != log_path or taint_found:
        with open(out_path, "w", encoding="utf-8") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

    return {
        "path": out_path,
        "taint_found": taint_found,
        "taint_thread_id": target_tid,
        "line_count": len(kept_lines),
    }


def parse_log_features(log_path: str) -> dict:
    """解析日志提取特征"""
    instr_addrs: List[str] = []
    bb_counts: Dict[str, int] = {}
    bb_set: set = set()
    branch_outcomes_first: Dict[str, str] = {}
    branch_sites: set = set()
    cmp_sites: set = set()
    cmp_count = 0
    instr_count = 0
    loop_instruction_count = 0
    taint_found = False

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            _, payload = split_thread_prefix(line)
            loop_marked, payload = split_loop_prefix(payload)
            if not taint_found:
                if payload.startswith("Taint\t"):
                    taint_found = True
                continue

            if payload.startswith("BasicBlock\t"):
                parts = payload.split("\t")
                if len(parts) >= 2:
                    addr = parts[1].strip()
                    bb_counts[addr] = bb_counts.get(addr, 0) + 1
                    bb_set.add(addr)
            elif payload.startswith("Instruction\t"):
                instr_count += 1
                if loop_marked:
                    loop_instruction_count += 1
                m = re.match(r"^Instruction\t([^:]+):\s+(.*)$", payload)
                if not m:
                    continue
                addr = m.group(1).strip()
                instr_addrs.append(addr)

                disasm = m.group(2).split("\t")[0].strip()
                mnemonic = disasm.split(" ", 1)[0].lower() if disasm else ""
                if mnemonic in ("cmp", "test"):
                    cmp_count += 1
                    cmp_sites.add(addr)

                # ⚠️ 重要：必须先检查 NOT_TAKEN，因为 NOT_TAKEN 也以 TAKEN 结尾
                if payload.endswith("NOT_TAKEN"):
                    branch_sites.add(addr)
                    if addr not in branch_outcomes_first:
                        branch_outcomes_first[addr] = "NOT_TAKEN"
                elif payload.endswith("TAKEN"):
                    branch_sites.add(addr)
                    if addr not in branch_outcomes_first:
                        branch_outcomes_first[addr] = "TAKEN"

    bb_top = sorted(bb_counts.items(), key=lambda x: (-x[1], x[0]))[:10]

    return {
        "instr_addrs": instr_addrs,
        "instr_count": instr_count,
        "bb_counts": bb_counts,
        "bb_set": list(bb_set),
        "branch_sites": list(branch_sites),
        "branch_outcomes_first": branch_outcomes_first,
        "cmp_sites": list(cmp_sites),
        "cmp_count": cmp_count,
        "loop_instruction_count": loop_instruction_count,
        "loop_density": loop_instruction_count / max(instr_count, 1),
        "debug": {
            "bb_top": bb_top,
            "branch_site_count": len(branch_sites),
            "taint_found": taint_found,
        },
    }


def compute_field_suffix_capacity(log_path: str, a: int, b: int) -> dict:
    """统计字段首次相关指令之后的 baseline 后续执行空间。

    这里的“后续”定义为：从该字段在 baseline 中第一次相关 tainted 指令出现位置开始，
    到日志结束为止的可影响执行空间。
    """
    target = set(range(int(a), int(b) + 1))
    suffix_instr_count = 0
    suffix_cmp_count = 0
    suffix_bb_exec_total = 0

    total_instr_count = 0
    total_cmp_count = 0
    total_bb_exec_total = 0

    taint_found = False
    activated = False
    first_use_line_no: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            _, payload = split_thread_prefix(line)
            if not taint_found:
                if payload.startswith("Taint\t"):
                    taint_found = True
                continue

            loop_marked, payload = split_loop_prefix(payload)
            _ = loop_marked  # 占位，保持与 parse_log_features 解析方式一致

            if payload.startswith("BasicBlock\t"):
                total_bb_exec_total += 1
                if activated:
                    suffix_bb_exec_total += 1
                continue

            if not payload.startswith("Instruction\t"):
                continue

            total_instr_count += 1
            parts = payload.split("\t")
            if len(parts) >= 3:
                idx_set = parse_field_indices(parts[2])
            else:
                idx_set = set()
            if not activated and idx_set & target:
                activated = True
                first_use_line_no = line_no

            if activated:
                suffix_instr_count += 1

            m = re.match(r"^Instruction\t([^:]+):\s+(.*)$", payload)
            if not m:
                continue
            disasm = m.group(2).split("\t")[0].strip()
            mnemonic = disasm.split(" ", 1)[0].lower() if disasm else ""
            if mnemonic in ("cmp", "test"):
                total_cmp_count += 1
                if activated:
                    suffix_cmp_count += 1

    if not activated:
        suffix_instr_count = total_instr_count
        suffix_cmp_count = total_cmp_count
        suffix_bb_exec_total = total_bb_exec_total

    return {
        "first_use_found": activated,
        "first_use_line_no": first_use_line_no,
        "suffix_instr_count": suffix_instr_count,
        "suffix_cmp_count": suffix_cmp_count,
        "suffix_bb_exec_total": suffix_bb_exec_total,
        "total_instr_count": total_instr_count,
        "total_cmp_count": total_cmp_count,
        "total_bb_exec_total": total_bb_exec_total,
    }


def jaccard(a: set, b: set) -> float:
    """计算 Jaccard 相似度"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def lcp_length(seq1: List[str], seq2: List[str]) -> int:
    """计算最长公共前缀长度"""
    n = min(len(seq1), len(seq2))
    for i in range(n):
        if seq1[i] != seq2[i]:
            return i
    return n


def diff_metrics(base: dict, other: dict, field_suffix_capacity: Optional[dict] = None) -> dict:
    """计算差分指标 (Behavioral Differential Metrics v2)."""
    base_branch_sites = set(base["branch_sites"])
    other_branch_sites = set(other["branch_sites"])
    common_branch_sites = base_branch_sites & other_branch_sites
    branch_flip = 0
    for site in common_branch_sites:
        if base["branch_outcomes_first"].get(site) != other["branch_outcomes_first"].get(site):
            branch_flip += 1

    bb_all = set(base["bb_counts"]) | set(other["bb_counts"])
    bb_l1 = 0
    for addr in bb_all:
        bb_l1 += abs(base["bb_counts"].get(addr, 0) - other["bb_counts"].get(addr, 0))

    cmp_base_set = set(base["cmp_sites"])
    cmp_other_set = set(other["cmp_sites"])

    lcp_len = lcp_length(base["instr_addrs"], other["instr_addrs"])
    base_instr_len = len(base["instr_addrs"])
    other_instr_len = len(other["instr_addrs"])
    if base_instr_len == 0:
        lcp_ratio = 1.0 if other_instr_len == 0 else 0.0
    else:
        lcp_ratio = lcp_len / base_instr_len

    suffix_instr_count = int((field_suffix_capacity or {}).get("suffix_instr_count") or base_instr_len)
    suffix_cmp_count = int((field_suffix_capacity or {}).get("suffix_cmp_count") or int(base.get("cmp_count", 0) or 0))
    suffix_bb_exec_total = int((field_suffix_capacity or {}).get("suffix_bb_exec_total") or sum(base["bb_counts"].values()))

    instr_delta_raw = abs(other_instr_len - base_instr_len) / max(suffix_instr_count, 1)
    cmp_delta_raw = abs(other["cmp_count"] - base["cmp_count"]) / max(suffix_cmp_count, 1)
    bb_multiset_l1_raw = bb_l1 / max(suffix_bb_exec_total, 1)

    # 对三个总量型长尾指标做单调压缩：
    # 先做后续执行空间归一化，再做 log1p，保留“值越大扰动越强”的排序，
    # 同时避免极端长尾值在后续摘要统计中主导 pairwise distance / variance。
    instr_delta_ratio = math.log1p(instr_delta_raw)
    cmp_delta_ratio = math.log1p(cmp_delta_raw)
    bb_multiset_l1_ratio = math.log1p(bb_multiset_l1_raw)
    branch_flip_ratio = branch_flip / max(len(common_branch_sites), 1)
    loop_delta_ratio = abs(float(other.get("loop_density", 0.0)) - float(base.get("loop_density", 0.0)))

    return {
        "branch_sites_jaccard": jaccard(base_branch_sites, other_branch_sites),
        "bb_set_jaccard": jaccard(set(base["bb_counts"]), set(other["bb_counts"])),
        "cmp_site_set_jaccard": jaccard(cmp_base_set, cmp_other_set),
        "lcp_ratio": lcp_ratio,
        "instr_delta_ratio": instr_delta_ratio,
        "bb_multiset_l1_ratio": bb_multiset_l1_ratio,
        "suffix_instr_count": suffix_instr_count,
        "suffix_cmp_count": suffix_cmp_count,
        "suffix_bb_exec_total": suffix_bb_exec_total,
        # Optional/diagnostic metrics in v2
        "cmp_delta_ratio": cmp_delta_ratio,
        "branch_flip_ratio": branch_flip_ratio,
        "loop_delta_ratio": loop_delta_ratio,
    }


def summarize_metrics(metric_list: List[dict]) -> dict:
    """汇总统计指标"""
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    summary = {}
    for key in keys:
        values = [m[key] for m in metric_list]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def summarize_metric_diversity(metric_list: List[dict], eps: float = 1e-6) -> dict:
    """汇总同字段下 mutation 指标的分化程度。"""
    if len(metric_list) <= 1:
        return {
            "deltaf_dispersion": 0.0,
            "unique_metric_vectors": len(metric_list),
            "low_diversity": 0,
        }
    keys = sorted(metric_list[0].keys())
    rounded_vectors = []
    for metrics in metric_list:
        rounded_vectors.append(tuple(round(float(metrics.get(k, 0.0)), 6) for k in keys))
    unique_count = len(set(rounded_vectors))

    # 以每个指标 across mutations 的总体标准差均值作为离散度。
    std_values = []
    for key in keys:
        vals = [float(item.get(key, 0.0)) for item in metric_list]
        std_values.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    dispersion = float(statistics.mean(std_values)) if std_values else 0.0

    return {
        "deltaf_dispersion": dispersion,
        "unique_metric_vectors": unique_count,
        "low_diversity": 1 if unique_count <= 1 or dispersion <= eps else 0,
    }


def parse_log_health(log_path: str) -> dict:
    """解析日志健康度：从第一个 Taint 标记开始统计 Instruction 和 BasicBlock"""
    instr_parsed = 0
    bb_parsed = 0
    branch_parsed = 0
    bad_lines = 0
    first_line = None
    last_line = None
    known_prefixes = ("Instruction\t", "BasicBlock\t", "Function\t", "Taint\t", "LOOP\t")
    taint_found = False

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            _, payload = split_thread_prefix(line)
            _, payload_for_count = split_loop_prefix(payload)
            if first_line is None:
                first_line = payload
            last_line = payload

            # 标记第一个 Taint 出现
            if not taint_found and payload_for_count.startswith("Taint\t"):
                taint_found = True

            # 只在 Taint 出现后才开始统计 Instruction 和 BasicBlock
            if taint_found:
                if payload_for_count.startswith("Instruction\t"):
                    instr_parsed += 1
                    if payload_for_count.endswith("TAKEN") or payload_for_count.endswith("NOT_TAKEN"):
                        branch_parsed += 1
                elif payload_for_count.startswith("BasicBlock\t"):
                    bb_parsed += 1
                elif payload_for_count.startswith(known_prefixes):
                    continue
                else:
                    bad_lines += 1
            # Taint 未出现前，只统计非法行（用于诊断）
            elif not payload_for_count.startswith(known_prefixes):
                bad_lines += 1

    start_ok = bool(first_line and first_line.startswith(known_prefixes))
    end_ok = bool(last_line and last_line.startswith(known_prefixes))

    return {
        "instr_parsed": instr_parsed,
        "bb_parsed": bb_parsed,
        "branch_parsed": branch_parsed,
        "bad_lines_dropped": bad_lines,
        "taint_found": taint_found,
        "slice_line_aligned": {
            "start_ok": start_ok,
            "end_ok": end_ok,
        },
    }


def relpath(path: str, outdir: str) -> str:
    return os.path.relpath(path, outdir)


def load_json_if_exists(path: str) -> dict:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return read_json(path)
    return {}


def parse_field_range(field_dir_name: str, meta: dict) -> Tuple[int, int]:
    field = meta.get("field") if isinstance(meta, dict) else None
    if isinstance(field, dict) and "a" in field and "b" in field:
        return int(field["a"]), int(field["b"])
    parts = field_dir_name.split("_")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    if len(parts) == 1 and parts[0].isdigit():
        value = int(parts[0])
        return value, value
    return -1, -1


def build_mutation_index(mutations_json: object) -> Dict[str, Dict[str, dict]]:
    index: Dict[str, Dict[str, dict]] = {}
    if not isinstance(mutations_json, list):
        return index
    for entry in mutations_json:
        if not isinstance(entry, dict):
            continue
        field = entry.get("field", {})
        if isinstance(field, dict) and "a" in field and "b" in field:
            field_key = safe_field_dir(field)
        else:
            continue
        muts = entry.get("mutations")
        if not isinstance(muts, list):
            continue
        index.setdefault(field_key, {})
        for mut in muts:
            if not isinstance(mut, dict):
                continue
            payload_hex = mut.get("payload_hex", "")
            if payload_hex:
                index[field_key][payload_hex.lower()] = mut
    return index


def build_field_unit_index(fields_json: dict, mutations_json: object) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for unit in units_from_fields_json(fields_json):
        index[safe_field_dir(unit)] = unit
    if isinstance(mutations_json, list):
        for entry in mutations_json:
            if not isinstance(entry, dict):
                continue
            field = entry.get("field")
            if isinstance(field, dict) and "a" in field and "b" in field:
                index[safe_field_dir(field)] = normalize_unit(field)
    return index


def format_ts(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(timestamp).isoformat()


def format_value_hex(value: object, width_bits: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and value.startswith("0x"):
            return value.lower()
        return None
    if width_bits is None or width_bits <= 0:
        return f"0x{ivalue:x}"
    mask = (1 << width_bits) - 1
    width_nibbles = max(1, (width_bits + 3) // 4)
    return f"0x{(ivalue & mask):0{width_nibbles}x}"


def field_id_sort_key(field_id: str) -> Tuple[int, int, str]:
    parts = field_id.split("_")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1]), field_id
    if len(parts) == 1 and parts[0].isdigit():
        value = int(parts[0])
        return value, value, field_id
    return sys.maxsize, sys.maxsize, field_id


def build_compact_report(report: dict) -> dict:
    compact_fields = []
    fields = report.get("fields", [])
    if isinstance(fields, list):
        for field in fields:
            if not isinstance(field, dict):
                continue
            diff = field.get("diff", {})
            per_mutation = diff.get("per_mutation", []) if isinstance(diff, dict) else []
            summary = diff.get("summary", {}) if isinstance(diff, dict) else {}
            diagnostic = summary.get("diagnostic", {}) if isinstance(summary, dict) else {}
            run_final_values = {}
            runs = field.get("runs", [])
            if isinstance(runs, list):
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    run_id = run.get("run_id")
                    field_value = run.get("field_value", {})
                    if not isinstance(run_id, str) or not isinstance(field_value, dict):
                        continue
                    run_final_values[run_id] = field_value.get("final_value_hex")

            compact_per_mutation = []
            if isinstance(per_mutation, list):
                for item in per_mutation:
                    if not isinstance(item, dict):
                        continue
                    run_id = str(item.get("run_id", ""))
                    row = dict(item)
                    row["final_value_hex"] = item.get("final_value_hex") or run_final_values.get(run_id)
                    compact_per_mutation.append(row)
            compact_fields.append({
                "field_id": field.get("field_id", ""),
                "field_kind": field.get("field_kind", "byte"),
                "field_unit": field.get("field_unit", {}),
                "baseline_field_hex": field.get("baseline_field_hex", ""),
                "per_mutation": compact_per_mutation,
            })
    compact_fields.sort(key=lambda item: field_id_sort_key(str(item.get("field_id", ""))))
    return {
        "created_at": report.get("created_at", ""),
        "fields": compact_fields,
    }


def cmd_diff(args: argparse.Namespace) -> None:
    """diff 子命令主逻辑"""
    outdir = args.outdir
    mutations_root = os.path.join(outdir, "mutations")
    if not os.path.exists(mutations_root):
        raise FileNotFoundError("mutations directory not found")

    sample = load_json_if_exists(os.path.join(outdir, "sample.json"))
    fields_json = load_json_if_exists(os.path.join(outdir, "fields.json"))
    mutations_json = load_json_if_exists(os.path.join(outdir, "mutations.json"))
    mutation_index = build_mutation_index(mutations_json)
    field_unit_index = build_field_unit_index(fields_json, mutations_json)

    baseline_meta_path = os.path.join(mutations_root, "baseline.json")
    baseline_log_path = os.path.join(mutations_root, "baseline.log")
    baseline_meta = load_json_if_exists(baseline_meta_path)
    if baseline_meta.get("log_slice_path"):
        baseline_log_path = baseline_meta["log_slice_path"]

    if not os.path.exists(baseline_log_path):
        raise RuntimeError("baseline log not found in mutations directory")

    baseline_payload_hex = (sample.get("payload_hex", "") or "").lower()
    payload_len = 0
    try:
        payload_len = len(bytes.fromhex(baseline_payload_hex)) if baseline_payload_hex else 0
    except ValueError:
        payload_len = 0

    fields_partition = []
    if isinstance(fields_json.get("fields"), list):
        for item in fields_json["fields"]:
            if isinstance(item, dict) and "a" in item and "b" in item:
                fields_partition.append(normalize_unit(item))

    experiment = {
        "mode": sample.get("mode", ""),
        "proto": sample.get("proto", baseline_meta.get("proto", "")),
        "target": {
            "host": baseline_meta.get("target", {}).get("host", ""),
            "port": baseline_meta.get("target", {}).get("port", ""),
        },
        "seed": sample.get("seed", 0),
        "outdir": outdir,
        "pin_tool_log_path": baseline_meta.get("pin_log", ""),
        "pcap_path": sample.get("pcap", ""),
        "pcap_packet_index": sample.get("index", None),
        "hex_payload": sample.get("hex", ""),
    }

    report = {
        "created_at": format_ts(time.time()),
        "experiment": experiment,
        "input": {
            "baseline_payload_hex": baseline_payload_hex,
            "payload_len": payload_len,
            "fields_partition": fields_partition,
            "field_partition_source": "fields.json",
        },
        "fields": [],
    }

    seen_taint_thread_ids: set = set()
    staged_baseline_preprocessed = baseline_meta.get("preprocessed_log_path")
    if staged_baseline_preprocessed and os.path.exists(staged_baseline_preprocessed):
        baseline_preprocessed_path = staged_baseline_preprocessed
        baseline_preprocess = {
            "path": baseline_preprocessed_path,
            "taint_found": True,
            "taint_thread_id": None,
            "line_count": None,
            "source": "staged_preprocessed_log",
        }
    else:
        baseline_preprocess = preprocess_log(baseline_log_path, seen_taint_thread_ids)
        baseline_preprocessed_path = baseline_preprocess["path"]

    base_features = None
    base_health = None
    base_status = "ok"
    base_error: Optional[str] = None
    try:
        base_health = parse_log_health(baseline_preprocessed_path)
        base_features = parse_log_features(baseline_preprocessed_path)
    except Exception as exc:
        base_status = "parse_fail"
        base_error = str(exc)

    for name in sorted(os.listdir(mutations_root)):
        field_dir = os.path.join(mutations_root, name)
        if not os.path.isdir(field_dir):
            continue

        field_unit = field_unit_index.get(name)
        if field_unit is None:
            a, b = parse_field_range(name, {})
            if a >= 0 and b >= 0:
                field_unit = {"kind": "byte", "a": a, "b": b, "repr": f"{a}" if a == b else f"{a},{b}"}
        if field_unit is None:
            continue
        field_unit = normalize_unit(field_unit)
        a = int(field_unit["a"])
        b = int(field_unit["b"])
        nbytes = b - a + 1 if a >= 0 and b >= 0 else 0
        w_bits = unit_width_bits(field_unit)
        if a >= 0 and b >= 0 and baseline_payload_hex:
            baseline_bytes = bytes.fromhex(baseline_payload_hex)
            baseline_field_hex = format_value_hex(read_value(baseline_bytes, field_unit, "big"), w_bits)
            baseline_value = read_value(baseline_bytes, field_unit, "big")
        else:
            baseline_field_hex = ""
            baseline_value = None

        baseline_run = {
            "run_id": "baseline",
            "kind": "baseline",
            "strategy": "baseline",
            "payload_hex": baseline_payload_hex,
            "field_value": {
                "requested_value_hex": format_value_hex(baseline_value, w_bits),
                "final_value_hex": format_value_hex(baseline_value, w_bits),
                "w_bits": w_bits,
            },
            "artifact": {
                "log_slice_path": relpath(baseline_log_path, outdir),
                "preprocessed_log_path": relpath(baseline_preprocessed_path, outdir),
                "log_slice_bytes": baseline_meta.get("slice_bytes", None),
            },
            "preprocess": baseline_preprocess,
            "parse_health": base_health or {
                "instr_parsed": 0,
                "bb_parsed": 0,
                "branch_parsed": 0,
                "bad_lines_dropped": 0,
                "taint_found": False,
                "slice_line_aligned": {"start_ok": False, "end_ok": False},
            },
            "status": base_status,
            "error": base_error,
        }

        field_runs = [baseline_run]
        per_mutation = []
        per_metric = []
        field_suffix_capacity = compute_field_suffix_capacity(baseline_preprocessed_path, a, b)
        for entry in sorted(os.listdir(field_dir)):
            if not entry.endswith(".log"):
                continue
            if entry.endswith(".preprocessed.log"):
                continue
            log_path = os.path.join(field_dir, entry)
            meta_path = log_path[:-4] + ".json"
            meta = load_json_if_exists(meta_path)
            meta_field = meta.get("field") if isinstance(meta, dict) else None
            run_unit = normalize_unit(meta_field) if isinstance(meta_field, dict) and "a" in meta_field and "b" in meta_field else field_unit

            strategy = meta.get("strategy") or meta.get("mutation_name") or ""
            payload_hex = (meta.get("payload_hex", "") or "").lower()

            a = int(run_unit["a"])
            b = int(run_unit["b"])
            nbytes = b - a + 1 if a >= 0 and b >= 0 else 0
            w_bits = unit_width_bits(run_unit)

            mutation_meta = mutation_index.get(name, {}).get(payload_hex)
            requested_value_hex = meta.get("requested_value_hex")
            final_value_hex = meta.get("final_value_hex")
            strategy_group = str(meta.get("strategy_group", ""))
            if isinstance(mutation_meta, dict):
                strategy = meta.get("strategy") or mutation_meta.get("strategy", strategy)
                if not strategy_group:
                    strategy_group = str(mutation_meta.get("strategy_group", ""))
                requested_value_hex = requested_value_hex or mutation_meta.get("requested_value_hex")
                final_value_hex = final_value_hex or mutation_meta.get("final_value_hex")

            if final_value_hex is None and payload_hex and a >= 0 and b >= 0:
                payload_bytes = bytes.fromhex(payload_hex)
                final_value = read_value(payload_bytes, run_unit, "big")
                final_value_hex = format_value_hex(final_value, w_bits)
            if requested_value_hex is None:
                requested_value_hex = final_value_hex
            if not strategy_group:
                strategy_group = str(meta.get("strategy_group", ""))

            status = "ok"
            error = None
            health = None
            features = None
            preprocess = {
                "path": os.path.splitext(log_path)[0] + ".preprocessed.log",
                "taint_found": False,
                "taint_thread_id": None,
                "line_count": 0,
            }
            if not os.path.exists(log_path):
                status = "invalid"
                error = "log not found"
            elif not payload_hex or a < 0 or b < 0:
                status = "invalid"
                error = "missing payload_hex or field range"
            else:
                try:
                    preprocess = preprocess_log(log_path, seen_taint_thread_ids)
                    health = parse_log_health(preprocess["path"])
                    features = parse_log_features(preprocess["path"])
                except Exception as exc:
                    status = "parse_fail"
                    error = str(exc)

            field_runs.append({
                "run_id": os.path.splitext(entry)[0],
                "kind": "mutation",
                "strategy_group": strategy_group,
                "strategy": strategy,
                "payload_hex": payload_hex,
                "field_value": {
                    "requested_value_hex": requested_value_hex,
                    "final_value_hex": final_value_hex,
                    "w_bits": w_bits,
                },
                "artifact": {
                    "log_slice_path": relpath(log_path, outdir),
                    "preprocessed_log_path": relpath(preprocess["path"], outdir),
                    "log_slice_bytes": meta.get("slice_bytes", None),
                },
                "preprocess": preprocess,
                "parse_health": health or {
                    "instr_parsed": 0,
                    "bb_parsed": 0,
                    "branch_parsed": 0,
                    "bad_lines_dropped": 0,
                    "taint_found": False,
                    "slice_line_aligned": {"start_ok": False, "end_ok": False},
                },
                "status": status,
                "error": error,
            })

            if status == "ok" and base_features is not None and features is not None:
                metrics = diff_metrics(base_features, features, field_suffix_capacity=field_suffix_capacity)
                per_metric.append(metrics)
                per_mutation.append({
                    "run_id": os.path.splitext(entry)[0],
                    "strategy_group": strategy_group,
                    "strategy": strategy,
                    "final_value_hex": final_value_hex,
                    "metrics": metrics,
                })

        report["fields"].append({
            "field_id": field_id(field_unit),
            "field_dir": name,
            "field_kind": field_unit.get("kind", "byte"),
            "field_unit": field_unit,
            "nbytes": nbytes,
            "range": {"a": a, "b": b},
            "baseline_field_hex": baseline_field_hex,
            "baseline_suffix_capacity": field_suffix_capacity,
            "runs": field_runs,
            "diff": {
                "per_mutation": per_mutation,
                "summary": {
                    "valid_mutations": len(per_metric),
                    "metrics": summarize_metrics(per_metric),
                    "diagnostic": summarize_metric_diversity(per_metric),
                },
            },
        })

    write_json(os.path.join(outdir, "report.json"), report)

    compact_report = build_compact_report(report)
    compact_report_path = os.path.join(outdir, "report_compact.json")
    with open(compact_report_path, "w", encoding="utf-8") as f:
        json.dump(compact_report, f, indent=2, ensure_ascii=False, sort_keys=False)

    print(os.path.join(outdir, "report.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算差分指标，输出 report.json"
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="包含 mutations/ 的目录")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cmd_diff(args)
    except FileNotFoundError as exc:
        print_user_error(str(exc))
        sys.exit(1)
    except ValueError as exc:
        print_user_error(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        print_user_error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
