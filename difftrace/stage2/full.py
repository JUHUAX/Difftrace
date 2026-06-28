#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full.py - 完整流程脚本
一键完成：选包 → baseline → 字段划分 → 生成变体 → 发送变体 → 差分
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from common import (
    DEFAULT_ENABLE_TAINT,
    DEFAULT_OUTDIR,
    DEFAULT_PIN_BIN,
    DEFAULT_PIN_LOG,
    DEFAULT_PROTO,
    DEFAULT_RECV,
    DEFAULT_RECV_TIMEOUT,
    DEFAULT_SEED,
    DEFAULT_SERVER_ARGS,
    DEFAULT_SERVER_BIN,
    DEFAULT_TAINT_KILL_EXISTING,
    DEFAULT_TAINT_KILL_TIMEOUT,
    DEFAULT_TAINT_LOG_MARKER,
    DEFAULT_TAINT_PREFIX,
    DEFAULT_TAINT_SHUTDOWN_TIMEOUT,
    DEFAULT_TAINT_STARTUP_TIME,
    DEFAULT_TAINT_STDOUT_LOG,
    DEFAULT_TAINT_TOOL,
    DEFAULT_TAINT_WORKDIR,
    DEFAULT_TARGET_HOST,
    DEFAULT_TARGET_PORT,
    DEFAULT_WAIT_MS,
    TaintProcessManager,
    build_taint_command,
    bytes_to_hex,
    collect_payload_candidates_from_pcap,
    ensure_dir,
    hex_to_bytes,
    print_user_error,
    send_payload,
    sample_payload_from_hex,
    sample_payload_from_pcap,
    slice_log,
    summarize_pcap_payloads,
    wait_for_log_growth,
    wait_for_file,
    read_json,
    write_json,
    write_sample_metadata,
)

# 导入其他模块的函数
from fields import extract_fields_from_log
from mutate import (
    mutate_payload,
    extract_field_compare_evidence,
    extract_field_instruction_stats,
    build_constraint_profile,
    build_v3_grouped_candidates,
    build_v3_mutations_for_unit,
    format_value_hex,
    parse_value_hex,
)
from diff import (
    cmd_diff,
    diff_metrics,
    parse_log_features,
    parse_log_health,
    preprocess_log,
    summarize_metric_diversity,
)
from build_field_training_samples import build_field_training_artifacts
from analyze_bitfields_planA import FieldRef, analyze_logs
from field_units import (
    byte_ranges_from_units,
    field_id,
    make_byte_unit,
    merge_bitfields,
    normalize_unit,
    read_value,
    safe_field_dir,
    values_from_bit_constraints,
    width_bits as unit_width_bits,
)


class _Ansi:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"


_STAGE_COLOR = {
    "RUN": _Ansi.CYAN,
    "TAINT": _Ansi.YELLOW,
    "SAMPLE": _Ansi.BLUE,
    "FIELD": _Ansi.MAGENTA,
    "SEND": _Ansi.GREEN,
    "DIFF": _Ansi.WHITE,
    "SUMMARY": _Ansi.CYAN,
}

_KEYWORD_HIGHLIGHTS = (
    (re.compile(r"(失败|错误|崩溃|不可用|超时)"), _Ansi.RED),
    (re.compile(r"(重启|重试|告警|警告)"), _Ansi.YELLOW),
    (re.compile(r"(成功|完成|已启动|已停止|结束)"), _Ansi.GREEN),
    (re.compile(r"(累计耗时|用时|开始|发送)"), _Ansi.CYAN),
    (re.compile(r"(baseline)"), _Ansi.BLUE),
    (re.compile(r"(mutation)"), _Ansi.MAGENTA),
)


def _highlight_keywords(message: str) -> str:
    """为回显中的关键字添加颜色，增强可读性。"""
    highlighted = message
    for pattern, color in _KEYWORD_HIGHLIGHTS:
        highlighted = pattern.sub(lambda m: f"{color}{_Ansi.BOLD}{m.group(1)}{_Ansi.RESET}", highlighted)
    return highlighted


def _progress(stage: str, message: str, step: int = 0, total: int = 0) -> None:
    """打印实时进度回显。"""
    ts = time.strftime("%H:%M:%S")
    stage_upper = stage.upper()
    color = _STAGE_COLOR.get(stage_upper, _Ansi.WHITE)
    stage_tag = f"[{stage_upper:<6}]"
    step_tag = ""
    if total > 0:
        step_tag = f"[{step:>3}/{total:<3}]"
    prefix = f"{_Ansi.DIM}[full {ts}]{_Ansi.RESET}"
    stage_colored = f"{color}{_Ansi.BOLD}{stage_tag}{_Ansi.RESET}"
    step_colored = f"{_Ansi.DIM}{step_tag}{_Ansi.RESET}" if step_tag else ""
    print(f"{prefix}{stage_colored}{step_colored} {_highlight_keywords(message)}", flush=True)


def _format_hms(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS。"""
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_value_bits(value: int, width_bits: int) -> str:
    """将整数按定位宽格式化为二进制（按字节分组）。"""
    if width_bits <= 0:
        return "0"
    raw = format(int(value) & ((1 << width_bits) - 1), f"0{width_bits}b")
    chunks = [raw[i:i + 8] for i in range(0, len(raw), 8)]
    return " ".join(chunks)


def _format_value_hex(value: int, width_bits: int) -> str:
    """将整数按定位宽格式化为十六进制。"""
    if not isinstance(value, int):
        return str(value)
    if width_bits <= 0:
        return hex(int(value))
    mask = (1 << width_bits) - 1
    width_nibbles = max(1, (width_bits + 3) // 4)
    return f"0x{(int(value) & mask):0{width_nibbles}x}"


def _resolve_field_byteorder(server_bin: str) -> str:
    """根据被测程序解析字段写回字节序。"""
    name = os.path.basename(server_bin or "")
    if name == "opener_server":
        return "little"
    return "big"


def _taint_unavailable_reason(taint_manager: TaintProcessManager, pin_log: str) -> str:
    """返回可读的 taint 不可用原因说明。"""
    pin_log_exists = os.path.exists(pin_log)
    if taint_manager.process is None:
        if pin_log_exists:
            return "污点进程未运行（进程对象不存在），但 pin 日志文件存在。"
        return "污点进程未运行（进程对象不存在），且 pin 日志文件不存在。"

    exit_code = taint_manager.process.poll()
    if exit_code is None:
        # 理论上 _ensure_taint_ready 中不应进入该分支，仅作保护说明。
        return "污点进程状态异常：检查阶段判定不可用，但进程看起来仍在运行。"

    if exit_code < 0:
        base = f"污点进程已被信号终止（signal={-exit_code}），可能是 server/pin/工具崩溃。"
    else:
        base = f"污点进程已退出（exit_code={exit_code}），可能是 server/pin/工具崩溃。"

    if not pin_log_exists:
        base += " 同时 pin 日志文件当前不可见。"
    return base


def _payload_binary_lines(payload: bytes, bytes_per_line: int = 16) -> List[str]:
    """将 payload 格式化为十六进制多行文本。"""
    lines: List[str] = []
    for offset in range(0, len(payload), bytes_per_line):
        chunk = payload[offset:offset + bytes_per_line]
        hex_text = " ".join(format(b, "02x") for b in chunk)
        lines.append(f"offset 0x{offset:04x}: {hex_text}")
    return lines


def _describe_field_mutation_value(
    mut: dict,
    payload_mut: bytes,
    field_unit: dict,
    byteorder: str = "big",
) -> str:
    """生成字段变异值说明文本。"""
    unit = normalize_unit(field_unit)
    width_bits = unit_width_bits(unit)
    final_value = parse_value_hex(mut.get("final_value_hex"))
    if not isinstance(final_value, int):
        final_value = read_value(payload_mut, unit, byteorder)
    requested_value = parse_value_hex(mut.get("requested_value_hex"))
    if not isinstance(requested_value, int):
        requested_value = final_value
    return f"req={format_value_hex(requested_value, width_bits)} final={format_value_hex(final_value, width_bits)}"


def _emit_field_candidates(field_unit: dict, mutations: List[dict], title: str) -> None:
    """在回显中输出字段候选值摘要。"""
    unit = normalize_unit(field_unit)
    width_bits = unit_width_bits(unit)
    label = field_id(unit)
    _progress("field", f"字段 {label} {title}（共 {len(mutations)} 个）")
    by_group = {}
    for mut in mutations:
        group = str(mut.get("strategy_group") or "legacy")
        value = parse_value_hex(mut.get("final_value_hex"))
        by_group.setdefault(group, []).append(format_value_hex(value or 0, width_bits))
    for group in sorted(by_group):
        _progress("field", f"  {group}: {len(by_group[group])} candidates [{', '.join(by_group[group])}]")


def _append_strategy_log(
    log_path: str,
    field_unit: dict,
    round_id: int,
    profile: dict,
    mutations: List[dict],
    evidence_details: Optional[List[dict]] = None,
    note: str = "",
) -> None:
    """追加精简 mutation strategy 日志。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    unit = normalize_unit(field_unit)
    width_bits = unit_width_bits(unit)
    label = field_id(unit)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] field={label} kind={unit.get('kind')} round={round_id}\n")
        if unit.get("kind") == "bit":
            f.write(f"  parent={unit.get('parent_repr')} bits={unit.get('bits')}\n")
        mode = profile.get("mode", "fallback")
        evidence_count = int(profile.get("evidence_count", 0))
        range_info = profile.get("range")
        f.write(f"  decision: mode={mode}, evidence_count={evidence_count}\n")
        importance = profile.get("importance")
        if importance:
            f.write(f"  importance: {importance}\n")
        stats = profile.get("instruction_stats")
        if isinstance(stats, dict):
            f.write(
                "  instruction_stats: "
                f"cmp={int(stats.get('compare_count', 0))}, "
                f"compute={int(stats.get('compute_count', 0))}, "
                f"move={int(stats.get('move_count', 0))}, "
                f"other={int(stats.get('other_count', 0))}\n"
            )
        budget = profile.get("budget")
        if isinstance(budget, dict):
            f.write(
                "  budget: "
                f"group_topk={int(budget.get('group_topk', 0))}, "
                f"round2={'yes' if budget.get('allow_round2') else 'no'}, "
                f"early_stop_min={int(budget.get('early_stop_min', 0))}\n"
            )
        if isinstance(range_info, dict):
            l_val = range_info.get("l")
            u_val = range_info.get("u")
            f.write(
                "  inferred_range: "
                f"[{_format_value_hex(l_val, width_bits)}, {_format_value_hex(u_val, width_bits)}]\n"
            )
        evidence_values = profile.get("evidence_values", [])
        evidence_hex = ", ".join(_format_value_hex(v, width_bits) for v in evidence_values)
        f.write(f"  evidence_values: [{evidence_hex}]\n")
        if note:
            f.write(f"  note: {note}\n")
        grouped = {}
        for mut in mutations:
            group = str(mut.get("strategy_group") or "legacy")
            strategy = str(mut.get("strategy", mut.get("name", "")))
            final_value = parse_value_hex(mut.get("final_value_hex"))
            grouped.setdefault(group, []).append((strategy, final_value or 0))
        for group in sorted(grouped):
            f.write(f"  group={group} count={len(grouped[group])}\n")
            for strategy, final_value in grouped[group]:
                f.write(f"    {strategy}={_format_value_hex(final_value, width_bits)}\n")
        f.write("\n")


def _infer_field_importance(profile: dict, instr_stats: dict) -> str:
    """依据约束和指令类型统计估计字段重要性。"""
    evidence_count = int(profile.get("evidence_count", 0))
    compare_count = int(instr_stats.get("compare_count", 0))
    compute_count = int(instr_stats.get("compute_count", 0))
    if evidence_count > 0 or compare_count > 0:
        return "high"
    if compute_count > 0:
        return "medium"
    return "low"


def _field_budget_for_importance(
    importance: str,
    default_group_topk: int,
    cg_rounds: int,
) -> dict:
    """按字段重要性分配 mutation 预算。"""
    if importance == "high":
        return {
            "group_topk": max(1, int(default_group_topk)),
            "allow_round2": cg_rounds > 1,
            "early_stop_min": 0,
        }
    if importance == "medium":
        return {
            "group_topk": min(max(1, int(default_group_topk)), 4),
            "allow_round2": False,
            "early_stop_min": 6,
        }
    return {
        "group_topk": min(max(1, int(default_group_topk)), 3),
        "allow_round2": False,
        "early_stop_min": 4,
    }


def _should_early_stop_field(metric_list: List[dict], early_stop_min: int) -> bool:
    """基于当前字段已观测到的差分指标决定是否早停。"""
    if early_stop_min <= 0 or len(metric_list) < early_stop_min:
        return False
    stats = summarize_metric_diversity(metric_list)
    return int(stats.get("low_diversity", 0)) == 1


def _build_unified_field_units(byte_fields: List[tuple], baseline_log: str, outdir: str) -> List[dict]:
    """Run byte field extraction + bitfield recovery and return unified FieldUnits."""
    ensure_dir(outdir)
    byte_units = [make_byte_unit(a, b) for a, b in byte_fields]
    byte_fields_json = {"fields": byte_units}
    byte_fields_path = os.path.join(outdir, "fields.byte.json")
    write_json(byte_fields_path, byte_fields_json)

    refs = [
        FieldRef(
            field_id=str(unit["repr"]),
            byte_offset_start=int(unit["a"]),
            byte_offset_end=int(unit["b"]),
            bit_width=int(unit["width_bits"]),
        )
        for unit in byte_units
    ]
    bitfield_result = analyze_logs(refs, [baseline_log])
    bitfields_path = os.path.join(outdir, "bitfields.json")
    write_json(bitfields_path, bitfield_result)
    return merge_bitfields(byte_units, bitfield_result)


def _collect_out_of_bounds_fields(payload_len: int, byte_fields: List[tuple], field_units: List[dict]) -> List[str]:
    """收集超出 payload 长度的字段描述。"""
    problems: List[str] = []

    for a, b in byte_fields:
        a_i = int(a)
        b_i = int(b)
        if a_i < 0 or b_i >= payload_len or a_i > b_i:
            label = f"{a_i}" if a_i == b_i else f"{a_i},{b_i}"
            problems.append(f"byte:{label}")

    for unit in field_units:
        normalized = normalize_unit(unit)
        a_i = int(normalized["a"])
        b_i = int(normalized["b"])
        if a_i < 0 or b_i >= payload_len or a_i > b_i:
            problems.append(f"{normalized.get('kind', 'field')}:{field_id(normalized)}")

    deduped: List[str] = []
    seen = set()
    for item in problems:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _baseline_retry_reason(
    payload_len: int,
    baseline_preprocess: dict,
    baseline_preprocess_check: dict,
    baseline_features: dict,
    byte_fields: List[tuple],
    field_units: List[dict],
) -> Optional[str]:
    """判断 baseline 是否异常；异常时返回单一原因描述。"""
    invalid_fields = _collect_out_of_bounds_fields(payload_len, byte_fields, field_units)
    if invalid_fields:
        return (
            "field range out of payload bounds: "
            f"payload_len=0x{payload_len:x}, invalid={', '.join(invalid_fields[:8])}"
        )
    if not baseline_preprocess.get("taint_found", False):
        return "no taint found in baseline log"
    if bool(baseline_preprocess_check.get("empty_file", False)):
        return "baseline preprocessed log is empty"
    if int(baseline_preprocess_check.get("instr_parsed", 0)) <= 0:
        return "baseline preprocessed log has no Instruction line"
    if int(baseline_features.get("instr_count", 0)) <= 0:
        return "no instruction parsed from baseline log"
    if not byte_fields:
        return "no fields extracted from baseline log"
    return None


def _inspect_preprocessed_log(preprocessed_path: str) -> dict:
    empty_file = (not os.path.exists(preprocessed_path)) or os.path.getsize(preprocessed_path) == 0
    health = parse_log_health(preprocessed_path) if os.path.exists(preprocessed_path) else {
        "instr_parsed": 0,
        "bb_parsed": 0,
        "branch_parsed": 0,
        "bad_lines_dropped": 0,
        "taint_found": False,
        "slice_line_aligned": {"start_ok": False, "end_ok": False},
    }
    instr_parsed = int(health.get("instr_parsed") or 0)
    return {
        "preprocessed_path": preprocessed_path,
        "exists": os.path.exists(preprocessed_path),
        "empty_file": empty_file,
        "taint_found": bool(health.get("taint_found", False)),
        "instr_parsed": instr_parsed,
        "bb_parsed": int(health.get("bb_parsed") or 0),
        "branch_parsed": int(health.get("branch_parsed") or 0),
        "bad_lines_dropped": int(health.get("bad_lines_dropped") or 0),
        "has_instruction": instr_parsed > 0,
    }


def _describe_preprocessed_log_issue(check: dict) -> Optional[str]:
    if not check.get("exists", False):
        return "preprocessed log missing"
    if check.get("empty_file", False):
        return "preprocessed log is empty"
    if not check.get("taint_found", False):
        return "preprocessed log has no taint"
    if not check.get("has_instruction", False):
        return "preprocessed log has no Instruction line"
    return None


def _record_preprocessed_check(meta_path: str, check: dict) -> None:
    if not os.path.exists(meta_path):
        return
    try:
        meta = read_json(meta_path)
    except Exception:
        return
    if not isinstance(meta, dict):
        return
    meta["preprocessed_check"] = check
    issue = _describe_preprocessed_log_issue(check)
    if issue:
        meta["preprocessed_issue"] = issue
    write_json(meta_path, meta)


def _run_baseline_with_field_sanity(
    args: argparse.Namespace,
    taint_manager: TaintProcessManager,
    payload: bytes,
    proto: str,
    outdir: str,
    mutations_root: str,
) -> dict:
    """发送 baseline，并在明显异常时重启污点分析后重发。"""
    baseline_log = os.path.join(mutations_root, "baseline.log")
    baseline_meta_path = os.path.join(mutations_root, "baseline.json")
    payload_len = len(payload)
    max_attempts = max(5, int(args.packet_retry_on_taint_fail) + 1)
    last_error_reason = ""

    for attempt in range(1, max_attempts + 1):
        marker_label = "baseline" if attempt == 1 else f"baseline 重试{attempt - 1}"
        baseline_send = _send_and_slice_with_recovery(
            args=args,
            taint_manager=taint_manager,
            marker_label=marker_label,
            payload=payload,
            proto=proto,
            log_path=baseline_log,
        )

        baseline_meta = {
            "role": "baseline",
            "proto": proto,
            "target": {"host": args.target_host, "port": args.target_port},
            "payload_hex": bytes_to_hex(payload),
            "pin_log": args.pin_log,
            "log_slice_path": baseline_log,
            "slice_bytes": baseline_send["slice_len"],
            "size_before": baseline_send["size_before"],
            "size_after": baseline_send["size_after"],
            "time_start": baseline_send["time_start"],
            "time_end": baseline_send["time_end"],
            "duration_sec": baseline_send["time_end"] - baseline_send["time_start"],
            "send_info": baseline_send["send_info"],
        }
        write_json(baseline_meta_path, baseline_meta)

        baseline_preprocess = preprocess_log(baseline_log, seen_taint_thread_ids=set())
        baseline_preprocess_check = _inspect_preprocessed_log(baseline_preprocess["path"])
        baseline_features = parse_log_features(baseline_preprocess["path"])
        byte_fields = extract_fields_from_log(baseline_meta["log_slice_path"])
        field_units = _build_unified_field_units(byte_fields, baseline_meta["log_slice_path"], outdir)
        error_reason = _baseline_retry_reason(
            payload_len,
            baseline_preprocess,
            baseline_preprocess_check,
            baseline_features,
            byte_fields,
            field_units,
        )
        baseline_meta["preprocessed_check"] = baseline_preprocess_check
        write_json(baseline_meta_path, baseline_meta)

        if error_reason is None:
            if taint_manager is not None:
                taint_manager.reset_restart_backoff()
            return {
                "baseline_send": baseline_send,
                "baseline_meta": baseline_meta,
                "baseline_preprocess": baseline_preprocess,
                "baseline_features": baseline_features,
                "byte_fields": byte_fields,
                "field_units": field_units,
            }

        last_error_reason = error_reason
        _progress("field", f"baseline 异常检测失败：{error_reason}")
        if attempt >= max_attempts:
            break
        if taint_manager is None:
            break
        _progress("taint", f"baseline 异常，重启污点分析后重发 baseline（第 {attempt + 1}/{max_attempts} 次）")
        _restart_taint_process(
            args,
            taint_manager,
            f"{error_reason}，准备重启并重发 baseline",
        )

    raise RuntimeError(
        "baseline sanity check failed\n"
        f"payload_len={payload_len}\n"
        f"reason={last_error_reason}"
    )


def _load_frozen_baseline(
    payload: bytes,
    proto: str,
    outdir: str,
    mutations_root: str,
    frozen_baseline_dir: str,
) -> dict:
    """Stage an existing baseline trace and derive Stage 2 fields from it."""
    source_dir = os.path.abspath(frozen_baseline_dir)
    source_raw_log = os.path.join(source_dir, "trace.log")
    source_preprocessed_log = os.path.join(source_dir, "trace.preprocessed.log")
    if not os.path.exists(source_preprocessed_log):
        raise FileNotFoundError(f"frozen baseline preprocessed log not found: {source_preprocessed_log}")

    baseline_log = os.path.join(mutations_root, "baseline.log")
    baseline_preprocessed_log = os.path.join(mutations_root, "baseline.preprocessed.log")
    baseline_meta_path = os.path.join(mutations_root, "baseline.json")
    ensure_dir(mutations_root)

    if os.path.exists(source_raw_log):
        shutil.copy2(source_raw_log, baseline_log)
    else:
        # Keep the historical artifact contract even if an audited raw trace is unavailable.
        shutil.copy2(source_preprocessed_log, baseline_log)
    shutil.copy2(source_preprocessed_log, baseline_preprocessed_log)

    baseline_preprocess_check = _inspect_preprocessed_log(baseline_preprocessed_log)
    baseline_features = parse_log_features(baseline_preprocessed_log)
    extracted_byte_fields = extract_fields_from_log(baseline_preprocessed_log)
    byte_fields = [
        (int(a), int(b))
        for a, b in extracted_byte_fields
        if 0 <= int(a) <= int(b) < len(payload)
    ]
    ignored_byte_fields = [
        (int(a), int(b))
        for a, b in extracted_byte_fields
        if not (0 <= int(a) <= int(b) < len(payload))
    ]
    if ignored_byte_fields:
        labels = ", ".join(
            str(a) if a == b else f"{a},{b}"
            for a, b in ignored_byte_fields
        )
        _progress(
            "field",
            "冻结 baseline 忽略超出 payload 范围的字段候选："
            f"payload_len={len(payload)}，ignored=[{labels}]",
        )
    field_units = _build_unified_field_units(byte_fields, baseline_preprocessed_log, outdir)
    error_reason = _baseline_retry_reason(
        len(payload),
        {
            "path": baseline_preprocessed_log,
            "taint_found": baseline_preprocess_check["taint_found"],
            "taint_thread_id": None,
            "line_count": baseline_preprocess_check["instr_parsed"],
        },
        baseline_preprocess_check,
        baseline_features,
        byte_fields,
        field_units,
    )
    if error_reason is not None:
        raise RuntimeError(
            "frozen baseline sanity check failed\n"
            f"source={source_dir}\n"
            f"payload_len={len(payload)}\n"
            f"reason={error_reason}"
        )

    baseline_meta = {
        "role": "baseline",
        "source": "frozen_preprocessed_log",
        "frozen_baseline_dir": source_dir,
        "proto": proto,
        "payload_hex": bytes_to_hex(payload),
        "log_slice_path": baseline_log,
        "preprocessed_log_path": baseline_preprocessed_log,
        "slice_bytes": os.path.getsize(baseline_log),
        "preprocessed_check": baseline_preprocess_check,
        "ignored_out_of_bounds_byte_fields": [
            {"a": a, "b": b}
            for a, b in ignored_byte_fields
        ],
    }
    write_json(baseline_meta_path, baseline_meta)
    return {
        "baseline_send": None,
        "baseline_meta": baseline_meta,
        "baseline_preprocess": {
            "path": baseline_preprocessed_log,
            "taint_found": baseline_preprocess_check["taint_found"],
            "taint_thread_id": None,
            "line_count": baseline_preprocess_check["instr_parsed"],
        },
        "baseline_features": baseline_features,
        "byte_fields": byte_fields,
        "field_units": field_units,
        "baseline_analysis_log": baseline_preprocessed_log,
    }


def _select_target_units(units: List[dict], field_spec: str) -> List[dict]:
    """Select one or all FieldUnits. field_spec may match repr or byte range."""
    if not field_spec:
        return list(units)
    normalized_spec = field_spec.strip()
    for unit in units:
        if normalized_spec == field_id(unit):
            return [unit]
        if unit.get("kind") == "byte":
            a = int(unit["a"])
            b = int(unit["b"])
            if normalized_spec in {str(a), f"{a},{b}", f"{a}-{b}", f"{a}_{b}"}:
                return [unit]
    raise RuntimeError(f"field not found: {field_spec}")


def should_enable_taint(args: argparse.Namespace) -> bool:
    """判断是否启用污点分析"""
    enable = bool(args.taint)
    if getattr(args, "no_taint", False):
        enable = False
    return enable


def resolve_taint_command(args: argparse.Namespace) -> str:
    """解析污点分析命令"""
    if args.taint_command:
        return args.taint_command
    return build_taint_command(
        args.taint_prefix,
        args.pin_bin,
        args.taint_tool,
        args.pin_log,
        args.server_bin,
        args.server_args,
    )


def resolve_server_name(args: argparse.Namespace) -> str:
    """解析 server 名称"""
    if args.taint_server_name:
        return args.taint_server_name
    if args.server_bin:
        return os.path.basename(args.server_bin)
    return ""


def _restart_taint_process(
    args: argparse.Namespace,
    taint_manager: TaintProcessManager,
    reason: str,
) -> None:
    """按配置重启污点分析进程。"""
    _progress("taint", f"污点进程重启：{reason}")
    max_restarts = max(1, int(args.taint_max_restarts))
    errors = []
    startup_wait = float(args.taint_startup_time) + 0.5 * float(getattr(taint_manager, "consecutive_restart_count", 0))
    for attempt in range(1, max_restarts + 1):
        _progress("taint", f"尝试重启污点进程（等待 {startup_wait:.1f}s）", attempt, max_restarts)
        try:
            taint_manager.restart(startup_wait=startup_wait)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"attempt {attempt}: restart exception: {exc}")
            if args.taint_restart_delay > 0:
                time.sleep(args.taint_restart_delay)
            continue

        if taint_manager.process is None or taint_manager.process.poll() is not None:
            errors.append(f"attempt {attempt}: process exited after restart")
            if args.taint_restart_delay > 0:
                time.sleep(args.taint_restart_delay)
            continue

        if not wait_for_file(args.pin_log, args.taint_startup_time):
            errors.append(f"attempt {attempt}: pin log not visible: {args.pin_log}")
            if args.taint_restart_delay > 0:
                time.sleep(args.taint_restart_delay)
            continue
        taint_manager.consecutive_restart_count = int(getattr(taint_manager, "consecutive_restart_count", 0)) + 1
        _progress("taint", f"污点进程重启成功（本次等待 {startup_wait:.1f}s）")
        return

    detail = "\n".join(errors[-5:])
    raise RuntimeError(
        "污点分析自动重启失败。\n"
        f"reason={reason}\n"
        f"max_restarts={max_restarts}\n"
        f"{_build_taint_startup_error(args, taint_manager)}\n"
        f"restart_errors:\n{detail}"
    )


def _ensure_taint_ready(
    args: argparse.Namespace,
    taint_manager: TaintProcessManager,
    reason: str,
) -> None:
    """确保污点分析进程可用，必要时按配置自动重启。"""
    if taint_manager is None:
        return
    if taint_manager.is_alive():
        return
    reason_text = _taint_unavailable_reason(taint_manager, args.pin_log)
    _progress("taint", f"检测到污点进程不可用，触发重启检查。{reason_text}")
    if not args.taint_auto_restart:
        raise RuntimeError(
            "污点分析进程已退出，且未启用自动重启。\n"
            f"reason={reason}\n"
            f"diagnosis={reason_text}\n"
            f"{_build_taint_startup_error(args, taint_manager)}"
        )
    _restart_taint_process(args, taint_manager, f"{reason}; {reason_text}")


def _send_and_slice_with_recovery(
    args: argparse.Namespace,
    taint_manager: TaintProcessManager,
    marker_label: str,
    payload: bytes,
    proto: str,
    log_path: str,
) -> dict:
    """发送 payload 并切片；若污点进程崩溃则自动重启并重试当前发送。"""
    max_attempts = max(1, int(args.packet_retry_on_taint_fail) + 1)
    for attempt in range(1, max_attempts + 1):
        _progress("send", f"{marker_label}: 发送尝试", attempt, max_attempts)
        _ensure_taint_ready(args, taint_manager, f"{marker_label} 前检查")
        if taint_manager and args.taint_log_marker:
            marker = f"发送 {marker_label}"
            if attempt > 1:
                marker += f" 重试{attempt - 1}"
            taint_manager.log_packet_marker(marker)

        size_before = os.path.getsize(args.pin_log)
        t_start = time.time()
        try:
            send_info = send_payload(
                payload=payload,
                proto=proto,
                host=args.target_host,
                port=args.target_port,
                recv=args.recv,
                recv_timeout=args.recv_timeout,
            )
        except Exception as exc:
            if taint_manager is not None and attempt < max_attempts and args.taint_auto_restart:
                _progress("taint", f"{marker_label}: 发送阶段失败，准备重启后重试。{type(exc).__name__}: {exc}")
                _restart_taint_process(
                    args,
                    taint_manager,
                    f"{marker_label} 发送阶段失败，准备重试; {type(exc).__name__}: {exc}",
                )
                continue
            raise RuntimeError(
                "payload send failed.\n"
                f"label={marker_label}\n"
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        size_after = wait_for_log_growth(args.pin_log, size_before, args.wait_ms)
        t_end = time.time()

        slice_len = slice_log(args.pin_log, size_before, size_after, log_path)
        if slice_len > 0:
            if taint_manager is not None:
                taint_manager.reset_restart_backoff()
            _progress("send", f"{marker_label}: 切片成功，{slice_len} bytes")
            return {
                "send_info": send_info,
                "size_before": size_before,
                "size_after": size_after,
                "time_start": t_start,
                "time_end": t_end,
                "slice_len": slice_len,
            }

        # A) 对空切片先做一次额外等待，再重新切片，避免把少数慢路径一律升级成重启。
        _progress("send", f"{marker_label}: 首次切片为空，额外等待 {args.wait_ms}ms 后重试切片")
        size_after_retry = wait_for_log_growth(args.pin_log, size_after, args.wait_ms)
        t_end_retry = time.time()
        if size_after_retry > size_after:
            slice_len_retry = slice_log(args.pin_log, size_before, size_after_retry, log_path)
            if slice_len_retry > 0:
                if taint_manager is not None:
                    taint_manager.reset_restart_backoff()
                _progress("send", f"{marker_label}: 额外等待后切片成功，{slice_len_retry} bytes")
                return {
                    "send_info": send_info,
                    "size_before": size_before,
                    "size_after": size_after_retry,
                    "time_start": t_start,
                    "time_end": t_end_retry,
                    "slice_len": slice_len_retry,
                }

        taint_dead = taint_manager is not None and not taint_manager.is_alive()
        if taint_dead and attempt < max_attempts and args.taint_auto_restart:
            reason_text = _taint_unavailable_reason(taint_manager, args.pin_log)
            _progress("taint", f"{marker_label}: 日志切片为空，且污点进程不可用。{reason_text}")
            _restart_taint_process(
                args,
                taint_manager,
                f"{marker_label} 切片为空且进程退出，准备重试; {reason_text}",
            )
            continue
        if taint_dead:
            raise RuntimeError(
                "log slice is empty and taint process exited.\n"
                f"label={marker_label}\n"
                f"{_build_taint_startup_error(args, taint_manager)}"
            )

        # B) 进程还活着但空切片，优先视为时序/慢路径问题；允许重启后重发当前 mutation。
        if taint_manager is not None and attempt < max_attempts and args.taint_auto_restart:
            _progress("taint", f"{marker_label}: 额外等待后仍为空切片，重启污点分析并重试当前 mutation")
            _restart_taint_process(
                args,
                taint_manager,
                f"{marker_label} 切片为空，额外等待后仍无日志，准备重试",
            )
            continue

        raise RuntimeError("log slice is empty; extra wait and retry exhausted")

    raise RuntimeError(f"发送失败：{marker_label}")


def _record_failed_mutation(
    args: argparse.Namespace,
    field_unit: dict,
    field_dir: str,
    mut: dict,
    idx: int,
    proto: str,
    error: Exception,
) -> None:
    """记录失败 mutation 的最小元信息，供后续排查。"""
    name_part = mut.get("name") or f"mut{idx}"
    meta_path = os.path.join(field_dir, f"{idx:03d}_{name_part}.failed.json")
    meta = {
        "field": field_unit,
        "mutation_index": idx,
        "mutation_name": name_part,
        "strategy_group": mut.get("strategy_group", ""),
        "strategy": mut.get("strategy", ""),
        "requested_value_hex": mut.get("requested_value_hex"),
        "final_value_hex": mut.get("final_value_hex"),
        "proto": proto,
        "target": {"host": args.target_host, "port": args.target_port},
        "payload_hex": str(mut.get("payload_hex", "")),
        "result_source": "failed",
        "error": f"{type(error).__name__}: {error}",
        "timestamp": time.time(),
    }
    write_json(meta_path, meta)
    mut["result_source"] = "failed"
    mut["error"] = meta["error"]


def _materialize_mapped_log(src_log_path: str, dst_log_path: str) -> None:
    """为映射复用的 mutation 生成独立 log 文件名，内容复用已执行切片。"""
    if os.path.exists(dst_log_path):
        os.remove(dst_log_path)
    try:
        os.link(src_log_path, dst_log_path)
    except OSError:
        shutil.copyfile(src_log_path, dst_log_path)


def _run_or_map_mutation(
    args: argparse.Namespace,
    taint_manager: TaintProcessManager,
    field_unit: dict,
    field_label: str,
    field_dir: str,
    mut: dict,
    idx: int,
    total: int,
    proto: str,
    field_byteorder: str,
    executed_by_payload: dict,
) -> str:
    """执行一次 mutation，或将其映射到已执行的相同 payload。"""
    name_part = mut.get("name") or f"mut{idx}"
    payload_hex = str(mut["payload_hex"]).lower()
    payload_mut = hex_to_bytes(mut["payload_hex"])
    log_path = os.path.join(field_dir, f"{idx:03d}_{name_part}.log")
    meta_path = os.path.join(field_dir, f"{idx:03d}_{name_part}.json")

    _progress("send", f"字段 {field_label} mutation: {name_part}", idx, total)
    _progress(
        "field",
        f"字段 {field_label} 变异值: "
        f"{_describe_field_mutation_value(mut, payload_mut, field_unit, field_byteorder)}",
    )

    reused = executed_by_payload.get(payload_hex)
    if reused is not None:
        _progress(
            "send",
            f"{field_label} {name_part}: 映射复用 {reused['mutation_name']} 的重放结果",
        )
        _materialize_mapped_log(reused["log_path"], log_path)
        meta = {
            "field": field_unit,
            "mutation_index": idx,
            "mutation_name": name_part,
            "strategy_group": mut.get("strategy_group", ""),
            "strategy": mut.get("strategy", ""),
            "requested_value_hex": mut.get("requested_value_hex"),
            "final_value_hex": mut.get("final_value_hex"),
            "proto": proto,
            "target": {"host": args.target_host, "port": args.target_port},
            "payload_hex": bytes_to_hex(payload_mut),
            "pin_log": args.pin_log,
            "log_slice_path": log_path,
            "slice_bytes": reused["slice_bytes"],
            "size_before": reused["size_before"],
            "size_after": reused["size_after"],
            "time_start": reused["time_start"],
            "time_end": reused["time_end"],
            "duration_sec": reused["duration_sec"],
            "send_info": reused["send_info"],
            "result_source": "mapped",
            "mapped_from": reused["mutation_name"],
            "mapped_from_log": reused["log_path"],
        }
        write_json(meta_path, meta)
        mut["result_source"] = "mapped"
        mut["mapped_from"] = reused["mutation_name"]
        return log_path

    max_attempts = max(1, int(args.packet_retry_on_taint_fail) + 1)
    last_issue: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        mutation_send = _send_and_slice_with_recovery(
            args=args,
            taint_manager=taint_manager,
            marker_label=f"{field_label} mut{idx}",
            payload=payload_mut,
            proto=proto,
            log_path=log_path,
        )
        meta = {
            "field": field_unit,
            "mutation_index": idx,
            "mutation_name": name_part,
            "strategy_group": mut.get("strategy_group", ""),
            "strategy": mut.get("strategy", ""),
            "requested_value_hex": mut.get("requested_value_hex"),
            "final_value_hex": mut.get("final_value_hex"),
            "proto": proto,
            "target": {"host": args.target_host, "port": args.target_port},
            "payload_hex": bytes_to_hex(payload_mut),
            "pin_log": args.pin_log,
            "log_slice_path": log_path,
            "slice_bytes": mutation_send["slice_len"],
            "size_before": mutation_send["size_before"],
            "size_after": mutation_send["size_after"],
            "time_start": mutation_send["time_start"],
            "time_end": mutation_send["time_end"],
            "duration_sec": mutation_send["time_end"] - mutation_send["time_start"],
            "send_info": mutation_send["send_info"],
            "result_source": "executed",
        }
        write_json(meta_path, meta)

        preprocess = preprocess_log(log_path)
        preprocess_check = _inspect_preprocessed_log(preprocess["path"])
        _record_preprocessed_check(meta_path, preprocess_check)
        preprocess_issue = _describe_preprocessed_log_issue(preprocess_check)
        if preprocess_issue is None:
            executed_by_payload[payload_hex] = {
                "mutation_name": name_part,
                "log_path": log_path,
                "slice_bytes": mutation_send["slice_len"],
                "size_before": mutation_send["size_before"],
                "size_after": mutation_send["size_after"],
                "time_start": mutation_send["time_start"],
                "time_end": mutation_send["time_end"],
                "duration_sec": mutation_send["time_end"] - mutation_send["time_start"],
                "send_info": mutation_send["send_info"],
            }
            mut["result_source"] = "executed"
            return log_path

        last_issue = preprocess_issue
        _progress("send", f"{field_label} {name_part}: {preprocess_issue}")
        if taint_manager is not None and attempt < max_attempts and args.taint_auto_restart:
            _progress("taint", f"{field_label} {name_part}: 预处理日志异常，重启污点分析并重发当前 mutation")
            _restart_taint_process(
                args,
                taint_manager,
                f"{field_label} {name_part} 预处理日志异常，准备重试; {preprocess_issue}",
            )
            continue
        raise RuntimeError(f"mutation preprocess validation failed: {preprocess_issue}")

    raise RuntimeError(f"mutation preprocess validation failed: {last_issue or 'unknown issue'}")


def _run_single_sample_pipeline(
    args: argparse.Namespace,
    outdir: str,
    payload: bytes,
    proto: str,
    sample_meta: dict,
    taint_manager: TaintProcessManager,
    run_start_ts: float,
) -> None:
    """执行单个样本的完整流程。"""
    ensure_dir(outdir)
    sample_start_ts = time.time()
    field_byteorder = _resolve_field_byteorder(args.server_bin)
    _progress("sample", f"开始样本流程，outdir={outdir}")
    if args.clean:
        _progress("sample", "清理旧产物")
        _clean_full_artifacts(outdir)

    write_sample_metadata(outdir, sample_meta)
    _progress("sample", "写入 sample.json 完成")
    strategy_log_path = os.path.join(outdir, "mutation_strategy.log")
    with open(strategy_log_path, "w", encoding="utf-8") as f:
        f.write("# Mutation Strategy Log\n")
        f.write(f"# sample_outdir={outdir}\n\n")

    # 2) Baseline run for slicing
    mutations_root = os.path.join(outdir, "mutations")
    ensure_dir(mutations_root)
    baseline_log = os.path.join(mutations_root, "baseline.log")

    frozen_baseline_dir = str(sample_meta.get("frozen_baseline_dir") or "")
    if not frozen_baseline_dir and not os.path.exists(args.pin_log):
        raise FileNotFoundError(f"pin log not found: {args.pin_log}")

    _progress("send", f"baseline payload 十六进制（len=0x{len(payload):x} bytes）")
    for line in _payload_binary_lines(payload):
        _progress("send", line)

    if frozen_baseline_dir:
        _progress("send", f"复用冻结 baseline：{frozen_baseline_dir}")
        baseline_result = _load_frozen_baseline(
            payload=payload,
            proto=proto,
            outdir=outdir,
            mutations_root=mutations_root,
            frozen_baseline_dir=frozen_baseline_dir,
        )
    else:
        baseline_result = _run_baseline_with_field_sanity(
            args=args,
            taint_manager=taint_manager,
            payload=payload,
            proto=proto,
            outdir=outdir,
            mutations_root=mutations_root,
        )
    baseline_send = baseline_result["baseline_send"]
    baseline_meta = baseline_result["baseline_meta"]
    baseline_preprocess = baseline_result["baseline_preprocess"]
    baseline_features = baseline_result["baseline_features"]
    byte_fields = baseline_result["byte_fields"]
    field_units = baseline_result["field_units"]
    baseline_log = baseline_result.get("baseline_analysis_log", baseline_log)
    if frozen_baseline_dir:
        _progress("send", "冻结 baseline staging 完成")
    else:
        _progress("send", "baseline 发送与切片完成")

    # 3) Extract byte fields, recover bit subfields, and build unified FieldUnits.
    if not byte_fields:
        raise RuntimeError("no fields extracted from baseline log")
    _progress("field", f"字节字段提取完成，共 {len(byte_fields)} 个字段")
    _progress("field", f"统一字段生成完成，共 {len(field_units)} 个字段单元")
    fields_json = {"fields": field_units}
    fields_json["baseline_packet_index"] = sample_meta.get("index")
    write_json(os.path.join(outdir, "fields.json"), fields_json)
    byte_field_ranges = byte_ranges_from_units([make_byte_unit(a, b) for a, b in byte_fields])

    # 4) Select target fields and generate mutations
    target_fields = _select_target_units(field_units, args.field)

    mutations_path = os.path.join(outdir, "mutations.json")
    selections = []
    if os.path.exists(mutations_path):
        if os.path.getsize(mutations_path) != 0:
            existing_data = read_json(mutations_path)
            if isinstance(existing_data, list):
                selections.extend(existing_data)
            elif isinstance(existing_data, dict):
                selections.append(existing_data)
            else:
                raise RuntimeError("mutations.json format is invalid")

    # 5) Run mutation payloads for each target field
    for field_idx, field_unit in enumerate(target_fields, start=1):
        field_unit = normalize_unit(field_unit)
        a = int(field_unit["a"])
        b = int(field_unit["b"])
        label = field_id(field_unit)
        field_t_start = time.time()
        _progress("field", f"处理字段 {label}", field_idx, len(target_fields))
        field_seed = args.seed + field_idx - 1
        constraint_values = values_from_bit_constraints(field_unit) if field_unit.get("kind") == "bit" else None
        instr_stats = extract_field_instruction_stats(
            baseline_log,
            a,
            b,
            field_ranges=byte_field_ranges,
        )
        width_bits = unit_width_bits(field_unit)
        if field_unit.get("kind") == "bit":
            evidence_round1 = []
            e0_for_round1 = list(constraint_values or [])
        else:
            evidence_round1 = (
                extract_field_compare_evidence(
                    baseline_log, a, b, width_bits,
                    source_bin=args.server_bin,
                    field_ranges=byte_field_ranges,
                )
                if args.constraint_guided else []
            )
            e0_for_round1 = [int(item["value"]) for item in evidence_round1]
        profile_round1 = build_constraint_profile(width_bits, e0_for_round1)
        importance = _infer_field_importance(profile_round1, instr_stats)
        field_budget = _field_budget_for_importance(importance, args.group_topk, args.cg_rounds)
        profile_round1["importance"] = importance
        profile_round1["instruction_stats"] = instr_stats
        profile_round1["budget"] = field_budget
        mutations = mutate_payload(
            payload, a, b, field_seed,
            constraint_guided=args.constraint_guided,
            baseline_log_path=baseline_log,
            field_ranges=byte_field_ranges,
            byteorder=field_byteorder,
            group_topk=field_budget["group_topk"],
            group_min_candidates=args.group_min_candidates,
            field_unit=field_unit,
            constraint_values=constraint_values,
        )
        _append_strategy_log(
            strategy_log_path,
            field_unit,
            round_id=1,
            profile=profile_round1,
            mutations=mutations,
            evidence_details=evidence_round1,
            note="initial round based on baseline trace",
        )
        _emit_field_candidates(field_unit, mutations, "候选值（初始轮）")
        selections.append({
            "field": field_unit,
            "mutations": mutations,
            "seed": field_seed,
            "timestamp": time.time(),
        })

        field_dir = os.path.join(mutations_root, safe_field_dir(field_unit))
        ensure_dir(field_dir)

        sent_logs_this_field: List[str] = []
        next_idx = 1
        executed_by_payload = {}
        field_metric_list: List[dict] = []
        early_stop_min = int(field_budget.get("early_stop_min", 0))
        executed_mutations: List[dict] = []
        for mut in mutations:
            idx = next_idx
            next_idx += 1
            try:
                log_path = _run_or_map_mutation(
                    args=args,
                    taint_manager=taint_manager,
                    field_unit=field_unit,
                    field_label=label,
                    field_dir=field_dir,
                    mut=mut,
                    idx=idx,
                    total=len(mutations),
                    proto=proto,
                    field_byteorder=field_byteorder,
                    executed_by_payload=executed_by_payload,
                )
            except Exception as exc:
                _progress("send", f"字段 {label} mutation 失败，跳过当前值并继续后续 mutation：{type(exc).__name__}: {exc}")
                _record_failed_mutation(args, field_unit, field_dir, mut, idx, proto, exc)
                executed_mutations.append(mut)
                if taint_manager is not None and args.taint_auto_restart:
                    _restart_taint_process(
                        args,
                        taint_manager,
                        f"字段 {label} mutation 失败后，重启污点分析并继续下一个 mutation",
                    )
                continue
            sent_logs_this_field.append(log_path)
            executed_mutations.append(mut)
            preprocess = preprocess_log(log_path)
            preprocess_check = _inspect_preprocessed_log(preprocess["path"])
            meta_name_part = mut.get("name") or f"mut{idx}"
            meta_path = os.path.join(field_dir, f"{idx:03d}_{meta_name_part}.json")
            _record_preprocessed_check(meta_path, preprocess_check)
            preprocess_issue = _describe_preprocessed_log_issue(preprocess_check)
            if preprocess_issue:
                _progress("send", f"字段 {label} mutation {meta_name_part}: {preprocess_issue}")
            features = parse_log_features(preprocess["path"])
            metrics = diff_metrics(baseline_features, features)
            field_metric_list.append(metrics)
            if _should_early_stop_field(field_metric_list, early_stop_min):
                _progress(
                    "field",
                    f"字段 {label} 早停：前 {len(field_metric_list)} 次 mutation 指标无明显分化",
                )
                break
        if len(executed_mutations) != len(mutations):
            mutations[:] = executed_mutations
            selections[-1]["mutations"] = mutations

        if (
            args.constraint_guided
            and field_budget.get("allow_round2")
            and args.cg_rounds > 1
            and field_unit.get("kind") != "bit"
        ):
            width_bits = unit_width_bits(field_unit)
            base_value = read_value(payload, field_unit, field_byteorder)

            baseline_evidence = extract_field_compare_evidence(
                baseline_log, a, b, width_bits,
                source_bin=args.server_bin,
                field_ranges=byte_field_ranges,
            )
            known_imm_values = [int(item["value"]) for item in baseline_evidence]
            known_imm_set = set(known_imm_values)
            known_evidence_map = {int(item["value"]): item for item in baseline_evidence}
            round_input_logs = list(sent_logs_this_field)
            for round_id in range(2, args.cg_rounds + 1):
                round_new_imms: List[int] = []
                for log_path in round_input_logs:
                    round_evidence = extract_field_compare_evidence(
                        log_path, a, b, width_bits,
                        source_bin=args.server_bin,
                        field_ranges=byte_field_ranges,
                    )
                    for item in round_evidence:
                        imm = int(item["value"])
                        if imm not in known_imm_set:
                            known_imm_set.add(imm)
                            known_imm_values.append(imm)
                            round_new_imms.append(imm)
                            known_evidence_map[imm] = item
                if not round_new_imms:
                    _progress("field", f"字段 {label} 闭环 round {round_id}: 无新增约束，结束")
                    break

                existing_values = {
                    int(value)
                    for item in mutations
                    for value in [parse_value_hex(item.get("final_value_hex"))]
                    if isinstance(value, int)
                }
                grouped_candidates = build_v3_grouped_candidates(
                    base_value=base_value,
                    width_bits=width_bits,
                    immediates=known_imm_values,
                    topk_per_group=field_budget["group_topk"],
                    min_per_group=args.group_min_candidates,
                )
                extra_mutations = build_v3_mutations_for_unit(
                    payload=payload,
                    field_unit=field_unit,
                    grouped={"enum": grouped_candidates.get("enum", [])},
                    existing_values=existing_values,
                    round_id=round_id,
                    byteorder=field_byteorder,
                    groups=("enum",),
                )
                if not extra_mutations:
                    _progress("field", f"字段 {label} 闭环 round {round_id}: 无新增候选，结束")
                    break

                profile_round_n = build_constraint_profile(width_bits, known_imm_values)
                _append_strategy_log(
                    strategy_log_path,
                    field_unit,
                    round_id=round_id,
                    profile=profile_round_n,
                    mutations=extra_mutations,
                    evidence_details=[known_evidence_map[v] for v in known_imm_values if v in known_evidence_map],
                    note=f"closed-loop round {round_id}, new_immediates={round_new_imms}",
                )
                _progress("field", f"字段 {label} 闭环 round {round_id}: 新增 {len(extra_mutations)} 个候选")
                _emit_field_candidates(field_unit, extra_mutations, f"候选值（闭环 round {round_id}）")
                mutations.extend(extra_mutations)
                selections[-1]["mutations"] = mutations

                next_round_logs: List[str] = []
                for mut in extra_mutations:
                    idx = next_idx
                    next_idx += 1
                    try:
                        log_path = _run_or_map_mutation(
                            args=args,
                            taint_manager=taint_manager,
                            field_unit=field_unit,
                            field_label=label,
                            field_dir=field_dir,
                            mut=mut,
                            idx=idx,
                            total=len(mutations),
                            proto=proto,
                            field_byteorder=field_byteorder,
                            executed_by_payload=executed_by_payload,
                        )
                    except Exception as exc:
                        _progress("send", f"字段 {label} 闭环 mutation 失败，跳过并继续：{type(exc).__name__}: {exc}")
                        _record_failed_mutation(args, field_unit, field_dir, mut, idx, proto, exc)
                        if taint_manager is not None and args.taint_auto_restart:
                            _restart_taint_process(
                                args,
                                taint_manager,
                                f"字段 {label} 闭环 mutation 失败后，重启污点分析并继续后续候选",
                            )
                        continue
                    next_round_logs.append(log_path)
                    preprocess = preprocess_log(log_path)
                    preprocess_check = _inspect_preprocessed_log(preprocess["path"])
                    meta_name_part = mut.get("name") or f"mut{idx}"
                    round_meta_path = os.path.join(field_dir, f"{idx:03d}_{meta_name_part}.json")
                    _record_preprocessed_check(round_meta_path, preprocess_check)
                    preprocess_issue = _describe_preprocessed_log_issue(preprocess_check)
                    if preprocess_issue:
                        _progress("send", f"字段 {label} 闭环 mutation {meta_name_part}: {preprocess_issue}")
                    features = parse_log_features(preprocess["path"])
                    metrics = diff_metrics(baseline_features, features)
                    field_metric_list.append(metrics)
                round_input_logs = next_round_logs

        field_elapsed = time.time() - field_t_start
        _progress("field", f"字段 {label} 完成，用时 {_format_hms(field_elapsed)}", field_idx, len(target_fields))
        sample_elapsed = time.time() - sample_start_ts
        run_elapsed = time.time() - run_start_ts
        _progress(
            "field",
            f"累计耗时：样本 {_format_hms(sample_elapsed)}，全流程 {_format_hms(run_elapsed)}",
            field_idx,
            len(target_fields),
        )

    write_json(mutations_path, selections)
    _progress("sample", "mutations.json 写入完成")

    # 6) Diff and report
    _progress("diff", "开始差分计算")
    cmd_diff(argparse.Namespace(outdir=outdir))
    _progress("diff", "差分计算完成")


def cmd_full(args: argparse.Namespace) -> None:
    """full 子命令主逻辑"""
    ensure_dir(args.outdir)
    if args.taint_max_restarts <= 0:
        raise ValueError("--taint-max-restarts must be >= 1")
    if args.packet_retry_on_taint_fail < 0:
        raise ValueError("--packet-retry-on-taint-fail must be >= 0")
    if args.taint_restart_delay < 0:
        raise ValueError("--taint-restart-delay must be >= 0")
    if args.cg_rounds < 1:
        raise ValueError("--cg-rounds must be >= 1")
    if args.group_topk <= 0:
        raise ValueError("--group-topk must be >= 1")
    if args.group_min_candidates <= 0:
        raise ValueError("--group-min-candidates must be >= 1")

    taint_manager = None
    run_start_ts = time.time()
    try:
        _progress("run", "full 流程启动")
        if should_enable_taint(args):
            _progress("taint", "准备启动污点分析进程")
            command = resolve_taint_command(args)
            taint_manager = TaintProcessManager(
                command=command,
                work_dir=args.taint_workdir,
                stdout_log=args.taint_stdout_log,
                startup_time=args.taint_startup_time,
                shutdown_timeout=args.taint_shutdown_timeout,
                kill_existing=args.taint_kill_existing,
                server_name=resolve_server_name(args),
                kill_wait=args.taint_kill_wait,
            )
            taint_manager.start()
            _progress("taint", "污点分析进程已启动，执行健康检查")
            if taint_manager.process is None or taint_manager.process.poll() is not None:
                if args.taint_auto_restart:
                    _restart_taint_process(args, taint_manager, "初次启动后进程未存活")
                else:
                    raise RuntimeError(_build_taint_startup_error(args, taint_manager))
            if not wait_for_file(args.pin_log, args.taint_startup_time):
                if args.taint_auto_restart:
                    _restart_taint_process(args, taint_manager, "初次启动后 pin 日志不可见")
                else:
                    raise RuntimeError(_build_taint_log_wait_error(args, taint_manager))

        if args.mode != "frozen" and args.sample_count <= 0:
            raise ValueError("--sample-count must be >= 1")

        if (args.sample_count > 1 or args.mode == "frozen") and args.clean and not args.resume:
            _progress("sample", "多样本模式：清理 outdir 根目录旧产物与 sample_* 目录")
            _clean_multisample_root(args.outdir)

        samples = []
        if args.mode == "pcap":
            if not args.pcap:
                raise ValueError("pcap 模式下必须提供 --pcap")
            if args.sample_count > 1 and args.index is not None:
                raise ValueError("--index cannot be used with --sample-count > 1")
            pcap_stats = summarize_pcap_payloads(args.pcap, args.proto)
            _progress(
                "run",
                "pcap 统计："
                f"原始总包数={pcap_stats['total_packets']}，"
                f"非空 Raw={pcap_stats['raw_nonempty']}，"
                f"可用候选={pcap_stats['usable']} "
                f"(tcp_raw={pcap_stats['tcp_raw']}, udp_raw={pcap_stats['udp_raw']})",
            )

            if args.sample_count > 1:
                # 多样本模式：按 pcap 顺序取前 N 个有效报文（不随机）。
                candidates = collect_payload_candidates_from_pcap(args.pcap, args.proto)
                if len(candidates) < args.sample_count:
                    raise RuntimeError(
                        f"not enough valid payloads in pcap: need {args.sample_count}, got {len(candidates)}"
                    )
                for idx, (payload, proto, pkt_idx) in enumerate(candidates[:args.sample_count], start=1):
                    samples.append({
                        "payload": payload,
                        "proto": proto,
                        "meta": {
                            "mode": "pcap",
                            "pcap": args.pcap,
                            "seed": args.seed + idx - 1,
                            "index": pkt_idx,
                            "proto": proto,
                            "payload_hex": bytes_to_hex(payload),
                            "timestamp": time.time(),
                        },
                    })
            else:
                seed_for_sample = args.seed
                payload, proto, chosen_pkt_index = sample_payload_from_pcap(
                    args.pcap, seed_for_sample, args.index, args.proto
                )
                samples.append({
                    "payload": payload,
                    "proto": proto,
                    "meta": {
                        "mode": "pcap",
                        "pcap": args.pcap,
                        "seed": seed_for_sample,
                        "index": chosen_pkt_index,
                        "proto": proto,
                        "payload_hex": bytes_to_hex(payload),
                        "timestamp": time.time(),
                    },
                })
            _progress("run", f"pcap 采样完成，共 {len(samples)} 个样本")
        elif args.mode == "hex":
            if not args.hex:
                raise ValueError("hex 模式下必须提供 --hex")
            if args.sample_count > 1:
                raise ValueError("hex 模式下 --sample-count 目前仅支持 1")
            payload, proto, chosen_index = sample_payload_from_hex(args.hex, args.proto)
            samples.append({
                "payload": payload,
                "proto": proto,
                "meta": {
                    "mode": "hex",
                    "hex": args.hex,
                    "seed": args.seed,
                    "index": chosen_index,
                    "proto": proto,
                    "payload_hex": bytes_to_hex(payload),
                    "timestamp": time.time(),
                },
            })
            _progress("run", "hex 样本准备完成，共 1 个样本")
        else:
            samples = _load_frozen_samples(args)
            _progress(
                "run",
                f"冻结 baseline 样本准备完成：protocol={args.frozen_protocol}，共 {len(samples)} 个样本",
            )

        for sample_no, sample in enumerate(samples, start=1):
            run_outdir = args.outdir
            if len(samples) > 1 or args.mode == "frozen":
                run_outdir = os.path.join(args.outdir, f"sample_{sample_no:03d}")
            if args.resume and _sample_pipeline_is_complete(run_outdir):
                _progress("sample", "断点续跑：已有完整产物，跳过样本", sample_no, len(samples))
                continue
            if args.resume and os.path.exists(run_outdir):
                _progress("sample", "断点续跑：清理不完整样本后从头重跑", sample_no, len(samples))
                shutil.rmtree(run_outdir)
            _progress("sample", "开始执行样本", sample_no, len(samples))
            sample_t_start = time.time()
            _run_single_sample_pipeline(
                args=args,
                outdir=run_outdir,
                payload=sample["payload"],
                proto=sample["proto"],
                sample_meta=sample["meta"],
                taint_manager=taint_manager,
                run_start_ts=run_start_ts,
            )
            sample_elapsed = time.time() - sample_t_start
            _progress("sample", f"样本执行完成，用时 {_format_hms(sample_elapsed)}", sample_no, len(samples))

        _progress("summary", "开始生成字段级摘要向量")
        if len(samples) > 1 or args.mode == "frozen":
            sample_dirs = [Path(args.outdir) / f"sample_{idx:03d}" for idx in range(1, len(samples) + 1)]
            summary_output_dir = Path(args.outdir) / "field_training_samples"
        else:
            sample_dirs = [Path(args.outdir)]
            summary_output_dir = Path(args.outdir) / "field_training_samples"
        summary_result = build_field_training_artifacts(
            sample_dirs=sample_dirs,
            output_dir=summary_output_dir,
            protocol_dir_name_override=os.path.basename(os.path.abspath(args.outdir)),
        )
        _progress(
            "summary",
            f"摘要向量生成完成：{summary_result['record_count']} 个字段，输出 {summary_result['output_dir']}",
        )
    finally:
        if taint_manager:
            _progress("taint", "停止污点分析进程")
            taint_manager.stop()
        _progress("run", "full 流程结束")


def _clean_full_artifacts(outdir: str) -> None:
    """清理 full 流程可能复用到的旧产物，避免干扰本次运行。"""
    mutations_root = os.path.join(outdir, "mutations")
    if os.path.isdir(mutations_root):
        shutil.rmtree(mutations_root)

    for filename in (
        "sample.json",
        "fields.json",
        "fields.byte.json",
        "bitfields.json",
        "mutations.json",
        "report.json",
        "report_compact.json",
        "mutation_strategy.log",
    ):
        path = os.path.join(outdir, filename)
        if os.path.exists(path):
            os.remove(path)

    summary_dir = os.path.join(outdir, "field_training_samples")
    if os.path.isdir(summary_dir):
        shutil.rmtree(summary_dir)


def _clean_multisample_root(outdir: str) -> None:
    """多样本模式下清理根目录历史产物与 sample_* 子目录。"""
    _clean_full_artifacts(outdir)
    if not os.path.isdir(outdir):
        return
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if os.path.isdir(path) and name.startswith("sample_"):
            shutil.rmtree(path)


def _sample_pipeline_is_complete(outdir: str) -> bool:
    """Return whether a prior sample has the durable artifacts needed downstream."""
    required = (
        ("mutations.json", list),
        ("report.json", dict),
        ("report_compact.json", dict),
    )
    for filename, expected_type in required:
        path = os.path.join(outdir, filename)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
        try:
            data = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(data, expected_type):
            return False
    return True


def _load_frozen_samples(args: argparse.Namespace) -> List[dict]:
    """Load frozen baseline packet metadata ordered by replay ordinal."""
    if not args.frozen_protocol:
        raise ValueError("frozen 模式下必须提供 --frozen-protocol")
    protocol_dir = Path(args.frozen_baseline_root) / args.frozen_protocol
    if not protocol_dir.is_dir():
        raise FileNotFoundError(f"frozen protocol directory not found: {protocol_dir}")

    records: List[tuple] = []
    for pkt_dir in sorted(protocol_dir.glob("pkt_*")):
        if not pkt_dir.is_dir():
            continue
        meta_path = pkt_dir / "meta.json"
        preprocessed_path = pkt_dir / "trace.preprocessed.log"
        if not meta_path.exists():
            raise FileNotFoundError(f"frozen baseline meta.json not found: {meta_path}")
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"frozen baseline preprocessed log not found: {preprocessed_path}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid frozen baseline metadata: {meta_path}: {exc}") from exc

        ordinal = int(meta.get("ordinal") or 0)
        payload_hex = str(meta.get("payload_hex") or "").strip()
        proto = str(meta.get("proto") or args.proto or "").strip()
        if ordinal <= 0:
            raise ValueError(f"frozen baseline metadata missing positive ordinal: {meta_path}")
        if not payload_hex:
            raise ValueError(f"frozen baseline metadata missing payload_hex: {meta_path}")
        if proto not in {"tcp", "udp"}:
            raise ValueError(f"frozen baseline metadata has unsupported proto={proto!r}: {meta_path}")
        try:
            payload = hex_to_bytes(payload_hex)
        except ValueError as exc:
            raise ValueError(f"invalid payload_hex in frozen baseline metadata: {meta_path}: {exc}") from exc

        sample_meta = {
            "mode": "frozen",
            "seed": args.seed + ordinal - 1,
            "index": meta.get("pcap_packet_index"),
            "proto": proto,
            "payload_hex": payload_hex.lower(),
            "timestamp": time.time(),
            "frozen_baseline_dir": str(pkt_dir.resolve()),
            "frozen_packet_id": pkt_dir.name,
            "frozen_ordinal": ordinal,
            "pcap": meta.get("pcap", ""),
        }
        records.append((ordinal, pkt_dir.name, {"payload": payload, "proto": proto, "meta": sample_meta}))

    records.sort(key=lambda item: (item[0], item[1]))
    ordinals = [item[0] for item in records]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError(f"duplicate frozen baseline ordinal under {protocol_dir}: {ordinals}")
    if not records:
        raise RuntimeError(f"no frozen baseline packets found under: {protocol_dir}")
    if args.frozen_limit > 0:
        records = records[:args.frozen_limit]
    return [item[2] for item in records]


def _tail_log(path: str, max_lines: int = 30) -> str:
    """读取日志末尾若干行（失败时返回空串）。"""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = lines[-max_lines:]
        return "".join(tail).strip()
    except OSError:
        return ""


def _build_taint_startup_error(args: argparse.Namespace, taint_manager: TaintProcessManager) -> str:
    """构建污点进程启动失败的详细错误信息。"""
    exit_code = None
    pid = None
    if taint_manager and taint_manager.process:
        pid = taint_manager.process.pid
        exit_code = taint_manager.process.poll()
    tail = _tail_log(args.taint_stdout_log, 40)
    parts = [
        "污点分析进程启动失败或已退出。",
        f"pid={pid}, exit_code={exit_code}",
        f"command={resolve_taint_command(args)}",
        f"workdir={args.taint_workdir}",
        f"stdout_log={args.taint_stdout_log}",
    ]
    if tail:
        parts.append("taint stdout 日志末尾：\n" + tail)
    return "\n".join(parts)


def _build_taint_log_wait_error(args: argparse.Namespace, taint_manager: TaintProcessManager) -> str:
    """构建 pin 日志等待失败的详细错误信息。"""
    poll_result = None
    if taint_manager and taint_manager.process:
        poll_result = taint_manager.process.poll()
    tail = _tail_log(args.taint_stdout_log, 40)
    parts = [
        "pin 日志在启动污点分析后仍未生成。",
        f"pin_log={args.pin_log}",
        f"taint_process_poll={poll_result} (None 表示仍在运行)",
        f"command={resolve_taint_command(args)}",
        f"workdir={args.taint_workdir}",
        f"stdout_log={args.taint_stdout_log}",
    ]
    if tail:
        parts.append("taint stdout 日志末尾：\n" + tail)
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="完整流程：选包 → baseline → 字段划分 → 变体 → 差分"
    )
    parser.add_argument("--mode", choices=["pcap", "hex", "frozen"], required=True,
                        help="数据来源模式")
    parser.add_argument("--pcap", help="pcap 文件路径")
    parser.add_argument("--hex", dest="hex", help="hex payload 字符串")
    parser.add_argument("--proto", choices=["auto", "tcp", "udp"],
                        default=DEFAULT_PROTO, help="协议类型")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子")
    parser.add_argument("--sample-count", type=int, default=1,
                        help="按 pcap 顺序选择的有效样本数量（仅 pcap 模式，默认 1）")
    parser.add_argument("--index", type=int,
                        help="指定原始 pcap 包号（pkt_idx）")
    parser.add_argument(
        "--frozen-baseline-root",
        default="/root/semvec/bitfield_groundtruth/replay_manual_latest/outputs",
        help="冻结 baseline 根目录（仅 frozen 模式）",
    )
    parser.add_argument(
        "--frozen-protocol",
        choices=["bacnet", "cip", "iec104", "iec61850", "modbus", "snap7"],
        help="读取冻结 baseline 的协议目录（仅 frozen 模式）",
    )
    parser.add_argument(
        "--frozen-limit",
        type=int,
        default=0,
        help="仅处理按 ordinal 排序后的前 N 个冻结样本；0 表示全量（仅 frozen 模式）",
    )
    parser.add_argument("--target-host", default=DEFAULT_TARGET_HOST,
                        help="目标主机")
    parser.add_argument("--target-port", type=int, default=DEFAULT_TARGET_PORT,
                        help="目标端口")
    parser.add_argument("--pin-log", default=DEFAULT_PIN_LOG,
                        help="pin 日志路径")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="输出目录")
    parser.add_argument("--clean", dest="clean", action="store_true", default=True,
                        help="运行前清理 outdir 下旧产物（默认开启）")
    parser.add_argument("--no-clean", dest="clean", action="store_false",
                        help="禁用运行前清理")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：跳过完整样本，清理并重跑不完整样本（仅 frozen 模式）")
    parser.add_argument("--wait-ms", type=int, default=DEFAULT_WAIT_MS,
                        help="发送后等待日志稳定的毫秒数")
    parser.add_argument("--field", help="指定字段范围 a,b")
    parser.add_argument("--constraint-guided", dest="constraint_guided",
                        action="store_true", default=True,
                        help="启用约束引导候选（默认开启）")
    parser.add_argument("--no-constraint-guided", dest="constraint_guided",
                        action="store_false",
                        help="禁用约束引导候选")
    parser.add_argument("--cg-rounds", type=int, default=2,
                        help="约束闭环轮数（默认 2）")
    parser.add_argument("--group-topk", type=int, default=6,
                        help="每个策略组候选上限（默认 6）")
    parser.add_argument("--group-min-candidates", type=int, default=6,
                        help="兼容参数；当前 V3 按 --group-topk 补满每个策略组")
    parser.add_argument("--recv", action="store_true", default=DEFAULT_RECV,
                        help="是否接收响应")
    parser.add_argument("--recv-timeout", type=float, default=DEFAULT_RECV_TIMEOUT,
                        help="接收超时时间（秒）")
    # 污点分析参数
    parser.add_argument("--taint", action="store_true", default=DEFAULT_ENABLE_TAINT,
                        help="启动污点分析")
    parser.add_argument("--no-taint", action="store_true",
                        help="禁用污点分析")
    parser.add_argument("--taint-command", default="",
                        help="自定义污点分析命令")
    parser.add_argument("--taint-prefix", default=DEFAULT_TAINT_PREFIX,
                        help="污点分析前置命令")
    parser.add_argument("--pin-bin", default=DEFAULT_PIN_BIN,
                        help="Pin 可执行文件路径")
    parser.add_argument("--taint-tool", default=DEFAULT_TAINT_TOOL,
                        help="taint 插桩 so 路径")
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN,
                        help="被测 server 可执行文件路径")
    parser.add_argument("--server-args", default=DEFAULT_SERVER_ARGS,
                        help="server 启动参数")
    parser.add_argument("--taint-server-name", default="",
                        help="server 进程名（用于清理残留）")
    parser.add_argument("--taint-workdir", default=DEFAULT_TAINT_WORKDIR,
                        help="污点分析工作目录")
    parser.add_argument("--taint-stdout-log", default=DEFAULT_TAINT_STDOUT_LOG,
                        help="污点分析标准输出日志")
    parser.add_argument("--taint-startup-time", type=float,
                        default=DEFAULT_TAINT_STARTUP_TIME,
                        help="污点分析启动等待时间（秒）")
    parser.add_argument("--taint-shutdown-timeout", type=float,
                        default=DEFAULT_TAINT_SHUTDOWN_TIMEOUT,
                        help="污点分析优雅退出超时（秒）")
    parser.add_argument("--taint-kill-existing", action="store_true",
                        default=DEFAULT_TAINT_KILL_EXISTING,
                        help="启动前清理残留进程")
    parser.add_argument("--taint-kill-wait", type=float,
                        default=DEFAULT_TAINT_KILL_TIMEOUT,
                        help="清理残留进程后的等待时间（秒）")
    parser.add_argument("--taint-log-marker", action="store_true",
                        default=DEFAULT_TAINT_LOG_MARKER,
                        help="在 taint 日志中写入发送标记")
    parser.add_argument("--taint-auto-restart", dest="taint_auto_restart",
                        action="store_true", default=True,
                        help="污点进程异常退出时自动重启（默认开启）")
    parser.add_argument("--no-taint-auto-restart", dest="taint_auto_restart",
                        action="store_false",
                        help="禁用污点进程自动重启")
    parser.add_argument("--taint-max-restarts", type=int, default=3,
                        help="污点进程单次故障允许的最大重启次数（默认 3）")
    parser.add_argument("--packet-retry-on-taint-fail", type=int, default=1,
                        help="污点崩溃后当前报文重发次数（默认 1）")
    parser.add_argument("--taint-restart-delay", type=float, default=1.0,
                        help="每次污点重启失败后的等待秒数（默认 1.0）")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "pcap" and not args.pcap:
        parser.error("--pcap is required for pcap mode")
    if args.mode == "hex" and not args.hex:
        parser.error("--hex is required for hex mode")
    if args.mode == "frozen" and not args.frozen_protocol:
        parser.error("--frozen-protocol is required for frozen mode")
    if args.frozen_limit < 0:
        parser.error("--frozen-limit must be >= 0")
    if args.resume and args.mode != "frozen":
        parser.error("--resume is currently supported only for frozen mode")

    try:
        cmd_full(args)
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
