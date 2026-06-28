#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mutate.py - 生成 payload 变体
读取 sample.json 和 fields.json，生成 mutations.json
"""

import argparse
import os
import random
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from common import (
    DEFAULT_OUTDIR,
    DEFAULT_SEED,
    bytes_to_hex,
    ensure_dir,
    hex_to_bytes,
    load_fields,
    load_sample,
    print_user_error,
    read_json,
    write_json,
)
from field_units import (
    byte_ranges_from_units,
    field_id,
    normalize_unit,
    read_value,
    units_from_fields_json,
    values_from_bit_constraints,
    width_bits as unit_width_bits,
    write_value,
)

THREAD_PREFIX_RE = re.compile(r"^THREADID\t([^\t]+)\t(.*)$")
IMM_HEX_RE = re.compile(r"(?<![0-9A-Za-z_])-?0x[0-9a-fA-F]+")
IMM_DEC_RE = re.compile(r"(?<![0-9A-Za-z_])-?\d+")
RANGE_RE = re.compile(r"\((\d+)\s*[,:\-]\s*(\d+)\)|(\d+)\s*-\s*(\d+)")
COMPARE_MNEMONICS = {"cmp", "test"}
COMPUTE_MNEMONICS = {
    "add", "sub", "adc", "sbb",
    "and", "or", "xor",
    "shl", "shr", "sar", "sal",
    "imul", "mul", "idiv", "div",
    "inc", "dec", "neg", "not",
    "lea", "rol", "ror", "rcl", "rcr",
}
MOVE_MNEMONICS = {
    "mov", "movzx", "movsx", "movsxd",
    "cmovz", "cmovnz", "cmove", "cmovne",
    "xchg",
}
ZERO_TEST_RE = re.compile(r"^\s*test\s+([^,\s]+)\s*,\s*([^,\s]+)\s*$", re.IGNORECASE)
ZERO_JUMP_RE = re.compile(r"^\s*j(?:z|e|nz|ne)\b", re.IGNORECASE)

V3_STRATEGY_GROUPS = ("neighborhood", "boundary", "enum", "extreme")


def format_value_hex(value: int, width_bits: int) -> str:
    """按字段位宽格式化十六进制值，供输出文件使用。"""
    if width_bits <= 0:
        return f"0x{int(value):x}"
    mask = (1 << width_bits) - 1
    width_nibbles = max(1, (width_bits + 3) // 4)
    return f"0x{(int(value) & mask):0{width_nibbles}x}"


def parse_value_hex(value: object) -> Optional[int]:
    """兼容读取十六进制字符串或旧整数值。"""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        try:
            return int(value, 16)
        except ValueError:
            return None
    return None


def _split_thread_prefix(line: str) -> Tuple[Optional[str], str]:
    match = THREAD_PREFIX_RE.match(line.rstrip("\n"))
    if not match:
        return None, line.rstrip("\n")
    return match.group(1), match.group(2)


def _extract_indices(idx_raw: str) -> Set[int]:
    idx_set: Set[int] = set()
    for m in RANGE_RE.finditer(idx_raw):
        a = m.group(1) or m.group(3)
        b = m.group(2) or m.group(4)
        if a is None or b is None:
            continue
        lo = int(a)
        hi = int(b)
        if lo > hi:
            lo, hi = hi, lo
        for i in range(lo, hi + 1):
            idx_set.add(i)
    for token in re.findall(r"\d+", idx_raw):
        idx_set.add(int(token))
    return idx_set


def _extract_immediates_from_operand_text(operand_text: str) -> List[int]:
    values: List[int] = []
    seen: Set[int] = set()
    # 仅从“非内存操作数”中提取立即数，避免把 [rbx+0x5] 这种地址位移当成比较约束值。
    operands = [part.strip() for part in operand_text.split(",") if part.strip()]
    for op in operands:
        if "[" in op and "]" in op:
            continue
        hex_tokens = IMM_HEX_RE.findall(op)
        for token in hex_tokens:
            try:
                value = int(token, 16)
            except ValueError:
                continue
            if value not in seen:
                seen.add(value)
                values.append(value)
        # 避免将 0x.. 内部的前导/片段数字再次按十进制提取（如 0xe0 -> 0）
        dec_text = IMM_HEX_RE.sub(" ", op)
        for token in IMM_DEC_RE.findall(dec_text):
            try:
                value = int(token, 10)
            except ValueError:
                continue
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _parse_instruction_payload(payload: str) -> Optional[Tuple[str, str, str]]:
    if not payload.startswith("Instruction\t"):
        return None
    cols = payload.split("\t")
    if len(cols) < 2:
        return None
    addr_disasm = cols[1].strip()
    colon = addr_disasm.find(": ")
    if colon < 0:
        return None
    addr = addr_disasm[:colon].strip()
    disasm = addr_disasm[colon + 2:].strip()
    idx_raw = cols[2].strip() if len(cols) > 2 else ""
    info_raw = cols[3].strip() if len(cols) > 3 else ""
    return addr, disasm, idx_raw, info_raw


def _is_zero_test_disasm(disasm: str) -> bool:
    """判断是否为 test reg, reg 形式的零值检查。"""
    match = ZERO_TEST_RE.match(disasm.strip())
    if not match:
        return False
    left = match.group(1).strip().lower()
    right = match.group(2).strip().lower()
    return left == right


def _is_zero_jump_disasm(disasm: str) -> bool:
    """判断是否为依赖 ZF 的零值分支。"""
    return bool(ZERO_JUMP_RE.match(disasm.strip()))


def _extract_non_tainted_cmp_values(info_raw: str) -> List[int]:
    """从日志第四列提取 cmp 的非污染操作数值。

    例如：
    - DST*=0x28;SRC=0x28 -> 提取 0x28
    - DST=0x10;SRC*=0x7  -> 提取 0x10
    """
    if not info_raw:
        return []
    values: List[int] = []
    seen: Set[int] = set()
    for token in info_raw.split(";"):
        token = token.strip()
        if not token or "=" not in token:
            continue
        lhs, rhs = token.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if "*" in lhs:
            continue
        extracted = _extract_immediates_from_operand_text(rhs)
        for value in extracted:
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _filter_compare_values_by_width(
    values: Sequence[int],
    width_bits: int,
    *,
    from_info_raw: bool = False,
) -> List[int]:
    """按字段位宽过滤 compare 证据值。

    设计目标：
    - 保留与字段位宽一致的真实约束值
    - 丢弃从 info_raw 回退路径提取出的明显运行时地址/指针值

    说明：
    - 对反汇编操作数字面量，仍按原逻辑保留
    - 对 info_raw 提取出的“非污染操作数值”，若其原始值已经超过字段位宽上界，
      则视为不适合作为该字段的约束证据，直接丢弃
    """
    if width_bits <= 0:
        return []
    mask = (1 << width_bits) - 1
    out: List[int] = []
    for value in values:
        try:
            ivalue = int(value)
        except Exception:
            continue
        if from_info_raw and ivalue > mask:
            continue
        out.append(ivalue)
    return out


def _field_overlap(a: int, b: int, idx_set: Set[int]) -> bool:
    if not idx_set:
        return False
    for i in range(a, b + 1):
        if i in idx_set:
            return True
    return False


def _resolve_evidence_owner_field(
    idx_set: Set[int],
    field_ranges: Sequence[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """为一条约束指令选择唯一归属字段。

    规则：
    - 只在有重叠的字段中选择
    - 优先选择重叠字节数最多的字段
    - 若重叠字节数相同，优先选择范围更大的字段
    - 若仍相同，按起始/结束边界稳定排序
    """
    if not idx_set or not field_ranges:
        return None

    candidates: List[Tuple[int, int, int, int]] = []
    for a, b in field_ranges:
        overlap = sum(1 for i in range(a, b + 1) if i in idx_set)
        if overlap <= 0:
            continue
        span = b - a + 1
        candidates.append((overlap, span, -a, -b))

    if not candidates:
        return None

    best = max(candidates)
    overlap_best, span_best, neg_a_best, neg_b_best = best
    return (-neg_a_best, -neg_b_best)


def _parse_module_offset(addr: str) -> Tuple[str, str]:
    if "+0x" in addr:
        mod, off = addr.split("+", 1)
        return mod.strip(), off.strip()
    return "", addr.strip()


def _resolve_source_location(addr: str, source_bin: str, cache: Dict[str, str]) -> str:
    """将地址解析为源码位置（best-effort）。"""
    key = f"{source_bin}|{addr}"
    if key in cache:
        return cache[key]
    if not source_bin or not os.path.exists(source_bin):
        cache[key] = "unknown"
        return cache[key]
    _, query_addr = _parse_module_offset(addr)
    try:
        result = subprocess.run(
            ["addr2line", "-f", "-C", "-e", source_bin, query_addr],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if len(lines) >= 2:
            loc = lines[-1]
        elif lines:
            loc = lines[0]
        else:
            loc = "unknown"
        cache[key] = loc
    except OSError:
        cache[key] = "unknown"
    return cache[key]


def extract_field_compare_evidence(
    log_path: str,
    a: int,
    b: int,
    width_bits: int,
    source_bin: str = "",
    field_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[dict]:
    """提取字段 compare 证据（含立即数、地址、trace 行号、源码位置）。"""
    if not log_path or not os.path.exists(log_path):
        return []
    mask = (1 << width_bits) - 1 if width_bits > 0 else 0
    taint_found = False
    seen_values: Set[int] = set()
    evidence_list: List[dict] = []
    source_cache: Dict[str, str] = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = list(enumerate(f, start=1))

    pending_zero_test: Optional[dict] = None

    for line_no, raw in lines:
        _, payload = _split_thread_prefix(raw)
        if not taint_found:
            if payload.startswith("Taint\t"):
                taint_found = True
            continue
        parsed = _parse_instruction_payload(payload)
        if not parsed:
            continue
        addr, disasm, idx_raw, info_raw = parsed
        mnemonic = disasm.split(" ", 1)[0].lower() if disasm else ""

        if pending_zero_test is not None:
            if _is_zero_jump_disasm(disasm):
                norm = 0 & mask
                if norm not in seen_values:
                    seen_values.add(norm)
                    evidence_list.append({
                        "value": norm,
                        "raw_imm": 0,
                        "address": pending_zero_test["address"],
                        "trace_line": pending_zero_test["trace_line"],
                        "disasm": pending_zero_test["disasm"],
                        "source_location": _resolve_source_location(
                            pending_zero_test["address"], source_bin, source_cache
                        ),
                        "trace_log_path": log_path,
                    })
            pending_zero_test = None

        if mnemonic not in COMPARE_MNEMONICS:
            continue
        idx_set = _extract_indices(idx_raw)
        if not _field_overlap(a, b, idx_set):
            continue
        if field_ranges:
            owner = _resolve_evidence_owner_field(idx_set, field_ranges)
            if owner != (a, b):
                continue

        if mnemonic == "test" and _is_zero_test_disasm(disasm):
            pending_zero_test = {
                "address": addr,
                "trace_line": line_no,
                "disasm": disasm,
            }
            continue

        operands = disasm.split(" ", 1)[1] if " " in disasm else ""
        compare_values = _extract_immediates_from_operand_text(operands)
        compare_values = _filter_compare_values_by_width(
            compare_values, width_bits, from_info_raw=False
        )
        if mnemonic == "cmp" and not compare_values:
            compare_values = _extract_non_tainted_cmp_values(info_raw)
            compare_values = _filter_compare_values_by_width(
                compare_values, width_bits, from_info_raw=True
            )

        for imm in compare_values:
            norm = int(imm) & mask
            if norm in seen_values:
                continue
            seen_values.add(norm)
            evidence_list.append({
                "value": norm,
                "raw_imm": imm,
                "address": addr,
                "trace_line": line_no,
                "disasm": disasm,
                "source_location": _resolve_source_location(addr, source_bin, source_cache),
                "trace_log_path": log_path,
            })
    return evidence_list


def extract_field_compare_immediates(
    log_path: str,
    a: int,
    b: int,
    width_bits: int,
    field_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[int]:
    """从日志中提取与字段相关 compare-like 指令立即数，按首次出现顺序返回。"""
    evidence = extract_field_compare_evidence(
        log_path, a, b, width_bits, source_bin="", field_ranges=field_ranges
    )
    return [int(item["value"]) for item in evidence if isinstance(item, dict) and "value" in item]


def extract_field_instruction_stats(
    log_path: str,
    a: int,
    b: int,
    field_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> dict:
    """提取字段相关指令类型统计，用于估计字段重要性。"""
    if not log_path or not os.path.exists(log_path):
        return {
            "total_count": 0,
            "compare_count": 0,
            "compute_count": 0,
            "move_count": 0,
            "other_count": 0,
        }

    taint_found = False
    compare_count = 0
    compute_count = 0
    move_count = 0
    other_count = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            _, payload = _split_thread_prefix(raw)
            if not taint_found:
                if payload.startswith("Taint\t"):
                    taint_found = True
                continue
            parsed = _parse_instruction_payload(payload)
            if not parsed:
                continue
            _, disasm, idx_raw, _ = parsed
            idx_set = _extract_indices(idx_raw)
            if not _field_overlap(a, b, idx_set):
                continue
            if field_ranges:
                owner = _resolve_evidence_owner_field(idx_set, field_ranges)
                if owner != (a, b):
                    continue

            mnemonic = disasm.split(" ", 1)[0].lower() if disasm else ""
            if mnemonic in COMPARE_MNEMONICS:
                compare_count += 1
            elif mnemonic in COMPUTE_MNEMONICS:
                compute_count += 1
            elif mnemonic in MOVE_MNEMONICS:
                move_count += 1
            else:
                other_count += 1

    return {
        "total_count": compare_count + compute_count + move_count + other_count,
        "compare_count": compare_count,
        "compute_count": compute_count,
        "move_count": move_count,
        "other_count": other_count,
    }


def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _looks_like_bitmask(values: Sequence[int], width_bits: int) -> bool:
    if not values:
        return False
    pow2_count = sum(1 for v in values if _is_pow2(v))
    low_popcount_count = sum(1 for v in values if v != 0 and bin(int(v)).count("1") <= 2)
    ratio = max(pow2_count, low_popcount_count) / max(len(values), 1)
    return ratio >= 0.5 and width_bits <= 32


def _infer_continuous_range(values: Sequence[int], width_bits: int) -> Optional[Tuple[int, int]]:
    """从常量集合推断连续范围 [L, U]（启发式）。"""
    if not values:
        return None
    mask = (1 << width_bits) - 1 if width_bits > 0 else 0
    uniq = sorted({int(v) & mask for v in values})
    if len(uniq) < 4:
        return None
    l = uniq[0]
    u = uniq[-1]
    span = u - l + 1
    if span <= 0:
        return None
    # 覆盖度足够高则视为连续范围证据
    density = len(uniq) / span
    if density >= 0.6:
        return l, u
    return None


def build_constraint_profile(width_bits: int, immediates: Sequence[int]) -> dict:
    """构建约束证据画像，用于调试与日志记录。"""
    if width_bits <= 0:
        return {
            "mode": "fallback",
            "evidence_count": 0,
            "evidence_values": [],
            "range": None,
            "density": 0.0,
            "bitmask_detected": False,
        }
    mask = (1 << width_bits) - 1
    uniq = sorted({int(v) & mask for v in immediates})
    inferred_range = _infer_continuous_range(uniq, width_bits) if uniq else None
    density = 0.0
    if uniq:
        l = uniq[0]
        u = uniq[-1]
        span = max(1, u - l + 1)
        density = len(uniq) / span
    if inferred_range is not None:
        mode = "range"
    elif uniq:
        mode = "enum"
    else:
        mode = "fallback"
    return {
        "mode": mode,
        "evidence_count": len(uniq),
        "evidence_values": uniq,
        "range": {"l": inferred_range[0], "u": inferred_range[1]} if inferred_range else None,
        "density": density,
        "bitmask_detected": _looks_like_bitmask(uniq, width_bits) if uniq else False,
    }


def _push_v3_candidate(
    grouped: Dict[str, List[dict]],
    group: str,
    strategy: str,
    value: int,
    width_bits: int,
    base_value: int,
) -> None:
    """向 V3 分组候选中追加一个值，组内去重并过滤 baseline 等值。"""
    if group not in grouped or width_bits <= 0:
        return
    mask = (1 << width_bits) - 1
    norm = int(value) & mask
    if norm == (int(base_value) & mask):
        return
    for item in grouped[group]:
        if int(item["value"]) == norm:
            return
    grouped[group].append({
        "group": group,
        "strategy": strategy,
        "value": norm,
    })


def _supplement_v3_group(
    grouped: Dict[str, List[dict]],
    group: str,
    width_bits: int,
    base_value: int,
    min_per_group: int,
    topk_per_group: int,
) -> None:
    """用已有候选邻域补足组内候选数量。"""
    if width_bits <= 0:
        return
    cursor = 0
    while len(grouped[group]) < min_per_group and cursor < len(grouped[group]):
        seed = int(grouped[group][cursor]["value"])
        cursor += 1
        _push_v3_candidate(grouped, group, "supplement_minus_1", seed - 1, width_bits, base_value)
        if len(grouped[group]) >= min_per_group:
            break
        _push_v3_candidate(grouped, group, "supplement_plus_1", seed + 1, width_bits, base_value)
    grouped[group] = grouped[group][:max(1, topk_per_group)]


def build_v3_grouped_candidates(
    base_value: int,
    width_bits: int,
    immediates: Sequence[int],
    topk_per_group: int = 6,
    min_per_group: int = 4,
) -> Dict[str, List[dict]]:
    """Mutation Strategy v3: 按固定策略组生成候选值。"""
    grouped: Dict[str, List[dict]] = {group: [] for group in V3_STRATEGY_GROUPS}
    if width_bits <= 0:
        return grouped

    mask = (1 << width_bits) - 1
    base_norm = int(base_value) & mask
    topk = max(1, int(topk_per_group))
    # V3 当前按策略组统一补满到 topk；min_per_group 仅保留为旧命令兼容参数。
    target_count = topk
    evidence_values = []
    seen_evidence: Set[int] = set()
    for raw in immediates:
        norm = int(raw) & mask
        if norm in seen_evidence:
            continue
        seen_evidence.add(norm)
        evidence_values.append(norm)

    profile = build_constraint_profile(width_bits, evidence_values)
    range_info = profile.get("range")

    # Neighborhood group: base±1 and e/e±1 around observed constraints.
    _push_v3_candidate(grouped, "neighborhood", "base_minus_1", base_norm - 1, width_bits, base_norm)
    _push_v3_candidate(grouped, "neighborhood", "base_plus_1", base_norm + 1, width_bits, base_norm)
    for value in evidence_values:
        _push_v3_candidate(grouped, "neighborhood", "constraint_minus_1", value - 1, width_bits, base_norm)
        _push_v3_candidate(grouped, "neighborhood", "constraint", value, width_bits, base_norm)
        _push_v3_candidate(grouped, "neighborhood", "constraint_plus_1", value + 1, width_bits, base_norm)

    # Boundary group: field-width boundaries plus inferred range boundaries.
    maxv = mask
    for strategy, value in (
        ("min", 0),
        ("min_plus_1", 1),
        ("max_minus_1", maxv - 1),
        ("max", maxv),
    ):
        _push_v3_candidate(grouped, "boundary", strategy, value, width_bits, base_norm)
    if isinstance(range_info, dict):
        l_val = int(range_info.get("l", 0))
        u_val = int(range_info.get("u", 0))
        for strategy, value in (
            ("range_l_minus_1", l_val - 1),
            ("range_l", l_val),
            ("range_l_plus_1", l_val + 1),
            ("range_u_minus_1", u_val - 1),
            ("range_u", u_val),
            ("range_u_plus_1", u_val + 1),
        ):
            _push_v3_candidate(grouped, "boundary", strategy, value, width_bits, base_norm)

    # Enum group: observed compare constants; if sparse, supplement around them.
    for value in evidence_values:
        _push_v3_candidate(grouped, "enum", "constraint", value, width_bits, base_norm)
    if not grouped["enum"]:
        for value in (0, 1, maxv):
            _push_v3_candidate(grouped, "enum", "enum_supplement", value, width_bits, base_norm)

    # Extreme group: degenerate/extreme values and larger offsets.
    for strategy, value in (
        ("zero", 0),
        ("all_ff", maxv),
        ("base_minus_0x10", base_norm - 0x10),
        ("base_plus_0x10", base_norm + 0x10),
    ):
        _push_v3_candidate(grouped, "extreme", strategy, value, width_bits, base_norm)

    for group in V3_STRATEGY_GROUPS:
        grouped[group] = grouped[group][:topk]
        _supplement_v3_group(grouped, group, width_bits, base_norm, target_count, topk)
    return grouped


def build_v3_mutations_for_unit(
    payload: bytes,
    field_unit: dict,
    grouped: Dict[str, List[dict]],
    existing_values: Optional[Set[int]] = None,
    round_id: int = 1,
    byteorder: str = "big",
    groups: Optional[Sequence[str]] = None,
) -> List[dict]:
    """将 V3 分组候选值转换为支持 byte/bit FieldUnit 的 mutation 条目。"""
    unit = normalize_unit(field_unit)
    width = unit_width_bits(unit)
    variants: List[dict] = []
    blocked_values: Set[int] = set(existing_values or set())
    group_names = tuple(groups) if groups is not None else V3_STRATEGY_GROUPS
    counters: Dict[str, int] = {group: 0 for group in group_names}

    for group in group_names:
        for item in grouped.get(group, []):
            strategy = str(item.get("strategy", "candidate"))
            value = int(item.get("value", 0))
            if value in blocked_values:
                continue
            counters[group] = counters.get(group, 0) + 1
            new_payload = write_value(payload, unit, value, byteorder)
            value_hex = format_value_hex(value, width)
            variants.append({
                "name": f"{group}_{strategy}_r{round_id}_{counters[group]:02d}",
                "strategy_group": group,
                "strategy": strategy,
                "payload_hex": bytes_to_hex(new_payload),
                "field_range": {"a": int(unit["a"]), "b": int(unit["b"])},
                "field_unit": unit,
                "field_id": str(unit["repr"]),
                "requested_value_hex": value_hex,
                "final_value_hex": value_hex,
                "constraint_guided": True,
            })
    return variants


def choose_field_unit(units: List[dict], seed: int, field_spec: str, outdir: str) -> dict:
    """选择一个统一字段单元，支持整字节字段和位子字段。"""
    if field_spec:
        for unit in units:
            normalized = normalize_unit(unit)
            if field_id(normalized) == field_spec:
                return normalized
            if normalized.get("kind") == "byte":
                a = int(normalized["a"])
                b = int(normalized["b"])
                if field_spec in {str(a), f"{a},{b}", f"{a}_{b}"}:
                    return normalized
        raise RuntimeError(f"field not found in fields.json: {field_spec}")

    existing = set()
    for item in read_existing_mutation_entries(outdir):
        field = item.get("field") if isinstance(item, dict) else None
        if isinstance(field, dict) and "a" in field and "b" in field:
            existing.add(field_id(field))
    candidates = [normalize_unit(unit) for unit in units if field_id(unit) not in existing]
    if not candidates:
        raise RuntimeError("no available fields after excluding existing mutations.json field")
    salt = sum(sum(ord(ch) for ch in item) for item in existing)
    rnd = random.Random(seed + salt)
    return rnd.choice(candidates)


def read_existing_mutation_entries(outdir: str) -> List[dict]:
    path = os.path.join(outdir, "mutations.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    data = read_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _validate_mutations(payload: bytes, a: int, b: int, variants: List[dict]) -> None:
    payload_hexes = [item["payload_hex"] for item in variants]

    for item in variants:
        new_payload = hex_to_bytes(item["payload_hex"])
        if len(new_payload) != len(payload):
            raise RuntimeError("mutation payload length mismatch")
        if new_payload[:a] != payload[:a] or new_payload[b + 1:] != payload[b + 1:]:
            raise RuntimeError(f"mutation modifies bytes outside field: {payload_hexes}")


def mutate_payload(payload: bytes, a: int, b: int, seed: int,
                   constraint_guided: bool = True,
                   baseline_log_path: str = "",
                   field_ranges: Optional[Sequence[Tuple[int, int]]] = None,
                   byteorder: str = "big",
                   group_topk: int = 6,
                   group_min_candidates: int = 4,
                   field_unit: Optional[dict] = None,
                   constraint_values: Optional[Sequence[int]] = None) -> List[dict]:
    """对指定字段生成变体 payload"""
    unit = normalize_unit(field_unit) if field_unit is not None else {"kind": "byte", "a": a, "b": b, "repr": f"{a}" if a == b else f"{a},{b}"}
    a = int(unit["a"])
    b = int(unit["b"])
    if a < 0 or b >= len(payload) or a > b:
        raise ValueError("field range out of payload bounds")

    field_int = read_value(payload, unit, byteorder)
    width_bits = unit_width_bits(unit)
    if constraint_values is not None:
        e0 = [int(value) for value in constraint_values]
    elif unit.get("kind") == "bit":
        e0 = values_from_bit_constraints(unit)
    else:
        e0 = (
            extract_field_compare_immediates(
                baseline_log_path, a, b, width_bits, field_ranges=field_ranges
            )
            if constraint_guided else []
        )
    grouped = build_v3_grouped_candidates(
        base_value=field_int,
        width_bits=width_bits,
        immediates=e0,
        topk_per_group=max(1, int(group_topk)),
        min_per_group=max(1, int(group_min_candidates)),
    )
    variants = build_v3_mutations_for_unit(
        payload=payload,
        field_unit=unit,
        grouped=grouped,
        existing_values=set(),
        round_id=1,
        byteorder=byteorder,
    )
    if not variants:
        raise RuntimeError(
            "no mutation candidates remain after filtering baseline-equal values; "
            "try increasing --group-topk or --group-min-candidates"
        )

    _validate_mutations(payload, a, b, variants)

    return variants


def cmd_mutate(args: argparse.Namespace) -> None:
    """mutate 子命令主逻辑"""
    ensure_dir(args.outdir)
    sample = load_sample(args.outdir)
    payload = hex_to_bytes(sample["payload_hex"])
    fields_json = load_fields(args.outdir)
    units = units_from_fields_json(fields_json)
    if not units:
        raise RuntimeError("fields list is empty")

    field_unit = choose_field_unit(units, args.seed, args.field, args.outdir)
    a = int(field_unit["a"])
    b = int(field_unit["b"])
    field_ranges = byte_ranges_from_units(units)
    baseline_log_path = os.path.join(args.outdir, "mutations", "baseline.log")
    mutations = mutate_payload(
        payload, a, b, args.seed,
        constraint_guided=args.constraint_guided,
        baseline_log_path=baseline_log_path,
        field_ranges=field_ranges,
        byteorder="big",
        group_topk=args.group_topk,
        group_min_candidates=args.group_min_candidates,
        field_unit=field_unit,
        constraint_values=values_from_bit_constraints(field_unit) if field_unit.get("kind") == "bit" else None,
    )

    selection = {
        "field": field_unit,
        "mutations": mutations,
        "seed": args.seed,
        "timestamp": time.time(),
    }

    mutations_path = os.path.join(args.outdir, "mutations.json")
    if os.path.exists(mutations_path):
        if os.path.getsize(mutations_path) == 0:
            selections = []
        else:
            existing_data = read_json(mutations_path)
            if isinstance(existing_data, list):
                selections = existing_data
            elif isinstance(existing_data, dict):
                selections = [existing_data]
            else:
                raise RuntimeError("mutations.json format is invalid")
        selections.append(selection)
        write_json(mutations_path, selections)
    else:
        write_json(mutations_path, [selection])
    print(field_id(selection["field"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 payload 变体"
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="输出目录（需包含 sample.json 和 fields.json）")
    parser.add_argument("--field",
                        help="指定字段范围 a,b")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子")
    parser.add_argument("--constraint-guided", dest="constraint_guided",
                        action="store_true", default=True,
                        help="启用基于执行约束的候选值生成（默认开启）")
    parser.add_argument("--no-constraint-guided", dest="constraint_guided",
                        action="store_false",
                        help="禁用基于执行约束的候选值生成")
    parser.add_argument("--group-topk", type=int, default=6,
                        help="每个策略组候选上限（默认 6）")
    parser.add_argument("--group-min-candidates", type=int, default=6,
                        help="兼容参数；当前 V3 按 --group-topk 补满每个策略组")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cmd_mutate(args)
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
