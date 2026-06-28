#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fields.py - 从污点日志中提取字段范围
解析 taint 日志，输出 fields.json
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from common import (
    DEFAULT_OUTDIR,
    ensure_dir,
    print_user_error,
    write_json,
)

# ============================================================
# 日志解析配置
# ============================================================

JUMP_INSTR_RE = re.compile(r"^\s*j[a-z]+\b", re.IGNORECASE)
CMP_INSTR_RE = re.compile(r"^\s*cmp\b", re.IGNORECASE)
TEST_INSTR_RE = re.compile(r"^\s*test\b", re.IGNORECASE)
SETCC_INSTR_RE = re.compile(r"^\s*set[a-z]+\b", re.IGNORECASE)
THREAD_PREFIX_RE = re.compile(r"^THREADID\t[^\t]+\t(.*)$")

# 默认行为
DEDUPLICATE_BY_ADDR = False
MERGE_SUBFIELDS = False
REMOVE_COVERED_RANGES = False


EFFECTIVE_VALUE_OPS = {
    "add", "adc", "sub", "sbb",
    "imul", "mul", "idiv", "div",
    "and", "or", "xor",
    "shl", "shr", "sar", "sal",
    "rol", "ror", "rcl", "rcr",
    "inc", "dec", "neg", "not",
    "lea",
}

MOVE_VALUE_OPS = {
    "mov", "movzx", "movsx", "movsxd", "movss", "movsd",
}

TYPED_FLOAT_LOAD_WIDTHS = {
    "movss": 4,
    "movsd": 8,
}

LEVEL_COMPARE = 3
LEVEL_COMPUTE = 2
LEVEL_MOVE = 1


# ============================================================
# 日志解析函数
# ============================================================

def parse_instruction_line(line: str):
    """解析 Instruction 行"""
    line = line.rstrip("\n")
    match = THREAD_PREFIX_RE.match(line)
    if match:
        line = match.group(1)

    if not line.startswith("Instruction"):
        return None
    cols = line.split("\t")
    # 日志格式: Instruction\taddr: instr\tidx\tinfo
    if len(cols) < 2:
        return None
    addr_instr = cols[1].strip()  # "libsnap7+0x19e93: movzx r15d, byte ptr [rbp+0x189]"
    colon_pos = addr_instr.find(": ")
    if colon_pos == -1:
        return None
    addr = addr_instr[:colon_pos].strip()
    instr = addr_instr[colon_pos + 2:].strip()
    idx_raw = cols[2].strip() if len(cols) > 2 else ""
    orig = line
    is_jump = bool(JUMP_INSTR_RE.match(instr))
    is_cmp = bool(CMP_INSTR_RE.match(instr) or TEST_INSTR_RE.match(instr))
    is_setcc = bool(SETCC_INSTR_RE.match(instr))
    return addr, instr, idx_raw, orig, is_jump, is_cmp, is_setcc


def instruction_priority_level(instr: str, is_jump: bool, is_cmp: bool, is_setcc: bool) -> int:
    """返回字段证据的消费等级。

    3: 控制流相关消费（cmp/test/jump/setcc）
    2: 数值计算或位运算消费
    1: mov 类数据传播
    0: 不参与字段排序
    """
    if is_jump or is_cmp or is_setcc:
        return LEVEL_COMPARE
    mnemonic = instr.strip().split(" ", 1)[0].lower()
    if mnemonic in EFFECTIVE_VALUE_OPS:
        return LEVEL_COMPUTE
    if mnemonic in MOVE_VALUE_OPS:
        return LEVEL_MOVE
    return 0


def is_effective_instruction(instr: str, is_jump: bool, is_cmp: bool, is_setcc: bool) -> bool:
    """判断指令是否应参与字段统计。"""
    return instruction_priority_level(instr, is_jump, is_cmp, is_setcc) > 0


def is_typed_float_memory_load(instr: str, width_bytes: int) -> bool:
    """判断指令是否从内存整体读取一个定宽浮点值。"""
    operands = instr.strip().split(None, 1)
    if len(operands) != 2:
        return False
    mnemonic = operands[0].lower()
    if TYPED_FLOAT_LOAD_WIDTHS.get(mnemonic) != width_bytes:
        return False
    dst_src = operands[1].split(",", 1)
    return len(dst_src) == 2 and "[" in dst_src[1]


def extract_numbers_list(s: str) -> List[int]:
    """从字符串中提取数字列表"""
    if not s:
        return []
    nums = set()
    field_part = s.strip()
    if not field_part:
        return []
    paren = re.findall(r"\((\s*\d+\s*)[,:\-]\s*(\d+\s*)\)", field_part)
    if paren:
        a, b = int(paren[0][0]), int(paren[0][1])
        lo, hi = min(a, b), max(a, b)
        return list(range(lo, hi + 1))
    for a, b in re.findall(r"\b(\d+)\s*-\s*(\d+)\b", field_part):
        a, b = int(a), int(b)
        lo, hi = min(a, b), max(a, b)
        nums.update(range(lo, hi + 1))
    for num in re.findall(r"\d+", field_part):
        nums.add(int(num))
    return sorted(nums)


def split_contiguous_runs(indices: List[int]) -> List[List[int]]:
    """将索引列表拆分为若干个连续子段。

    例如:
    - [3, 8] -> [[3], [8]]
    - [3, 4, 8] -> [[3, 4], [8]]
    - [12, 13, 14] -> [[12, 13, 14]]
    """
    if not indices:
        return []

    sorted_indices = sorted(set(indices))
    runs: List[List[int]] = []
    current_run = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        if idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]

    runs.append(current_run)
    return runs


def split_range_by_covered_bytes(a: int, b: int, covered_bytes: set) -> List[Tuple[int, int]]:
    """将字段范围按已覆盖字节切分，只保留未覆盖的连续子段。

    例如：
    - field=2,3，covered={2} -> [(3,3)]
    - field=27,30，covered={27} -> [(28,30)]
    - field=10,14，covered={11,13} -> [(10,10), (12,12), (14,14)]
    """
    remaining = [i for i in range(a, b + 1) if i not in covered_bytes]
    runs = split_contiguous_runs(remaining)
    return [(run[0], run[-1]) for run in runs if run]


def extract_number_groups(s: str) -> List[List[int]]:
    """按分号切分为多个字段范围，并进一步按连续性拆分。

    只有连续字节序列才会被视为同一个多字节字段。
    非连续组合会拆成多个字段。
    """
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    groups: List[List[int]] = []
    for part in parts:
        nums = extract_numbers_list(part)
        if nums:
            groups.extend(split_contiguous_runs(nums))
    return groups


def field_key_from_indices(indices: List[int]) -> str:
    """从索引列表生成字段键"""
    if not indices:
        return "unknown"
    if len(indices) == 1:
        return str(indices[0])
    return f"{min(indices)}-{max(indices)}"


def merge_subfields_into_ranges(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """将子字段合并到范围"""
    field_indices = {}
    for key in groups.keys():
        if key == "unknown":
            field_indices[key] = set()
            continue
        if "-" in key and all(p.isdigit() for p in key.split("-")):
            a, b = map(int, key.split("-"))
            field_indices[key] = set(range(min(a, b), max(a, b) + 1))
        elif key.isdigit():
            field_indices[key] = {int(key)}
        else:
            field_indices[key] = set()
    keys_to_delete = set()
    for k1, s1 in field_indices.items():
        if not s1 or k1 == "unknown":
            continue
        for k2, s2 in field_indices.items():
            if k1 == k2:
                continue
            if "-" in k2 and s1.issubset(s2):
                keys_to_delete.add(k1)
                break
    merged = defaultdict(list)
    for key, lst in groups.items():
        if key in keys_to_delete:
            s1 = field_indices[key]
            for k2, s2 in field_indices.items():
                if k2 != key and "-" in k2 and s1.issubset(s2):
                    merged[k2].extend(lst)
                    break
        else:
            merged[key].extend(lst)
    return dict(merged)


def remove_covered_field_ranges(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """移除被覆盖的字段范围"""
    field_indices = {}
    for key in groups.keys():
        if key == "unknown":
            field_indices[key] = set()
            continue
        if "-" in key and all(p.isdigit() for p in key.split("-")):
            a, b = map(int, key.split("-"))
            field_indices[key] = set(range(min(a, b), max(a, b) + 1))
        elif key.isdigit():
            field_indices[key] = {int(key)}
        else:
            field_indices[key] = set()
    keys_to_delete = set()
    for k1, s1 in field_indices.items():
        if not s1 or k1 == "unknown":
            continue
        for k2, s2 in field_indices.items():
            if k1 == k2:
                continue
            if s1.issubset(s2) and s1 != s2:
                keys_to_delete.add(k1)
                break
    result = {}
    for key, lst in groups.items():
        if key not in keys_to_delete:
            result[key] = lst
    return result


def process_file_once(filepath: str) -> Dict[str, List[Tuple[str, int]]]:
    """处理日志文件，返回字段分组及其证据等级。"""
    groups = defaultdict(list)
    seen_addrs = set()
    last_cmp_field_key = "unknown"
    has_function_since_last_cmp = False
    taint_found = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            marker_line = line.rstrip("\n")
            match = THREAD_PREFIX_RE.match(marker_line)
            if match:
                marker_line = match.group(1)
            if marker_line.startswith("Taint\t"):
                taint_found = True
                continue
            if not taint_found:
                continue

            parsed = parse_instruction_line(line)
            if not parsed:
                continue
            addr, instr, idx_raw, orig, is_jump, is_cmp, is_setcc = parsed

            if DEDUPLICATE_BY_ADDR and addr in seen_addrs:
                continue
            seen_addrs.add(addr)

            if not is_effective_instruction(instr, is_jump, is_cmp, is_setcc):
                continue

            groups_list = extract_number_groups(idx_raw)
            if not groups_list:
                groups_list = [[]]

            for indices in groups_list:
                key = field_key_from_indices(indices)
                level = instruction_priority_level(instr, is_jump, is_cmp, is_setcc)

                if (is_jump or is_setcc) and key == "unknown":
                    if not has_function_since_last_cmp:
                        key = last_cmp_field_key
                elif is_cmp:
                    last_cmp_field_key = key
                    has_function_since_last_cmp = False

                groups[key].append((orig, level))

    if MERGE_SUBFIELDS:
        groups = merge_subfields_into_ranges(groups)

    if REMOVE_COVERED_RANGES:
        groups = remove_covered_field_ranges(groups)

    return groups


# ============================================================
# 字段提取核心逻辑
# ============================================================

def extract_fields_from_log(log_path: str) -> List[Tuple[int, int]]:
    """
    从日志中提取字段范围，按消费等级和出现次数排序，去除重复覆盖的字段。
    
    逻辑：
    1. 统计每个字段范围的最高消费等级及出现次数
    2. 先按消费等级降序，再按频率降序排序
    3. 如果当前字段的字节已在之前输出的字段中出现过，则仅保留未覆盖连续子段
    """
    result, _ = extract_fields_with_counts(log_path)
    return result


def _build_field_stats(
    groups: Dict[str, List[Tuple[str, int]]]
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """将字段分组转换为排序所需的统计信息。"""
    field_stats: Dict[Tuple[int, int], Dict[str, int]] = {}
    for key, occurrences in groups.items():
        if key == "unknown":
            continue
        if "-" in key:
            a_str, b_str = key.split("-", 1)
            if not (a_str.isdigit() and b_str.isdigit()):
                continue
            a = int(a_str)
            b = int(b_str)
            field = (min(a, b), max(a, b))
        elif key.isdigit():
            v = int(key)
            field = (v, v)
        else:
            continue

        level_counts = {
            LEVEL_COMPARE: 0,
            LEVEL_COMPUTE: 0,
            LEVEL_MOVE: 0,
        }
        for _, level in occurrences:
            if level in level_counts:
                level_counts[level] += 1
        max_level = 0
        count_at_max_level = 0
        for level in (LEVEL_COMPARE, LEVEL_COMPUTE, LEVEL_MOVE):
            if level_counts[level] > 0:
                max_level = level
                count_at_max_level = level_counts[level]
                break

        field_stats[field] = {
            "max_level": max_level,
            "effective_level": max_level,
            "count_at_max_level": count_at_max_level,
            "total_count": len(occurrences),
            "compare_count": level_counts[LEVEL_COMPARE],
            "compute_count": level_counts[LEVEL_COMPUTE],
            "move_count": level_counts[LEVEL_MOVE],
            "assembled_full_value": 0,
            "typed_float_consumption": 0,
        }

    # `movss` / `movsd` 从内存读取时具有明确的定宽浮点语义。
    # 与普通 dword/qword 块搬运不同，这足以将完整范围提升为强字段候选。
    for field, stats in field_stats.items():
        a, b = field
        width_bytes = b - a + 1
        occurrences = groups.get(field_key_from_indices(list(range(a, b + 1))), [])
        if any(
            is_typed_float_memory_load(
                parse_instruction_line(orig)[1], width_bytes
            )
            for orig, _ in occurrences
            if parse_instruction_line(orig)
        ):
            stats["typed_float_consumption"] = 1
            stats["assembled_full_value"] = 1
            stats["effective_level"] = max(
                int(stats.get("max_level", 0)), LEVEL_COMPUTE
            )

    # “组装完成奖励”：
    # 对连续多字节字段，只有在“完整字段整体落地”之外，
    # 还观察到沿字段边界逐步扩展的 compute 级组装痕迹时，
    # 才给完整字段额外优先级。
    #
    # 这里刻意不把“孤立的大块 mov/word/dword/qword 搬运”当作充分证据。
    # 否则容易把实现层面的结构体块拷贝，误当成一个真实的协议字段。
    # 典型正例：CIP 中 byte -> shl -> or -> dword 的逐步组装。
    # 典型反例：Snap7 / IEC104 中已经存在稳定子字段时，又出现更大块的联合搬运。
    for field, stats in field_stats.items():
        a, b = field
        if b <= a:
            continue

        occurrences = groups.get(field_key_from_indices(list(range(a, b + 1))), [])
        has_full_landing = False
        for orig, level in occurrences:
            parsed = parse_instruction_line(orig)
            if not parsed:
                continue
            _, instr, _, _, is_jump, is_cmp, is_setcc = parsed
            mnemonic = instr.strip().split(" ", 1)[0].lower()
            if is_cmp or is_setcc or mnemonic in MOVE_VALUE_OPS:
                has_full_landing = True
                break
        if not has_full_landing:
            continue

        boundary_compute_subfields = []
        for subfield, substats in field_stats.items():
            sa, sb = subfield
            if subfield == field:
                continue
            if sa < a or sb > b:
                continue
            # 这里只承认 compute 级的边界子段为“真实组装链”。
            # 单纯 compare 或 move 痕迹不足以证明完整字段是由这些子段逐步拼装出来的。
            if int(substats.get("compute_count", 0)) < 1:
                continue
            # 只奖励“沿着完整字段边界逐步扩展”的连续子段，
            # 避免普通内部片段也被误判成完整组装链。
            if sa == a or sb == b:
                boundary_compute_subfields.append(subfield)
        if len(boundary_compute_subfields) >= 2:
            stats["assembled_full_value"] = 1
            # 完整长字段虽然常常只在最终 mov/store 中整体落地，
            # 但其前面已经出现了明确的中等级组装痕迹。
            # 因此这里至少将其排序等级抬到 compute 级，
            # 从而不再被单字节零件字段天然压制。
            stats["effective_level"] = max(int(stats.get("max_level", 0)), LEVEL_COMPUTE)
    return field_stats


def _sort_field_stats(
    field_stats: Dict[Tuple[int, int], Dict[str, int]]
) -> List[Tuple[Tuple[int, int], Dict[str, int]]]:
    """按消费等级优先、同等级内频率优先、最后短字段优先排序。"""
    return sorted(
        field_stats.items(),
        key=lambda item: (
            -int(item[1].get("effective_level", item[1]["max_level"])),
            -int(item[1].get("assembled_full_value", 0)),
            -int(item[1]["count_at_max_level"]),
            -int(item[1]["total_count"]),
            (item[0][1] - item[0][0]),
            item[0][0],
            item[0][1],
        ),
    )


def _find_joint_use_suppressed_fields(
    field_stats: Dict[Tuple[int, int], Dict[str, int]]
) -> set:
    """识别应被压制的“大字段联合使用”候选。

    目标是避免把两个已经稳定成立的相邻子字段，仅因为后续一起参与
    lea/add/sub/cmp 等联合计算，就误提升成一个更大的原始字段。

    当前仅做非常收敛的特判：
    - 父字段长度至少 4 字节，且能恰好均分成两个相邻子字段；
    - 两个子字段长度至少 2 字节；
    - 两个子字段本身已经稳定成立（有 compare 证据，或组装完成且证据数量足够）；
    - 父字段自身也有较强证据，但这类证据可能只是后续联合计算/边界检查。
    """
    suppressed = set()
    for (a, b), parent_stats in field_stats.items():
        width_bytes = b - a + 1
        if width_bytes < 4 or (width_bytes % 2) != 0:
            continue

        left = (a, a + width_bytes // 2 - 1)
        right = (left[1] + 1, b)
        if left not in field_stats or right not in field_stats:
            continue

        left_stats = field_stats[left]
        right_stats = field_stats[right]
        left_width = left[1] - left[0] + 1
        right_width = right[1] - right[0] + 1
        if left_width < 2 or right_width < 2:
            continue

        def is_stable_child(stats: Dict[str, int]) -> bool:
            return (
                int(stats.get("compare_count", 0)) >= 1
                or (
                    int(stats.get("assembled_full_value", 0)) >= 1
                    and int(stats.get("effective_level", stats.get("max_level", 0))) >= LEVEL_COMPUTE
                    and int(stats.get("total_count", 0)) >= 3
                )
            )

        if not (is_stable_child(left_stats) and is_stable_child(right_stats)):
            continue

        parent_compare = int(parent_stats.get("compare_count", 0))
        parent_total = int(parent_stats.get("total_count", 0))
        if parent_compare < 1 and parent_total < 4:
            continue

        suppressed.add((a, b))

    return suppressed


def extract_fields_with_counts(log_path: str) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    从日志中提取字段范围，返回字段列表和所有字段的计数信息。
    
    Returns:
        (fields, all_counts): fields 是过滤后的字段列表，all_counts 是所有字段的出现次数
    """
    groups = process_file_once(log_path)
    field_stats = _build_field_stats(groups)
    suppressed_fields = _find_joint_use_suppressed_fields(field_stats)
    field_counts: Dict[Tuple[int, int], int] = {
        field: int(stats["total_count"]) for field, stats in field_stats.items()
    }
    sorted_fields = _sort_field_stats(field_stats)
    
    # 过滤：如果当前字段与已输出字段部分重叠，则仅保留未覆盖部分的连续子段
    covered_bytes: set = set()
    result: List[Tuple[int, int]] = []
    
    for (a, b), stats in sorted_fields:
        if (a, b) in suppressed_fields:
            continue
        count = int(stats["total_count"])
        residual_ranges = split_range_by_covered_bytes(a, b, covered_bytes)
        if not residual_ranges:
            continue
        for residual in residual_ranges:
            result.append(residual)
            covered_bytes.update(range(residual[0], residual[1] + 1))
            if residual not in field_counts:
                field_counts[residual] = count
    
    # 过滤后的字段按边界先后排序输出
    result = sorted(result, key=lambda x: (x[0], x[1]))
    return result, field_counts


# ============================================================
# JSON 输出和命令行接口
# ============================================================

def fields_to_json(fields: List[Tuple[int, int]], counts: Dict[Tuple[int, int], int] = None) -> dict:
    """将字段列表转换为 JSON 格式"""
    result = []
    for a, b in fields:
        entry = {"a": a, "b": b, "repr": f"{a}" if a == b else f"{a},{b}"}
        if counts and (a, b) in counts:
            entry["count"] = counts[(a, b)]
        result.append(entry)
    return {"fields": result}


def cmd_fields(args: argparse.Namespace) -> None:
    """fields 子命令主逻辑"""
    if args.verbose:
        groups = process_file_once(args.log)
        field_stats = _build_field_stats(groups)
        fields, all_counts = extract_fields_with_counts(args.log)
        # 按消费等级优先、同等级内频率优先显示字段统计
        print("=== 字段出现次数统计 ===")
        sorted_counts = _sort_field_stats(field_stats)
        for (a, b), stats in sorted_counts:
            count = int(stats["total_count"])
            field_repr = f"{a}" if a == b else f"{a},{b}"
            print(
                f"  {field_repr}: total={count}, "
                f"level={stats['max_level']}, "
                f"effective={stats.get('effective_level', stats['max_level'])}, "
                f"assembled={stats.get('assembled_full_value', 0)}, "
                f"typed_float={stats.get('typed_float_consumption', 0)}, "
                f"cmp={stats['compare_count']}, "
                f"compute={stats['compute_count']}, "
                f"move={stats['move_count']}"
            )
        print(f"\n=== 过滤后的字段 ({len(fields)} 个) ===")
    else:
        fields = extract_fields_from_log(args.log)
        all_counts = None
    
    if not fields:
        raise RuntimeError("no fields extracted from log")
    
    fields_json = fields_to_json(fields, all_counts if args.verbose else None)
    ensure_dir(args.outdir)
    write_json(os.path.join(args.outdir, "fields.json"), fields_json)

    for f in fields_json["fields"]:
        if args.verbose and "count" in f:
            print(f"{f['repr']} (count={f['count']})")
        else:
            print(f["repr"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从污点日志中提取字段范围"
    )
    parser.add_argument("--log", required=True,
                        help="要解析的日志文件路径")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="输出目录")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示字段出现次数统计")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cmd_fields(args)
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
