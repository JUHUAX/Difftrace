#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_bitfields_planA.py

独立的位字段识别、划分和约束提取实验脚本。

目标：
1. 识别哪些字段是疑似位字段
2. 恢复候选子字段（bit / bit-range）
3. 提取与子字段相关的约束与来源指令

说明：
- 本脚本不接入 full.py 流程，只作为独立实验工具。
- 只依赖动态执行日志、字段划分和指令语义，不依赖协议先验。
- 优先保证“结果可解释”，每个结论都尽量回溯到具体指令。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


THREAD_PREFIX_RE = re.compile(r"^THREADID\t([^\t]+)\t(.*)$")
INSTR_PREFIX = "Instruction\t"
BB_PREFIX = "BasicBlock\t"
TAINT_PREFIX = "Taint\t"

JCC_RE = re.compile(r"^\s*j([a-z]+)\b", re.IGNORECASE)
SETCC_RE = re.compile(r"^\s*set[a-z]+\b", re.IGNORECASE)
IMM_HEX_RE = re.compile(r"(?<![0-9A-Za-z_])-?0x[0-9a-fA-F]+")
IMM_DEC_RE = re.compile(r"(?<![0-9A-Za-z_])-?\d+")
INFO_VALUE_RE = re.compile(r"([A-Za-z0-9_*]+)\s*=\s*([^;\t]+)")

BIT_OPS = {"and", "or", "xor", "shr", "sar", "shl", "sal", "rol", "ror", "rcl", "rcr"}
VALUE_OPS = BIT_OPS | {"cmp", "test", "add", "adc", "sub", "sbb"}
USEFUL_OPS = VALUE_OPS | {"mov", "movzx", "movsx", "movsxd", "lea", "seta", "setb", "sete", "setne"}
RECOVERY_MODES = ("full", "operation_driven", "flat_evidence")

MEM_WIDTH_HINTS = {
    "dword ptr": 32,
    "qword ptr": 64,
    "word ptr": 16,
    "byte ptr": 8,
}


@dataclass(frozen=True)
class FieldRef:
    field_id: str
    byte_offset_start: int
    byte_offset_end: int
    bit_width: int


@dataclass(frozen=True)
class BitOrigin:
    kind: str
    field_id: Optional[str] = None
    source_bit_index: Optional[int] = None

    def to_json(self) -> dict:
        data = {"kind": self.kind}
        if self.field_id is not None:
            data["field_id"] = self.field_id
        if self.source_bit_index is not None:
            data["source_bit_index"] = self.source_bit_index
        return data


CONST0 = BitOrigin("CONST0")
CONST1 = BitOrigin("CONST1")
UNKNOWN = BitOrigin("UNKNOWN")


@dataclass
class BitOriginMap:
    width_bits: int
    origins: List[BitOrigin]

    @classmethod
    def unknown(cls, width_bits: int) -> "BitOriginMap":
        return cls(width_bits=width_bits, origins=[UNKNOWN] * width_bits)

    @classmethod
    def const_zero(cls, width_bits: int) -> "BitOriginMap":
        return cls(width_bits=width_bits, origins=[CONST0] * width_bits)

    def clone(self) -> "BitOriginMap":
        return BitOriginMap(self.width_bits, list(self.origins))

    def slice(self, offset: int, width_bits: int) -> "BitOriginMap":
        sl = self.origins[offset:offset + width_bits]
        if len(sl) < width_bits:
            sl = sl + [UNKNOWN] * (width_bits - len(sl))
        return BitOriginMap(width_bits=width_bits, origins=sl)

    def assign_slice(self, offset: int, other: "BitOriginMap") -> None:
        for i in range(other.width_bits):
            pos = offset + i
            if 0 <= pos < self.width_bits:
                self.origins[pos] = other.origins[i]

    def used_source_bits(self, field_id: Optional[str] = None, bit_mask: Optional[int] = None) -> List[int]:
        bits: List[int] = []
        for i, origin in enumerate(self.origins):
            if bit_mask is not None and ((bit_mask >> i) & 1) == 0:
                continue
            if origin.kind != "SOURCE":
                continue
            if field_id is not None and origin.field_id != field_id:
                continue
            if origin.source_bit_index is not None:
                bits.append(origin.source_bit_index)
        return sorted(set(bits))


@dataclass
class UseEvent:
    field_id: str
    trace_line: int
    address: str
    instruction: str
    event_kind: str
    used_source_bits: List[int]
    mask: Optional[int] = None
    shift: Optional[int] = None
    compare_value: Optional[int] = None
    branch_taken: Optional[bool] = None
    normalized_constraint: Optional[str] = None
    structured_constraint: Optional[dict] = None
    raw_operands: str = ""
    explanation: str = ""
    variable_shift: bool = False

    def to_json(self) -> dict:
        return {
            "field_id": self.field_id,
            "trace_line": self.trace_line,
            "address": self.address,
            "instruction": self.instruction,
            "event_kind": self.event_kind,
            "used_source_bits": self.used_source_bits,
            "mask": self.mask,
            "shift": self.shift,
            "compare_value": self.compare_value,
            "branch_taken": self.branch_taken,
            "normalized_constraint": self.normalized_constraint,
            "structured_constraint": self.structured_constraint,
            "raw_operands": self.raw_operands,
            "explanation": self.explanation,
            "variable_shift": self.variable_shift,
        }


@dataclass
class CandidateSubfield:
    field_id: str
    bits: List[int]
    kind: str
    evidence_events: List[int] = field(default_factory=list)
    constraints: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    source_event_kinds: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "label": bits_to_layout_label(self.bits),
            "constraints": self.constraints,
        }


@dataclass
class ParsedInstruction:
    trace_line: int
    address: str
    disasm: str
    idx_raw: str
    info_raw: str
    payload: str

    @property
    def mnemonic(self) -> str:
        return self.disasm.split(" ", 1)[0].lower() if self.disasm else ""

    @property
    def operands_text(self) -> str:
        return self.disasm.split(" ", 1)[1] if " " in self.disasm else ""


@dataclass(frozen=True)
class BitfieldRecoveryConfig:
    mode: str = "full"
    use_consumption_guidance: bool = True
    resolve_hierarchical_evidence: bool = True


def recovery_config(mode: str) -> BitfieldRecoveryConfig:
    if mode == "full":
        return BitfieldRecoveryConfig()
    if mode == "operation_driven":
        return BitfieldRecoveryConfig(
            mode=mode,
            use_consumption_guidance=False,
        )
    if mode == "flat_evidence":
        return BitfieldRecoveryConfig(
            mode=mode,
            resolve_hierarchical_evidence=False,
        )
    raise ValueError(f"unsupported recovery mode: {mode!r}")


REG_ALIASES: Dict[str, Tuple[str, int, int]] = {
    "rax": ("rax", 64, 0), "eax": ("rax", 32, 0), "ax": ("rax", 16, 0), "al": ("rax", 8, 0), "ah": ("rax", 8, 8),
    "rbx": ("rbx", 64, 0), "ebx": ("rbx", 32, 0), "bx": ("rbx", 16, 0), "bl": ("rbx", 8, 0), "bh": ("rbx", 8, 8),
    "rcx": ("rcx", 64, 0), "ecx": ("rcx", 32, 0), "cx": ("rcx", 16, 0), "cl": ("rcx", 8, 0), "ch": ("rcx", 8, 8),
    "rdx": ("rdx", 64, 0), "edx": ("rdx", 32, 0), "dx": ("rdx", 16, 0), "dl": ("rdx", 8, 0), "dh": ("rdx", 8, 8),
    "rsi": ("rsi", 64, 0), "esi": ("rsi", 32, 0), "si": ("rsi", 16, 0), "sil": ("rsi", 8, 0),
    "rdi": ("rdi", 64, 0), "edi": ("rdi", 32, 0), "di": ("rdi", 16, 0), "dil": ("rdi", 8, 0),
    "rbp": ("rbp", 64, 0), "ebp": ("rbp", 32, 0), "bp": ("rbp", 16, 0), "bpl": ("rbp", 8, 0),
    "rsp": ("rsp", 64, 0), "esp": ("rsp", 32, 0), "sp": ("rsp", 16, 0), "spl": ("rsp", 8, 0),
}
for i in range(8, 16):
    REG_ALIASES[f"r{i}"] = (f"r{i}", 64, 0)
    REG_ALIASES[f"r{i}d"] = (f"r{i}", 32, 0)
    REG_ALIASES[f"r{i}w"] = (f"r{i}", 16, 0)
    REG_ALIASES[f"r{i}b"] = (f"r{i}", 8, 0)


def load_fields(path: str) -> List[FieldRef]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("fields", [])
    fields: List[FieldRef] = []
    for item in items:
        a = int(item["a"])
        b = int(item["b"])
        field_id = str(item.get("repr", f"{a}" if a == b else f"{a},{b}"))
        bit_width = (b - a + 1) * 8
        fields.append(FieldRef(field_id, a, b, bit_width))
    return fields


def split_thread_prefix(line: str) -> Tuple[Optional[str], str]:
    match = THREAD_PREFIX_RE.match(line.rstrip("\n"))
    if not match:
        return None, line.rstrip("\n")
    return match.group(1), match.group(2)


def parse_indices(idx_raw: str) -> Set[int]:
    idx_set: Set[int] = set()
    if not idx_raw or idx_raw.strip() == "-":
        return idx_set
    for a, b in re.findall(r"\((\d+)\s*[,:\-]\s*(\d+)\)|(\d+)\s*-\s*(\d+)", idx_raw):
        left = a or b
    for token in re.findall(r"\d+", idx_raw):
        idx_set.add(int(token))
    return idx_set


def parse_field_tokens(idx_raw: str) -> List[str]:
    if not idx_raw or idx_raw.strip() == "-":
        return []
    tokens: List[str] = []
    for part in idx_raw.split(";"):
        token = part.strip()
        if not token or token == "-":
            continue
        tokens.append(token)
    return tokens


def token_byte_span(token: str) -> Optional[Tuple[int, int]]:
    nums = [int(x) for x in re.findall(r"\d+", token)]
    if not nums:
        return None
    uniq = sorted(set(nums))
    if uniq[-1] - uniq[0] + 1 != len(uniq):
        return None
    return uniq[0], uniq[-1]


def parse_info_values(info_raw: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    if not info_raw:
        return result
    for lhs, rhs in INFO_VALUE_RE.findall(info_raw):
        token = rhs.strip()
        try:
            if token.lower().startswith("-0x") or token.lower().startswith("0x"):
                result[lhs] = int(token, 16)
            else:
                result[lhs] = int(token, 10)
        except ValueError:
            continue
    return result


def parse_non_tainted_runtime_value(info_raw: str) -> Optional[int]:
    """从日志的 info_raw 中提取非污染操作数的运行时值。

    适配类似：
    - DST*=0x20;SRC=0xe0
    - DST=0xe0;SRC*=0x20

    其中带 * 的键表示污染字段值；不带 * 的键表示另一侧实际参与运算的值。
    这里返回第一个非污染值，用于按 imm-mask 语义处理 and/or/xor。
    """
    info_values = parse_info_values(info_raw)
    for key, value in info_values.items():
        if "*" not in key:
            return value
    return None


def parse_instruction(payload: str, line_no: int) -> Optional[ParsedInstruction]:
    if not payload.startswith(INSTR_PREFIX):
        return None
    cols = payload.split("\t")
    if len(cols) < 2:
        return None
    addr_disasm = cols[1].strip()
    colon = addr_disasm.find(": ")
    if colon < 0:
        return None
    return ParsedInstruction(
        trace_line=line_no,
        address=addr_disasm[:colon].strip(),
        disasm=addr_disasm[colon + 2:].strip(),
        idx_raw=cols[2].strip() if len(cols) > 2 else "",
        info_raw=cols[3].strip() if len(cols) > 3 else "",
        payload=payload,
    )


def split_operands(operands_text: str) -> List[str]:
    return [part.strip() for part in operands_text.split(",") if part.strip()]


def operand_is_memory(op: str) -> bool:
    return "[" in op and "]" in op


def operand_is_register(op: str) -> bool:
    return op.strip().lower() in REG_ALIASES


def operand_is_immediate(op: str) -> bool:
    op = op.strip()
    return bool(IMM_HEX_RE.fullmatch(op) or IMM_DEC_RE.fullmatch(op))


def parse_immediate(op: str) -> Optional[int]:
    op = op.strip()
    try:
        if op.lower().startswith("-0x") or op.lower().startswith("0x"):
            return int(op, 16)
        if IMM_DEC_RE.fullmatch(op):
            return int(op, 10)
    except ValueError:
        return None
    return None


def operand_width_bits(op: str) -> int:
    text = op.strip().lower()
    if text in REG_ALIASES:
        return REG_ALIASES[text][1]
    for hint, width in MEM_WIDTH_HINTS.items():
        if hint in text:
            return width
    return 0


def normalize_memory_key(op: str) -> str:
    text = op.strip().lower()
    for hint in MEM_WIDTH_HINTS:
        text = text.replace(hint, "")
    return re.sub(r"\s+", " ", text).strip()


def build_field_source_map(field_ref: FieldRef) -> BitOriginMap:
    origins: List[BitOrigin] = []
    for byte_off in range(field_ref.byte_offset_start, field_ref.byte_offset_end + 1):
        local_byte = byte_off - field_ref.byte_offset_start
        for bit in range(8):
            origins.append(BitOrigin("SOURCE", field_ref.field_id, local_byte * 8 + bit))
    return BitOriginMap(width_bits=field_ref.bit_width, origins=origins)


def low_extend(src: BitOriginMap, width_bits: int) -> BitOriginMap:
    if width_bits <= src.width_bits:
        return BitOriginMap(width_bits, list(src.origins[:width_bits]))
    return BitOriginMap(width_bits, list(src.origins) + [CONST0] * (width_bits - src.width_bits))


def sign_extend(src: BitOriginMap, width_bits: int) -> BitOriginMap:
    if width_bits <= src.width_bits:
        return BitOriginMap(width_bits, list(src.origins[:width_bits]))
    sign_origin = src.origins[src.width_bits - 1] if src.width_bits > 0 else UNKNOWN
    return BitOriginMap(width_bits, list(src.origins) + [sign_origin] * (width_bits - src.width_bits))


def shift_right(src: BitOriginMap, shift: int, arithmetic: bool) -> BitOriginMap:
    shift = max(0, int(shift))
    if shift == 0:
        return src.clone()
    fill = src.origins[-1] if arithmetic and src.width_bits > 0 else CONST0
    origins: List[BitOrigin] = []
    for i in range(src.width_bits):
        src_idx = i + shift
        origins.append(src.origins[src_idx] if src_idx < src.width_bits else fill)
    return BitOriginMap(src.width_bits, origins)


def shift_left(src: BitOriginMap, shift: int) -> BitOriginMap:
    shift = max(0, int(shift))
    if shift == 0:
        return src.clone()
    origins: List[BitOrigin] = []
    for i in range(src.width_bits):
        src_idx = i - shift
        origins.append(src.origins[src_idx] if src_idx >= 0 else CONST0)
    return BitOriginMap(src.width_bits, origins)


def apply_and_imm(src: BitOriginMap, imm: int) -> BitOriginMap:
    origins: List[BitOrigin] = []
    for i, origin in enumerate(src.origins):
        origins.append(origin if ((imm >> i) & 1) else CONST0)
    return BitOriginMap(src.width_bits, origins)


def apply_or_imm(src: BitOriginMap, imm: int) -> BitOriginMap:
    origins: List[BitOrigin] = []
    for i, origin in enumerate(src.origins):
        origins.append(CONST1 if ((imm >> i) & 1) else origin)
    return BitOriginMap(src.width_bits, origins)


def apply_xor_imm(src: BitOriginMap, imm: int) -> BitOriginMap:
    origins: List[BitOrigin] = []
    for i, origin in enumerate(src.origins):
        if ((imm >> i) & 1) == 0:
            origins.append(origin)
        elif origin.kind == "CONST0":
            origins.append(CONST1)
        elif origin.kind == "CONST1":
            origins.append(CONST0)
        else:
            origins.append(UNKNOWN)
    return BitOriginMap(src.width_bits, origins)


def contiguous_range(bits: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not bits:
        return None
    sorted_bits = sorted(set(bits))
    if sorted_bits[-1] - sorted_bits[0] + 1 != len(sorted_bits):
        return None
    return sorted_bits[0], sorted_bits[-1]


def contiguous_runs(bits: Sequence[int]) -> List[List[int]]:
    sorted_bits = sorted(set(bits))
    if not sorted_bits:
        return []
    runs: List[List[int]] = [[sorted_bits[0]]]
    for bit in sorted_bits[1:]:
        if bit == runs[-1][-1] + 1:
            runs[-1].append(bit)
        else:
            runs.append([bit])
    return runs


def subfield_evidence_strength(subfield: CandidateSubfield) -> int:
    event_kinds = set(subfield.source_event_kinds)
    if {"cmp", "test_mask", "test_reg"} & event_kinds:
        return 3
    if {"and", "or", "xor"} & event_kinds:
        return 2
    if {"shr", "sar", "shl", "sal", "mov_derived", "boundary_shrink"} & event_kinds:
        return 1
    return 0


def bits_to_label(bits: Sequence[int]) -> str:
    bits = sorted(set(bits))
    if not bits:
        return "unknown_bits"
    if len(bits) == 1:
        return f"bit{bits[0]}"
    span = contiguous_range(bits)
    if span is not None:
        lo, hi = span
        return f"bits[{hi}:{lo}]"
    return "bits{" + ",".join(str(x) for x in bits) + "}"


def bits_to_layout_label(bits: Sequence[int]) -> str:
    bits = sorted(set(bits))
    if not bits:
        return "[]"
    if len(bits) == 1:
        return f"[{bits[0]}]"
    span = contiguous_range(bits)
    if span is not None:
        lo, hi = span
        return f"[{hi}:{lo}]"
    return "[" + ",".join(str(x) for x in bits) + "]"


def format_recovered_layout(bit_width: int, subfields: Sequence[CandidateSubfield]) -> Tuple[List[str], str]:
    if not subfields:
        return [], ""
    ordered = sorted(subfields, key=lambda item: (-max(item.bits), -len(item.bits), item.bits))
    labels = [bits_to_layout_label(item.bits) for item in ordered]
    return labels, " ".join(labels)


def normalize_constraint(bits: Sequence[int], compare_value: Optional[int], op: str) -> str:
    label = bits_to_label(bits)
    if op == "zero":
        return f"{label} == 0"
    if op == "nonzero":
        return f"{label} != 0"
    if compare_value is None:
        return f"{label} {op} ?"
    return f"{label} {op} 0x{int(compare_value):x}"


def bits_mask(bits: Sequence[int]) -> int:
    mask = 0
    for bit in sorted(set(bits)):
        mask |= (1 << bit)
    return mask


def build_structured_constraint(
    bits: Sequence[int],
    op: str,
    compare_value: Optional[int] = None,
    explicit_mask: Optional[int] = None,
) -> dict:
    bits = sorted(set(bits))
    mask = int(explicit_mask if explicit_mask is not None else bits_mask(bits))
    label = bits_to_label(bits)
    constraint: dict = {
        "mask": f"0x{mask:x}",
        "operator": op,
        "value": f"0x{int(compare_value):x}" if compare_value is not None else None,
        "subfield_constraint": None,
    }

    span = contiguous_range(bits)
    shifted_compare_value: Optional[int] = None
    subfield_compare_value: Optional[int] = compare_value
    if compare_value is not None and span is not None:
        lo, hi = span
        subfield_width = hi - lo + 1
        max_subfield_value = (1 << subfield_width) - 1
        if 0 <= int(compare_value) <= max_subfield_value:
            shifted_compare_value = (int(compare_value) << lo) & mask
            subfield_compare_value = int(compare_value)
        else:
            shifted_compare_value = int(compare_value) & mask
            subfield_compare_value = (int(compare_value) & mask) >> lo
    elif compare_value is not None:
        shifted_compare_value = int(compare_value) & mask

    if op == "zero":
        field_constraint = f"(field & 0x{mask:x}) == 0x0"
    elif op == "nonzero":
        field_constraint = f"(field & 0x{mask:x}) != 0"
    elif op == "eq" and compare_value is not None:
        masked_value = shifted_compare_value if shifted_compare_value is not None else (int(compare_value) & mask)
        field_constraint = f"(field & 0x{mask:x}) == 0x{masked_value:x}"
    else:
        field_constraint = None

    if len(bits) == 1:
        bit = bits[0]
        if op == "zero":
            constraint["subfield_constraint"] = f"bit{bit} == 0"
        elif op == "nonzero":
            constraint["subfield_constraint"] = f"bit{bit} == 1"
        elif op == "eq" and compare_value is not None:
            bit_val = int(subfield_compare_value or 0) & 1
            constraint["subfield_constraint"] = f"bit{bit} == {bit_val}"
    elif span is not None:
        lo, hi = span
        span_mask = ((1 << (hi - lo + 1)) - 1) << lo
        if span_mask == mask:
            if op == "zero":
                constraint["subfield_constraint"] = f"bits[{hi}:{lo}] == 0"
            elif op == "nonzero":
                constraint["subfield_constraint"] = f"bits[{hi}:{lo}] != 0"
            elif op == "eq" and compare_value is not None:
                sub_value = int(subfield_compare_value or 0)
                constraint["subfield_constraint"] = f"bits[{hi}:{lo}] == {sub_value}"

    if constraint["subfield_constraint"] is None:
        if op == "zero":
            constraint["subfield_constraint"] = f"{label} == 0"
        elif op == "nonzero":
            constraint["subfield_constraint"] = f"{label} != 0"
        elif op == "eq" and compare_value is not None:
            constraint["subfield_constraint"] = f"{label} == 0x{(int(compare_value) & mask):x}"

    constraint["expression"] = constraint["subfield_constraint"] or field_constraint
    return constraint


def source_field_ids(bit_map: BitOriginMap) -> List[str]:
    ids: Set[str] = set()
    for origin in bit_map.origins:
        if origin.kind == "SOURCE" and origin.field_id is not None:
            ids.add(origin.field_id)
    return sorted(ids)


def parse_log(path: str) -> List[ParsedInstruction]:
    items: List[ParsedInstruction] = []
    taint_seen = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            _, payload = split_thread_prefix(raw)
            if not taint_seen:
                if payload.startswith(TAINT_PREFIX):
                    taint_seen = True
                continue
            parsed = parse_instruction(payload, line_no)
            if parsed:
                items.append(parsed)
    return items


class BitfieldAnalyzer:
    def __init__(self, fields: Sequence[FieldRef]) -> None:
        self.fields = list(fields)
        self.fields_by_id = {f.field_id: f for f in self.fields}
        self.reg_state: Dict[str, BitOriginMap] = {}
        self.mem_state: Dict[str, BitOriginMap] = {}
        self.events_by_field: Dict[str, List[UseEvent]] = {f.field_id: [] for f in self.fields}
        self.pending_test_event: Optional[Tuple[str, UseEvent]] = None

    def _field_for_idx_raw(self, idx_raw: str) -> Optional[FieldRef]:
        tokens = parse_field_tokens(idx_raw)
        matched = sorted({token for token in tokens if token in self.fields_by_id})
        if len(matched) == 1:
            return self.fields_by_id[matched[0]]
        if not matched and tokens:
            span_matched: List[FieldRef] = []
            for token in tokens:
                span = token_byte_span(token)
                if span is None:
                    continue
                for field in self.fields:
                    if field.byte_offset_start == span[0] and field.byte_offset_end == span[1]:
                        span_matched.append(field)
            uniq = {(f.field_id, f.byte_offset_start, f.byte_offset_end): f for f in span_matched}
            if len(uniq) == 1:
                return next(iter(uniq.values()))
        if len(matched) > 1:
            return None
        return None

    def _get_reg_alias(self, op: str) -> Optional[Tuple[str, int, int]]:
        return REG_ALIASES.get(op.strip().lower())

    def _read_reg(self, op: str) -> BitOriginMap:
        alias = self._get_reg_alias(op)
        if alias is None:
            return BitOriginMap.unknown(max(operand_width_bits(op), 8))
        base, width, offset = alias
        full = self.reg_state.get(base)
        if full is None:
            full = BitOriginMap.unknown(64)
            self.reg_state[base] = full
        return full.slice(offset, width)

    def _write_reg(self, op: str, value: BitOriginMap) -> None:
        alias = self._get_reg_alias(op)
        if alias is None:
            return
        base, width, offset = alias
        full = self.reg_state.get(base)
        if full is None:
            full = BitOriginMap.unknown(64)
            self.reg_state[base] = full
        normalized = low_extend(value, width)
        full.assign_slice(offset, normalized)
        upper_limit = offset + width
        for i in range(upper_limit, 64):
            if width == 32 and offset == 0:
                full.origins[i] = CONST0

    def _read_memory(self, op: str) -> BitOriginMap:
        key = normalize_memory_key(op)
        width = operand_width_bits(op) or 8
        return self.mem_state.get(key, BitOriginMap.unknown(width))

    def _write_memory(self, op: str, value: BitOriginMap) -> None:
        key = normalize_memory_key(op)
        width = operand_width_bits(op) or value.width_bits
        self.mem_state[key] = low_extend(value, width)

    def _operand_source_map(self, op: str, instr: ParsedInstruction, field_ref: Optional[FieldRef]) -> BitOriginMap:
        op = op.strip()
        if operand_is_register(op):
            return self._read_reg(op)
        if operand_is_memory(op):
            key = normalize_memory_key(op)
            width = operand_width_bits(op) or 8
            # 字段标签直接信任日志；若该内存槽已经在分析过程中获得了
            # 更精确的 bit 映射（例如 mov byte 落地后再 movzx 读回），
            # 应优先复用这份映射，而不是再次退化成“整字段原始映射”。
            existing = self.mem_state.get(key)
            if existing is not None and (
                field_ref is None or existing.used_source_bits(field_ref.field_id)
            ):
                return low_extend(existing, width)
            idx_set = parse_indices(instr.idx_raw)
            if field_ref is not None and idx_set:
                overlap = any(i in idx_set for i in range(field_ref.byte_offset_start, field_ref.byte_offset_end + 1))
                if overlap:
                    src = build_field_source_map(field_ref)
                    return low_extend(src, width or src.width_bits)
            if existing is not None:
                return low_extend(existing, width)
            return self._read_memory(op)
        if operand_is_immediate(op):
            width = max(operand_width_bits(op), 8)
            value = parse_immediate(op) or 0
            origins = [CONST1 if ((value >> i) & 1) else CONST0 for i in range(width)]
            return BitOriginMap(width, origins)
        return BitOriginMap.unknown(max(operand_width_bits(op), 8))

    def _append_event(self, event: UseEvent) -> None:
        self.events_by_field.setdefault(event.field_id, []).append(event)

    def _maybe_append_derived_mov_event(self, instr: ParsedInstruction, bit_map: BitOriginMap) -> None:
        for field_id in source_field_ids(bit_map):
            field_ref = self.fields_by_id.get(field_id)
            if field_ref is None:
                continue
            used_bits = bit_map.used_source_bits(field_id)
            if not used_bits:
                continue
            if used_bits == list(range(field_ref.bit_width)):
                continue

            source_positions = [
                i for i, origin in enumerate(bit_map.origins)
                if origin.kind == "SOURCE" and origin.field_id == field_id
            ]
            if not source_positions:
                continue
            source_span = contiguous_range(source_positions)
            if source_span is None:
                continue

            left_fill = all(origin.kind in {"CONST0", "CONST1"} for origin in bit_map.origins[:source_span[0]])
            right_fill = all(origin.kind in {"CONST0", "CONST1"} for origin in bit_map.origins[source_span[1] + 1:])
            if not (left_fill or right_fill):
                continue

            self._append_event(UseEvent(
                field_id=field_id,
                trace_line=instr.trace_line,
                address=instr.address,
                instruction=instr.disasm,
                event_kind="mov_derived",
                used_source_bits=used_bits,
                raw_operands=instr.operands_text,
                explanation="mov 保存了位移/掩码后的局部结果",
            ))

    def _maybe_close_pending_test(self, instr: ParsedInstruction) -> None:
        if self.pending_test_event is None:
            return
        field_id, event = self.pending_test_event
        if JCC_RE.match(instr.disasm):
            m = JCC_RE.match(instr.disasm)
            cc = m.group(1).lower() if m else ""
            taken = "TAKEN" in instr.payload
            event.branch_taken = taken
            if cc in {"z", "e"}:
                event.normalized_constraint = normalize_constraint(event.used_source_bits, None, "zero" if taken else "nonzero")
                event.structured_constraint = build_structured_constraint(
                    event.used_source_bits,
                    "zero" if taken else "nonzero",
                    explicit_mask=event.mask,
                )
            elif cc in {"nz", "ne"}:
                event.normalized_constraint = normalize_constraint(event.used_source_bits, None, "nonzero" if taken else "zero")
                event.structured_constraint = build_structured_constraint(
                    event.used_source_bits,
                    "nonzero" if taken else "zero",
                    explicit_mask=event.mask,
                )
            elif cc in {"s", "ns"} and event.used_source_bits:
                bit = max(event.used_source_bits)
                event.normalized_constraint = f"bit{bit} {'== 1' if (cc == 's' and taken) or (cc == 'ns' and not taken) else '== 0'}"
                event.structured_constraint = build_structured_constraint(
                    [bit],
                    "nonzero" if ((cc == 's' and taken) or (cc == 'ns' and not taken)) else "zero",
                    explicit_mask=(1 << bit),
                )
            self._append_event(event)
        else:
            if event.event_kind == "test_mask":
                event.structured_constraint = build_structured_constraint(
                    event.used_source_bits,
                    "nonzero",
                    explicit_mask=event.mask,
                )
            elif event.event_kind == "test_reg":
                event.structured_constraint = build_structured_constraint(
                    event.used_source_bits,
                    "nonzero",
                )
            self._append_event(event)
        self.pending_test_event = None

    def process_instruction(self, instr: ParsedInstruction) -> None:
        self._maybe_close_pending_test(instr)
        mnemonic = instr.mnemonic
        if mnemonic not in USEFUL_OPS and not JCC_RE.match(instr.disasm):
            return
        operands = split_operands(instr.operands_text)
        idx_set = parse_indices(instr.idx_raw)
        field_ref = self._field_for_idx_raw(instr.idx_raw)

        if mnemonic == "mov" and len(operands) == 2:
            src_map = self._operand_source_map(operands[1], instr, field_ref)
            if operand_is_register(operands[1]) or operand_is_memory(operands[1]):
                self._maybe_append_derived_mov_event(instr, src_map)
            if operand_is_register(operands[0]):
                self._write_reg(operands[0], src_map)
            elif operand_is_memory(operands[0]):
                self._write_memory(operands[0], src_map)
            return

        if mnemonic in {"movzx", "movsx", "movsxd"} and len(operands) == 2:
            src_map = self._operand_source_map(operands[1], instr, field_ref)
            dst_width = operand_width_bits(operands[0]) or src_map.width_bits
            if mnemonic == "movzx":
                out_map = low_extend(src_map, dst_width)
            else:
                out_map = sign_extend(src_map, dst_width)
            if operand_is_register(operands[0]):
                self._write_reg(operands[0], out_map)
            return

        if mnemonic == "lea" and len(operands) == 2:
            idx_set = parse_indices(instr.idx_raw)
            if field_ref is not None and idx_set:
                out_map = low_extend(build_field_source_map(field_ref), operand_width_bits(operands[0]) or field_ref.bit_width)
            else:
                out_map = BitOriginMap.unknown(operand_width_bits(operands[0]) or 64)
            self._write_reg(operands[0], out_map)
            return

        if mnemonic == "xor" and len(operands) == 2 and operands[0].strip().lower() == operands[1].strip().lower():
            dst_width = operand_width_bits(operands[0]) or 8
            out_map = BitOriginMap.const_zero(dst_width)
            if operand_is_register(operands[0]):
                self._write_reg(operands[0], out_map)
            elif operand_is_memory(operands[0]):
                self._write_memory(operands[0], out_map)
            return

        runtime_mask = None
        if mnemonic in {"and", "or", "xor"} and len(operands) == 2:
            if operand_is_immediate(operands[1]):
                runtime_mask = parse_immediate(operands[1]) or 0
            else:
                runtime_mask = parse_non_tainted_runtime_value(instr.info_raw)

        if mnemonic in {"and", "or", "xor"} and len(operands) == 2 and runtime_mask is not None:
            dst_map = self._operand_source_map(operands[0], instr, field_ref)
            imm = runtime_mask
            if mnemonic == "and":
                out_map = apply_and_imm(dst_map, imm)
            elif mnemonic == "or":
                out_map = apply_or_imm(dst_map, imm)
            else:
                out_map = apply_xor_imm(dst_map, imm)
            if operand_is_register(operands[0]):
                self._write_reg(operands[0], out_map)
            elif operand_is_memory(operands[0]):
                self._write_memory(operands[0], out_map)

            if field_ref is not None:
                used_bits = dst_map.used_source_bits(field_ref.field_id, imm)
                if used_bits:
                    explanation = "按位掩码处理字段位" if mnemonic == "and" else "按位逻辑处理字段位"
                    self._append_event(UseEvent(
                        field_id=field_ref.field_id,
                        trace_line=instr.trace_line,
                        address=instr.address,
                        instruction=instr.disasm,
                        event_kind=mnemonic,
                        used_source_bits=used_bits,
                        mask=imm,
                        raw_operands=instr.operands_text,
                        explanation=explanation,
                    ))
            return

        if mnemonic in {"shr", "sar", "shl", "sal"} and len(operands) == 2 and operand_is_immediate(operands[1]):
            dst_map = self._operand_source_map(operands[0], instr, field_ref)
            shift = parse_immediate(operands[1]) or 0
            if mnemonic in {"shr", "sar"}:
                out_map = shift_right(dst_map, shift, arithmetic=(mnemonic == "sar"))
            else:
                out_map = shift_left(dst_map, shift)
            if operand_is_register(operands[0]):
                self._write_reg(operands[0], out_map)
            elif operand_is_memory(operands[0]):
                self._write_memory(operands[0], out_map)
            if field_ref is not None:
                used_bits = dst_map.used_source_bits(field_ref.field_id)
                if used_bits:
                    self._append_event(UseEvent(
                        field_id=field_ref.field_id,
                        trace_line=instr.trace_line,
                        address=instr.address,
                        instruction=instr.disasm,
                        event_kind=mnemonic,
                        used_source_bits=used_bits,
                        shift=shift,
                        raw_operands=instr.operands_text,
                        explanation="位移后继续使用该字段的部分 bit",
                    ))
            return

        if mnemonic in {"shr", "sar", "shl", "sal"} and len(operands) == 2 and operand_is_register(operands[1]):
            dst_map = self._operand_source_map(operands[0], instr, field_ref)
            if operand_is_register(operands[0]):
                self._write_reg(operands[0], BitOriginMap.unknown(dst_map.width_bits))
            elif operand_is_memory(operands[0]):
                self._write_memory(operands[0], BitOriginMap.unknown(dst_map.width_bits))
            if field_ref is not None:
                used_bits = dst_map.used_source_bits(field_ref.field_id)
                if used_bits:
                    self._append_event(UseEvent(
                        field_id=field_ref.field_id,
                        trace_line=instr.trace_line,
                        address=instr.address,
                        instruction=instr.disasm,
                        event_kind=mnemonic,
                        used_source_bits=used_bits,
                        raw_operands=instr.operands_text,
                        explanation="变位数位移后继续逐位使用该字段",
                        variable_shift=True,
                    ))
            return

        if mnemonic == "test" and len(operands) == 2:
            lhs_map = self._operand_source_map(operands[0], instr, field_ref)
            if operand_is_immediate(operands[1]):
                imm = parse_immediate(operands[1]) or 0
                if field_ref is not None:
                    used_bits = lhs_map.used_source_bits(field_ref.field_id, imm)
                    if used_bits:
                        event = UseEvent(
                            field_id=field_ref.field_id,
                            trace_line=instr.trace_line,
                            address=instr.address,
                            instruction=instr.disasm,
                            event_kind="test_mask",
                            used_source_bits=used_bits,
                            mask=imm,
                            raw_operands=instr.operands_text,
                            explanation="使用掩码测试该字段的部分 bit",
                        )
                        self.pending_test_event = (field_ref.field_id, event)
                return
            rhs_map = self._operand_source_map(operands[1], instr, field_ref)
            if field_ref is not None:
                used_bits = sorted(set(lhs_map.used_source_bits(field_ref.field_id) + rhs_map.used_source_bits(field_ref.field_id)))
                if used_bits:
                    event = UseEvent(
                        field_id=field_ref.field_id,
                        trace_line=instr.trace_line,
                        address=instr.address,
                        instruction=instr.disasm,
                        event_kind="test_reg",
                        used_source_bits=used_bits,
                        structured_constraint=None,
                        raw_operands=instr.operands_text,
                        explanation="整体测试该字段相关值是否为 0 / 非 0",
                    )
                    self.pending_test_event = (field_ref.field_id, event)
            return

        if mnemonic == "cmp" and len(operands) == 2:
            lhs_map = self._operand_source_map(operands[0], instr, field_ref)
            rhs_map = self._operand_source_map(operands[1], instr, field_ref)
            compare_value = parse_immediate(operands[1])
            if compare_value is None:
                info_values = parse_info_values(instr.info_raw)
                non_tainted = [(k, v) for k, v in info_values.items() if "*" not in k]
                if non_tainted:
                    compare_value = non_tainted[0][1]

            if field_ref is not None:
                used_bits = sorted(set(lhs_map.used_source_bits(field_ref.field_id) + rhs_map.used_source_bits(field_ref.field_id)))
                if used_bits:
                    normalized = normalize_constraint(used_bits, compare_value, "==") if compare_value is not None else None
                    structured = build_structured_constraint(used_bits, "eq", compare_value) if compare_value is not None else None
                    self._append_event(UseEvent(
                        field_id=field_ref.field_id,
                        trace_line=instr.trace_line,
                        address=instr.address,
                        instruction=instr.disasm,
                        event_kind="cmp",
                        used_source_bits=used_bits,
                        compare_value=compare_value,
                        normalized_constraint=normalized,
                        structured_constraint=structured,
                        raw_operands=instr.operands_text,
                        explanation="比较该字段相关位与常量或另一个运行时值",
                    ))
            return

        if SETCC_RE.match(instr.disasm) and len(operands) == 1 and operand_is_register(operands[0]):
            dst_width = operand_width_bits(operands[0]) or 8
            self._write_reg(operands[0], BitOriginMap.unknown(dst_width))
            return

    def finalize(self) -> None:
        if self.pending_test_event is not None:
            _, event = self.pending_test_event
            self._append_event(event)
            self.pending_test_event = None


def events_for_recovery_mode(events: Sequence[UseEvent], config: BitfieldRecoveryConfig) -> List[UseEvent]:
    if config.use_consumption_guidance:
        return list(events)
    operation_event_kinds = {"and", "or", "xor", "shr", "sar", "shl", "sal", "test_mask"}
    return [event for event in events if event.event_kind in operation_event_kinds]


def infer_subfields(
    field_ref: FieldRef,
    events: Sequence[UseEvent],
    config: Optional[BitfieldRecoveryConfig] = None,
) -> Tuple[bool, List[dict], List[CandidateSubfield]]:
    config = config or recovery_config("full")
    events = events_for_recovery_mode(events, config)
    bitfield_evidence: List[dict] = []
    bit_op_count = 0
    distinct_masks: Set[int] = set()
    grouped: Dict[Tuple[int, ...], List[UseEvent]] = {}
    full_width_bits = list(range(field_ref.bit_width))
    has_partial_bit_use = False
    has_strong_bit_evidence = False
    strong_event_kinds = {"and", "or", "xor", "shr", "sar", "shl", "sal", "test_mask"}
    full_width_compare_like = False
    partial_compare_like = False
    weak_partial_only = True
    first_full_width_compare_line: Optional[int] = None
    has_partial_before_full_width_compare = False
    partial_compare_groups: Set[Tuple[int, ...]] = set()
    full_width_compare_count = 0
    has_variable_shift_scan = False
    only_single_bit_scan = True

    for event in events:
        if event.event_kind in {"and", "or", "xor", "shr", "sar", "shl", "sal", "test_mask", "test_reg", "mov_derived"}:
            bit_op_count += 1
            if event.mask is not None:
                distinct_masks.add(int(event.mask))
            bitfield_evidence.append({
                "trace_line": event.trace_line,
                "address": event.address,
                "instruction": event.instruction,
                "used_source_bits": event.used_source_bits,
                "explanation": event.explanation,
            })
        key = tuple(sorted(set(event.used_source_bits)))
        if key:
            grouped.setdefault(key, []).append(event)
            if list(key) != full_width_bits:
                has_partial_bit_use = True
                compare_line = first_full_width_compare_line
                if compare_line is None or event.trace_line < compare_line:
                    has_partial_before_full_width_compare = True
                if event.event_kind in {"cmp", "test_mask", "test_reg"}:
                    partial_compare_like = True
                    partial_compare_groups.add(tuple(key))
                if event.event_kind not in {"and", "or", "xor", "shr", "sar", "shl", "sal", "mov_derived"}:
                    weak_partial_only = False
            elif event.event_kind in {"cmp", "test_reg"}:
                full_width_compare_like = True
                full_width_compare_count += 1
                if first_full_width_compare_line is None or event.trace_line < first_full_width_compare_line:
                    first_full_width_compare_line = event.trace_line
        if (
            event.event_kind in strong_event_kinds
            and event.used_source_bits
            and list(sorted(set(event.used_source_bits))) != full_width_bits
        ):
            has_strong_bit_evidence = True
        if event.variable_shift:
            has_variable_shift_scan = True
        if event.event_kind == "and" and event.mask not in {None, 1}:
            only_single_bit_scan = False
        if event.event_kind == "cmp":
            only_single_bit_scan = False
        if len(event.used_source_bits) != 1 and event.event_kind in {"and", "cmp", "test_mask", "test_reg"}:
            only_single_bit_scan = False

    # “是否是位字段”优先看是否存在明确的“部分 bit 消费”语义，而不是简单计数。
    # 这样可以保留类似 field 6 这种只出现一次掩码提取高 4 bit 的情况，
    # 同时避免把整字节 cmp/test_reg 的枚举字段误判成 bitfield。
    is_bitfield = has_strong_bit_evidence or has_partial_bit_use

    # 收紧规则：
    # 如果一个字段已经被整字段 cmp/test_reg 明确当作整体值消费，
    # 而后续又没有出现局部 cmp/test 这类更强的局部消费，
    # 仅凭 shift/and/mov_derived 这类弱局部痕迹，不再把它判成位字段。
    #
    # 但若已经存在明确的强位证据（例如 shr 提取高位段、and mask 提取低位段），
    # 则不能再因为后面出现整字段 cmp 就把它压回普通整字段。否则像
    # BACnet Object Identifier 这类“先拆位段、再分别作为 object_type /
    # object_instance 继续使用”的经典模式会被误杀。
    if (
        config.resolve_hierarchical_evidence
        and full_width_compare_like
        and not partial_compare_like
        and weak_partial_only
        and not has_strong_bit_evidence
    ):
        is_bitfield = False

    # 若字段本身已经稳定地作为整字段枚举/状态值参与大量比较，而所谓“局部位证据”
    # 仅表现为一次接近整宽的 mask 后比较（例如仅屏蔽 1~2 个 bit 再按整体值比较），
    # 则更像“整体字段归一化后比较”，不应恢复成位字段。
    #
    # 典型例子：8-bit 枚举值比较前先 and 掉一个 don't-care bit；此时虽然日志层面
    # 看起来像部分 bit 被消费，但程序语义上仍是整个字节作为状态码/枚举值在使用。
    if (
        config.resolve_hierarchical_evidence
        and is_bitfield
        and field_ref.bit_width <= 8
        and full_width_compare_count >= 2
        and partial_compare_groups
    ):
        all_near_full = all(len(bits) >= field_ref.bit_width - 2 for bits in partial_compare_groups)
        has_small_partial_group = any(len(bits) <= max(1, field_ref.bit_width // 2) for bits in partial_compare_groups)
        if all_near_full and not has_small_partial_group:
            is_bitfield = False

    # 若字段只是通过变位数移位（如 sar/shr reg, cl）配合 and 0x1
    # 在循环中做逐位扫描，而没有恢复出固定位置的局部比较边界，
    # 则更接近“普通字段上的位遍历”，不稳定输出为位字段。
    if config.resolve_hierarchical_evidence and has_variable_shift_scan and only_single_bit_scan and not partial_compare_like:
        is_bitfield = False

    partial_events = [
        ev for ev in events
        if ev.used_source_bits and sorted(set(ev.used_source_bits)) != full_width_bits
    ]
    bit_occurrence: Dict[int, int] = {}
    for ev in partial_events:
        for bit in sorted(set(ev.used_source_bits)):
            bit_occurrence[bit] = bit_occurrence.get(bit, 0) + 1

    subfields: List[CandidateSubfield] = []
    for bits_tuple, group_events in sorted(grouped.items(), key=lambda item: (len(item[0]), item[0])):
        bits = list(bits_tuple)
        if bits == full_width_bits:
            continue

        # 若字段在第一次 full-width cmp/test 之前已经出现过局部 bit 证据，
        # 则第一次 full-width 比较之后、且再未进入 cmp/test 的局部位操作，
        # 更可能是后续编码/搬运阶段的中间结果，不再用于主导最终子字段边界恢复。
        #
        # 典型例子：BACnet Object Identifier 先被拆成高 10 位 / 低 22 位参与比较，
        # 后面又在 encode 阶段重新拼回 32 位并按字节输出。后半段会产生很多
        # [0], [0:7], [1:9] 这类局部痕迹，但它们不应反向切碎前面已经比较过的
        # 真正语义子字段。
        if config.resolve_hierarchical_evidence and first_full_width_compare_line is not None and has_partial_before_full_width_compare:
            has_group_compare_like = any(
                ev.event_kind in {"cmp", "test_mask", "test_reg"} or ev.compare_value is not None
                for ev in group_events
            )
            has_pre_compare_support = any(ev.trace_line < first_full_width_compare_line for ev in group_events)
            has_post_compare_support = any(ev.trace_line > first_full_width_compare_line for ev in group_events)
            weak_only_group = all(ev.event_kind in {"and", "or", "xor", "shr", "sar", "shl", "sal", "mov_derived"} for ev in group_events)
            if has_post_compare_support and not has_pre_compare_support and not has_group_compare_like and weak_only_group:
                continue

        support = len(group_events)
        max_bit_support = max(bit_occurrence.get(bit, support) for bit in bits)
        together_ratio = float(support) / float(max(max_bit_support, 1))
        has_compare_like = any(ev.event_kind in {"cmp", "test_mask", "test_reg"} or ev.compare_value is not None for ev in group_events)
        has_arith_like = any(ev.event_kind in {"and", "or", "xor", "shr", "sar", "shl", "sal"} for ev in group_events)
        has_derived_mov = any(ev.event_kind == "mov_derived" for ev in group_events)

        if len(bits) == 1:
            kind = "flag"
        else:
            if contiguous_range(bits) is not None:
                kind = "enum_or_small_int" if has_compare_like else "bit_range"
            else:
                kind = "bit_group"

        # 子字段恢复遵循“共同使用模式”：
        # 1. 单 bit 始终保留为 flag 候选
        # 2. 多 bit 组若经常整体出现，或属于显式位操作/比较后的整体消费，则保留
        # 3. 若某些 bit 也经常单独出现，不压制单 bit 候选，允许大小位组并存
        if len(bits) > 1:
            keep_group = (
                support >= 2
                or together_ratio >= 0.75
                or has_compare_like
                or has_arith_like
                or has_derived_mov
            )
            if not keep_group:
                continue

        constraint_map: Dict[str, List[int]] = {}
        structured_constraints: Dict[str, dict] = {}
        for ev in group_events:
            if ev.structured_constraint:
                key = json.dumps(ev.structured_constraint, sort_keys=True, ensure_ascii=False)
                entry = structured_constraints.setdefault(key, dict(ev.structured_constraint))
                entry.setdefault("trace_lines", [])
                entry["trace_lines"] = sorted(set(entry["trace_lines"] + [ev.trace_line]))
            elif ev.normalized_constraint:
                constraint_map.setdefault(ev.normalized_constraint, []).append(ev.trace_line)
        constraints = list(sorted(structured_constraints.values(), key=lambda item: (item.get("subfield_constraint") or "", item.get("field_constraint") or "")))
        constraints.extend(
            {"expression": text, "trace_lines": lines}
            for text, lines in sorted(constraint_map.items())
        )
        confidence = 0.35
        confidence += 0.08 * support
        confidence += 0.10 * len(constraints)
        confidence += 0.15 * together_ratio
        if has_compare_like:
            confidence += 0.10
        if len(bits) == 1:
            confidence += 0.05
        confidence = min(0.99, confidence)
        subfields.append(CandidateSubfield(
            field_id=field_ref.field_id,
            bits=bits,
            kind=kind,
            evidence_events=[ev.trace_line for ev in group_events],
            constraints=constraints,
            confidence=confidence,
            source_event_kinds=sorted(set(ev.event_kind for ev in group_events)),
        ))

    # 非连续位集合先拆成连续段，避免类似 [0,1,2,3,4,5,7] 直接作为一个最终子字段输出。
    expanded: List[CandidateSubfield] = []
    for sf in subfields:
        runs = contiguous_runs(sf.bits)
        if len(runs) <= 1:
            expanded.append(sf)
            continue
        for run in runs:
            expanded.append(CandidateSubfield(
                field_id=sf.field_id,
                bits=run,
                kind="flag" if len(run) == 1 else ("enum_or_small_int" if contiguous_range(run) is not None else "bit_group"),
                evidence_events=sf.evidence_events,
                constraints=sf.constraints,
                confidence=max(0.55, sf.confidence - 0.05),
                source_event_kinds=sf.source_event_kinds + ["split_noncontiguous"],
            ))
    subfields = expanded

    if not config.resolve_hierarchical_evidence:
        flat_candidates: Dict[Tuple[int, ...], CandidateSubfield] = {}
        for sf in subfields:
            key = tuple(sf.bits)
            if key not in flat_candidates:
                flat_candidates[key] = sf
                continue
            existing = flat_candidates[key]
            existing.evidence_events = sorted(set(existing.evidence_events + sf.evidence_events))
            existing.constraints = sorted(
                {json.dumps(item, sort_keys=True): item for item in existing.constraints + sf.constraints}.values(),
                key=lambda item: (item.get("subfield_constraint", ""), item.get("field_constraint", "")),
            )
            existing.confidence = min(0.99, max(existing.confidence, sf.confidence))
            existing.source_event_kinds = sorted(set(existing.source_event_kinds + sf.source_event_kinds))
        return is_bitfield, bitfield_evidence, sorted(flat_candidates.values(), key=lambda item: (len(item.bits), item.bits))

    # 若 shift+mov 导出的宽区间内部，已经有多个更直接的单 bit / 小位段证据，
    # 则将该宽区间收缩成有效侧边界位。这里仅对“位移链派生值”收缩；
    # 若宽区间来自 mask 操作本身（如 and 0x3f），则保留 mask 对应位段。
    direct_subfields = [
        sf for sf in subfields
        if "mov_derived" not in sf.source_event_kinds
    ]
    direct_bits_union: Set[int] = set()
    direct_single_bits: Set[int] = set()
    for sf in direct_subfields:
        direct_bits_union.update(sf.bits)
        if len(sf.bits) == 1:
            direct_single_bits.add(sf.bits[0])

    rewritten: List[CandidateSubfield] = []
    seen_keys: Set[Tuple[Tuple[int, ...], Tuple[str, ...]]] = set()
    for sf in subfields:
        bits = list(sf.bits)
        is_shift_derived = (
            "mov_derived" in sf.source_event_kinds
            and not any(kind in {"and", "or", "xor"} for kind in sf.source_event_kinds)
        )
        if len(bits) > 1 and is_shift_derived:
            overlap_bits = [bit for bit in bits if bit in direct_bits_union]
            overlap_single_bits = [bit for bit in bits if bit in direct_single_bits]
            if len(overlap_bits) >= 2 or len(overlap_single_bits) >= 2:
                edge_bit = min(bits)
                sf = CandidateSubfield(
                    field_id=sf.field_id,
                    bits=[edge_bit],
                    kind="flag",
                    evidence_events=sf.evidence_events,
                    constraints=sf.constraints,
                    confidence=min(0.99, max(sf.confidence, 0.72)),
                    source_event_kinds=sf.source_event_kinds + ["boundary_shrink"],
                )

        key = (tuple(sf.bits), tuple(sorted(sf.source_event_kinds)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rewritten.append(sf)

    # 若收缩后的单 bit 与已有直接单 bit 重复，则合并证据而不是重复输出
    merged: Dict[Tuple[int, ...], CandidateSubfield] = {}
    for sf in rewritten:
        key = tuple(sf.bits)
        if key not in merged:
            merged[key] = sf
            continue
        existing = merged[key]
        existing.evidence_events = sorted(set(existing.evidence_events + sf.evidence_events))
        existing.constraints = sorted(
            {json.dumps(item, sort_keys=True): item for item in existing.constraints + sf.constraints}.values(),
            key=lambda item: (item.get("normalized", ""), item.get("sources", [])),
        )
        existing.confidence = min(0.99, max(existing.confidence, sf.confidence))
        existing.source_event_kinds = sorted(set(existing.source_event_kinds + sf.source_event_kinds))

    resolved: List[CandidateSubfield] = list(merged.values())
    dropped_small_keys: Set[Tuple[int, ...]] = set()
    split_large_keys: Set[Tuple[int, ...]] = set()
    split_residuals: List[CandidateSubfield] = []

    for sf in sorted(resolved, key=lambda item: (-len(item.bits), item.bits)):
        if len(sf.bits) <= 1:
            continue
        if contiguous_range(sf.bits) is None:
            continue

        larger_strength = subfield_evidence_strength(sf)
        contained_smalls = [
            other for other in resolved
            if other is not sf
            and other.bits
            and len(other.bits) < len(sf.bits)
            and set(other.bits).issubset(sf.bits)
        ]
        if not contained_smalls:
            continue

        max_small_strength = max(subfield_evidence_strength(other) for other in contained_smalls)
        if larger_strength > max_small_strength:
            for other in contained_smalls:
                dropped_small_keys.add(tuple(other.bits))
            continue

        covered_bits: Set[int] = set()
        for other in contained_smalls:
            covered_bits.update(other.bits)
        residual_bits = [bit for bit in sf.bits if bit not in covered_bits]
        split_large_keys.add(tuple(sf.bits))
        for run in contiguous_runs(residual_bits):
            if not run:
                continue
            split_residuals.append(CandidateSubfield(
                field_id=sf.field_id,
                bits=run,
                kind="flag" if len(run) == 1 else "bit_range",
                evidence_events=sf.evidence_events,
                constraints=sf.constraints,
                confidence=max(0.50, sf.confidence - 0.08),
                source_event_kinds=sf.source_event_kinds + ["coverage_split"],
            ))

    coverage_resolved: Dict[Tuple[int, ...], CandidateSubfield] = {}
    for sf in resolved + split_residuals:
        key = tuple(sf.bits)
        if key in dropped_small_keys:
            continue
        if key in split_large_keys:
            continue
        if key not in coverage_resolved:
            coverage_resolved[key] = sf
            continue
        existing = coverage_resolved[key]
        existing.evidence_events = sorted(set(existing.evidence_events + sf.evidence_events))
        existing.constraints = sorted(
            {json.dumps(item, sort_keys=True): item for item in existing.constraints + sf.constraints}.values(),
            key=lambda item: (item.get("subfield_constraint", ""), item.get("field_constraint", "")),
        )
        existing.confidence = min(0.99, max(existing.confidence, sf.confidence))
        existing.source_event_kinds = sorted(set(existing.source_event_kinds + sf.source_event_kinds))

    subfields = sorted(coverage_resolved.values(), key=lambda item: (len(item.bits), item.bits))

    # 若一个多字节字段仅仅表现为“按字节拆开再写出”的序列化路径，
    # 例如 32-bit 整数字段经过 shr 8/16/24 后逐字节 mov/store，
    # 则虽然日志层面会出现 [7:0] / [15:8] / ... 这样的局部范围，
    # 但它们并不对应稳定的语义子字段，只是编码输出时的字节拆包。
    #
    # 这里做一个非常收敛的压制：
    # 1. 字段宽度至少 16 bit 且按整字节对齐；
    # 2. 当前恢复出的所有子字段恰好按字节完整分割整个字段；
    # 3. 这些子字段的证据仅来自 mov_derived / boundary_shrink / coverage_split
    #    等弱传播痕迹，没有 cmp/test/mask 之类固定局部消费。
    # 满足时将其回退为普通字段，避免把“整数字段序列化”误判成位字段。
    if is_bitfield and field_ref.bit_width >= 16 and field_ref.bit_width % 8 == 0 and subfields:
        covered_bits = sorted({bit for sf in subfields for bit in sf.bits})
        expected_bits = list(range(field_ref.bit_width))
        exact_byte_partition = (
            covered_bits == expected_bits
            and all(len(sf.bits) == 8 and contiguous_range(sf.bits) is not None for sf in subfields)
            and all(sf.bits[0] % 8 == 0 for sf in subfields)
        )
        byte_serialization_only = all(
            set(sf.source_event_kinds).issubset({"mov_derived", "boundary_shrink", "coverage_split"})
            and not sf.constraints
            for sf in subfields
        )
        if exact_byte_partition and byte_serialization_only:
            is_bitfield = False
            subfields = []

    if has_variable_shift_scan and only_single_bit_scan and not partial_compare_like:
        subfields = []

    return is_bitfield, bitfield_evidence, subfields


def analyze_logs(fields: Sequence[FieldRef], log_paths: Sequence[str], mode: str = "full") -> dict:
    config = recovery_config(mode)
    analyzer = BitfieldAnalyzer(fields)
    for path in log_paths:
        instructions = parse_log(path)
        for instr in instructions:
            analyzer.process_instruction(instr)
    analyzer.finalize()

    output_fields = []
    for field_ref in fields:
        events = analyzer.events_by_field.get(field_ref.field_id, [])
        is_bitfield, evidence, subfields = infer_subfields(field_ref, events, config=config)
        if not is_bitfield:
            continue
        recovered_layout, recovered_layout_text = format_recovered_layout(field_ref.bit_width, subfields)
        output_fields.append({
            "field_id": field_ref.field_id,
            "subfields": [item.to_json() for item in subfields],
        })
    return {"fields": output_fields}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="位字段识别、划分和约束提取实验脚本（Plan A）")
    parser.add_argument("--log", required=True, nargs="+", help="baseline / mutation 日志路径，可传多个")
    parser.add_argument("--fields", required=True, help="fields.json 路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--mode",
        choices=RECOVERY_MODES,
        default="full",
        help="位字段恢复模式：full、operation_driven 或 flat_evidence",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fields = load_fields(args.fields)
    result = analyze_logs(fields, args.log, mode=args.mode)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
