#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified byte-field and bit-field helpers for difftrace."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def make_byte_unit(a: int, b: int, repr_text: Optional[str] = None, count: Optional[int] = None) -> dict:
    unit = {
        "kind": "byte",
        "a": int(a),
        "b": int(b),
        "repr": repr_text or (f"{a}" if a == b else f"{a},{b}"),
        "width_bits": (int(b) - int(a) + 1) * 8,
    }
    if count is not None:
        unit["count"] = int(count)
    return unit


def make_bit_unit(parent: dict, bits: Sequence[int], constraints: Optional[List[dict]] = None) -> dict:
    clean_bits = sorted({int(bit) for bit in bits})
    if not clean_bits:
        raise ValueError("bit field requires at least one bit")
    parent_repr = str(parent.get("repr") or parent.get("field_id") or parent.get("a"))
    bit_repr = bits_repr(clean_bits)
    return {
        "kind": "bit",
        "a": int(parent["a"]),
        "b": int(parent["b"]),
        "parent_repr": parent_repr,
        "bits": clean_bits,
        "repr": f"{parent_repr}{bit_repr}",
        "width_bits": len(clean_bits),
        "constraints": constraints or [],
    }


def normalize_unit(unit: dict) -> dict:
    kind = str(unit.get("kind") or "byte")
    if kind == "bit":
        bits = sorted({int(bit) for bit in unit.get("bits", [])})
        parent_repr = str(unit.get("parent_repr") or unit.get("repr", "").split("[", 1)[0] or unit.get("a"))
        return {
            **unit,
            "kind": "bit",
            "a": int(unit["a"]),
            "b": int(unit["b"]),
            "parent_repr": parent_repr,
            "bits": bits,
            "repr": str(unit.get("repr") or f"{parent_repr}{bits_repr(bits)}"),
            "width_bits": int(unit.get("width_bits") or len(bits)),
        }
    a = int(unit["a"])
    b = int(unit["b"])
    return {
        **unit,
        "kind": "byte",
        "a": a,
        "b": b,
        "repr": str(unit.get("repr") or (f"{a}" if a == b else f"{a},{b}")),
        "width_bits": int(unit.get("width_bits") or ((b - a + 1) * 8)),
    }


def unit_from_tuple(a: int, b: int) -> dict:
    return make_byte_unit(a, b)


def units_from_fields_json(fields_json: dict) -> List[dict]:
    result = []
    for item in fields_json.get("fields", []) if isinstance(fields_json, dict) else []:
        if isinstance(item, dict) and "a" in item and "b" in item:
            result.append(normalize_unit(item))
    return result


def byte_ranges_from_units(units: Sequence[dict]) -> List[Tuple[int, int]]:
    ranges = []
    seen = set()
    for unit in units:
        normalized = normalize_unit(unit)
        key = (int(normalized["a"]), int(normalized["b"]))
        if key not in seen:
            seen.add(key)
            ranges.append(key)
    return ranges


def field_id(unit: dict) -> str:
    normalized = normalize_unit(unit)
    return str(normalized["repr"])


def safe_field_dir(unit: dict) -> str:
    text = field_id(unit)
    text = text.replace(",", "_")
    text = text.replace("[", "__bits_")
    text = text.replace("]", "")
    text = text.replace(":", "_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    return text.strip("_") or "field"


def bits_repr(bits: Sequence[int]) -> str:
    bits = sorted({int(bit) for bit in bits})
    if not bits:
        return "[]"
    if len(bits) == 1:
        return f"[{bits[0]}]"
    if bits[-1] - bits[0] + 1 == len(bits):
        return f"[{bits[-1]}:{bits[0]}]"
    return "[" + ",".join(str(bit) for bit in bits) + "]"


def width_bits(unit: dict) -> int:
    return int(normalize_unit(unit)["width_bits"])


def read_value(payload: bytes, unit: dict, byteorder: str = "big") -> int:
    normalized = normalize_unit(unit)
    a = int(normalized["a"])
    b = int(normalized["b"])
    raw = payload[a:b + 1]
    if normalized["kind"] == "byte":
        return int.from_bytes(raw, byteorder, signed=False)
    parent_value = int.from_bytes(raw, byteorder, signed=False)
    result = 0
    for out_bit, source_bit in enumerate(normalized["bits"]):
        if (parent_value >> int(source_bit)) & 1:
            result |= (1 << out_bit)
    return result


def write_value(payload: bytes, unit: dict, value: int, byteorder: str = "big") -> bytes:
    normalized = normalize_unit(unit)
    a = int(normalized["a"])
    b = int(normalized["b"])
    if normalized["kind"] == "byte":
        length = b - a + 1
        mask = (1 << (length * 8)) - 1
        value_bytes = (int(value) & mask).to_bytes(length, byteorder, signed=False)
        return payload[:a] + value_bytes + payload[b + 1:]

    raw = payload[a:b + 1]
    parent_width = len(raw) * 8
    parent_value = int.from_bytes(raw, byteorder, signed=False)
    for out_bit, source_bit in enumerate(normalized["bits"]):
        bit_mask = 1 << int(source_bit)
        if (int(value) >> out_bit) & 1:
            parent_value |= bit_mask
        else:
            parent_value &= ~bit_mask
    parent_value &= (1 << parent_width) - 1
    value_bytes = parent_value.to_bytes(len(raw), byteorder, signed=False)
    return payload[:a] + value_bytes + payload[b + 1:]


def values_from_bit_constraints(unit: dict) -> List[int]:
    normalized = normalize_unit(unit)
    if normalized["kind"] != "bit":
        return []
    width = width_bits(normalized)
    mask = (1 << width) - 1 if width > 0 else 0
    values = []
    seen = set()
    for constraint in normalized.get("constraints", []) or []:
        if not isinstance(constraint, dict):
            continue
        op = str(constraint.get("operator") or "").lower()
        raw_value = constraint.get("value")
        value = _parse_subfield_constraint_value(constraint.get("subfield_constraint") or constraint.get("expression"))
        if isinstance(raw_value, str) and raw_value:
            if value is None:
                try:
                    value = int(raw_value, 16)
                except ValueError:
                    value = None
        elif isinstance(raw_value, int):
            if value is None:
                value = raw_value
        if value is None:
            if op == "zero":
                value = 0
            elif op == "nonzero":
                value = 1
        if value is None:
            continue
        value &= mask
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _parse_subfield_constraint_value(text: object) -> Optional[int]:
    if not isinstance(text, str) or "==" not in text:
        return None
    rhs = text.split("==", 1)[1].strip()
    token = rhs.split()[0] if rhs else ""
    try:
        if token.lower().startswith("0x"):
            return int(token, 16)
        return int(token, 10)
    except ValueError:
        return None


def merge_bitfields(byte_units: Sequence[dict], bitfield_result: dict) -> List[dict]:
    """Replace byte fields identified as bitfields with recovered bit subfields."""
    parent_to_subfields: Dict[str, List[dict]] = {}
    for field in bitfield_result.get("fields", []) if isinstance(bitfield_result, dict) else []:
        if not isinstance(field, dict):
            continue
        fid = str(field.get("field_id", ""))
        subs = field.get("subfields", [])
        if fid and isinstance(subs, list) and subs:
            parent_to_subfields[fid] = subs

    merged: List[dict] = []
    for unit in byte_units:
        normalized = normalize_unit(unit)
        parent_id = field_id(normalized)
        subfields = parent_to_subfields.get(parent_id)
        if not subfields:
            merged.append(normalized)
            continue
        for sub in subfields:
            if not isinstance(sub, dict):
                continue
            bits = _parse_subfield_bits(sub.get("label", ""))
            if not bits:
                continue
            merged.append(make_bit_unit(normalized, bits, constraints=sub.get("constraints") or []))
    return merged


def _parse_subfield_bits(label: object) -> List[int]:
    text = str(label or "").strip()
    if not text.startswith("[") or not text.endswith("]"):
        return []
    body = text[1:-1].strip()
    if not body:
        return []
    if ":" in body:
        left, right = body.split(":", 1)
        hi = int(left)
        lo = int(right)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(token.strip()) for token in body.split(",") if token.strip()]
