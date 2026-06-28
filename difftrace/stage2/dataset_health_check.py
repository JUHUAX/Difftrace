#!/usr/bin/env python3
"""Dataset health check for field-level training samples."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


V3_CONTEXT_COLS = [
    "relative_start",
    "field_instr_ratio",
    "compare_ratio",
    "constraint_value_diversity",
]
V3_GROUPS = ["neighborhood", "boundary", "enum", "extreme"]
V3_GROUP_SUMMARY_COLS = [
    "mean_baseline_distance",
    "mean_pairwise_distance",
    "max_pairwise_distance",
    "metric_vector_variance",
    "unique_vector_ratio",
    "loop_dispersion",
]
V3_MODEL_FEATURE_COLS = V3_CONTEXT_COLS + [
    f"{group}_{feature}"
    for group in V3_GROUPS
    for feature in V3_GROUP_SUMMARY_COLS
]
MODEL_FEATURE_COLS = V3_MODEL_FEATURE_COLS
DIAGNOSIS_COLS = [
    "valid_mutations",
    "mutation_count",
    "boundary_miss",
    "deltaf_dispersion",
    "unique_metric_vectors",
    "metrics_source",
    "mutations_json_count",
    "mutation_status_ok",
    "mutation_status_non_ok",
    "baseline_status",
    "baseline_taint_found",
    "baseline_bb_parsed",
    "baseline_branch_parsed",
    "baseline_instr_parsed",
    "baseline_bad_lines_dropped",
    "baseline_preprocess_line_count",
    "baseline_taint_thread_id",
]
DEFAULT_CSV = Path("/root/semvec/difftrace/out/field_training_samples.csv")
DEFAULT_SUMMARY = Path("/root/semvec/difftrace/out/field_training_summary.json")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/difftrace/out/dataset_health_report")


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[INFO] {message}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> List[str]:
    available = [col for col in columns if col in df.columns]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        warn(f"Missing {label} columns: {missing}")
    return available


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    model_feature_cols = ensure_columns(df, MODEL_FEATURE_COLS, "model feature")
    diagnosis_cols = [col for col in DIAGNOSIS_COLS if col in df.columns]
    metadata_cols = [col for col in df.columns if col not in set(model_feature_cols) | set(diagnosis_cols)]
    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    return {
        "metadata_cols": metadata_cols,
        "diagnosis_cols": diagnosis_cols,
        "model_feature_cols": model_feature_cols,
        "numeric_cols": numeric_cols,
    }


def maybe_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    converted = df.copy()
    for col in columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def distribution_stats(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    quantiles = clean.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "max": float(clean.max()),
    }


def frame_distribution_stats(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {col: distribution_stats(df[col]) for col in columns if col in df.columns}


def protocol_stats_table(df: pd.DataFrame, summary_json: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if "protocol_name" not in df.columns:
        raise ValueError("Missing protocol_name column")
    if "sample_id" not in df.columns:
        raise ValueError("Missing sample_id column")

    grouped = df.groupby("protocol_name", dropna=False)
    rows: List[Dict[str, Any]] = []
    total_fields = len(df)
    for protocol, group in grouped:
        sample_count = group["sample_id"].nunique(dropna=True)
        row = {
            "protocol_name": protocol,
            "field_sample_count": int(len(group)),
            "sample_count": int(sample_count),
            "avg_fields_per_sample": float(len(group) / sample_count) if sample_count else np.nan,
            "field_sample_ratio": float(len(group) / total_fields) if total_fields else np.nan,
            "boundary_miss_count": int(pd.to_numeric(group.get("boundary_miss"), errors="coerce").fillna(0).astype(float).gt(0).sum())
            if "boundary_miss" in group
            else 0,
            "boundary_miss_ratio": float(pd.to_numeric(group.get("boundary_miss"), errors="coerce").fillna(0).astype(float).gt(0).mean())
            if "boundary_miss" in group
            else np.nan,
            "mutation_count_mean": float(pd.to_numeric(group.get("mutation_count"), errors="coerce").mean())
            if "mutation_count" in group
            else np.nan,
            "valid_mutations_mean": float(pd.to_numeric(group.get("valid_mutations"), errors="coerce").mean())
            if "valid_mutations" in group
            else np.nan,
            "unique_metric_vectors_mean": float(pd.to_numeric(group.get("unique_metric_vectors"), errors="coerce").mean())
            if "unique_metric_vectors" in group
            else np.nan,
            "deltaf_dispersion_mean": float(pd.to_numeric(group.get("deltaf_dispersion"), errors="coerce").mean())
            if "deltaf_dispersion" in group
            else np.nan,
        }
        if summary_json and isinstance(summary_json.get("per_protocol"), dict):
            summary_row = summary_json["per_protocol"].get(str(protocol), {})
            if isinstance(summary_row, dict):
                row["summary_field_sample_count"] = summary_row.get("field_sample_count")
                row["summary_sample_count"] = summary_row.get("sample_count")
                row["summary_avg_fields_per_sample"] = summary_row.get("avg_fields_per_sample")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("field_sample_count", ascending=False)


def grouped_distribution_table(df: pd.DataFrame, group_col: str, columns: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if group_col not in df.columns:
        return pd.DataFrame()
    for group_name, group in df.groupby(group_col, dropna=False):
        for col in columns:
            if col not in group.columns:
                continue
            stats = distribution_stats(group[col])
            rows.append({"group": group_name, "column": col, **stats})
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in columns:
        if col not in df.columns:
            rows.append(
                {
                    "feature": col,
                    "exists": False,
                    "missing_count": total,
                    "missing_ratio": 1.0 if total else np.nan,
                }
            )
            continue
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "exists": True,
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_ratio", "feature"], ascending=[False, True])


def variance_report(df: pd.DataFrame, columns: Sequence[str], low_variance_threshold: float) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        clean = series.dropna()
        unique_count = int(clean.nunique(dropna=True))
        std = float(clean.std(ddof=0)) if len(clean) > 1 else 0.0
        rows.append(
            {
                "feature": col,
                "non_null_count": int(clean.count()),
                "unique_value_count": unique_count,
                "std": std,
                "is_constant": bool(unique_count <= 1),
                "is_low_variance": bool(std <= low_variance_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_constant", "is_low_variance", "std", "feature"], ascending=[False, False, True, True])


def iqr_outlier_report(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            rows.append({"feature": col, "non_null_count": 0})
            continue
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (clean < lower) | (clean > upper)
        rows.append(
            {
                "feature": col,
                "non_null_count": int(clean.count()),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "p99": float(clean.quantile(0.99)),
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "iqr_lower_bound": float(lower),
                "iqr_upper_bound": float(upper),
                "outlier_count_iqr": int(outlier_mask.sum()),
                "outlier_ratio_iqr": float(outlier_mask.mean()),
                "global_ratio": float(outlier_mask.sum() / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["outlier_ratio_iqr", "feature"], ascending=[False, True])


def correlation_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.DataFrame()
    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    return numeric.corr()


def high_correlation_pairs(corr: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    if corr.empty:
        return pairs
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "corr": float(value)})
    pairs.sort(key=lambda item: abs(item["corr"]), reverse=True)
    return pairs


def make_readiness_assessment(
    df: pd.DataFrame,
    missing_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    model_feature_cols: Sequence[str],
) -> Dict[str, Any]:
    blocking_issues: List[str] = []
    cautions: List[str] = []

    total_fields = len(df)
    boundary_ratio = float(pd.to_numeric(df.get("boundary_miss"), errors="coerce").fillna(0).gt(0).mean()) if "boundary_miss" in df else np.nan
    low_mutation_ratio_lt3 = (
        float(pd.to_numeric(df.get("mutation_count"), errors="coerce").lt(3).mean()) if "mutation_count" in df else np.nan
    )
    any_missing_model = bool((missing_df["missing_count"] > 0).any()) if not missing_df.empty else False
    constant_features = variance_df.loc[variance_df["is_constant"], "feature"].tolist() if not variance_df.empty else []

    if total_fields < 100:
        blocking_issues.append("Field sample size is small (<100); latent structure learning may be unstable.")
    if len(model_feature_cols) < 28:
        blocking_issues.append("Model feature columns are incomplete.")
    if any_missing_model:
        cautions.append("Some model feature columns contain missing values; imputation or filtering is needed.")
    if constant_features:
        cautions.append(f"Constant model features detected: {constant_features}")
    if pd.notna(boundary_ratio) and boundary_ratio > 0.5:
        cautions.append(f"boundary_miss ratio is high ({boundary_ratio:.3f}); field partitions may be noisy.")
    if pd.notna(low_mutation_ratio_lt3) and low_mutation_ratio_lt3 > 0.25:
        cautions.append(
            f"A large fraction of fields have mutation_count < 3 ({low_mutation_ratio_lt3:.3f}); mutation coverage may be thin."
        )

    ready = len(blocking_issues) == 0
    return {
        "ready_for_first_pass_unsupervised_learning": ready,
        "blocking_issues": blocking_issues,
        "cautions": cautions,
        "recommended_model_input": list(model_feature_cols),
    }


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_lines(values: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


FONT_5X7: Dict[str, List[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


class Canvas:
    def __init__(self, width: int, height: int, bg: Tuple[int, int, int] = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
        self.pixels[:, :] = bg

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        x0, x1 = sorted((max(0, x0), min(self.width, x1)))
        y0, y1 = sorted((max(0, y0), min(self.height, y1)))
        if x0 >= x1 or y0 >= y1:
            return
        self.pixels[y0:y1, x0:x1] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], thickness: int = 1) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            t = step / steps
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            self.fill_rect(x - thickness // 2, y - thickness // 2, x + thickness // 2 + 1, y + thickness // 2 + 1, color)

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = (0, 0, 0),
        scale: int = 1,
        vertical: bool = False,
    ) -> None:
        cursor_x = x
        cursor_y = y
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            for row_idx, row in enumerate(glyph):
                for col_idx, value in enumerate(row):
                    if value == "1":
                        px = cursor_x + col_idx * scale
                        py = cursor_y + row_idx * scale
                        self.fill_rect(px, py, px + scale, py + scale, color)
            if vertical:
                cursor_y += 8 * scale
            else:
                cursor_x += 6 * scale

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = []
        for row in self.pixels:
            raw_rows.append(b"\x00" + row.tobytes())
        raw = b"".join(raw_rows)
        compressed = zlib.compress(raw)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", compressed),
            chunk(b"IEND", b""),
        ]
        path.write_bytes(b"".join(png))


def value_to_y(value: float, min_val: float, max_val: float, top: int, bottom: int) -> int:
    if not np.isfinite(value):
        return bottom
    if max_val <= min_val:
        return (top + bottom) // 2
    ratio = (value - min_val) / (max_val - min_val)
    ratio = min(max(ratio, 0.0), 1.0)
    return int(round(bottom - ratio * (bottom - top)))


def render_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
    path: Path,
    width: int = 1100,
    height: int = 700,
) -> None:
    canvas = Canvas(width, height)
    left, right, top, bottom = 90, width - 40, 80, height - 120
    canvas.draw_text(20, 20, title, scale=2)
    canvas.line(left, top, left, bottom, (0, 0, 0), 2)
    canvas.line(left, bottom, right, bottom, (0, 0, 0), 2)
    canvas.draw_text(20, 120, y_label, vertical=True)
    max_val = max(values) if values else 1.0
    max_val = max(max_val, 1.0)
    bar_space = (right - left) / max(len(values), 1)
    bar_width = max(12, int(bar_space * 0.65))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = int(left + idx * bar_space + (bar_space - bar_width) / 2)
        x1 = x0 + bar_width
        y1 = bottom
        y0 = value_to_y(float(value), 0.0, max_val, top, bottom)
        canvas.fill_rect(x0, y0, x1, y1, (60, 120, 216))
        canvas.draw_text(x0, min(bottom + 10, height - 20), label[:10], scale=1)
        canvas.draw_text(x0, max(y0 - 12, top), f"{float(value):.2f}"[:8], scale=1)
    for tick in range(6):
        val = max_val * tick / 5.0
        y = value_to_y(val, 0.0, max_val, top, bottom)
        canvas.line(left - 5, y, left, y, (0, 0, 0), 1)
        canvas.line(left, y, right, y, (230, 230, 230), 1)
        canvas.draw_text(10, y - 4, f"{val:.1f}"[:8], scale=1)
    canvas.save_png(path)


def render_histogram(values: Sequence[float], bins: int, title: str, path: Path) -> None:
    clean = np.array([v for v in values if pd.notna(v)], dtype=float)
    if clean.size == 0:
        clean = np.array([0.0])
    counts, edges = np.histogram(clean, bins=bins)
    labels = []
    for i in range(len(counts)):
        labels.append(f"{edges[i]:.0f}")
    render_bar_chart(labels, counts.tolist(), title, "COUNT", path)


def render_boundary_ratio_chart(protocol_df: pd.DataFrame, path: Path) -> None:
    labels = protocol_df["protocol_name"].astype(str).tolist()
    values = protocol_df["boundary_miss_ratio"].fillna(0).tolist()
    render_bar_chart(labels, values, "BOUNDARY MISS RATIO BY PROTOCOL", "RATIO", path)


def render_boxplot(
    grouped_values: Dict[str, Sequence[float]],
    title: str,
    path: Path,
    width: int = 1200,
    height: int = 700,
) -> None:
    canvas = Canvas(width, height)
    left, right, top, bottom = 90, width - 40, 80, height - 120
    canvas.draw_text(20, 20, title, scale=2)
    canvas.line(left, top, left, bottom, (0, 0, 0), 2)
    canvas.line(left, bottom, right, bottom, (0, 0, 0), 2)

    all_values = [float(v) for values in grouped_values.values() for v in values if pd.notna(v)]
    if not all_values:
        all_values = [0.0]
    min_val = min(all_values)
    max_val = max(all_values)
    if max_val == min_val:
        max_val = min_val + 1.0

    labels = list(grouped_values.keys())
    box_space = (right - left) / max(len(labels), 1)
    for idx, label in enumerate(labels):
        values = pd.Series(grouped_values[label]).dropna().astype(float)
        if values.empty:
            continue
        q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
        whisker_low = float(values.min())
        whisker_high = float(values.max())
        x_center = int(left + idx * box_space + box_space / 2)
        box_half = max(8, int(box_space * 0.18))
        y_q1 = value_to_y(float(q1), min_val, max_val, top, bottom)
        y_q2 = value_to_y(float(q2), min_val, max_val, top, bottom)
        y_q3 = value_to_y(float(q3), min_val, max_val, top, bottom)
        y_low = value_to_y(whisker_low, min_val, max_val, top, bottom)
        y_high = value_to_y(whisker_high, min_val, max_val, top, bottom)
        canvas.line(x_center, y_low, x_center, y_q1, (0, 0, 0), 1)
        canvas.line(x_center, y_q3, x_center, y_high, (0, 0, 0), 1)
        canvas.fill_rect(x_center - box_half, y_q3, x_center + box_half, y_q1, (171, 204, 255))
        canvas.line(x_center - box_half, y_q2, x_center + box_half, y_q2, (200, 50, 50), 2)
        canvas.line(x_center - box_half, y_low, x_center + box_half, y_low, (0, 0, 0), 1)
        canvas.line(x_center - box_half, y_high, x_center + box_half, y_high, (0, 0, 0), 1)
        canvas.draw_text(x_center - box_half, bottom + 12, label[:10], scale=1)
    for tick in range(6):
        val = min_val + (max_val - min_val) * tick / 5.0
        y = value_to_y(val, min_val, max_val, top, bottom)
        canvas.line(left, y, right, y, (235, 235, 235), 1)
        canvas.draw_text(10, y - 4, f"{val:.2f}"[:8], scale=1)
    canvas.save_png(path)


def render_heatmap(corr: pd.DataFrame, title: str, path: Path, cell_size: int = 22) -> None:
    if corr.empty:
        corr = pd.DataFrame(np.zeros((1, 1)), columns=["NA"], index=["NA"])
    n = len(corr)
    width = 420 + n * cell_size
    height = 220 + n * cell_size
    canvas = Canvas(width, height)
    left, top = 220, 120
    canvas.draw_text(20, 20, title, scale=2)

    def corr_color(value: float) -> Tuple[int, int, int]:
        if not np.isfinite(value):
            return (220, 220, 220)
        value = max(-1.0, min(1.0, value))
        if value >= 0:
            base = int(255 - 135 * value)
            return (255, base, base)
        base = int(255 - 135 * abs(value))
        return (base, base, 255)

    for row_idx, row_name in enumerate(corr.index):
        canvas.draw_text(10, top + row_idx * cell_size + 5, str(row_name)[:28], scale=1)
        canvas.draw_text(left + row_idx * cell_size + 2, 80, str(row_name)[:8], scale=1, vertical=True)
        for col_idx, col_name in enumerate(corr.columns):
            value = float(corr.loc[row_name, col_name])
            x0 = left + col_idx * cell_size
            y0 = top + row_idx * cell_size
            canvas.fill_rect(x0, y0, x0 + cell_size, y0 + cell_size, corr_color(value))
            canvas.line(x0, y0, x0 + cell_size, y0, (255, 255, 255), 1)
            canvas.line(x0, y0, x0, y0 + cell_size, (255, 255, 255), 1)
    canvas.save_png(path)


def pick_boxplot_features(outlier_df: pd.DataFrame, model_feature_cols: Sequence[str], limit: int = 6) -> List[str]:
    chosen = []
    focus_prefixes = ["instr_delta_ratio", "cmp_delta_ratio", "bb_multiset_l1_ratio"]
    for prefix in focus_prefixes:
        for col in model_feature_cols:
            if col.startswith(prefix):
                chosen.append(col)
                if len(chosen) >= limit:
                    return chosen[:limit]
    if not outlier_df.empty:
        chosen.extend(outlier_df["feature"].head(limit).tolist())
    deduped: List[str] = []
    for col in chosen:
        if col not in deduped:
            deduped.append(col)
    return deduped[:limit]


def summary_protocol_section(protocol_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for _, row in protocol_df.iterrows():
        rows.append({key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Health check for field-level training samples.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV, help="Path to field_training_samples.csv")
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_SUMMARY, help="Path to field_training_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--low-variance-threshold", type=float, default=1e-6, help="Threshold for low-variance flagging")
    parser.add_argument("--corr-threshold", type=float, default=0.9, help="Absolute correlation threshold for reporting")
    parser.add_argument("--hist-bins", type=int, default=15, help="Histogram bins")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output")
    args = parser.parse_args()

    if not args.input_csv.exists():
        warn(f"Missing input CSV: {args.input_csv}")
        return 1
    if not args.input_summary.exists():
        warn(f"Missing input summary JSON: {args.input_summary}")
        return 1

    info(f"Reading {args.input_csv}", args.quiet)
    df_raw = pd.read_csv(args.input_csv)
    if df_raw.empty:
        warn("Input CSV is empty.")
        return 1

    info(f"Reading {args.input_summary}", args.quiet)
    summary_json = load_json(args.input_summary)

    column_groups = classify_columns(df_raw)
    numeric_cols = column_groups["numeric_cols"]
    df = maybe_numeric(df_raw, numeric_cols)
    model_feature_cols = column_groups["model_feature_cols"]
    diagnosis_numeric = [
        col
        for col in ["mutation_count", "valid_mutations", "boundary_miss", "deltaf_dispersion", "unique_metric_vectors"]
        if col in df.columns
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    info("Computing protocol-level statistics", args.quiet)
    protocol_df = protocol_stats_table(df, summary_json)
    write_table(protocol_df, output_dir / "per_protocol_stats.csv")

    info("Computing diagnosis distributions", args.quiet)
    diagnosis_global = frame_distribution_stats(df, diagnosis_numeric)
    diagnosis_by_protocol_df = grouped_distribution_table(df, "protocol_name", diagnosis_numeric)
    if not diagnosis_by_protocol_df.empty:
        write_table(diagnosis_by_protocol_df, output_dir / "diagnosis_by_protocol_stats.csv")

    boundary_miss_global_ratio = (
        float(pd.to_numeric(df["boundary_miss"], errors="coerce").fillna(0).gt(0).mean()) if "boundary_miss" in df else np.nan
    )
    mutation_lt3_count = int(pd.to_numeric(df["mutation_count"], errors="coerce").lt(3).sum()) if "mutation_count" in df else 0
    mutation_lt5_count = int(pd.to_numeric(df["mutation_count"], errors="coerce").lt(5).sum()) if "mutation_count" in df else 0
    unique_eq1_count = (
        int(pd.to_numeric(df["unique_metric_vectors"], errors="coerce").fillna(np.nan).eq(1).sum())
        if "unique_metric_vectors" in df
        else 0
    )

    info("Computing feature quality reports", args.quiet)
    missing_df = missing_report(df, model_feature_cols)
    variance_df = variance_report(df, model_feature_cols, args.low_variance_threshold)
    outlier_df = iqr_outlier_report(df, model_feature_cols)
    write_table(missing_df, output_dir / "feature_missing_report.csv")
    write_table(variance_df, output_dir / "feature_variance_report.csv")
    write_table(outlier_df, output_dir / "feature_outlier_report.csv")
    write_lines(model_feature_cols, output_dir / "model_feature_cols.txt")
    write_json({"model_feature_cols": model_feature_cols}, output_dir / "model_feature_cols.json")

    info("Computing correlations", args.quiet)
    corr = correlation_matrix(df, model_feature_cols)
    if not corr.empty:
        corr.to_csv(output_dir / "feature_correlation_matrix.csv", index=True)
    high_corr_pairs = high_correlation_pairs(corr, args.corr_threshold)
    if high_corr_pairs:
        write_table(pd.DataFrame(high_corr_pairs), output_dir / "high_correlation_pairs.csv")

    info("Rendering charts", args.quiet)
    render_bar_chart(
        protocol_df["protocol_name"].astype(str).tolist(),
        protocol_df["field_sample_count"].astype(float).tolist(),
        "FIELD SAMPLE DISTRIBUTION BY PROTOCOL",
        "FIELDS",
        output_dir / "protocol_sample_distribution.png",
    )
    if "mutation_count" in df:
        render_histogram(
            pd.to_numeric(df["mutation_count"], errors="coerce").dropna().tolist(),
            bins=args.hist_bins,
            title="MUTATION COUNT HISTOGRAM",
            path=output_dir / "mutation_count_hist.png",
        )
    if "unique_metric_vectors" in df:
        render_histogram(
            pd.to_numeric(df["unique_metric_vectors"], errors="coerce").dropna().tolist(),
            bins=args.hist_bins,
            title="UNIQUE METRIC VECTORS HISTOGRAM",
            path=output_dir / "unique_metric_vectors_hist.png",
        )
    render_boundary_ratio_chart(protocol_df, output_dir / "boundary_miss_by_protocol.png")
    render_heatmap(corr, "MODEL FEATURE CORRELATION", output_dir / "feature_correlation_heatmap.png")

    if "deltaf_dispersion" in df and "protocol_name" in df:
        grouped_delta = {
            str(protocol): pd.to_numeric(group["deltaf_dispersion"], errors="coerce").dropna().tolist()
            for protocol, group in df.groupby("protocol_name", dropna=False)
        }
        render_boxplot(grouped_delta, "DELTAF DISPERSION BY PROTOCOL", output_dir / "deltaf_dispersion_boxplot.png")

    chosen_boxplot_features = pick_boxplot_features(outlier_df, model_feature_cols)
    if chosen_boxplot_features:
        grouped_boxplot = {}
        for feature in chosen_boxplot_features:
            grouped_boxplot[feature[:10]] = pd.to_numeric(df[feature], errors="coerce").dropna().tolist()
        render_boxplot(grouped_boxplot, "SELECTED FEATURE BOXPLOTS", output_dir / "selected_feature_boxplots.png")

    readiness = make_readiness_assessment(df, missing_df, variance_df, model_feature_cols)
    total_samples = None
    if "protocol_name" in df.columns and "sample_id" in df.columns:
        total_samples = int(df[["protocol_name", "sample_id"]].drop_duplicates().shape[0])
    elif "sample_id" in df.columns:
        total_samples = int(df["sample_id"].nunique())
    health_summary = {
        "input_paths": {
            "field_training_samples_csv": str(args.input_csv),
            "field_training_summary_json": str(args.input_summary),
        },
        "dataset_size": {
            "total_field_samples": int(len(df)),
            "total_protocols": int(df["protocol_name"].nunique()) if "protocol_name" in df else None,
            "total_samples": total_samples,
        },
        "column_groups": column_groups,
        "protocol_distribution": summary_protocol_section(protocol_df),
        "diagnosis_global": diagnosis_global,
        "diagnosis_flags": {
            "boundary_miss_global_ratio": boundary_miss_global_ratio,
            "fields_with_mutation_count_lt_3": mutation_lt3_count,
            "fields_with_mutation_count_lt_5": mutation_lt5_count,
            "fields_with_unique_metric_vectors_eq_1": unique_eq1_count,
        },
        "feature_quality": {
            "model_feature_count": len(model_feature_cols),
            "missing_features": missing_df.to_dict(orient="records"),
            "variance_summary": {
                "constant_features": variance_df.loc[variance_df["is_constant"], "feature"].tolist() if not variance_df.empty else [],
                "low_variance_features": variance_df.loc[variance_df["is_low_variance"], "feature"].tolist() if not variance_df.empty else [],
            },
            "outlier_focus": outlier_df.head(12).to_dict(orient="records"),
            "high_correlation_pairs": high_corr_pairs[:50],
        },
        "readiness_assessment": readiness,
        "output_files": {
            "summary_json": str(output_dir / "dataset_health_summary.json"),
            "per_protocol_stats_csv": str(output_dir / "per_protocol_stats.csv"),
            "feature_missing_report_csv": str(output_dir / "feature_missing_report.csv"),
            "feature_variance_report_csv": str(output_dir / "feature_variance_report.csv"),
            "feature_outlier_report_csv": str(output_dir / "feature_outlier_report.csv"),
            "model_feature_cols_txt": str(output_dir / "model_feature_cols.txt"),
            "protocol_sample_distribution_png": str(output_dir / "protocol_sample_distribution.png"),
            "mutation_count_hist_png": str(output_dir / "mutation_count_hist.png"),
            "unique_metric_vectors_hist_png": str(output_dir / "unique_metric_vectors_hist.png"),
            "boundary_miss_by_protocol_png": str(output_dir / "boundary_miss_by_protocol.png"),
            "feature_correlation_heatmap_png": str(output_dir / "feature_correlation_heatmap.png"),
            "selected_feature_boxplots_png": str(output_dir / "selected_feature_boxplots.png"),
            "deltaf_dispersion_boxplot_png": str(output_dir / "deltaf_dispersion_boxplot.png"),
        },
    }
    write_json(health_summary, output_dir / "dataset_health_summary.json")

    info(f"Wrote dataset health report to {output_dir}", args.quiet)
    info(
        f"Ready for first-pass unsupervised learning: {readiness['ready_for_first_pass_unsupervised_learning']}",
        args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
