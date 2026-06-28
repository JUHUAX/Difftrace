#!/usr/bin/env python3
"""Summarize RQ4 program-log judge results on the RQ2-B eval key set."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FULL_CSV = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval/"
    "pairwise_judge_split_seed0_eval_v4/pairwise_judge_results.csv"
)
DEFAULT_SHUFFLED_CSV = Path(
    "/root/semvec/RQ4/out/shuffled_seed_0/program_log_pairwise_judge_results.csv"
)
DEFAULT_NO_LATENT_CSV = Path(
    "/root/semvec/RQ4/out/no_latent_direct/program_log_pairwise_judge_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("/root/semvec/RQ4/out/program_log_judge_on_rq2b_eval")

VALID_VERDICTS = [
    "same_behavior",
    "mostly_same_behavior",
    "weakly_same_behavior",
    "different_behavior",
    "insufficient_information",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RQ4 program-log judge results on the RQ2-B V4 eval field set."
    )
    parser.add_argument("--full-csv", type=Path, default=DEFAULT_FULL_CSV)
    parser.add_argument("--shuffled-csv", type=Path, default=DEFAULT_SHUFFLED_CSV)
    parser.add_argument("--no-latent-csv", type=Path, default=DEFAULT_NO_LATENT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protocol_name", "sample_id", "field_id", "verdict"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            key = (
                str(row.get("protocol_name", "")).strip(),
                str(row.get("sample_id", "")).strip(),
                str(row.get("field_id", "")).strip(),
            )
            if all(key):
                rows[key] = row
    return rows


def pct(value: float) -> str:
    return f"{value:.4f}"


def summarize(
    name: str,
    rows: dict[tuple[str, str, str], dict[str, str]],
    reference_keys: list[tuple[str, str, str]],
) -> dict[str, Any]:
    present_keys = [key for key in reference_keys if key in rows]
    missing_keys = [key for key in reference_keys if key not in rows]
    counts = Counter(rows[key].get("verdict", "") for key in present_keys)
    total = len(present_keys)
    same = counts["same_behavior"]
    mostly = counts["mostly_same_behavior"]
    weakly = counts["weakly_same_behavior"]
    strong = same + mostly
    any_overlap = strong + weakly
    reference_total = len(reference_keys)
    return {
        "name": name,
        "reference_fields": reference_total,
        "covered_fields": total,
        "missing_fields": len(missing_keys),
        "coverage_rate": 0.0 if reference_total == 0 else total / reference_total,
        "same_behavior": same,
        "mostly_same_behavior": mostly,
        "weakly_same_behavior": weakly,
        "different_behavior": counts["different_behavior"],
        "insufficient_information": counts["insufficient_information"],
        "same_behavior_rate": 0.0 if total == 0 else same / total,
        "strong_agreement_rate": 0.0 if total == 0 else strong / total,
        "any_overlap_rate": 0.0 if total == 0 else any_overlap / total,
        "same_behavior_vs_reference": 0.0 if reference_total == 0 else same / reference_total,
        "strong_agreement_vs_reference": 0.0 if reference_total == 0 else strong / reference_total,
        "any_overlap_vs_reference": 0.0 if reference_total == 0 else any_overlap / reference_total,
        "missing_keys": missing_keys,
    }


def summarize_by_protocol(
    rows: dict[tuple[str, str, str], dict[str, str]],
    reference_keys: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    by_protocol: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in reference_keys:
        by_protocol[key[0]].append(key)
    result = []
    for protocol in sorted(by_protocol):
        summary = summarize(protocol, rows, by_protocol[protocol])
        summary["protocol_name"] = protocol
        result.append(summary)
    return result


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "reference_fields",
        "covered_fields",
        "missing_fields",
        "coverage_rate",
        "same_behavior",
        "mostly_same_behavior",
        "weakly_same_behavior",
        "different_behavior",
        "insufficient_information",
        "same_behavior_rate",
        "strong_agreement_rate",
        "any_overlap_rate",
        "same_behavior_vs_reference",
        "strong_agreement_vs_reference",
        "any_overlap_vs_reference",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_protocol_csv(path: Path, rows_by_method: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = [
        "method",
        "protocol_name",
        "reference_fields",
        "covered_fields",
        "missing_fields",
        "coverage_rate",
        "same_behavior",
        "mostly_same_behavior",
        "weakly_same_behavior",
        "different_behavior",
        "insufficient_information",
        "same_behavior_rate",
        "strong_agreement_rate",
        "any_overlap_rate",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in rows_by_method.items():
            for row in rows:
                out = {key: row.get(key, "") for key in fieldnames}
                out["method"] = method
                writer.writerow(out)


def write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    protocol_rows: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = [
        "# RQ4 Program-log Judge on RQ2-B Eval Fields",
        "",
        "Reference field set: RQ2-B V4 split seed0 eval matched fields.",
        "",
        "## Overall",
        "",
        "| Method | Reference Fields | Covered | Missing | Coverage | Same | Mostly | Weakly | Different | Insufficient | Strong/Covered | Any/Covered | Strong/Reference | Any/Reference |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {name} | {reference_fields} | {covered_fields} | {missing_fields} | {coverage_rate} | "
            "{same_behavior} | {mostly_same_behavior} | {weakly_same_behavior} | "
            "{different_behavior} | {insufficient_information} | {same_behavior_rate} | "
            "{strong_agreement_rate} | {any_overlap_rate} | "
            "{strong_agreement_vs_reference} | {any_overlap_vs_reference} |".format(
                name=row["name"],
                reference_fields=row["reference_fields"],
                covered_fields=row["covered_fields"],
                missing_fields=row["missing_fields"],
                coverage_rate=pct(row["coverage_rate"]),
                same_behavior=row["same_behavior"],
                mostly_same_behavior=row["mostly_same_behavior"],
                weakly_same_behavior=row["weakly_same_behavior"],
                different_behavior=row["different_behavior"],
                insufficient_information=row["insufficient_information"],
                same_behavior_rate=pct(row["same_behavior_rate"]),
                strong_agreement_rate=pct(row["strong_agreement_rate"]),
                any_overlap_rate=pct(row["any_overlap_rate"]),
                strong_agreement_vs_reference=pct(row["strong_agreement_vs_reference"]),
                any_overlap_vs_reference=pct(row["any_overlap_vs_reference"]),
            )
        )
    lines.extend(["", "## By Protocol", ""])
    for method, rows in protocol_rows.items():
        lines.extend(
            [
                f"### {method}",
                "",
                "| Protocol | Reference Fields | Covered | Coverage | Same | Mostly | Weakly | Different | Insufficient | Strong/Covered | Any/Covered | Strong/Reference | Any/Reference |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| {protocol_name} | {reference_fields} | {covered_fields} | {coverage_rate} | "
                "{same_behavior} | {mostly_same_behavior} | {weakly_same_behavior} | "
                "{different_behavior} | {insufficient_information} | "
                "{strong_agreement_rate} | {any_overlap_rate} | "
                "{strong_agreement_vs_reference} | {any_overlap_vs_reference} |".format(
                    protocol_name=row["protocol_name"],
                    reference_fields=row["reference_fields"],
                    covered_fields=row["covered_fields"],
                    coverage_rate=pct(row["coverage_rate"]),
                    same_behavior=row["same_behavior"],
                    mostly_same_behavior=row["mostly_same_behavior"],
                    weakly_same_behavior=row["weakly_same_behavior"],
                    different_behavior=row["different_behavior"],
                    insufficient_information=row["insufficient_information"],
                    strong_agreement_rate=pct(row["strong_agreement_rate"]),
                    any_overlap_rate=pct(row["any_overlap_rate"]),
                    strong_agreement_vs_reference=pct(row["strong_agreement_vs_reference"]),
                    any_overlap_vs_reference=pct(row["any_overlap_vs_reference"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_rows = read_rows(args.full_csv)
    shuffled_rows = read_rows(args.shuffled_csv)
    no_latent_rows = read_rows(args.no_latent_csv)
    reference_keys = sorted(full_rows)

    methods = {
        "Full baseline": full_rows,
        "Shuffled groups": shuffled_rows,
        "No-latent direct": no_latent_rows,
    }
    summaries = [summarize(name, rows, reference_keys) for name, rows in methods.items()]
    protocol_rows = {
        name: summarize_by_protocol(rows, reference_keys) for name, rows in methods.items()
    }

    write_summary_csv(args.output_dir / "rq4_program_log_judge_on_rq2b_eval_summary.csv", summaries)
    write_protocol_csv(
        args.output_dir / "rq4_program_log_judge_on_rq2b_eval_by_protocol.csv",
        protocol_rows,
    )
    write_markdown(
        args.output_dir / "rq4_program_log_judge_on_rq2b_eval_readable.md",
        summaries,
        protocol_rows,
    )
    print(
        "[rq4-program-log-summary] wrote "
        f"{args.output_dir / 'rq4_program_log_judge_on_rq2b_eval_readable.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
