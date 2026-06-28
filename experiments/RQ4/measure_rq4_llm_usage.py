#!/usr/bin/env python3
"""Measure actual API token usage for RQ4 Full vs No-latent prompts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import runpy
import time
from pathlib import Path
from typing import Any


DEFAULT_FULL_INPUT = Path("/root/semvec/difftrace/stage4/out/stage4_field_profiles/field_semantic_profiles.jsonl")
DEFAULT_NO_LATENT_INPUT = Path("/root/semvec/difftrace/stage3/out/stage3_filtered/stage3_dataset_semantic_fields.csv")
DEFAULT_OUTPUT_DIR = Path("/root/semvec/RQ4/out/llm_usage_measurement")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

FULL_SCRIPT = Path("/root/semvec/difftrace/stage4/run_stage4_field_semantic_fusion.py")
NO_LATENT_SCRIPT = Path("/root/semvec/RQ4/run_rq4_no_latent_direct_summary.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-input", type=Path, default=DEFAULT_FULL_INPUT)
    parser.add_argument("--no-latent-input", type=Path, default=DEFAULT_NO_LATENT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--api-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-top-p", type=float, default=1.0)
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and estimate text size without API calls.")
    return parser.parse_args()


def field_uid(row: dict[str, Any]) -> str:
    return f"{row.get('protocol_name')}-{row.get('sample_id')}-{row.get('field_id')}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    result = {}
    for key in dir(usage):
        if key.startswith("_"):
            continue
        value = getattr(usage, key)
        if isinstance(value, (str, int, float, bool, type(None), dict, list)):
            result[key] = value
    return result


def call_api(prompt: str, args: argparse.Namespace) -> tuple[str, dict[str, Any], float]:
    from openai import OpenAI

    api_key = args.api_key or DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing API key")
    client = OpenAI(api_key=api_key, base_url=args.api_base_url, timeout=args.api_timeout)
    start = time.time()
    response = client.chat.completions.create(
        model=args.api_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.api_temperature,
        top_p=args.api_top_p,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content or ""
    return content, usage_to_dict(getattr(response, "usage", None)), elapsed


def flatten_usage(prefix: str, usage: dict[str, Any], row: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, dict):
            flatten_usage(f"{prefix}{key}.", value, row)
        elif isinstance(value, list):
            row[f"{prefix}{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            row[f"{prefix}{key}"] = value


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_ns = runpy.run_path(str(FULL_SCRIPT), run_name="__rq4_usage_full__")
    no_ns = runpy.run_path(str(NO_LATENT_SCRIPT), run_name="__rq4_usage_no_latent__")

    full_template = full_ns["PROMPT_TEMPLATE"]
    full_rows = load_jsonl(args.full_input)
    no_rows = load_csv_rows(args.no_latent_input)
    no_by_uid = {field_uid(row): row for row in no_rows}
    full_by_uid = {field_uid(row): row for row in full_rows}
    common_uids = sorted(set(full_by_uid) & set(no_by_uid))
    if not common_uids:
        raise SystemExit("no common field_uid between Full and No-latent inputs")

    rng = random.Random(args.random_seed)
    sample_uids = common_uids[:]
    rng.shuffle(sample_uids)
    sample_uids = sample_uids[: args.sample_size]

    records: list[dict[str, Any]] = []
    for index, uid in enumerate(sample_uids, start=1):
        prompts = {
            "full_field_fusion": full_ns["build_prompt"](full_template, full_by_uid[uid]),
            "no_latent_direct": no_ns["build_prompt"](no_by_uid[uid]),
        }
        for method, prompt in prompts.items():
            row: dict[str, Any] = {
                "index": index,
                "field_uid": uid,
                "method": method,
                "prompt_chars": len(prompt),
                "estimated_prompt_tokens_chars_div4": len(prompt) / 4,
            }
            if args.dry_run:
                row["response_chars"] = 0
                row["elapsed_seconds"] = 0.0
            else:
                print(f"[measure] {index}/{len(sample_uids)} {method} {uid}")
                response_text, usage, elapsed = call_api(prompt, args)
                row["response_chars"] = len(response_text)
                row["elapsed_seconds"] = round(elapsed, 3)
                flatten_usage("", usage, row)
            records.append(row)

    csv_path = args.output_dir / "rq4_llm_usage_measurement.csv"
    fieldnames = sorted({key for row in records for key in row})
    preferred = [
        "index",
        "field_uid",
        "method",
        "prompt_chars",
        "estimated_prompt_tokens_chars_div4",
        "response_chars",
        "elapsed_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {}
    for method in sorted({row["method"] for row in records}):
        method_rows = [row for row in records if row["method"] == method]
        summary[method] = {
            "calls": len(method_rows),
            "prompt_chars": sum(float(row.get("prompt_chars") or 0) for row in method_rows),
            "response_chars": sum(float(row.get("response_chars") or 0) for row in method_rows),
            "prompt_tokens": sum(float(row.get("prompt_tokens") or 0) for row in method_rows),
            "completion_tokens": sum(float(row.get("completion_tokens") or 0) for row in method_rows),
            "total_tokens": sum(float(row.get("total_tokens") or 0) for row in method_rows),
            "prompt_cache_hit_tokens": sum(float(row.get("prompt_cache_hit_tokens") or 0) for row in method_rows),
            "prompt_cache_miss_tokens": sum(float(row.get("prompt_cache_miss_tokens") or 0) for row in method_rows),
            "elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in method_rows),
        }
    summary_path = args.output_dir / "rq4_llm_usage_measurement_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[measure] wrote {csv_path}")
    print(f"[measure] wrote {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
