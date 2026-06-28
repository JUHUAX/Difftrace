#!/usr/bin/env python3
"""Run pairwise summary judge for RQ2-B program-log semantic evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROGRAM_LOG_JSONL = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval/program_log_groundtruth_candidates.jsonl"
)
DEFAULT_STAGE4_PROFILES = Path(
    "/root/semvec/difftrace/stage4/out/stage4_field_semantic_fusion/field_semantic_fused_profiles.jsonl"
)
DEFAULT_OUTPUT_CSV = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval/pairwise_judge_results.csv"
)
DEFAULT_OUTPUT_MD = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval/pairwise_judge_readable.md"
)
DEFAULT_RUN_LOG = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/eval/pairwise_judge_run.log"
)
DEFAULT_RETRY_DELAY_SECONDS = 30 * 60
DEFAULT_WORKERS = 5
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

# Keep aligned with the other RQ2-B/Stage4 LLM scripts.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

VALID_VERDICTS = {
    "same_behavior",
    "mostly_same_behavior",
    "weakly_same_behavior",
    "different_behavior",
    "insufficient_information",
}

PRINT_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()


PROMPT_TEMPLATE = """你是一个程序行为描述一致性评估助手。

# 任务

你将看到协议处理程序对同一个字段的解析和消费行为的两段描述：Input A 和 Input B。

这两段描述关注的不是字段在协议规范中的角色，而是程序如何使用该字段，例如读取、比较、分支、拆位、组装、存储、传播、影响循环、影响处理范围、后期回写或普通数据搬运等。

你的任务是判断：

```text
Input A 和 Input B 是否描述了相同或兼容的程序解析行为？
```

请只比较两段描述中的程序行为含义，不要判断字段的传统协议类型，也不要推断协议规范。

# 输入

## Input A

```text
{input_a_description}
```

## Input B

```text
{input_b_description}
```

# 判断原则

请按以下步骤判断：

1. 分别抽取 Input A 和 Input B 中的程序行为主张。
   - 例如：早期常量比较、条件分支、范围检查、多路径处理、bit 子段提取、多字节数值组装、循环或批量处理、地址/索引访问、存储/传播、后期回写、透明搬运等。
   - 如果某段描述包含多个行为，请拆开判断。

2. 判断两段描述的核心行为是否一致。
   - 如果两段都描述同一类主要行为，即使措辞或抽象层级不同，也应视为一致或大体一致。
   - 如果一段更具体、一段更抽象，只要抽象描述能覆盖具体行为，可以视为一致或大体一致。
   - 如果两段仅有明确的局部行为重叠，但无法确认核心行为一致，应视为弱一致。
   - 如果两段只共享很弱的泛化词，但关键行为不同，应视为不一致。

3. 判断是否存在单侧行为。
   - 如果某个重要行为只在 Input A 中出现，记录到 `input_a_only_behaviors`。
   - 如果某个重要行为只在 Input B 中出现，记录到 `input_b_only_behaviors`。
   - 不要因为某段没有复述所有细节就直接判为不一致；只记录对核心语义有影响的重要遗漏。

4. 判断是否存在冲突行为。
   - 如果两段对字段行为的主要性质描述相反或明显不兼容，记录到 `conflicting_behaviors`。
   - 例如一段主要描述条件分支/路径选择，另一段主要描述无控制作用的数据搬运；或者一段描述循环规模影响，另一段明确只描述单次存储/回写。

{neutral_equivalence_rules}

# Verdict

`verdict` 必须是以下五个之一：

```text
same_behavior
mostly_same_behavior
weakly_same_behavior
different_behavior
insufficient_information
```

请按以下标准选择：

- `same_behavior`：两段描述的核心程序行为一致；允许措辞不同、抽象层级不同、次要细节不同。
- `mostly_same_behavior`：两段描述的核心程序行为一致，但其中一段遗漏重要的补充行为，或抽象层级明显更高。
- `weakly_same_behavior`：两段描述存在明确的局部行为重叠，但重叠不足以确认核心程序行为一致。
- `different_behavior`：两段描述的核心程序行为不同、缺乏明确重叠，或存在明显冲突。
- `insufficient_information`：任一输入为空、过短、只有占位内容，或两段描述都不足以可靠比较。

# 重要约束

1. 只能使用 Input A 和 Input B 的文本内容。
2. 不要使用协议规范、字段名常识、payload 内容、源码知识或你对具体协议的知识。
3. 不要根据输入顺序、措辞风格或其他外部上下文推断字段语义。
4. 不要要求逐字匹配；重点比较程序行为含义。
5. 不要把传统字段类型作为判断依据。
6. 不要输出推理过程，只输出要求的 JSON。
7. 不要输出连续分数或置信分数。
8. 输出必须是严格 JSON，不要输出 markdown 或额外解释。

# 输出 JSON Schema

请输出一个 JSON object：

```json
{
  "verdict": "same_behavior|mostly_same_behavior|weakly_same_behavior|different_behavior|insufficient_information",
  "shared_behaviors": [
    {
      "behavior": "Both inputs describe an early constant comparison followed by conditional branching.",
      "input_a_evidence": "Input A mentions constant comparison and branch control.",
      "input_b_evidence": "Input B mentions comparison-driven path selection."
    }
  ],
  "input_a_only_behaviors": [
    {
      "behavior": "Input A describes later output write-back.",
      "impact": "important|minor"
    }
  ],
  "input_b_only_behaviors": [
    {
      "behavior": "Input B describes loop-scale influence.",
      "impact": "important|minor"
    }
  ],
  "conflicting_behaviors": [
    {
      "input_a_behavior": "Input A describes ordinary storage only.",
      "input_b_behavior": "Input B describes multi-path branch selection."
    }
  ],
  "rationale": "The two inputs share the main comparison and branch behavior, but Input A also mentions a later write-back that Input B does not cover.",
  "needs_manual_review": false,
  "manual_review_reason": ""
}
```

# 输出字段要求

- `verdict` 必须从五个固定值中选择；
- 不要输出任何连续分数、置信分数或数值评分字段；
- `shared_behaviors` 可以为空数组；
- `input_a_only_behaviors` 可以为空数组；
- `input_b_only_behaviors` 可以为空数组；
- `conflicting_behaviors` 可以为空数组；
- `impact` 只能是 `important` 或 `minor`；
- `rationale` 应简洁说明为什么给出该 verdict；
- 如果任一输入信息不足、字段对齐可疑或描述无法比较，应设置 `needs_manual_review=true`。
"""


NEUTRAL_EQUIVALENCE_RULES = """5. 以下两类描述可视为程序行为含义相近。
   - “某个 bit 被拆出后测试，控制单个跳转”和“字段取值驱动多类别处理分派”意思相近。
   - “字段控制循环次数、游标步进或后续数据消费范围”和“字段触发边界检查和异常路径爆发式分化”意思相近。"""

NO_NEUTRAL_EQUIVALENCE_RULES = """5. 不额外提供程序行为等价示例。
   - 只根据 Input A 和 Input B 自身文本判断两段描述是否一致、部分一致或不一致。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ2-B pairwise summary judge.")
    parser.add_argument("--program-log-jsonl", type=Path, default=DEFAULT_PROGRAM_LOG_JSONL)
    parser.add_argument("--stage4-profiles", type=Path, default=DEFAULT_STAGE4_PROFILES)
    parser.add_argument(
        "--sample-map-manifest",
        type=Path,
        default=None,
        help=(
            "Optional held-out split manifest used to map frozen-baseline sample IDs "
            "to their original packet IDs."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument(
        "--backend",
        choices=["api", "codex"],
        default="api",
        help="LLM backend. Default: api (DeepSeek-compatible OpenAI API). Use codex only when Codex CLI is available.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Debug mode: randomly sample this many pending matched fields.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used by --random-sample. Default: 0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the Markdown report from --output-csv without calling the LLM.",
    )
    parser.add_argument("--retry-delay-seconds", type=int, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-cwd", type=Path, default=Path("/root"))
    parser.add_argument("--api-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-reasoning-effort", default="high")
    parser.add_argument("--api-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--api-top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--api-thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-neutral-equivalence-rules",
        action="store_true",
        help=(
            "Disable the optional neutral equivalence examples in the judge prompt. "
            "By default they are included, matching the V4-style judge prompt."
        ),
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")


def print_line(message: str) -> None:
    with PRINT_LOCK:
        print(f"\r{message}")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_sample_id_map(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], str] = {}
    for protocol, protocol_row in manifest.get("protocols", {}).items():
        for packet_row in protocol_row.get("packet_alignment", []):
            sample_id = str(packet_row.get("sample_id") or "")
            packet_id = str(packet_row.get("packet_id") or "")
            if sample_id and packet_id:
                result[(str(protocol), sample_id)] = packet_id
    return result


def load_eval_packet_keys(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result: set[tuple[str, str]] = set()
    for protocol, protocol_row in manifest.get("protocols", {}).items():
        packet_by_sample = {
            str(packet_row.get("sample_id") or ""): str(packet_row.get("packet_id") or "")
            for packet_row in protocol_row.get("packet_alignment", [])
        }
        for sample_id in protocol_row.get("eval", []):
            packet_id = packet_by_sample.get(str(sample_id), "")
            if packet_id:
                result.add((str(protocol), packet_id))
    return result


def normalize_sample_id(
    protocol_name: str,
    sample_id: str,
    sample_id_map: dict[tuple[str, str], str] | None = None,
) -> str:
    sample_id = str(sample_id or "")
    if sample_id_map:
        mapped = sample_id_map.get((str(protocol_name or ""), sample_id))
        if mapped:
            return mapped
    if re.fullmatch(r"pkt_\d+", sample_id):
        return sample_id
    match = re.fullmatch(r"sample_(\d+)", sample_id)
    if match:
        index = max(int(match.group(1)) - 1, 0)
        return f"pkt_{index:04d}"
    return sample_id


def key_of(
    row: dict[str, Any],
    sample_id_map: dict[tuple[str, str], str] | None = None,
) -> tuple[str, str, str]:
    protocol_name = str(row.get("protocol_name") or "")
    return (
        protocol_name,
        normalize_sample_id(protocol_name, str(row.get("sample_id") or ""), sample_id_map),
        str(row.get("field_id") or ""),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: JSON line must be an object")
            rows.append(obj)
    return rows


def load_program_log_rows(
    path: Path,
    sample_id_map: dict[tuple[str, str], str] | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = key_of(row, sample_id_map)
        if all(key):
            rows[key] = row
    return rows


def load_stage4_rows(
    path: Path,
    sample_id_map: dict[tuple[str, str], str] | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = key_of(row, sample_id_map)
        summary = str(row.get("field_program_semantic_summary") or row.get("semantic_summary") or "").strip()
        if all(key) and summary:
            row["semantic_summary"] = summary
            rows[key] = row
    return rows


def sort_key(key: tuple[str, str, str]) -> tuple[str, int, str, str]:
    proto, sample, field_id = key
    match = re.fullmatch(r"pkt_(\d+)", sample)
    sample_index = int(match.group(1)) if match else 10**9
    return proto, sample_index, sample, field_id


def coverage_by_protocol(
    program_keys: set[tuple[str, str, str]],
    stage4_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocols = sorted({key[0] for key in program_keys | stage4_keys})
    for protocol in protocols:
        p_keys = {key for key in program_keys if key[0] == protocol}
        s_keys = {key for key in stage4_keys if key[0] == protocol}
        matched = p_keys & s_keys
        rows.append(
            {
                "protocol_name": protocol,
                "program_log_fields": len(p_keys),
                "stage4_fields": len(s_keys),
                "matched_fields": len(matched),
                "program_log_only_fields": len(p_keys - s_keys),
                "stage4_only_fields": len(s_keys - p_keys),
                "match_rate_vs_program_log": 0.0 if not p_keys else len(matched) / len(p_keys),
                "match_rate_vs_stage4": 0.0 if not s_keys else len(matched) / len(s_keys),
            }
        )
    return rows


def build_prompt(input_a: str, input_b: str, include_neutral_equivalence_rules: bool = True) -> str:
    neutral_rules = (
        NEUTRAL_EQUIVALENCE_RULES
        if include_neutral_equivalence_rules
        else NO_NEUTRAL_EQUIVALENCE_RULES
    )
    return (
        PROMPT_TEMPLATE.replace("{input_a_description}", input_a.strip())
        .replace("{input_b_description}", input_b.strip())
        .replace("{neutral_equivalence_rules}", neutral_rules)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        starts = [match.start() for match in re.finditer(r"\{", stripped)]
        for start in starts:
            candidate = stripped[start:]
            end = candidate.rfind("}")
            if end == -1:
                continue
            try:
                parsed = json.loads(candidate[: end + 1])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def validate_response(obj: dict[str, Any]) -> dict[str, Any]:
    verdict = obj.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r}")
    result = dict(obj)
    for key in (
        "shared_behaviors",
        "input_a_only_behaviors",
        "input_b_only_behaviors",
        "conflicting_behaviors",
    ):
        if not isinstance(result.get(key), list):
            result[key] = []
    if not isinstance(result.get("rationale"), str):
        result["rationale"] = ""
    if not isinstance(result.get("needs_manual_review"), bool):
        result["needs_manual_review"] = False
    if not isinstance(result.get("manual_review_reason"), str):
        result["manual_review_reason"] = ""
    return result


def looks_retryable(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "429",
            "too many requests",
            "rate limit",
            "exceeded retry limit",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
        )
    )


def wait_with_countdown(seconds: int, run_log: Path, reason: str) -> None:
    log_line(run_log, f"waiting {seconds}s before retry: {reason}")
    time.sleep(seconds)


def run_codex(prompt: str, args: argparse.Namespace) -> str:
    with tempfile.NamedTemporaryFile("r", encoding="utf-8") as output_file, tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8"
    ) as stdout_file, tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stderr_file:
        command = [
            args.codex_command,
            "exec",
            "--skip-git-repo-check",
            "-C",
            str(args.codex_cwd),
            "-o",
            output_file.name,
        ]
        if args.codex_model:
            command.extend(["-m", args.codex_model])
        command.append("-")
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        assert process.stdin is not None
        process.stdin.write(prompt)
        process.stdin.close()
        while process.poll() is None:
            time.sleep(1)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        if process.returncode != 0:
            raise RuntimeError(f"returncode={process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        output_file.seek(0)
        return output_file.read()


def get_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing API key: fill DEEPSEEK_API_KEY, set env DEEPSEEK_API_KEY, or pass --api-key")
    return api_key


def run_api(prompt: str, args: argparse.Namespace) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("missing Python package: install openai to use --backend api") from exc
    client = OpenAI(api_key=get_api_key(args), base_url=args.api_base_url, timeout=args.api_timeout)
    response = client.chat.completions.create(
        model=args.api_model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=args.api_reasoning_effort,
        temperature=args.api_temperature,
        top_p=args.api_top_p,
        extra_body={"thinking": {"type": args.api_thinking}},
    )
    return response.choices[0].message.content or ""


def call_llm(prompt: str, args: argparse.Namespace) -> str:
    if args.backend == "api":
        return run_api(prompt, args)
    return run_codex(prompt, args)


def result_row(
    key: tuple[str, str, str],
    input_a: str,
    input_b: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    protocol, sample, field_id = key
    return {
        "protocol_name": protocol,
        "sample_id": sample,
        "field_id": field_id,
        "input_a": input_a,
        "input_b": input_b,
        "verdict": response["verdict"],
        "shared_behaviors": response["shared_behaviors"],
        "input_a_only_behaviors": response["input_a_only_behaviors"],
        "input_b_only_behaviors": response["input_b_only_behaviors"],
        "conflicting_behaviors": response["conflicting_behaviors"],
        "rationale": response["rationale"],
        "needs_manual_review": response["needs_manual_review"],
        "manual_review_reason": response["manual_review_reason"],
    }


def load_existing_results(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            verdict = row.get("verdict", "")
            if verdict not in VALID_VERDICTS:
                raise ValueError(
                    f"{path} contains legacy or invalid verdict {verdict!r}; "
                    "rerun with --overwrite or use a new output path"
                )
            key = (
                row.get("protocol_name", ""),
                row.get("sample_id", ""),
                row.get("field_id", ""),
            )
            if all(key):
                existing[key] = row
    return existing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol_name",
        "sample_id",
        "field_id",
        "input_a",
        "input_b",
        "verdict",
        "shared_behaviors",
        "input_a_only_behaviors",
        "input_b_only_behaviors",
        "conflicting_behaviors",
        "rationale",
        "needs_manual_review",
        "manual_review_reason",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: sort_key((item["protocol_name"], item["sample_id"], item["field_id"]))):
            csv_row = dict(row)
            for key in (
                "shared_behaviors",
                "input_a_only_behaviors",
                "input_b_only_behaviors",
                "conflicting_behaviors",
            ):
                if not isinstance(csv_row.get(key), str):
                    csv_row[key] = json_dumps(csv_row[key])
            writer.writerow(csv_row)


def verdict_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {verdict: 0 for verdict in sorted(VALID_VERDICTS)}
    for row in rows:
        verdict = row.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    return counts


def verdict_counts_by_protocol(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("protocol_name") or ""), []).append(row)
    out: list[dict[str, Any]] = []
    for protocol in sorted(grouped):
        protocol_rows = grouped[protocol]
        counts = verdict_counts(protocol_rows)
        matched = len(protocol_rows)
        out.append(
            {
                "protocol_name": protocol,
                "matched_fields": matched,
                "same_behavior": counts.get("same_behavior", 0),
                "mostly_same_behavior": counts.get("mostly_same_behavior", 0),
                "weakly_same_behavior": counts.get("weakly_same_behavior", 0),
                "different_behavior": counts.get("different_behavior", 0),
                "insufficient_information": counts.get("insufficient_information", 0),
                "same_behavior_rate": 0.0 if matched == 0 else counts.get("same_behavior", 0) / matched,
                "strong_agreement_rate": 0.0
                if matched == 0
                else (counts.get("same_behavior", 0) + counts.get("mostly_same_behavior", 0)) / matched,
                "any_overlap_rate": 0.0
                if matched == 0
                else (
                    counts.get("same_behavior", 0)
                    + counts.get("mostly_same_behavior", 0)
                    + counts.get("weakly_same_behavior", 0)
                )
                / matched,
            }
        )
    return out


def write_markdown(
    path: Path,
    coverage_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    program_only: list[tuple[str, str, str]],
    stage4_only: list[tuple[str, str, str]],
    run_started_at: str,
    run_elapsed: str,
    completed_this_run: int,
    existing_results: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = verdict_counts(result_rows)
    protocol_verdict_rows = verdict_counts_by_protocol(result_rows)
    matched = len(result_rows)
    lines = [
        "# RQ2-B Pairwise Judge Results",
        "",
        f"generated-at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Run Summary",
        "",
        f"- run_started_at: {run_started_at}",
        f"- total_elapsed: {run_elapsed}",
        f"- completed_this_run: {completed_this_run}",
        f"- existing_results_reused: {existing_results}",
        f"- total_results_written: {len(result_rows)}",
        "",
        "## Coverage Summary",
        "",
        "| Protocol | Program-log fields | Stage4 fields | Matched | Program-log only | Stage4 only | Match vs program-log | Match vs Stage4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in coverage_rows:
        lines.append(
            "| {protocol_name} | {program_log_fields} | {stage4_fields} | {matched_fields} | {program_log_only_fields} | {stage4_only_fields} | {match_rate_vs_program_log:.4f} | {match_rate_vs_stage4:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Verdict Summary",
            "",
            f"- matched_fields: {matched}",
            f"- same_behavior: {counts.get('same_behavior', 0)}",
            f"- mostly_same_behavior: {counts.get('mostly_same_behavior', 0)}",
            f"- weakly_same_behavior: {counts.get('weakly_same_behavior', 0)}",
            f"- different_behavior: {counts.get('different_behavior', 0)}",
            f"- insufficient_information: {counts.get('insufficient_information', 0)}",
            f"- same_behavior_rate: {0.0 if matched == 0 else counts.get('same_behavior', 0) / matched:.4f}",
            f"- strong_agreement_rate: {0.0 if matched == 0 else (counts.get('same_behavior', 0) + counts.get('mostly_same_behavior', 0)) / matched:.4f}",
            f"- any_overlap_rate: {0.0 if matched == 0 else (counts.get('same_behavior', 0) + counts.get('mostly_same_behavior', 0) + counts.get('weakly_same_behavior', 0)) / matched:.4f}",
            "",
            "## Verdict By Protocol",
            "",
            "| Protocol | Matched | Same | Mostly same | Weakly same | Different | Insufficient | Same rate | Strong agreement | Any overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in protocol_verdict_rows:
        lines.append(
            "| {protocol_name} | {matched_fields} | {same_behavior} | {mostly_same_behavior} | {weakly_same_behavior} | {different_behavior} | {insufficient_information} | {same_behavior_rate:.4f} | {strong_agreement_rate:.4f} | {any_overlap_rate:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Unmatched Fields",
            "",
            f"- program_log_only_fields: {len(program_only)}",
            f"- stage4_only_fields: {len(stage4_only)}",
            "",
            "program_log_only preview:",
            "",
            "```text",
        ]
    )
    lines.extend(" / ".join(key) for key in program_only[:100])
    if len(program_only) > 100:
        lines.append(f"... ({len(program_only) - 100} more)")
    lines.extend(["```", "", "stage4_only preview:", "", "```text"])
    lines.extend(" / ".join(key) for key in stage4_only[:100])
    if len(stage4_only) > 100:
        lines.append(f"... ({len(stage4_only) - 100} more)")
    lines.extend(["```", "", "## Pairwise Results", ""])
    for row in sorted(result_rows, key=lambda item: sort_key((item["protocol_name"], item["sample_id"], item["field_id"]))):
        lines.extend(
            [
                f"### {row['protocol_name']} / {row['sample_id']} / {row['field_id']}",
                "",
                "Input A:",
                "",
                "```text",
                str(row["input_a"]),
                "```",
                "",
                "Input B:",
                "",
                "```text",
                str(row["input_b"]),
                "```",
                "",
                f"verdict: `{row['verdict']}`",
                "",
                f"rationale: {row['rationale']}",
                "",
                f"shared_behaviors: `{json_dumps(row['shared_behaviors'])}`",
                "",
                f"input_a_only_behaviors: `{json_dumps(row['input_a_only_behaviors'])}`",
                "",
                f"input_b_only_behaviors: `{json_dumps(row['input_b_only_behaviors'])}`",
                "",
                f"conflicting_behaviors: `{json_dumps(row['conflicting_behaviors'])}`",
                "",
                f"needs_manual_review: `{row['needs_manual_review']}`",
                "",
                f"manual_review_reason: {row['manual_review_reason']}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_one(
    key: tuple[str, str, str],
    program_row: dict[str, Any],
    stage4_row: dict[str, Any],
    args: argparse.Namespace,
    index: int,
    total: int,
    run_start_time: float,
) -> dict[str, Any]:
    input_a = str(program_row.get("program_log_description") or "").strip()
    input_b = str(stage4_row.get("semantic_summary") or "").strip()
    prompt = build_prompt(
        input_a,
        input_b,
        include_neutral_equivalence_rules=not args.no_neutral_equivalence_rules,
    )
    attempt = 0
    while True:
        attempt += 1
        start = time.time()
        total_elapsed = format_duration(time.time() - run_start_time)
        print_line(
            f"[judge] start {index}/{total} {'/'.join(key)} "
            f"attempt={attempt} total_elapsed={total_elapsed}"
        )
        try:
            response_text = call_llm(prompt, args)
            parsed = validate_response(extract_json_object(response_text))
            elapsed = format_duration(time.time() - start)
            total_elapsed = format_duration(time.time() - run_start_time)
            print_line(
                f"[judge] done {index}/{total} {'/'.join(key)} "
                f"verdict={parsed['verdict']} elapsed={elapsed} total_elapsed={total_elapsed}"
            )
            log_line(
                args.run_log,
                f"done {'/'.join(key)} verdict={parsed['verdict']} "
                f"elapsed={elapsed} total_elapsed={total_elapsed}",
            )
            return result_row(key, input_a, input_b, parsed)
        except Exception as exc:
            error_text = str(exc)
            retry_reason = "retryable error" if looks_retryable(error_text) else "error"
            total_elapsed = format_duration(time.time() - run_start_time)
            print_line(
                f"[judge] error {'/'.join(key)} attempt={attempt} "
                f"total_elapsed={total_elapsed}: {error_text[:1000]}"
            )
            log_line(
                args.run_log,
                f"error {'/'.join(key)} attempt={attempt} "
                f"total_elapsed={total_elapsed}: {error_text[:1000]}",
            )
            wait_with_countdown(args.retry_delay_seconds, args.run_log, retry_reason)


def main() -> None:
    script_start_time = time.time()
    run_started_at = datetime.now().isoformat(timespec="seconds")
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.run_log.parent.mkdir(parents=True, exist_ok=True)

    sample_id_map = load_sample_id_map(args.sample_map_manifest)
    eval_packet_keys = load_eval_packet_keys(args.sample_map_manifest)
    program_rows = load_program_log_rows(args.program_log_jsonl, sample_id_map)
    stage4_rows = load_stage4_rows(args.stage4_profiles, sample_id_map)
    if eval_packet_keys:
        program_rows = {
            key: row
            for key, row in program_rows.items()
            if key[:2] in eval_packet_keys
        }
    program_keys = set(program_rows)
    stage4_keys = set(stage4_rows)
    matched_keys = sorted(program_keys & stage4_keys, key=sort_key)
    program_only = sorted(program_keys - stage4_keys, key=sort_key)
    stage4_only = sorted(stage4_keys - program_keys, key=sort_key)
    coverage_rows = coverage_by_protocol(program_keys, stage4_keys)

    if args.report_only:
        if not args.output_csv.exists():
            raise SystemExit(f"--report-only requires existing --output-csv: {args.output_csv}")
        existing = load_existing_results(args.output_csv)
        result_rows = list(existing.values())
        elapsed = format_duration(time.time() - script_start_time)
        write_markdown(
            args.output_md,
            coverage_rows,
            result_rows,
            program_only,
            stage4_only,
            run_started_at,
            elapsed,
            0,
            len(existing),
        )
        print_line(f"[judge] report-only results={len(result_rows)} md={args.output_md}")
        return

    existing = {} if args.overwrite else load_existing_results(args.output_csv)
    rows: list[dict[str, Any]] = list(existing.values())
    pending_keys = [key for key in matched_keys if key not in existing]
    if args.random_sample is not None:
        if args.random_sample < 1:
            raise SystemExit("--random-sample must be >= 1")
        rng = random.Random(args.random_seed)
        sample_size = min(args.random_sample, len(pending_keys))
        pending_keys = sorted(rng.sample(pending_keys, sample_size), key=sort_key)
    elif args.limit is not None:
        pending_keys = pending_keys[: args.limit]
    total = len(pending_keys)
    log_line(
        args.run_log,
        f"start backend={args.backend} matched={len(matched_keys)} pending={total} existing={len(existing)}",
    )
    print_line(
        f"[judge] loaded program_log={len(program_rows)} stage4={len(stage4_rows)} matched={len(matched_keys)} "
        f"pending={total} random_sample={args.random_sample} random_seed={args.random_seed}"
    )

    start_time = script_start_time
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, max(total, 1))) as executor:
        futures = [
            executor.submit(
                process_one,
                key,
                program_rows[key],
                stage4_rows[key],
                args,
                index,
                total,
                start_time,
            )
            for index, key in enumerate(pending_keys, start=1)
        ]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            with WRITE_LOCK:
                merged_rows = rows + completed
                write_csv(args.output_csv, merged_rows)
                write_markdown(
                    args.output_md,
                    coverage_rows,
                    merged_rows,
                    program_only,
                    stage4_only,
                    run_started_at,
                    format_duration(time.time() - script_start_time),
                    len(completed),
                    len(existing),
                )

    final_rows = rows + completed
    write_csv(args.output_csv, final_rows)
    elapsed = format_duration(time.time() - script_start_time)
    write_markdown(
        args.output_md,
        coverage_rows,
        final_rows,
        program_only,
        stage4_only,
        run_started_at,
        elapsed,
        len(completed),
        len(existing),
    )
    log_line(args.run_log, f"finished completed={len(completed)} total_results={len(final_rows)} elapsed={elapsed}")
    print_line(f"[judge] finished completed={len(completed)} total_results={len(final_rows)} elapsed={elapsed}")
    print_line(f"[judge] csv: {args.output_csv}")
    print_line(f"[judge] md: {args.output_md}")


if __name__ == "__main__":
    main()
