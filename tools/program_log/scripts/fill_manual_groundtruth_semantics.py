#!/usr/bin/env python3
"""Fill program semantics for manually corrected program-log groundtruth fields.

This script keeps field boundaries fixed. It only updates semantic fields for
entries that were manually added/corrected during boundary cleanup.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_LLM_OUTPUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/llm_output"
)
DEFAULT_LOG_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/preprocessed_logs"
)
DEFAULT_RUN_LOG = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/fill_manual_semantics.log"
)
DEFAULT_BACKUP_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/groundtruth_result/manual_semantics_backups"
)
DEFAULT_RETRY_DELAY_SECONDS = 30 * 60
DEFAULT_WORKERS = 5
DEFAULT_BASE_URL = None
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

# Manual program-log semantic completion follows the PG-generation backend.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

LOG_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
BACKUP_LOCK = threading.Lock()

MANUAL_MARKERS = (
    "人工校验补齐",
    "人工校验修正",
    "人工校验保留",
    "manual boundary correction",
)

SEMANTIC_KEYS = (
    "program_log_description",
    "observed_behaviors",
    "field_partition_evidence",
    "needs_review",
    "review_reason",
)


PROMPT_TEMPLATE = r"""你是一个程序执行日志分析助手。

# 任务

你将看到一个协议 server 处理单个输入报文时产生的 taint execution log，以及一组已经人工确认的字段边界。

请严格遵守：

1. 不要重新划分字段。
2. 不要新增、删除、合并、拆分、重命名任何字段。
3. 只为输入中列出的 `field_id` 补全程序行为语义。
4. 输出中必须且只能包含输入给你的这些 `field_id`。
5. 字段边界已经人工确认，即使你认为边界可疑，也不能修改 field_id。
6. 语义描述必须只基于 log 中可观察到的 tainted input 使用方式。
7. 不要根据协议规范、协议名、函数名中的协议术语或自身知识库推断字段角色。
8. 如果 log 中找不到足够证据，请仍然输出该 field_id，并设置 `needs_review=true`，在 `review_reason` 中说明证据不足。

# 输入字段

```json
{target_fields_json}
```

# program_execution_log

```text
{program_execution_log}
```

# 字段 ID 格式

- byte 字段：`b:start:end`
- bit 字段：`bit:start:end:low:high`

对于 bit 字段，请描述程序如何单独使用这些 bit，例如 mask、shift、test、cmp、条件分支、传播或存储。

# 输出格式

只输出一个 JSON object，不要输出 markdown，不要输出解释文字。

```json
{
  "fields": [
    {
      "field_id": "bit:42:42:0:2",
      "program_log_description": "一到两句话，描述该字段在程序中如何被消费。",
      "observed_behaviors": [
        {
          "behavior": "短程序行为短语",
          "evidence": [
            {
              "line_no": 123,
              "function": "function_name_or_empty",
              "instruction": "assembly_or_summary",
              "field_refs": "42"
            }
          ]
        }
      ],
      "field_partition_evidence": [
        {
          "line_no": 123,
          "function": "function_name_or_empty",
          "instruction": "assembly_or_summary",
          "field_refs": "42",
          "reason": "为什么该 evidence 支持这个字段的程序行为语义"
        }
      ],
      "needs_review": false,
      "review_reason": ""
    }
  ]
}
```

# 描述风格约束

- `program_log_description` 应描述程序行为，不要写传统字段类型名、协议角色名或规范术语。
- 好的描述例子：`该 bit 段被掩码提取后参与条件测试，并决定是否进入另一条处理路径。`
- 好的描述例子：`该 byte 范围在循环中被逐字节复制到内部缓冲，后续没有观察到数值比较或分支消费。`
- 不好的描述例子：`这是 flags 字段。`
- 不好的描述例子：`这是协议中的对象 ID。`
- `observed_behaviors[].behavior` 应是短程序行为短语，例如 `bit 子段提取后参与测试`、`逐字节复制到内部缓冲`。
- 每个 evidence object 中无法确定的字段可以填空字符串，但不要编造不存在的行号或指令。
- 如果 evidence 来自 `RepeatSummary`，可以把摘要中的 `instruction`、`field_refs` 和 `reason` 作为证据，并在 reason 中说明这是压缩摘要证据。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill semantics for manually corrected program-log groundtruth fields."
    )
    parser.add_argument("--llm-output-dir", type=Path, default=DEFAULT_LLM_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--backend", choices=["api", "codex"], default="api")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--start-seq", type=int, default=1)
    parser.add_argument("--only-seq", type=int, default=None)
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Debug mode: randomly sample N manual fields across all selected files.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for CLI symmetry; target selection is still limited to manual-placeholder fields.",
    )
    parser.add_argument("--retry-delay-seconds", type=int, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-cwd", type=Path, default=Path("/root"))
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_BASE_URL,
        help="Optional OpenAI-compatible API base URL. Default: use the OpenAI SDK default.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-reasoning-effort", default="high")
    parser.add_argument("--api-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--api-top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--api-timeout", type=float, default=600.0)
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def seq_of(path: Path) -> int:
    match = re.match(r"^(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")


def print_line(message: str) -> None:
    with PRINT_LOCK:
        print(f"\r{message}")


def print_status(message: str) -> None:
    with PRINT_LOCK:
        print(f"\r{message}", end="", flush=True)


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


def wait_with_countdown(seconds: int, stop_event: threading.Event | None = None) -> None:
    start = time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if time.time() - start >= seconds:
            return
        time.sleep(1)


def field_text(field: dict[str, Any]) -> str:
    parts = [
        str(field.get("program_log_description", "")),
        str(field.get("review_reason", "")),
    ]
    for evidence in field.get("field_partition_evidence") or []:
        if isinstance(evidence, dict):
            parts.extend(str(evidence.get(key, "")) for key in ("reason", "function", "instruction"))
    for behavior in field.get("observed_behaviors") or []:
        if isinstance(behavior, dict):
            parts.append(str(behavior.get("behavior", "")))
    return "\n".join(parts)


def is_manual_placeholder(field: dict[str, Any]) -> bool:
    if field.get("review_reason") == "manual boundary correction":
        return True
    text = field_text(field)
    return any(marker in text for marker in MANUAL_MARKERS)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        starts = [m.start() for m in re.finditer(r"\{", stripped)]
        for start in starts:
            candidate = stripped[start:]
            end = candidate.rfind("}")
            if end < 0:
                continue
            try:
                parsed = json.loads(candidate[: end + 1])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON must be an object")
    return parsed


def build_prompt(log_text: str, fields: list[dict[str, Any]]) -> str:
    target_fields = [
        {
            "field_id": field.get("field_id", ""),
            "current_program_log_description": field.get("program_log_description", ""),
            "current_review_reason": field.get("review_reason", ""),
        }
        for field in fields
    ]
    return (
        PROMPT_TEMPLATE.replace(
            "{target_fields_json}",
            json.dumps(target_fields, ensure_ascii=False, indent=2),
        )
        .replace("{program_execution_log}", log_text)
    )


def get_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing API key: set env OPENAI_API_KEY or pass --api-key")
    return api_key


def run_api(prompt: str, args: argparse.Namespace) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("missing Python package: install openai to use --backend api") from exc

    client_kwargs = {"api_key": get_api_key(args), "timeout": args.api_timeout}
    if args.api_base_url:
        client_kwargs["base_url"] = args.api_base_url
    client = OpenAI(**client_kwargs)
    request_kwargs = {
        "model": args.api_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.api_temperature,
        "top_p": args.api_top_p,
    }
    if args.api_reasoning_effort:
        request_kwargs["reasoning_effort"] = args.api_reasoning_effort
    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or ""


def run_codex(prompt: str, args: argparse.Namespace) -> str:
    command = [
        args.codex_command,
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(args.codex_cwd),
        "-o",
    ]
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".json") as output_file:
        command.append(output_file.name)
        if args.codex_model:
            command.extend(["-m", args.codex_model])
        command.append("-")
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stdout_file, tempfile.NamedTemporaryFile(
            "w+", encoding="utf-8"
        ) as stderr_file:
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


def run_llm(prompt: str, args: argparse.Namespace) -> str:
    if args.backend == "api":
        return run_api(prompt, args)
    return run_codex(prompt, args)


def log_path_for(json_path: Path, args: argparse.Namespace) -> Path:
    return args.log_dir / f"{json_path.stem}.log"


def selected_json_files(args: argparse.Namespace) -> list[Path]:
    files = sorted(args.llm_output_dir.glob("*.json"), key=seq_of)
    if args.only_seq is not None:
        files = [path for path in files if seq_of(path) == args.only_seq]
    else:
        files = [path for path in files if seq_of(path) >= args.start_seq]
    return files


def collect_targets(args: argparse.Namespace) -> dict[Path, list[str]]:
    targets: list[tuple[Path, str]] = []
    for path in selected_json_files(args):
        data = json.loads(path.read_text(encoding="utf-8"))
        for field in data.get("fields", []):
            if not isinstance(field, dict):
                continue
            field_id = str(field.get("field_id", "")).strip()
            if not field_id:
                continue
            if is_manual_placeholder(field):
                targets.append((path, field_id))

    if args.random_sample is not None:
        rng = random.Random(args.random_seed)
        if args.random_sample < len(targets):
            targets = rng.sample(targets, args.random_sample)

    grouped: dict[Path, list[str]] = defaultdict(list)
    for path, field_id in targets:
        grouped[path].append(field_id)
    return dict(sorted(grouped.items(), key=lambda item: seq_of(item[0])))


def ensure_backup(path: Path, args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = args.backup_dir / path.name
    with BACKUP_LOCK:
        if not backup_path.exists():
            shutil.copy2(path, backup_path)


def merge_semantics(
    json_path: Path,
    target_ids: list[str],
    response_text: str,
    args: argparse.Namespace,
) -> int:
    response = extract_json_object(response_text)
    response_fields = response.get("fields")
    if not isinstance(response_fields, list):
        raise ValueError("LLM response missing fields list")

    target_set = set(target_ids)
    updates: dict[str, dict[str, Any]] = {}
    for field in response_fields:
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id", "")).strip()
        if field_id not in target_set:
            raise ValueError(f"LLM returned unexpected field_id {field_id!r}")
        updates[field_id] = field

    missing = sorted(target_set - set(updates))
    if missing:
        raise ValueError(f"LLM response missing target field_ids: {missing}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = 0
    for field in data.get("fields", []):
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id", "")).strip()
        if field_id not in updates:
            continue
        update = updates[field_id]
        for key in SEMANTIC_KEYS:
            if key in update:
                field[key] = update[key]
        changed += 1

    if changed != len(target_set):
        raise ValueError(f"updated {changed} fields but expected {len(target_set)}")

    if not args.dry_run:
        ensure_backup(json_path, args)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def process_file(
    json_path: Path,
    target_ids: list[str],
    args: argparse.Namespace,
    counters: dict[str, int],
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    seq = seq_of(json_path)
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        start = time.time()
        log_path = log_path_for(json_path, args)
        try:
            if not log_path.exists():
                raise FileNotFoundError(f"missing preprocessed log: {log_path}")
            data = json.loads(json_path.read_text(encoding="utf-8"))
            fields_by_id = {
                str(field.get("field_id", "")).strip(): field
                for field in data.get("fields", [])
                if isinstance(field, dict)
            }
            target_fields = [fields_by_id[field_id] for field_id in target_ids if field_id in fields_by_id]
            if len(target_fields) != len(target_ids):
                missing = sorted(set(target_ids) - set(fields_by_id))
                raise ValueError(f"target fields disappeared from {json_path.name}: {missing}")
            prompt = build_prompt(log_path.read_text(encoding="utf-8", errors="replace"), target_fields)
            print_line(
                f"[fill] start seq={seq} file={json_path.name} fields={len(target_ids)} attempt={attempt}"
            )
            response_text = run_llm(prompt, args)
            changed = merge_semantics(json_path, target_ids, response_text, args)
            elapsed = format_duration(time.time() - start)
            with state_lock:
                counters["done_files"] += 1
                counters["done_fields"] += changed
            msg = f"done seq={seq} file={json_path.name} fields={changed} elapsed={elapsed}"
            print_line(f"[fill] {msg}")
            log_line(args.run_log, msg)
            return
        except Exception as exc:
            if stop_event.is_set():
                return
            error_text = str(exc)
            with state_lock:
                counters["errors"] += 1
            msg = f"error seq={seq} file={json_path.name} attempt={attempt}: {error_text[:1000]}"
            print_line(f"[fill] {msg}")
            log_line(args.run_log, msg)
            reason = "retryable error" if looks_retryable(error_text) else "error"
            log_line(args.run_log, f"waiting {args.retry_delay_seconds}s after {reason}")
            wait_with_countdown(args.retry_delay_seconds, stop_event)


def status_loop(
    total_files: int,
    total_fields: int,
    counters: dict[str, int],
    start_time: float,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        print_status(
            "[status] "
            f"files={counters['done_files']}/{total_files} "
            f"fields={counters['done_fields']}/{total_fields} "
            f"errors={counters['errors']} elapsed={format_duration(time.time() - start_time)}"
        )
        time.sleep(1)
    print()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    args.run_log.parent.mkdir(parents=True, exist_ok=True)

    grouped = collect_targets(args)
    total_files = len(grouped)
    total_fields = sum(len(fields) for fields in grouped.values())
    print_line(
        f"[fill] selected files={total_files} manual_fields={total_fields} "
        f"backend={args.backend} dry_run={args.dry_run}"
    )
    log_line(
        args.run_log,
        f"start backend={args.backend} files={total_files} fields={total_fields} "
        f"only_seq={args.only_seq} start_seq={args.start_seq} random_sample={args.random_sample}",
    )
    if args.dry_run or total_fields == 0:
        for path, field_ids in grouped.items():
            print_line(f"[fill] target {path.name}: {len(field_ids)} fields")
        return

    stop_event = threading.Event()
    state_lock = threading.Lock()
    counters = {"done_files": 0, "done_fields": 0, "errors": 0}
    start_time = time.time()

    def handle_sigint(signum: int, frame: Any) -> None:
        stop_event.set()
        log_line(args.run_log, "interrupted by Ctrl+C")
        print_line("[fill] interrupted by Ctrl+C; waiting for active calls to settle")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    status_thread = threading.Thread(
        target=status_loop,
        args=(total_files, total_fields, counters, start_time, stop_event),
        daemon=True,
    )
    status_thread.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, max(total_files, 1))) as executor:
            futures = [
                executor.submit(process_file, path, field_ids, args, counters, state_lock, stop_event)
                for path, field_ids in grouped.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                if stop_event.is_set():
                    break
                future.result()
    except KeyboardInterrupt:
        stop_event.set()
        log_line(args.run_log, f"stopped elapsed={format_duration(time.time() - start_time)}")
        return
    finally:
        stop_event.set()
        status_thread.join(timeout=2)

    log_line(
        args.run_log,
        f"finished files={counters['done_files']}/{total_files} "
        f"fields={counters['done_fields']}/{total_fields} "
        f"errors={counters['errors']} elapsed={format_duration(time.time() - start_time)}",
    )
    print_line(
        f"[fill] finished files={counters['done_files']}/{total_files} "
        f"fields={counters['done_fields']}/{total_fields} "
        f"errors={counters['errors']} elapsed={format_duration(time.time() - start_time)}"
    )


if __name__ == "__main__":
    main()
