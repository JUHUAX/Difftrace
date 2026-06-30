#!/usr/bin/env python3
"""Run Codex/API groundtruth generation over preprocessed RQ2-B logs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/preprocessed_logs"
)
DEFAULT_OUTPUT_DIR = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/llm_outputs"
)
DEFAULT_RUN_LOG = Path(
    "/root/semvec/bitfield_groundtruth/evaluation_from_program_log/llm_outputs/run.log"
)
DEFAULT_RETRY_DELAY_SECONDS = 30 * 60
DEFAULT_WORKERS = 5
DEFAULT_BASE_URL = None
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

# Program-log ground-truth generation uses GPT by default.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

LOG_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
MERGE_LOCK = threading.Lock()

PROMPT_TEMPLATE = r"""你是一个程序执行日志分析助手。

# 任务

你将看到一个协议 server 处理单个输入报文时产生的 taint execution log。

请你只根据该 log 中可观察到的 tainted input 使用方式，完成两个任务：

1. 恢复程序执行视角下的字段划分；
2. 为每个字段生成程序语义描述。

这里的字段划分和语义都必须来自程序实际如何消费 tainted input，而不是协议规范、tshark 字段名、函数名暗示或常识。

# 输入

program_execution_log:

```text
{program_execution_log}
```

# Log Format

日志行通常具有如下格式：

```text
THREADID<TAB><thread_id><TAB><event_type><TAB><event_payload...>
```

常见 event_type 包括：

## Taint

`Taint` 表示输入报文 taint 被创建。

示例：

```text
THREADID	0	Taint	memory+0x55ee2916e0e0	12
```

`Taint` 行中的最后一个数字表示当前 taint input 的字节数。例如上例中的 `12` 表示当前输入包含 12 个被 taint 标记的字节，后续 Instruction 行中的 byte index 应理解为这些输入字节的 0 基下标。

## Function

`Function` 表示函数进入或退出。

示例：

```text
THREADID	0	Function	enter	bvlc_decode_header	bacnet_server+0x4e820
THREADID	0	Function	exit	bvlc_decode_header
```

函数名可用于判断字段相关指令位于 parser、decoder、handler、check、copy 或 storage 逻辑中。

## BasicBlock

`BasicBlock` 表示基本块执行。BasicBlock 可辅助观察控制流路径，但单独 BasicBlock 不足以说明字段语义。应优先使用 Instruction 证据。

## Instruction

`Instruction` 表示某条指令使用了 tainted input。

一般格式：

```text
THREADID	0	Instruction	<module+addr>: <assembly>	<field_byte_indices>	<SRC/DST/value info> [LOOP]
```

其中 `<field_byte_indices>` 表示该指令关联的数据包字节下标：

- `0` 表示使用了报文字节 0；
- `2,3` 表示 taint/dataflow 记录中报文字节 2 和 3 的数据流在当前位置被合并使用；这只是程序行为证据之一，不能单独决定字段边界；
- `2;3` 表示当前指令中同时使用了报文字节 2 和 3，但不表示程序把它们合并成同一个数值字段；它们可能是两个字段在同一条指令或同一段逻辑中被使用；
- `-` 表示该指令本身不直接对应某个输入字节，常见于 tainted branch 指令；
- 行尾 `LOOP` 表示该指令处于循环相关执行中。

## RepeatSummary

`RepeatSummary` 是预处理器插入的压缩摘要行，用于表示一段高度重复的执行片段。

它不是原始程序指令，也不是新的 taint source。它表示原始 log 中有大量相似的 Function、BasicBlock 或 Instruction 事件被压缩。

一般格式：

```text
THREADID	0	RepeatSummary	kind=<instruction|function|basicblock>	function=<function_name>	action=<enter|exit_if_function>	instruction=<assembly_or_basicblock_if_any>	field_refs=<byte_indices_if_any>	repeated=<count>	loop=<true|false>	values=<representative_values_if_any>	reason=<compression_reason>
```

示例：

```text
THREADID	0	RepeatSummary	kind=instruction	function=days_since_epoch	instruction="cmp r9w, ax"	field_refs=9	repeated=17133	loop=true	values="DST*=0x7ea; SRC range 0x76d..0x7ea"	reason="date conversion loop repeatedly compares tainted year against loop counter"
THREADID	0	RepeatSummary	kind=basicblock	function=days_since_epoch	instruction="bacnet_server+0xacd93"	repeated=11526	reason="same basic block appears repeatedly in a loop or recurring execution path"
THREADID	0	RepeatSummary	kind=function	function=StringUtils_compareChars	action=enter	repeated=3086	reason="string comparison helper repeatedly called during name-list sorting"
THREADID	0	RepeatSummary	kind=function	function=CheckEncapsulationInactivity	action=enter	repeated=1032	field_refs=48,49	reason="periodic socket/session inactivity check repeatedly consumes a tainted timeout/state value"
```

解释规则：

- `repeated` 表示被压缩的相似事件数量；
- `action` 如果存在，表示被压缩的是函数进入还是函数退出事件；
- `field_refs` 如果存在，表示这些重复事件主要关联的输入字节；
- `loop=true` 表示该重复片段来自循环相关执行；
- `values` 如果存在，给出代表性 SRC/DST 值或值范围；
- `reason` 是预处理器对压缩原因的简短说明。

你可以把 `RepeatSummary` 作为 evidence，用来说明某字段被循环比较、重复消费、后期状态维护、字符串/列表遍历、批量编码/解码或周期任务反复使用。

但请注意：

- 不要把 `RepeatSummary` 当作普通 Instruction；
- 不要仅凭 `RepeatSummary` 创建新的字段边界；
- 不要因为 `repeated` 很大就夸大字段重要性；
- 对 `kind=function` 且没有 `field_refs` 的摘要，只能把它理解为调用路径、周期逻辑或后期维护逻辑被反复执行；除非附近有原始 Instruction 或带 `field_refs` 的摘要支持，否则不要把它作为字段语义的核心证据；
- 若原始 Instruction 证据与 `RepeatSummary` 都存在，应优先用原始 Instruction 作为字段边界证据，用 `RepeatSummary` 补充说明重复使用模式。

# 字段划分规则

请根据 tainted byte indices 的实际共同使用方式恢复字段。

你可以输出两类字段：

1. byte 字段
   - 格式为 `b:start:end`
   - 例如 `b:2:3`
   - 单字节字段也必须写完整的 start 和 end。
   - 例如第 5 个字节必须写成 `b:5:5`，不能写成 `b:5`。
   - 例如第 40 个字节必须写成 `b:40:40`，不能写成 `b:40`。

2. bit 字段
   - 格式为 `bit:start:end:low:high`
   - 例如 `bit:5:5:0:1`
   - 单 bit 字段也必须写完整的 low 和 high。
   - 例如 byte 5 的 bit 7 必须写成 `bit:5:5:7:7`，不能写成 `bit:5:7` 或 `bit:5:5:7`。
   - 跨多个连续字节的 bit 字段仍必须写成 `bit:start:end:low:high`。
   - 只有当程序语义上确实把某个字节或几个连续字节中的某一位或某几位单独取出，并对这些 bit 进行计算、比较、约束检查、传播或分支使用时，才输出 bit 字段。

禁止输出任何 field_id 简写格式，例如 `b:5`、`b:40`、`bit:5:7`、`bit:5:5:7`。如果字段边界不确定，仍然必须在 `field_id` 中输出一个结构合法的最佳估计，并设置 `needs_review=true`，在 `review_reason` 中说明不确定性。

字段划分原则：

- 如果多个连续字节经常在同一组 `movzx / shl / or / add / cmp` 等指令中被组合使用，可以划为一个多字节字段。
- 如果多个字节只以分号形式出现，例如 `2;3`，只能说明当前指令同时使用了多个字节，不能单独作为合并字段的证据。
- 如果相邻字节分别参与不同比较、不同分支或不同处理逻辑，应拆成不同字段。
- 如果某个字节只被单独比较或单独消费，应保留为单字节字段。
- 如果一段字节只出现大块搬运或 copy，没有证据说明它们作为一个数值字段被组装，不要仅凭 copy 把整段合并成一个字段。
- 如果字段内部存在稳定 bit mask、shift、and/test 等位级证据，并且这些证据表明程序正在单独使用某一位或某几位的语义，可以将对应 bit range 输出为 bit 字段。
- 不要仅因为出现位运算指令就输出 bit 字段；必须能说明某个 bit range 被程序单独计算、比较、传播或用于分支。
- 如果字段边界证据不足，应选择更保守、更小的字段划分，并设置 `needs_review=true`。

# 程序语义描述规则

请为每个字段输出一段自然语言程序语义描述，描述该字段在程序中如何被使用。

可观察行为包括但不限于：

- 字段是否在解析早期被消费；
- 字段是否参与 `cmp`、`test` 或条件分支；
- 字段是否与常量、候选值或合法值比较；
- 字段是否参与范围、边界、最小值或最大值检查；
- 字段是否驱动 switch/case、handler 选择或多路径分派；
- 字段是否影响循环、长度、数量、解析范围或批处理规模；
- 字段是否作为地址、索引、偏移或对象定位信息使用；
- 字段是否主要作为业务数据、payload、测量值或参数被读取；
- 字段是否只在后期被搬运、存储、复制或透传；
- 字段是否证据不足，无法可靠判断。

每个语义描述都必须有 evidence 支持。Evidence 应尽量引用：

- log line number；
- function name；
- instruction assembly；
- field byte indices；
- branch taken / not taken；
- LOOP 标记；
- mask / shift / constant value / memory index 等可观察信息。

# 程序行为描述风格

`program_log_description` 和 `observed_behaviors[].behavior` 必须描述程序行为，而不是协议字段角色。

推荐使用以下行为动词：

- 读取；
- 组装；
- 拆分；
- 提取；
- 掩码；
- 比较；
- 测试；
- 分支；
- 约束；
- 选择；
- 写入；
- 存储；
- 回写；
- 传播；
- 控制循环；
- 影响处理范围。

推荐描述以下程序行为对象：

- 条件分支；
- 常量比较；
- 范围检查；
- 边界检查；
- 多路径处理；
- 后续解码路径；
- 循环或批量处理；
- 地址、索引或偏移访问；
- 数据保存或输出构造；
- bit 子段；
- 多字节数值。

优先使用这类句式：

- `该字段被程序...`
- `该字段参与...`
- `该字段驱动...`
- `该字段影响...`
- `该字段在...之后被...`

避免使用这类句式：

- `该字段是...`
- `该字节是...`
- `这是一个...字段`

禁止把字段命名为协议角色、协议层对象或规范字段类型，例如：

- version / type / service / APDU / NPDU；
- object id / instance / request parameter；
- descriptor / control byte / capability parameter；
- identifier / length field / counter / address field；
- function code / command / status。

如果函数名中包含协议术语，只能把函数名当作定位信息，不能把其中的协议术语写入 `program_log_description` 或 `observed_behaviors[].behavior`。

改写示例：

- 不写：`该字段是 APDU 控制字节。`
- 应写：`该字段被程序拆分为类别位段和布尔标志位段，并参与多处分支判断。`
- 不写：`该字段是对象标识。`
- 应写：`该字段被程序组装为 32 位数值，随后拆分为高位段和低位段，并分别参与比较和索引相关检查。`
- 不写：`该字段作为请求参数被回显。`
- 应写：`该字段被读取后保存，并在后续输出构造过程中被直接写回。`
- 不写：`该字段是描述符。`
- 应写：`该字段在读取后续字段前被掩码、比较和标志检查，用于约束后续解码路径。`

# 严格规则

1. 只能根据输入 log 判断。
2. 不要使用协议规范、tshark 字段名、payload 内容或源码。
3. 不要假设字段的传统协议语义。
4. 不要根据函数名、模块名、server 名、协议库名或其他命名细节推断具体协议名或协议规范字段。
5. 函数名只能作为定位证据，例如说明某条指令位于某个函数中；不能作为协议知识来源。
6. 字段划分和字段程序语义必须完全基于汇编指令体现的数据流、比较、分支、位操作、循环、内存访问、copy/store 等程序行为。
7. `program_log_description` 和 `observed_behaviors[].behavior` 必须使用程序行为描述风格，不要使用协议角色或规范字段类型命名字段。
8. 如果证据不足，必须设置 `needs_review=true` 并说明原因。
9. 不要输出 confidence。
10. 输出必须是严格 JSON，不要输出 markdown 或额外解释。

# 输出 JSON Schema

请输出一个 JSON object，格式如下：

```json
{
  "packet_summary": {
    "taint_start_line": 0,
    "log_line_count": 0,
    "notes": ""
  },
  "fields": [
    {
      "field_id": "b:0:0",
      "field_partition_evidence": [
        {
          "line_no": 12,
          "function": "decode_header",
          "instruction": "cmp byte ptr [rdi], 0x81",
          "field_refs": "0",
          "reason": "byte 0 is consumed independently by a constant comparison"
        }
      ],
      "program_log_description": "This field is consumed early and compared against a constant before a conditional branch, so it gates parser execution.",
      "observed_behaviors": [
        {
          "behavior": "early constant comparison before conditional branch",
          "evidence": [
            {
              "line_no": 12,
              "function": "decode_header",
              "instruction": "cmp byte ptr [rdi], 0x81",
              "field_refs": "0"
            }
          ]
        }
      ],
      "needs_review": false,
      "review_reason": ""
    }
  ]
}
```

# 输出要求

- `fields` 必须覆盖 log 中能观察到的 tainted input 字节或 bit 字段。
- `field_id` 必须使用 `b:start:end` 或 `bit:start:end:low:high`；单字节字段必须写成 `b:x:x`，禁止写成 `b:x`。
- `field_partition_evidence` 至少包含一条支持字段边界的证据；若证据不足，仍需列出最相关证据并设置 `needs_review=true`。
- `program_log_description` 应是一到两句话，描述程序行为语义，不要写传统字段类型名称、协议角色或协议层对象。
- `observed_behaviors[].behavior` 应使用短程序行为短语，例如 `常量比较后控制分支`、`多字节数值组装并存储`、`bit 子段提取后参与测试`，不要使用 `对象标识`、`控制字节`、`请求参数` 等角色名。
- `observed_behaviors` 可以为空，但若为空则必须设置 `needs_review=true`。
- 每个 evidence object 中无法确定的字段可填空字符串，但不要编造。
- 如果 evidence 来自 `RepeatSummary`，`instruction` 可填写摘要中的 `instruction` 或空字符串，`reason` 中应明确说明这是压缩摘要证据。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate program-log groundtruth with Codex or API.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Write all successful results into one merge JSON file instead of keeping per-log result files.",
    )
    parser.add_argument("--backend", choices=["codex", "api"], default="codex")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of logs to analyze concurrently. Default: 5. Ignored by --only-seq except it still runs one task.",
    )
    parser.add_argument("--start-seq", type=int, default=1)
    parser.add_argument(
        "--only-seq",
        type=int,
        default=None,
        help="Debug mode: process only the log whose filename starts with this sequence number.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-delay-seconds", type=int, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-cwd", type=Path, default=Path("/root"))
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_BASE_URL,
        help="Optional OpenAI-compatible API base URL. Default: use the OpenAI SDK default.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to env OPENAI_API_KEY.",
    )
    parser.add_argument("--api-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-reasoning-effort", default="high")
    parser.add_argument(
        "--api-temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for API backend. Default 0.0.",
    )
    parser.add_argument(
        "--api-top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling top_p for API backend. Default 1.0.",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds for API backend. Default: 600.",
    )
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
    if not match:
        return -1
    return int(match.group(1))


def metadata_from_log_name(path: Path) -> dict[str, Any]:
    match = re.match(r"^(\d+)_(.+)_(pkt_\d+)\.log$", path.name)
    if not match:
        return {
            "seq": seq_of(path),
            "protocol_name": "",
            "sample_id": "",
            "input_log_name": path.name,
        }
    return {
        "seq": int(match.group(1)),
        "protocol_name": match.group(2),
        "sample_id": match.group(3),
        "input_log_name": path.name,
    }


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")


def build_prompt(log_text: str) -> str:
    return PROMPT_TEMPLATE.replace("{program_execution_log}", log_text)


def looks_retryable(text: str) -> bool:
    lowered = text.lower()
    retry_markers = [
        "429",
        "too many requests",
        "rate limit",
        "exceeded retry limit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
    ]
    return any(marker in lowered for marker in retry_markers)


def print_status(message: str) -> None:
    with PRINT_LOCK:
        print(f"\r{message}", end="", flush=True)


def print_line(message: str) -> None:
    with PRINT_LOCK:
        print(f"\r{message}")


def wait_with_countdown(
    seconds: int,
    run_log: Path,
    reason: str,
    stop_event: threading.Event | None = None,
) -> None:
    log_line(run_log, f"waiting {seconds}s before retry: {reason}")
    start = time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        elapsed = time.time() - start
        remaining = seconds - int(elapsed)
        if remaining <= 0:
            return
        time.sleep(1)


def run_codex(prompt: str, output_path: Path, args: argparse.Namespace, run_log: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.codex_command,
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(args.codex_cwd),
        "-o",
        str(output_path),
    ]
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
        combined = f"returncode={process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        raise RuntimeError(combined)
    if not output_path.exists():
        combined = f"codex completed but did not write output file\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        raise RuntimeError(combined)
    return output_path.read_text(encoding="utf-8", errors="replace")


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


def output_for(input_path: Path, args: argparse.Namespace) -> Path:
    if args.merge_json is not None:
        return args.output_dir / ".tmp_merged_outputs" / f"{input_path.stem}.json"
    return args.output_dir / f"{input_path.stem}.json"


def load_merge_completed_seqs(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", []) if isinstance(data, dict) else []
    completed: set[int] = set()
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("seq"), int):
            completed.add(item["seq"])
    return completed


def append_merged_result(merge_path: Path, input_path: Path, response_text: str) -> None:
    merge_path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata_from_log_name(input_path)
    record = {
        **meta,
        "source_log": str(input_path),
        "response_text": response_text.rstrip() + "\n",
        "updated_at": now(),
    }
    with MERGE_LOCK:
        if merge_path.exists():
            data = json.loads(merge_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {"results": []}
        else:
            data = {"created_at": now(), "results": []}
        results = data.setdefault("results", [])
        if not isinstance(results, list):
            results = []
            data["results"] = results
        seq = record["seq"]
        replaced = False
        for index, existing in enumerate(results):
            if isinstance(existing, dict) and existing.get("seq") == seq:
                results[index] = record
                replaced = True
                break
        if not replaced:
            results.append(record)
        results.sort(key=lambda item: item.get("seq", -1) if isinstance(item, dict) else -1)
        data["updated_at"] = now()
        data["result_count"] = len(results)
        tmp_path = merge_path.with_suffix(merge_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(merge_path)


def process_one(input_path: Path, args: argparse.Namespace) -> None:
    output_path = output_for(input_path, args)
    log_text = input_path.read_text(encoding="utf-8", errors="replace")
    prompt = build_prompt(log_text)

    if args.backend == "codex":
        run_codex(prompt, output_path, args, args.run_log)
    else:
        response = run_api(prompt, args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response.rstrip() + "\n", encoding="utf-8")

    if args.merge_json is not None:
        response_text = output_path.read_text(encoding="utf-8", errors="replace")
        append_merged_result(args.merge_json, input_path, response_text)
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def process_with_retry(
    input_path: Path,
    args: argparse.Namespace,
    item_index: int,
    total_items: int,
    start_time: float,
    counters: dict[str, int],
    active: dict[int, dict[str, Any]],
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    seq = seq_of(input_path)
    output_path = output_for(input_path, args)
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        item_start = time.time()
        thread_id = threading.get_ident()
        with state_lock:
            active[thread_id] = {
                "seq": seq,
                "file": input_path.name,
                "attempt": attempt,
                "status": "running",
                "started_at": item_start,
            }
        msg = (
            f"start seq={seq} file={input_path.name} "
            f"item={item_index}/{total_items} attempt={attempt}"
        )
        print_line(f"[llm] {msg}")
        log_line(args.run_log, msg)
        try:
            process_one(input_path, args)
            elapsed = time.time() - item_start
            with state_lock:
                counters["done"] += 1
                active.pop(thread_id, None)
            done_msg = (
                f"done seq={seq} output={args.merge_json if args.merge_json else output_path.name} "
                f"elapsed={format_duration(elapsed)} total_elapsed={format_duration(time.time() - start_time)}"
            )
            print_line(f"[llm] {done_msg}")
            log_line(args.run_log, done_msg)
            return
        except Exception as exc:
            if stop_event.is_set():
                with state_lock:
                    active.pop(thread_id, None)
                return
            error_text = str(exc)
            retry_reason = "retryable error" if looks_retryable(error_text) else "error"
            with state_lock:
                counters["errors"] += 1
                active[thread_id] = {
                    "seq": seq,
                    "file": input_path.name,
                    "attempt": attempt,
                    "status": f"waiting after {retry_reason}",
                    "started_at": item_start,
                    "retry_until": time.time() + args.retry_delay_seconds,
                }
            err_msg = (
                f"error seq={seq} file={input_path.name} "
                f"attempt={attempt}: {error_text[:1000]}"
            )
            print_line(f"[llm] {err_msg}")
            log_line(args.run_log, err_msg)
            wait_with_countdown(args.retry_delay_seconds, args.run_log, retry_reason, stop_event)


def status_loop(
    total_items: int,
    start_time: float,
    counters: dict[str, int],
    active: dict[int, dict[str, Any]],
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        with state_lock:
            done = counters["done"]
            skipped = counters["skipped"]
            errors = counters["errors"]
            active_items = list(active.values())
        remaining = max(total_items - done - skipped, 0)
        active_text = "; ".join(
            f"{item['seq']}:{item['status']}#{item['attempt']}:{format_duration(time.time() - item['started_at'])}"
            for item in active_items[:5]
        )
        print_status(
            "[status] "
            f"done={done} skipped={skipped} active={len(active_items)} remaining={remaining} "
            f"errors={errors} elapsed={format_duration(time.time() - start_time)}"
            + (f" | {active_text}" if active_text else "")
        )
        time.sleep(1)
    print()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_log.parent.mkdir(parents=True, exist_ok=True)
    merge_completed_seqs: set[int] = set()
    if args.merge_json is not None and args.merge_json.exists() and not args.overwrite:
        merge_completed_seqs = load_merge_completed_seqs(args.merge_json)
    if args.only_seq is not None:
        all_inputs = sorted(
            [path for path in args.input_dir.glob("*.log") if seq_of(path) == args.only_seq],
            key=seq_of,
        )
        if not all_inputs:
            raise SystemExit(f"no log found with sequence {args.only_seq} under {args.input_dir}")
    else:
        all_inputs = sorted(
            [path for path in args.input_dir.glob("*.log") if seq_of(path) >= args.start_seq],
            key=seq_of,
        )
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    start_time = time.time()
    log_line(
        args.run_log,
        (
            f"start backend={args.backend} input_dir={args.input_dir} "
            f"start_seq={args.start_seq} only_seq={args.only_seq} workers={args.workers} "
            f"merge_json={args.merge_json}"
        ),
    )

    stop_event = threading.Event()
    state_lock = threading.Lock()
    counters = {"done": 0, "skipped": 0, "errors": 0}
    active: dict[int, dict[str, Any]] = {}

    def handle_sigint(signum: int, frame: Any) -> None:
        stop_event.set()
        log_line(args.run_log, "interrupted by Ctrl+C")
        print_line("[llm] interrupted by Ctrl+C; waiting for active worker calls to settle")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    pending_inputs: list[tuple[int, Path]] = []
    for index, input_path in enumerate(all_inputs, start=1):
        output_path = output_for(input_path, args)
        remaining = len(all_inputs) - index
        already_done = (
            seq_of(input_path) in merge_completed_seqs
            if args.merge_json is not None
            else output_path.exists()
        )
        if already_done and not args.overwrite:
            with state_lock:
                counters["skipped"] += 1
            msg = f"skip existing seq={seq_of(input_path)} input={input_path.name} remaining={remaining}"
            print_line(f"[llm] {msg}")
            log_line(args.run_log, msg)
            continue
        pending_inputs.append((index, input_path))

    worker_count = 1 if args.only_seq is not None else min(args.workers, max(len(pending_inputs), 1))
    status_thread = threading.Thread(
        target=status_loop,
        args=(len(all_inputs), start_time, counters, active, state_lock, stop_event),
        daemon=True,
    )
    status_thread.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_with_retry,
                    input_path,
                    args,
                    index,
                    len(all_inputs),
                    start_time,
                    counters,
                    active,
                    state_lock,
                    stop_event,
                )
                for index, input_path in pending_inputs
            ]
            for future in concurrent.futures.as_completed(futures):
                if stop_event.is_set():
                    break
                future.result()
    except KeyboardInterrupt:
        stop_event.set()
        log_line(args.run_log, f"stopped total_elapsed={format_duration(time.time() - start_time)}")
        return
    finally:
        stop_event.set()
        status_thread.join(timeout=2)

    log_line(args.run_log, f"finished total_elapsed={format_duration(time.time() - start_time)}")
    print_line(f"[llm] finished total_elapsed={format_duration(time.time() - start_time)}")


if __name__ == "__main__":
    main()
