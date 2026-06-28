#!/usr/bin/env python3
"""Fuse active z-axis descriptions into field-level semantic summaries.

This script is the new Stage 4 field-level LLM step. It reads per-field active
axis semantic items produced by build_stage4_field_profiles.py, sends each
field's active z-axis descriptions and weights to an LLM, and writes field-level
program semantic summaries plus optional traditional coarse tags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/root/semvec/difftrace/stage4/out/stage4_field_profiles/field_semantic_profiles.jsonl")
DEFAULT_OUT_DIR = Path("/root/semvec/difftrace/stage4/out/stage4_field_semantic_fusion")
DEFAULT_JSONL = DEFAULT_OUT_DIR / "field_semantic_fused_profiles.jsonl"
DEFAULT_CSV = DEFAULT_OUT_DIR / "field_semantic_fused_vectors.csv"
DEFAULT_RESPONSES = DEFAULT_OUT_DIR / "field_semantic_fusion_prompt_responses.md"
DEFAULT_RUN_LOG = DEFAULT_OUT_DIR / "field_semantic_fusion_run.log"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_WORKERS = 5
DEFAULT_RETRY_DELAY_SECONDS = 30 * 60

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

COARSE_FIELD_SEMANTIC_TAGS = {
    "identifier",
    "length_or_count",
    "control_or_flags",
    "addressing",
    "data_value",
    "other_or_unknown",
}

PROMPT_TEMPLATE = """# 任务

我们正在做一个协议字段程序语义分析流程。

整体链路如下：

1. Stage 1: 字段恢复。
   系统先重放原始协议报文，通过程序执行污点日志观察每个输入字节或 bit 如何被程序消费，从而恢复字段边界。这里得到的是“程序实际如何使用字段”，而不是协议规范里的人工模板。

2. Stage 2: 字段级 mutation 与执行差分。
   对 Stage 1 得到的每个字段，系统会构造多组 mutation 值并重新执行程序。
   每个 mutation 执行后，系统会把 mutation 执行日志与 baseline 执行日志比较，得到程序行为差分结果。

3. Stage 3: 字段表示空间学习。
   系统把每个字段的差分结果编码为程序行为向量，并训练 AutoEncoder，将字段行为表示压缩为 8 维 latent 表示 z1...z8。

4. Stage 4: latent 维度命名与字段级描述生成。
   Stage 4 首先根据每个 z 维与程序行为探针的相关性，为每个 z 维 high_value / low_value 端生成可读、受差分材料约束的程序行为语义名称。
   然后，Stage 4 会根据每个字段激活的若干 z 轴 high/low 端语义及其权重，生成字段级程序行为描述，并进一步投影到传统 coarse semantic tags。

当前的任务是：读取同一个字段相关的多个 active z 轴语义项，将它们总结成一句字段级程序行为语义描述，并基于该描述选择传统 coarse semantic tags。

你将看到同一个字段在多个 active latent z 轴上的程序行为语义项。每个语义项描述该字段在某个程序行为方向上的表现，并带有该语义项对当前字段的权重 `axis_score`。

这些语义项可能涉及：

- 字段取值变化导致程序执行路径偏离 baseline；
- 字段触发边界值、极端值或异常路径分化；
- 字段参与密集比较、范围检查、条件分支；
- 字段影响循环次数、处理规模、消费范围或资源使用；
- 字段驱动多类别分派、模式选择或离散处理路径；
- 字段在程序后期被消费、存储、传播或回写；
- 字段主要表现为普通数据搬运或弱语义信号。

你的任务有两步：

1. 总结这些 active z 轴语义项，生成一句字段级程序行为语义描述。
2. 根据融合后的字段级程序行为语义，将该字段投影到传统 coarse semantic tags。

请注意：

- active z 轴语义项是输入材料，不是最终答案；
- 多个 z 轴可能描述同一字段的不同行为侧面；
- 你必须综合所有重要 z 轴，尤其要考虑 `axis_score` 较高的语义项；
- 不要简单拼接 z 轴描述；
- 不要只选择最高分 z 轴；
- 不要直接照搬某个 z 轴的 `latent_name` 作为字段级总结；
- 字段级总结应描述“程序如何使用该字段”，而不是字段在协议规范中的名称。

# 输入

输入是一个 JSON object：

```json
{
  "axis_semantics": [
    {
      "axis": "z7",
      "side": "high",
      "axis_score": 0.49,
      "latent_name": "字段触发边界与异常多路径处理",
      "definition": "字段取边界值或极端值时，程序内部呈现出多种处理路径、范围检查、异常路径或拒绝路径。",
      "percentile": 0.99
    }
  ]
}
```

实际输入如下：

```json
{field_payload_json}
```

其中：

- `axis_semantics` 是该字段激活的 z 轴语义项列表；
- `axis_semantics[].axis_score` 表示该 z 轴语义项对当前字段语义的权重；
- `axis_semantics[].latent_name` 是该 z 轴端的简短程序行为名称；
- `axis_semantics[].definition` 是该 z 轴端的程序行为定义；
- `axis_semantics[].percentile` 表示该字段在当前 z 轴上的相对位置。

# 输出

请输出严格 JSON object，不要输出 markdown，不要输出额外解释。

JSON schema：

```json
{
  "field_program_semantic_summary": "one concise sentence describing the fused program behavior of this field",
  "traditional_semantic_tags": [
    "identifier"
  ],
  "tag_rationale": "brief reason for the traditional tag projection",
  "needs_review": false,
  "review_reason": ""
}
```

# 字段级程序行为语义描述规则

`field_program_semantic_summary` 必须满足：

1. 使用一句话，长度尽量控制在 40 个汉字以内。
2. 描述程序行为，而不是协议规范角色。
3. 不要出现具体协议名、字段名、函数名、源码模块名。
4. 不要使用 payload 字节值、样本编号或字段 id 推断语义。
5. 不要说“这是 length 字段 / identifier 字段 / flag 字段”等传统标签名称。
6. 可以描述以下程序行为：
   - 早期常量比较或入口门控；
   - 范围检查、边界检查、异常路径或拒绝路径；
   - 多路径分支、离散分派或处理模式选择；
   - 循环次数、处理规模、消费范围或资源使用变化；
   - 多字节组装、bit 提取、数值传播；
   - 后期消费、存储、回写、普通数据搬运；
   - 弱信号或无法形成稳定行为解释。
7. 如果多个高权重语义项共同指向一个更高层行为，应写融合后的行为。
8. 如果语义项之间存在张力，应优先保留最能解释高权重语义项的共同程序行为，并在 `needs_review` 中标记不确定性。

好的字段级总结示例：

```text
该字段取值变化主要触发范围检查和异常路径分化。
```

不好的字段级总结示例：

```text
z7 high: 字段触发边界与异常多路径处理; z6 high: 字段触发边界检查与异常路径分化。
```

原因：这只是 z 轴描述拼接，不是字段级融合。

# 传统 coarse semantic tags

`traditional_semantic_tags` 只能从以下 6 个标签中选择：

```text
identifier
length_or_count
control_or_flags
addressing
data_value
other_or_unknown
```

输出规则：

1. 至少输出 1 个标签，最多输出 3 个标签。
2. 标签按相关性从高到低排序。
3. 如果材料不足或无法归类，输出 `other_or_unknown`。
4. 必须先基于融合后的字段级程序行为语义判断，再选择传统标签。
5. 不要因为出现“比较”“分支”“异常路径”“处理偏离”就自动选择 `control_or_flags`。
6. 如果只是常量/合法性校验导致拒绝或早退，优先考虑 `identifier`，不是 `control_or_flags`。
7. 如果主要是范围、边界、极端值、消费范围、循环或处理规模变化，优先考虑 `length_or_count`。
8. 只有明确体现“离散模式选择、命令/状态分派、bit 控制位、多个互斥处理模式”时，才优先选择 `control_or_flags`。

# 传统标签判定参考

以下判定参考用于把字段级程序行为语义映射到传统 coarse tags。你的输入不是协议字段名，也不是底层数值指标，而是若干 z 轴的 `latent_name` / `definition` 及其权重。因此判断时要先理解高权重 z 轴语义项共同表达的“程序行为形态”，再谨慎映射到传统标签。

特别注意：输入中的 z 轴语义项通常比较抽象，例如“多路径处理偏离”“边界与异常路径分化”“多类别行为分化”“后期参与约束检查”。这些是程序行为形态，不等于传统标签本身。不要把某个行为词直接当作标签词。

## 推荐判定顺序

请按以下顺序判断，不要直接做关键词匹配：

1. 高权重语义项是否主要是“边界/极端/范围/异常路径/资源状态”一组？如果是，先考虑 `length_or_count` 或 `addressing`，不要直接选 `control_or_flags`。
2. 高权重语义项是否主要是“后期消费/后期约束/保存/传播/回写/普通值解析”一组？如果是，先考虑 `data_value`。
3. 高权重语义项是否主要是“固定值、入口门控、常量比较、合法性确认、错误值统一拒绝”一组？如果是，选 `identifier`。
4. 高权重语义项是否明确表达“不同取值选择不同模式、命令、状态、handler/case 或 bit 控制位”？如果是，选 `control_or_flags`。
5. 高权重语义项是否表达“对象/实例/索引/偏移/地址/查表/访问目标定位”？如果是，选 `addressing`。
6. 如果无法判断，选 `other_or_unknown`。

## identifier

当高权重 z 轴语义项主要表达以下含义时，优先考虑 `identifier`：

- 字段取值变化导致入口门控、固定值校验、少量候选值校验或合法性确认失败；
- 字段像“必须匹配某个值或某类值”一样被程序使用；
- 错误或非预期取值主要导致拒绝、早退、统一异常路径或不进入后续解析；
- z 轴名称中出现“入口门控”“常量比较”“合法性校验”“固定值检查”“早期拒绝”等含义。

如果输入只显示“全局偏离”“密集比较”“异常路径”，但没有明确长度/范围/地址/模式线索，可以把 `identifier` 作为候选，因为固定值或合法性字段 mutation 后也常造成整体偏离。不要因为该字段触发了比较、分支或异常路径就自动选 `control_or_flags`。

## length_or_count

当高权重 z 轴语义项主要表达以下含义时，优先考虑 `length_or_count`：

- 字段取值变化影响处理规模、输入消费范围、循环次数、重复次数、解析边界、资源使用或路径长短；
- 字段对范围、边界值、极端值高度敏感，并且这种敏感性可以解释为“处理多少、读多少、循环多少、覆盖多大范围”；
- 字段变化导致截断、扩张、越界、资源放大、提前停止或后续处理量变化；
- z 轴名称中出现“边界检查”“极端处理”“消费范围”“循环/规模变化”“资源放大/截断”“长度约束”等含义。

在当前输入中，“边界检查”“极端处理”“异常路径分化”“资源相关状态”非常常见。若这些语义项占主导，且没有明确的离散模式/命令/bit 控制线索，应优先把它解释为 `length_or_count` 候选，而不是 `control_or_flags`。因为长度、数量、范围类字段在 mutation 到边界或极端值时最容易产生这种行为形态。

## control_or_flags

当高权重 z 轴语义项主要表达以下含义时，优先考虑 `control_or_flags`：

- 字段不同离散取值选择不同处理模式、命令、状态、选项、handler、case 或互斥分支；
- bit 或 bit 段被单独提取后作为开关、标志位、模式位或条件控制位使用；
- z 轴语义项强调“多类别分派”“离散模式选择”“处理等价类区分”“命令/状态路径选择”“bit 控制”等含义；
- 字段改变的是“走哪类处理逻辑”，而不是“是否通过固定校验”或“处理多少数据”。

严格限制：`control_or_flags` 不是“比较、分支、异常路径、多路径偏离、多类别分化”的默认标签。许多 identifier、length_or_count、addressing 字段 mutation 后也会出现多路径或多类别行为。只有高权重语义项清楚表达“离散取值在正常解析中选择不同功能/模式/命令/状态/bit 控制”时，才把 `control_or_flags` 放在首位。

## addressing

当高权重 z 轴语义项主要表达以下含义时，优先考虑 `addressing`：

- 字段用于定位对象、索引、偏移、地址、寄存器、表项、实例、通道或集合元素；
- 字段变化导致查表、对象匹配、访问范围、索引范围或目标有效性相关差异；
- 越界或非法定位值导致异常路径，但合法值主要改变访问目标，而不是改变整体处理规模；
- z 轴名称中出现“对象/实例匹配”“索引/偏移访问”“地址定位”“查表路径”“访问范围校验”等含义。

`addressing` 与 `length_or_count` 都可能边界敏感。如果语义项只说“边界/异常路径”，但没有目标定位线索，优先不要强行选 `addressing`。如果语义项更像“访问哪个目标/位置”，选 `addressing`；如果更像“处理多少数据/循环多少次”，选 `length_or_count`。

## data_value

当高权重 z 轴语义项主要表达以下含义时，优先考虑 `data_value`：

- 字段主要被保存、传播、回写、输出、参与普通计算或在程序后期被消费；
- 字段变化没有清楚地决定入口门控、离散模式、长度规模、索引定位或解析结构；
- 字段像业务参数、测量值、时间值、普通载荷或可传递数据一样被程序使用；
- z 轴名称中出现“后期消费”“存储/传播/回写”“普通数据搬运”“弱控制流影响”等含义。

在当前输入中，“后期参与约束检查”“后期参与约束值分派”“晚期进入程序消费”不应自动推向 `control_or_flags`。如果这些语义项占主导，而没有明确模式/命令/长度/地址线索，应优先考虑 `data_value`。字段确实被比较或检查，但这些行为不能稳定解释为 identifier、length_or_count、control_or_flags 或 addressing 时，也可以把 `data_value` 作为候选。

## other_or_unknown

当高权重 z 轴语义项不足以形成稳定解释时，选择 `other_or_unknown`：

- active z 轴语义项很少、权重很低或彼此冲突；
- 字段级总结只能写成很泛的“程序行为变化”；
- 语义项混合了边界、分支、传播、异常等多个方向，但无法判断主导传统语义；
- 字段可能是透明字段、低价值字段或当前材料无法可靠分类。

如果字段语义信号强但传统标签难以区分，可以把 `other_or_unknown` 作为第二或第三候选，并设置 `needs_review=true`。不要为了避免 unknown 而强行把首选改成 `control_or_flags`。

# 冲突与不确定性处理

如果语义项同时支持多个标签：

1. 优先根据高 `axis_score` 语义项判断；
2. 优先根据字段级融合语义判断，而不是逐 z 轴投票；
3. 可以输出多个候选标签；
4. 如果最高优先级标签很不确定，设置 `needs_review=true` 并说明原因。

如果出现以下情况，应设置 `needs_review=true`：

- 高权重语义项互相冲突；
- 字段级行为描述只能写得很泛；
- traditional tag 投影依赖很弱；
- 多个标签几乎同等合理；
- active 语义项太少。

# 输出字段要求

- `field_program_semantic_summary`：非空字符串；
- `traditional_semantic_tags`：字符串数组，长度 1 到 3；
- `tag_rationale`：简短说明为什么选择这些 traditional tags；
- `needs_review`：布尔值；
- `review_reason`：如果 `needs_review=false`，输出空字符串；如果为 true，简要说明原因。"""

PRINT_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
RESPONSE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse field active z-axis semantic items into one field-level semantic summary.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--responses-md", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Prompt template file. It must contain {field_payload_json}.",
    )
    parser.add_argument("--backend", choices=["api", "codex"], default="api")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--only-field-uid", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-delay-seconds", type=int, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-cwd", type=Path, default=Path("/root"))
    parser.add_argument("--api-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--api-top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and write a preview without calling the LLM.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_line(message: str) -> None:
    with PRINT_LOCK:
        print(f"\r{message}")


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_no}: JSON line must be an object")
            rows.append(obj)
    return rows


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = [str(item).strip() for item in value]
    else:
        raw = [part.strip() for part in str(value or "").replace(",", ";").split(";")]
    tags: list[str] = []
    for tag in raw:
        if tag in COARSE_FIELD_SEMANTIC_TAGS and tag not in tags:
            tags.append(tag)
    return tags


def compact_axis_semantics(row: dict[str, Any]) -> list[dict[str, Any]]:
    semantics: list[dict[str, Any]] = []
    for item in row.get("active_axis_explanations") or []:
        if not isinstance(item, dict):
            continue
        semantics.append(
            {
                "axis": item.get("axis", ""),
                "side": item.get("side", ""),
                "axis_score": item.get("axis_score", 0.0),
                "latent_name": item.get("latent_name", ""),
                "definition": item.get("definition", ""),
                "percentile": item.get("percentile", ""),
            }
        )
    semantics.sort(key=lambda item: float(item.get("axis_score") or 0.0), reverse=True)
    return semantics


def build_field_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis_semantics": compact_axis_semantics(row),
    }


def axis_list_from_semantics(axis_semantics: list[dict[str, Any]]) -> str:
    return ",".join(
        f"{item.get('axis', '')}:{item.get('side', '')}"
        for item in axis_semantics
        if item.get("axis") and item.get("side")
    )


def load_prompt_template(args: argparse.Namespace) -> str:
    if args.prompt_file:
        template = args.prompt_file.read_text(encoding="utf-8")
    else:
        template = PROMPT_TEMPLATE
    if not template.strip():
        raise SystemExit(
            "field-level fusion prompt is empty; discuss/fill PROMPT_TEMPLATE or pass --prompt-file"
        )
    if "{field_payload_json}" not in template:
        raise SystemExit("prompt template must contain {field_payload_json}")
    return template


def build_prompt(template: str, row: dict[str, Any]) -> str:
    payload = build_field_payload(row)
    return template.replace("{field_payload_json}", json.dumps(payload, ensure_ascii=False, indent=2))


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        obj = json.loads(stripped[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM response must be a JSON object")
    return obj


def validate_response(obj: dict[str, Any]) -> dict[str, Any]:
    summary = str(obj.get("field_program_semantic_summary", "")).strip()
    tags = normalize_tags(obj.get("traditional_semantic_tags"))
    if not summary:
        raise ValueError("missing field_program_semantic_summary")
    if not tags:
        tags = ["other_or_unknown"]
    tags = tags[:3]
    return {
        "field_program_semantic_summary": summary,
        "traditional_semantic_tags": tags,
        "tag_rationale": str(obj.get("tag_rationale", "")).strip(),
        "needs_review": bool(obj.get("needs_review", False)),
        "review_reason": str(obj.get("review_reason", "")).strip(),
    }


def call_api(prompt: str, args: argparse.Namespace) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("missing Python package: install openai to use --backend api") from exc
    api_key = args.api_key or DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing API key")
    client = OpenAI(api_key=api_key, base_url=args.api_base_url, timeout=args.api_timeout)
    response = client.chat.completions.create(
        model=args.api_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.api_temperature,
        top_p=args.api_top_p,
    )
    return response.choices[0].message.content or ""


def call_codex(prompt: str, args: argparse.Namespace) -> str:
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=True) as stdout_file, tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8", delete=True
    ) as stderr_file, tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=True) as output_file:
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


def call_llm(prompt: str, args: argparse.Namespace) -> str:
    if args.backend == "api":
        return call_api(prompt, args)
    return call_codex(prompt, args)


def load_done(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    done: set[str] = set()
    for row in load_jsonl(output_jsonl):
        field_uid = str(row.get("field_uid", "")).strip()
        if field_uid:
            done.add(field_uid)
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with RESPONSE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_response_md(path: Path, field_uid: str, prompt: str, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with RESPONSE_LOCK:
        exists = path.exists()
        with path.open("a", encoding="utf-8") as handle:
            if not exists:
                handle.write("# Stage 4D Field Semantic Fusion Prompt Responses\n\n")
            handle.write(f"## {field_uid}\n\n")
            handle.write("### Prompt\n\n```text\n")
            handle.write(prompt)
            handle.write("\n```\n\n### Response\n\n```text\n")
            handle.write(response)
            handle.write("\n```\n\n")


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    if not jsonl_path.exists():
        return
    rows = load_jsonl(jsonl_path)
    fieldnames = [
        "protocol_name",
        "sample_id",
        "field_id",
        "field_uid",
        "field_program_semantic_summary",
        "traditional_semantic_tags",
        "tag_rationale",
        "needs_review",
        "review_reason",
        "active_axes",
        "dominant_axes",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            if isinstance(out["traditional_semantic_tags"], list):
                out["traditional_semantic_tags"] = ";".join(out["traditional_semantic_tags"])
            writer.writerow(out)


def process_one(row: dict[str, Any], template: str, args: argparse.Namespace, index: int, total: int) -> dict[str, Any]:
    field_uid = str(row.get("field_uid") or f"{row.get('protocol_name')}-{row.get('sample_id')}-{row.get('field_id')}")
    prompt = build_prompt(template, row)
    if args.dry_run:
        append_response_md(args.responses_md, field_uid, prompt, "[dry-run skipped]")
        return {"field_uid": field_uid, "dry_run": True}
    attempt = 0
    while True:
        attempt += 1
        start = time.time()
        print_line(f"[stage4-field-fusion] start {index}/{total} {field_uid} attempt={attempt}")
        try:
            response = call_llm(prompt, args)
            parsed = validate_response(extract_json_object(response))
            axis_semantics = compact_axis_semantics(row)
            active_axes = row.get("active_axes") or axis_list_from_semantics(axis_semantics)
            dominant_axes = row.get("dominant_axes") or active_axes
            out = {
                "protocol_name": row.get("protocol_name", ""),
                "sample_id": row.get("sample_id", ""),
                "field_id": row.get("field_id", ""),
                "field_uid": row.get("field_uid", field_uid),
                "active_axes": active_axes,
                "dominant_axes": dominant_axes,
                "axis_semantics": axis_semantics,
                **parsed,
            }
            append_jsonl(args.output_jsonl, out)
            append_response_md(args.responses_md, field_uid, prompt, response)
            elapsed = format_duration(time.time() - start)
            print_line(f"[stage4-field-fusion] done {index}/{total} {field_uid} elapsed={elapsed}")
            log_line(args.run_log, f"done field_uid={field_uid} attempt={attempt} elapsed={elapsed}")
            return out
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            message = f"error field_uid={field_uid} attempt={attempt}: {exc}"
            print_line(f"[stage4-field-fusion] {message}; retry in {args.retry_delay_seconds}s")
            log_line(args.run_log, message)
            time.sleep(args.retry_delay_seconds)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    template = load_prompt_template(args)
    rows = load_jsonl(args.input)
    if args.only_field_uid:
        rows = [row for row in rows if str(row.get("field_uid", "")) == args.only_field_uid]
    else:
        rows = rows[max(args.start_index - 1, 0) :]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.overwrite:
        done = load_done(args.output_jsonl)
        rows = [row for row in rows if str(row.get("field_uid", "")) not in done]
    if args.overwrite:
        args.output_jsonl.unlink(missing_ok=True)
        args.output_csv.unlink(missing_ok=True)
        args.responses_md.unlink(missing_ok=True)
    total = len(rows)
    log_line(args.run_log, f"start total={total} input={args.input}")
    print_line(f"[stage4-field-fusion] pending={total} backend={args.backend} workers={args.workers}")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_one, row, template, args, index, total)
                for index, row in enumerate(rows, start=1)
            ]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        log_line(args.run_log, "interrupted by user")
        raise
    write_csv_from_jsonl(args.output_jsonl, args.output_csv)
    log_line(args.run_log, f"finished total={total} output={args.output_jsonl}")
    print_line(f"[stage4-field-fusion] finished output={args.output_jsonl}")


if __name__ == "__main__":
    main()
