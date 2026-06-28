#!/usr/bin/env python3
"""RQ4-B ablation: summarize field semantics directly from raw probe features.

This script removes the Stage 3/4 latent semantic abstraction:

    28D strategy-aware behavior probes -> LLM field summary

It writes outputs with the same key columns and semantic columns used by the
normal Stage 4 field fusion result, so existing RQ2 tshark and program-log
semantic evaluators can consume it through their --predictions/--stage4-profiles
arguments.
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


DEFAULT_INPUT = Path("/root/semvec/difftrace/stage3/out/stage3_filtered/stage3_dataset_semantic_fields.csv")
DEFAULT_OUT_DIR = Path("/root/semvec/RQ4/out/no_latent_direct")
DEFAULT_JSONL = DEFAULT_OUT_DIR / "field_semantic_direct_profiles.jsonl"
DEFAULT_CSV = DEFAULT_OUT_DIR / "field_semantic_direct_vectors.csv"
DEFAULT_RESPONSES = DEFAULT_OUT_DIR / "field_semantic_direct_prompt_responses.md"
DEFAULT_RUN_LOG = DEFAULT_OUT_DIR / "field_semantic_direct_run.log"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_WORKERS = 5
DEFAULT_RETRY_DELAY_SECONDS = 30 * 60

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

KEY_COLS = ["protocol_name", "sample_id", "field_id"]
CONTEXT_COLS = [
    "relative_start",
    "field_instr_ratio",
    "compare_ratio",
    "constraint_value_diversity",
]
GROUPS = ["neighborhood", "boundary", "enum", "extreme"]
GROUP_FEATURES = [
    "mean_baseline_distance",
    "mean_pairwise_distance",
    "max_pairwise_distance",
    "metric_vector_variance",
    "unique_vector_ratio",
    "loop_dispersion",
]
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
   完整方法会把每个字段的差分结果编码为程序行为向量，并训练 AutoEncoder，将字段行为表示压缩为 8 维 latent 表示 z1...z8。

4. Stage 4: latent 维度命名与字段级描述生成。
   完整方法会先为每个 z 维 high_value / low_value 端生成受差分材料约束的程序行为语义名称，再根据字段激活的若干 z 轴语义及其权重，生成字段级程序行为描述，并投影到传统 coarse semantic tags。

当前任务是 no-latent direct 版本：不使用 latent z 轴语义项，而是直接读取同一个字段的 28 维程序行为探针摘要，生成一句字段级程序行为语义描述，并基于该描述选择传统 coarse semantic tags。

你将看到同一个字段在四组 mutation 值上的程序执行差分摘要。它们可能体现路径偏离、边界/极端处理、范围检查、条件分支、循环或消费范围变化、离散分派、后期存储传播或弱语义信号。

你的任务有两步：

1. 总结这些 28 维程序行为探针摘要，生成一句字段级程序行为语义描述。
2. 根据字段级程序行为语义，将该字段投影到传统 coarse semantic tags。

请注意：

- 28 维程序行为探针摘要是输入材料，不是最终答案；
- 你必须综合 context、四组 mutation 摘要和 diagnostics；
- 不要简单复述指标名，不要只根据某一个最高指标做判断；
- 字段级总结应描述“程序如何使用该字段”，而不是字段在协议规范中的名称；
- 不要根据协议名、字段 id、字段位置或常识推断具体协议字段名；
- 不要编造 trace 中没有体现的协议语义；
- 不要把统计摘要包装成确定的协议含义；
- 输出中不要提 latent、z 轴、AE、消融实验等方法词。

判断时请把四组 mutation 摘要当作互补证据：`boundary` / `extreme` 更容易暴露范围、长度、资源和异常路径行为；`enum` 更容易暴露离散分派、候选值约束或模式选择；`neighborhood` 更容易区分字段是否对小幅扰动敏感。不要把某一组的高偏离直接等同于某个传统标签，而应观察不同组之间的相对模式。例如，所有组都产生相似的统一拒绝，更像固定值或合法性校验；只有边界/极端值明显放大差异，更像范围或规模相关；枚举组内部出现多个稳定行为等价类，才更支持离散控制或分派。

# 输入

输入是一个 JSON object。实际输入如下：

```json
{field_payload_json}
```

其中：

- `context.relative_start` 表示字段首次被程序消费的位置；越大表示越晚被消费；
- `context.field_instr_ratio` 表示 baseline 中字段相关指令占比；越大表示处理强度越高；
- `context.compare_ratio` 表示字段相关比较行为比例；越大表示更多参与条件判断、约束检查或分支；
- `context.constraint_value_diversity` 表示程序能区分的候选值或约束值丰富度；
- `mutation_groups` 有四组：`neighborhood` 原值附近扰动，`boundary` 边界值，`enum` 候选离散值，`extreme` 极端/异常值；
- 每组 6 个摘要：`mean_baseline_distance` 表示相对 baseline 的平均偏离，`mean_pairwise_distance` / `max_pairwise_distance` 表示组内处理分歧，`metric_vector_variance` 表示响应是否不稳定，`unique_vector_ratio` 表示是否形成多个行为等价类，`loop_dispersion` 表示是否影响循环、批量处理或重复执行；
- `diagnostics` 提供 mutation 数量、有效样本、整体差分离散度、约束数量和字段类型等辅助诊断信息。

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
6. 不要直接写“mean_baseline_distance 很高”“unique_vector_ratio 较低”这类指标描述，要转写为程序行为。
7. 可以描述以下程序行为：
   - 早期常量比较或入口门控；
   - 范围检查、边界检查、异常路径或拒绝路径；
   - 多路径分支、离散分派或处理模式选择；
   - 循环次数、处理规模、消费范围或资源使用变化；
   - 多字节组装、bit 提取、数值传播；
   - 后期消费、存储、回写、普通数据搬运；
   - 弱信号或无法形成稳定行为解释。
8. 如果多个指标方向共同指向一个更高层行为，应写融合后的行为。
9. 如果指标之间存在张力，应优先保留最能解释主要差分模式的共同程序行为，并在 `needs_review` 中标记不确定性。
10. 如果差分只显示“程序发生变化”，但无法判断变化属于校验、长度、控制、地址还是普通数据，应把总结写得保守，并优先设置 `needs_review=true`。
11. 如果 `valid_mutations` 很少、`unique_metric_vectors` 很低或多组摘要几乎相同，不要过度解释为复杂语义。

好的字段级总结示例：

```text
该字段取值变化主要触发范围检查和异常路径分化。
```

不好的字段级总结示例：

```text
boundary mean_baseline_distance 高，enum unique_vector_ratio 高。
```

原因：这只是指标复述，不是字段级程序行为总结。

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
4. 必须先基于字段级程序行为语义判断，再选择传统标签。
5. 不要因为出现“比较”“分支”“异常路径”“处理偏离”就自动选择 `control_or_flags`。
6. 如果只是常量/合法性校验导致拒绝或早退，优先考虑 `identifier`，不是 `control_or_flags`。
7. 如果主要是范围、边界、极端值、消费范围、循环或处理规模变化，优先考虑 `length_or_count`。
8. 只有明确体现“离散模式选择、命令/状态分派、bit 控制位、多个互斥处理模式”时，才优先选择 `control_or_flags`。

# 传统标签判定参考

以下判定参考用于把字段级程序行为语义映射到传统 coarse tags。你的输入不是协议字段名，也不是 z 轴语义名称，而是四组 mutation 程序行为差分摘要。因此判断时要先理解各组 mutation 共同表达的“程序行为形态”，再谨慎映射到传统标签。

特别注意：输入中的差分摘要通常比较抽象，例如“boundary / extreme 组产生强偏离”“enum 组内部行为分化明显”“loop_dispersion 升高”。这些是程序行为形态，不等于传统标签本身。不要把某个指标词直接当作标签词。

## 推荐判定顺序

请按以下顺序判断，不要直接做关键词匹配：

1. 主导差分是否主要是“边界/极端/范围/异常路径/资源状态”一组？如果是，先考虑 `length_or_count` 或 `addressing`，不要直接选 `control_or_flags`。
2. 主导差分是否主要是“后期消费/后期约束/保存/传播/回写/普通值解析”一组？如果是，先考虑 `data_value`。
3. 主导差分是否主要是“固定值、入口门控、常量比较、合法性确认、错误值统一拒绝”一组？如果是，选 `identifier`。
4. 主导差分是否明确表达“不同取值选择不同模式、命令、状态、handler/case 或 bit 控制位”？如果是，选 `control_or_flags`。
5. 主导差分是否表达“对象/实例/索引/偏移/地址/查表/访问目标定位”？如果是，选 `addressing`。
6. 如果无法判断，选 `other_or_unknown`。

## identifier

当程序行为摘要主要表达以下含义时，优先考虑 `identifier`：

- 字段取值变化导致入口门控、固定值校验、少量候选值校验或合法性确认失败；
- 字段像“必须匹配某个值或某类值”一样被程序使用；
- 错误或非预期取值主要导致拒绝、早退、统一异常路径或不进入后续解析；
- `constraint_value_diversity` 或 `constraint_count` 显示存在候选值约束，同时 mutation 后主要表现为通过/拒绝差异。

不要因为该字段触发了比较、分支或异常路径就自动选 `control_or_flags`。

## length_or_count

当程序行为摘要主要表达以下含义时，优先考虑 `length_or_count`：

- 字段取值变化影响处理规模、输入消费范围、循环次数、重复次数、解析边界、资源使用或路径长短；
- `boundary` 或 `extreme` 组差分显著，并且这种敏感性可以解释为“处理多少、读多少、循环多少、覆盖多大范围”；
- `loop_dispersion` 明显升高，或极端值导致截断、扩张、越界、资源放大、提前停止；
- mutation 主要体现范围、边界、极端值、消费范围、循环/规模变化、长度约束等行为。

若边界/极端差分占主导，且没有明确的离散模式/命令/bit 控制线索，应优先考虑 `length_or_count`，而不是 `control_or_flags`。

## control_or_flags

当程序行为摘要主要表达以下含义时，优先考虑 `control_or_flags`：

- 字段不同离散取值选择不同处理模式、命令、状态、选项、handler、case 或互斥分支；
- bit 或 bit 段被单独提取后作为开关、标志位、模式位或条件控制位使用；
- `enum` 组内部行为分化明显，并且分化更像正常功能/模式选择，而不是合法性通过/拒绝；
- 字段改变的是“走哪类处理逻辑”，而不是“是否通过固定校验”或“处理多少数据”。

严格限制：`control_or_flags` 不是“比较、分支、异常路径、多路径偏离、多类别分化”的默认标签。只有差分摘要清楚表达“离散取值在正常解析中选择不同功能/模式/命令/状态/bit 控制”时，才把它放在首位。

## addressing

当程序行为摘要主要表达以下含义时，优先考虑 `addressing`：

- 字段用于定位对象、索引、偏移、地址、寄存器、表项、实例、通道或集合元素；
- 字段变化导致查表、对象匹配、访问范围、索引范围或目标有效性相关差异；
- 越界或非法定位值导致异常路径，但合法值主要改变访问目标，而不是改变整体处理规模；
- 主导差分更像“访问哪个目标/位置”，而不是“处理多少数据/循环多少次”。

`addressing` 与 `length_or_count` 都可能边界敏感；若没有目标定位线索，不要强行选 `addressing`。

## data_value

当程序行为摘要主要表达以下含义时，优先考虑 `data_value`：

- 字段主要被保存、传播、回写、输出、参与普通计算或在程序后期被消费；
- 字段变化没有清楚地决定入口门控、离散模式、长度规模、索引定位或解析结构；
- 字段像业务参数、测量值、时间值、普通载荷或可传递数据一样被程序使用；
- `relative_start` 较晚、比较/分支信号较弱，或 mutation 主要造成普通数值传播差异。

字段确实被比较或检查，但这些行为不能稳定解释为 identifier、length_or_count、control_or_flags 或 addressing 时，也可以把 `data_value` 作为候选。

## other_or_unknown

当程序行为摘要不足以形成稳定解释时，选择 `other_or_unknown`：

- 差分信号很弱、有效 mutation 较少或不同指标彼此冲突；
- 字段级总结只能写成很泛的“程序行为变化”；
- 语义混合了边界、分支、传播、异常等多个方向，但无法判断主导传统语义；
- 字段可能是透明字段、低价值字段或当前材料无法可靠分类。

不要为了避免 unknown 而强行把首选改成 `control_or_flags`。

# 冲突与不确定性处理

如果差分摘要同时支持多个标签：

1. 优先根据主导差分模式判断；
2. 优先根据字段级程序行为语义判断，而不是逐指标投票；
3. 可以输出多个候选标签；
4. 如果最高优先级标签很不确定，设置 `needs_review=true` 并说明原因。

如果出现以下情况，应设置 `needs_review=true`：

- 关键差分指标互相冲突；
- 字段级行为描述只能写得很泛；
- traditional tag 投影依赖很弱；
- 多个标签几乎同等合理；
- 差分信号太弱。

# 输出字段要求

- `field_program_semantic_summary`：非空字符串；
- `traditional_semantic_tags`：字符串数组，长度 1 到 3；
- `tag_rationale`：简短说明为什么选择这些 traditional tags；
- `needs_review`：布尔值；
- `review_reason`：如果 `needs_review=false`，输出空字符串；如果为 true，简要说明原因。"""

PRINT_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ4-B no-latent direct semantic summary.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--responses-md", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--backend", choices=["api", "codex"], default="api")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--only-field-uid", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def field_uid(row: dict[str, Any]) -> str:
    return f"{row.get('protocol_name', '')}-{row.get('sample_id', '')}-{row.get('field_id', '')}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY_COLS + CONTEXT_COLS + [f"{g}_{f}" for g in GROUPS for f in GROUP_FEATURES])
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        for raw in reader:
            row = dict(raw)
            for col in CONTEXT_COLS:
                row[col] = parse_float(row.get(col))
            for group in GROUPS:
                for feature in GROUP_FEATURES:
                    col = f"{group}_{feature}"
                    row[col] = parse_float(row.get(col))
            row["field_uid"] = field_uid(row)
            rows.append(row)
    return rows


def build_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": {col: row[col] for col in CONTEXT_COLS},
        "mutation_groups": {
            group: {feature: row[f"{group}_{feature}"] for feature in GROUP_FEATURES}
            for group in GROUPS
        },
        "diagnostics": {
            "mutation_count": parse_float(row.get("mutation_count")),
            "valid_mutations": parse_float(row.get("valid_mutations")),
            "unique_metric_vectors": parse_float(row.get("unique_metric_vectors")),
            "deltaf_dispersion": parse_float(row.get("deltaf_dispersion")),
            "constraint_count": parse_float(row.get("constraint_count")),
            "field_kind": str(row.get("field_kind", "")),
        },
    }


def build_prompt(row: dict[str, Any]) -> str:
    payload = json.dumps(build_payload(row), ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.replace("{field_payload_json}", payload)


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


def validate_response(obj: dict[str, Any]) -> dict[str, Any]:
    summary = str(obj.get("field_program_semantic_summary", "")).strip()
    if not summary:
        raise ValueError("missing field_program_semantic_summary")
    tags = normalize_tags(obj.get("traditional_semantic_tags")) or ["other_or_unknown"]
    return {
        "field_program_semantic_summary": summary,
        "traditional_semantic_tags": tags[:3],
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
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout_file, stderr=stderr_file, text=True)
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
    return call_api(prompt, args) if args.backend == "api" else call_codex(prompt, args)


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row.get("field_uid", "")).strip()
            if uid:
                done.add(uid)
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_response_md(path: Path, uid: str, prompt: str, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        exists = path.exists()
        with path.open("a", encoding="utf-8") as handle:
            if not exists:
                handle.write("# RQ4-B No-Latent Direct Prompt Responses\n\n")
            handle.write(f"## {uid}\n\n### Prompt\n\n```text\n{prompt}\n```\n\n")
            handle.write(f"### Response\n\n```text\n{response}\n```\n\n")


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    if not jsonl_path.exists():
        return
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
    ]
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            if isinstance(out["traditional_semantic_tags"], list):
                out["traditional_semantic_tags"] = ";".join(out["traditional_semantic_tags"])
            writer.writerow(out)


def process_one(row: dict[str, Any], args: argparse.Namespace, index: int, total: int) -> dict[str, Any]:
    uid = str(row["field_uid"])
    prompt = build_prompt(row)
    if args.dry_run:
        append_response_md(args.responses_md, uid, prompt, "[dry-run skipped]")
        return {"field_uid": uid, "dry_run": True}
    attempt = 0
    while True:
        attempt += 1
        start = time.time()
        print_line(f"[rq4-no-latent] start {index}/{total} {uid} attempt={attempt}")
        try:
            response = call_llm(prompt, args)
            parsed = validate_response(extract_json_object(response))
            out = {
                "protocol_name": row.get("protocol_name", ""),
                "sample_id": row.get("sample_id", ""),
                "field_id": row.get("field_id", ""),
                "field_uid": uid,
                **parsed,
            }
            append_jsonl(args.output_jsonl, out)
            append_response_md(args.responses_md, uid, prompt, response)
            elapsed = format_duration(time.time() - start)
            print_line(f"[rq4-no-latent] done {index}/{total} {uid} elapsed={elapsed}")
            log_line(args.run_log, f"done field_uid={uid} attempt={attempt} elapsed={elapsed}")
            return out
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            message = f"error field_uid={uid} attempt={attempt}: {exc}"
            print_line(f"[rq4-no-latent] {message}; retry in {args.retry_delay_seconds}s")
            log_line(args.run_log, message)
            time.sleep(args.retry_delay_seconds)


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    rows = load_rows(args.input)
    if args.only_field_uid:
        rows = [row for row in rows if str(row["field_uid"]) == args.only_field_uid]
    else:
        rows = rows[max(args.start_index - 1, 0):]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.overwrite:
        args.output_jsonl.unlink(missing_ok=True)
        args.output_csv.unlink(missing_ok=True)
        args.responses_md.unlink(missing_ok=True)
    else:
        done = load_done(args.output_jsonl)
        rows = [row for row in rows if str(row["field_uid"]) not in done]
    total = len(rows)
    log_line(args.run_log, f"start total={total} input={args.input}")
    print_line(f"[rq4-no-latent] pending={total} backend={args.backend} workers={args.workers}")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, row, args, index, total) for index, row in enumerate(rows, start=1)]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        log_line(args.run_log, "interrupted by user")
        raise
    write_csv_from_jsonl(args.output_jsonl, args.output_csv)
    log_line(args.run_log, f"finished total={total} output={args.output_jsonl}")
    print_line(f"[rq4-no-latent] finished output={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
