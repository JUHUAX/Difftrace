#!/usr/bin/env python3
"""Generate Stage 4 latent-axis naming prompts.

Input is the computed Stage 4A top-k probe evidence. The main Stage 4 path uses
only latent behavior naming. The old per-axis coarse projection mode is kept as
a compatibility path, but field-level traditional tags should now be produced by
run_stage4_field_semantic_fusion.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming/z_topk_probe_evidence.json")
DEFAULT_OUT = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming/z_llm_prompt_responses.md")
DEFAULT_SEMANTICS_OUT = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming/z_axis_semantics.json")
DEFAULT_COARSE_OUT = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming/z_coarse_projection_prompt_responses.md")
DEFAULT_COARSE_TAGS_OUT = Path("/root/semvec/difftrace/stage4/out/stage4_latent_naming/z_axis_coarse_tags.json")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
COARSE_FIELD_SEMANTIC_TAGS = [
    "identifier",
    "length_or_count",
    "control_or_flags",
    "addressing",
    "data_value",
    "other_or_unknown",
]

# Fill this value if you prefer configuring the key inside the script.
# If left empty, the script falls back to --api-key, DEEPSEEK_API_KEY, then OPENAI_API_KEY.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:05.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:05.2f}s"


BACKGROUND = """\
我们正在做一个协议字段语义分析流程。

整体链路如下：

1. Stage 1: 字段恢复。
   系统先重放原始协议报文，通过程序执行污点日志观察每个输入字节或 bit 如何被程序消费，
   从而恢复字段边界。这里得到的是“程序实际如何使用字段”，而不是协议规范里的人工模板。

2. Stage 2: 字段级 mutation 与执行差分。
   对 Stage 1 得到的每个字段，系统会构造多组 mutation 值并重新执行程序。
   mutation 的目的不是随机破坏报文，而是用不同取值策略主动探测字段背后的程序行为。
   当前使用四个策略组：
   - neighborhood：选择字段原值附近的小幅扰动，用于探测局部连续取值、数值敏感性和局部处理区域。
   - boundary：选择最小值、最大值和边界附近值，用于探测范围检查、边界分支、错误路径和阈值行为。
   - enum：选择候选枚举值、约束值或离散代表值，用于探测 handler/case 分派和离散语义类别。
   - extreme：选择极端值或异常值，用于探测异常路径、拒绝路径、资源放大、循环变化或长度爆炸。
   每个 mutation 执行后，系统会把 mutation 执行日志与 baseline 执行日志比较，得到程序行为差分指标。
   这些差分指标描述字段取值变化后，程序执行路径、比较行为、循环行为、处理强度等是否发生变化。

3. Stage 3: 字段表示空间学习。
   对每个字段，系统把 Stage 2 中每个策略组的多次 mutation 差分结果压缩成组级摘要指标。
   每个策略组产生 6 个摘要维度：
   - mean_baseline_distance：该组 mutation 相对 baseline 的平均偏离程度。
   - mean_pairwise_distance：该组 mutation 之间是否触发多种处理状态或执行路径。
   - max_pairwise_distance：该组 mutation 内最强的一次处理分歧。
   - metric_vector_variance：程序对该组不同 mutation 的响应是否稳定，是否只对少数值高度敏感。
   - unique_vector_ratio：该组取值是否被程序划分成多个行为等价类。
   - loop_dispersion：该组取值是否影响循环、批量处理、重复执行或资源相关行为。
   四个策略组共产生 24 维摘要指标。
   另外加入 4 维上下文向量：
   - relative_start：字段首次被消费的位置；数值越大表示字段首次被程序消费得越晚。
   - field_instr_ratio：字段相关指令在 baseline 中的占比；数值越大表示程序对该字段投入的处理越多。
   - compare_ratio：字段相关比较行为比例；数值越大表示字段越多参与条件判断、约束检查或分支决策。
   - constraint_value_diversity：字段约束值多样性；数值越大表示程序能区分出的候选值或约束值越丰富。
   因此，每个字段最终得到 28 维程序行为探针向量。
   随后系统训练 AutoEncoder，把 28 维程序行为探针压缩为 8 维 latent 表示 z1...z8。

4. Stage 4: latent 维度命名。
   现在要做的不是从原始 trace 重新推断字段语义，也不是给字段套传统协议标签。
   目标是根据每个 z 维与 28 维程序行为探针的相关性，给 z 维生成一个可读、受证据约束的程序行为语义名称。
   这个名称应描述“字段变化会让程序表现出什么行为”，而不是描述统计分布或协议特定字段名。
"""

TASK = """\
你将看到一个 AE 隐空间维度与若干 28 维程序行为探针之间的相关性结果。

系统已经计算了当前 z 维和每个程序行为探针之间的相关性，并分别从正相关和负相关方向选择 top-k 探针。
这些探针已经经过相关性门槛筛选：只有 abs(correlation) >= correlation_threshold 的探针才会进入输入。

你的任务不是重新分析 trace，也不是推断协议规范字段类型。
你的任务是根据当前 z 维相关的几个程序行为探针，概括该 z 维高值端和低值端分别表达的字段程序行为语义。

请根据给定的 positive_evidence 和 negative_evidence，分别为该 z 维的 high_value 端和 low_value 端生成：
- latent_name
- definition
- evidence
- confidence

最后生成一句 axis_summary，概括 high_value 与 low_value 的双端对比关系。
"""

INPUT_NOTES = """\
输入 key 含义：
- axis：当前 latent 维度，例如 z1。
- positive_evidence：与该 z 维正相关的 probe；表示 z 高值端具备这些 probe 的 probe_high_value_meaning。
- negative_evidence：与该 z 维负相关的 probe；表示 z 低值端具备这些 probe 的 probe_high_value_meaning。
- evidence_strength：该端通过相关性阈值的 probe 数量强度，取值为 strong / medium / weak / insufficient。
- correlation：z 维与 probe 的相关系数。
- probe_meaning：probe 的总体程序行为语义。
- probe_high_value_meaning：该 probe 数值越大时对应的程序行为。

关键方向规则：
- positive_evidence 用 probe_high_value_meaning 解释 z 高值端。
- negative_evidence 也用 probe_high_value_meaning 解释 z 低值端。
- 不要把负相关解释为“没有该行为”。
"""

STRICT_RULES = [
    "只能使用下方给定的 evidence，不允许引入输入 probe 之外的新语义概念。",
    "不允许使用协议特定字段类型作为 latent 名称。",
    "不允许使用“变化轴”“响应轴”“分布模式”“相关性模式”等统计分析式表述。",
    "high_value 只能主要根据 positive_evidence 生成。",
    "low_value 只能主要根据 negative_evidence 生成。",
    "definition 必须基于对应方向 evidence 的 probe_high_value_meaning。",
    "如果 evidence_strength 为 insufficient，该端 latent_name 必须输出 \"insufficient evidence\"，definition 只能说明证据不足。",
    "如果 evidence_strength 为 weak，该端只能做低置信单 probe 解释，confidence 必须为 low。",
    "只有 evidence_strength 为 medium 或 strong 时，才允许给出正常程序行为名称。",
    "不同 z 维不要求具有不同名称；如果 evidence 相似，可以给出相似或相同名称。",
    "不允许为了让维度显得独立而编造细微语义差异。",
]

LATENT_NAME_RULES = [
    "必须是“字段 + 动词 + 程序行为对象”的短语。",
    "优先使用程序动作词，例如：触发、参与、驱动、进入、选择、分派、区分、检查、约束、解析、消费。",
    "避免使用指标化/统计化名词，例如：敏感度、丰富度、分化度、扰动度、偏离度、强度、比例、分布、模式。",
    "不要把 probe 名或指标名直接改写成名称。",
    "名称应描述字段在程序中造成或参与的行为，而不是描述数值大小。",
]

LATENT_NAME_BAD_EXAMPLES = [
    "字段行为扰动敏感度",
    "字段全局行为偏离度",
    "字段离散类别丰富度",
    "字段边界极端处理分歧度",
]

LATENT_NAME_GOOD_EXAMPLES = [
    "字段触发多路径处理",
    "字段参与条件分支",
    "字段驱动异常处理",
    "字段选择离散处理分支",
    "字段参与边界检查",
    "字段被程序多类别区分",
    "字段后期参与约束检查",
]

COARSE_TAG_DESCRIPTIONS = {
    "identifier": "事务号、对象 ID、会话 ID、序列号、引用号等用于标识实体或关联请求响应的字段。",
    "length_or_count": "长度、数量、计数、数组元素个数、剩余长度等控制数据规模或重复次数的字段。",
    "control_or_flags": "类型、功能码、命令码、状态码、控制位、选项位、布尔标志等影响分支或处理模式的字段。",
    "addressing": "源/目的地址、设备地址、寄存器地址、对象地址、偏移或索引等定位目标的字段。",
    "data_value": "测量值、参数值、时间值、载荷数据、业务数值等承载应用数据的字段。",
    "other_or_unknown": "保留、填充、未知、证据不足或无法可靠归入其他类别的字段。",
}

COARSE_DIFFERENTIAL_SIGNATURES = """\
## identifier

semantic intuition:
- identifier 更接近“常量/标识校验字段”：程序通常期望它等于某个正确值或少数合法值。
- 它可用于报文识别、事务/对象/会话标识、协议常量或魔数式校验。

expected differential signature:
- 字段一般较早被消费，因为程序需要先确认它是否匹配预期标识。
- mutation 与 baseline 可能明显不同，因为错误 identifier 会导致拒绝、提前退出或进入统一错误处理。
- 多个 mutation 之间可能彼此相似，因为不同错误值都进入相近的拒绝路径。
- 因此可能出现：mean_baseline_distance 较高，但 mean_pairwise_distance、unique_vector_ratio 或组内分化不一定高。

key discriminators:
- 透明字段的 mutation 与 baseline 基本一致；identifier 的 mutation 彼此一致，但整体偏离 baseline。
- control_or_flags 更像不同取值选择不同处理模式；identifier 更像错误取值统一不通过校验。
- 如果 evidence 显示 mutation 之间行为相似，但都远离 baseline，应优先考虑 identifier。

## length_or_count

semantic intuition:
- length_or_count 表示字段值控制后续处理规模，例如长度、数量、重复次数、数组元素数或剩余输入预算。

expected differential signature:
- 字段值变化倾向于改变后续执行规模、循环次数、输入消费范围或解析边界。
- boundary / extreme 组可能比 neighborhood 组更容易触发明显差异。
- loop_dispersion、与执行规模相关的差分、或大范围 mean_baseline_distance 可能更有指示性。
- 极端值可能导致提前拒绝、截断、资源放大或路径缩短/拉长。

key discriminators:
- 如果字段变化主要改变“处理多少数据”或“重复多少次”，应优先考虑 length_or_count，而不是 control_or_flags。
- length_or_count 应更明显影响执行规模、循环/批处理或输入消费范围；addressing 可能边界敏感，但不应显著改变规模。

## control_or_flags

semantic intuition:
- control_or_flags 表示字段值选择处理模式、状态、命令、功能、选项或开关路径。

expected differential signature:
- enum 组、候选离散值或不同取值可能触发多个行为等价类。
- unique_vector_ratio、mean_pairwise_distance 或多类别分化相关 probe 可能较强。
- 多个合法或候选取值之间的程序响应可能不同，而不只是全部错误值进入同一路径。
- compare_ratio 可作为辅助证据，但不能单独决定该标签。

key discriminators:
- 不能仅因为字段参与比较、约束或分支就选择 control_or_flags。
- 只有当差分证据更像“字段取值选择程序如何处理”，而不是“字段取值表示长度、地址、身份或数据值”时，才优先考虑 control_or_flags。
- 如果 evidence 只说明 mutation 与 baseline 差异大，但没有多类别/离散处理分派迹象，应避免选择 control_or_flags。

## addressing

semantic intuition:
- addressing 更像 identifier 与 length_or_count 的交叉。
- 它和 identifier 相似：字段值需要通过合法性检查，非法值会走错误处理、拒绝路径或提前退出。
- 它和 identifier 的区别是：合法值通常不是单个常量或少数枚举，而是一段范围。
- 它和 length_or_count 相似：合法值具有范围属性，边界值和越界值更容易触发差异。
- 它和 length_or_count 的区别是：合法范围内取值通常不应明显改变程序执行规模、循环次数或输入消费量。

expected differential signature:
- mutation 可能表现出范围/边界敏感：合法范围内变化相对稳定，越界或极端值明显偏离 baseline。
- 多个越界 mutation 之间可能行为相似，因为它们都进入相近错误路径；这一点接近 identifier。
- 与 length_or_count 不同，合法范围内取值不应显著改变循环次数、批量处理长度、资源规模或输入消费总量。
- 可能出现 boundary / extreme 响应较强，但 loop_dispersion 或规模型差异不强。

key discriminators:
- identifier 更像常量/少量合法值校验；addressing 更像范围合法性校验。
- length_or_count 更明显改变执行规模、循环次数、批量处理长度或输入消费范围；addressing 不应主要改变这些规模因素。
- 如果 evidence 显示字段对边界/范围敏感，但没有明显改变处理规模，应优先考虑 addressing，而不是 length_or_count。

## data_value

semantic intuition:
- data_value 表示字段主要承载业务数据、测量值、参数值、时间值或 payload 内容。

expected differential signature:
- data_value 的差分特点通常是字段值变化不会显著触发约束、比较、分派、边界检查或执行规模变化。
- 它可能被程序消费、保存、传递或参与普通数据处理，但不会明显决定解析结构或控制流。
- 它一般更可能在程序执行后期被消费，或在前置解析、长度检查、类型分派完成后才作为业务内容被读取。
- compare_ratio、constraint_value_diversity、多类别分化、边界异常响应、loop_dispersion 通常不应同时强。
- mutation 与 baseline 的差异可能较弱，或者只表现为局部数据处理差异。

key discriminators:
- 如果 mutation 与 baseline 的差异较弱，且缺少 compare/constraint/dispatch/loop 等强语义证据，应优先考虑 data_value 或 other_or_unknown。
- 如果字段变化主要体现为“数据内容发生变化”，而不是“程序选择不同处理方式”，应优先考虑 data_value。
- 如果字段值变化明显选择不同 handler、状态或命令路径，则更可能是 control_or_flags。

## other_or_unknown

semantic intuition:
- other_or_unknown 表示证据不足、字段语义弱、字段透明，或无法可靠归入其他类别。

expected differential signature:
- 差分证据不足；
- 字段变化对程序行为影响很弱或不稳定；
- mutation 与 baseline 可能非常接近；
- 或 top-k evidence 混杂，无法形成稳定标签判断。

key discriminators:
- 如果字段 mutation 与 baseline 基本一致，优先考虑 other_or_unknown 或透明/低价值字段，而不是 identifier。
- 如果字段证据强但难以区分具体传统标签，可以保留 other_or_unknown 作为第二候选，而不是强行选择 control_or_flags。
"""

ANALYSIS_GUIDE = [
    "分别处理 high_value 端和 low_value 端；两端使用同一套分析流程。",
    "先查看该端的 evidence_strength；如果为 insufficient 或 weak，按低证据规则处理。",
    "查看该端对应的 evidence：high_value 使用 positive_evidence，low_value 使用 negative_evidence。",
    "基于该端 evidence 中的 probe_high_value_meaning 解释该端程序行为。",
    "根据该端程序行为解释生成 definition。",
    "根据 definition 压缩出 latent_name。",
    "最后根据 high_value 和 low_value 的定义生成一句 axis_summary。",
]


OUTPUT_SCHEMA = {
    "axis": "...",
    "high_value": {
        "evidence_strength": "strong|medium|weak|insufficient",
        "evidence_count": 0,
        "correlation_threshold": 0.0,
        "latent_name": "...",
        "definition": "...",
        "evidence": [
            {
                "probe": "...",
                "direction": "positive",
                "correlation": 0.0,
                "high_value_meaning": "...",
                "used_for_name": True,
            }
        ],
        "confidence": "high|medium|low",
    },
    "low_value": {
        "evidence_strength": "strong|medium|weak|insufficient",
        "evidence_count": 0,
        "correlation_threshold": 0.0,
        "latent_name": "...",
        "definition": "...",
        "evidence": [
            {
                "probe": "...",
                "direction": "negative",
                "correlation": 0.0,
                "high_value_meaning": "...",
                "used_for_name": True,
            }
        ],
        "confidence": "high|medium|low",
    },
    "axis_summary": "...",
    "notes": "...",
}

COARSE_OUTPUT_SCHEMA = {
    "axis": "...",
    "high_value": {
        "possible_field_semantics": ["control_or_flags"],
        "differential_signature_match": "...",
        "tag_reason": "...",
    },
    "low_value": {
        "possible_field_semantics": ["identifier", "control_or_flags"],
        "differential_signature_match": "...",
        "tag_reason": "...",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 4B prompts and query an LLM.")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--semantics-out",
        type=Path,
        default=DEFAULT_SEMANTICS_OUT,
        help="Structured z-axis semantics JSON output. Not written in --dry-run mode.",
    )
    parser.add_argument(
        "--coarse-out",
        type=Path,
        default=DEFAULT_COARSE_OUT,
        help="Coarse projection prompt/response Markdown output.",
    )
    parser.add_argument(
        "--coarse-tags-out",
        type=Path,
        default=DEFAULT_COARSE_TAGS_OUT,
        help="Structured z-axis coarse semantic tags JSON output. Not written in --dry-run mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "naming", "coarse"],
        default="naming",
        help="Run latent naming, legacy coarse projection, or both. Default: naming.",
    )
    parser.add_argument(
        "--no-semantics-out",
        action="store_true",
        help="Do not write the structured z-axis semantics JSON.",
    )
    parser.add_argument(
        "--strict-json",
        dest="strict_json",
        action="store_true",
        default=True,
        help="Fail if any LLM response cannot be parsed as JSON. Default: true.",
    )
    parser.add_argument(
        "--no-strict-json",
        dest="strict_json",
        action="store_false",
        help="Continue even if an LLM response cannot be parsed as JSON.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to in-script DEEPSEEK_API_KEY, then env DEEPSEEK_API_KEY, then env OPENAI_API_KEY.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature. Default 0.0 for more deterministic naming.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling top_p. Default 1.0; use a lower value only if the provider supports it well.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for providers that support seeded generation.",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="enabled",
        help="DeepSeek thinking mode passed through extra_body.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only render prompts; do not call the API.",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Use one cumulative conversation across axes. Default is one independent request per axis.",
    )
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        starts = [match.start() for match in re.finditer(r"\{", stripped)]
        for start in reversed(starts):
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


def compact_side(side: dict[str, Any]) -> dict[str, Any]:
    return {
        "latent_name": side.get("latent_name", ""),
        "definition": side.get("definition", ""),
        "confidence": side.get("confidence", ""),
    }


def normalize_coarse_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        candidates = []
    normalized: list[str] = []
    for candidate in candidates:
        if candidate in COARSE_FIELD_SEMANTIC_TAGS and candidate not in normalized:
            normalized.append(candidate)
    return normalized[:2] or ["other_or_unknown"]


def compact_axis_semantics(axis: str, response_text: str) -> dict[str, Any]:
    parsed = extract_json_object(response_text)
    high_value = parsed.get("high_value", {})
    low_value = parsed.get("low_value", {})
    if not isinstance(high_value, dict) or not isinstance(low_value, dict):
        raise ValueError(f"{axis}: high_value and low_value must be JSON objects")
    return {
        "axis": parsed.get("axis", axis),
        "high_value": compact_side(high_value),
        "low_value": compact_side(low_value),
        "axis_summary": parsed.get("axis_summary", ""),
        "notes": parsed.get("notes", ""),
    }


def compact_coarse_side(side: dict[str, Any]) -> dict[str, Any]:
    return {
        "possible_field_semantics": normalize_coarse_tags(side.get("possible_field_semantics")),
        "differential_signature_match": side.get("differential_signature_match", ""),
        "tag_reason": side.get("tag_reason", ""),
    }


def compact_axis_coarse_tags(axis: str, response_text: str) -> dict[str, Any]:
    parsed = extract_json_object(response_text)
    high_value = parsed.get("high_value", {})
    low_value = parsed.get("low_value", {})
    if not isinstance(high_value, dict) or not isinstance(low_value, dict):
        raise ValueError(f"{axis}: high_value and low_value must be JSON objects")
    return {
        "axis": parsed.get("axis", axis),
        "high_value": compact_coarse_side(high_value),
        "low_value": compact_coarse_side(low_value),
    }


def format_probe(item: dict[str, Any], direction: str) -> str:
    corr = item.get("correlation")
    corr_text = "null" if corr is None else f"{corr:.6f}"
    return (
        f"- probe: `{item['probe']}`\n"
        f"  - direction: {direction}\n"
        f"  - correlation_metric: {item.get('correlation_metric')}\n"
        f"  - correlation: {corr_text}\n"
        f"  - feature_group: {item.get('feature_group')}\n"
        f"  - predefined_meaning: {item.get('probe_meaning')}\n"
        f"  - high_value_meaning: {item.get('probe_high_value_meaning')}"
    )


def render_prompt(axis_obj: dict[str, Any]) -> str:
    axis = axis_obj["axis"]
    threshold = axis_obj.get("correlation_threshold")
    lines: list[str] = []
    lines.append(f"# 背景")
    lines.append("")
    lines.append(BACKGROUND)
    lines.append("")
    lines.append("# 任务")
    lines.append("")
    lines.append(f"当前 latent 维度：`{axis}`")
    lines.append("")
    lines.append(TASK)
    lines.append("")
    lines.append("# 分析引导")
    lines.append("")
    for index, step in enumerate(ANALYSIS_GUIDE, start=1):
        lines.append(f"{index}. {step}")
    lines.append("")
    lines.append("注意：分析引导只用于约束你的生成过程，不要把推理过程写入输出。")
    lines.append("")
    lines.append("# 输入说明")
    lines.append("")
    lines.append(INPUT_NOTES)
    lines.append("")
    lines.append("# 严格规则")
    lines.append("")
    for rule in STRICT_RULES:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("# latent_name 命名规范")
    lines.append("")
    lines.append("latent_name 风格要求：")
    for rule in LATENT_NAME_RULES:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("不推荐：")
    for example in LATENT_NAME_BAD_EXAMPLES:
        lines.append(f"- {example}")
    lines.append("")
    lines.append("推荐：")
    for example in LATENT_NAME_GOOD_EXAMPLES:
        lines.append(f"- {example}")
    lines.append("")
    lines.append("# Evidence")
    lines.append("")
    lines.append("## positive_evidence（high_value 端）")
    lines.append("")
    lines.append(f"- evidence_strength: {axis_obj.get('positive_evidence_strength', 'unknown')}")
    lines.append(f"- evidence_count: {axis_obj.get('positive_evidence_count', len(axis_obj.get('positive_evidence', [])))}")
    lines.append(f"- correlation_threshold: {threshold}")
    for item in axis_obj.get("positive_evidence", []):
        lines.append(format_probe(item, "positive"))
    if not axis_obj.get("positive_evidence"):
        lines.append("- none")
    lines.append("")
    lines.append("## negative_evidence（low_value 端）")
    lines.append("")
    lines.append(f"- evidence_strength: {axis_obj.get('negative_evidence_strength', 'unknown')}")
    lines.append(f"- evidence_count: {axis_obj.get('negative_evidence_count', len(axis_obj.get('negative_evidence', [])))}")
    lines.append(f"- correlation_threshold: {threshold}")
    for item in axis_obj.get("negative_evidence", []):
        lines.append(format_probe(item, "negative"))
    if not axis_obj.get("negative_evidence"):
        lines.append("- none")
    lines.append("")
    lines.append("# 输出格式")
    lines.append("")
    lines.append("请只输出 JSON，不要输出额外解释。")
    lines.append("evidence 中必须保留用于命名 probe 的 high_value_meaning。")
    lines.append("输出格式如下：")
    lines.append(json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def projection_background() -> str:
    old = """4. Stage 4: latent 维度命名。
   现在要做的不是从原始 trace 重新推断字段语义，也不是给字段套传统协议标签。
   目标是根据每个 z 维与 28 维程序行为探针的相关性，给 z 维生成一个可读、受证据约束的程序行为语义名称。
   这个名称应描述“字段变化会让程序表现出什么行为”，而不是描述统计分布或协议特定字段名。"""
    new = """4. Stage 4: latent 维度与传统字段语义的投影。
   前一步已经根据每个 z 维与 28 维程序行为探针的相关性，生成了每个 z 维 high_value / low_value 端的程序行为语义名称和定义。
   现在要做的不是重新命名 z 轴，也不是从原始 trace 推断字段语义。
   当前任务是：根据 z 端的程序行为语义、top-k probe evidence，以及 coarse semantic tags 的 differential signature table，
   将每个 z 端投影到最相关的一个或多个传统粗粒度字段语义标签。"""
    return BACKGROUND.replace(old, new)


def format_axis_semantics(axis_semantics: dict[str, Any]) -> str:
    lines: list[str] = []
    for side_name in ["high_value", "low_value"]:
        side = axis_semantics.get(side_name, {})
        lines.append(f"## {side_name}")
        lines.append(f"- latent_name: {side.get('latent_name', '')}")
        lines.append(f"- definition: {side.get('definition', '')}")
        lines.append(f"- confidence: {side.get('confidence', '')}")
        lines.append("")
    lines.append(f"- axis_summary: {axis_semantics.get('axis_summary', '')}")
    if axis_semantics.get("notes"):
        lines.append(f"- notes: {axis_semantics.get('notes')}")
    return "\n".join(lines)


def render_coarse_projection_prompt(axis_obj: dict[str, Any], axis_semantics: dict[str, Any]) -> str:
    axis = axis_obj["axis"]
    lines: list[str] = []
    lines.append("# 背景")
    lines.append("")
    lines.append(projection_background())
    lines.append("")
    lines.append("# 任务")
    lines.append("")
    lines.append(f"当前 latent 维度：`{axis}`")
    lines.append("")
    lines.append("你将看到一个 AE 隐空间维度的 high_value / low_value 语义、对应的 top-k probe evidence，以及 6 个 coarse semantic tags 的 differential signature table。")
    lines.append("你的任务是根据 differential signature table，为该 z 维的 high_value 端和 low_value 端分别选择最相关的 coarse semantic tags。")
    lines.append("")
    lines.append("允许的 coarse semantic tags 只有：")
    for tag in COARSE_FIELD_SEMANTIC_TAGS:
        lines.append(f"- {tag}")
    lines.append("")
    lines.append("每个 high_value / low_value 端最多选择 2 个标签，优先选择 1 个最相关标签。")
    lines.append("你不需要生成 latent_name，不需要重写 definition，也不需要分析原始 trace。")
    lines.append("你只能基于输入中的 latent behavior semantics、probe evidence 和 differential signature table 做投影。")
    lines.append("")
    lines.append("# 分析引导")
    lines.append("")
    guide = [
        "读取当前 z 端的 latent behavior name / definition。",
        "读取该 z 端对应的 top-k probe evidence。",
        "根据 probe evidence 判断该 z 端更接近哪种 differential signature：常量/标识校验、执行规模或循环/消费范围变化、离散处理模式选择、范围合法性但不改变规模、数据值消费且缺少强控制证据、或证据不足。",
        "再根据 differential signature 选择 coarse semantic tags。",
        "如果 evidence 同时支持多个标签，最多保留 2 个。",
        "如果只是“参与比较/分支/约束”，不能自动选择 control_or_flags，必须判断该比较或差分响应更像哪种字段作用。",
    ]
    for index, step in enumerate(guide, start=1):
        lines.append(f"{index}. {step}")
    lines.append("")
    lines.append("注意：分析引导只用于约束你的生成过程，不要把完整推理过程写入输出。")
    lines.append("")
    lines.append("# 输入说明")
    lines.append("")
    lines.append("- axis：当前 latent 维度，例如 z1。")
    lines.append("- high_value / low_value：该 z 维高值端和低值端的程序行为语义。")
    lines.append("- latent_name：前一步已经生成的 z 端程序行为语义名称。")
    lines.append("- definition：前一步已经生成的 z 端程序行为定义。")
    lines.append("- confidence：前一步 latent behavior naming 的置信度。")
    lines.append("- positive_evidence：与该 z 维正相关的 probe；用于 high_value 端。")
    lines.append("- negative_evidence：与该 z 维负相关的 probe；用于 low_value 端。")
    lines.append("- probe_high_value_meaning：该 probe 数值越大时对应的程序行为。")
    lines.append("- correlation：z 维与 probe 的相关系数。")
    lines.append("- feature_group：probe 所属策略组或上下文组。")
    lines.append("")
    lines.append("方向规则：")
    lines.append("- high_value 端主要使用 positive_evidence。")
    lines.append("- low_value 端主要使用 negative_evidence。")
    lines.append("- negative_evidence 表示 z 低值端具备该 probe 的 probe_high_value_meaning。")
    lines.append("- 不要把负相关解释为“没有该行为”。")
    lines.append("")
    lines.append("# Coarse Semantic Tags Differential Signatures")
    lines.append("")
    lines.append(COARSE_DIFFERENTIAL_SIGNATURES)
    lines.append("")
    lines.append("# 严格规则")
    lines.append("")
    lines.append("- 只能从 6 个 coarse semantic tags 中选择。")
    lines.append("- 每个 high_value / low_value 端最多选择 2 个标签，优先选择 1 个。")
    lines.append("- 不能因为 evidence 提到比较、分支、约束，就自动选择 control_or_flags。")
    lines.append("- 必须先判断 differential signature，再选择 coarse tag。")
    lines.append("- 不允许使用协议字段名、原始 trace、tshark 字段名或源码上下文。")
    lines.append("- 如果证据不足或 signature 混杂，允许输出 other_or_unknown。")
    lines.append("- 不要重新生成 latent_name 或 definition。")
    lines.append("- 输出必须是 JSON，不要输出额外解释。")
    lines.append("")
    lines.append("# Evidence")
    lines.append("")
    lines.append("## latent_behavior_semantics")
    lines.append("")
    lines.append(format_axis_semantics(axis_semantics))
    lines.append("")
    lines.append("## positive_evidence（high_value 端）")
    lines.append("")
    for item in axis_obj.get("positive_evidence", []):
        lines.append(format_probe(item, "positive"))
    if not axis_obj.get("positive_evidence"):
        lines.append("- none")
    lines.append("")
    lines.append("## negative_evidence（low_value 端）")
    lines.append("")
    for item in axis_obj.get("negative_evidence", []):
        lines.append(format_probe(item, "negative"))
    if not axis_obj.get("negative_evidence"):
        lines.append("- none")
    lines.append("")
    lines.append("# 输出格式")
    lines.append("")
    lines.append("请只输出 JSON，不要输出额外解释。")
    lines.append("输出格式如下：")
    lines.append(json.dumps(COARSE_OUTPUT_SCHEMA | {"axis": axis}, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def get_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("missing API key: fill DEEPSEEK_API_KEY in the script, set env DEEPSEEK_API_KEY, or pass --api-key")
    return api_key


def call_llm(client: Any, args: argparse.Namespace, messages: list[dict[str, Any]]) -> tuple[str | None, str]:
    request_kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": {"thinking": {"type": args.thinking}},
    }
    if args.seed is not None:
        request_kwargs["seed"] = args.seed
    response = client.chat.completions.create(
        **request_kwargs,
    )
    message = response.choices[0].message
    reasoning_content = getattr(message, "reasoning_content", None)
    content = message.content or ""
    return reasoning_content, content


def start_live_timer(axis_label: str, axis_start: float, total_start: float, interval: float = 1.0) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def run() -> None:
        while not stop_event.wait(interval):
            axis_elapsed = time.perf_counter() - axis_start
            total_elapsed = time.perf_counter() - total_start
            message = (
                f"\r[stage4b] {axis_label} waiting | "
                f"axis={format_duration(axis_elapsed)} total={format_duration(total_elapsed)}"
            )
            sys.stdout.write(message)
            sys.stdout.flush()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop_event, thread


def stop_live_timer(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=2.0)
    sys.stdout.write("\r")
    sys.stdout.flush()


def append_prompt_block(lines: list[str], axis: str, prompt: str, reasoning: str | None, response: str | None) -> None:
    lines.append(f"## {axis}")
    lines.append("")
    lines.append("### Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(prompt)
    lines.append("```")
    lines.append("")
    if reasoning is not None:
        lines.append("### Reasoning Content")
        lines.append("")
        lines.append("```text")
        lines.append(reasoning)
        lines.append("```")
        lines.append("")
    lines.append("### Response")
    lines.append("")
    lines.append("```json")
    lines.append(response if response is not None else "")
    lines.append("```")
    lines.append("")


def load_evidence(path: Path) -> list[dict[str, Any]]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, list):
        raise SystemExit(f"{path} must contain a JSON list")
    return evidence


def load_axis_semantics(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    axes = data.get("axes")
    if not isinstance(axes, list):
        raise SystemExit(f"{path} must contain an 'axes' list")
    return axes


def axes_by_name(axes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["axis"]: item for item in axes if isinstance(item, dict) and item.get("axis")}


def make_report_header(title: str, description: str, args: argparse.Namespace) -> list[str]:
    return [
        title,
        "",
        description,
        "",
        "## Run Config",
        "",
        f"- evidence: `{args.evidence}`",
        f"- model: `{args.model}`",
        f"- base_url: `{args.base_url}`",
        f"- reasoning_effort: `{args.reasoning_effort}`",
        f"- temperature: `{args.temperature}`",
        f"- top_p: `{args.top_p}`",
        f"- seed: `{args.seed}`",
        f"- thinking: `{args.thinking}`",
        f"- dry_run: `{args.dry_run}`",
        f"- cumulative: `{args.cumulative}`",
        f"- mode: `{args.mode}`",
        "",
    ]


def get_client(args: argparse.Namespace) -> Any:
    if args.dry_run:
        return None
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("missing Python package: install openai to call the LLM API") from exc
    return OpenAI(api_key=get_api_key(args), base_url=args.base_url)


def run_prompt_batch(
    *,
    args: argparse.Namespace,
    client: Any,
    evidence: list[dict[str, Any]],
    total_start: float,
    stage_label: str,
    lines: list[str],
    prompt_builder: Any,
    parser: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    cumulative_messages: list[dict[str, Any]] = []
    parsed_items: list[dict[str, Any]] = []
    json_errors: list[str] = []
    for index, axis_obj in enumerate(evidence, start=1):
        axis_start = time.perf_counter()
        axis = axis_obj["axis"]
        elapsed_before = axis_start - total_start
        print(
            f"[{stage_label}] ({index}/{len(evidence)}) {axis} start | elapsed={format_duration(elapsed_before)}",
            flush=True,
        )
        prompt = prompt_builder(axis_obj)
        reasoning_content: str | None = None
        content: str | None = None
        if not args.dry_run:
            axis_label = f"({index}/{len(evidence)}) {axis}"
            timer_stop, timer_thread = start_live_timer(axis_label, axis_start, total_start)
            if args.cumulative:
                try:
                    cumulative_messages.append({"role": "user", "content": prompt})
                    reasoning_content, content = call_llm(client, args, cumulative_messages)  # type: ignore[arg-type]
                    cumulative_messages.append({"role": "assistant", "content": content})
                finally:
                    stop_live_timer(timer_stop, timer_thread)
            else:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    reasoning_content, content = call_llm(client, args, messages)  # type: ignore[arg-type]
                finally:
                    stop_live_timer(timer_stop, timer_thread)
        if content is not None and not args.dry_run:
            try:
                parsed_items.append(parser(axis, content))
            except Exception as exc:
                message = f"{axis}: {exc}"
                json_errors.append(message)
                if args.strict_json:
                    print(f"[{stage_label}][json-error] {message}", flush=True)
        append_prompt_block(lines, axis, prompt, reasoning_content, content)
        axis_elapsed = time.perf_counter() - axis_start
        total_elapsed = time.perf_counter() - total_start
        print(
            f"[{stage_label}] ({index}/{len(evidence)}) {axis} done | axis={format_duration(axis_elapsed)} total={format_duration(total_elapsed)}",
            flush=True,
        )
    return parsed_items, json_errors


def write_json_doc(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    evidence = load_evidence(args.evidence)
    print(f"[stage4] start: axes={len(evidence)} dry_run={args.dry_run} mode={args.mode} model={args.model}", flush=True)
    client = get_client(args)

    axis_semantics: list[dict[str, Any]] = []
    if args.mode in {"both", "naming"}:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = make_report_header(
            "# Stage 4B Latent Naming Prompt Responses",
            "This file records the complete latent naming prompt and LLM response for each latent dimension.",
            args,
        )
        axis_semantics, json_errors = run_prompt_batch(
            args=args,
            client=client,
            evidence=evidence,
            total_start=total_start,
            stage_label="stage4-naming",
            lines=lines,
            prompt_builder=render_prompt,
            parser=compact_axis_semantics,
        )
        write_start = time.perf_counter()
        args.out.write_text("\n".join(lines), encoding="utf-8")
        write_elapsed = time.perf_counter() - write_start
        print(f"[stage4-naming] wrote: {args.out} | write={format_duration(write_elapsed)}", flush=True)
        if json_errors and args.strict_json:
            raise SystemExit(
                "failed to parse one or more latent naming responses as JSON; "
                f"prompt/response report was written to {args.out}, but semantics JSON was not written: "
                + "; ".join(json_errors)
            )
        if not args.dry_run and not args.no_semantics_out:
            write_json_doc(
                args.semantics_out,
                {
                    "source_response_file": str(args.out),
                    "model": args.model,
                    "base_url": args.base_url,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "thinking": args.thinking,
                    "axes": axis_semantics,
                },
            )
            print(f"[stage4-naming] semantics: {args.semantics_out}", flush=True)
        elif args.dry_run:
            print("[stage4-naming] semantics JSON skipped in dry-run mode", flush=True)
        else:
            print("[stage4-naming] semantics JSON disabled by --no-semantics-out", flush=True)

    if args.mode in {"both", "coarse"}:
        if not axis_semantics:
            axis_semantics = load_axis_semantics(args.semantics_out)
        semantics_by_axis = axes_by_name(axis_semantics)
        missing = [item["axis"] for item in evidence if item.get("axis") not in semantics_by_axis]
        if missing:
            raise SystemExit(f"missing axis semantics for coarse projection: {', '.join(missing)}")
        args.coarse_out.parent.mkdir(parents=True, exist_ok=True)
        lines = make_report_header(
            "# Stage 4B Coarse Projection Prompt Responses",
            "This file records the coarse semantic projection prompt and LLM response for each latent dimension.",
            args,
        )

        def build_projection(axis_obj: dict[str, Any]) -> str:
            return render_coarse_projection_prompt(axis_obj, semantics_by_axis[axis_obj["axis"]])

        axis_coarse_tags, json_errors = run_prompt_batch(
            args=args,
            client=client,
            evidence=evidence,
            total_start=total_start,
            stage_label="stage4-coarse",
            lines=lines,
            prompt_builder=build_projection,
            parser=compact_axis_coarse_tags,
        )
        write_start = time.perf_counter()
        args.coarse_out.write_text("\n".join(lines), encoding="utf-8")
        write_elapsed = time.perf_counter() - write_start
        print(f"[stage4-coarse] wrote: {args.coarse_out} | write={format_duration(write_elapsed)}", flush=True)
        if json_errors and args.strict_json:
            raise SystemExit(
                "failed to parse one or more coarse projection responses as JSON; "
                f"prompt/response report was written to {args.coarse_out}, but coarse tags JSON was not written: "
                + "; ".join(json_errors)
            )
        if not args.dry_run:
            write_json_doc(
                args.coarse_tags_out,
                {
                    "source_response_file": str(args.coarse_out),
                    "source_axis_semantics_file": str(args.semantics_out),
                    "model": args.model,
                    "base_url": args.base_url,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "thinking": args.thinking,
                    "axes": axis_coarse_tags,
                },
            )
            print(f"[stage4-coarse] coarse tags: {args.coarse_tags_out}", flush=True)
        else:
            print("[stage4-coarse] coarse tags JSON skipped in dry-run mode", flush=True)

    total_elapsed = time.perf_counter() - total_start
    print(f"[stage4] axes: {len(evidence)}", flush=True)
    print(f"[stage4] total elapsed: {format_duration(total_elapsed)}", flush=True)


if __name__ == "__main__":
    main()
