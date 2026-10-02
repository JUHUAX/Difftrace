# 评估辅助工具

该目录包含用于解析流量、评估 baseline 输出，以及运行 program-log 分析的辅助工具。

目录仅包含工具代码与标签映射配置，不包含TG/PG数据、对比工具预测、judge输出或实验结果。运行以下命令时，从工件根目录出发并替换全部占位路径；未显式配置的默认路径可能仍指向原实验环境。

## V7字段边界指标

`boundary_metrics.py`统一计算byte/bit的Accuracy、F1-score和Perfection。所有可能的边界位置（含两端）参与判定；Perfection要求真值字段的两端均被预测，且内部没有多余边界。bit边界沿用父字段身份，以真值与预测父字段的并集确定候选域，避免丢弃仅预测出的父字段。

TG/PG评估器在JSON中新增`paper_metrics`，并输出`field_boundary_v7_metrics.csv/.md`；旧指标只作诊断。SOTA评估器主表及CSV新增Accuracy、F1-score和Perfection。`Protocol Value Avg`是有相应参考字段的协议的算术平均，`Overall Micro`汇总原始计数，两者不能混用。没有参考bit字段的协议不计入bit主表平均；旧的Macro诊断项不作为论文主表。

报文、标注和预测需运行时提供，工件不包含这些数据。`update_manifest.py`更新或检查不含Git内部文件和缓存的清单。

## Ground truth 构建流程

论文使用两套互补参考：TShark-derived ground truth（TG）和程序日志参考标注（PG）。TG描述协议解析视角，PG补充具体程序的已观测消费行为，不是唯一的协议规范真值。本目录下工具可按如下方式串联使用。

### 构建 TShark-derived ground truth（TG）

TG 来自 TShark 对 pcap 的结构化解析，用于协议规范/解析器视角评估。

1. 将 pcap 解析为逐包 TShark JSON：

   ```bash
   python tools/tshark/field_boundary/generate_groundtruthA_tshark.py \
     --pcap bacnet=<bacnet.pcap> \
     --pcap cip=<cip.pcap> \
     --output-dir <tshark_json_dir>
   ```

   对 IEC61850/MMS，需要去掉 ISO stack 的 MMS payload，可先使用 `field_boundary/strip_iec61850_iso_to_mms_pcap.py`。

2. 导出与 replay sample ID 对齐的 byte 字段边界真值：

   ```bash
   python tools/sota_evaluation/scripts/export_tshark_boundary_groundtruth.py \
     --tshark-root <tshark_json_dir> \
     --replay-root <difftrace_replay_outputs> \
     --output <tg_boundary_groundtruth.jsonl>
   ```

3. 从 TShark 字段构建传统语义标签候选：

   ```bash
   python tools/tshark/semantic_inference/build_tshark_semantic_groundtruth.py \
     --input-dir <tshark_json_dir> \
     --output-dir <tg_semantic_dir>
   ```

   输出 CSV 是规则映射得到的候选文件，作为最终统一标签语义真值使用前需要人工校验。

### 构建 Program-log-derived ground truth（PG）

PG 来自污点执行日志，用于程序执行视角评估。它关注被测实现如何实际消费输入字节，而不是结果是否符合协议规范标签。

1. 先通过 DiffTrace/Pin replay 管线获得执行日志。每个 packet 目录应包含 replay 过程中生成的执行日志和元数据。

2. 预处理原始程序日志：

   ```bash
   python tools/program_log/scripts/preprocess_program_logs.py \
     --input-root <difftrace_replay_outputs> \
     --output-dir <preprocessed_log_dir> \
     --overwrite
   ```

3. 使用 LLM ground-truth builder 生成初步 PG 候选：

   ```bash
   python tools/program_log/scripts/run_program_log_groundtruth_llm.py \
     --input-dir <preprocessed_log_dir> \
     --output-dir <pg_llm_output_dir> \
     --backend api \
     --api-key $OPENAI_API_KEY
   ```

   这些输出只是初步候选。日志会剔除协议名和函数名，prompt 也禁止模型依据协议规范或自身先验知识推断结果。

4. 将逐日志 LLM 输出整理为结构化 JSONL/CSV：

   ```bash
   python tools/program_log/scripts/collect_program_log_groundtruth_outputs.py \
     --input-dir <pg_llm_output_dir> \
     --output-dir <pg_eval_dir> \
     --replay-root <difftrace_replay_outputs>
   ```

5. 对收集后的候选进行人工验证和修正。人工校验重点是字段边界和程序语义是否符合执行日志中的消费行为，而不是是否符合协议规范标签。补全语义条目时可使用 `program_log/scripts/fill_manual_groundtruth_semantics.py`。

最终 PG 文件通常会被 `evaluate_program_log_field_boundary.py` 用于字段边界评估，并被 `run_program_log_pairwise_judge.py` 用于程序语义一致性评估。

## `tshark/`

TShark 工具用于解析抓取到的 pcap，并构建或评估 TShark-derived 协议视图。

- `field_boundary/generate_groundtruthA_tshark.py`：从 pcap 中提取结构化的 per-packet TShark 字段数据。
- `field_boundary/strip_iec61850_iso_to_mms_pcap.py`：将 IEC61850 抓包转换为 MMS 层级 pcap 的辅助脚本。
- `field_boundary/replay_groundtruth_pcap.py`：重放数据包，用于对齐和分析。
- `field_boundary/analyze_replay_logs.py`：分析重放日志。
- `field_boundary/evaluate_tshark_vs_experiment.py`：基于 TShark-derived 数据评估字段边界预测。
- `semantic_inference/build_tshark_semantic_groundtruth.py`：从 TShark 字段构建统一粗粒度语义标签。
- `semantic_inference/evaluate_stage4_semantic_predictions.py`：在统一标签空间下评估字段语义预测。

示例：

```bash
python tools/tshark/field_boundary/generate_groundtruthA_tshark.py \
  --pcap bacnet=<bacnet.pcap> \
  --output-dir <tshark_output_dir>
```

## `sota_evaluation/`

SOTA 评估工具用于归一化 baseline 输出并计算统一指标。

这些脚本不包含七套SOTA工具的实现；须自行提供对应工具输出。固定映射依据源标签含义/关键字，统一标签为`identifier`、`length_or_count`、`control_or_flags`、`addressing`、`data_value`和`other_or_unknown`。

- `config/semantic_label_mapping.json`：统一粗粒度语义标签映射。
- `scripts/export_boundary_predictions.py`：将字段边界预测导出为统一格式。
- `scripts/export_tshark_boundary_groundtruth.py`：导出 TShark-derived 字段边界真值。
- `scripts/evaluate_boundary_predictions.py`：计算字段边界指标。
- `scripts/export_unified_semantic_candidates.py`：将语义候选导出到统一标签空间。
- `scripts/evaluate_unified_semantics.py`：评估语义预测结果。
- `scripts/evaluate_fsibp_native.py`：评估 FSIBP 输出。
- `scripts/export_tshark_sample_alignment.py`、`scripts/export_program_log_semantic_pairs.py`：样本对齐和导出辅助脚本。

示例：

```bash
python tools/sota_evaluation/scripts/evaluate_boundary_predictions.py \
  --predictions <boundary_predictions.jsonl> \
  --groundtruth <boundary_groundtruth.jsonl> \
  --outdir <metrics_output_dir>
```

## `program_log/`

Program-log 工具用于预处理执行日志，并运行程序执行视角的分析。

- `scripts/preprocess_program_logs.py`：预处理原始执行日志。
- `scripts/run_program_log_groundtruth_llm.py`：运行基于 LLM 的 program-log 分析。
- `scripts/collect_program_log_groundtruth_outputs.py`：将 LLM 输出整理为结构化文件。
- `scripts/evaluate_program_log_field_boundary.py`：从 program-log 视角评估字段边界。
- `scripts/run_program_log_pairwise_judge.py`：运行 pairwise semantic judge。
- `scripts/fill_manual_groundtruth_semantics.py`：补全语义条目的辅助脚本。

示例：

```bash
python tools/program_log/scripts/run_program_log_pairwise_judge.py \
  --program-log-jsonl <program_log_semantics.jsonl> \
  --stage4-profiles <field_semantic_fused_profiles.jsonl> \
  --output-csv <judge_results.csv> \
  --output-md <judge_report.md> \
  --run-log <judge_run_log.jsonl> \
  --backend api
```

LLM 支持的 program-log 脚本会根据任务使用不同默认后端：

- PG 生成脚本使用 `gpt-5.5`，读取 `OPENAI_API_KEY` 或显式传入的 `--api-key`。
- Pairwise judge 使用 `MiMo-V2.6-flash`，读取 `MIMO_API_KEY` 或显式传入的 `--api-key`；如果 provider 需要自定义 endpoint，需要设置 `MIMO_API_BASE_URL` 或传入 `--api-base-url`。

`--stage4-profiles`在这里必须指向含最终字段程序语义的融合输出，不能使用仅含激活维度解释的profile文件。Judge的`same_behavior`和`mostly_same_behavior`计入Strong Agreement；再加入`weakly_same_behavior`得到Any Overlap。`different_behavior`和`insufficient_information`不计入这两项。

涉及LLM的命令会产生API费用。PG builder和judge不能假定都有`--dry-run`；只在脚本确实支持时使用。`--report-only`用于judge已有结果的报告，不会重新判定描述；五类verdict汇总可用`../experiments/RQ4/summarize_program_log_judge_on_rq2b_eval.py`。生成PG候选不等于完成了论文所述人工核验。
