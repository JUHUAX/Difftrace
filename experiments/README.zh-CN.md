# 实验复现指南

该目录提供四个 research question 评估的复现入口。RQ3 和 RQ4 在该目录下有专用脚本；RQ1 和 RQ2 复用 `difftrace/` 与 `tools/` 下的通用管线和评估工具。

目录仅包含脚本和说明，不包含实验输入、TG/PG数据或结果。

代码目录`stage3/`和`stage4/`共同实现论文Stage 3。以下命令是接口示例；RQ3生成器、RQ4 shuffled管线及若干辅助脚本仍依赖原工作区路径和外部输入，需适配后运行。小样例测试不等于六协议实验已复现。

## V7指标与成本输出

RQ1/RQ3主指标为Accuracy、F1-score和Perfection。TG/PG评估器输出`field_boundary_v7_metrics.csv/.md`，RQ3采用有参考bit字段的协议平均。旧汇总需用新评估器重新计算，不能将旧Subfield F1直接改名为边界F1-score。

RQ4成本脚本默认抽样100个共同字段，不足时明确报错；显式设置更小的`--sample-size`可试跑。JSON的`avg_*`是每字段平均成本，原总量也保留，Markdown主表对应V7。Reasoning Tokens来自API统计，是Completion Tokens的子集，不重复计入Total Tokens。缺失统计及dry-run测量显示N/A。

```bash
python experiments/RQ4/measure_rq4_llm_usage.py \
  --full-input <field_semantic_profiles.jsonl> \
  --no-latent-input <stage3_dataset_semantic_fields.csv> \
  --output-dir <cost_output_dir> \
  --sample-size 100
```

增加`--dry-run`只检查提示词，不调用LLM，不产生实测成本。

运行实验前，请在 artifact 根目录设置 Python 路径：

```bash
cd /path/to/artifact
export PYTHONPATH=$PWD/difftrace/common:$PWD/difftrace/stage1:$PWD/difftrace/stage2:$PWD/difftrace/stage3:$PWD/difftrace/stage4:$PYTHONPATH
```

LLM 相关脚本按任务使用不同凭证：DiffTrace 语义生成使用 `DEEPSEEK_API_KEY`，PG 生成使用 `OPENAI_API_KEY`，program-log pairwise judge 使用 `MIMO_API_KEY`。

## RQ1：细粒度字段划分

RQ1 评估 byte-level 和 bit-level 字段划分。复现流程如下：

1. 使用 `benchmark/` 下的二进制程序和脚本生成或提供协议 pcap。
2. 运行 DiffTrace Stage 1/2，生成字段划分输出。
3. 使用 `tools/tshark/` 构建 TShark-derived 协议视图。
4. 使用 `tools/sota_evaluation/` 和 program-log 工具评估 DiffTrace 与 SOTA 输出。

主要脚本：

- `difftrace/stage2/full.py`：端到端重放、跟踪、字段划分、扰动和执行差分驱动脚本。
- `tools/tshark/field_boundary/generate_groundtruthA_tshark.py`：从 pcap 中提取 TShark 字段信息。
- `tools/tshark/field_boundary/evaluate_tshark_vs_experiment.py`：在 TShark-derived 视角下评估字段边界。
- `tools/program_log/scripts/evaluate_program_log_field_boundary.py`：在 program-log-derived 视角下评估字段边界。
- `tools/sota_evaluation/scripts/evaluate_boundary_predictions.py`：计算 SOTA 输出的统一字段边界指标。

示例命令形式：

```bash
python difftrace/stage2/full.py \
  --mode pcap \
  --proto tcp \
  --pcap <input.pcap> \
  --target-host 127.0.0.1 \
  --target-port <port> \
  --server-bin benchmark/binaries/<server> \
  --pin-bin <path-to-pin> \
  --taint-tool <path-to-pintool.so> \
  --outdir <difftrace_output_dir> \
  --taint

python tools/tshark/field_boundary/generate_groundtruthA_tshark.py \
  --pcap <name>=<input.pcap> \
  --output-dir <tshark_output_dir>

python tools/sota_evaluation/scripts/evaluate_boundary_predictions.py \
  --predictions <boundary_predictions.jsonl> \
  --groundtruth <boundary_groundtruth.jsonl> \
  --outdir <rq1_metrics_dir>
```

每个脚本可通过 `--help` 查看具体路径参数。

预期输出：

- `<difftrace_output_dir>/` 下的 DiffTrace 重放结果，包括每包的 `fields.json`、`bitfields.json`、扰动记录和执行差分报告。
- `<tshark_output_dir>/` 下的 TShark-derived 字段视图，用作 TG 字段边界真值。
- SOTA评估器输出`boundary_metrics_summary.json/.csv/.md`；TG评估器输出`metrics_summary.json`，PG评估器输出`field_boundary_metrics_summary.json`。TG/PG的新主指标表均为`field_boundary_v7_metrics.csv/.md`，JSON中的主指标位于`paper_metrics`。

## RQ2：字段程序语义推理

RQ2 在统一传统标签和 program-log semantic agreement 两个视角下评估字段语义推理。复现流程如下：

1. 运行 Stage 2，获得字段行为摘要。
2. 运行 Stage 3，学习并投影低维字段表示。
3. 运行 Stage 4，生成字段程序语义描述。
4. 使用 TShark-derived 语义标签评估传统标签准确率。
5. 使用 pairwise judge 评估程序语义一致性。

为了横向比较，将各工具标签按含义/关键字固定映射到六类：`identifier`、`length_or_count`、`control_or_flags`、`addressing`、`data_value`和`other_or_unknown`。DiffTrace在自然语言语义之外输出固定标签。TG报告标签Accuracy；PG报告Strong Agreement（same + mostly same）和Any Overlap（再加入weakly same），主要指标为Strong Agreement。

RQ2使用按报文划分的70%训练/30%留出集；划分清单可由`difftrace/stage4/generate_heldout_packet_split.py`生成，并通过矩阵构建器的`--split-manifest`使用。缩放器、AE训练及相关性解释只使用训练部分；留出部分只接受相同缩放与编码，激活阈值以训练空间分布校准。AE内部10%验证集仅用于检查点选择，不代替30%留出集。工件不附划分数据、训练权重或结果。

主要脚本：

- `difftrace/stage3/build_stage3_dataset.py`：从字段行为摘要构建 Stage 3 数据集。
- `difftrace/stage3/build_stage3_training_matrix.py`：准备归一化训练/投影矩阵。
- `difftrace/stage3/train_stage3_autoencoder.py`：训练并应用自编码器表示模型。
- `difftrace/stage4/build_stage4_latent_names.py`：准备表示空间维度解释证据。
- `difftrace/stage4/run_stage4_llm_naming.py`：生成表示空间维度语义定义。
- `difftrace/stage4/build_stage4_field_profiles.py`：构建字段级激活维度 profile。
- `difftrace/stage4/run_stage4_field_semantic_fusion.py`：生成最终字段程序语义描述。
- `tools/tshark/semantic_inference/build_tshark_semantic_groundtruth.py`：从 TShark 输出构建统一传统语义标签。
- `tools/tshark/semantic_inference/evaluate_stage4_semantic_predictions.py`：在传统标签下评估语义预测。
- `tools/program_log/scripts/run_program_log_pairwise_judge.py`：使用 pairwise judge 评估字段程序语义。
- `tools/sota_evaluation/scripts/evaluate_unified_semantics.py`：在统一标签空间下评估 SOTA 语义输出。

示例命令形式：

```bash
python difftrace/stage3/train_stage3_autoencoder.py \
  --input-csv <train_only_matrix.csv> \
  --projection-csv <all_fields_matrix.csv> \
  --eval-csv <heldout_matrix.csv> \
  --output-dir <stage3_output_dir> \
  --latent-dims 8

python difftrace/stage4/run_stage4_llm_naming.py \
  --evidence <z_topk_probe_evidence.json> \
  --out <latent_naming_report.md> \
  --semantics-out <z_axis_semantics.json> \
  --api-key $DEEPSEEK_API_KEY

python tools/tshark/semantic_inference/evaluate_stage4_semantic_predictions.py \
  --predictions <field_semantic_fused_vectors.csv> \
  --groundtruth <tshark_semantic_groundtruth.csv> \
  --out-dir <rq2_tshark_metrics_dir>

python tools/program_log/scripts/run_program_log_pairwise_judge.py \
  --program-log-jsonl <program_log_semantics.jsonl> \
  --stage4-profiles <field_semantic_fused_profiles.jsonl> \
  --output-csv <judge_results.csv> \
  --output-md <judge_report.md> \
  --run-log <judge_run_log.jsonl> \
  --backend api
```

预期输出：

- Stage 3 表示文件，包括 `stage3_training_matrix.csv`、`ae_embeddings.csv` 等自编码器表示结果和训练摘要。
- Stage 4 语义文件，包括 `z_topk_probe_evidence.json`、`z_axis_semantics.json`、`field_semantic_profiles.jsonl` 以及最终语义向量/profile。
- 评估结果，包括传统标签准确率汇总和 pairwise judge 结果，例如 `judge_results.csv`。

两次语义生成默认`deepseek-v4.1-flash`，PG初始标注默认`gpt-5.5`，独立judge默认`MiMo-V2.6-flash`。这些名称是调用配置，不证明已有实验结果是在更新模型后重新生成的。PG初始候选需按论文流程核验。

## RQ3：bit field 划分消融

`RQ3/` 下的文件用于评估 bit field 划分模块的不同变体：

- `run_rq3_bitfield_ablation.py`：生成 Full、Operation-Driven Recovery 和 Flat Evidence Aggregation 的输出。
- `evaluate_rq3_bitfield_ablation.py`：主表计算bit边界Accuracy、F1-score和Perfection，旧检测/子字段匹配指标仅保留在JSON诊断项中。

ODR从局部位操作直接形成候选，不等待后续消费修正；FEA扫描完整轨迹收集消费事件，但不按指令证据强弱裁决边界；Full使用完整流程。

示例：

```bash
python experiments/RQ3/run_rq3_bitfield_ablation.py \
  --replay-root <stage2_replay_outputs> \
  --outdir <rq3_output_dir> \
  --overwrite

python experiments/RQ3/evaluate_rq3_bitfield_ablation.py \
  --outdir <rq3_output_dir> \
  --groundtruth-jsonl <verified_pg.jsonl>
```

预期输出：

- `<rq3_output_dir>/<mode>/` 下的消融输出，每个 mode 包含按包生成的 `bitfields.json`。
- `rq3_generation_manifest.json`，记录生成过程和输入输出。
- `<rq3_output_dir>/metrics/<mode>/` 下的指标汇总，以及 `rq3_bitfield_ablation_summary.json` 和 `rq3_bitfield_ablation_summary.md`。

`--skip-evaluate`只能在新评估器已经生成`paper_metrics`时重建汇总，不能用它把旧结果转换为新指标。没有参考bit字段的协议不参与主表协议平均，整体微平均只作另行报告。

## RQ4：语义表示消融

`RQ4/` 下的文件用于评估策略组划分和低维表示空间学习：

Shuffled groups保持候选值和差分结果，但打乱策略分组；No-latent direct保持28维摘要，跳过AE及潜在维度解释。比较时使用相同字段、同一语义生成模型和同一评估参考。

- `build_rq4_shuffled_group_dataset.py`：构建 shuffled-group 行为摘要。
- `run_rq4_shuffled_group_pipeline.py`：运行 shuffled-groups 消融管线。
- `run_rq4_no_latent_direct_summary.py`：运行 no-latent direct 语义生成 baseline。
- `measure_rq4_llm_usage.py`：在抽样字段上测量 LLM 调用成本。
- `summarize_program_log_judge_on_rq2b_eval.py`：汇总 pairwise-judge 输出。

示例：

```bash
python experiments/RQ4/run_rq4_shuffled_group_pipeline.py \
  --seed 0 \
  --out-root <rq4_output_dir> \
  --latent-dim 8 \
  --backend api
```

预期输出：

- shuffled-group 数据集，例如 `stage3_dataset_semantic_fields.csv` 和 `rq4_shuffled_group_manifest.json`。
- `<rq4_output_dir>/shuffled_seed_<seed>/` 下的 shuffled pipeline 结果，包括 Stage 3 矩阵、AE embedding、维度语义、字段 profile 和融合后的语义输出。
- no-latent direct 输出，例如 `field_semantic_direct_profiles.jsonl` 和 `field_semantic_direct_vectors.csv`。
- LLM调用成本的逐调用CSV、含`avg_*`及总量的`rq4_llm_usage_measurement_summary.json`和每字段平均成本Markdown主表。
- program-log judge 汇总，例如 `rq4_program_log_judge_on_rq2b_eval_summary.csv`、按协议 CSV 和可读 Markdown 报告。

本轮没有调用LLM或重跑真实实验，不改变V7中的结果数值。所有上述输出均在运行时产生，不是工件附带文件。首次试跑应显式减小样本并检查质量，再进行付费完整运行。
