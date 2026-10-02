# DiffTrace 管线代码

该目录包含 DiffTrace 的核心实现。代码按管线阶段组织，可用于从协议流量和 benchmark 二进制程序开始，运行完整流程并输出 byte/bit 字段边界和字段程序语义描述。

本目录仅包含实现与说明，不包含报文、执行轨迹、训练矩阵、AE权重或语义结果。以下命令说明数据流和接口，尚未在独立机器上验证为可直接运行的完整流程。

## 与V7方法阶段的对应关系

| V7正文 | 主要代码 |
| --- | --- |
| Stage 1：细粒度字段划分 | `stage1/`及`../pintool/` |
| Stage 2：策略化执行差分计算 | `stage2/`；`stage3/build_stage3_dataset.py`整理其字段摘要 |
| Stage 3：字段程序语义表示 | `stage3/`的AE学习和`stage4/`的维度解释、字段语义聚合 |

四个策略组各生成六项摘要统计，并加入四项上下文特征，得到28维行为摘要；每次扰动的九维差分不直接作为AE输入。代码中的Stage 4不是论文新增阶段。

## 运行环境设置

在 artifact 根目录运行命令，并通过 `PYTHONPATH` 暴露各阶段目录：

```bash
cd /path/to/artifact
export PYTHONPATH=$PWD/difftrace/common:$PWD/difftrace/stage1:$PWD/difftrace/stage2:$PWD/difftrace/stage3:$PWD/difftrace/stage4:$PYTHONPATH
```

先构建 pintool，并设置示例命令所需路径：

```bash
export PIN_BIN=/path/to/pin
export TAINT_TOOL=$PWD/pintool/obj-intel64/pintool.so
export DEEPSEEK_API_KEY=<your-api-key>   # 仅 LLM 语义生成需要
```

## 文件组织

### `common/`：共享工具

- `common.py`：进程管理、重放辅助、trace 预处理和指标工具。
- `field_units.py`：通用字段单元数据结构。
- `send.py`：数据包发送/重放辅助。
- `verify_branch_parsing.py`：执行轨迹中的分支解析检查器。

### `stage1/`：细粒度字段划分

Stage 1 从 taint-analysis traces 中恢复 byte/bit 字段边界。

- `fields.py`：基于污点记录和字段相关指令证据进行 byte-level 字段划分。
- `analyze_bitfields_planA.py`：bit-level consumption tracking 和 instruction-evidence arbitration。

常规使用中，Stage 1 会在端到端运行时由 `stage2/full.py` 调用。

### `stage2/`：策略化执行差分计算

Stage 2 负责重放数据包、收集轨迹、划分字段、扰动字段、计算执行差分指标，并构建字段行为摘要。

- `full.py`：重放、跟踪、字段划分、扰动和执行差分的端到端主驱动脚本。
- `mutate.py`：字段扰动候选值生成。
- `diff.py`：执行差分指标计算。
- `compare_all_mutations.py`：批量比较 mutation 输出。
- `build_field_training_samples.py`：将 Stage 2 输出转换为字段级训练样本。
- `dataset_health_check.py`：检查生成样本质量。
- `debug_metric.py`：检查单个执行差分指标。
- `run_frozen_stage2_protocol.sh`：重新运行指定协议 Stage 2 的辅助脚本。

### `stage3/`：表示空间学习

Stage 3 将字段行为摘要转换为低维字段表示。

- `build_stage3_dataset.py`：从 Stage 2 输出构建 all-field Stage 3 数据集。
- `filter_stage3_transparent_fields.py`：过滤几乎不产生可观测执行差异的透明字段。
- `build_stage3_training_matrix.py`：归一化特征并创建训练/评估矩阵。
- `train_stage3_autoencoder.py`：训练并应用自编码器。
- `check_stage3_dataset.py`：检查 Stage 3 数据集，并定义矩阵构建和自编码器共享的 28 维特征列。

### `stage4/`：语义解释和聚合

Stage 4 实现表示空间维度解释和字段程序语义聚合，对应论文Stage 3的后两部分。Spearman相关性默认按正负方向各筛选最多5项且满足绝对相关系数不小于0.30的特征；字段激活阈值默认为高20%和低20%。

- `build_stage4_latent_names.py`：计算表示维度与行为摘要 probe 之间的相关性。
- `run_stage4_llm_naming.py`：调用 LLM 为表示维度生成名称和定义。
- `build_stage4_field_profiles.py`：识别每个字段激活的语义维度。
- `run_stage4_field_semantic_fusion.py`：将激活维度语义聚合为字段级程序语义。
- `generate_heldout_packet_split.py`：生成 held-out packet split。

## 端到端流程

下面命令展示完整流程。请将 `<input.pcap>`、`<server>` 和 `<output-root>` 等占位符替换为具体协议路径。

### 1. 运行 Stage 1/2：跟踪、划分、扰动和执行差分

`full.py` 是主入口。它会在 Pin 下启动 benchmark server，重放 pcap 中的数据包，收集污点/执行日志，恢复字段边界，扰动字段并计算执行差分输出。

```bash
python difftrace/stage2/full.py \
  --mode pcap \
  --proto tcp \
  --pcap <input.pcap> \
  --target-host 127.0.0.1 \
  --target-port <port> \
  --server-bin benchmark/binaries/<server> \
  --pin-bin $PIN_BIN \
  --taint-tool $TAINT_TOOL \
  --outdir <output-root>/<protocol> \
  --taint
```

对于 BACnet 等 UDP 协议，需要设置 `--proto udp`，并使用对应 server 二进制程序和端口。

`<output-root>/<protocol>/` 下的主要输出包括 per-packet 字段划分结果、mutation 记录、执行差分指标，以及后续阶段使用的日志。

### 2. 为 Stage 3 构建字段行为摘要

对一个或多个协议运行 Stage 2 后，从 Stage 2 输出根目录构建字段级数据集：

```bash
python difftrace/stage3/build_stage3_dataset.py \
  --input-root <output-root> \
  --output-dir <stage3-dataset-dir>
```

过滤透明或低信息字段：

```bash
python difftrace/stage3/filter_stage3_transparent_fields.py \
  --input-csv <stage3-dataset-dir>/stage3_dataset_all_fields.csv \
  --output-dir <stage3-filtered-dir>
```

构建归一化训练/投影矩阵：

```bash
python difftrace/stage3/build_stage3_training_matrix.py \
  --input-csv <stage3-filtered-dir>/stage3_dataset_semantic_fields.csv \
  --output-dir <stage3-matrix-dir>
```

此命令未设置外部留出集。复现RQ2的按报文70%/30%划分时，先准备`stage4/generate_heldout_packet_split.py`输出的划分清单，再向矩阵构建器传入`--split-manifest <split.json>`；缩放器仅在训练部分拟合。不要将AE内部10%验证划分当成RQ2留出集。

### 3. 训练/投影低维表示空间

训练自编码器并将字段投影到学习到的表示空间：

```bash
python difftrace/stage3/train_stage3_autoencoder.py \
  --input-csv <stage3-matrix-dir>/stage3_training_matrix_train.csv \
  --projection-csv <stage3-matrix-dir>/stage3_training_matrix.csv \
  --output-dir <stage3-ae-dir> \
  --latent-dims 8
```

`ae_embeddings.csv`对应投影矩阵，`train_ae_embeddings.csv`对应训练矩阵；指定`--eval-csv`还会输出`eval_ae_embeddings.csv`。输出还包含训练历史、重构误差及模型检查点，均是运行时生成文件，不在工件中。

默认编码器为28→16→12→d，解码器对称，隐藏层ReLU、输出层Sigmoid；使用MSE、Adam、150轮、batch size 128、学习率0.001、weight decay 0.00001和seed 1337。脚本默认比较4/8/12维，上例显式使用V7配置8维；文档不据此断言8维已通过最优性实验验证。

### 4. 解释表示空间维度

计算 probe 相关性，并准备 LLM 维度解释证据：

```bash
python difftrace/stage4/build_stage4_latent_names.py \
  --embeddings <stage3-ae-dir>/ae_latent8/train_ae_embeddings.csv \
  --training-matrix <stage3-matrix-dir>/stage3_training_matrix_train.csv \
  --out-dir <stage4-latent-dir>
```

为表示维度生成自然语言语义：

```bash
python difftrace/stage4/run_stage4_llm_naming.py \
  --evidence <stage4-latent-dir>/z_topk_probe_evidence.json \
  --out <stage4-latent-dir>/latent_naming_report.md \
  --semantics-out <stage4-latent-dir>/z_axis_semantics.json \
  --api-key $DEEPSEEK_API_KEY
```

可以使用 `--dry-run` 预览prompt，避免实际调用API。该调用和后面的字段语义融合均默认使用`deepseek-v4.1-flash`，temperature为0、top-p为1，未设置采样seed。完整Prompt在对应Python文件中，不依赖外部Prompt图片。

### 5. 构建字段 profile 并生成字段程序语义

构建字段级激活维度 profile：

```bash
python difftrace/stage4/build_stage4_field_profiles.py \
  --embeddings <stage3-ae-dir>/ae_latent8/ae_embeddings.csv \
  --threshold-embeddings <stage3-ae-dir>/ae_latent8/train_ae_embeddings.csv \
  --axis-semantics <stage4-latent-dir>/z_axis_semantics.json \
  --out-dir <stage4-profile-dir>
```

将激活维度语义融合为一句字段级程序语义描述：

```bash
python difftrace/stage4/run_stage4_field_semantic_fusion.py \
  --input <stage4-profile-dir>/field_semantic_profiles.jsonl \
  --output-jsonl <stage4-semantic-dir>/field_semantic_fused_profiles.jsonl \
  --output-csv <stage4-semantic-dir>/field_semantic_fused_vectors.csv \
  --backend api \
  --api-key $DEEPSEEK_API_KEY
```

最终输出包括字段程序语义描述、用于横向比较的粗粒度标签及辅助得分；8维表示保存在AE嵌入文件中。语义输出可用`../tools/`下的脚本评估，标签映射配置为`../tools/sota_evaluation/config/semantic_label_mapping.json`。

## 新输入与现有实现的限制

`--threshold-embeddings`让字段profile使用已有参考嵌入分布计算百分位。不指定时，脚本会用当前输入字段自行排序，因此不能省略该参数后将单个新报文视为“复用已有空间”的推理。缩放器、AE、轴解释和参考分布必须来自同一次空间构建。

`train_stage3_autoencoder.py --projection-csv`是在本次训练后编码另一个矩阵，并非加载既有检查点的独立推理入口。新私有协议复用固定空间的完整入口尚待整理和验证。另一个待修正问题是：`relative_start`实际表示报文内相对位置，但`build_stage4_latent_names.py`中的行为描述仍写成首次消费位置。

部分默认路径、`common.py`的默认pintool路径及`run_frozen_stage2_protocol.sh`仍指向原工作区；后者还调用工件外脚本。迁移时必须检查路径和依赖，不能认为设置`PYTHONPATH`就已经消除了这些依赖。

## 调试建议

- 使用 `--help` 查看任意脚本的路径参数。
- 主驱动可用`--sample-count`控制报文数；数据集构建器使用`--samples`/`--limit-samples`，语义融合使用`--limit`。不要将某个脚本的参数直接套用到其他脚本。
- LLM 脚本建议先使用 `--dry-run` 检查 prompt 构造。
- 如果出现 import 错误，检查 `PYTHONPATH` 是否包含上述所有 `difftrace/` 阶段目录。
