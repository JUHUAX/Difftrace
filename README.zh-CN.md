# DiffTrace Artifact

该 artifact 包含 DiffTrace 的实现代码、benchmark 二进制程序、自定义 pintool 源码，以及用于复现主要实验流程的脚本。

工件不包含报文数据、TG/PG标注、实验结果或模型权重。名称包含groundtruth的文件是生成与评估代码，不是数据；运行时需要自行提供或生成输入。benchmark中的12个预编译程序仍予保留。

## 目录结构

- `difftrace/`：DiffTrace 核心管线实现，按阶段组织。
- `pintool/`：自定义 Intel Pin pintool 源码和构建说明。
- `benchmark/`：已编译的 benchmark client/server 二进制程序和流量抓取脚本。
- `experiments/`：RQ3 和 RQ4 消融实验脚本。
- `tools/`：用于 TShark 解析、SOTA 评估和 program-log 分析的辅助工具。
- `tests/`：仅使用临时构造样例的检查代码，不包含实验数据。
- `MANIFEST.txt`：工件文件清单，不包含Git内部文件和Python缓存。

## 环境概览

代码主要由 Python 脚本和 C++ Intel Pin pintool 组成。典型环境需要：

- Linux x86-64。
- Python 3.10+。
- `requirements.txt` 中列出的 Python 包。
- TShark/Wireshark 命令行工具，用于基于 TShark 的解析。
- Intel Pin 需要单独安装，详见 `pintool/README.md`。

在 artifact 根目录安装 Python 依赖：

```bash
pip install -r requirements.txt
```

LLM 相关脚本会根据用途从环境变量读取凭证：

```bash
export OPENAI_API_KEY=<your-openai-api-key>       # program-log ground-truth 生成，默认模型 gpt-5.5
export MIMO_API_KEY=<your-mimo-api-key>           # program-log pairwise judge，默认模型 MiMo-V2.6-flash
export MIMO_API_BASE_URL=<your-mimo-base-url>     # MiMo provider 使用自定义 OpenAI-compatible endpoint 时需要
export DEEPSEEK_API_KEY=<your-deepseek-api-key>   # DiffTrace 语义生成，默认模型 deepseek-v4.1-flash
```

## 典型流程

1. 在 `pintool/` 下构建 pintool。
2. 使用 `benchmark/binaries/` 下的 benchmark 二进制程序和 `benchmark/scripts/` 下的流量脚本运行协议 client/server 通信并抓取流量。
3. 运行 `difftrace/` 下的 DiffTrace Stage 1 和 Stage 2 脚本，收集执行轨迹、划分字段、扰动字段并计算执行差分。
4. 运行代码目录 `stage3/` 和 `stage4/`，完成论文Stage 3的表示学习、维度解释和语义聚合；代码目录名称不代表论文有第四个方法阶段。
5. 使用 `tools/` 和 `experiments/` 下的脚本运行评估工具和消融实验。

各目录提供中英文说明：[核心管线](difftrace/README.zh-CN.md)、[benchmark](benchmark/README.zh-CN.md)、[pintool](pintool/README.zh-CN.md)、[评估工具](tools/README.zh-CN.md)、[实验入口](experiments/README.zh-CN.md)。所有命令中的占位路径都必须替换后再运行。

## V7评估口径与工件清单

RQ1和RQ3均使用Accuracy、F1-score和Perfection。TG/PG评估结果的`paper_metrics`及`field_boundary_v7_metrics.csv/.md`采用该口径；旧字段匹配指标仅作诊断。协议平均与整体微平均分别报告，RQ3主表采用有参考bit字段的协议平均。

RQ4默认抽样100个共同字段，报告每字段平均Prompt、Completion、Reasoning、Total Tokens及耗时，同时保留总量。缺失的API统计和dry-run结果标为N/A，不视为实测零值。

运行`python tools/update_manifest.py`更新清单，使用`--check`核验。分发时排除`.git/`和缓存，不删除本地仓库历史。清单描述本地工作树，不表示修改已经上传。本次对齐不改变其他脚本的原实验路径，运行时仍需按环境配置。

运行`PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`进行无数据、无LLM调用的小样例检查。

## 当前验证范围与待处理限制

已通过22项临时样例检查，覆盖TG/PG和SOTA评估入口、边界指标、RQ3汇总及RQ4 dry-run；这不是六协议真实实验或独立机器上的端到端验证，也未重新计算论文中的实验数值。测试需要Python 3.10+；系统默认Python可能不满足要求。

以下实现问题尚未由文档更新修复：

- `relative_start`计算报文内相对位置，但维度解释代码中的自然语言描述仍误写为首次消费位置。
- 新协议复用已有缩放器、AE、维度解释和参考分布的完整推理入口尚未整理；不能在单个新报文上重新建立百分位分布。核心README解释了现有参数的作用，但不代表该流程已完成验证。
- 部分脚本仍硬编码原实验路径，benchmark抓包脚本还依赖未打包的构建目录；不同环境需先配置，不能直接一键运行。

README和文件清单说明工件内容；发布状态以远程提交和匿名工件页面的核验结果为准。
