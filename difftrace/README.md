# DiffTrace Pipeline Code

This directory contains the core DiffTrace implementation. The code is organized according to the pipeline stages used in the paper.

When running scripts from this reorganized artifact directory, set `PYTHONPATH` so that shared modules can be imported:

```bash
cd /root/semvec/data_avaliable
export PYTHONPATH=$PWD/difftrace/common:$PWD/difftrace/stage1:$PWD/difftrace/stage2:$PWD/difftrace/stage3:$PWD/difftrace/stage4:$PYTHONPATH
```

## `common/`: shared utilities

- `common.py`: shared helpers for packet replay, process management, trace preprocessing, and metric utilities.
- `field_units.py`: common field-unit data structures and helpers.
- `send.py`: packet sending/replay helper.
- `verify_branch_parsing.py`: helper for checking branch parsing in execution traces.

## `stage1/`: fine-grained field segmentation

Stage 1 recovers byte/bit field boundaries from taint-analysis traces.

- `fields.py`: byte-level field segmentation from taint records and field-related instruction evidence.
- `analyze_bitfields_planA.py`: bit-level consumption tracking and instruction-evidence arbitration for recovering bit subfields.

Typical use is through the Stage 2 driver `stage2/full.py`, which calls the segmentation logic as part of the end-to-end pipeline. For debugging bitfield recovery directly, run:

```bash
python difftrace/stage1/analyze_bitfields_planA.py --help
```

## `stage2/`: strategic execution-difference computation

Stage 2 performs field perturbation, replays perturbed packets, computes execution-difference metrics, and builds field behavior summaries.

- `full.py`: main end-to-end driver for packet replay, taint collection, field segmentation, perturbation, and execution-difference computation.
- `mutate.py`: field perturbation candidate generation.
- `diff.py`: execution-difference metric computation between original and perturbed executions.
- `compare_all_mutations.py`: batch comparison over mutation outputs.
- `build_field_training_samples.py`: builds field-level behavior-summary samples for representation learning.
- `dataset_health_check.py`: checks generated training data quality.
- `debug_metric.py`: debug helper for inspecting individual execution-difference metrics.
- `run_frozen_stage2_protocol.sh`: helper script for rerunning Stage 2 on a selected protocol.

Example command shape:

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
  --outdir <output-dir> \
  --taint
```

Use `python difftrace/stage2/full.py --help` for all available options.

## `stage3/`: representation-space learning

Stage 3 learns low-dimensional representations from 28-dimensional field behavior summaries.

- `build_stage3_dataset.py`: builds the Stage 3 dataset from field behavior summaries.
- `filter_stage3_transparent_fields.py`: filters fields with little observable execution difference.
- `build_stage3_training_matrix.py`: normalizes and prepares the training matrix.
- `train_stage3_autoencoder.py`: trains autoencoders with configurable latent dimensions.
- `train_stage3_pca.py`: PCA baseline/helper.
- `analyze_stage3_latent_space.py`, `visualize_stage3_latent_space.py`, `find_stage3_neighbors.py`: analysis and visualization helpers.
- `check_stage3_dataset.py`, `observe_stage3_dataset.py`: data inspection helpers.
- `basic_visual_analysis.py`, `analyze_latent_space_structure.py`, `train_unsupervised_v1.py`, `probe_latent_space_v1.py`, `infer_field_role_v1.py`: auxiliary analysis scripts retained for representation-space inspection.

Example:

```bash
python difftrace/stage3/train_stage3_autoencoder.py \
  --input-csv <stage3_training_matrix.csv> \
  --projection-csv <all_fields_matrix.csv> \
  --output-dir <stage3_output_dir> \
  --latent-dims 8
```

## `stage4/`: semantic interpretation and aggregation

This directory implements representation-dimension interpretation and field program-semantic aggregation. It corresponds to the semantic-representation part of Stage 3 in the paper.

- `build_stage4_latent_names.py`: builds evidence for interpreting representation-space dimensions.
- `run_stage4_llm_naming.py`: queries an LLM to assign program-behavior semantics to representation dimensions.
- `build_stage4_field_profiles.py`: builds field-level activated-dimension profiles.
- `run_stage4_field_semantic_fusion.py`: aggregates activated dimension semantics into field program semantics.
- `generate_heldout_packet_split.py`: creates held-out packet splits for evaluation.

Example:

```bash
python difftrace/stage4/run_stage4_llm_naming.py \
  --evidence <z_topk_probe_evidence.json> \
  --out <latent_naming_report.md> \
  --semantics-out <z_axis_semantics.json> \
  --api-key $DEEPSEEK_API_KEY
```

For scripts that invoke LLMs, set `DEEPSEEK_API_KEY` or pass `--api-key` explicitly.
