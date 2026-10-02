# DiffTrace Pipeline Code

This directory contains the core DiffTrace implementation. It is organized by pipeline stage and can be used to run the full workflow from protocol traffic and a benchmark binary to byte/bit field boundaries and field program-semantic descriptions.

Only code/documentation is included: no packets, traces, training matrices, AE weights, or semantic outputs. Commands describe data flow and interfaces; they are not an independently validated turnkey workflow.

## Mapping to V7 method stages

| V7 section | Main code |
| --- | --- |
| Stage 1: fine-grained field segmentation | `stage1/` and `../pintool/` |
| Stage 2: strategic execution-difference computation | `stage2/`; `stage3/build_stage3_dataset.py` collects its field summaries |
| Stage 3: field program-semantic representation | AE learning in `stage3/`, followed by interpretation/aggregation in `stage4/` |

Six statistics per strategy group across four groups, plus four context features, form the 28-D summary. Per-perturbation 9-D differences are not direct AE inputs. Code Stage 4 is not an additional paper stage.

## Runtime setup

Run commands from the artifact root and expose the stage directories through `PYTHONPATH`:

```bash
cd /path/to/artifact
export PYTHONPATH=$PWD/difftrace/common:$PWD/difftrace/stage1:$PWD/difftrace/stage2:$PWD/difftrace/stage3:$PWD/difftrace/stage4:$PYTHONPATH
```

Build the pintool first and set paths used by the examples:

```bash
export PIN_BIN=/path/to/pin
export TAINT_TOOL=$PWD/pintool/obj-intel64/pintool.so
export DEEPSEEK_API_KEY=<your-api-key>   # only needed for LLM-backed semantic generation
```

## File organization

### `common/`: shared utilities

- `common.py`: process management, replay helpers, trace preprocessing, and metric utilities.
- `field_units.py`: common field-unit data structures.
- `send.py`: packet sending/replay helper.
- `verify_branch_parsing.py`: branch parsing checker for execution traces.

### `stage1/`: fine-grained field segmentation

Stage 1 recovers byte/bit field boundaries from taint-analysis traces.

- `fields.py`: byte-level field segmentation from taint records and field-related instruction evidence.
- `analyze_bitfields_planA.py`: bit-level consumption tracking and instruction-evidence arbitration.

In normal use, Stage 1 is invoked by `stage2/full.py` during the end-to-end run.

### `stage2/`: strategic execution-difference computation

Stage 2 replays packets, collects traces, segments fields, perturbs fields, computes execution-difference metrics, and builds field behavior summaries.

- `full.py`: main end-to-end driver for replay, tracing, segmentation, perturbation, and differencing.
- `mutate.py`: field perturbation candidate generation.
- `diff.py`: execution-difference metric computation.
- `compare_all_mutations.py`: batch comparison over mutation outputs.
- `build_field_training_samples.py`: converts Stage 2 outputs into field-level training samples.
- `dataset_health_check.py`: checks generated sample quality.
- `debug_metric.py`: inspects individual execution-difference metrics.
- `run_frozen_stage2_protocol.sh`: helper for rerunning Stage 2 on selected protocols.

### `stage3/`: representation-space learning

Stage 3 converts field behavior summaries into low-dimensional field representations.

- `build_stage3_dataset.py`: builds all-field Stage 3 datasets from Stage 2 outputs.
- `filter_stage3_transparent_fields.py`: filters fields with little observable execution difference.
- `build_stage3_training_matrix.py`: normalizes features and creates train/eval matrices.
- `train_stage3_autoencoder.py`: trains and applies autoencoders.
- `check_stage3_dataset.py`: checks Stage 3 datasets and defines the shared 28-D feature columns used by the matrix builder and autoencoder.

### `stage4/`: semantic interpretation and aggregation

Stage 4 implements the last two parts of paper Stage 3: dimension interpretation and field program-semantic aggregation. Spearman evidence selection defaults to up to five features per positive/negative direction with absolute correlation at least 0.30; activation thresholds are the top/bottom 20%.

- `build_stage4_latent_names.py`: computes correlations between representation dimensions and behavior-summary probes.
- `run_stage4_llm_naming.py`: queries an LLM to name/define representation-dimension semantics.
- `build_stage4_field_profiles.py`: identifies activated semantic dimensions for each field.
- `run_stage4_field_semantic_fusion.py`: aggregates activated dimension semantics into field-level program semantics.
- `generate_heldout_packet_split.py`: creates held-out packet splits.

## End-to-end workflow

The following commands show the complete workflow. Replace placeholders such as `<input.pcap>`, `<server>`, and `<output-root>` with protocol-specific paths.

### 1. Run Stage 1/2: trace, segment, perturb, and compute execution differences

`full.py` is the main entry point. It starts the benchmark server under Pin, replays packets from a pcap, collects taint/execution logs, recovers field boundaries, perturbs fields, and computes execution-difference outputs.

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

For UDP protocols such as BACnet, set `--proto udp` and use the corresponding server binary and port.

Useful outputs under `<output-root>/<protocol>/` include per-packet field segmentation results, mutation records, execution-difference metrics, and logs used by later stages.

### 2. Build field behavior summaries for Stage 3

After running Stage 2 for one or more protocols, build a field-level dataset from the Stage 2 output root:

```bash
python difftrace/stage3/build_stage3_dataset.py \
  --input-root <output-root> \
  --output-dir <stage3-dataset-dir>
```

Filter transparent or low-information fields:

```bash
python difftrace/stage3/filter_stage3_transparent_fields.py \
  --input-csv <stage3-dataset-dir>/stage3_dataset_all_fields.csv \
  --output-dir <stage3-filtered-dir>
```

Build normalized training/projection matrices:

```bash
python difftrace/stage3/build_stage3_training_matrix.py \
  --input-csv <stage3-filtered-dir>/stage3_dataset_semantic_fields.csv \
  --output-dir <stage3-matrix-dir>
```

This command does not define an external held-out set. For RQ2's 70%/30% packet split, first prepare the manifest produced by `stage4/generate_heldout_packet_split.py`, then pass `--split-manifest <split.json>` to the matrix builder; its scaler fits the training partition only. The AE's internal 10% validation split is not the RQ2 held-out set.

### 3. Train/project the low-dimensional representation space

Train autoencoders and project fields into the learned representation space:

```bash
python difftrace/stage3/train_stage3_autoencoder.py \
  --input-csv <stage3-matrix-dir>/stage3_training_matrix_train.csv \
  --projection-csv <stage3-matrix-dir>/stage3_training_matrix.csv \
  --output-dir <stage3-ae-dir> \
  --latent-dims 8
```

`ae_embeddings.csv` corresponds to the projection matrix; `train_ae_embeddings.csv` corresponds to training rows. `--eval-csv` additionally produces `eval_ae_embeddings.csv`. Histories, reconstruction errors, and checkpoints are runtime outputs, not bundled files.

The default encoder is 28→16→12→d with a symmetric decoder, ReLU hidden layers, and Sigmoid output. Defaults are MSE, Adam, 150 epochs, batch size 128, learning rate 0.001, weight decay 0.00001, and seed 1337. The script defaults to comparing 4/8/12 dimensions; the example explicitly uses V7's 8-D configuration without asserting experimentally verified optimality.

### 4. Interpret representation dimensions

Compute probe correlations and prepare evidence for LLM-based dimension interpretation:

```bash
python difftrace/stage4/build_stage4_latent_names.py \
  --embeddings <stage3-ae-dir>/ae_latent8/train_ae_embeddings.csv \
  --training-matrix <stage3-matrix-dir>/stage3_training_matrix_train.csv \
  --out-dir <stage4-latent-dir>
```

Generate natural-language semantics for representation dimensions:

```bash
python difftrace/stage4/run_stage4_llm_naming.py \
  --evidence <stage4-latent-dir>/z_topk_probe_evidence.json \
  --out <stage4-latent-dir>/latent_naming_report.md \
  --semantics-out <stage4-latent-dir>/z_axis_semantics.json \
  --api-key $DEEPSEEK_API_KEY
```

Use `--dry-run` to preview prompts without API calls. This call and field fusion default to `deepseek-v4.1-flash`, temperature 0, top-p 1, and no sampling seed. Complete prompts are in their Python files, not dependent on an external prompt image.

### 5. Build field profiles and generate field program semantics

Build field-level activated-dimension profiles:

```bash
python difftrace/stage4/build_stage4_field_profiles.py \
  --embeddings <stage3-ae-dir>/ae_latent8/ae_embeddings.csv \
  --threshold-embeddings <stage3-ae-dir>/ae_latent8/train_ae_embeddings.csv \
  --axis-semantics <stage4-latent-dir>/z_axis_semantics.json \
  --out-dir <stage4-profile-dir>
```

Fuse activated dimension semantics into one field-level program-semantic description:

```bash
python difftrace/stage4/run_stage4_field_semantic_fusion.py \
  --input <stage4-profile-dir>/field_semantic_profiles.jsonl \
  --output-jsonl <stage4-semantic-dir>/field_semantic_fused_profiles.jsonl \
  --output-csv <stage4-semantic-dir>/field_semantic_fused_vectors.csv \
  --backend api \
  --api-key $DEEPSEEK_API_KEY
```

Final outputs include program-semantic descriptions, coarse comparison labels, and auxiliary scores. The 8-D representations remain in AE embedding files. Evaluate semantic outputs with `../tools/`; mappings are in `../tools/sota_evaluation/config/semantic_label_mapping.json`.

## New inputs and implementation limitations

`--threshold-embeddings` calibrates profiles against an existing reference embedding distribution. Omitting it ranks the current input fields themselves; doing so on a single new packet is not fixed-space inference. The scaler, AE, axis interpretations, and reference distribution must come from the same space construction.

`train_stage3_autoencoder.py --projection-csv` encodes another matrix after training in the current run; it is not a standalone checkpoint-loading inference entry point. Complete fixed-space inference for a new private protocol remains to be organized and validated. Also, `relative_start` measures packet-relative position, but its description in `build_stage4_latent_names.py` still incorrectly denotes first-consumption position.

Some defaults, `common.py`'s pintool path, and `run_frozen_stage2_protocol.sh` retain original workspace paths. The frozen helper also invokes an external script. Check these dependencies when relocating; setting `PYTHONPATH` alone does not remove them.

## Debugging tips

- Run any script with `--help` to inspect path options.
- The main driver uses `--sample-count`; dataset construction uses `--samples`/`--limit-samples`, while field fusion uses `--limit`. Options are not interchangeable between scripts.
- Use LLM scripts with `--dry-run` first to verify prompt construction.
- If imports fail, check that `PYTHONPATH` includes all `difftrace/` stage directories as shown above.
