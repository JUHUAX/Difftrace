# Experiment Reproduction Guide

This directory provides entry points for reproducing the four research-question evaluations. RQ3 and RQ4 have dedicated scripts under this directory. RQ1 and RQ2 reuse the common pipeline and evaluation utilities under `difftrace/` and `tools/`.

Only scripts/documentation are included, not experimental inputs, TG/PG datasets, or results.

Code directories `stage3/` and `stage4/` jointly implement paper Stage 3. Commands are interface examples. The RQ3 generator, RQ4 shuffled pipeline, and some helpers still depend on original workspace paths and external inputs; adapt them before use. Synthetic tests do not establish six-protocol reproduction.

## V7 metric and cost outputs

RQ1/RQ3 use Accuracy, F1-score, and Perfection. TG/PG evaluators write `field_boundary_v7_metrics.csv/.md`; RQ3 averages protocols with reference bit fields. Rerun the updated evaluator on old inputs; do not rename legacy Subfield F1 to boundary F1-score.

RQ4 defaults to 100 shared fields and fails if fewer are available. Set a smaller `--sample-size` explicitly for a pilot. JSON includes per-field `avg_*` costs alongside totals; the Markdown table follows V7. API-reported reasoning tokens are a subset of completion tokens and are not added again to total tokens. Missing counters and dry-run measurements are N/A.

```bash
python experiments/RQ4/measure_rq4_llm_usage.py \
  --full-input <field_semantic_profiles.jsonl> \
  --no-latent-input <stage3_dataset_semantic_fields.csv> \
  --output-dir <cost_output_dir> \
  --sample-size 100
```

Add `--dry-run` to check prompts without LLM calls or measured costs.

Before running experiments, set the Python path from the artifact root:

```bash
cd /path/to/artifact
export PYTHONPATH=$PWD/difftrace/common:$PWD/difftrace/stage1:$PWD/difftrace/stage2:$PWD/difftrace/stage3:$PWD/difftrace/stage4:$PYTHONPATH
```

LLM-backed scripts use task-specific credentials: DiffTrace semantic generation uses `DEEPSEEK_API_KEY`, PG generation uses `OPENAI_API_KEY`, and the program-log pairwise judge uses `MIMO_API_KEY`.

## RQ1: fine-grained field segmentation

RQ1 evaluates byte-level and bit-level field segmentation. The reproduction workflow is:

1. Generate or provide protocol pcaps using benchmark binaries and scripts under `benchmark/`.
2. Run DiffTrace Stage 1/2 to produce field segmentation outputs.
3. Build TShark-derived protocol views using `tools/tshark/`.
4. Evaluate DiffTrace and SOTA outputs using `tools/sota_evaluation/` and program-log utilities where applicable.

Main scripts:

- `difftrace/stage2/full.py`: end-to-end replay, tracing, field segmentation, perturbation, and execution-difference driver.
- `tools/tshark/field_boundary/generate_groundtruthA_tshark.py`: extracts TShark field information from pcaps.
- `tools/tshark/field_boundary/evaluate_tshark_vs_experiment.py`: evaluates field boundaries under the TShark-derived view.
- `tools/program_log/scripts/evaluate_program_log_field_boundary.py`: evaluates field boundaries under the program-log-derived view.
- `tools/sota_evaluation/scripts/evaluate_boundary_predictions.py`: computes shared field-boundary metrics for SOTA outputs.

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

Use `--help` on each script for concrete path options.

Expected outputs:

- DiffTrace replay outputs under `<difftrace_output_dir>/`, including per-packet field files such as `fields.json`, `bitfields.json`, mutation records, and execution-difference reports.
- TShark-derived field views under `<tshark_output_dir>/`, used as TG boundary ground truth.
- The SOTA evaluator writes `boundary_metrics_summary.json/.csv/.md`; TG writes `metrics_summary.json`, and PG writes `field_boundary_metrics_summary.json`. Both TG/PG expose the new main tables as `field_boundary_v7_metrics.csv/.md`, with primary JSON metrics under `paper_metrics`.

## RQ2: field program-semantic inference

RQ2 evaluates field semantic inference under unified traditional labels and program-log semantic agreement. The reproduction workflow is:

1. Run Stage 2 to obtain field behavior summaries.
2. Run Stage 3 to learn/project low-dimensional field representations.
3. Run Stage 4 to generate field program-semantic descriptions.
4. Evaluate traditional-label accuracy with TShark-derived semantic labels.
5. Evaluate program-semantic agreement with the pairwise judge.

For comparison, fixed source-label meaning/keyword mappings produce six labels: `identifier`, `length_or_count`, `control_or_flags`, `addressing`, `data_value`, and `other_or_unknown`. DiffTrace outputs fixed labels alongside natural-language semantics. TG reports label Accuracy; PG reports Strong Agreement (same + mostly same) and Any Overlap (add weakly same), with Strong Agreement as the primary metric.

RQ2 uses a packet-level 70% training/30% held-out split. Generate the manifest with `difftrace/stage4/generate_heldout_packet_split.py` and pass it as the matrix builder's `--split-manifest`. Fit scaling, train the AE, and interpret correlations on training data only. Held-out fields use the same scaling/encoding and training-space activation distribution. The AE's internal 10% validation split selects checkpoints, not the 30% held-out set. Split data, weights, and results are not bundled.

Main scripts:

- `difftrace/stage3/build_stage3_dataset.py`: builds Stage 3 datasets from field behavior summaries.
- `difftrace/stage3/build_stage3_training_matrix.py`: prepares normalized training/projection matrices.
- `difftrace/stage3/train_stage3_autoencoder.py`: trains and applies the autoencoder representation model.
- `difftrace/stage4/build_stage4_latent_names.py`: prepares evidence for representation-dimension interpretation.
- `difftrace/stage4/run_stage4_llm_naming.py`: generates semantic definitions for representation dimensions.
- `difftrace/stage4/build_stage4_field_profiles.py`: builds field-level activated-dimension profiles.
- `difftrace/stage4/run_stage4_field_semantic_fusion.py`: generates final field program-semantic descriptions.
- `tools/tshark/semantic_inference/build_tshark_semantic_groundtruth.py`: builds unified traditional semantic labels from TShark output.
- `tools/tshark/semantic_inference/evaluate_stage4_semantic_predictions.py`: evaluates semantic predictions under traditional labels.
- `tools/program_log/scripts/run_program_log_pairwise_judge.py`: evaluates field program-semantics with a pairwise judge.
- `tools/sota_evaluation/scripts/evaluate_unified_semantics.py`: evaluates SOTA semantic outputs under the unified label space.

Example command shape:

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

Expected outputs:

- Stage 3 representation files, including `stage3_training_matrix.csv`, autoencoder embeddings such as `ae_embeddings.csv`, and training summaries.
- Stage 4 semantic files, including `z_topk_probe_evidence.json`, `z_axis_semantics.json`, `field_semantic_profiles.jsonl`, and final semantic vectors/profiles.
- Evaluation outputs, including traditional-label accuracy summaries and pairwise-judge results such as `judge_results.csv`.

Both generation calls default to `deepseek-v4.1-flash`; initial PG annotation defaults to `gpt-5.5`, and the independent judge to `MiMo-V2.6-flash`. These are invocation settings, not proof that existing results were regenerated with updated models. Initial PG candidates require the paper's review workflow.

## RQ3: bitfield segmentation ablation

Files under `RQ3/` evaluate variants of the bitfield segmentation module:

- `run_rq3_bitfield_ablation.py`: generates outputs for Full, Operation-Driven Recovery, and Flat Evidence Aggregation.
- `evaluate_rq3_bitfield_ablation.py`: reports bit-boundary Accuracy, F1-score, and Perfection in the main table; legacy detection/subfield-matching scores remain JSON diagnostics.

ODR forms local bit-operation candidates without waiting for later consumption to revise them. FEA scans the complete trace but does not arbitrate according to evidence strength. Full uses the complete pipeline.

Example:

```bash
python experiments/RQ3/run_rq3_bitfield_ablation.py \
  --replay-root <stage2_replay_outputs> \
  --outdir <rq3_output_dir> \
  --overwrite

python experiments/RQ3/evaluate_rq3_bitfield_ablation.py \
  --outdir <rq3_output_dir> \
  --groundtruth-jsonl <verified_pg.jsonl>
```

Expected outputs:

- Ablation outputs under `<rq3_output_dir>/<mode>/`, where each mode contains per-packet `bitfields.json` files.
- Generation metadata in `rq3_generation_manifest.json`.
- Metric summaries under `<rq3_output_dir>/metrics/<mode>/`, plus `rq3_bitfield_ablation_summary.json` and `rq3_bitfield_ablation_summary.md`.

`--skip-evaluate` can rebuild a summary only after the updated evaluator has produced `paper_metrics`; it does not convert old results to new metrics. Protocols without reference bit fields are excluded from the main protocol average. Pooled micro scores are reported separately.

## RQ4: semantic representation ablation

Files under `RQ4/` evaluate strategic grouping and low-dimensional representation learning:

Shuffled groups keeps candidate values/differences but shuffles strategy membership. No-latent direct keeps the 28-D summaries but skips AE and latent interpretation. Compare the same fields with the same generation model and evaluation reference.

- `build_rq4_shuffled_group_dataset.py`: builds shuffled-group behavior summaries.
- `run_rq4_shuffled_group_pipeline.py`: runs the shuffled-groups ablation pipeline.
- `run_rq4_no_latent_direct_summary.py`: runs the no-latent direct semantic-generation baseline.
- `measure_rq4_llm_usage.py`: measures LLM invocation cost on sampled fields.
- `summarize_program_log_judge_on_rq2b_eval.py`: summarizes pairwise-judge outputs.

Example:

```bash
python experiments/RQ4/run_rq4_shuffled_group_pipeline.py \
  --seed 0 \
  --out-root <rq4_output_dir> \
  --latent-dim 8 \
  --backend api
```

Expected outputs:

- Shuffled-group datasets such as `stage3_dataset_semantic_fields.csv` and `rq4_shuffled_group_manifest.json`.
- Shuffled pipeline artifacts under `<rq4_output_dir>/shuffled_seed_<seed>/`, including Stage 3 matrices, AE embeddings, dimension semantics, field profiles, and fused semantic outputs.
- No-latent direct outputs such as `field_semantic_direct_profiles.jsonl` and `field_semantic_direct_vectors.csv`.
- Per-call cost CSV, `rq4_llm_usage_measurement_summary.json` with `avg_*` and totals, and the per-field-average Markdown table.
- Program-log judge summaries such as `rq4_program_log_judge_on_rq2b_eval_summary.csv`, per-protocol CSVs, and readable Markdown reports.

This revision made no LLM calls or real-experiment reruns and does not change V7 result values. All listed outputs are generated at runtime, not bundled. Use an explicit small pilot and inspect its quality before a paid full run.
