# Evaluation Utilities

This directory contains helper utilities for parsing traffic, evaluating baseline outputs, and running program-log-based analyses.

## `tshark/`

TShark utilities parse captured pcaps and construct or evaluate TShark-derived protocol views.

- `field_boundary/generate_groundtruthA_tshark.py`: extracts structured per-packet TShark field data.
- `field_boundary/strip_iec61850_iso_to_mms_pcap.py`: helper for converting IEC61850 captures to MMS-level pcaps.
- `field_boundary/replay_groundtruth_pcap.py`: replays packets for alignment and analysis.
- `field_boundary/analyze_replay_logs.py`: analyzes replay logs.
- `field_boundary/evaluate_tshark_vs_experiment.py`: evaluates field-boundary predictions against TShark-derived data.
- `semantic_inference/build_tshark_semantic_groundtruth.py`: builds coarse-grained semantic labels from TShark fields.
- `semantic_inference/evaluate_stage4_semantic_predictions.py`: evaluates predicted field semantics under the unified label space.

Example:

```bash
python tools/tshark/field_boundary/generate_groundtruthA_tshark.py \
  --pcap bacnet=<bacnet.pcap> \
  --output-dir <tshark_output_dir>
```

## `sota_evaluation/`

SOTA evaluation utilities normalize baseline outputs and compute shared metrics.

- `config/semantic_label_mapping.json`: unified coarse-grained semantic label mapping.
- `scripts/export_boundary_predictions.py`: exports boundary predictions into the common format.
- `scripts/export_tshark_boundary_groundtruth.py`: exports TShark-derived boundary data.
- `scripts/evaluate_boundary_predictions.py`: computes field-boundary metrics.
- `scripts/export_unified_semantic_candidates.py`: exports semantic candidates into the unified label space.
- `scripts/evaluate_unified_semantics.py`: evaluates semantic predictions.
- `scripts/evaluate_fsibp_native.py`: evaluates FSIBP outputs.
- `scripts/export_tshark_sample_alignment.py`, `scripts/export_program_log_semantic_pairs.py`: alignment/export helpers.

Example:

```bash
python tools/sota_evaluation/scripts/evaluate_boundary_predictions.py \
  --predictions <boundary_predictions.jsonl> \
  --groundtruth <boundary_groundtruth.jsonl> \
  --outdir <metrics_output_dir>
```

## `program_log/`

Program-log utilities preprocess execution logs and run program-execution-perspective analyses.

- `scripts/preprocess_program_logs.py`: preprocesses raw execution logs.
- `scripts/run_program_log_groundtruth_llm.py`: runs LLM-based program-log analysis.
- `scripts/collect_program_log_groundtruth_outputs.py`: collects LLM outputs into structured files.
- `scripts/evaluate_program_log_field_boundary.py`: evaluates field boundaries from the program-log perspective.
- `scripts/run_program_log_pairwise_judge.py`: runs the pairwise semantic judge.
- `scripts/fill_manual_groundtruth_semantics.py`: helper for completing semantic entries.

Example:

```bash
python tools/program_log/scripts/run_program_log_pairwise_judge.py \
  --program-log-jsonl <program_log_semantics.jsonl> \
  --stage4-profiles <field_semantic_profiles.jsonl> \
  --output-csv <judge_results.csv> \
  --backend api
```

LLM-backed scripts use `DEEPSEEK_API_KEY` or an explicit `--api-key` argument.
