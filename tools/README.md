# Evaluation Utilities

This directory contains helper utilities for parsing traffic, evaluating baseline outputs, and running program-log-based analyses.

Only code and label-mapping configuration are included: no TG/PG datasets, baseline predictions, judge outputs, or results. Run examples from the artifact root and replace every placeholder; unspecified defaults may still point to the original experiment environment.

## V7 field-boundary metrics

`boundary_metrics.py` computes byte/bit Accuracy, F1-score, and Perfection over all possible gap positions, including endpoints. Perfection requires both reference-field endpoints and no extra interior predicted boundary. Bit evaluation preserves parent identity and uses the union of reference/predicted parents as its candidate domain, retaining prediction-only parents.

TG/PG evaluators expose `paper_metrics` and `field_boundary_v7_metrics.csv/.md`; legacy scores are diagnostic. The SOTA main table/CSV exposes Accuracy, F1-score, and Perfection. `Protocol Value Avg` averages protocols with reference fields at that granularity; `Overall Micro` pools counts. Do not mix them. Protocols without reference bit fields are excluded from bit-table averages; legacy Macro diagnostics are not the paper table.

Inputs and annotations are supplied at runtime, not bundled. `update_manifest.py` updates/checks the list without Git internals or caches.

## Ground-truth construction workflows

The paper uses complementary references: TShark-derived ground truth (TG) and program-log reference annotations (PG). TG reflects protocol parsing; PG complements it with observed consumption in a concrete implementation, not unique protocol-specification truth. Use the utilities as follows.

### Build TShark-derived ground truth (TG)

TG is constructed from TShark's structured packet parsing and is used for the protocol-specification/dissector perspective.

1. Parse pcaps into structured per-packet TShark JSON files:

   ```bash
   python tools/tshark/field_boundary/generate_groundtruthA_tshark.py \
     --pcap bacnet=<bacnet.pcap> \
     --pcap cip=<cip.pcap> \
     --output-dir <tshark_json_dir>
   ```

   For IEC61850/MMS captures, use `field_boundary/strip_iec61850_iso_to_mms_pcap.py` first if the replay/evaluation pipeline expects MMS payloads without the full ISO stack.

2. Export byte-field boundary ground truth aligned with replay sample IDs:

   ```bash
   python tools/sota_evaluation/scripts/export_tshark_boundary_groundtruth.py \
     --tshark-root <tshark_json_dir> \
     --replay-root <difftrace_replay_outputs> \
     --output <tg_boundary_groundtruth.jsonl>
   ```

3. Build traditional semantic-label candidates from TShark fields:

   ```bash
   python tools/tshark/semantic_inference/build_tshark_semantic_groundtruth.py \
     --input-dir <tshark_json_dir> \
     --output-dir <tg_semantic_dir>
   ```

   The output CSV is a rule-mapped candidate file and should be manually reviewed before being used as the final unified-label semantic ground truth.

### Build Program-log-derived ground truth (PG)

PG is constructed from taint execution logs and is used for the program-execution perspective. It reflects how the tested implementation consumes input bytes, rather than whether a result matches the protocol specification.

1. Obtain replay logs from the DiffTrace/Pin replay pipeline. Each packet directory should contain the execution log and metadata produced during replay.

2. Preprocess raw program logs:

   ```bash
   python tools/program_log/scripts/preprocess_program_logs.py \
     --input-root <difftrace_replay_outputs> \
     --output-dir <preprocessed_log_dir> \
     --overwrite
   ```

3. Generate initial PG candidates with the LLM ground-truth builder:

   ```bash
   python tools/program_log/scripts/run_program_log_groundtruth_llm.py \
     --input-dir <preprocessed_log_dir> \
     --output-dir <pg_llm_output_dir> \
     --backend api \
     --api-key $OPENAI_API_KEY
   ```

   These outputs are initial candidates. The logs remove protocol names and function names where possible, and the prompt forbids using protocol specifications or model prior knowledge.

4. Collect per-log LLM outputs into structured JSONL/CSV files:

   ```bash
   python tools/program_log/scripts/collect_program_log_groundtruth_outputs.py \
     --input-dir <pg_llm_output_dir> \
     --output-dir <pg_eval_dir> \
     --replay-root <difftrace_replay_outputs>
   ```

5. Manually verify and correct the collected candidates. The review should check whether the field boundaries and program-semantics follow the execution logs, not whether they match protocol-specification labels. The helper `program_log/scripts/fill_manual_groundtruth_semantics.py` can be used when completing semantic entries.

The final PG files are typically consumed by `evaluate_program_log_field_boundary.py` for boundary evaluation and by `run_program_log_pairwise_judge.py` for program-semantic agreement evaluation.

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

They do not bundle implementations of all seven baselines; provide each tool's outputs separately. Fixed mapping follows source-label meanings/keywords. Unified labels are `identifier`, `length_or_count`, `control_or_flags`, `addressing`, `data_value`, and `other_or_unknown`.

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
  --stage4-profiles <field_semantic_fused_profiles.jsonl> \
  --output-csv <judge_results.csv> \
  --output-md <judge_report.md> \
  --run-log <judge_run_log.jsonl> \
  --backend api
```

LLM-backed program-log scripts use different default backends by task:

- PG generation scripts use `gpt-5.5` and read `OPENAI_API_KEY` or an explicit `--api-key`.
- The pairwise judge uses `MiMo-V2.6-flash` and reads `MIMO_API_KEY` or an explicit `--api-key`; set `MIMO_API_BASE_URL` or pass `--api-base-url` if the provider requires a custom endpoint.

Here `--stage4-profiles` must point to fused outputs containing final field program semantics, not the profile containing only activated-axis explanations. `same_behavior` plus `mostly_same_behavior` forms Strong Agreement; adding `weakly_same_behavior` gives Any Overlap. `different_behavior` and `insufficient_information` contribute to neither.

LLM commands incur API costs. Do not assume the PG builder or judge supports `--dry-run`; use it only where implemented. Judge `--report-only` reports existing results without rerunning judgments. Aggregate five-verdict outputs with `../experiments/RQ4/summarize_program_log_judge_on_rq2b_eval.py`. PG candidate generation does not establish completion of the paper's manual review.
