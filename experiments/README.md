# Experiment Drivers

This directory contains experiment drivers for the ablation studies.

## RQ3: bitfield segmentation ablation

Files under `RQ3/` evaluate variants of the bitfield segmentation module:

- `run_rq3_bitfield_ablation.py`: generates outputs for Full, Operation-Driven Recovery, and Flat Evidence Aggregation.
- `evaluate_rq3_bitfield_ablation.py`: computes bitfield detection and bit-subfield boundary metrics.

Example:

```bash
python experiments/RQ3/run_rq3_bitfield_ablation.py \
  --replay-root <stage2_replay_outputs> \
  --outdir <rq3_output_dir> \
  --overwrite

python experiments/RQ3/evaluate_rq3_bitfield_ablation.py \
  --help
```

## RQ4: semantic representation ablation

Files under `RQ4/` evaluate strategic grouping and low-dimensional representation learning:

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
  --backend api
```

LLM-backed scripts read credentials from `DEEPSEEK_API_KEY` or an explicit `--api-key` argument.
