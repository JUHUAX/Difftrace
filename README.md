# DiffTrace Artifact

This artifact contains the DiffTrace implementation, benchmark binaries, the custom pintool source, and scripts for reproducing the main evaluation workflow.

The submission package intentionally excludes TG/PG datasets and generated experimental results; the included evaluation scripts consume paths supplied by the user.

## Directory layout

- `difftrace/`: core DiffTrace pipeline implementation, organized by stage.
- `pintool/`: custom Intel Pin pintool source and build instructions.
- `benchmark/`: compiled benchmark client/server binaries and traffic-capture scripts.
- `experiments/`: scripts for RQ3 and RQ4 ablation experiments.
- `tools/`: helper utilities for TShark parsing, SOTA evaluation, and program-log analysis.
- `tests/`: checks using temporary synthetic fixtures, not experimental data.
- `MANIFEST.txt`: artifact file list excluding Git internals and Python caches.

## Environment overview

The code is primarily Python plus a C++ Intel Pin pintool. A typical environment needs:

- Linux x86-64.
- Python 3.10+.
- Python packages listed in `requirements.txt`.
- TShark/Wireshark command-line tools for TShark-based parsing.
- Intel Pin installed separately. See `pintool/README.md`.

Install Python dependencies from the artifact root:

```bash
pip install -r requirements.txt
```

LLM-backed scripts read credentials from environment variables according to their role:

```bash
export OPENAI_API_KEY=<your-openai-api-key>       # program-log ground-truth generation, default model gpt-5.5
export MIMO_API_KEY=<your-mimo-api-key>           # program-log pairwise judge, default model MiMo-V2.6-flash
export MIMO_API_BASE_URL=<your-mimo-base-url>     # required when the MiMo provider uses a custom OpenAI-compatible endpoint
export DEEPSEEK_API_KEY=<your-deepseek-api-key>   # DiffTrace semantic generation, default model deepseek-v4.1-flash
```

## Typical workflow

1. Build the pintool under `pintool/`.
2. Use benchmark binaries under `benchmark/binaries/` and traffic scripts under `benchmark/scripts/` to run protocol client/server communication and capture traffic.
3. Run DiffTrace Stage 1 and Stage 2 scripts under `difftrace/` to collect traces, segment fields, perturb fields, and compute execution differences.
4. Run code directories `stage3/` and `stage4/` to perform the representation learning, dimension interpretation, and aggregation of paper Stage 3. Directory names do not imply a fourth method stage in the paper.
5. Use scripts under `tools/` and `experiments/` to run evaluation utilities and ablation experiments.

Each directory provides English/Chinese instructions: [pipeline](difftrace/README.md), [benchmark](benchmark/README.md), [pintool](pintool/README.md), [evaluation tools](tools/README.md), and [experiments](experiments/README.md). Replace every placeholder path before running commands.

## V7 evaluation and packaging

No packet datasets, TG/PG annotations, results, or model weights are bundled. Files named groundtruth are generation/evaluation code. The 12 precompiled benchmark programs are retained.

RQ1/RQ3 use Accuracy, F1-score, and Perfection, exposed in TG/PG `paper_metrics` and `field_boundary_v7_metrics.csv/.md`. Legacy field-matching metrics are diagnostic only. Protocol averages and pooled micro scores are separate; RQ3 averages protocols with reference bit fields.

RQ4 defaults to 100 shared fields and reports per-field average prompt, completion, reasoning, and total tokens and elapsed time alongside totals. Missing API counters and dry-run measurements are N/A, not measured zeros.

Run `python tools/update_manifest.py` to update the list, or add `--check` to validate it. Exclude `.git/` and caches from distributions without deleting local history. The list describes the local working tree, not uploaded changes. Other scripts retain original experiment paths and still require local configuration.

Run `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v` for data-free checks without LLM calls.

## Verification scope and remaining limitations

Twenty-two temporary-fixture checks have passed, covering TG/PG and SOTA evaluation entry points, boundary metrics, RQ3 summaries, and RQ4 dry runs. This is not six-protocol experimental reproduction or independent-machine end-to-end validation; paper results have not been recomputed. Tests require Python 3.10+; the system default may be older.

Documentation updates do not resolve these implementation issues:

- `relative_start` measures packet-relative position, but the dimension-interpretation code still describes it incorrectly as first-consumption position.
- The complete inference entry point for reusing an existing scaler, AE, axis interpretations, and reference distribution on new protocols remains unorganized. Do not derive a new percentile distribution from a single new packet. The pipeline README explains existing options without claiming validated inference support.
- Some scripts retain original absolute paths; capture helpers also depend on unbundled build directories. Configure them before use rather than assuming a one-command portable workflow.

README and manifest describe artifact contents; publication status must be verified against the remote commit and anonymous artifact page.
