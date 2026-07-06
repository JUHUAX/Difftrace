# DiffTrace Artifact

This artifact contains the DiffTrace implementation, benchmark binaries, the custom pintool source, and scripts for reproducing the main evaluation workflow.

## Directory layout

- `difftrace/`: core DiffTrace pipeline implementation, organized by stage.
- `pintool/`: custom Intel Pin pintool source and build instructions.
- `benchmark/`: compiled benchmark client/server binaries and traffic-capture scripts.
- `experiments/`: scripts for RQ3 and RQ4 ablation experiments.
- `tools/`: helper utilities for TShark parsing, SOTA evaluation, and program-log analysis.
- `MANIFEST.txt`: complete file list of this artifact directory.

## Environment overview

The code is primarily Python plus a C++ Intel Pin pintool. A typical environment needs:

- Linux x86-64.
- Python 3.8+.
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
export MIMO_API_KEY=<your-mimo-api-key>           # program-log pairwise judge, default model MiMo-V2.5-Pro
export MIMO_API_BASE_URL=<your-mimo-base-url>     # required when the MiMo provider uses a custom OpenAI-compatible endpoint
export DEEPSEEK_API_KEY=<your-deepseek-api-key>   # DiffTrace semantic generation, default model deepseek-v4-pro
```

## Typical workflow

1. Build the pintool under `pintool/`.
2. Use benchmark binaries under `benchmark/binaries/` and traffic scripts under `benchmark/scripts/` to run protocol client/server communication and capture traffic.
3. Run DiffTrace Stage 1 and Stage 2 scripts under `difftrace/` to collect traces, segment fields, perturb fields, and compute execution differences.
4. Run Stage 3 and Stage 4 scripts to learn the representation space and generate field program semantics.
5. Use scripts under `tools/` and `experiments/` to run evaluation utilities and ablation experiments.

See the README in each top-level directory for file-level descriptions and example commands.
