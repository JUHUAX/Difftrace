#!/usr/bin/env bash
set -euo pipefail

REPLAY_ROOT="${1:-/root/semvec/bitfield_groundtruth/replay_manual_latest/outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=("${REPLAY_ROOT}")

if [[ "${FORCE:-0}" == "1" ]]; then
  args+=("--force")
fi

if [[ -n "${PROTOCOL:-}" ]]; then
  args+=("--protocol" "${PROTOCOL}")
fi

if [[ -n "${FIELDS_SCRIPT:-}" ]]; then
  args+=("--fields-script" "${FIELDS_SCRIPT}")
fi

if [[ -n "${BITFIELD_SCRIPT:-}" ]]; then
  args+=("--bitfield-script" "${BITFIELD_SCRIPT}")
fi

if [[ "${SKIP_BITFIELDS:-0}" == "1" ]]; then
  args+=("--skip-bitfields")
fi

if [[ "${FILL_GAPS:-0}" == "1" ]]; then
  args+=("--fill-gaps")
fi

exec python3 "${SCRIPT_DIR}/analyze_replay_logs.py" "${args[@]}"
