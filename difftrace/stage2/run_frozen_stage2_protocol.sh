#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bacnet|cip|iec104|iec61850|modbus|snap7> [extra full.py args...]" >&2
  exit 2
fi

protocol="$1"
shift

out_root="${OUT_ROOT:-/root/semvec/difftrace/out_frozen}"
mkdir -p "$out_root/logs" "$out_root/outputs"

server_args=()
case "$protocol" in
  bacnet)
    target_port=47808
    server_bin=/root/semvec/bitfield_groundtruth/server/bacnet_server
    server_args=(--server-args "123 LoopServer")
    startup_time=4
    ;;
  cip)
    target_port=44818
    server_bin=/root/semvec/bitfield_groundtruth/server/CIP_server
    server_args=(--server-args "lo")
    startup_time=3
    ;;
  iec104)
    target_port=2404
    server_bin=/root/semvec/bitfield_groundtruth/server/iec104_server
    startup_time=3
    ;;
  iec61850)
    target_port=8102
    server_bin=/root/semvec/bitfield_groundtruth/server/iec61850_server
    server_args=(--server-args "8102")
    startup_time=18
    ;;
  modbus)
    target_port=502
    server_bin=/root/semvec/bitfield_groundtruth/server/modbus_server
    startup_time=3
    ;;
  snap7)
    target_port=102
    server_bin=/root/semvec/bitfield_groundtruth/server/snap7_server
    startup_time=3
    ;;
  *)
    echo "unsupported protocol: $protocol" >&2
    exit 2
    ;;
esac

python3 /root/semvec/difftrace/full.py \
  --mode frozen \
  --frozen-protocol "$protocol" \
  --target-host 127.0.0.1 \
  --target-port "$target_port" \
  --pin-log "$out_root/logs/${protocol}.taint_record.log" \
  --outdir "$out_root/outputs/$protocol" \
  --constraint-guided \
  --group-topk 6 \
  --group-min-candidates 6 \
  --cg-rounds 2 \
  --wait-ms 3000 \
  --taint \
  --pin-bin pin \
  --taint-tool /root/semvec/pintool_new/obj-intel64/pintool.so \
  --server-bin "$server_bin" \
  "${server_args[@]}" \
  --taint-kill-existing \
  --taint-max-restarts 3 \
  --packet-retry-on-taint-fail 1 \
  --taint-startup-time "$startup_time" \
  --taint-workdir /root/semvec \
  --taint-stdout-log "$out_root/logs/${protocol}.taint_stdout.log" \
  "$@"
