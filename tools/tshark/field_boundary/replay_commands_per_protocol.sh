#!/usr/bin/env bash
set -euo pipefail

# 用途：
# 重新重放 6 个协议的所有合法报文，并为每个数据包生成：
# - trace.log
# - trace.preprocessed.log
# - field_layout.txt
# - meta.json
#
# 说明：
# 1. 每条命令都通过 replay_groundtruth_pcap.py 自动启动 pin + pintool + server。
# 2. 输出会写到 RUN_ROOT/outputs/<proto>/pkt_xxxx/ 下。
# 3. 日志会写到 RUN_ROOT/logs/ 下。
# 4. iec61850 的重放应继续使用“剥离到 TCP 直接承载 MMS”的 pcap，
#    也就是 iec61850_client_to_server_only.pcap。
#    original_iso_stack 那份主要用于 Wireshark 分层解析，不适用于当前 raw-MMS server 重放。

ROOT="/root/semvec/bitfield_groundtruth"
TOOL="/root/semvec/pintool_new/obj-intel64/pintool.so"
WORKDIR="/root/semvec"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT}/replay_manual_${TS}"
OUT_ROOT="${RUN_ROOT}/outputs"
LOG_ROOT="${RUN_ROOT}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
echo
echo "下面 6 条命令可以分别执行；也可以按需复制单条运行。"
echo

cat <<EOF
# 1. snap7
python3 "${ROOT}/evaluation_from_tshark/field_boundary/replay_groundtruth_pcap.py" \\
  --pcap "${ROOT}/pcap/snap7_client_to_server_only.pcap" \\
  --proto tcp \\
  --target-host 127.0.0.1 \\
  --target-port 102 \\
  --pin-log "${LOG_ROOT}/snap7_taint_record.log" \\
  --outdir "${OUT_ROOT}/snap7" \\
  --wait-ms 3000 \\
  --taint \\
  --pin-bin pin \\
  --taint-tool "${TOOL}" \\
  --server-bin "${ROOT}/server/snap7_server" \\
  --taint-startup-time 3 \\
  --taint-workdir "${WORKDIR}" \\
  --taint-stdout-log "${LOG_ROOT}/snap7_taint_stdout.log"

# 2. modbus
python3 "${ROOT}/evaluation_from_tshark/field_boundary/replay_groundtruth_pcap.py" \\
  --pcap "${ROOT}/pcap/modbus_client_to_server_only.pcap" \\
  --proto tcp \\
  --target-host 127.0.0.1 \\
  --target-port 502 \\
  --pin-log "${LOG_ROOT}/modbus_taint_record.log" \\
  --outdir "${OUT_ROOT}/modbus" \\
  --wait-ms 3000 \\
  --taint \\
  --pin-bin pin \\
  --taint-tool "${TOOL}" \\
  --server-bin "${ROOT}/server/modbus_server" \\
  --taint-startup-time 3 \\
  --taint-workdir "${WORKDIR}" \\
  --taint-stdout-log "${LOG_ROOT}/modbus_taint_stdout.log"

# 3. bacnet
python3 "${ROOT}/evaluation_from_tshark/field_boundary/replay_groundtruth_pcap.py" \\
  --pcap "${ROOT}/pcap/bacnet_client_to_server_only.pcap" \\
  --proto udp \\
  --target-host 127.0.0.1 \\
  --target-port 47808 \\
  --pin-log "${LOG_ROOT}/bacnet_taint_record.log" \\
  --outdir "${OUT_ROOT}/bacnet" \\
  --wait-ms 3000 \\
  --taint \\
  --pin-bin pin \\
  --taint-tool "${TOOL}" \\
  --server-bin "${ROOT}/server/bacnet_server" \\
  --server-args "123 LoopServer" \\
  --taint-startup-time 4 \\
  --taint-workdir "${WORKDIR}" \\
  --taint-stdout-log "${LOG_ROOT}/bacnet_taint_stdout.log"

# 4. iec104
python3 "${ROOT}/evaluation_from_tshark/field_boundary/replay_groundtruth_pcap.py" \\
  --pcap "${ROOT}/pcap/iec104_client_to_server_only.pcap" \\
  --proto tcp \\
  --target-host 127.0.0.1 \\
  --target-port 2404 \\
  --pin-log "${LOG_ROOT}/iec104_taint_record.log" \\
  --outdir "${OUT_ROOT}/iec104" \\
  --wait-ms 3000 \\
  --taint \\
  --pin-bin pin \\
  --taint-tool "${TOOL}" \\
  --server-bin "${ROOT}/server/iec104_server" \\
  --taint-startup-time 3 \\
  --taint-workdir "${WORKDIR}" \\
  --taint-stdout-log "${LOG_ROOT}/iec104_taint_stdout.log"

# 5. cip
python3 "${ROOT}/evaluation_from_tshark/field_boundary/replay_groundtruth_pcap.py" \\
  --pcap "${ROOT}/pcap/CIP_client_to_server_only.no_tail.pcap" \\
  --proto tcp \\
  --target-host 127.0.0.1 \\
  --target-port 44818 \\
  --pin-log "${LOG_ROOT}/cip_taint_record.log" \\
  --outdir "${OUT_ROOT}/cip" \\
  --wait-ms 3000 \\
  --taint \\
  --pin-bin pin \\
  --taint-tool "${TOOL}" \\
  --server-bin "${ROOT}/server/CIP_server" \\
  --server-args "lo" \\
  --taint-startup-time 3 \\
  --taint-workdir "${WORKDIR}" \\
  --taint-stdout-log "${LOG_ROOT}/cip_taint_stdout.log"

# 6. iec61850
python3 "${ROOT}/evaluation_from_tshark/field_boundary/replay_groundtruth_pcap.py" \\
  --pcap "${ROOT}/pcap/iec61850_client_to_server_only.pcap" \\
  --proto tcp \\
  --target-host 127.0.0.1 \\
  --target-port 8102 \\
  --pin-log "${LOG_ROOT}/iec61850_taint_record.log" \\
  --outdir "${OUT_ROOT}/iec61850" \\
  --wait-ms 3000 \\
  --taint \\
  --pin-bin pin \\
  --taint-tool "${TOOL}" \\
  --server-bin "${ROOT}/server/iec61850_server" \\
  --server-args "8102" \\
  --server-port-check 8102 \\
  --taint-startup-time 18 \\
  --taint-workdir "${WORKDIR}" \\
  --taint-stdout-log "${LOG_ROOT}/iec61850_taint_stdout.log"
EOF
