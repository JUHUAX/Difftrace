#!/usr/bin/env bash
set -euo pipefail

BASE="/root/semvec/bitfield_groundtruth"
PCAP_DIR="${BASE}/pcap"
SERVER_DIR="${BASE}/server"
LOG_DIR="${BASE}/run_logs"
COUNT_FILE="${BASE}/packet_counts.tsv"

mkdir -p "${PCAP_DIR}" "${SERVER_DIR}" "${LOG_DIR}"
mkdir -p "${BASE}/iec61850/runtime/mms_server_filestore" \
         "${BASE}/iec61850/runtime/client_filestore" \
         "${BASE}/iec61850/runtime/downloads"
printf 'seed\n' > "${BASE}/iec61850/runtime/mms_server_filestore/mms_seed.txt"

cp "${BASE}/generated_src/bacnet/bacnet_client.sh" "${BASE}/bacnet/bin/bacnet_client.sh"
chmod +x "${BASE}/bacnet/bin/bacnet_client.sh"

cleanup_servers() {
  pkill -f 'bacnet/server' >/dev/null 2>&1 || true
  pkill -f 'CIP_server' >/dev/null 2>&1 || true
  pkill -f 'iec104_server' >/dev/null 2>&1 || true
  pkill -f 'iec61850_server' >/dev/null 2>&1 || true
  pkill -f 'modbus_server' >/dev/null 2>&1 || true
  pkill -f 'snap7_server' >/dev/null 2>&1 || true
  sleep 1
}

count_packets() {
  tcpdump -nn -r "$1" 2>/dev/null | wc -l | tr -d ' '
}

payload_capture_bpf() {
  local port="$1"
  printf "tcp and dst port %s and (((ip[2:2] - ((ip[0] & 0x0f) << 2)) - ((tcp[12] & 0xf0) >> 2)) > 0)" "${port}"
}

run_protocol() {
  local name="$1"
  local filter_expr="$2"
  local server_cmd="$3"
  local client_cmd="$4"
  local server_src="$5"

  local pcap_path="${PCAP_DIR}/${name}_client_to_server_only.pcap"
  local server_log="${LOG_DIR}/${name}_server.log"
  local client_log="${LOG_DIR}/${name}_client.log"
  local tcpdump_log="${LOG_DIR}/${name}_tcpdump.log"

  rm -f "${pcap_path}" "${server_log}" "${client_log}" "${tcpdump_log}"
  cleanup_servers

  stdbuf -oL -eL bash -lc "${server_cmd}" >"${server_log}" 2>&1 &
  local server_pid=$!
  sleep 2

  stdbuf -oL -eL bash -lc "exec tcpdump -i lo -U -nn -s 0 -w '${pcap_path}' ${filter_expr}" \
    >"${tcpdump_log}" 2>&1 &
  local tcpdump_pid=$!
  sleep 1

  set +e
  timeout 90 bash -lc "${client_cmd}" >"${client_log}" 2>&1
  local client_rc=$?
  set -e

  sleep 2
  kill "${tcpdump_pid}" >/dev/null 2>&1 || true
  wait "${tcpdump_pid}" >/dev/null 2>&1 || true
  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true

  cp "${server_src}" "${SERVER_DIR}/${name}_server"
  chmod +x "${SERVER_DIR}/${name}_server"

  local packet_count
  packet_count="$(count_packets "${pcap_path}")"
  printf '%s\t%s\t%s\n' "${name}" "${packet_count}" "${client_rc}" >> "${COUNT_FILE}"
}

run_cip_protocol() {
  local name="CIP"
  local server_cmd="cd '${BASE}' && exec ./build_work/CIP/CIP_server lo"
  local client_cmd="cd '${BASE}' && exec ./build_work/CIP/CIP_client 127.0.0.1"
  local server_src="${BASE}/build_work/CIP/CIP_server"

  local full_pcap_path="${PCAP_DIR}/${name}_full_capture.pcap"
  local pcap_path="${PCAP_DIR}/${name}_client_to_server_only.pcap"
  local server_log="${LOG_DIR}/${name}_server.log"
  local client_log="${LOG_DIR}/${name}_client.log"
  local tcpdump_log="${LOG_DIR}/${name}_tcpdump.log"

  rm -f "${full_pcap_path}" "${pcap_path}" "${server_log}" "${client_log}" "${tcpdump_log}"
  cleanup_servers

  stdbuf -oL -eL bash -lc "${server_cmd}" >"${server_log}" 2>&1 &
  local server_pid=$!
  sleep 2

  stdbuf -oL -eL bash -lc "exec tcpdump -i lo -U -nn -s 0 -w '${full_pcap_path}'" \
    >"${tcpdump_log}" 2>&1 &
  local tcpdump_pid=$!
  sleep 1

  set +e
  timeout 90 bash -lc "${client_cmd}" >"${client_log}" 2>&1
  local client_rc=$?
  set -e

  sleep 2
  kill "${tcpdump_pid}" >/dev/null 2>&1 || true
  wait "${tcpdump_pid}" >/dev/null 2>&1 || true
  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true

  tshark -r "${full_pcap_path}" \
    -Y 'tcp.dstport == 44818 || udp.dstport == 44818 || udp.dstport == 2222' \
    -w "${pcap_path}"

  cp "${server_src}" "${SERVER_DIR}/${name}_server"
  chmod +x "${SERVER_DIR}/${name}_server"

  local packet_count
  packet_count="$(count_packets "${pcap_path}")"
  printf '%s\t%s\t%s\n' "${name}" "${packet_count}" "${client_rc}" >> "${COUNT_FILE}"
}

printf 'protocol\tpacket_count\tclient_rc\n' > "${COUNT_FILE}"

run_protocol \
  "bacnet" \
  "'udp and src port 47809 and dst port 47808'" \
  "cd '${BASE}' && export BACNET_IFACE=lo BACNET_IP_PORT=47808 && exec ./build_work/bacnet/server 123" \
  "cd '${BASE}' && export BACNET_IFACE=lo BACNET_IP_PORT=47809 && exec ./bacnet/bin/bacnet_client.sh 123 127.0.0.1:47808" \
  "${BASE}/build_work/bacnet/server"

run_cip_protocol

run_protocol \
  "iec104" \
  "'tcp and dst port 2404'" \
  "cd '${BASE}' && exec ./build_work/iec104_copy/iec104_server" \
  "cd '${BASE}' && exec ./build_work/iec104_copy/iec104_client 127.0.0.1 2404" \
  "${BASE}/build_work/iec104_copy/iec104_server"

run_protocol \
  "iec61850" \
  "'tcp and dst port 102'" \
  "cd '${BASE}' && exec ./build_work/iec61850/iec61850_server" \
  "cd '${BASE}' && exec ./build_work/iec61850/iec61850_client 127.0.0.1 102" \
  "${BASE}/build_work/iec61850/iec61850_server"

run_protocol \
  "modbus" \
  "\"$(payload_capture_bpf 502)\"" \
  "cd '${BASE}' && exec ./build_work/modbus/modbus_server 33" \
  "cd '${BASE}' && exec ./build_work/modbus/modbus_client 33" \
  "${BASE}/build_work/modbus/modbus_server"

run_protocol \
  "snap7" \
  "\"$(payload_capture_bpf 102)\"" \
  "cd '${BASE}' && (sleep 120) | ./build_work/snap7/snap7_server" \
  "cd '${BASE}' && exec ./build_work/snap7/snap7_client 30" \
  "${BASE}/build_work/snap7/snap7_server"

cleanup_servers
