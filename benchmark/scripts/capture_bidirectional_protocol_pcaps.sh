#!/usr/bin/env bash
set -euo pipefail

BASE="/root/semvec/bitfield_groundtruth"
PCAP_DIR="${BASE}/pcap"
LOG_DIR="${BASE}/run_logs/bidirectional_capture"
COUNT_FILE="${PCAP_DIR}/bidirectional_packet_counts.tsv"

mkdir -p "${PCAP_DIR}" "${LOG_DIR}"
mkdir -p "${BASE}/iec61850/runtime/mms_server_filestore" \
         "${BASE}/iec61850/runtime/client_filestore" \
         "${BASE}/iec61850/runtime/downloads"
printf 'seed\n' > "${BASE}/iec61850/runtime/mms_server_filestore/mms_seed.txt"

cp "${BASE}/generated_src/bacnet/bacnet_client.sh" "${BASE}/bacnet/bin/bacnet_client.sh"
chmod +x "${BASE}/bacnet/bin/bacnet_client.sh"

server_pid=""
tcpdump_pid=""

stop_active_processes() {
  if [[ -n "${tcpdump_pid}" ]]; then
    kill "${tcpdump_pid}" >/dev/null 2>&1 || true
    wait "${tcpdump_pid}" >/dev/null 2>&1 || true
    tcpdump_pid=""
  fi
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    server_pid=""
  fi
}

cleanup_servers() {
  stop_active_processes
  pkill -f 'bacnet/server' >/dev/null 2>&1 || true
  pkill -f 'CIP_server' >/dev/null 2>&1 || true
  pkill -f 'iec104_server' >/dev/null 2>&1 || true
  pkill -f 'iec61850_server' >/dev/null 2>&1 || true
  pkill -f 'modbus_server' >/dev/null 2>&1 || true
  pkill -f 'snap7_server' >/dev/null 2>&1 || true
  sleep 1
}

trap cleanup_servers EXIT
trap 'cleanup_servers; exit 130' INT TERM

count_packets() {
  tcpdump -nn -r "$1" 2>/dev/null | wc -l | tr -d ' '
}

count_payload_packets() {
  tshark -r "$1" -Y 'tcp.len > 0 || udp.length > 8' -T fields -e frame.number 2>/dev/null \
    | wc -l | tr -d ' '
}

run_protocol() {
  local name="$1"
  local filter_expr="$2"
  local server_cmd="$3"
  local client_cmd="$4"

  local pcap_path="${PCAP_DIR}/${name}_bidirectional.pcap"
  local pcap_tmp="${pcap_path}.tmp"
  local server_log="${LOG_DIR}/${name}_server.log"
  local client_log="${LOG_DIR}/${name}_client.log"
  local tcpdump_log="${LOG_DIR}/${name}_tcpdump.log"

  printf '\n[capture] start protocol=%s\n' "${name}"
  rm -f "${pcap_tmp}" "${server_log}" "${client_log}" "${tcpdump_log}"
  cleanup_servers

  stdbuf -oL -eL bash -lc "${server_cmd}" >"${server_log}" 2>&1 &
  server_pid=$!
  sleep 2

  stdbuf -oL -eL tcpdump -i lo -U -nn -s 0 -w "${pcap_tmp}" ${filter_expr} \
    >"${tcpdump_log}" 2>&1 &
  tcpdump_pid=$!
  sleep 1

  set +e
  timeout 90 bash -lc "${client_cmd}" >"${client_log}" 2>&1
  local client_rc=$?
  set -e

  sleep 2
  stop_active_processes
  mv -f "${pcap_tmp}" "${pcap_path}"

  local packet_count
  local payload_packet_count
  packet_count="$(count_packets "${pcap_path}")"
  payload_packet_count="$(count_payload_packets "${pcap_path}")"
  printf '%s\t%s\t%s\t%s\n' \
    "${name}" "${packet_count}" "${payload_packet_count}" "${client_rc}" >> "${COUNT_FILE}"
  printf '[capture] done protocol=%s packets=%s payload_packets=%s client_rc=%s output=%s\n' \
    "${name}" "${packet_count}" "${payload_packet_count}" "${client_rc}" "${pcap_path}"
}

capture_bacnet() {
  run_protocol \
    "bacnet" \
    "udp port 47808 or udp port 47809" \
    "cd '${BASE}' && export BACNET_IFACE=lo BACNET_IP_PORT=47808 && exec ./build_work/bacnet/server 123" \
    "cd '${BASE}' && export BACNET_IFACE=lo BACNET_IP_PORT=47809 && exec ./bacnet/bin/bacnet_client.sh 123 127.0.0.1:47808"
}

capture_cip() {
  run_protocol \
    "CIP" \
    "tcp port 44818 or udp port 44818 or udp port 2222" \
    "cd '${BASE}' && exec ./build_work/CIP/CIP_server lo" \
    "cd '${BASE}' && exec ./build_work/CIP/CIP_client 127.0.0.1"
}

capture_iec104() {
  run_protocol \
    "iec104" \
    "tcp port 2404" \
    "cd '${BASE}' && exec ./build_work/iec104_copy/iec104_server" \
    "cd '${BASE}' && exec ./build_work/iec104_copy/iec104_client 127.0.0.1 2404"
}

capture_iec61850() {
  local source_pcap="${PCAP_DIR}/iec61850_full_iso_stack_recapture.pcap"
  local output_pcap="${PCAP_DIR}/iec61850_bidirectional.pcap"

  if [[ ! -f "${source_pcap}" ]]; then
    printf '[capture] missing IEC61850 ISO-stack reference capture: %s\n' "${source_pcap}" >&2
    exit 1
  fi

  # The active Semvec IEC61850 server intentionally accepts stripped raw MMS
  # replay payloads. It cannot negotiate the ISO stack used by the original
  # client. Keep using the previously captured full ISO-stack conversation for
  # offline tools such as FieldHunter.
  cp -f "${source_pcap}" "${output_pcap}"

  local packet_count
  local payload_packet_count
  packet_count="$(count_packets "${output_pcap}")"
  payload_packet_count="$(count_payload_packets "${output_pcap}")"
  printf '%s\t%s\t%s\t%s\n' \
    "iec61850" "${packet_count}" "${payload_packet_count}" "reused_iso_stack_capture" >> "${COUNT_FILE}"
  printf '[capture] reuse protocol=iec61850 packets=%s payload_packets=%s output=%s\n' \
    "${packet_count}" "${payload_packet_count}" "${output_pcap}"
}

capture_modbus() {
  run_protocol \
    "modbus" \
    "tcp port 502" \
    "cd '${BASE}' && exec ./build_work/modbus/modbus_server 33" \
    "cd '${BASE}' && exec ./build_work/modbus/modbus_client 33"
}

capture_snap7() {
  run_protocol \
    "snap7" \
    "tcp port 102" \
    "cd '${BASE}' && (sleep 120) | ./build_work/snap7/snap7_server" \
    "cd '${BASE}' && exec ./build_work/snap7/snap7_client 30"
}

usage() {
  cat <<'EOF'
Usage:
  ./capture_bidirectional_protocol_pcaps.sh all
  ./capture_bidirectional_protocol_pcaps.sh <protocol> [<protocol> ...]

Protocols:
  modbus cip snap7 bacnet iec104 iec61850
EOF
}

capture_one() {
  case "${1,,}" in
    modbus) capture_modbus ;;
    cip) capture_cip ;;
    snap7) capture_snap7 ;;
    bacnet) capture_bacnet ;;
    iec104) capture_iec104 ;;
    iec61850) capture_iec61850 ;;
    *)
      printf 'Unknown protocol: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

printf 'protocol\tpacket_count\tpayload_packet_count\tclient_rc\n' > "${COUNT_FILE}"

if [[ "$#" -eq 1 && "${1,,}" == "all" ]]; then
  for protocol in modbus cip snap7 bacnet iec104 iec61850; do
    capture_one "${protocol}"
  done
else
  for protocol in "$@"; do
    capture_one "${protocol}"
  done
fi

cleanup_servers
trap - EXIT INT TERM
printf '\n[capture] all requested protocols completed. Summary: %s\n' "${COUNT_FILE}"
