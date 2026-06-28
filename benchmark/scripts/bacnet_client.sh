#!/usr/bin/env bash
set -u

ROOT_DIR="/root/semvec/bitfield_groundtruth"
BUILD_DIR="${ROOT_DIR}/build_work/bacnet"
BIN_DIR="${ROOT_DIR}/bacnet/bin"
LOG_DIR="${ROOT_DIR}/bacnet/logs"

TARGET_DEVICE_ID="${1:-123}"
TARGET_MAC="${2:-127.0.0.1:47808}"

export BACNET_IFACE="${BACNET_IFACE:-lo}"
export BACNET_IP_PORT="${BACNET_IP_PORT:-47809}"

mkdir -p "${LOG_DIR}"

failures=0

run_step() {
    local name="$1"
    shift
    local log_file="${LOG_DIR}/${name}.log"
    echo "=== ${name}" | tee "${log_file}"
    if "$@" >>"${log_file}" 2>&1; then
        echo "OK ${name}" | tee -a "${log_file}"
    else
        local rc=$?
        echo "FAIL ${name} rc=${rc}" | tee -a "${log_file}"
        failures=$((failures + 1))
    fi
}

run_step "coverage_client" \
    "${BIN_DIR}/bacnet_coverage_client" "${TARGET_DEVICE_ID}" "${TARGET_MAC}"

run_step "coverage_client_repeat" \
    "${BIN_DIR}/bacnet_coverage_client" "${TARGET_DEVICE_ID}" "${TARGET_MAC}"

run_step "coverage_client_repeat_2" \
    "${BIN_DIR}/bacnet_coverage_client" "${TARGET_DEVICE_ID}" "${TARGET_MAC}"

if [[ "${failures}" -gt 0 ]]; then
    echo "Client run finished with ${failures} failed step(s)."
    exit 1
fi

echo "Client run finished successfully."
