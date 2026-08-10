#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
SUITE="${SUITE:-daily-omni}"
CASE_NAME="${CASE_NAME:-optimized}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8093}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/accuracy/${CASE_NAME}/${SUITE}}"
SERVER_LOG="${RESULT_DIR}/server.log"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/tmp/minicpmo_b_daily_omni}"
VIDEOMME_ROOT="${VIDEOMME_ROOT:-/tmp/minicpmo_b_videomme}"

case "${SUITE}" in
  daily-omni) ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${DAILY_OMNI_ROOT}}" ;;
  videomme) ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${VIDEOMME_ROOT}}" ;;
  seed-tts) ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}" ;;
  *) echo "Unknown SUITE=${SUITE}" >&2; exit 2 ;;
esac

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"
git rev-parse HEAD > "${RESULT_DIR}/git_commit.txt"
npu-smi info > "${RESULT_DIR}/npu_smi.txt" 2>&1 || true

DEPLOY_CONFIG="${DEPLOY_CONFIG}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
NPU_DEVICE="${NPU_DEVICE}" \
HOST="${HOST}" \
PORT="${PORT}" \
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH}" \
competition/minicpmo_b/scripts/start_server.sh > "${SERVER_LOG}" 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
  fi
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source competition/minicpmo_b/scripts/server_utils.sh
wait_for_served_model \
  "${HOST}" "${PORT}" "${SERVED_MODEL_NAME}" \
  "${server_pid}" "${SERVER_LOG}" "${READY_TIMEOUT}"

SUITE="${SUITE}" \
HOST="${HOST}" \
PORT="${PORT}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
RESULT_DIR="${RESULT_DIR}" \
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT}" \
VIDEOMME_ROOT="${VIDEOMME_ROOT}" \
competition/minicpmo_b/scripts/benchmark_accuracy.sh
