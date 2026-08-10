#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
CASE_NAME="${CASE_NAME:-optimized_matrix}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8092}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
REQUIRE_STAGE1_FULL_DECODE="${REQUIRE_STAGE1_FULL_DECODE:-1}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/${CASE_NAME}}"
SERVER_LOG="${RESULT_DIR}/server.log"

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

git rev-parse HEAD > "${RESULT_DIR}/git_commit.txt"
git status --short > "${RESULT_DIR}/git_status.txt"
npu-smi info > "${RESULT_DIR}/npu_smi.txt" 2>&1 || true
cp "${DEPLOY_CONFIG}" "${RESULT_DIR}/deploy_config.yaml"

DEPLOY_CONFIG="${DEPLOY_CONFIG}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
NPU_DEVICE="${NPU_DEVICE}" \
HOST="${HOST}" \
PORT="${PORT}" \
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
if [[ "${REQUIRE_STAGE1_FULL_DECODE}" == "1" ]]; then
  validate_stage1_full_decode_graph "${SERVER_LOG}"
fi

run_case() {
  local concurrency="$1"
  local prompts="$2"
  local result_filename="seed_tts_c${concurrency}_n${prompts}.json"
  HOST="${HOST}" \
  PORT="${PORT}" \
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  NUM_PROMPTS="${prompts}" \
  MAX_CONCURRENCY="${concurrency}" \
  RESULT_DIR="${RESULT_DIR}" \
  RESULT_FILENAME="${result_filename}" \
  competition/minicpmo_b/scripts/benchmark_seed_tts.sh
  validate_performance_result "${RESULT_DIR}/${result_filename}" "${prompts}"
}

run_case 1 "${C1_PROMPTS:-32}"
run_case 4 "${C4_PROMPTS:-64}"
run_case 8 "${C8_PROMPTS:-128}"
