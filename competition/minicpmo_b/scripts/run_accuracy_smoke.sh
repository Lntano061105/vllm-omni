#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8096}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-3}"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/tmp/minicpmo_b_daily_omni_smoke}"
VIDEOMME_ROOT="${VIDEOMME_ROOT:-/tmp/minicpmo_b_videomme_smoke}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/accuracy/smoke_all}"
SERVER_LOG="${RESULT_DIR}/server.log"

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

python competition/minicpmo_b/scripts/prepare_accuracy_data.py \
  daily-omni --limit "${SMOKE_PROMPTS}" --destination "${DAILY_OMNI_ROOT}"
python competition/minicpmo_b/scripts/prepare_accuracy_data.py \
  videomme --limit "${SMOKE_PROMPTS}" --destination "${VIDEOMME_ROOT}"

DEPLOY_CONFIG="${DEPLOY_CONFIG}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
NPU_DEVICE="${NPU_DEVICE}" \
HOST="${HOST}" \
PORT="${PORT}" \
ALLOWED_LOCAL_MEDIA_PATH=/tmp \
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

SUITE=daily-omni \
NUM_PROMPTS="${SMOKE_PROMPTS}" \
MAX_CONCURRENCY=1 \
MIN_DAILY_OMNI_ACCURACY=0 \
HOST="${HOST}" \
PORT="${PORT}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT}" \
RESULT_DIR="${RESULT_DIR}/daily-omni" \
competition/minicpmo_b/scripts/benchmark_accuracy.sh

SUITE=videomme \
NUM_PROMPTS="${SMOKE_PROMPTS}" \
MAX_CONCURRENCY=1 \
MIN_VIDEOMME_ACCURACY=0 \
HOST="${HOST}" \
PORT="${PORT}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
VIDEOMME_ROOT="${VIDEOMME_ROOT}" \
RESULT_DIR="${RESULT_DIR}/videomme" \
competition/minicpmo_b/scripts/benchmark_accuracy.sh
