#!/usr/bin/env bash
set -euo pipefail

# Start one expensive three-stage service and use it for both the performance
# screen and the Seed-TTS accuracy/audio-quality gate. This keeps candidate
# comparisons reproducible while avoiding a second 5-8 minute model startup.

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
CASE_NAME="${CASE_NAME:-candidate}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:?DEPLOY_CONFIG is required}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8094}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
REQUIRE_STAGE1_FULL_DECODE="${REQUIRE_STAGE1_FULL_DECODE:-0}"
PERF_NUM_PROMPTS="${PERF_NUM_PROMPTS:-4}"
PERF_MAX_CONCURRENCY="${PERF_MAX_CONCURRENCY:-1}"
ACCURACY_NUM_PROMPTS="${ACCURACY_NUM_PROMPTS:-3}"
ACCURACY_MAX_CONCURRENCY="${ACCURACY_MAX_CONCURRENCY:-1}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/competition/minicpmo_b/results/candidates/${CASE_NAME}}"
PERF_DIR="${RESULT_ROOT}/performance"
ACCURACY_DIR="${RESULT_ROOT}/accuracy/seed-tts"
AUDIO_DIR="${RESULT_ROOT}/audio"
SERVER_LOG="${RESULT_ROOT}/server.log"

mkdir -p "${PERF_DIR}" "${ACCURACY_DIR}" "${AUDIO_DIR}"
cd "${REPO_ROOT}"

git rev-parse HEAD > "${RESULT_ROOT}/git_commit.txt"
npu-smi info > "${RESULT_ROOT}/npu_smi.txt" 2>&1 || true

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

perf_filename="seed_tts_c${PERF_MAX_CONCURRENCY}_n${PERF_NUM_PROMPTS}.json"
HOST="${HOST}" \
PORT="${PORT}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
NUM_PROMPTS="${PERF_NUM_PROMPTS}" \
MAX_CONCURRENCY="${PERF_MAX_CONCURRENCY}" \
RESULT_DIR="${PERF_DIR}" \
RESULT_FILENAME="${perf_filename}" \
competition/minicpmo_b/scripts/benchmark_seed_tts.sh
validate_performance_result "${PERF_DIR}/${perf_filename}" "${PERF_NUM_PROMPTS}"

SEED_TTS_WER_SAVE_AUDIO_DIR="${AUDIO_DIR}" \
SUITE=seed-tts \
HOST="${HOST}" \
PORT="${PORT}" \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
NUM_PROMPTS="${ACCURACY_NUM_PROMPTS}" \
MAX_CONCURRENCY="${ACCURACY_MAX_CONCURRENCY}" \
NUM_WARMUPS=0 \
RESULT_DIR="${ACCURACY_DIR}" \
competition/minicpmo_b/scripts/benchmark_accuracy.sh

python competition/minicpmo_b/scripts/analyze_audio_quality.py \
  "${AUDIO_DIR}" \
  --output "${RESULT_ROOT}/audio_quality.json"
