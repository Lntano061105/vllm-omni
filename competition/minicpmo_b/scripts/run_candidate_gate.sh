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
REQUIRE_STAGE1_CPU_SLOT_MAPPING="${REQUIRE_STAGE1_CPU_SLOT_MAPPING:-0}"
REQUIRE_STAGE1_GRAPH_SAMPLER="${REQUIRE_STAGE1_GRAPH_SAMPLER:-0}"
REQUIRE_STAGE1_BINARY_ARGMAX="${REQUIRE_STAGE1_BINARY_ARGMAX:-0}"
REQUIRE_STAGE0_REF_CACHE="${REQUIRE_STAGE0_REF_CACHE:-0}"
REQUIRE_STAGE2_PROMPT_CACHE="${REQUIRE_STAGE2_PROMPT_CACHE:-0}"
REQUIRE_STAGE2_NPUGRAPH="${REQUIRE_STAGE2_NPUGRAPH:-0}"
MIN_NPUGRAPH_BUCKETS="${MIN_NPUGRAPH_BUCKETS:-1}"
DUPLEX_GATE_PROFILE="${DUPLEX_GATE_PROFILE:-}"
BASELINE_DUPLEX_RESULT="${BASELINE_DUPLEX_RESULT:-}"
DUPLEX_NUM_PROMPTS="${DUPLEX_NUM_PROMPTS:-4}"
DUPLEX_MAX_CONCURRENCY="${DUPLEX_MAX_CONCURRENCY:-1}"
DUPLEX_NUM_WARMUPS="${DUPLEX_NUM_WARMUPS:-1}"
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

if [[ -n "${DUPLEX_GATE_PROFILE}" ]]; then
  if [[ -z "${BASELINE_DUPLEX_RESULT}" ]]; then
    echo "BASELINE_DUPLEX_RESULT is required when DUPLEX_GATE_PROFILE is set" >&2
    exit 2
  fi
  duplex_dir="${RESULT_ROOT}/duplex"
  duplex_filename="native_duplex_c${DUPLEX_MAX_CONCURRENCY}_n${DUPLEX_NUM_PROMPTS}.json"
  HOST="${HOST}" \
  PORT="${PORT}" \
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  NUM_PROMPTS="${DUPLEX_NUM_PROMPTS}" \
  MAX_CONCURRENCY="${DUPLEX_MAX_CONCURRENCY}" \
  NUM_WARMUPS="${DUPLEX_NUM_WARMUPS}" \
  RESULT_DIR="${duplex_dir}" \
  RESULT_FILENAME="${duplex_filename}" \
  competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh

  python competition/minicpmo_b/scripts/gate_duplex_candidate.py \
    "${BASELINE_DUPLEX_RESULT}" \
    "${duplex_dir}/${duplex_filename}" \
    --profile "${DUPLEX_GATE_PROFILE}" \
    --output "${RESULT_ROOT}/duplex_gate.json"
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

log_gate_args=(
  "${SERVER_LOG}"
  --min-npugraph-buckets "${MIN_NPUGRAPH_BUCKETS}"
  --output "${RESULT_ROOT}/activation_gate.json"
)
if [[ "${REQUIRE_STAGE1_FULL_DECODE}" == "1" ]]; then
  log_gate_args+=(--require-stage1-full-decode)
fi
if [[ "${REQUIRE_STAGE1_CPU_SLOT_MAPPING}" == "1" ]]; then
  log_gate_args+=(--require-stage1-cpu-slot-mapping)
fi
if [[ "${REQUIRE_STAGE1_GRAPH_SAMPLER}" == "1" ]]; then
  log_gate_args+=(--require-stage1-graph-sampler)
fi
if [[ "${REQUIRE_STAGE1_BINARY_ARGMAX}" == "1" ]]; then
  log_gate_args+=(--require-stage1-binary-argmax)
fi
if [[ "${REQUIRE_STAGE0_REF_CACHE}" == "1" ]]; then
  log_gate_args+=(--require-stage0-ref-cache)
fi
if [[ "${REQUIRE_STAGE2_PROMPT_CACHE}" == "1" ]]; then
  log_gate_args+=(--require-stage2-prompt-cache)
fi
if [[ "${REQUIRE_STAGE2_NPUGRAPH}" == "1" ]]; then
  log_gate_args+=(--require-stage2-npugraph)
fi
python competition/minicpmo_b/scripts/validate_candidate_log.py "${log_gate_args[@]}"
