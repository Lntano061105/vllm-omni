#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
CASE_NAME="${CASE_NAME:-optimized_duplex_matrix}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
REQUIRE_STAGE1_FULL_DECODE="${REQUIRE_STAGE1_FULL_DECODE:-1}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/${CASE_NAME}/duplex_rtf}"
SERVER_LOG="${SERVER_LOG:-${RESULT_DIR}/server.log}"

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"
git rev-parse HEAD > "${RESULT_DIR}/git_commit.txt"
git status --short > "${RESULT_DIR}/git_status.txt"
sha256sum "${DEPLOY_CONFIG}" > "${RESULT_DIR}/deploy_config_sha256.txt"
npu-smi info > "${RESULT_DIR}/npu_smi.txt" 2>&1 || true
python competition/minicpmo_b/scripts/run_protocol.py write \
  --output "${RESULT_DIR}/run_protocol.json" \
  --kind realtime-duplex-speak-generation-matrix \
  --field served_model_name="${SERVED_MODEL_NAME}" \
  --field transcript="${TRANSCRIPT:-The two men hurried back and found the cylinder still lying in the same position.}" \
  --field turns_per_session=1 \
  --field input_chunk_ms=200 \
  --field turn_duration_ms=0 \
  --field num_warmups="${NUM_WARMUPS:-2}" \
  --field c1_prompts="${C1_PROMPTS:-32}" \
  --field c4_prompts="${C4_PROMPTS:-64}" \
  --field c8_prompts="${C8_PROMPTS:-128}" \
  --file input_wav="${INPUT_WAV:-/tmp/minicpmo_b_seedtts/en/wavs/common_voice_en_125386-common_voice_en_125388.wav}" \
  --file ref_audio="${REF_AUDIO:-/workspace/MiniCPM-o-4_5/assets/system_ref_audio.wav}" \
  --variant-field label="${CASE_NAME}" \
  --variant-file deploy_config="${DEPLOY_CONFIG}" \
  > "${RESULT_DIR}/run_protocol.log"

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
    # SIGINT is the programmatic equivalent of one terminal Ctrl-C and lets
    # the API server/orchestrator/stages run their normal cleanup path.
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
  local filename="native_duplex_c${concurrency}_n${prompts}.json"
  HOST="${HOST}" PORT="${PORT}" \
  MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  NUM_PROMPTS="${prompts}" MAX_CONCURRENCY="${concurrency}" \
  NUM_WARMUPS="${NUM_WARMUPS:-2}" \
  RESULT_DIR="${RESULT_DIR}" RESULT_FILENAME="${filename}" \
  competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh
  python - "${RESULT_DIR}/${filename}" "${prompts}" <<'PY'
import json
import sys

path, expected_raw = sys.argv[1:]
expected = int(expected_raw)
data = json.load(open(path, encoding="utf-8"))
sessions = int(data.get("sessions", data.get("completed", 0)) or 0)
errors = int(data.get("errors", data.get("failed", 0)) or 0)
chunks = int(data.get("audio_speak_generation_chunk_count", 0) or 0)
if sessions != expected or errors != 0 or chunks <= 0:
    raise SystemExit(
        f"invalid duplex result {path}: sessions={sessions}, errors={errors}, "
        f"speak_generation_chunks={chunks}, expected_sessions={expected}"
    )
PY
}

run_case 1 "${C1_PROMPTS:-32}"
run_case 4 "${C4_PROMPTS:-64}"
run_case 8 "${C8_PROMPTS:-128}"

if [[ "${RUN_MULTITURN_GATE:-1}" == "1" ]]; then
  multiturn_dir="${RESULT_DIR}/multiturn_s2_t3"
  multiturn_result="${multiturn_dir}/native_duplex_rtf.json"
  HOST="${HOST}" PORT="${PORT}" \
  NUM_PROMPTS=2 MAX_CONCURRENCY=1 TURNS_PER_SESSION=3 NUM_WARMUPS=0 \
  RESULT_DIR="${multiturn_dir}" \
  RESULT_FILENAME=native_duplex_rtf.json \
  competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh

  python competition/minicpmo_b/scripts/gate_multiturn_duplex.py \
    "${multiturn_result}" \
    --server-log "${SERVER_LOG}" \
    --output "${multiturn_dir}/multiturn_gate.json"
fi

if [[ "${REQUIRE_ACTIVATION_GATE:-0}" == "1" ]]; then
  python competition/minicpmo_b/scripts/validate_candidate_log.py \
    "${SERVER_LOG}" \
    --require-stage0-mm-cache-disabled \
    --require-stage1-full-decode \
    --require-stage0-ref-cache \
    --require-stage2-prompt-cache \
    --require-stage2-runner-prewarm \
    --require-stage1-cpu-slot-mapping \
    --require-stage1-graph-sampler \
    --require-stage1-binary-argmax \
    --output "${RESULT_DIR}/activation_gate.json"
fi
