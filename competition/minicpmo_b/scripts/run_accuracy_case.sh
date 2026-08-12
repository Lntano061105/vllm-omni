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
BENCHMARK_SEED="${BENCHMARK_SEED:-0}"

if [[ "${SUITE}" == "daily-omni" && -z "${NUM_PROMPTS:-}" ]]; then
  NUM_PROMPTS="$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${DAILY_OMNI_ROOT}/qa.json")"
fi

case "${SUITE}" in
  daily-omni) ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${DAILY_OMNI_ROOT}}" ;;
  videomme) ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${VIDEOMME_ROOT}}" ;;
  seed-tts) ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}" ;;
  *) echo "Unknown SUITE=${SUITE}" >&2; exit 2 ;;
esac

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"
git rev-parse HEAD > "${RESULT_DIR}/git_commit.txt"
git status --short > "${RESULT_DIR}/git_status.txt"
sha256sum "${DEPLOY_CONFIG}" > "${RESULT_DIR}/deploy_config_sha256.txt"
npu-smi info > "${RESULT_DIR}/npu_smi.txt" 2>&1 || true

protocol_args=(
  write --output "${RESULT_DIR}/run_protocol.json"
  --kind "accuracy-${SUITE}"
  --field served_model_name="${SERVED_MODEL_NAME}"
  --field suite="${SUITE}"
  --field max_concurrency="${MAX_CONCURRENCY:-1}"
  --field num_warmups="${NUM_WARMUPS:-0}"
  --field benchmark_seed="${BENCHMARK_SEED}"
  --field no_oversample=true
  --field request_rate=inf
  --field temperature=0
  --variant-field label="${CASE_NAME}"
  --variant-file deploy_config="${DEPLOY_CONFIG}"
)
case "${SUITE}" in
  daily-omni)
    protocol_args+=(
      --field num_prompts="${NUM_PROMPTS}" --field disable_shuffle=false
      --field input_mode=all --field pack_mode=minicpm-interleave
      --field output_len=512
      --field extra_body='{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'
      --file qa_json="${DAILY_OMNI_ROOT}/qa.json"
      --tree media="${DAILY_OMNI_ROOT}/Videos"
    )
    ;;
  videomme)
    protocol_args+=(
      --field num_prompts="${NUM_PROMPTS:-2700}"
      --field disable_shuffle=true --field pack_mode=minicpm-frames
      --field max_frames=96 --field duration=all --field output_len=128
      --field inline_local_video="${VIDEOMME_INLINE_LOCAL_VIDEO:-0}"
      --field extra_body='{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'
      --file parquet="${VIDEOMME_ROOT}/videomme/test-00000-of-00001.parquet"
      --tree media="${VIDEOMME_ROOT}/video"
    )
    ;;
  seed-tts)
    protocol_args+=(
      --field num_prompts="${NUM_PROMPTS:-1000}" --field locale=en
      --field disable_shuffle=false --field output_len=null
      --field turns_per_session=1
      --field eval_device="${SEED_TTS_EVAL_DEVICE:-cpu}"
      --field sim_eval="${SEED_TTS_SIM_EVAL:-0}"
      --field utmos_eval="${SEED_TTS_UTMOS_EVAL:-0}"
      --field extra_body='{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'
      --file meta="${SEED_TTS_ROOT:-/tmp/minicpmo_b_seedtts}/en/meta.lst"
      --tree reference_audio="${SEED_TTS_ROOT:-/tmp/minicpmo_b_seedtts}/en/prompt-wavs"
    )
    if [[ -n "${SEED_TTS_HF_WHISPER_MODEL:-}" ]]; then
      protocol_args+=(--tree whisper_model="${SEED_TTS_HF_WHISPER_MODEL}")
    fi
    if [[ "${SEED_TTS_SIM_EVAL:-0}" == "1" ]]; then
      protocol_args+=(--tree wavlm_model="${SEED_TTS_WAVLM_MODEL:?SEED_TTS_WAVLM_MODEL is required when SIM evaluation is enabled}")
    fi
    if [[ "${SEED_TTS_UTMOS_EVAL:-0}" == "1" ]]; then
      protocol_args+=(--file utmos_model="${SEED_TTS_UTMOS_JIT_FILE:?SEED_TTS_UTMOS_JIT_FILE is required when UTMOS evaluation is enabled}")
    fi
    ;;
esac
python competition/minicpmo_b/scripts/run_protocol.py "${protocol_args[@]}" \
  > "${RESULT_DIR}/run_protocol.log"

case "${SUITE}" in
  daily-omni)
    sha256sum "${DAILY_OMNI_ROOT}/qa.json" > "${RESULT_DIR}/input_sha256.txt"
    ;;
  videomme)
    sha256sum "${VIDEOMME_ROOT}/videomme/test-00000-of-00001.parquet" \
      > "${RESULT_DIR}/input_sha256.txt"
    ;;
  seed-tts)
    sha256sum "${SEED_TTS_ROOT:-/tmp/minicpmo_b_seedtts}/en/meta.lst" \
      > "${RESULT_DIR}/input_sha256.txt"
    ;;
esac

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

benchmark_status=0
SUITE="${SUITE}" \
HOST="${HOST}" \
PORT="${PORT}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
RESULT_DIR="${RESULT_DIR}" \
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT}" \
VIDEOMME_ROOT="${VIDEOMME_ROOT}" \
NUM_PROMPTS="${NUM_PROMPTS:-}" \
BENCHMARK_SEED="${BENCHMARK_SEED}" \
competition/minicpmo_b/scripts/benchmark_accuracy.sh || benchmark_status=$?

activation_args=()
if [[ "${REQUIRE_STAGE0_MM_CACHE_DISABLED:-0}" == "1" ]]; then
  activation_args+=(--require-stage0-mm-cache-disabled)
fi
activation_status=0
python competition/minicpmo_b/scripts/validate_candidate_log.py \
  "${SERVER_LOG}" \
  "${activation_args[@]}" \
  --output "${RESULT_DIR}/activation_gate.json" || activation_status=$?

if (( benchmark_status != 0 || activation_status != 0 )); then
  echo "Accuracy case failed: benchmark_status=${benchmark_status}, activation_status=${activation_status}" >&2
  exit 1
fi
