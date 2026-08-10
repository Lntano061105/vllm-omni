#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
SEED_TTS_PATH="${SEED_TTS_PATH:-/workspace/seed-tts}"
SEED_TTS_ARCHIVE="${SEED_TTS_ARCHIVE:-/workspace/seed-tts/seedtts_testset.tar}"
SEED_TTS_EXTRACT_DIR="${SEED_TTS_EXTRACT_DIR:-/tmp/minicpmo_b_seedtts}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/optimized}"
RESULT_FILENAME="${RESULT_FILENAME:-seed_tts_c${MAX_CONCURRENCY}_n${NUM_PROMPTS}.json}"

if [[ ! -f "${SEED_TTS_PATH}/en/meta.lst" ]]; then
  if [[ ! -f "${SEED_TTS_ARCHIVE}" ]]; then
    echo "Seed-TTS meta.lst and archive are both missing" >&2
    exit 1
  fi
  mkdir -p "${SEED_TTS_EXTRACT_DIR}"
  if [[ ! -f "${SEED_TTS_EXTRACT_DIR}/en/meta.lst" ]]; then
    tar --no-same-owner -xf "${SEED_TTS_ARCHIVE}" --strip-components=1 -C "${SEED_TTS_EXTRACT_DIR}"
  fi
  SEED_TTS_PATH="${SEED_TTS_EXTRACT_DIR}"
fi

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

exec vllm bench serve --omni \
  --backend openai-chat-omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --endpoint /v1/chat/completions \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dataset-name seed-tts \
  --dataset-path "${SEED_TTS_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --no-oversample \
  --trust-remote-code \
  --seed-tts-locale en \
  --temperature 0 \
  --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}' \
  --percentile-metrics ttft,e2el,audio_ttfp,audio_rtf,audio_chunk_rtf,audio_duration,audio_underrun \
  --num-warmups "${NUM_WARMUPS}" \
  --save-result \
  --result-dir "${RESULT_DIR}" \
  --result-filename "${RESULT_FILENAME}"
