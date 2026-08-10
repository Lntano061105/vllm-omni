#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
SUITE="${SUITE:-daily-omni}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/accuracy/${SUITE}}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
MIN_DAILY_OMNI_ACCURACY="${MIN_DAILY_OMNI_ACCURACY:-0.78}"
MIN_VIDEOMME_ACCURACY="${MIN_VIDEOMME_ACCURACY:-0.68}"
MAX_SEED_TTS_MEAN_WER="${MAX_SEED_TTS_MEAN_WER:-0.05}"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/tmp/minicpmo_b_daily_omni}"
VIDEOMME_ROOT="${VIDEOMME_ROOT:-/tmp/minicpmo_b_videomme}"
SEED_TTS_ROOT="${SEED_TTS_ROOT:-/tmp/minicpmo_b_seedtts}"
HF_HOME="${HF_HOME:-/tmp/minicpmo_b_hf}"
LOCAL_WHISPER_MODEL="${LOCAL_WHISPER_MODEL:-/tmp/minicpmo_b_models/whisper-large-v3-modelscope}"

mkdir -p "${RESULT_DIR}" "${HF_HOME}"
cd "${REPO_ROOT}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

common=(
  python tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --num-warmups "${NUM_WARMUPS}"
  --result-dir "${RESULT_DIR}"
  --trust-remote-code
)

case "${SUITE}" in
  daily-omni)
    NUM_PROMPTS="${NUM_PROMPTS:-1197}"
    exec "${common[@]}" \
      --num-prompts "${NUM_PROMPTS}" \
      --skip-seed-tts \
      --skip-videomme \
      --daily-omni-qa-json "${DAILY_OMNI_ROOT}/qa.json" \
      --daily-omni-video-dir "${DAILY_OMNI_ROOT}/Videos" \
      --daily-omni-input-mode all \
      --daily-omni-pack-mode minicpm-interleave \
      --daily-extra-body-json '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}' \
      --daily-omni-save-eval-items \
      --min-daily-omni-accuracy "${MIN_DAILY_OMNI_ACCURACY}" \
      --temperature 0 \
      --output-len 512
    ;;
  videomme)
    NUM_PROMPTS="${NUM_PROMPTS:-2700}"
    videomme_media_args=()
    if [[ "${VIDEOMME_INLINE_LOCAL_VIDEO:-0}" == "1" ]]; then
      videomme_media_args+=(--videomme-inline-local-video)
    fi
    exec "${common[@]}" \
      --num-prompts "${NUM_PROMPTS}" \
      --skip-daily-omni \
      --skip-seed-tts \
      --run-videomme \
      --videomme-dataset-path "${VIDEOMME_ROOT}" \
      --videomme-parquet "${VIDEOMME_ROOT}/videomme/test-00000-of-00001.parquet" \
      --videomme-video-dir "${VIDEOMME_ROOT}/video" \
      --videomme-pack-mode minicpm-frames \
      --videomme-max-frames 96 \
      --videomme-duration all \
      --videomme-extra-body-json '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}' \
      --videomme-save-eval-items \
      "${videomme_media_args[@]}" \
      --min-videomme-accuracy "${MIN_VIDEOMME_ACCURACY}" \
      --temperature 0 \
      --output-len 128
    ;;
  seed-tts)
    NUM_PROMPTS="${NUM_PROMPTS:-1000}"
    export SEED_TTS_SIM_EVAL=0
    export SEED_TTS_UTMOS_EVAL=0
    # Competition images are commonly network-isolated. Prefer the prepared
    # local Whisper checkpoint so an accuracy gate never blocks on a Hub
    # metadata request. An explicit SEED_TTS_HF_WHISPER_MODEL still wins.
    if [[ -z "${SEED_TTS_HF_WHISPER_MODEL:-}" && -d "${LOCAL_WHISPER_MODEL}" ]]; then
      export SEED_TTS_HF_WHISPER_MODEL="${LOCAL_WHISPER_MODEL}"
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
    fi
    exec "${common[@]}" \
      --num-prompts "${NUM_PROMPTS}" \
      --skip-daily-omni \
      --skip-videomme \
      --seed-tts-dataset-path "${SEED_TTS_ROOT}" \
      --seed-tts-root "${SEED_TTS_ROOT}" \
      --seed-tts-locale en \
      --seed-tts-wer-save-items \
      --seed-tts-eval-device "${SEED_TTS_EVAL_DEVICE:-cpu}" \
      --seed-extra-body-json '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}' \
      --temperature 0 \
      --max-seed-tts-mean-wer "${MAX_SEED_TTS_MEAN_WER}"
    ;;
  *)
    echo "Unknown SUITE=${SUITE}; expected daily-omni, videomme, or seed-tts" >&2
    exit 2
    ;;
esac
