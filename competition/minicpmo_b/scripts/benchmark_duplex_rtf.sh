#!/usr/bin/env bash
set -euo pipefail

# Challenge RTF probe for the native Realtime full-duplex path.  It streams
# user audio with model-owned LISTEN/SPEAK enabled and uses the explicit
# Thinker ``<turn_eos>`` boundary propagated as ``metadata.speak_tail``.

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
INPUT_WAV="${INPUT_WAV:-/tmp/minicpmo_b_seedtts/en/wavs/common_voice_en_125386-common_voice_en_125388.wav}"
REF_AUDIO="${REF_AUDIO:-/workspace/MiniCPM-o-4_5/assets/system_ref_audio.wav}"
TRANSCRIPT="${TRANSCRIPT:-The two men hurried back and found the cylinder still lying in the same position.}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TURNS_PER_SESSION="${TURNS_PER_SESSION:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/duplex_rtf}"
RESULT_FILENAME="${RESULT_FILENAME:-native_duplex_c${MAX_CONCURRENCY}_n${NUM_PROMPTS}.json}"

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

exec python competition/minicpmo_b/scripts/benchmark_native_duplex_rtf.py \
  --url "ws://${HOST}:${PORT}/v1/realtime?duplex=1" \
  --model "${SERVED_MODEL_NAME}" \
  --input-wav "${INPUT_WAV}" \
  --ref-audio "${REF_AUDIO}" \
  --transcript "${TRANSCRIPT}" \
  --num-sessions "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --turns-per-session "${TURNS_PER_SESSION}" \
  --num-warmups "${NUM_WARMUPS}" \
  --output-dir "${RESULT_DIR}" \
  --result-filename "${RESULT_FILENAME}"
