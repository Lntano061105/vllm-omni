#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml}"
NPU_DEVICE="${NPU_DEVICE:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
STAGE_INIT_TIMEOUT="${STAGE_INIT_TIMEOUT:-900}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}"

extra_args=()
if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH}" ]]; then
  extra_args+=(--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}")
fi

cd "${REPO_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec vllm serve "${MODEL_PATH}" --omni \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --deploy-config "${DEPLOY_CONFIG}" \
  --stage-init-timeout "${STAGE_INIT_TIMEOUT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${extra_args[@]}"
