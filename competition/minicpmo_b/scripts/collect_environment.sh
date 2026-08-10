#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/environment}"
HASH_MODEL_WEIGHTS="${HASH_MODEL_WEIGHTS:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

date -u --iso-8601=seconds > "${OUTPUT_DIR}/timestamp_utc.txt"
uname -a > "${OUTPUT_DIR}/uname.txt"
python --version > "${OUTPUT_DIR}/python_version.txt" 2>&1
python -m pip show \
  torch torch-npu vllm vllm-omni vllm-ascend transformers \
  > "${OUTPUT_DIR}/python_packages.txt" 2>&1 || true
npu-smi info > "${OUTPUT_DIR}/npu_smi.txt" 2>&1 || true

git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
git diff --stat > "${OUTPUT_DIR}/git_diff_stat.txt"
git diff --binary > "${OUTPUT_DIR}/working_tree.patch"

if [[ -d "${MODEL_PATH}" ]]; then
  find "${MODEL_PATH}" -maxdepth 2 -type f -printf '%P\t%s\n' \
    | sort > "${OUTPUT_DIR}/model_manifest.tsv"
  find "${MODEL_PATH}" -maxdepth 2 -type f \
    \( -name 'config.json' -o -name '*index*.json' -o -name '*.yaml' \) \
    -print0 | sort -z | xargs -0 -r sha256sum \
    > "${OUTPUT_DIR}/model_metadata_sha256.txt"
  if [[ "${HASH_MODEL_WEIGHTS}" == "1" ]]; then
    find "${MODEL_PATH}" -maxdepth 2 -type f \
      \( -name '*.safetensors' -o -name '*.pt' -o -name '*.onnx' \) \
      -print0 | sort -z | xargs -0 -r sha256sum \
      > "${OUTPUT_DIR}/model_weights_sha256.txt"
  fi
else
  printf 'Model path not visible: %s\n' "${MODEL_PATH}" > "${OUTPUT_DIR}/model_manifest.tsv"
fi

echo "Environment manifest written to ${OUTPUT_DIR}"
