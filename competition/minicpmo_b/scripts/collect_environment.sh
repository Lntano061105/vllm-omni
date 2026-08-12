#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/competition/minicpmo_b/results/environment}"
HASH_MODEL_WEIGHTS="${HASH_MODEL_WEIGHTS:-0}"
DAILY_OMNI_SOURCE="${DAILY_OMNI_SOURCE:-/workspace/MTEB/Daily-Omni}"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/tmp/minicpmo_b_daily_omni}"
VIDEOMME_SOURCE="${VIDEOMME_SOURCE:-/workspace/Video-MME}"
VIDEOMME_ROOT="${VIDEOMME_ROOT:-/tmp/minicpmo_b_videomme}"
SEED_TTS_ROOT="${SEED_TTS_ROOT:-/tmp/minicpmo_b_seedtts}"
SEED_TTS_WHISPER_MODEL="${SEED_TTS_WHISPER_MODEL:-/workspace/whisper-large-v3}"
SEED_TTS_WAVLM_MODEL="${SEED_TTS_WAVLM_MODEL:-/workspace/wavlm-base-plus}"
SEED_TTS_UTMOS_FILE="${SEED_TTS_UTMOS_FILE:-${SEED_TTS_UTMOS_JIT_FILE:-/workspace/utmos/utmos.jit}}"
CONTAINER_IMAGE_DIGEST="${CONTAINER_IMAGE_DIGEST:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

date -u --iso-8601=seconds > "${OUTPUT_DIR}/timestamp_utc.txt"
uname -a > "${OUTPUT_DIR}/uname.txt"
python --version > "${OUTPUT_DIR}/python_version.txt" 2>&1
python -m pip show \
  torch torch-npu vllm vllm-omni vllm-ascend transformers \
  > "${OUTPUT_DIR}/python_packages.txt" 2>&1 || true
npu-smi info > "${OUTPUT_DIR}/npu_smi.txt" 2>&1 || true
printf '%s\n' "${CONTAINER_IMAGE_DIGEST}" > "${OUTPUT_DIR}/container_image_digest.txt"
if [[ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ]]; then
  cp /usr/local/Ascend/ascend-toolkit/latest/version.cfg "${OUTPUT_DIR}/cann_version.txt"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  sha256sum /usr/local/Ascend/ascend-toolkit/set_env.sh > "${OUTPUT_DIR}/cann_version.txt"
else
  printf 'CANN version file not found\n' > "${OUTPUT_DIR}/cann_version.txt"
fi

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

{
  for eval_model in \
    "${SEED_TTS_WHISPER_MODEL}" \
    "${SEED_TTS_WAVLM_MODEL}" \
    "${SEED_TTS_UTMOS_FILE}"; do
    if [[ -d "${eval_model}" ]]; then
      find "${eval_model}" -maxdepth 1 -type f -printf '%p\t%s\n'
    elif [[ -f "${eval_model}" ]]; then
      find "${eval_model}" -maxdepth 0 -type f -printf '%p\t%s\n'
    else
      printf '%s\tMISSING\n' "${eval_model}"
    fi
  done
} | sort > "${OUTPUT_DIR}/seed_tts_eval_model_manifest.tsv"

if [[ "${HASH_MODEL_WEIGHTS}" == "1" ]]; then
  {
    for eval_model in \
      "${SEED_TTS_WHISPER_MODEL}" \
      "${SEED_TTS_WAVLM_MODEL}" \
      "${SEED_TTS_UTMOS_FILE}"; do
      if [[ -d "${eval_model}" ]]; then
        find "${eval_model}" -maxdepth 1 -type f \
          \( -name '*.safetensors' -o -name '*.bin' -o -name '*.jit' -o -name 'config.json' \) \
          -print0 | sort -z | xargs -0 -r sha256sum
      elif [[ -f "${eval_model}" ]]; then
        sha256sum "${eval_model}"
      fi
    done
  } > "${OUTPUT_DIR}/seed_tts_eval_model_sha256.txt"
fi

# Record the exact local dataset snapshot without hashing tens of gigabytes of
# media. Small authoritative indexes/QA files are hashed; all source archives,
# parquet shards and prepared media are still listed with byte sizes.
{
  for root in \
    "${DAILY_OMNI_SOURCE}" "${DAILY_OMNI_ROOT}" \
    "${VIDEOMME_SOURCE}" "${VIDEOMME_ROOT}" "${SEED_TTS_ROOT}"; do
    if [[ -d "${root}" ]]; then
      find "${root}" -maxdepth 2 -type f -printf '%p\t%s\n'
    else
      printf '%s\tMISSING\n' "${root}"
    fi
  done
} | sort > "${OUTPUT_DIR}/dataset_manifest.tsv"

{
  for metadata in \
    "${DAILY_OMNI_SOURCE}/dataset_infos.json" \
    "${DAILY_OMNI_ROOT}/qa.json" \
    "${VIDEOMME_SOURCE}/videomme/test-00000-of-00001.parquet" \
    "${VIDEOMME_ROOT}/videomme/test-00000-of-00001.parquet" \
    "${SEED_TTS_ROOT}/en/meta.lst"; do
    if [[ -f "${metadata}" ]]; then
      sha256sum "${metadata}"
    fi
  done
} > "${OUTPUT_DIR}/dataset_metadata_sha256.txt"

echo "Environment manifest written to ${OUTPUT_DIR}"
