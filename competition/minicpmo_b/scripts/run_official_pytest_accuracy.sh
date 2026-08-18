#!/usr/bin/env bash
set -Eeuo pipefail

# Run the organizer-provided MiniCPM-o 4.5 full-model accuracy pytest suite.
# Each benchmark is isolated into its own result directory, so a long
# Video-MME/Seed-TTS interruption can be resumed without re-running a passed
# Daily-Omni job.  The default maps logical NPU 0 inside pytest to physical NPU
# 5, matching the current development-machine allocation policy.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/official_pytest_common.sh"

TEST_FILE="tests/e2e/accuracy/minicpmo_4_5/test_minicpmo_4_5.py"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${REPO_ROOT}/vllm_omni/deploy/minicpmo_4_5.yaml}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-5}"
SUITE="all"
RESUME=0
CHECK_ONLY=0
RESULT_DIR=""

usage() {
  cat <<'EOF'
Usage: run_official_pytest_accuracy.sh [options]

Runs the official full-model pytest nodes for Daily-Omni, Video-MME and
Seed-TTS.  It uses ASCEND_RT_VISIBLE_DEVICES so the selected physical device
appears to the test as logical device 0.

Options:
  --suite NAME       all (default), daily-omni, videomme, or seed-tts
  --device ID        physical NPU id (default: 5)
  --result-dir PATH  immutable run directory (default: timestamped under results/)
  --deploy-config P  deployment YAML; default is the official baseline YAML
  --model-path PATH  local MiniCPM-o 4.5 model directory
  --resume           skip only suites that already passed with the same inputs
  --check-only       run dependency/NPU preflight, but do not invoke pytest
  -h, --help         show this help

Environment knobs retained from the official command include SEED_TTS_WER_EVAL,
SEED_TTS_SIM_EVAL, ACC_BENCH_VIDEOMME_NUM_PROMPTS and evaluator model paths.
Set REQUIRE_IDLE_NPU=0 only when deliberately bypassing the safety check.
EOF
}

while (($#)); do
  case "$1" in
    --suite) SUITE="${2:-}"; shift 2 ;;
    --device) NPU_DEVICE="${2:-}"; shift 2 ;;
    --result-dir) RESULT_DIR="${2:-}"; shift 2 ;;
    --deploy-config) DEPLOY_CONFIG="${2:-}"; shift 2 ;;
    --model-path) MODEL_PATH="${2:-}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) official_bench_die "unknown option: $1" ;;
  esac
done

case "${SUITE}" in
  all|daily-omni|videomme|seed-tts) ;;
  *) official_bench_die "--suite must be all, daily-omni, videomme, or seed-tts" ;;
esac
official_bench_require_nonnegative_integer "--device" "${NPU_DEVICE}"
[[ -f "${REPO_ROOT}/${TEST_FILE}" ]] || official_bench_die "missing test file: ${REPO_ROOT}/${TEST_FILE}"
[[ -f "${DEPLOY_CONFIG}" ]] || official_bench_die "missing deploy config: ${DEPLOY_CONFIG}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  official_bench_die "model directory does not exist: ${MODEL_PATH}"
fi

if [[ -z "${RESULT_DIR}" ]]; then
  RESULT_DIR="${REPO_ROOT}/competition/minicpmo_b/results/official_pytest_accuracy_$(official_bench_timestamp)"
fi
if [[ -e "${RESULT_DIR}/summary.tsv" ]] && (( ! RESUME )); then
  official_bench_die "result directory already contains a run: ${RESULT_DIR}; choose a new directory or use --resume"
fi
mkdir -p "${RESULT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}"
export VLLM_TEST_MINICPMO_4_5_MODEL="${MODEL_PATH}"
export VLLM_TEST_MINICPMO_4_5_DEPLOY_CONFIG="${DEPLOY_CONFIG}"
export SEED_TTS_WER_EVAL="${SEED_TTS_WER_EVAL:-1}"
export SEED_TTS_SIM_EVAL="${SEED_TTS_SIM_EVAL:-1}"

official_bench_capture_metadata "${RESULT_DIR}" "${NPU_DEVICE}" "${REPO_ROOT}"
printf '%s\n' "${DEPLOY_CONFIG}" > "${RESULT_DIR}/deploy_config_path.txt"
sha256sum "${DEPLOY_CONFIG}" > "${RESULT_DIR}/deploy_config_sha256.txt"

modules=(pytest pytest_asyncio huggingface_hub)
case "${SUITE}" in
  all) modules+=(datasets av jiwer zhon) ;;
  daily-omni|videomme) modules+=(datasets av) ;;
  seed-tts) modules+=(jiwer zhon) ;;
esac
official_bench_require_python_modules "${modules[@]}"
official_bench_require_idle_npu "${NPU_DEVICE}" "${RESULT_DIR}/npu_smi_idle_check.txt"

if (( CHECK_ONLY )); then
  official_bench_note "Preflight passed; no pytest process was started."
  exit 0
fi

declare -A TEST_NODE=(
  [daily-omni]='test_minicpmo_4_5_daily_omni_accuracy_bench'
  [videomme]='test_minicpmo_4_5_videomme_accuracy_bench'
  [seed-tts]='test_minicpmo_4_5_seed_tts_wer_bench'
)
if [[ "${SUITE}" == "all" ]]; then
  suites=(daily-omni videomme seed-tts)
else
  suites=("${SUITE}")
fi

printf 'suite\tstatus\tresult_dir\tlog\n' > "${RESULT_DIR}/summary.tsv"
overall_status=0
for current_suite in "${suites[@]}"; do
  suite_dir="${RESULT_DIR}/${current_suite}"
  mkdir -p "${suite_dir}"
  fingerprint="$(official_bench_fingerprint "${REPO_ROOT}" "${NPU_DEVICE}" "${MODEL_PATH}" "${DEPLOY_CONFIG}" "${REPO_ROOT}/${TEST_FILE}" "${current_suite}")"
  marker="${suite_dir}/pytest_passed.fingerprint"
  if (( RESUME )) && [[ -f "${marker}" ]] && [[ "$(< "${marker}")" == "${fingerprint}" ]]; then
    official_bench_note "Skipping resumed ${current_suite}; matching successful marker exists."
    printf '%s\t%s\t%s\t%s\n' "${current_suite}" skipped "${suite_dir}" "${suite_dir}/pytest.log" >> "${RESULT_DIR}/summary.tsv"
    continue
  fi

  # A previous pytest process must have relinquished the selected physical NPU
  # before the next independent suite starts.
  official_bench_require_idle_npu "${NPU_DEVICE}" "${suite_dir}/npu_smi_idle_check.txt"
  cmd=(pytest -s -v "${TEST_FILE}::${TEST_NODE[${current_suite}]}" -m full_model --run-level full_model)
  export ACC_BENCH_RESULT_DIR="${suite_dir}"
  if official_bench_run_logged "${suite_dir}/pytest.log" "${cmd[@]}"; then
    printf '%s\n' "${fingerprint}" > "${marker}"
    printf '%s\t%s\t%s\t%s\n' "${current_suite}" passed "${suite_dir}" "${suite_dir}/pytest.log" >> "${RESULT_DIR}/summary.tsv"
  else
    status=$?
    printf '%s\n' "${status}" > "${suite_dir}/pytest_exit_code.txt"
    printf '%s\t%s\t%s\t%s\n' "${current_suite}" "failed(${status})" "${suite_dir}" "${suite_dir}/pytest.log" >> "${RESULT_DIR}/summary.tsv"
    overall_status=1
  fi
done

official_bench_capture_npu_after "${RESULT_DIR}"
if (( overall_status )); then
  official_bench_note "One or more accuracy suites failed. Resume the same directory with --resume after fixing the cause."
  exit 1
fi
official_bench_note "Accuracy suites passed. Summary: ${RESULT_DIR}/summary.tsv"
