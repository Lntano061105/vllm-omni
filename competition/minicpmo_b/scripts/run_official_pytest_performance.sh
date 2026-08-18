#!/usr/bin/env bash
set -Eeuo pipefail

# Run the two organizer-provided performance pytest configurations: the
# chat-completions Seed-TTS matrix (TTFT/TTFP/request-level RTF) and the native
# Realtime duplex Seed-TTS matrix (the closest official SPEAK->WAV RTF probe).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/official_pytest_common.sh"

TEST_FILE="tests/dfx/perf/scripts/run_benchmark.py"
SIMPLEX_CONFIG="${SIMPLEX_CONFIG:-tests/dfx/perf/tests/test_minicpmo_4_5.json}"
DUPLEX_CONFIG="${DUPLEX_CONFIG:-tests/dfx/perf/tests/test_minicpmo_4_5_duplex_seed_tts.json}"
NPU_DEVICE="${NPU_DEVICE:-5}"
SUITE="all"
RESUME=0
CHECK_ONLY=0
RESULT_DIR=""

usage() {
  cat <<'EOF'
Usage: run_official_pytest_performance.sh [options]

Runs the official performance pytest commands, keeping raw JSON produced via
BENCHMARK_DIR and pytest logs in separate simplex/ and duplex/ directories.

Options:
  --suite NAME       all (default), simplex, or duplex
  --device ID        physical NPU id (default: 5)
  --result-dir PATH  immutable run directory (default: timestamped under results/)
  --simplex-config P official/simplex benchmark JSON to use
  --duplex-config P  official/duplex benchmark JSON to use
  --resume           skip only suites that already passed with the same inputs
  --check-only       run dependency/NPU preflight, but do not invoke pytest
  -h, --help         show this help

The commands are equivalent to the organizer examples:
  BENCHMARK_DIR=... pytest -s -v tests/dfx/perf/scripts/run_benchmark.py \
      --test-config-file tests/dfx/perf/tests/test_minicpmo_4_5.json
and its duplex Seed-TTS configuration counterpart.
EOF
}

while (($#)); do
  case "$1" in
    --suite) SUITE="${2:-}"; shift 2 ;;
    --device) NPU_DEVICE="${2:-}"; shift 2 ;;
    --result-dir) RESULT_DIR="${2:-}"; shift 2 ;;
    --simplex-config) SIMPLEX_CONFIG="${2:-}"; shift 2 ;;
    --duplex-config) DUPLEX_CONFIG="${2:-}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) official_bench_die "unknown option: $1" ;;
  esac
done

case "${SUITE}" in
  all|simplex|duplex) ;;
  *) official_bench_die "--suite must be all, simplex, or duplex" ;;
esac
official_bench_require_nonnegative_integer "--device" "${NPU_DEVICE}"
[[ -f "${REPO_ROOT}/${TEST_FILE}" ]] || official_bench_die "missing test file: ${REPO_ROOT}/${TEST_FILE}"
[[ -f "${REPO_ROOT}/${SIMPLEX_CONFIG}" ]] || official_bench_die "missing simplex config: ${REPO_ROOT}/${SIMPLEX_CONFIG}"
[[ -f "${REPO_ROOT}/${DUPLEX_CONFIG}" ]] || official_bench_die "missing duplex config: ${REPO_ROOT}/${DUPLEX_CONFIG}"

if [[ -z "${RESULT_DIR}" ]]; then
  RESULT_DIR="${REPO_ROOT}/competition/minicpmo_b/results/official_pytest_performance_$(official_bench_timestamp)"
fi
if [[ -e "${RESULT_DIR}/summary.tsv" ]] && (( ! RESUME )); then
  official_bench_die "result directory already contains a run: ${RESULT_DIR}; choose a new directory or use --resume"
fi
mkdir -p "${RESULT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}"

official_bench_capture_metadata "${RESULT_DIR}" "${NPU_DEVICE}" "${REPO_ROOT}"
sha256sum "${REPO_ROOT}/${SIMPLEX_CONFIG}" > "${RESULT_DIR}/simplex_config_sha256.txt"
sha256sum "${REPO_ROOT}/${DUPLEX_CONFIG}" > "${RESULT_DIR}/duplex_config_sha256.txt"
official_bench_require_python_modules pytest pytest_asyncio
official_bench_require_idle_npu "${NPU_DEVICE}" "${RESULT_DIR}/npu_smi_idle_check.txt"

if (( CHECK_ONLY )); then
  official_bench_note "Preflight passed; no pytest process was started."
  exit 0
fi

if [[ "${SUITE}" == "all" ]]; then
  suites=(simplex duplex)
else
  suites=("${SUITE}")
fi

printf 'suite\tstatus\tresult_dir\tlog\n' > "${RESULT_DIR}/summary.tsv"
overall_status=0
for current_suite in "${suites[@]}"; do
  if [[ "${current_suite}" == "simplex" ]]; then
    config="${SIMPLEX_CONFIG}"
  else
    config="${DUPLEX_CONFIG}"
  fi
  suite_dir="${RESULT_DIR}/${current_suite}"
  mkdir -p "${suite_dir}"
  fingerprint="$(official_bench_fingerprint "${REPO_ROOT}" "${NPU_DEVICE}" "${REPO_ROOT}/${TEST_FILE}" "${REPO_ROOT}/${config}" "${current_suite}")"
  marker="${suite_dir}/pytest_passed.fingerprint"
  if (( RESUME )) && [[ -f "${marker}" ]] && [[ "$(< "${marker}")" == "${fingerprint}" ]]; then
    official_bench_note "Skipping resumed ${current_suite}; matching successful marker exists."
    printf '%s\t%s\t%s\t%s\n' "${current_suite}" skipped "${suite_dir}" "${suite_dir}/pytest.log" >> "${RESULT_DIR}/summary.tsv"
    continue
  fi

  official_bench_require_idle_npu "${NPU_DEVICE}" "${suite_dir}/npu_smi_idle_check.txt"
  cmd=(pytest -s -v "${TEST_FILE}" --test-config-file "${config}")
  export BENCHMARK_DIR="${suite_dir}"
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
  official_bench_note "One or more performance suites failed. Resume the same directory with --resume after fixing the cause."
  exit 1
fi
official_bench_note "Performance suites passed. Summary: ${RESULT_DIR}/summary.tsv"
