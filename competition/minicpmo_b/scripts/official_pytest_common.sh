#!/usr/bin/env bash

# Shared helpers for the lightweight wrappers around the organizer-provided
# pytest commands.  The helpers deliberately never kill an existing service:
# a benchmark must start from an idle card to make its result reproducible.

official_bench_note() {
  printf '[official-bench] %s\n' "$*"
}

official_bench_die() {
  printf '[official-bench] ERROR: %s\n' "$*" >&2
  exit 2
}

official_bench_timestamp() {
  date -u +%Y%m%dT%H%M%SZ
}

official_bench_require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || official_bench_die "${name} must be a non-negative integer, got: ${value}"
}

official_bench_require_python_modules() {
  # Use import discovery rather than importing the packages: some optional ML
  # packages perform expensive initialization at import time.
  python -c '
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python package(s): " + ", ".join(missing))
' "$@"
}

official_bench_capture_metadata() {
  local output_dir="$1"
  local physical_device="$2"
  local repo_root="$3"

  mkdir -p "${output_dir}"
  {
    printf 'started_utc=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'physical_npu=%s\n' "${physical_device}"
    printf 'ascend_rt_visible_devices=%s\n' "${ASCEND_RT_VISIBLE_DEVICES:-}"
    printf 'vllm_worker_multiproc_method=%s\n' "${VLLM_WORKER_MULTIPROC_METHOD:-}"
    printf 'python=%s\n' "$(python --version 2>&1)"
    # ``pytest --version`` eagerly loads every installed third-party plugin and
    # can hang in a partially configured accelerator image. Importing pytest is
    # sufficient for provenance and has no plugin-discovery side effect.
    printf 'pytest=%s\n' "$(python -c 'import pytest; print(pytest.__version__)' 2>&1)"
  } > "${output_dir}/run.env"

  (
    cd "${repo_root}"
    git rev-parse HEAD
  ) > "${output_dir}/git_commit.txt"
  (
    cd "${repo_root}"
    git status --short
  ) > "${output_dir}/git_status.txt"

  if command -v npu-smi >/dev/null 2>&1; then
    timeout "${NPU_SMI_TIMEOUT_S:-20}" npu-smi info > "${output_dir}/npu_smi_before.txt" 2>&1 || true
  else
    printf 'npu-smi is unavailable in PATH\n' > "${output_dir}/npu_smi_before.txt"
  fi
}

official_bench_capture_npu_after() {
  local output_dir="$1"
  if command -v npu-smi >/dev/null 2>&1; then
    timeout "${NPU_SMI_TIMEOUT_S:-20}" npu-smi info > "${output_dir}/npu_smi_after.txt" 2>&1 || true
  fi
}

official_bench_require_idle_npu() {
  local physical_device="$1"
  local output_file="$2"

  if [[ "${REQUIRE_IDLE_NPU:-1}" != "1" ]]; then
    official_bench_note "Skipping NPU-idle check because REQUIRE_IDLE_NPU=${REQUIRE_IDLE_NPU}."
    return 0
  fi
  command -v npu-smi >/dev/null 2>&1 || {
    official_bench_die "npu-smi is required for the default idle-card safety check; set REQUIRE_IDLE_NPU=0 only for a deliberate override."
  }

  # Some 910B/910C driver versions reject ``info proc -i`` even though their
  # normal ``info`` view includes a per-NPU process table.  Parse that portable
  # table instead of relying on the unsupported per-card proc subcommand.
  if ! timeout "${NPU_SMI_TIMEOUT_S:-20}" npu-smi info > "${output_file}" 2>&1; then
    official_bench_die "could not query physical NPU ${physical_device}; see ${output_file}"
  fi
  if grep -qiE "No running processes found in NPU[[:space:]]+${physical_device}([^0-9]|$)" "${output_file}"; then
    return 0
  fi
  if grep -qE "^[[:space:]]*\\|[[:space:]]*${physical_device}[[:space:]]+" "${output_file}"; then
    official_bench_die "physical NPU ${physical_device} is not idle; see ${output_file}. Refusing to mix this benchmark with an existing process."
  fi

  official_bench_die "could not determine whether physical NPU ${physical_device} is idle; see ${output_file}. Refusing to start an unverified benchmark."
}

official_bench_write_command() {
  local output_file="$1"
  shift
  printf '%q ' "$@" > "${output_file}"
  printf '\n' >> "${output_file}"
}

official_bench_fingerprint() {
  # Inputs are already shell-safe paths/values controlled by the wrapper.  A
  # fingerprint lets --resume skip only a previously successful, identical
  # suite; changed source/configuration is run again instead of being masked.
  local repo_root="$1"
  shift
  {
    (
      cd "${repo_root}"
      git rev-parse HEAD
    )
    printf '%s\n' "$@"
    local item
    for item in "$@"; do
      if [[ -f "${item}" ]]; then
        sha256sum "${item}"
      fi
    done
  } | sha256sum | awk '{print $1}'
}

official_bench_run_logged() {
  local log_file="$1"
  shift
  official_bench_write_command "${log_file}.command" "$@"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local pipe_status=("${PIPESTATUS[@]}")
  set -e
  if (( pipe_status[0] != 0 )); then
    return "${pipe_status[0]}"
  fi
  return "${pipe_status[1]}"
}
