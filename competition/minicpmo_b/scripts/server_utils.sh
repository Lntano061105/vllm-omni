#!/usr/bin/env bash

wait_for_served_model() {
  local host="$1"
  local port="$2"
  local served_model_name="$3"
  local server_pid="$4"
  local server_log="$5"
  local timeout_s="$6"
  local deadline=$((SECONDS + timeout_s))
  local models_json

  while true; do
    if models_json="$(curl --fail --silent "http://${host}:${port}/v1/models" 2>/dev/null)"; then
      if EXPECTED_MODEL="${served_model_name}" python -c '
import json
import os
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
models = payload.get("data", []) if isinstance(payload, dict) else []
raise SystemExit(0 if any(isinstance(row, dict) and row.get("id") == os.environ["EXPECTED_MODEL"] for row in models) else 1)
' <<< "${models_json}" 2>/dev/null; then
        return 0
      fi
    fi

    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Server exited before serving ${served_model_name}; see ${server_log}" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Server readiness timed out after ${timeout_s}s; see ${server_log}" >&2
      return 1
    fi
    sleep 2
  done
}

validate_performance_result() {
  local result_json="$1"
  local expected_requests="$2"
  python -c '
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
with open(path, encoding="utf-8") as stream:
    result = json.load(stream)
completed = int(result.get("completed", 0) or 0)
failed = int(result.get("failed", 0) or 0)
if completed != expected or failed != 0:
    raise SystemExit(
        f"Invalid performance result {path}: completed={completed}, "
        f"failed={failed}, expected={expected}"
    )
' "${result_json}" "${expected_requests}"
}

validate_stage1_full_decode_graph() {
  local server_log="$1"
  python -c '
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
stage1 = "\n".join(
    line for line in text.splitlines()
    if "StageEngineCoreProc_stage1" in line
)
required = {
    "Stage 1 requested FULL_DECODE_ONLY": "CUDAGraphMode.FULL_DECODE_ONLY",
    "Npugraph_ex enabled": "enable_npugraph_ex\x27: True",
}
missing = [label for label, marker in required.items() if marker not in stage1]
if "Capturing CUDA graphs (decode, FULL)" not in text:
    missing.append("FULL decode graph captured")
if missing:
    raise SystemExit(
        f"Stage 1 full-decode graph validation failed for {path}: "
        + ", ".join(missing)
    )
if "CUDAGraphMode.PIECEWISE" in stage1:
    raise SystemExit(
        f"Stage 1 unexpectedly contains PIECEWISE graph mode in {path}"
    )
print(f"Validated Stage 1 FULL_DECODE_ONLY graph activation: {path}")
' "${server_log}"
}
