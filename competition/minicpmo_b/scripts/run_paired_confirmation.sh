#!/usr/bin/env bash
set -euo pipefail

# Optional order-bias confirmation after the authoritative A1→B1 26-phase run.
# It reverses the fresh-start order (B2→A2) and never overwrites official data.

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${REPO_ROOT}/competition/minicpmo_b/results/official_910c}"
CONFIRM_ROOT="${CONFIRM_ROOT:-${OFFICIAL_ROOT}/paired_confirmation}"
MODEL_PATH="${MODEL_PATH:-/workspace/MiniCPM-o-4_5}"
NPU_DEVICE="${NPU_DEVICE:-0}"
PORT="${PORT:-8091}"
OPTIMIZED_CONFIG="${OPTIMIZED_CONFIG:-${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-${REPO_ROOT}/vllm_omni/deploy/minicpmo_4_5.yaml}"

A1="${OFFICIAL_ROOT}/official_910c_baseline/duplex_rtf/native_duplex_c1_n32.json"
B1="${OFFICIAL_ROOT}/official_910c_optimized/duplex_rtf/native_duplex_c1_n32.json"
for required in "${A1}" "${B1}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing authoritative first-pair result: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${CONFIRM_ROOT}"
cd "${REPO_ROOT}"

# B2 first, then A2: the reverse of the authoritative A1→B1 order.
CASE_NAME=official_910c_optimized_confirmation \
RESULT_DIR="${CONFIRM_ROOT}/optimized_b2/duplex_rtf" \
DEPLOY_CONFIG="${OPTIMIZED_CONFIG}" \
MODEL_PATH="${MODEL_PATH}" NPU_DEVICE="${NPU_DEVICE}" PORT="${PORT}" \
REQUIRE_STAGE1_FULL_DECODE=1 REQUIRE_ACTIVATION_GATE=1 \
RUN_MULTITURN_GATE=0 \
competition/minicpmo_b/scripts/run_duplex_matrix.sh

CASE_NAME=official_910c_baseline_confirmation \
RESULT_DIR="${CONFIRM_ROOT}/baseline_a2/duplex_rtf" \
DEPLOY_CONFIG="${BASELINE_CONFIG}" \
MODEL_PATH="${MODEL_PATH}" NPU_DEVICE="${NPU_DEVICE}" PORT="${PORT}" \
REQUIRE_STAGE1_FULL_DECODE=0 REQUIRE_ACTIVATION_GATE=0 \
RUN_MULTITURN_GATE=0 \
competition/minicpmo_b/scripts/run_duplex_matrix.sh

python competition/minicpmo_b/scripts/run_protocol.py compare \
  "${OFFICIAL_ROOT}/official_910c_baseline/duplex_rtf/run_protocol.json" \
  "${CONFIRM_ROOT}/baseline_a2/duplex_rtf/run_protocol.json" \
  --require-field num_warmups=2 \
  --require-field turns_per_session=1 \
  --require-field input_chunk_ms=200 \
  --require-field turn_duration_ms=0 \
  --require-field c1_prompts=32 \
  --require-field c4_prompts=64 \
  --require-field c8_prompts=128 \
  --output "${CONFIRM_ROOT}/protocol_gate_baseline_repeat.json"

python competition/minicpmo_b/scripts/run_protocol.py compare \
  "${OFFICIAL_ROOT}/official_910c_optimized/duplex_rtf/run_protocol.json" \
  "${CONFIRM_ROOT}/optimized_b2/duplex_rtf/run_protocol.json" \
  --require-field num_warmups=2 \
  --require-field turns_per_session=1 \
  --require-field input_chunk_ms=200 \
  --require-field turn_duration_ms=0 \
  --require-field c1_prompts=32 \
  --require-field c4_prompts=64 \
  --require-field c8_prompts=128 \
  --output "${CONFIRM_ROOT}/protocol_gate_optimized_repeat.json"

python competition/minicpmo_b/scripts/run_protocol.py compare \
  "${CONFIRM_ROOT}/baseline_a2/duplex_rtf/run_protocol.json" \
  "${CONFIRM_ROOT}/optimized_b2/duplex_rtf/run_protocol.json" \
  --require-field num_warmups=2 \
  --require-field turns_per_session=1 \
  --require-field input_chunk_ms=200 \
  --require-field turn_duration_ms=0 \
  --require-field c1_prompts=32 \
  --require-field c4_prompts=64 \
  --require-field c8_prompts=128 \
  --output "${CONFIRM_ROOT}/protocol_gate_second_pair.json"

python competition/minicpmo_b/scripts/analyze_paired_confirmation.py \
  "${A1}" "${B1}" \
  "${CONFIRM_ROOT}/optimized_b2/duplex_rtf/native_duplex_c1_n32.json" \
  "${CONFIRM_ROOT}/baseline_a2/duplex_rtf/native_duplex_c1_n32.json" \
  --output "${CONFIRM_ROOT}/paired_confirmation_gate.json"

echo "Paired confirmation passed: ${CONFIRM_ROOT}/paired_confirmation_gate.json"
