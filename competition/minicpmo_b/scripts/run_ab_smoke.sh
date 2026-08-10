#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vllm-workspace/vllm-omni}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
PORT="${PORT:-8092}"
OPTIMIZED_PORT="${OPTIMIZED_PORT:-$((PORT + 1))}"

CASE_NAME=baseline \
DEPLOY_CONFIG="${REPO_ROOT}/vllm_omni/deploy/minicpmo_4_5.yaml" \
NUM_PROMPTS="${NUM_PROMPTS}" \
MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
PORT="${PORT}" \
competition/minicpmo_b/scripts/run_perf_case.sh

CASE_NAME=optimized \
DEPLOY_CONFIG="${REPO_ROOT}/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml" \
NUM_PROMPTS="${NUM_PROMPTS}" \
MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
PORT="${OPTIMIZED_PORT}" \
competition/minicpmo_b/scripts/run_perf_case.sh
