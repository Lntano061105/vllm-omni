#!/usr/bin/env python3
"""Verify that requested MiniCPM-o optimizations actually ran."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FATAL_MARKERS = (
    "Orchestrator thread crashed",
    "Engine core initialization failed",
    "MTE DDR",
    "ERR99999 UNKNOWN",
    "Traceback (most recent call last)",
    "AssertionError: Expected a cached item for mm_hash=",
)


def validate_log(
    text: str,
    *,
    require_stage1_full_decode: bool = False,
    require_stage0_ref_cache: bool = False,
    require_stage2_prompt_cache: bool = False,
    require_stage2_runner_prewarm: bool = False,
    require_stage2_npugraph: bool = False,
    require_stage1_cpu_slot_mapping: bool = False,
    require_stage1_graph_sampler: bool = False,
    require_stage1_binary_argmax: bool = False,
    require_stage0_mm_cache_disabled: bool = False,
    min_npugraph_buckets: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    fatal_hits = [marker for marker in FATAL_MARKERS if marker in text]
    failures.extend(f"fatal log marker present: {marker}" for marker in fatal_hits)

    checks: dict[str, bool] = {}

    def require(label: str, marker: str) -> None:
        present = marker in text
        checks[label] = present
        if not present:
            failures.append(f"missing activation marker: {label} ({marker})")

    if require_stage1_full_decode:
        require("Stage 1 FULL_DECODE_ONLY", "CUDAGraphMode.FULL_DECODE_ONLY")
        require("Stage 1 NPUGraph_ex", "enable_npugraph_ex': True")
        require("Stage 1 full graph capture", "Capturing CUDA graphs (decode, FULL)")
        if "CUDAGraphMode.PIECEWISE" in "\n".join(
            line for line in text.splitlines() if "StageEngineCoreProc_stage1" in line
        ):
            failures.append("Stage 1 silently fell back to PIECEWISE")

    if require_stage1_cpu_slot_mapping:
        require(
            "Stage 1 CPU slot mapping configured",
            "NPU Talker runner-local CPU slot mapping enabled",
        )
        require(
            "Stage 1 CPU slot mapping used",
            "Used NPU Talker runner-local CPU slot mapping fast path",
        )
        if "falling back to the standard GPU kernel" in text:
            failures.append("Stage 1 CPU slot mapping fell back to the GPU kernel")

    if require_stage1_graph_sampler:
        require(
            "Stage 1 graph sampler model enabled",
            "MiniCPM-o Talker graph-contained exact compact sampler enabled",
        )
        require(
            "Stage 1 graph sampler runner enabled",
            "NPU Talker graph sampler inputs enabled",
        )
        require(
            "Stage 1 graph sampler incremental history used",
            "Used NPU Talker incremental graph sampler history fast path",
        )

    if require_stage1_binary_argmax:
        require(
            "Stage 1 binary control argmax enabled",
            "MiniCPM-o Talker deterministic binary control argmax enabled",
        )
        require(
            "Stage 1 binary control argmax used",
            "Used MiniCPM-o Talker deterministic binary control argmax",
        )

    if require_stage0_ref_cache:
        require(
            "Stage 0 reference embedding cached",
            "Cached MiniCPM-o Stage-0 reference audio embedding",
        )
        require(
            "Stage 0 reference embedding reused",
            "Reused MiniCPM-o Stage-0 reference audio embedding cache",
        )

    if require_stage0_mm_cache_disabled:
        require(
            "Stage 0 multimodal processor cache disabled",
            "Stage 0 multimodal processor cache: disabled (mm_processor_cache_gb=0)",
        )

    if require_stage2_prompt_cache:
        require("Stage 2 prompt preload", "Preloaded MiniCPM-o runtime prompt")
        require(
            "Stage 2 immutable initial state cached",
            "Cached MiniCPM-o Token2Wav immutable initial state",
        )
        require(
            "Stage 2 immutable initial state reused",
            "Reused MiniCPM-o Token2Wav immutable initial state",
        )
        require("Stage 2 bounded prompt cache", "runtime_prompt_cache_size=4")
        require("Stage 2 runtime initial-state switch", "cache_runtime_initial_state=True")

    if require_stage2_runner_prewarm:
        require(
            "Stage 2 backend 13/25 prewarm",
            "live codec chunks [13, 25] with 3 left-context frames",
        )
        require(
            "Stage 2 runner 13/25 prewarm",
            "Code2Wav runner prewarm completed: prompts=1 live codec chunks=[13, 25] left_context=3",
        )

    captured_buckets = text.count("Captured steady MiniCPM-o Token2Wav NPUGraph bucket")
    if require_stage2_npugraph:
        checks["Stage 2 NPUGraph bucket count"] = captured_buckets >= min_npugraph_buckets
        if captured_buckets < min_npugraph_buckets:
            failures.append(
                "insufficient Stage 2 NPUGraph buckets: "
                f"captured={captured_buckets} required={min_npugraph_buckets}"
            )
        require(
            "Stage 2 NPUGraph replay",
            "Replayed steady MiniCPM-o Token2Wav NPUGraph",
        )

    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "captured_npugraph_buckets": captured_buckets,
        "fatal_markers": fatal_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--require-stage1-full-decode", action="store_true")
    parser.add_argument("--require-stage0-ref-cache", action="store_true")
    parser.add_argument("--require-stage2-prompt-cache", action="store_true")
    parser.add_argument("--require-stage2-runner-prewarm", action="store_true")
    parser.add_argument("--require-stage2-npugraph", action="store_true")
    parser.add_argument("--require-stage1-cpu-slot-mapping", action="store_true")
    parser.add_argument("--require-stage1-graph-sampler", action="store_true")
    parser.add_argument("--require-stage1-binary-argmax", action="store_true")
    parser.add_argument("--require-stage0-mm-cache-disabled", action="store_true")
    parser.add_argument("--min-npugraph-buckets", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.min_npugraph_buckets < 1:
        parser.error("--min-npugraph-buckets must be >= 1")
    result = validate_log(
        args.server_log.read_text(encoding="utf-8", errors="replace"),
        require_stage1_full_decode=args.require_stage1_full_decode,
        require_stage0_ref_cache=args.require_stage0_ref_cache,
        require_stage2_prompt_cache=args.require_stage2_prompt_cache,
        require_stage2_runner_prewarm=args.require_stage2_runner_prewarm,
        require_stage2_npugraph=args.require_stage2_npugraph,
        require_stage1_cpu_slot_mapping=args.require_stage1_cpu_slot_mapping,
        require_stage1_graph_sampler=args.require_stage1_graph_sampler,
        require_stage1_binary_argmax=args.require_stage1_binary_argmax,
        require_stage0_mm_cache_disabled=args.require_stage0_mm_cache_disabled,
        min_npugraph_buckets=args.min_npugraph_buckets,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
