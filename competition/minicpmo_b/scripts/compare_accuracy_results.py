#!/usr/bin/env python3
"""Compare complete baseline/optimized accuracy results with a 2pp gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_METRICS = {
    "daily-omni": (("daily_omni_accuracy", "higher"),),
    "videomme": (("videomme_accuracy", "higher"),),
    # WER is an error rate; SIM and UTMOS are quality scores. Official retests
    # require all three so a fast but degraded vocoder cannot pass on ASR alone.
    "seed-tts": (
        ("seed_tts_content_error_mean", "lower"),
        ("seed_tts_sim_mean", "higher"),
        ("seed_tts_utmos_mean", "higher"),
    ),
}
_ABSOLUTE_OPTIMIZED_GATES = {
    "daily-omni": ("daily_omni_accuracy", "higher", 0.78),
    "videomme": ("videomme_accuracy", "higher", 0.68),
    "seed-tts": ("seed_tts_content_error_mean", "lower", 0.05),
}


def _load_result(path: Path) -> dict[str, Any]:
    if path.is_dir():
        complete_seed = sorted(path.glob("seed_tts_quality_resumed*.json"))
        benchmark_results = sorted(path.glob("qwen_omni_acc_*.json")) + sorted(
            path.glob("omni_acc_videomme_*.json")
        )
        matches = complete_seed or benchmark_results
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one accuracy result under {path}, found {len(matches)}"
            )
        path = matches[0]
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_expected_requests(path: Path) -> int | None:
    directory = path if path.is_dir() else path.parent
    protocol_path = directory / "run_protocol.json"
    if not protocol_path.is_file():
        return None
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    comparison = protocol.get("comparison")
    fields = comparison.get("fields") if isinstance(comparison, dict) else None
    value = fields.get("num_prompts") if isinstance(fields, dict) else None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"protocol has no positive integer num_prompts: {protocol_path}")
    return value


def compare_accuracy(
    suite: str,
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    *,
    max_regression: float = 0.02,
    expected_requests: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    metric_results: dict[str, Any] = {}
    if suite == "seed-tts":
        for label, result in (("baseline", baseline), ("optimized", optimized)):
            if result.get("seed_tts_quality_complete") is not True:
                failures.append(f"{label} seed_tts_quality_complete is not true")
            expected = int(result.get("completed", 0) or 0)
            sessions = int(result.get("seed_tts_session_count", 0) or 0)
            turns = int(result.get("seed_tts_turn_count", 0) or 0)
            if expected <= 0:
                failures.append(f"{label} completed is not positive")
            if expected_requests is not None and expected != expected_requests:
                failures.append(
                    f"{label} completed={expected} != expected {expected_requests}"
                )
            if int(result.get("failed", 0) or 0) != 0:
                failures.append(f"{label} failed is not zero")
            if sessions != expected:
                failures.append(
                    f"{label} seed_tts_session_count={sessions} != completed={expected}"
                )
            for key in (
                "seed_tts_content_evaluated",
                "seed_tts_sim_evaluated",
                "seed_tts_utmos_evaluated",
            ):
                value = int(result.get(key, 0) or 0)
                if value != turns or turns <= 0:
                    failures.append(f"{label} {key}={value} != turns={turns}")
            for key in (
                "seed_tts_request_failed",
                "seed_tts_no_pcm",
                "seed_tts_asr_failed",
                "seed_tts_save_audio_failed",
                "seed_tts_sim_failed",
                "seed_tts_sim_skipped_no_ref",
                "seed_tts_utmos_failed",
            ):
                if int(result.get(key, 0) or 0) != 0:
                    failures.append(f"{label} {key} is not zero")
    for metric, direction in _METRICS[suite]:
        try:
            baseline_value = float(baseline[metric])
            if not math.isfinite(baseline_value):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            baseline_value = None
            failures.append(f"baseline missing numeric {metric}")
        try:
            optimized_value = float(optimized[metric])
            if not math.isfinite(optimized_value):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            optimized_value = None
            failures.append(f"optimized missing numeric {metric}")

        regression = None
        if baseline_value is not None and optimized_value is not None:
            regression = (
                baseline_value - optimized_value
                if direction == "higher"
                else optimized_value - baseline_value
            )
            if regression > max_regression + 1e-12:
                failures.append(
                    f"{metric} regression={regression:.6f} exceeds {max_regression:.6f}"
                )
        metric_results[metric] = {
            "direction": direction,
            "baseline": baseline_value,
            "optimized": optimized_value,
            "regression": regression,
        }

    primary_metric, primary_direction = _METRICS[suite][0]
    primary = metric_results[primary_metric]
    absolute_metric, absolute_direction, absolute_threshold = _ABSOLUTE_OPTIMIZED_GATES[suite]
    for label in ("baseline", "optimized"):
        value = metric_results[absolute_metric][label]
        failed_absolute = value is not None and (
            value < absolute_threshold - 1e-12
            if absolute_direction == "higher"
            else value > absolute_threshold + 1e-12
        )
        if failed_absolute:
            if suite == "seed-tts":
                failures.append(
                    f"{label} Seed-TTS mean WER={value:.6f} exceeds "
                    f"absolute gate {absolute_threshold:.6f}"
                )
            else:
                comparator = "below" if absolute_direction == "higher" else "exceeds"
                failures.append(
                    f"{label} {absolute_metric}={value:.6f} {comparator} "
                    f"absolute gate {absolute_threshold:.6f}"
                )

    return {
        "passed": not failures,
        "suite": suite,
        "metric": primary_metric,
        "direction": primary_direction,
        "baseline": primary["baseline"],
        "optimized": primary["optimized"],
        "regression": primary["regression"],
        "metrics": metric_results,
        "max_regression": max_regression,
        "expected_requests": expected_requests,
        "absolute_gate": {
            "metric": absolute_metric,
            "direction": absolute_direction,
            "threshold": absolute_threshold,
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(_METRICS))
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument(
        "--expected-requests",
        type=int,
        help="Require completed/session count; defaults to 1000 for seed-tts.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_regression < 0:
        parser.error("--max-regression must be >= 0")

    expected_requests = args.expected_requests
    if expected_requests is None:
        baseline_expected = _protocol_expected_requests(args.baseline)
        optimized_expected = _protocol_expected_requests(args.optimized)
        if baseline_expected is not None or optimized_expected is not None:
            if baseline_expected != optimized_expected:
                raise SystemExit(
                    "baseline/optimized protocol num_prompts differ: "
                    f"{baseline_expected!r} != {optimized_expected!r}"
                )
            expected_requests = baseline_expected
        elif args.suite == "seed-tts":
            expected_requests = 1000
    result = compare_accuracy(
        args.suite,
        _load_result(args.baseline),
        _load_result(args.optimized),
        max_regression=args.max_regression,
        expected_requests=expected_requests,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
