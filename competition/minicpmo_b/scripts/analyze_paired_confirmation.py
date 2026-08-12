#!/usr/bin/env python3
"""Gate an A1→B1, B2→A2 fresh-start confirmation for 910C metrics.

The official matrix remains authoritative. This auxiliary gate detects a
candidate whose apparent gain is explained by service-start order or host/NPU
drift by requiring the same direction and minimum gain in both fresh-start
pairs, plus bounded repeat drift within each variant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = {
    "ttft_ms": ("ttft_ms", "mean", 10.0),
    "ttfp_ms": ("ttfp_ms", "mean", 10.0),
    "speak_generation_rtf": ("speak_generation_rtf", "mean", 15.0),
}


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing finite metric: {label}")
    value = float(value)
    if value <= 0:
        raise ValueError(f"metric must be positive: {label}={value}")
    return value


def _metric(payload: dict[str, Any], name: str, statistic: str) -> float:
    nested = payload.get(name)
    if isinstance(nested, dict):
        return _number(nested.get(statistic), f"{name}.{statistic}")
    legacy = {
        "ttft_ms": "mean_ttft_ms",
        "ttfp_ms": "mean_audio_ttfp_ms",
        "speak_generation_rtf": "mean_audio_speak_generation_rtf",
    }[name]
    return _number(payload.get(legacy), legacy)


def _validate_result(payload: dict[str, Any], label: str, failures: list[str]) -> None:
    sessions = int(payload.get("sessions", 0) or 0)
    runs = payload.get("runs")
    if sessions != 32:
        failures.append(f"{label} sessions={sessions}, expected 32")
    if not isinstance(runs, list) or len(runs) != 32:
        failures.append(f"{label} runs[] is not complete 32/32")
    elif not all(isinstance(run, dict) and run.get("ok") is True for run in runs):
        failures.append(f"{label} contains unsuccessful runs")
    if int(payload.get("audio_speak_generation_chunk_count", 0) or 0) <= 0:
        failures.append(f"{label} has no SPEAK-generation chunks")


def _improvement(baseline: float, optimized: float) -> float:
    return (1.0 - optimized / baseline) * 100.0


def _drift(first: float, second: float) -> float:
    return abs(second / first - 1.0) * 100.0


def evaluate(
    a1: dict[str, Any],
    b1: dict[str, Any],
    b2: dict[str, Any],
    a2: dict[str, Any],
    *,
    max_repeat_drift_pct: float = 10.0,
) -> dict[str, Any]:
    failures: list[str] = []
    for label, payload in (("A1", a1), ("B1", b1), ("B2", b2), ("A2", a2)):
        _validate_result(payload, label, failures)

    metrics: dict[str, Any] = {}
    for label, (name, statistic, minimum_gain) in METRICS.items():
        values = {
            "a1": _metric(a1, name, statistic),
            "b1": _metric(b1, name, statistic),
            "b2": _metric(b2, name, statistic),
            "a2": _metric(a2, name, statistic),
        }
        pair1_gain = _improvement(values["a1"], values["b1"])
        pair2_gain = _improvement(values["a2"], values["b2"])
        baseline_repeat_mean = (values["a1"] + values["a2"]) / 2.0
        optimized_repeat_mean = (values["b1"] + values["b2"]) / 2.0
        combined_gain = _improvement(baseline_repeat_mean, optimized_repeat_mean)
        baseline_drift = _drift(values["a1"], values["a2"])
        optimized_drift = _drift(values["b1"], values["b2"])
        if pair1_gain < minimum_gain:
            failures.append(
                f"{label} A1→B1 improvement {pair1_gain:.3f}% < {minimum_gain:.3f}%"
            )
        if pair2_gain < minimum_gain:
            failures.append(
                f"{label} A2→B2 improvement {pair2_gain:.3f}% < {minimum_gain:.3f}%"
            )
        if combined_gain < minimum_gain:
            failures.append(
                f"{label} two-repeat improvement {combined_gain:.3f}% < {minimum_gain:.3f}%"
            )
        if baseline_drift > max_repeat_drift_pct:
            failures.append(
                f"{label} baseline repeat drift {baseline_drift:.3f}% > {max_repeat_drift_pct:.3f}%"
            )
        if optimized_drift > max_repeat_drift_pct:
            failures.append(
                f"{label} optimized repeat drift {optimized_drift:.3f}% > {max_repeat_drift_pct:.3f}%"
            )
        metrics[label] = {
            **values,
            "pair1_improvement_pct": pair1_gain,
            "pair2_improvement_pct": pair2_gain,
            "two_repeat_improvement_pct": combined_gain,
            "baseline_two_repeat_mean": baseline_repeat_mean,
            "optimized_two_repeat_mean": optimized_repeat_mean,
            "baseline_repeat_drift_pct": baseline_drift,
            "optimized_repeat_drift_pct": optimized_drift,
            "minimum_improvement_pct": minimum_gain,
        }
    return {
        "passed": not failures,
        "method": "fresh-start A1-B1-B2-A2 order-bias confirmation",
        "max_repeat_drift_pct": max_repeat_drift_pct,
        "metrics": metrics,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a1_baseline", type=Path)
    parser.add_argument("b1_optimized", type=Path)
    parser.add_argument("b2_optimized", type=Path)
    parser.add_argument("a2_baseline", type=Path)
    parser.add_argument("--max-repeat-drift-pct", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.a1_baseline, args.b1_optimized, args.b2_optimized, args.a2_baseline)
    ]
    result = evaluate(*payloads, max_repeat_drift_pct=args.max_repeat_drift_pct)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
