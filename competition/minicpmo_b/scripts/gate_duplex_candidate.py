#!/usr/bin/env python3
"""Machine-check a MiniCPM-o native-duplex performance candidate.

Only ``speak_generation_rtf`` is accepted as the competition RTF proxy. The
request-level and all-chunk RTF fields are deliberately ignored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    min_speak_rtf_improvement_pct: float
    max_speak_rtf_p99_regression_pct: float
    min_ttft_improvement_pct: float
    max_ttft_regression_pct: float
    min_ttfp_improvement_pct: float
    max_ttfp_regression_pct: float


PROFILES = {
    "host-control": GateThresholds(
        min_speak_rtf_improvement_pct=3.0,
        max_speak_rtf_p99_regression_pct=0.0,
        min_ttft_improvement_pct=0.0,
        max_ttft_regression_pct=2.0,
        min_ttfp_improvement_pct=0.0,
        max_ttfp_regression_pct=2.0,
    ),
    "slot-mapping": GateThresholds(
        min_speak_rtf_improvement_pct=5.0,
        max_speak_rtf_p99_regression_pct=0.0,
        min_ttft_improvement_pct=0.0,
        max_ttft_regression_pct=2.0,
        min_ttfp_improvement_pct=0.0,
        max_ttfp_regression_pct=2.0,
    ),
    "stage1-hotpath": GateThresholds(
        min_speak_rtf_improvement_pct=12.0,
        max_speak_rtf_p99_regression_pct=0.0,
        min_ttft_improvement_pct=0.0,
        max_ttft_regression_pct=2.0,
        min_ttfp_improvement_pct=0.0,
        max_ttfp_regression_pct=2.0,
    ),
    "npugraph": GateThresholds(
        min_speak_rtf_improvement_pct=15.0,
        max_speak_rtf_p99_regression_pct=0.0,
        min_ttft_improvement_pct=0.0,
        max_ttft_regression_pct=2.0,
        min_ttfp_improvement_pct=0.0,
        max_ttfp_regression_pct=2.0,
    ),
    "prompt-cache": GateThresholds(
        min_speak_rtf_improvement_pct=-2.0,
        max_speak_rtf_p99_regression_pct=2.0,
        min_ttft_improvement_pct=10.0,
        max_ttft_regression_pct=0.0,
        min_ttfp_improvement_pct=10.0,
        max_ttfp_regression_pct=0.0,
    ),
    "combined": GateThresholds(
        min_speak_rtf_improvement_pct=15.0,
        max_speak_rtf_p99_regression_pct=0.0,
        min_ttft_improvement_pct=10.0,
        max_ttft_regression_pct=0.0,
        min_ttfp_improvement_pct=10.0,
        max_ttfp_regression_pct=0.0,
    ),
}


def _number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric: {label}")
    return float(value)


def _summary_metric(payload: dict[str, Any], name: str, statistic: str) -> float:
    nested = payload.get(name)
    if isinstance(nested, dict):
        return _number(nested.get(statistic), label=f"{name}.{statistic}")
    legacy = {
        ("ttft_ms", "mean"): "mean_ttft_ms",
        ("ttfp_ms", "mean"): "mean_audio_ttfp_ms",
        ("speak_generation_rtf", "mean"): "mean_audio_speak_generation_rtf",
        ("speak_generation_rtf", "median"): "median_audio_speak_generation_rtf",
        ("speak_generation_rtf", "p99"): "p99_audio_speak_generation_rtf",
    }.get((name, statistic))
    return _number(payload.get(legacy), label=legacy or f"{name}.{statistic}")


def _improvement_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError(f"baseline metric must be positive, got {baseline}")
    return (1.0 - candidate / baseline) * 100.0


def _successful(payload: dict[str, Any]) -> tuple[int, int]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("native-duplex result is missing runs[]")
    return sum(bool(run.get("ok")) for run in runs if isinstance(run, dict)), len(runs)


def evaluate_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    baseline_ok, baseline_total = _successful(baseline)
    candidate_ok, candidate_total = _successful(candidate)
    failures: list[str] = []
    if baseline_ok != baseline_total:
        failures.append(f"baseline runs incomplete: {baseline_ok}/{baseline_total}")
    if candidate_ok != candidate_total:
        failures.append(f"candidate runs incomplete: {candidate_ok}/{candidate_total}")
    if candidate_total != baseline_total:
        failures.append(
            f"run count changed: baseline={baseline_total} candidate={candidate_total}"
        )

    baseline_turns = int(baseline.get("audio_turns", 0) or 0)
    candidate_turns = int(candidate.get("audio_turns", 0) or 0)
    if candidate_turns != baseline_turns:
        failures.append(
            f"audio turn count changed: baseline={baseline_turns} candidate={candidate_turns}"
        )
    baseline_chunks = int(baseline.get("audio_speak_generation_chunk_count", 0) or 0)
    candidate_chunks = int(candidate.get("audio_speak_generation_chunk_count", 0) or 0)
    # Packetization itself is an allowed latency optimization (the production
    # candidate uses a shorter initial codec packet), so equal request sets can
    # legitimately yield different SPEAK-generation chunk counts. Require at
    # least one measured competition-phase chunk per audio turn on both sides,
    # but do not reject a candidate solely for changing packet boundaries.
    if (
        baseline_chunks < max(1, baseline_turns)
        or candidate_chunks < max(1, candidate_turns)
    ):
        failures.append(
            "insufficient SPEAK generation chunks: "
            f"baseline={baseline_chunks}/{baseline_turns} turns, "
            f"candidate={candidate_chunks}/{candidate_turns} turns"
        )

    metric_specs = {
        "ttft_mean_ms": ("ttft_ms", "mean"),
        "ttfp_mean_ms": ("ttfp_ms", "mean"),
        "speak_rtf_mean": ("speak_generation_rtf", "mean"),
        "speak_rtf_median": ("speak_generation_rtf", "median"),
        "speak_rtf_p99": ("speak_generation_rtf", "p99"),
    }
    metrics: dict[str, dict[str, float]] = {}
    for label, (name, statistic) in metric_specs.items():
        base_value = _summary_metric(baseline, name, statistic)
        candidate_value = _summary_metric(candidate, name, statistic)
        metrics[label] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "improvement_pct": _improvement_pct(base_value, candidate_value),
        }

    checks = {
        "speak_rtf_mean": (
            metrics["speak_rtf_mean"]["improvement_pct"]
            >= thresholds.min_speak_rtf_improvement_pct,
            f"SPEAK RTF mean improvement below {thresholds.min_speak_rtf_improvement_pct:.2f}%",
        ),
        "speak_rtf_p99": (
            metrics["speak_rtf_p99"]["improvement_pct"]
            >= -thresholds.max_speak_rtf_p99_regression_pct,
            f"SPEAK RTF p99 regression exceeds {thresholds.max_speak_rtf_p99_regression_pct:.2f}%",
        ),
        "ttft_mean": (
            metrics["ttft_mean_ms"]["improvement_pct"]
            >= (
                thresholds.min_ttft_improvement_pct
                if thresholds.min_ttft_improvement_pct > 0
                else -thresholds.max_ttft_regression_pct
            ),
            "TTFT gate failed",
        ),
        "ttfp_mean": (
            metrics["ttfp_mean_ms"]["improvement_pct"]
            >= (
                thresholds.min_ttfp_improvement_pct
                if thresholds.min_ttfp_improvement_pct > 0
                else -thresholds.max_ttfp_regression_pct
            ),
            "TTFP gate failed",
        ),
    }
    failures.extend(message for passed, message in checks.values() if not passed)
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "baseline": {
            "successful_runs": baseline_ok,
            "total_runs": baseline_total,
            "audio_turns": baseline_turns,
            "speak_generation_chunks": baseline_chunks,
        },
        "candidate": {
            "successful_runs": candidate_ok,
            "total_runs": candidate_total,
            "audio_turns": candidate_turns,
            "speak_generation_chunks": candidate_chunks,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_candidate(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        PROFILES[args.profile],
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
