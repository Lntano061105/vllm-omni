#!/usr/bin/env python3
"""Validate the strict 2-session x 3-turn native-duplex lifecycle gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    runs = payload.get("runs")
    if payload.get("sessions") != 2:
        failures.append(f"sessions={payload.get('sessions')!r}, expected 2")
    if payload.get("audio_turns") != 6:
        failures.append(f"audio_turns={payload.get('audio_turns')!r}, expected 6")
    if not isinstance(runs, list) or len(runs) != 2:
        failures.append(f"runs={len(runs) if isinstance(runs, list) else None}, expected 2")
    else:
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                failures.append(f"run[{index}] is not an object")
                continue
            expected = {
                "ok": True,
                "done_count": 3,
                "cancelled_count": 0,
                "stale_audio_delta_count": 0,
                "truncate_count": 0,
                "error_count": 0,
                "lifecycle_counts_ok": True,
                "cross_turn_independent_ok": True,
            }
            for key, value in expected.items():
                if run.get(key) != value:
                    failures.append(
                        f"run[{index}].{key}={run.get(key)!r}, expected {value!r}"
                    )
    return {
        "passed": not failures,
        "expected_sessions": 2,
        "expected_turns_per_session": 3,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate(json.loads(args.result.read_text(encoding="utf-8")))
    report["result"] = str(args.result)
    if args.server_log is not None:
        warning_lines = [
            line
            for line in args.server_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if "Enqueue save_async" in line and "previous_chunks_sent=" in line
        ]
        report["server_log"] = str(args.server_log)
        report["watermark_regression_count"] = len(warning_lines)
        if warning_lines:
            report["passed"] = False
            report["failures"].append(
                f"server log contains {len(warning_lines)} resumable-segment watermark warning(s)"
            )

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
