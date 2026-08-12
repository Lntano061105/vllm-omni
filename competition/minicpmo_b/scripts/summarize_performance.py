#!/usr/bin/env python3
"""Render vLLM-Omni performance JSON files as a Markdown A/B table."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path


METRICS = (
    ("mean_ttft_ms", "TTFT ms", False),
    ("mean_audio_ttfp_ms", "TTFP ms", False),
    ("mean_audio_rtf", "Audio RTF", False),
    ("mean_audio_chunk_rtf", "Steady chunk RTF", False),
    ("mean_audio_speak_generation_rtf", "SPEAK generation RTF", False),
    ("mean_audio_speak_tail_rtf", "SPEAK tail RTF", False),
    ("mean_e2el_ms", "E2EL ms", False),
    ("request_throughput", "Req/s", True),
    ("audio_continuity_ok_rate", "Continuity", True),
    ("mean_audio_underrun_s", "Underrun s", False),
)


def parse_case(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        return label, Path(raw_path)
    path = Path(value)
    return path.parent.name or path.stem, path


def display(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cases",
        nargs="+",
        help="JSON path or LABEL=JSON_PATH; the first case is the delta baseline",
    )
    parser.add_argument("--output", type=Path, help="Also write the Markdown table here.")
    args = parser.parse_args()

    cases: list[tuple[str, dict[str, object]]] = []
    for raw in args.cases:
        label, path = parse_case(raw)
        cases.append((label, json.loads(path.read_text(encoding="utf-8"))))

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        labels = [label for label, _ in cases]
        print("| Metric | " + " | ".join(labels) + " | " + " | ".join(f"{label} vs {labels[0]}" for label in labels[1:]) + " |")
        print("|---|" + "---:|" * (len(labels) * 2 - 1))
        baseline = cases[0][1]
        for key, title, higher_is_better in METRICS:
            values = [data.get(key) for _, data in cases]
            deltas: list[str] = []
            base_value = baseline.get(key)
            for value in values[1:]:
                if not isinstance(base_value, (int, float)) or not isinstance(value, (int, float)) or base_value == 0:
                    deltas.append("N/A")
                    continue
                raw_delta = (float(value) / float(base_value) - 1.0) * 100.0
                score_delta = raw_delta if higher_is_better else -raw_delta
                deltas.append(f"{raw_delta:+.2f}% ({score_delta:+.2f}% score)")
            print("| " + title + " | " + " | ".join(display(value) for value in values) + " | " + " | ".join(deltas) + " |")

    rendered = stream.getvalue()
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
