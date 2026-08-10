#!/usr/bin/env python3
"""Compute deterministic waveform sanity metrics for saved Seed-TTS audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze(path: Path) -> dict[str, float | int | str]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    absolute = np.abs(mono)
    adjacent = np.abs(np.diff(mono)) if mono.size > 1 else np.zeros(0, dtype=np.float32)
    near_silence = absolute < 1e-4
    return {
        "file": str(path),
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_s": float(mono.size / sample_rate) if sample_rate > 0 else 0.0,
        "peak": float(absolute.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0,
        "dc_offset": float(np.mean(mono)) if mono.size else 0.0,
        "clipping_rate": float(np.mean(absolute >= 0.999)) if mono.size else 0.0,
        "near_silence_rate": float(np.mean(near_silence)) if mono.size else 1.0,
        "max_adjacent_jump": float(adjacent.max(initial=0.0)),
        "p999_adjacent_jump": float(np.percentile(adjacent, 99.9)) if adjacent.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="WAV files or directories")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.wav")))
        elif path.suffix.lower() == ".wav":
            files.append(path)
    rows = [analyze(path) for path in files]
    payload = {"files": rows, "count": len(rows)}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
