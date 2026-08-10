#!/usr/bin/env python3
"""Repeat the native MiniCPM-o duplex scenario and aggregate challenge metrics.

This runner deliberately uses the model-owned LISTEN/SPEAK path
(``auto_response=true`` and ``minicpmo45_native_duplex=true``).  The legacy
Realtime Seed-TTS backend sends text with native duplex disabled, so it cannot
measure the challenge's SPEAK-generation phase.

The local phase boundary is carried by ``metadata.speak_tail``.  It is set on
every audio packet descended from the final Thinker ``<turn_eos>`` handoff;
those packets are excluded from SPEAK-generation RTF and reported separately.
The organizer's official script remains authoritative for final ranking.
"""

from __future__ import annotations

import argparse
import audioop
import concurrent.futures
import json
import math
import statistics
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO = REPO_ROOT / "tests/e2e/online_serving/minicpmo_realtime_duplex_scenarios.py"


def _prepare_input_wav(source: Path, output_dir: Path) -> Path:
    """Normalize a PCM16 WAV to the native duplex model's 16 kHz mono input."""
    with wave.open(str(source), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"native duplex fixture must be PCM16, got sample width {sample_width}: {source}")
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    if channels != 1:
        raise ValueError(f"native duplex fixture must be mono or stereo, got {channels} channels: {source}")
    if sample_rate == 16_000:
        return source
    frames, _ = audioop.ratecv(frames, sample_width, channels, sample_rate, 16_000, None)
    normalized = output_dir / "input_16k_mono_pcm16.wav"
    with wave.open(str(normalized), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(frames)
    return normalized


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p99": _percentile(values, 99.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _extract_result(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [0]
    candidates.extend(index + 1 for index, char in enumerate(stdout) if char == "\n")
    parsed: list[dict[str, Any]] = []
    for index in candidates:
        if index >= len(stdout) or stdout[index] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "session_metrics" in value:
            parsed.append(value)
    if not parsed:
        raise RuntimeError(f"scenario produced no result JSON; output tail:\n{stdout[-4000:]}")
    return parsed[-1]


def _scenario_command(args: argparse.Namespace, run_index: int, *, warmup: bool) -> list[str]:
    run_dir = args.output_dir / (f"warmup_{run_index:03d}" if warmup else f"run_{run_index:03d}")
    command = [
        sys.executable,
        str(SCENARIO),
        "--url",
        args.url,
        "--model",
        args.model,
        "--input-wav",
        str(args.input_wav),
        "--output-dir",
        str(run_dir),
        "--output-audio-format",
        "pcm16",
        "--chunk-ms",
        str(args.input_chunk_ms),
        "--turns",
        str(args.turns_per_session),
        "--first-turn-transcript",
        args.transcript,
        "--validation-mode",
        "response-required",
        "--temperature",
        "0",
        "--timeout-s",
        str(args.timeout_s),
        "--require-audio",
    ]
    if args.ref_audio is not None:
        command.extend(("--ref-audio", str(args.ref_audio)))
    for _ in range(max(0, args.turns_per_session - 1)):
        command.extend(("--turn-input-wav", str(args.input_wav)))
    for _ in range(args.turns_per_session):
        command.extend(("--turn-duration-ms", str(args.turn_duration_ms)))
    return command


def _run_once(args: argparse.Namespace, run_index: int, *, warmup: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        _scenario_command(args, run_index, warmup=warmup),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.process_timeout_s,
        check=False,
    )
    try:
        result = _extract_result(completed.stdout)
    except Exception:
        log_path = args.output_dir / (f"warmup_{run_index:03d}.log" if warmup else f"run_{run_index:03d}.log")
        log_path.write_text(completed.stdout, encoding="utf-8")
        raise
    if completed.returncode != 0 or not result.get("ok"):
        raise RuntimeError(
            f"native duplex run {run_index} failed (rc={completed.returncode}): "
            f"errors={result.get('errors')} outcomes={result.get('turn_outcomes')}"
        )
    return result


def _finite(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]


def _aggregate(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [
        request
        for result in results
        for request in result.get("request_metrics", [])
        if isinstance(request, dict)
    ]
    generation_rtfs = [
        value
        for request in requests
        for value in _finite(request.get("speak_generation_chunk_rtfs"))
    ]
    tail_rtfs = [
        value
        for request in requests
        for value in _finite(request.get("speak_tail_chunk_rtfs"))
    ]
    ttfts = [float(request["ttft_ms"]) for request in requests if isinstance(request.get("ttft_ms"), int | float)]
    ttfps = [float(request["ttfp_ms"]) for request in requests if isinstance(request.get("ttfp_ms"), int | float)]
    request_rtfs = [float(request["rtf"]) for request in requests if isinstance(request.get("rtf"), int | float)]
    generation = _summary(generation_rtfs)
    tail = _summary(tail_rtfs)
    return {
        "benchmark": "minicpmo45-native-duplex-speak-generation-rtf",
        "metric_contract": {
            "speak_generation_rtf": (
                "inter-packet client receive interval / current packet audio duration; "
                "only packets with metadata.speak_tail=false; first packet excluded"
            ),
            "speak_tail_boundary": "final Thinker <turn_eos> handoff propagated through Talker and Code2Wav",
            "authority": "local proxy; organizer official evaluator is authoritative",
        },
        "server": {"url": args.url, "model": args.model},
        "input": {
            "source_wav": str(args.source_input_wav),
            "wav": str(args.input_wav),
            "ref_audio": str(args.ref_audio) if args.ref_audio is not None else None,
            "transcript_hint": args.transcript,
            "turn_duration_ms": args.turn_duration_ms,
            "turns_per_session": args.turns_per_session,
            "input_chunk_ms": args.input_chunk_ms,
        },
        "sessions": len(results),
        "audio_turns": len(requests),
        "max_concurrency": args.max_concurrency,
        "warmups": args.num_warmups,
        "mean_audio_speak_generation_rtf": generation["mean"],
        "median_audio_speak_generation_rtf": generation["median"],
        "p99_audio_speak_generation_rtf": generation["p99"],
        "audio_speak_generation_chunk_count": generation["count"],
        "mean_audio_speak_tail_rtf": tail["mean"],
        "median_audio_speak_tail_rtf": tail["median"],
        "p99_audio_speak_tail_rtf": tail["p99"],
        "audio_speak_tail_chunk_count": tail["count"],
        "ttft_ms": _summary(ttfts),
        "ttfp_ms": _summary(ttfps),
        "request_audio_rtf": _summary(request_rtfs),
        "speak_generation_rtf": generation,
        "speak_tail_rtf": tail,
        "request_metrics": requests,
        "runs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8091/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-filename", default="native_duplex_rtf.json")
    parser.add_argument("--num-sessions", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--turns-per-session", type=int, default=1)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--input-chunk-ms", type=int, default=200)
    parser.add_argument("--turn-duration-ms", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--process-timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    if args.num_sessions <= 0 or args.max_concurrency <= 0 or args.turns_per_session <= 0:
        parser.error("session, concurrency, and turn counts must be positive")
    if args.turn_duration_ms < 0:
        parser.error("--turn-duration-ms must be non-negative")
    if not args.input_wav.is_file():
        parser.error(f"input WAV not found: {args.input_wav}")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        parser.error(f"reference audio not found: {args.ref_audio}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_input_wav = args.input_wav
    args.input_wav = _prepare_input_wav(args.input_wav, args.output_dir)
    for index in range(args.num_warmups):
        _run_once(args, index, warmup=True)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = [executor.submit(_run_once, args, index) for index in range(args.num_sessions)]
        for future in futures:
            results.append(future.result())

    aggregate = _aggregate(args, results)
    result_path = args.output_dir / args.result_filename
    result_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if not aggregate["audio_speak_generation_chunk_count"]:
        raise SystemExit(
            "No SPEAK-generation chunks were observed. Confirm that the server contains "
            "the explicit speak_tail propagation and that the fixture produces a multi-unit spoken response."
        )


if __name__ == "__main__":
    main()
