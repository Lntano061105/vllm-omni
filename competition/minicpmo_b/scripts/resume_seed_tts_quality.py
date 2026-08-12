#!/usr/bin/env python3
"""Resume Seed-TTS WER/SIM/UTMOS from a generated-audio checkpoint."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from types import SimpleNamespace

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSSampleRequest
from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
    compute_seed_tts_wer_metrics,
    print_seed_tts_wer_summary,
)


def _wav_pcm_s16le_24k_mono(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (24000, 1, 2):
            raise ValueError(
                f"checkpoint WAV must be 24 kHz mono PCM16: {path} "
                f"got rate={wf.getframerate()} channels={wf.getnchannels()} "
                f"width={wf.getsampwidth()}"
            )
        return wf.readframes(wf.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-result",
        type=Path,
        help="Merge quality fields into the pre-ASR generation/performance checkpoint.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    requests = []
    outputs = []
    missing = 0
    for row in rows:
        audio_raw = row.get("audio_path")
        audio_path = Path(audio_raw) if audio_raw else None
        if audio_path is not None and not audio_path.is_absolute():
            from_cwd = audio_path.resolve()
            from_manifest = (args.manifest.parent / audio_path.name).resolve()
            audio_path = from_cwd if from_cwd.is_file() else from_manifest
        success = bool(row.get("request_success")) and audio_path is not None and audio_path.is_file()
        pcm = _wav_pcm_s16le_24k_mono(audio_path) if success and audio_path is not None else None
        if not success:
            missing += 1
        requests.append(
            SeedTTSSampleRequest(
                prompt=str(row.get("reference_text", "")),
                prompt_len=0,
                expected_output_len=0,
                multi_modal_data=None,
                request_id=f"resume-{row.get('index', len(requests))}",
                seed_tts_utterance_id=str(row.get("utterance_id", "")),
                seed_tts_locale=str(row.get("locale", "en")),
                seed_tts_ref_wav_path=str(row.get("reference_wav_path", "")),
            )
        )
        outputs.append(
            SimpleNamespace(
                success=success,
                error=str(row.get("request_error") or row.get("checkpoint_error") or ""),
                tts_output_pcm_bytes=pcm,
                tts_turn_pcm_bytes=None,
            )
        )

    if missing and not args.allow_incomplete:
        raise SystemExit(f"checkpoint is incomplete: {missing}/{len(rows)} rows have no usable audio")
    metrics = compute_seed_tts_wer_metrics(requests, outputs, include_per_item=True)
    if metrics is None:
        raise SystemExit("manifest did not produce Seed-TTS metrics")
    result = metrics
    if args.base_result is not None:
        base = json.loads(args.base_result.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise SystemExit("--base-result must contain one JSON object")
        result = {**base, **metrics}
        result["seed_tts_generation_checkpoint"] = False
        result["seed_tts_quality_complete"] = not bool(
            metrics.get("seed_tts_eval_setup_error")
        )
        result["seed_tts_quality_resumed_from_manifest"] = str(args.manifest)
        result["seed_tts_quality_base_result"] = str(args.base_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print_seed_tts_wer_summary(metrics)
    print(f"Saved resumed Seed-TTS quality metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
