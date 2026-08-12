from __future__ import annotations

import json
import runpy
import sys
import types
import wave
from pathlib import Path


def test_resume_seed_tts_quality_merges_performance_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "00000_utt_en.wav"
    with wave.open(str(audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 32)

    manifest = tmp_path / "seed_tts_eval_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "index": 0,
                "utterance_id": "utt",
                "locale": "en",
                "reference_text": "hello",
                "reference_wav_path": "/dataset/ref.wav",
                "audio_path": str(audio),
                "request_success": True,
                "request_error": "",
                "checkpoint_error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base = tmp_path / "performance.json"
    base.write_text(
        json.dumps(
            {
                "completed": 1,
                "mean_ttft_ms": 123.0,
                "seed_tts_generation_checkpoint": True,
                "seed_tts_quality_complete": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "complete.json"

    dataset_mod = types.ModuleType(
        "vllm_omni.benchmarks.data_modules.seed_tts_dataset"
    )

    class FakeRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dataset_mod.SeedTTSSampleRequest = FakeRequest
    eval_mod = types.ModuleType("vllm_omni.benchmarks.data_modules.seed_tts_eval")
    eval_mod.compute_seed_tts_wer_metrics = lambda *_args, **_kwargs: {
        "seed_tts_content_evaluated": 1,
        "seed_tts_content_error_mean": 0.0,
    }
    eval_mod.print_seed_tts_wer_summary = lambda _metrics: None
    monkeypatch.setitem(sys.modules, dataset_mod.__name__, dataset_mod)
    monkeypatch.setitem(sys.modules, eval_mod.__name__, eval_mod)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resume_seed_tts_quality.py",
            str(manifest),
            "--base-result",
            str(base),
            "--output",
            str(output),
        ],
    )

    try:
        runpy.run_path(
            str(
                Path(__file__).resolve().parents[2]
                / "competition/minicpmo_b/scripts/resume_seed_tts_quality.py"
            ),
            run_name="__main__",
        )
    except SystemExit as exc:
        assert exc.code == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["completed"] == 1
    assert result["mean_ttft_ms"] == 123.0
    assert result["seed_tts_content_evaluated"] == 1
    assert result["seed_tts_generation_checkpoint"] is False
    assert result["seed_tts_quality_complete"] is True
