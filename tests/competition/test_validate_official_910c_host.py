from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/validate_official_910c_host.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_official_910c_host", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return {
        "model_path": model,
        "daily_omni_root": tmp_path / "daily",
        "videomme_root": tmp_path / "video",
        "seed_tts_root": tmp_path / "seed",
        "image_digest": "quay.io/ascend/vllm-omni@sha256:" + "a" * 64,
    }


def _asset_tree(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "model-assets"
    files = (
        "config.json",
        "assets/system_ref_audio.wav",
        "assets/token2wav/speech_tokenizer_v2_25hz.onnx",
        "assets/token2wav/campplus.onnx",
        "assets/token2wav/flow.pt",
        "assets/token2wav/flow.yaml",
        "assets/token2wav/hift.pt",
        "model-00001-of-00001.safetensors",
    )
    for name in files:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}', encoding="utf-8"
    )
    daily = tmp_path / "daily-assets"
    daily.mkdir()
    (daily / "qa.json").write_text('[{"q":"x"}]', encoding="utf-8")
    video = tmp_path / "video-assets/videomme"
    video.mkdir(parents=True)
    (video / "test-00000-of-00001.parquet").write_bytes(b"x")
    seed = tmp_path / "seed-assets/en"
    seed.mkdir(parents=True)
    (seed / "meta.lst").write_text("id|prompt|prompt-wavs/ref.wav|target\n" * 1000, encoding="utf-8")
    (seed / "prompt-wavs").mkdir()
    (seed / "prompt-wavs/ref.wav").write_bytes(b"x")
    whisper = tmp_path / "whisper"
    wavlm = tmp_path / "wavlm"
    for root in (whisper, wavlm):
        root.mkdir()
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"x")
    utmos = tmp_path / "utmos.jit"
    utmos.write_bytes(b"x")
    return {
        "model_path": model,
        "daily_omni_root": daily,
        "videomme_root": video.parent,
        "seed_tts_root": seed.parent,
        "whisper_model": whisper,
        "wavlm_model": wavlm,
        "utmos_model": utmos,
        "result_root": tmp_path / "results",
    }


def test_accepts_one_idle_910c(tmp_path: Path) -> None:
    result = _MOD.inspect_host(
        device=0,
        npu_smi_text="| 0  Ascend 910C | OK |\n",
        process_text="1 0 S /sbin/init\n",
        **_inputs(tmp_path),
    )
    assert result["passed"] is True
    assert result["single_910c"] is True
    assert result["warnings"] == []


def test_rejects_910b_or_stale_server(tmp_path: Path) -> None:
    result = _MOD.inspect_host(
        device=0,
        npu_smi_text="| 0  Ascend 910B4 | OK |\n",
        process_text="99 1 S vllm serve /workspace/MiniCPM-o-4_5\n",
        **_inputs(tmp_path),
    )
    assert result["passed"] is False
    assert any("not a 910C" in failure for failure in result["failures"])
    assert any("stale" in failure for failure in result["failures"])


def test_rejects_multiple_visible_devices(tmp_path: Path) -> None:
    result = _MOD.inspect_host(
        device=0,
        npu_smi_text="| 0  Ascend 910C | OK |\n| 1  Ascend 910C | OK |\n",
        process_text="",
        **_inputs(tmp_path),
    )
    assert result["passed"] is False
    assert any("exactly one visible NPU" in failure for failure in result["failures"])


def test_rejects_selected_910c_when_occupied(tmp_path: Path) -> None:
    result = _MOD.inspect_host(
        device=0,
        npu_smi_text=(
            "| 0  Ascend 910C | OK |\n"
            "| 0       0                 | 12345         | python | 2048 |\n"
        ),
        process_text="12345 1 S python other_project.py\n",
        **_inputs(tmp_path),
    )
    assert result["passed"] is False
    assert any("occupied" in failure for failure in result["failures"])


def test_asset_preflight_accepts_complete_offline_assets(tmp_path: Path) -> None:
    assets = _MOD.inspect_assets(
        **_asset_tree(tmp_path),
        host="127.0.0.1",
        port=0,
        check_port=False,
        min_free_gb=0,
        check_media_files=False,
        video_parquet_rows=2700,
        package_versions={
            "s3tokenizer": "0.3.0",
            "onnxruntime": "1.23.2",
            "torch": "2.8.0",
            "torch-npu": "2.8.0",
            "vllm": "0.11.0",
            "vllm-omni": "0.25.0",
            "vllm-ascend": "0.11.0",
            "transformers": "4.57.0",
        },
    )
    assert assets["passed"] is True, assets["failures"]
    assert assets["daily_omni_rows"] == 1
    assert assets["seed_tts_rows"] == 1000


def test_asset_preflight_rejects_missing_evaluator_and_short_seed_set(tmp_path: Path) -> None:
    inputs = _asset_tree(tmp_path)
    inputs["utmos_model"].unlink()
    (inputs["seed_tts_root"] / "en/meta.lst").write_text("x\n" * 999, encoding="utf-8")
    assets = _MOD.inspect_assets(
        **inputs,
        host="127.0.0.1",
        port=0,
        check_port=False,
        min_free_gb=0,
        check_media_files=False,
        video_parquet_rows=2700,
        package_versions={
            "s3tokenizer": "0.3.0",
            "onnxruntime": "1",
            "torch": "1",
            "torch-npu": "1",
            "vllm": "1",
            "vllm-omni": "0.25.0",
            "vllm-ascend": "1",
            "transformers": "1",
        },
    )
    assert assets["passed"] is False
    assert any("expected at least 1000" in failure for failure in assets["failures"])
    assert any("utmos.jit" in failure for failure in assets["failures"])
