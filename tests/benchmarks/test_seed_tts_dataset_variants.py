"""Tests for SeedTTSTextDataset, SeedTTSTextSampleRequest, SeedTTSDesignDataset,
and SeedTTSDesignSampleRequest.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the data module directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "seed_tts_dataset.py"
_MODULE_NAME = "vllm_omni.benchmarks.data_modules.seed_tts_dataset"

if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (  # noqa: E402
    SeedTTSDataset,
    SeedTTSDesignDataset,
    SeedTTSDesignSampleRequest,
    SeedTTSSampleRequest,
    SeedTTSTextDataset,
    SeedTTSTextSampleRequest,
    SeedTTSTurn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_tts_root(tmp_path: Path) -> Path:
    """Minimal seed-tts-style directory with 5 entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    wav_dir = locale_dir / "prompt-wavs"
    wav_dir.mkdir()
    for i in range(5):
        (wav_dir / f"utt{i:03d}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    meta = "\n".join(f"utt{i:03d}|ref text {i}|prompt-wavs/utt{i:03d}.wav|target text {i}" for i in range(5))
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mock_tokenizer(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    tokenizer.get_vocab.return_value = {"<pad>": 0}
    tokenizer.all_special_ids = []
    tokenizer.all_special_tokens = []
    tokenizer.vocab_size = 1
    tokenizer.__len__.return_value = 1
    return tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seed_tts_dataset_groups_four_turns_with_one_reference(seed_tts_root, mock_tokenizer):
    ds = SeedTTSDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )

    requests = ds.sample(
        mock_tokenizer,
        num_requests=1,
        turns_per_session=4,
        no_oversample=True,
    )

    assert len(requests) == 1
    request = requests[0]
    assert [turn.utterance_id for turn in request.seed_tts_turns] == [
        "utt000",
        "utt001",
        "utt002",
        "utt003",
    ]
    assert [turn.target_text for turn in request.seed_tts_turns] == [
        "target text 0",
        "target text 1",
        "target text 2",
        "target text 3",
    ]
    assert request.seed_tts_speech_extra["ref_text"] == "ref text 0"
    assert request.seed_tts_ref_wav_path.endswith("utt000.wav")


def test_seed_tts_eval_expands_grouped_pcm_by_turn():
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
        _expand_seed_tts_turn_outputs,
    )

    request = SeedTTSSampleRequest(
        prompt="target 0",
        prompt_len=4,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="session-0",
        seed_tts_turns=tuple(
            SeedTTSTurn(utterance_id=f"utt-{index}", target_text=f"target {index}") for index in range(4)
        ),
    )
    output = types.SimpleNamespace(
        success=True,
        tts_turn_pcm_bytes=[bytes([index]) for index in range(4)],
    )

    requests, outputs = _expand_seed_tts_turn_outputs([request], [output])

    assert [item.prompt for item in requests] == [f"target {index}" for index in range(4)]
    assert [item.seed_tts_utterance_id for item in requests] == [f"utt-{index}" for index in range(4)]
    assert [item.tts_output_pcm_bytes for item in outputs] == [bytes([index]) for index in range(4)]


def test_seed_tts_eval_keeps_session_pcm_for_single_turn_chat():
    """Chat-omni Seed-TTS sets session PCM only; expand must not wipe it."""
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
        _expand_seed_tts_turn_outputs,
    )

    request = SeedTTSSampleRequest(
        prompt="target 0",
        prompt_len=4,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="session-0",
        seed_tts_turns=(SeedTTSTurn(utterance_id="utt-0", target_text="target 0"),),
    )
    pcm = b"\x01\x02\x03\x04"
    output = types.SimpleNamespace(
        success=True,
        tts_output_pcm_bytes=pcm,
        tts_turn_pcm_bytes=None,
    )

    requests, outputs = _expand_seed_tts_turn_outputs([request], [output])

    assert len(requests) == 1
    assert requests[0].prompt == "target 0"
    assert outputs[0].tts_output_pcm_bytes == pcm


def test_seed_tts_eval_checkpoints_audio_before_batch_asr(monkeypatch, tmp_path):
    """An interrupted Whisper pass must not discard the generation phase."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    request = SeedTTSSampleRequest(
        prompt="target words",
        prompt_len=2,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="session-0",
        seed_tts_utterance_id="utt-0",
        seed_tts_locale="en",
        seed_tts_ref_wav_path="/dataset/ref.wav",
    )
    output = types.SimpleNamespace(
        success=True,
        error="",
        tts_output_pcm_bytes=b"\x00\x00" * 240,
        tts_turn_pcm_bytes=None,
    )
    monkeypatch.setenv("SEED_TTS_WER_SAVE_AUDIO_DIR", str(tmp_path))
    monkeypatch.setenv("SEED_TTS_WHISPER_BATCH_SIZE", "8")
    monkeypatch.setattr(seed_tts_eval, "_missing_deps_message", lambda _lang: None)
    monkeypatch.setattr(
        seed_tts_eval,
        "_pcm_s16le_to_f32_16k",
        lambda *_args, **_kwargs: np.zeros(160, dtype=np.float32),
    )

    def interrupted_batch(_wavs):
        raise KeyboardInterrupt

    monkeypatch.setattr(seed_tts_eval, "_transcribe_en_batch_f32_16k", interrupted_batch)

    with pytest.raises(KeyboardInterrupt):
        seed_tts_eval.compute_seed_tts_wer_metrics([request], [output])

    manifest = tmp_path / "seed_tts_eval_manifest.jsonl"
    rows = [__import__("json").loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["utterance_id"] == "utt-0"
    assert rows[0]["reference_text"] == "target words"
    assert rows[0]["request_success"] is True
    assert Path(rows[0]["audio_path"]).is_file()


def test_seed_tts_text_dataset_omits_ref_audio(seed_tts_root, mock_tokenizer):
    ds = SeedTTSTextDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSTextSampleRequest)
        assert req.seed_tts_speech_extra is None or "ref_audio" not in (req.seed_tts_speech_extra or {})
        assert req.seed_tts_ref_wav_path == ""
        assert "target text" in req.prompt


# ---------------------------------------------------------------------------
# SeedTTSDesignDataset tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_tts_design_root(tmp_path: Path) -> Path:
    """seed-tts-design directory with 5-field meta.lst entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    meta = "\n".join(
        f"des{i:03d}|||target text {i}|A warm {['female', 'male'][i % 2]} voice with neutral accent." for i in range(5)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


def test_seed_tts_design_dataset_has_instructions(seed_tts_design_root, mock_tokenizer):
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        extra = req.seed_tts_speech_extra or {}
        assert "instructions" in extra
        assert extra["instructions"], "instructions must be non-empty"
        assert extra.get("task_type") == "VoiceDesign"
        assert "ref_audio" not in extra
        assert req.seed_tts_ref_wav_path == ""


def test_seed_tts_design_dataset_rejects_missing_description(seed_tts_design_root, mock_tokenizer):
    """Lines without a voice_description should be skipped."""
    locale_dir = seed_tts_design_root / "en"
    # The bad line has 4 fields, not 5, so will be filtered
    meta = "bad|||target text without description\n" + "\n".join(
        f"ok|||target text {i}|A clear female voice." for i in range(9)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=10, no_oversample=True)
    assert len(requests) == 9  # since we filter the bad row out and don't oversample
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        assert req.seed_tts_utterance_id == "ok"


def test_attach_sets_seed_tts_row_even_without_extra_body():
    """seed_tts_row=True must be set for SeedTTSTextSampleRequest (no extra body)."""
    from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSTextSampleRequest

    req = SeedTTSTextSampleRequest(
        prompt="hello world",
        prompt_len=2,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="test-0",
        seed_tts_speech_extra=None,
        seed_tts_ref_wav_path="",
    )
    assert req.seed_tts_speech_extra is None
    assert req.seed_tts_ref_wav_path == ""
    # The fix ensures that even with speech_extra=None, the function
    # sets seed_tts_row=True. We verify the source code has the fix.
    import inspect

    import vllm_omni.benchmarks.patch.patch as patch_mod

    src = inspect.getsource(patch_mod._attach_seed_tts_to_request_func_input)
    # seed_tts_row must be set BEFORE the 'if not ex: return' check
    row_pos = src.index("seed_tts_row")
    not_ex_pos = src.index("if not ex:")
    assert row_pos < not_ex_pos, "seed_tts_row must be set before 'if not ex: return'"


def test_seed_tts_whisper_transcribe_passes_attention_mask(monkeypatch):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    calls = {}

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __call__(self, wav, *, sampling_rate, return_tensors, return_attention_mask=False):
            calls["return_attention_mask"] = return_attention_mask
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            assert len(wav) > 0
            return types.SimpleNamespace(
                input_features=FakeTensor("features"),
                attention_mask=FakeTensor("mask") if return_attention_mask else None,
            )

        def get_decoder_prompt_ids(self, *, language, task):
            assert language == "english"
            assert task == "transcribe"
            return [(1, 2)]

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert skip_special_tokens
            assert predicted_ids == [[42]]
            return ["hello"]

    class FakeModel:
        def generate(self, input_features, **kwargs):
            calls["input_features"] = input_features
            calls["generate_kwargs"] = kwargs
            return [[42]]

    monkeypatch.setattr(seed_tts_eval, "_ensure_en_asr", lambda: None)
    monkeypatch.setattr(seed_tts_eval, "_en_processor", FakeProcessor())
    monkeypatch.setattr(seed_tts_eval, "_en_model", FakeModel())
    monkeypatch.setattr(seed_tts_eval, "_device", "cuda:1")

    text = seed_tts_eval._transcribe_en_f32_16k(np.ones(1600, dtype=np.float32))

    assert text == "hello"
    assert calls["return_attention_mask"] is True
    assert calls["input_features"].device == "cuda:1"
    assert calls["generate_kwargs"]["attention_mask"].device == "cuda:1"
    assert calls["generate_kwargs"]["forced_decoder_ids"] == [(1, 2)]


def test_seed_tts_whisper_batch_transcribe_uses_padding_and_attention_mask(
    monkeypatch,
):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    calls = {}

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __call__(
            self,
            wavs,
            *,
            sampling_rate,
            return_tensors,
            padding,
            return_attention_mask=False,
        ):
            calls["batch_size"] = len(wavs)
            calls["padding"] = padding
            calls["return_attention_mask"] = return_attention_mask
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            return types.SimpleNamespace(
                input_features=FakeTensor("features"),
                attention_mask=FakeTensor("mask"),
            )

        def get_decoder_prompt_ids(self, *, language, task):
            assert language == "english"
            assert task == "transcribe"
            return [(1, 2)]

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert skip_special_tokens
            assert predicted_ids == [[41], [42]]
            return [" first ", "second"]

    class FakeModel:
        def generate(self, input_features, **kwargs):
            calls["input_features"] = input_features
            calls["generate_kwargs"] = kwargs
            return [[41], [42]]

    monkeypatch.setattr(seed_tts_eval, "_ensure_en_asr", lambda: None)
    monkeypatch.setattr(seed_tts_eval, "_en_processor", FakeProcessor())
    monkeypatch.setattr(seed_tts_eval, "_en_model", FakeModel())
    monkeypatch.setattr(seed_tts_eval, "_device", "cpu")

    texts = seed_tts_eval._transcribe_en_batch_f32_16k(
        [np.ones(1600, dtype=np.float32), np.ones(800, dtype=np.float32)]
    )

    assert texts == ["first", "second"]
    assert calls["batch_size"] == 2
    assert calls["padding"] is True
    assert calls["return_attention_mask"] is True
    assert calls["input_features"].device == "cpu"
    assert calls["generate_kwargs"]["attention_mask"].device == "cpu"
    assert calls["generate_kwargs"]["forced_decoder_ids"] == [(1, 2)]


def test_seed_tts_utmos_loads_explicit_local_jit_without_hub(
    monkeypatch, tmp_path: Path
):
    import huggingface_hub
    import torch

    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    jit_path = tmp_path / "utmos.jit"
    jit_path.write_bytes(b"local-utmos-placeholder")
    loaded = {}

    class FakeModel:
        def eval(self):
            loaded["eval"] = True
            return self

    def fake_load(path, *, map_location):
        loaded["path"] = path
        loaded["map_location"] = map_location
        return FakeModel()

    def fail_hub_download(**kwargs):
        raise AssertionError(f"local UTMOS path unexpectedly used the Hub: {kwargs}")

    monkeypatch.setenv("SEED_TTS_UTMOS_JIT_FILE", str(jit_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_hub_download)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.jit, "load", fake_load)
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_model", None)
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_device", None)
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_load_failed", False)

    model = seed_tts_eval._ensure_utmos_jit_model()

    assert isinstance(model, FakeModel)
    assert loaded == {
        "path": str(jit_path.resolve()),
        "map_location": "cpu",
        "eval": True,
    }
    assert seed_tts_eval._utmos_jit_device == "cpu"
