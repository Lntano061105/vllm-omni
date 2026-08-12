from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    PromptFeatures,
    _SteadyNPUGraph,
    state_shape_signature,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _RuntimePrompt,
    _remove_weight_norm_for_inference,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.last_chunk_calls: list[bool] = []

    def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
        batch, length, _ = xs.shape
        self.calls.append(batch)
        self.last_chunk_calls.append(last_chunk)
        old_length = 0 if att_cache is None else att_cache.shape[3]
        output = xs[:, : max(1, length - 1)]
        cnn = xs[:, :1, :].transpose(1, 2).contiguous()
        marker = xs[:, 0, 0].reshape(1, batch, 1, 1, 1)
        att = marker.expand(1, batch, 1, old_length + output.shape[1], 1).clone()
        return output, cnn, att


class _FakeBlock:
    def __init__(self):
        conv1 = SimpleNamespace(causal_padding=(1, 0))
        self.conv = SimpleNamespace(
            in_channels=1,
            out_channels=1,
            block=[None, conv1],
        )
        self.attn = SimpleNamespace(num_heads=1, head_dim=1)


class _FakeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_FakeBlock()]
        self.cfg_batches: list[int] = []
        self.speaker_order: list[list[float]] = []
        self.times: list[float] = []
        self.t_embedder_calls = 0

    def t_embedder(self, time):
        self.t_embedder_calls += 1
        return time[:, None]

    def blocks_forward_chunk(
        self,
        inputs,
        time,
        mask,
        cnn_cache,
        att_cache,
        cnn_out,
        att_out,
    ):
        del mask, cnn_cache, att_cache
        self.cfg_batches.append(inputs.shape[0])
        self.speaker_order.append(inputs[:, 2, 0].tolist())
        self.times.append(float(time[0, 0, 0]))
        marker = inputs[:, 1, 0]
        cnn_out.copy_(marker.reshape(1, -1, 1, 1).expand_as(cnn_out))
        att_out.copy_(marker.reshape(1, -1, 1, 1, 1).expand_as(att_out))
        return inputs[:, 1:2]


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _FakeEstimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.zeros(1, 1, 100), persistent=False)


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.encoder = _FakeEncoder()
        self.encoder_proj = nn.Identity()
        self.decoder = _FakeDecoder()
        self.spk_embed_affine_layer = nn.Identity()

    def input_embedding(self, tokens):
        return tokens.to(self.dummy.dtype).unsqueeze(-1)


class _FakeHiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.calls: list[int] = []

    def forward(self, mel, source):
        del source
        self.calls.append(mel.shape[0])
        speech = mel[:, 0].repeat_interleave(3, dim=1)
        generated_source = speech[:, None]
        return speech, generated_source


class _FakeToken2Wav:
    def __init__(self):
        self.flow = _FakeFlow()
        self.hift = _FakeHiFT()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.hamming_window(4, periodic=False)
        self.prompt_calls = 0

    def _prepare_prompt(self, prompt_wav):
        del prompt_wav
        self.prompt_calls += 1
        return (
            torch.tensor([[5, 6]], dtype=torch.long),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1, 1),
            torch.ones(1, 4, 1),
            torch.tensor([4], dtype=torch.int32),
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("sequential stream fallback must never be called")


def test_remove_weight_norm_for_inference_supports_modern_parametrizations():
    layer = nn.Conv1d(2, 3, kernel_size=3, padding=1).eval()
    layer = torch.nn.utils.parametrizations.weight_norm(layer)
    inputs = torch.randn(1, 2, 5)
    expected = layer(inputs)

    assert _remove_weight_norm_for_inference(layer) == 1
    assert not torch.nn.utils.parametrize.is_parametrized(layer, "weight")
    torch.testing.assert_close(layer(inputs), expected)

    def __call__(self, *args, **kwargs):
        raise AssertionError("sequential __call__ fallback must never be called")


def _config(
    minimum: int = 1,
    *,
    cache_initial_state: bool = False,
    cross_prompt_batching: bool = False,
    runtime_prompt_cache_size: int = 0,
    cache_runtime_initial_state: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config={
                "extra": {
                    "code2wav_min_batch_size": minimum,
                    "code2wav_cache_initial_state": cache_initial_state,
                    "code2wav_cross_prompt_batching": cross_prompt_batching,
                    "code2wav_runtime_prompt_cache_size": runtime_prompt_cache_size,
                    "code2wav_cache_runtime_initial_state": cache_runtime_initial_state,
                    "prompt_cache_id": "shared",
                    "prompt_wav": "/fake/prompt.wav",
                }
            },
        )
    )


def _model(
    *,
    cache_initial_state: bool = False,
    cross_prompt_batching: bool = False,
    runtime_prompt_cache_size: int = 0,
    cache_runtime_initial_state: bool = False,
):
    token2wav = _FakeToken2Wav()
    backend = BatchedToken2Wav(token2wav)
    model = MiniCPMO45Code2Wav(
        vllm_config=_config(
            cache_initial_state=cache_initial_state,
            cross_prompt_batching=cross_prompt_batching,
            runtime_prompt_cache_size=runtime_prompt_cache_size,
            cache_runtime_initial_state=cache_runtime_initial_state,
        )
    )
    model.backend = backend
    return model, token2wav


def test_code2wav_resolves_hf_model_id_for_assets(mocker, tmp_path):
    resolved_root = tmp_path / "snapshot"
    resolved_root.mkdir()
    config = _config()
    config.model_config.model = "openbmb/MiniCPM-o-4_5"
    config.model_config.revision = "test-revision"
    config.model_config.stage_connector_config["extra"].pop("prompt_wav")
    config.load_config = SimpleNamespace(download_dir="/model-cache")
    model = MiniCPMO45Code2Wav(vllm_config=config)
    mock_download = mocker.patch(
        "vllm_omni.model_executor.model_loader.weight_utils.download_weights_from_hf_specific",
        return_value=str(resolved_root),
    )

    assert model._resolve_model_root() == resolved_root
    assert model.model_path == str(resolved_root)
    assert model._default_prompt_wav == str(resolved_root / "assets" / "HT_ref_audio.wav")
    mock_download.assert_called_once_with(
        "openbmb/MiniCPM-o-4_5",
        "/model-cache",
        allow_patterns=[
            "assets/HT_ref_audio.wav",
            "assets/token2wav/*",
        ],
        revision="test-revision",
        require_all=True,
    )


def _info(
    request_id: str,
    chunk_seq: int,
    codes: list[int],
    *,
    last_chunk: bool = False,
    cache_epoch: int = 0,
):
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long)},
        "meta": {
            "request_id": request_id,
            "chunk_seq": chunk_seq,
            "cache_epoch": cache_epoch,
            "last_chunk": last_chunk,
            "prompt_cache_id": "shared",
        },
    }


def _forward(model, infos, placeholder_counts=None, request_ids=None):
    placeholder_counts = placeholder_counts or [1] * len(infos)
    input_ids = torch.zeros(sum(placeholder_counts), dtype=torch.long)
    return model(
        input_ids=input_ids,
        seq_token_counts=placeholder_counts,
        runtime_additional_information=infos,
        request_ids=request_ids,
    )


def test_adapter_runs_true_batch_cfg_and_splits_request_caches():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.prompt_calls == 1
    assert token2wav.flow.decoder.estimator.t_embedder_calls == 1
    assert token2wav.flow.encoder.calls == [2, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [4, 4, 4, 4]
    assert all(order == [1.0, 1.0, 0.0, 0.0] for order in token2wav.flow.decoder.estimator.speaker_order)
    assert token2wav.hift.calls == [2]
    assert len(audios) == 2
    cache0 = states[0].flow_cache["estimator_cnn_cache"]
    cache1 = states[1].flow_cache["estimator_cnn_cache"]
    assert cache0.data_ptr() != cache1.data_ptr()
    assert cache0[0, 0, 0, 0, 0].item() == 10
    assert cache1[0, 0, 0, 0, 0].item() == 20


def test_adapter_batches_distinct_prompt_speakers_after_initialization():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    first = adapter.prepare_prompt("first", "/fake/first.wav")
    second = PromptFeatures(
        speech_tokens=first.speech_tokens,
        projected_speaker_embedding=torch.full_like(first.projected_speaker_embedding, 2.0),
        mels=first.mels,
    )
    states = [adapter.setup_batch(first, 1)[0], adapter.setup_batch(second, 1)[0]]
    token2wav.flow.decoder.estimator.speaker_order.clear()

    adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        [first, second],
        states,
        last_chunk=False,
    )

    assert token2wav.flow.decoder.estimator.speaker_order == [
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("cfg_mode", "expected_scale"),
    [("conditional", 1.0), ("conditional_scale", 1.7)],
)
def test_adapter_conditional_only_cfg_modes_halve_estimator_batch(cfg_mode, expected_scale):
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, cfg_mode=cfg_mode)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)
    audios, _ = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.flow.decoder.estimator.cfg_batches == [1, 1, 1, 1]
    torch.testing.assert_close(audios[0][0], torch.tensor(expected_scale * 10))


def test_adapter_rk4_uses_four_evaluations_per_solver_step():
    token2wav = _FakeToken2Wav()
    token2wav.n_timesteps = 1
    adapter = BatchedToken2Wav(token2wav, cfg_mode="conditional", solver="rk4")
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)
    estimator = token2wav.flow.decoder.estimator
    estimator.cfg_batches.clear()
    estimator.times.clear()

    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert adapter.num_evaluations == 4
    assert estimator.cfg_batches == [1, 1, 1, 1]
    assert estimator.times == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert states[0].flow_cache["estimator_cnn_cache"].shape[0] == 4
    torch.testing.assert_close(audios[0][0], torch.tensor(10.0))


def test_adapter_keeps_half_flow_and_float_hift_dtype_boundaries_explicit():
    token2wav = _FakeToken2Wav()
    token2wav.flow.half()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    assert prompt.projected_speaker_embedding.dtype == torch.float16
    assert prompt.mels.dtype == torch.float16
    assert states[0].hift_cache["mel"].dtype == torch.float32

    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert audios[0].dtype == torch.float32
    assert states[0].hift_cache["mel"].dtype == torch.float32


def test_cached_initial_state_reuses_read_only_template_then_isolates_live_caches():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)

    first_features, first_states = adapter.setup_cached_batch("shared", "/fake/prompt.wav", 2)
    second_features, second_states = adapter.setup_cached_batch("shared", "/fake/prompt.wav", 1)

    assert first_features is second_features
    assert token2wav.prompt_calls == 1
    assert token2wav.flow.encoder.calls == [1]
    assert len(first_states) == 2
    assert len(second_states) == 1
    first_cache = first_states[0].flow_cache["estimator_cnn_cache"]
    sibling_cache = first_states[1].flow_cache["estimator_cnn_cache"]
    second_cache = second_states[0].flow_cache["estimator_cnn_cache"]
    assert first_cache.data_ptr() == sibling_cache.data_ptr() == second_cache.data_ptr()
    template_snapshot = first_cache.clone()

    _, live_states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        first_features,
        first_states,
        last_chunk=False,
    )

    torch.testing.assert_close(first_cache, template_snapshot)
    live_cache0 = live_states[0].flow_cache["estimator_cnn_cache"]
    live_cache1 = live_states[1].flow_cache["estimator_cnn_cache"]
    assert live_cache0.data_ptr() != live_cache1.data_ptr()
    assert live_cache0.data_ptr() != first_cache.data_ptr()
    assert live_cache1.data_ptr() != first_cache.data_ptr()


def test_evict_prompt_removes_cached_initial_state() -> None:
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)

    adapter.setup_cached_batch("shared", "/fake/prompt.wav", 1)
    adapter.evict_prompt("shared", "/fake/prompt.wav")
    adapter.setup_cached_batch("shared", "/fake/prompt.wav", 1)

    assert token2wav.prompt_calls == 2
    assert token2wav.flow.encoder.calls == [1, 1]


def test_prompt_features_are_padded_to_reusable_shape_bucket() -> None:
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, prompt_bucket_frames=4)

    features = adapter.prepare_prompt("voice", "/fake/prompt.wav")

    assert features.speech_tokens.tolist() == [[5, 6, 4218, 4218]]
    assert features.mels.shape == (1, 8, 1)
    assert features.mels[:, 4:].eq(1).all()


def test_code2wav_prewarm_runs_initial_and_steady_shapes_then_synchronizes() -> None:
    model, token2wav = _model()

    class _FakePlatform:
        synchronized = False

        @classmethod
        def synchronize(cls):
            cls.synchronized = True

    model._prewarm_backend(
        {
            "initial_codec_chunk_frames": 4,
            "codec_chunk_frames": 25,
            "codec_left_context_frames": 3,
            "code2wav_prewarm_prompt_buckets": [2],
        },
        _FakePlatform,
    )

    assert token2wav.prompt_calls == 1
    # One prompt shape, one cached default state, then initial + steady live
    # codec chunks. All are single-request warmups.
    assert token2wav.flow.encoder.calls == [1, 1, 1, 1]
    assert token2wav.hift.calls == [1, 1]
    assert _FakePlatform.synchronized is True


def test_code2wav_prewarm_covers_preloaded_runtime_prompt_shapes() -> None:
    model, token2wav = _model(
        runtime_prompt_cache_size=1,
        cache_runtime_initial_state=True,
    )
    model._runtime_prompts["runtime-key"] = _RuntimePrompt(
        cache_id="runtime-ref",
        path="/fake/runtime.wav",
        owners=set(),
    )

    class _FakePlatform:
        @staticmethod
        def synchronize():
            pass

    model._prewarm_backend(
        {
            "initial_codec_chunk_frames": 13,
            "codec_chunk_frames": 50,
            "codec_left_context_frames": 3,
        },
        _FakePlatform,
    )

    # Default and runtime prompts each build one initial state and execute the
    # two live shapes. The runtime initial state remains cached for request 1.
    assert token2wav.prompt_calls == 2
    assert token2wav.flow.encoder.calls == [1, 1, 1, 1, 1, 1]
    assert token2wav.hift.calls == [1, 1, 1, 1]
    assert ("runtime-ref", "/fake/runtime.wav") in model.backend._initial_states


def test_code2wav_runner_prewarm_executes_full_live_packet_path(
    monkeypatch,
    tmp_path,
) -> None:
    from vllm_omni.platforms import current_omni_platform

    model, token2wav = _model(
        runtime_prompt_cache_size=1,
        cache_runtime_initial_state=True,
    )
    extra = model.vllm_config.model_config.stage_connector_config["extra"]
    extra.update(
        {
            "code2wav_runner_prewarm": True,
            "initial_codec_chunk_frames": 4,
            "codec_chunk_frames": 25,
            "codec_left_context_frames": 3,
        }
    )
    waveform = torch.linspace(-0.25, 0.25, 1600)
    _, entry = model._materialize_runtime_prompt(waveform, 16000)
    monkeypatch.setattr(current_omni_platform, "synchronize", lambda: None)

    model.runner_prewarm()

    assert token2wav.prompt_calls == 1
    assert token2wav.flow.encoder.calls == [1, 1, 1]
    assert token2wav.hift.calls == [1, 1]
    assert model._states == {}


def test_steady_npugraph_dispatch_clones_outputs_and_shape_mismatch_falls_back(
    monkeypatch,
) -> None:
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    state = adapter.setup_batch(features, 1)[0]
    static_state = adapter._clone_state(state)
    output_state = adapter._clone_state(state)
    static_tokens = torch.zeros((1, 2), dtype=torch.long)
    graph_audio = torch.zeros(4)

    class _FakeGraph:
        calls = 0

        @classmethod
        def replay(cls):
            cls.calls += 1
            graph_audio.fill_(
                float(
                    static_tokens.sum()
                    + static_features.speech_tokens.sum()
                    + static_features.projected_speaker_embedding.sum()
                    + static_features.mels.sum()
                )
            )
            for cache_name in ("flow_cache", "hift_cache"):
                inputs = getattr(static_state, cache_name)
                outputs = getattr(output_state, cache_name)
                for name in outputs:
                    outputs[name].copy_(inputs[name] + 1)

    graph_key = (
        (tuple(static_tokens.shape), str(static_tokens.dtype), static_tokens.device.type),
        int(features.mels.shape[1]),
        state_shape_signature(state),
    )
    static_features = PromptFeatures(
        speech_tokens=features.speech_tokens.clone(),
        projected_speaker_embedding=features.projected_speaker_embedding.clone(),
        mels=features.mels.clone(),
    )
    adapter._steady_npugraphs[graph_key] = _SteadyNPUGraph(
        graph=_FakeGraph(),
        tokens=static_tokens,
        features=static_features,
        input_state=static_state,
        audio=graph_audio,
        output_state=output_state,
        state_signature=state_shape_signature(state),
        prompt_mel_length=int(features.mels.shape[1]),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)

    live_features = PromptFeatures(
        speech_tokens=features.speech_tokens + 10,
        projected_speaker_embedding=features.projected_speaker_embedding + 20,
        mels=features.mels + 30,
    )
    audios, next_states = adapter.decode_batch(
        torch.tensor([[3, 4]]),
        live_features,
        [state],
        last_chunk=False,
    )

    assert _FakeGraph.calls == 1
    assert adapter.steady_npugraph_replays == 1
    expected_audio = float(
        torch.tensor([[3, 4]]).sum()
        + live_features.speech_tokens.sum()
        + live_features.projected_speaker_embedding.sum()
        + live_features.mels.sum()
    )
    assert audios[0].eq(expected_audio).all()
    torch.testing.assert_close(static_features.speech_tokens, live_features.speech_tokens)
    torch.testing.assert_close(
        static_features.projected_speaker_embedding,
        live_features.projected_speaker_embedding,
    )
    torch.testing.assert_close(static_features.mels, live_features.mels)
    for cache_name in ("flow_cache", "hift_cache"):
        original = getattr(state, cache_name)
        result = getattr(next_states[0], cache_name)
        for name in result:
            torch.testing.assert_close(result[name], original[name] + 1)
    graph_audio.zero_()
    for cache in (output_state.flow_cache, output_state.hift_cache):
        for tensor in cache.values():
            tensor.zero_()
    assert audios[0].eq(expected_audio).all()
    assert any(tensor.count_nonzero() for tensor in next_states[0].flow_cache.values())

    encoder_calls = len(token2wav.flow.encoder.calls)
    adapter.decode_batch(
        torch.tensor([[3, 4, 5]]),
        features,
        [state],
        last_chunk=False,
    )
    assert _FakeGraph.calls == 1
    assert adapter.steady_npugraph_replays == 1
    assert len(token2wav.flow.encoder.calls) == encoder_calls + 1


def test_model_reuses_cached_initial_state_for_default_voice() -> None:
    model, token2wav = _model(cache_initial_state=True)

    _forward(model, [_info("a", 0, [10, 11], last_chunk=True)])
    _forward(model, [_info("b", 0, [12, 13], last_chunk=True)])

    assert token2wav.prompt_calls == 1
    # One prompt setup plus one live decode per request.
    assert token2wav.flow.encoder.calls == [1, 1, 1]


def test_fade_in_out_limits_overlap_to_available_previous_audio():
    speech = torch.arange(6, dtype=torch.float32).reshape(1, -1)
    previous = torch.full((1, 3), 2.0)
    window = torch.hamming_window(8, periodic=False)

    actual = BatchedToken2Wav._fade_in_out(speech, previous, window)

    expected = speech.clone()
    expected[..., :3] = speech[..., :3] * window[:3] + previous * window[-3:]
    torch.testing.assert_close(actual, expected)


def test_estimator_cache_stack_split_round_trip_preserves_cfg_rows():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    stacked = adapter._stack_flow_cache(states)
    assert stacked["estimator_cnn_cache"].shape[2] == 4
    assert stacked["estimator_att_cache"].shape[2] == 4
    restored = adapter._split_flow_cache(stacked, 2)
    for original, round_tripped in zip(states, restored, strict=True):
        torch.testing.assert_close(
            round_tripped["estimator_cnn_cache"],
            original.flow_cache["estimator_cnn_cache"],
        )
        torch.testing.assert_close(
            round_tripped["estimator_att_cache"],
            original.flow_cache["estimator_att_cache"],
        )


def test_single_request_cache_stack_split_reuses_existing_storage():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    state = adapter.setup_batch(prompt, 1)[0]

    stacked = adapter._stack_flow_cache([state])
    restored = adapter._split_flow_cache(stacked, 1)[0]

    for name, original in state.flow_cache.items():
        assert stacked[name].data_ptr() == original.data_ptr()
        assert restored[name].data_ptr() == original.data_ptr()


def test_model_preserves_output_slots_and_prefers_runtime_codes():
    model, token2wav = _model()
    output = _forward(
        model,
        [_info("a", 0, [10, 11]), _info("b", 0, [20, 21])],
        placeholder_counts=[3, 1],
    )

    audios = output.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert len(output.multimodal_outputs["sr"]) == 2
    assert all(sr.item() == 24000 for sr in output.multimodal_outputs["sr"])
    assert all(audio.dtype == torch.float32 for audio in audios)
    # Fake CFM uses two Euler steps whose deltas sum to one. Its conditional
    # row is mu and its unconditional row is zero, so CFG produces 1.7 * mu.
    torch.testing.assert_close(audios[0][0], torch.tensor(1.7 * 10))
    torch.testing.assert_close(audios[1][0], torch.tensor(1.7 * 20))
    assert token2wav.flow.encoder.calls[-1] == 2


def test_code2wav_projects_duplex_metadata_to_final_audio_output():
    model, token2wav = _model()
    segment = _info("duplex", 0, [10, 11])
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    segment["meta"].update(
        {
            "duplex_epoch": 3,
            "duplex_turn_id": 7,
            "llm_output_text_utf8": segment_text_utf8,
            "tts_is_last_chunk": True,
            "speak_tail": False,
            "turn_end": False,
        }
    )

    segment_output = _forward(model, [segment])

    assert segment_output.multimodal_outputs["meta.turn_end"][0].item() is False
    assert segment_output.multimodal_outputs["meta.speak_tail"][0].item() is False
    # A Talker unit boundary only drains pending codec tokens. The official
    # streaming path keeps Token2wav open until the assistant turn ends.
    assert token2wav.flow.encoder.last_chunk_calls[-1] is False
    assert "duplex" in model._states

    final = _info("duplex", 1, [12, 13], last_chunk=True)
    final["meta"].update(segment["meta"])
    final["meta"]["chunk_seq"] = 1
    final["meta"]["last_chunk"] = True
    final["meta"]["turn_end"] = True
    final["meta"]["speak_tail"] = True
    output = _forward(model, [final])

    payload = output.multimodal_outputs
    assert "meta" not in payload
    assert payload["meta.duplex_epoch"][0].item() == 3
    assert payload["meta.duplex_turn_id"][0].item() == 7
    torch.testing.assert_close(
        payload["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert payload["meta.tts_is_last_chunk"][0].item() is True
    assert payload["meta.speak_tail"][0].item() is True
    assert payload["meta.turn_end"][0].item() is True
    assert token2wav.flow.encoder.last_chunk_calls[-1] is True
    assert "duplex" not in model._states


def test_initial_empty_segment_marker_initializes_stream_without_audio():
    model, token2wav = _model()
    boundary = _info("duplex", 0, [])
    boundary["meta"].update(
        {
            "code_flat_numel": 0,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    output = _forward(model, [boundary])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert "duplex" in model._states
    assert token2wav.hift.calls == []

    resumed = _info(
        "duplex",
        1,
        [4218, 4218, 4218, 10, 11, 12, 13, 14],
    )
    output = _forward(model, [resumed])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert "duplex" in model._states


def test_shared_runtime_prompt_recreates_missing_file_before_second_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    first = _info("voice-a", 0, [10, 11])
    first["codes"]["ref"] = reference
    first["meta"]["ref_audio_sr"] = 16000
    first["meta"].pop("prompt_cache_id")
    _forward(model, [first], request_ids=["internal-a"])

    prompt_key = model._request_prompt_keys["voice-a"]
    prompt_path = Path(model._runtime_prompts[prompt_key].path)
    prompt_path.unlink()

    second = _info("voice-b", 0, [12, 13])
    second["codes"]["ref"] = reference
    second["meta"]["ref_audio_sr"] = 16000
    second["meta"].pop("prompt_cache_id")
    _forward(model, [second], request_ids=["internal-b"])

    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"voice-a", "voice-b"}

    _forward(
        model,
        [_info("voice-a", 1, [14, 15], last_chunk=True)],
        request_ids=["internal-a"],
    )
    model.on_requests_finished(["internal-a"])
    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"voice-b"}

    _forward(
        model,
        [_info("voice-b", 1, [16, 17], last_chunk=True)],
        request_ids=["internal-b"],
    )
    model.on_requests_finished(["internal-b"])
    assert not prompt_path.exists()
    assert prompt_key not in model._runtime_prompts


def test_bounded_runtime_prompt_cache_reuses_features_and_evicts_lru(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, token2wav = _model(runtime_prompt_cache_size=1)

    def runtime_ref_info(request_id: str, reference: torch.Tensor):
        info = _info(request_id, 0, [10, 11], last_chunk=True)
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    first_reference = torch.tensor([0.0, 0.25, -0.25, 0.0])
    _forward(model, [runtime_ref_info("voice-a", first_reference)])
    first_key = next(iter(model._runtime_prompts))
    first_entry = model._runtime_prompts[first_key]
    first_path = Path(first_entry.path)

    assert token2wav.prompt_calls == 1
    assert first_path.is_file()
    assert first_entry.owners == set()

    _forward(model, [runtime_ref_info("voice-b", first_reference.clone())])

    assert token2wav.prompt_calls == 1
    assert first_key in model._runtime_prompts
    assert first_path.is_file()

    second_reference = torch.tensor([0.0, 0.5, -0.5, 0.0])
    _forward(model, [runtime_ref_info("voice-c", second_reference)])

    assert token2wav.prompt_calls == 2
    assert len(model._runtime_prompts) == 1
    assert first_key not in model._runtime_prompts
    assert not first_path.exists()


def test_runtime_prompt_cache_reuses_immutable_initial_state(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, token2wav = _model(
        runtime_prompt_cache_size=1,
        cache_runtime_initial_state=True,
    )
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def runtime_ref_info(request_id: str):
        info = _info(request_id, 0, [10, 11], last_chunk=True)
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    _forward(model, [runtime_ref_info("voice-a")])
    first_encoder_calls = len(token2wav.flow.encoder.calls)
    _forward(model, [runtime_ref_info("voice-b")])

    assert token2wav.prompt_calls == 1
    # First request: prompt setup + live decode. Second request reuses the
    # immutable initial flow state and executes only its live decode.
    assert first_encoder_calls == 2
    assert len(token2wav.flow.encoder.calls) == 3


def test_runtime_initial_state_cache_requires_bounded_prompt_cache():
    with pytest.raises(ValueError, match="runtime_prompt_cache_size > 0"):
        MiniCPMO45Code2Wav(
            vllm_config=_config(cache_runtime_initial_state=True)
        )


def test_runtime_prompt_preload_matches_native_trim_and_first_request(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    source_path = tmp_path / "system_ref.wav"
    source = torch.linspace(-0.5, 0.5, 3210)
    sf.write(source_path, source.numpy(), 16000, subtype="PCM_16")
    model, token2wav = _model(
        runtime_prompt_cache_size=1,
        cache_runtime_initial_state=True,
    )

    model._preload_runtime_prompts(
        [source_path],
        target_sample_rate=16000,
        frame_samples=1600,
    )

    assert token2wav.prompt_calls == 1
    assert len(token2wav.flow.encoder.calls) == 1
    assert len(model._runtime_prompts) == 1
    waveform, sample_rate = model._load_reference_waveform(
        source_path,
        target_sample_rate=16000,
        frame_samples=1600,
    )
    assert waveform.numel() == 3200

    info = _info("voice-a", 0, [10, 11], last_chunk=True)
    info["codes"]["ref"] = waveform
    info["meta"]["ref_audio_sr"] = sample_rate
    info["meta"].pop("prompt_cache_id")
    _forward(model, [info])

    assert token2wav.prompt_calls == 1
    # Preload did prompt setup; the first live request only decodes its chunk.
    assert len(token2wav.flow.encoder.calls) == 2


def test_runtime_prompt_write_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.sf.write",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated write failure"):
        model._materialize_runtime_prompt(reference, 16000)

    assert len(model._runtime_prompts) == 1
    entry = next(iter(model._runtime_prompts.values()))
    assert not Path(entry.path).exists()
    assert list(Path(entry.path).parent.iterdir()) == []


def test_runtime_prompt_files_are_isolated_between_model_instances(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    first_model, _ = _model()
    second_model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def runtime_ref_info(request_id: str):
        info = _info(request_id, 0, [10, 11])
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    _forward(first_model, [runtime_ref_info("voice-a")], request_ids=["internal-a"])
    _forward(second_model, [runtime_ref_info("voice-b")], request_ids=["internal-b"])

    first_key = first_model._request_prompt_keys["voice-a"]
    second_key = second_model._request_prompt_keys["voice-b"]
    first_path = Path(first_model._runtime_prompts[first_key].path)
    second_path = Path(second_model._runtime_prompts[second_key].path)
    assert first_key == second_key
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()

    _forward(
        first_model,
        [_info("voice-a", 1, [12, 13], last_chunk=True)],
        request_ids=["internal-a"],
    )
    first_model.on_requests_finished(["internal-a"])
    assert not first_path.exists()
    assert second_path.is_file()

    _forward(
        second_model,
        [_info("voice-b", 1, [12, 13], last_chunk=True)],
        request_ids=["internal-b"],
    )
    second_model.on_requests_finished(["internal-b"])
    assert not second_path.exists()


def test_mixed_final_exact_buckets_keep_order_and_release_only_final_states():
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    output = _forward(
        model,
        [
            _info("a", 1, [11, 12]),
            _info("c", 1, [31, 32, 33], last_chunk=True),
            _info("b", 1, [21, 22]),
            _info("d", 1, [41, 42, 43], last_chunk=True),
        ],
    )

    audios = output.multimodal_outputs["model_outputs"]
    window = torch.hamming_window(4, periodic=False)
    overlap_scale = 1.7 * (window[0] + window[2])
    expected = torch.tensor([1, 3, 2, 4], dtype=torch.float32) * overlap_scale
    actual = torch.stack([audio[0] for audio in audios])
    torch.testing.assert_close(actual, expected)
    assert set(model._states) == {"a", "b"}


def test_empty_final_sentinel_emits_empty_and_releases_state_without_compute():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    hift_calls = list(token2wav.hift.calls)
    output = _forward(
        model,
        [
            _info("a", 1, [], last_chunk=True),
            _info("b", 1, [], last_chunk=True),
        ],
    )

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}
    assert token2wav.hift.calls == hift_calls


def test_empty_final_ignores_generation_scheduler_placeholder_token():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    infos = [_info("a", 1, [], last_chunk=True), _info("b", 1, [], last_chunk=True)]
    for info in infos:
        info.pop("codes")
        info["meta"]["code_flat_numel"] = 0

    output = _forward(model, infos, placeholder_counts=[1, 1])

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}


@pytest.mark.parametrize(
    "info",
    [
        # The runner injects the engine request id on every step (GPU
        # _preprocess, NPU _gather_runtime_additional_information)...
        {"request_id": "a", "meta": {"request_id": "a"}},
        # ...but a pre-warm step can also reach the model with nothing at all.
        {},
    ],
)
def test_prewarm_placeholder_step_emits_silence_without_touching_state(info):
    # async-chunk pre-warm submits Stage 2 with a reserved placeholder prompt.
    # If it gets scheduled before the first codec window lands, those reserved
    # tokens must neither be vocoded nor held to the codec payload contract.
    model, token2wav = _model()

    output = _forward(model, [info], request_ids=["a"])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._states == {}
    assert token2wav.hift.calls == []


def test_metadata_only_payload_still_decodes_codec_from_prompt_tokens():
    # The connector strips 1-D codec tensors out of additional_information and
    # leaves them in the prompt tokens, so a real chunk reaches the model as
    # producer metadata plus input ids. It must still be vocoded.
    model, _ = _model()
    info = {
        "request_id": "a",
        "meta": {
            "request_id": "a",
            "chunk_seq": 0,
            "code_flat_numel": 2,
            "prompt_cache_id": "shared",
        },
    }

    output = _forward(model, [info], placeholder_counts=[2])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert set(model._states) == {"a"}


def test_non_final_chunk_shorter_than_lookahead_window_is_rejected():
    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    with pytest.raises(RuntimeError, match="chunk_below_lookahead_window"):
        adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=False)

    # The final chunk is zero-padded by the encoder, so it stays decodable.
    audios, _ = adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=True)
    assert len(audios) == 1


def test_forward_builds_backend_when_weight_loading_was_skipped(monkeypatch):
    # load_format=dummy never calls load_weights(), so Stage 2 would otherwise
    # reach its first request with no Token2wav assets at all.
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    token2wav = _FakeToken2Wav()
    builds = 0

    def build_backend():
        nonlocal builds
        builds += 1
        model.backend = BatchedToken2Wav(token2wav)

    monkeypatch.setattr(model, "_build_backend", build_backend)

    output = _forward(model, [_info("a", 0, [10, 11])])
    _forward(model, [_info("a", 1, [12, 13])])

    assert builds == 1
    assert output.multimodal_outputs["model_outputs"][0].numel() > 0


@pytest.mark.parametrize(
    ("info", "reason"),
    [
        (_info("a", 0, [1, 2], cache_epoch=-1), "negative_stream_position"),
        (_info("a", 0, [1, 2]), "stale_or_reordered_chunk"),
        (_info("a", 2, [1, 2]), "stale_or_reordered_chunk"),
    ],
)
def test_stale_epoch_and_reordered_chunks_are_rejected(info, reason):
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])

    with pytest.raises(RuntimeError, match=reason):
        _forward(model, [info, _info("b", 1, [3, 4])])


def test_singleton_and_mixed_shape_buckets_use_same_batched_backend_without_fallback():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    output = _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8, 9])])

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    # Exact-shape buckets execute independently but both use the same vectorized
    # adapter; there is no Token2wav.stream/__call__ fallback.
    assert token2wav.hift.calls[-2:] == [1, 1]


def test_initialized_requests_with_distinct_prompts_share_live_decode_batch():
    model, token2wav = _model(cross_prompt_batching=True)
    first_a = _info("a", 0, [1, 2])
    first_b = _info("b", 0, [3, 4])
    first_a["meta"].update(prompt_cache_id="voice-a", prompt_wav="/fake/voice-a.wav")
    first_b["meta"].update(prompt_cache_id="voice-b", prompt_wav="/fake/voice-b.wav")

    _forward(model, [first_a, first_b])
    assert token2wav.hift.calls[-2:] == [1, 1]
    token2wav.hift.calls.clear()

    _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8])])

    assert token2wav.hift.calls == [2]


def test_cross_prompt_batching_is_opt_in():
    model, token2wav = _model()
    first_a = _info("a", 0, [1, 2])
    first_b = _info("b", 0, [3, 4])
    first_a["meta"].update(prompt_cache_id="voice-a", prompt_wav="/fake/voice-a.wav")
    first_b["meta"].update(prompt_cache_id="voice-b", prompt_wav="/fake/voice-b.wav")
    _forward(model, [first_a, first_b])
    token2wav.hift.calls.clear()

    _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8])])

    assert token2wav.hift.calls == [1, 1]


def test_backend_failure_does_not_commit_any_request_state(monkeypatch):
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    before = dict(model._states)
    original = model.backend.decode_batch
    call_count = 0

    def fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "decode_batch", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _forward(
            model,
            [
                _info("a", 1, [5, 6]),
                _info("b", 1, [7, 8]),
                _info("c", 1, [9, 10, 11]),
                _info("d", 1, [12, 13, 14]),
            ],
        )
    assert call_count == 2
    assert model._states == before


def test_engine_step_completion_keeps_live_stream_state_for_next_chunk():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    model.on_requests_finished(["a"])
    assert set(model._states) == {"a", "b"}

    _forward(model, [_info("a", 1, [5, 6])])
    assert model._states["a"].chunk_seq == 1

    profile = model(
        input_ids=torch.zeros(5, dtype=torch.long),
        seq_token_counts=[2, 3],
    )
    assert [audio.numel() for audio in profile.multimodal_outputs["model_outputs"]] == [0, 0]
    assert set(model._states) == {"a", "b"}


def test_stream_state_uses_stable_payload_request_ids_across_runner_ids():
    model, _ = _model()
    _forward(
        model,
        [_info("external-a", 0, [1, 2]), _info("external-b", 0, [3, 4])],
        request_ids=["internal-a", "internal-b"],
    )

    model.on_requests_finished(["internal-a"])

    assert set(model._states) == {"external-a", "external-b"}

    _forward(
        model,
        [_info("external-a", 1, [5, 6], last_chunk=True)],
        request_ids=["internal-a"],
    )
    model.on_requests_finished(["internal-a"])

    assert set(model._states) == {"external-b"}


def test_stream_state_uses_top_level_payload_id_when_meta_omits_id():
    model, _ = _model()
    first = _info("external-a", 0, [1, 2])
    first["meta"].pop("request_id")
    first["request_id"] = "external-a"
    _forward(model, [first], request_ids=["internal-chunk-0"])

    assert set(model._states) == {"external-a"}

    second = _info("external-a", 1, [5, 6], last_chunk=True)
    second["meta"].pop("request_id")
    second["request_id"] = "external-a"
    _forward(model, [second], request_ids=["internal-chunk-1"])

    assert model._states == {}


def test_replayed_terminal_chunk_is_idempotently_ignored():
    model, _ = _model()
    _forward(model, [_info("stream-a", 0, [1, 2])])
    terminal = _info("stream-a", 1, [3, 4], last_chunk=True)
    terminal["meta"].update(
        tts_is_last_chunk=True,
        speak_tail=True,
        turn_end=True,
    )
    first = _forward(model, [terminal])

    assert first.multimodal_outputs["model_outputs"][0].numel() > 0
    assert model._states == {}

    replay = _forward(model, [terminal])

    torch.testing.assert_close(
        replay.multimodal_outputs["model_outputs"][0],
        first.multimodal_outputs["model_outputs"][0],
    )
    assert replay.multimodal_outputs["meta.turn_end"][0].item() is True
    assert replay.multimodal_outputs["meta.speak_tail"][0].item() is True
    assert replay.multimodal_outputs["meta.tts_is_last_chunk"][0].item() is True
    assert model._states == {}


def test_reference_voice_and_duplex_metadata_follow_request_lifecycle():
    model, _ = _model()
    first = _info("voice-a", 0, [1, 2])
    first["codes"]["ref"] = torch.linspace(-0.1, 0.1, 160)
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    first["meta"].update(
        ref_audio_sr=16000,
        llm_output_text_utf8=segment_text_utf8,
        duplex_turn_id=7,
        duplex_epoch=3,
    )
    first["meta"].pop("prompt_cache_id")

    output = _forward(model, [first])
    prompt_key = model._request_prompt_keys["voice-a"]
    prompt = model._runtime_prompts[prompt_key]
    prompt_cache_id, prompt_wav = prompt.cache_id, prompt.path
    assert prompt_cache_id.startswith("runtime-ref-")
    assert Path(prompt_wav).is_file()
    torch.testing.assert_close(
        output.multimodal_outputs["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert output.multimodal_outputs["meta.duplex_turn_id"][0].item() == 7
    assert output.multimodal_outputs["meta.duplex_epoch"][0].item() == 3

    final = _info("voice-a", 1, [3, 4], last_chunk=True)
    final["meta"].pop("prompt_cache_id")
    final["meta"]["tts_is_last_chunk"] = True
    output = _forward(model, [final])

    assert output.multimodal_outputs["meta.tts_is_last_chunk"][0].item() is True
    # Explicit last_chunk owns the persistent stream and runtime prompt
    # lifetime; no runner-ID cleanup callback is required.
    assert "voice-a" not in model._request_prompt_keys
    assert prompt_key not in model._runtime_prompts
    assert not Path(prompt_wav).exists()
    assert (prompt_cache_id, prompt_wav) not in model.backend._prompt_features
