# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _apply_repetition_penalty,
    _apply_repetition_penalty_scatter,
    _apply_top_k_top_p,
    _compact_top_k_top_p_candidates,
    _exact_top_p_then_top_k_candidates,
    _fixed_window_repetition_penalty,
    _max_audio_tokens,
    _restore_weight_norm_weight,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.ones(2, 4)


class _FakeTTSBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
    ) -> torch.Tensor:
        return self.linear(inputs_embeds)


def test_wrapper_always_delegates_talker_to_native_ar_path() -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.talker = _FakeNativeTalker()

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.arange(2),
        model_intermediate_buffer=[{"request_id": "req"}],
    )

    assert output.shape == (2, 4)
    assert model.talker.forward_kwargs["model_intermediate_buffer"][0]["request_id"] == "req"


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._batch_stop_logits = None
    talker._talker_binary_argmax_uses = 0
    talker._request_generators = {}
    talker._request_audio_states = {}
    talker._deferred_cleanup_ids = set()
    talker._codec_min_tokens = 50
    talker._codec_seed = 42
    talker._talker_emit_codec_chunks = False
    talker._talker_initial_codec_chunk_frames = 4
    talker._talker_codec_chunk_frames = 25
    talker._talker_compact_sampling = False
    talker._talker_npu_fused_filter = False
    talker._talker_npu_sampling_params = None
    talker._talker_scatter_repetition = False
    talker._talker_graph_head = False
    talker._talker_graph_sampler = False
    talker._graph_head_valid = False
    talker._graph_sampler_valid = False
    talker.register_buffer(
        "_binary_control_logit_templates",
        torch.tensor(
            [[0.0, float("-inf")], [float("-inf"), 0.0]],
            dtype=torch.float32,
        ),
        persistent=False,
    )
    return talker


def _routed(output, index: int):
    return to_payload_element(
        output.multimodal_outputs,
        index,
        index,
        index + 1,
        seq_len=2,
        scheduled_seq_len=2,
    )


@pytest.mark.parametrize(
    ("condition_tokens", "expected"),
    [(3, 64), (100, 1000), (1000, 2048)],
)
def test_audio_token_limit_scales_with_condition_length(
    condition_tokens: int,
    expected: int,
) -> None:
    assert _max_audio_tokens(condition_tokens) == expected


def test_weight_norm_restore_matches_checkpoint_parametrization_in_bfloat16() -> None:
    generator = torch.Generator().manual_seed(42)
    weight_v = torch.randn(8, 16, generator=generator, dtype=torch.bfloat16)
    weight_g = torch.rand(8, 1, generator=generator, dtype=torch.bfloat16)
    linear = nn.utils.parametrizations.weight_norm(
        nn.Linear(16, 8, bias=False, dtype=torch.bfloat16),
        dim=0,
    )
    with torch.no_grad():
        linear.parametrizations.weight.original0.copy_(weight_g)
        linear.parametrizations.weight.original1.copy_(weight_v)

    restored = _restore_weight_norm_weight(weight_g, weight_v)

    assert torch.equal(restored, linear.weight)


def test_repetition_penalty_matches_dense_reference() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(2, 6562, generator=generator)
    history = torch.tensor([3, 9, 3, 1024, 9, 9, 6550, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3])
    recent = history[-16:]
    frequencies = torch.bincount(recent, minlength=logits.shape[-1]).to(logits.dtype)
    alpha = torch.pow(torch.tensor(1.05, dtype=logits.dtype), frequencies)
    expected = torch.where(logits < 0, logits * alpha, logits / alpha)

    actual = _apply_repetition_penalty(
        logits,
        history,
        penalty=1.05,
        window_size=16,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scatter_repetition_penalty_is_bit_exact(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 6562, generator=generator, dtype=dtype)
    history = torch.tensor([3, 9, 3, 1024, 9, 9, 6550, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3])

    expected = _apply_repetition_penalty(
        logits,
        history,
        penalty=1.05,
        window_size=16,
    )
    actual = _apply_repetition_penalty_scatter(
        logits,
        history,
        penalty=1.05,
        window_size=16,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("concentrated", [False, True])
def test_top_k_top_p_matches_full_sort_reference(concentrated: bool) -> None:
    generator = torch.Generator().manual_seed(123)
    logits = torch.randn(3, 6562, generator=generator)
    if concentrated:
        logits[:, :8] += 12.0
    sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative_probs <= 0.15
    remove[..., -3:] = False
    remove = remove.scatter(-1, sorted_indices, remove)
    expected = logits.masked_fill(remove, float("-inf"))
    threshold = torch.topk(expected, 25, dim=-1).values[..., -1, None]
    expected.masked_fill_(expected < threshold, float("-inf"))

    actual = _apply_top_k_top_p(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )

    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    torch.testing.assert_close(actual[torch.isfinite(actual)], expected[torch.isfinite(expected)])


@pytest.mark.parametrize("concentrated", [False, True])
def test_compact_top_k_candidates_preserve_dense_distribution(concentrated: bool) -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(2, 127, generator=generator)
    if concentrated:
        logits[:, :8] += 12.0
    dense = _apply_top_k_top_p(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )
    candidate_logits, candidate_ids = _compact_top_k_top_p_candidates(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )

    dense_probabilities = torch.softmax(dense, dim=-1)
    compact_probabilities = torch.softmax(candidate_logits, dim=-1)
    torch.testing.assert_close(
        dense_probabilities.gather(-1, candidate_ids),
        compact_probabilities,
    )
    assert torch.equal(
        torch.isfinite(dense).gather(-1, candidate_ids),
        torch.isfinite(candidate_logits),
    )


@pytest.mark.parametrize("concentrated", [False, True])
def test_exact_top_p_then_top_k_reconstructs_dense_distribution(concentrated: bool) -> None:
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(3, 6562, generator=generator)
    if concentrated:
        logits[:, :8] += 12.0
    dense = _apply_top_k_top_p(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )

    candidate_logits, candidate_ids = _exact_top_p_then_top_k_candidates(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )

    assert torch.equal(
        torch.isfinite(dense).gather(-1, candidate_ids),
        torch.isfinite(candidate_logits),
    )
    torch.testing.assert_close(
        torch.softmax(dense, dim=-1).gather(-1, candidate_ids),
        torch.softmax(candidate_logits, dim=-1),
    )


def test_fixed_window_repetition_penalty_matches_bincount() -> None:
    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(2, 127, generator=generator)
    histories = torch.tensor(
        [
            [-1, -1, -1, 3, 9, 3, 4, 4],
            [7, 7, 7, 7, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )
    penalty_lut = torch.pow(
        torch.tensor(1.05),
        torch.arange(histories.shape[1], dtype=torch.float32).add(1),
    )
    penalty_lut = torch.cat((torch.ones(1), penalty_lut))

    expected = torch.cat(
        [
            _apply_repetition_penalty(
                logits[row : row + 1],
                histories[row][histories[row] >= 0],
                penalty=1.05,
                window_size=histories.shape[1],
            )
            for row in range(logits.shape[0])
        ],
        dim=0,
    )
    actual = _fixed_window_repetition_penalty(
        logits,
        histories,
        penalty=1.05,
        penalty_lut=penalty_lut,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_talker_emits_request_aligned_codec_deltas_after_compaction(mocker) -> None:
    talker = _make_talker()
    seen: list[tuple[str, list[float], list[int]]] = []

    def sample(hidden, history, request_id, step):
        assert step == 0
        seen.append((request_id, hidden.reshape(-1).tolist(), history.tolist()))
        return torch.tensor(2 if request_id == "req-a" else 3)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.tensor([1])}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
    )

    assert seen == [
        ("req-a", [2.0, 0.0], [1]),
        ("req-b", [3.0, 0.0], []),
    ]
    assert infos[0]["audio_codes"]["accumulated"].tolist() == [1, 2]
    assert infos[1]["audio_codes"]["accumulated"].tolist() == [3]
    assert set(output.multimodal_outputs) == {"codes", "meta"}
    assert "model_outputs" not in output.multimodal_outputs
    assert "sr" not in output.multimodal_outputs
    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[3]]
    assert _routed(output, 0)["meta"]["finished"].item() is False
    assert set(output.multimodal_outputs["meta"]) == {"finished"}
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [0, 0]


def test_graph_head_projects_only_runner_selected_rows() -> None:
    talker = _make_talker()
    talker._talker_graph_head = True
    talker.tts_model = mock_model = _FakeTTSBackbone()
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    talker.register_buffer("_graph_head_logits", torch.empty(2, 8), persistent=False)
    inputs = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    hidden, projected = talker(
        inputs_embeds=inputs,
        omni_logits_indices=torch.tensor([0, 2]),
    )

    expected_hidden = mock_model(inputs_embeds=inputs)
    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(
        projected,
        talker.head_code[0](expected_hidden[[0, 2]]),
    )


def test_graph_head_full_decode_projects_static_row_prefix() -> None:
    talker = _make_talker()
    talker._talker_graph_head = True
    talker.tts_model = mock_model = _FakeTTSBackbone()
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    hidden, projected = talker(
        inputs_embeds=inputs,
        omni_num_sample_rows=2,
    )

    expected_hidden = mock_model(inputs_embeds=inputs)
    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(projected, talker.head_code[0](expected_hidden[:2]))


def test_graph_sampler_builds_exact_compact_candidates_inside_forward() -> None:
    talker = _make_talker()
    talker._talker_graph_head = True
    talker._talker_graph_sampler = True
    talker._codec_temperature = 0.8
    talker._codec_repetition_penalty = 1.05
    talker._codec_top_k = 3
    talker._codec_top_p = 0.85
    talker.tts_model = _FakeTTSBackbone()
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    talker.register_buffer("_graph_head_logits", torch.empty(1, 8), persistent=False)
    talker.register_buffer("_graph_sampler_logits", torch.empty(1, 3), persistent=False)
    talker.register_buffer(
        "_graph_sampler_ids",
        torch.empty(1, 3, dtype=torch.long),
        persistent=False,
    )
    talker.register_buffer(
        "_codec_repetition_penalty_lut",
        torch.pow(torch.tensor(1.05), torch.arange(17, dtype=torch.float32)),
        persistent=False,
    )
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    history = torch.tensor([[-1] * 14 + [2, 2]], dtype=torch.long)

    hidden, candidate_logits, candidate_ids = talker(
        inputs_embeds=inputs,
        omni_logits_indices=torch.tensor([1]),
        omni_codec_history=history,
        omni_codec_eos_allowed=torch.tensor([False]),
    )

    projected = talker.head_code[0](hidden[[1]]).float() / 0.8
    projected = _fixed_window_repetition_penalty(
        projected,
        history,
        penalty=1.05,
        penalty_lut=talker._codec_repetition_penalty_lut,
    )
    projected[:, 7] = float("-inf")
    expected_logits, expected_ids = _exact_top_p_then_top_k_candidates(
        projected,
        top_k=3,
        top_p=0.85,
    )

    torch.testing.assert_close(candidate_logits, expected_logits)
    torch.testing.assert_close(candidate_ids, expected_ids)


def test_graph_sampler_masks_eos_without_indexed_inplace_write() -> None:
    import ast
    import inspect

    import textwrap

    source = textwrap.dedent(
        inspect.getsource(MiniCPMO45OmniTTSForConditionalGeneration.forward)
    )
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) for target in node.targets)
        for node in ast.walk(tree)
    )


def test_make_omni_output_reuses_graph_head_logits(mocker) -> None:
    talker = _make_talker()
    projected = torch.tensor([[11.0, 12.0], [21.0, 22.0]])
    seen: list[torch.Tensor] = []

    def sample(hidden, history, request_id, step, *, projected_logits=None):
        assert projected_logits is not None
        seen.append(projected_logits.clone())
        return torch.tensor(2 if request_id == "req-a" else 3)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    talker.make_omni_output(
        (torch.ones(2, 4), projected),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    assert len(seen) == 2
    torch.testing.assert_close(seen[0], projected[0:1])
    torch.testing.assert_close(seen[1], projected[1:2])


def test_make_omni_output_reuses_graph_sampler_candidates(mocker) -> None:
    talker = _make_talker()
    talker._talker_graph_sampler = True
    candidate_logits = torch.tensor([[3.0, 2.0, 1.0]])
    candidate_ids = torch.tensor([[5, 4, 3]])
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sample(hidden, history, request_id, step, *, compact_candidates=None, **kwargs):
        assert compact_candidates is not None
        seen.append(tuple(value.clone() for value in compact_candidates))
        return torch.tensor(5)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    info = {
        "request_id": "req-graph-sampler",
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    talker.make_omni_output(
        (torch.ones(1, 4), candidate_logits, candidate_ids),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert len(seen) == 1
    torch.testing.assert_close(seen[0][0], candidate_logits)
    torch.testing.assert_close(seen[0][1], candidate_ids)


def test_talker_binary_sampler_uses_exact_control_argmax() -> None:
    talker = _make_talker()
    talker._talker_binary_argmax = True
    logits = torch.tensor(
        [
            [0.0, float("-inf")],
            [float("-inf"), 0.0],
            [float("-inf"), float("-inf")],
        ]
    )

    output = talker.sample(logits, sampling_metadata=None)

    assert output.logprobs_tensors is None
    assert output.sampled_token_ids.dtype == torch.int32
    assert output.sampled_token_ids.tolist() == [[0], [1], [0]]


def test_talker_projects_request_aligned_duplex_metadata(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-a",
            "native_duplex": True,
            "duplex": {"epoch": 3, "turn_id": 7},
            "ids": {"tts": [41]},
            "meta": {
                "native_duplex_segment_text": "first",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-b",
            "native_duplex": True,
            "duplex": {"epoch": 4, "turn_id": 8},
            "ids": {"tts": [42, 99]},
            "meta": {
                "native_duplex_segment_text": "second",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
    ]

    output = talker.make_omni_output(
        torch.ones(2, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    meta = output.multimodal_outputs["meta"]
    assert [value.item() for value in meta["native_duplex"]] == [True, True]
    assert [value.item() for value in meta["duplex_epoch"]] == [3, 4]
    assert [value.item() for value in meta["duplex_turn_id"]] == [7, 8]
    assert "native_duplex_segment_text" not in meta
    assert [bytes(value.tolist()).decode("utf-8") for value in meta["llm_output_text_utf8"]] == [
        "first",
        "second",
    ]
    assert [value.item() for value in meta["turn_end"]] == [False, True]


def test_talker_reuses_explicit_request_duplex_metadata_without_rescanning_ids(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    any_mock = mocker.patch("torch.any", side_effect=AssertionError("unexpected TTS ID rescan"))
    info = {
        "request_id": "req-cached-meta",
        "native_duplex": True,
        "duplex": {"epoch": 3, "turn_id": 7},
        "ids": {"tts": torch.tensor([41, 99])},
        "meta": {
            "native_duplex_segment_text": "cached",
            "turn_eos_token_id": 99,
            "turn_end": True,
        },
        "audio_state": {"step": 0, "min_tokens": 26, "max_tokens": 26},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    ).multimodal_outputs["meta"]
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    ).multimodal_outputs["meta"]

    any_mock.assert_not_called()
    assert first["native_duplex"][0] is second["native_duplex"][0]
    assert first["llm_output_text_utf8"][0] is second["llm_output_text_utf8"][0]
    assert second["turn_end"][0].item() is True


def test_talker_rejects_native_duplex_without_fence_identity(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    info = {
        "request_id": "req-missing-fence",
        "native_duplex": True,
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    with pytest.raises(RuntimeError, match="requires non-negative integer epoch and turn_id"):
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )


def test_incomplete_prefill_emits_no_code_and_does_not_advance_state(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-prefill",
            "audio_state": {"step": 0},
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-decode",
            "audio_state": {"step": 4},
            "audio_codes": {"accumulated": torch.tensor([1])},
        },
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
        request_sample_eligible=[False, True],
    )

    sample.assert_called_once()
    assert sample.call_args.args[2] == "req-decode"
    assert infos[0]["audio_state"]["step"] == 0
    assert infos[0]["audio_codes"]["accumulated"].numel() == 0
    assert infos[1]["audio_state"]["step"] == 5
    assert _routed(output, 0)["codes"]["audio"].shape == (0, 1)
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[2]]


def test_eos_is_terminal_once_and_never_enters_codec_history(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(7))
    info = {
        "request_id": "req-stop",
        # The real sampler masks EOS before min_tokens.  Set zero here because
        # this test replaces the sampler with an unconditional EOS token.
        "audio_state": {"step": 3, "min_tokens": 0},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    first_logits = talker.compute_logits(first.text_hidden_states)
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    sample.assert_called_once()
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert first.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert first.multimodal_outputs["meta"]["finished"][0].item() is True
    assert second.multimodal_outputs["meta"]["finished"][0].item() is False
    assert first_logits.argmax(dim=-1).tolist() == [1]
    assert talker.compute_logits(second.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_sparse_codec_output_emits_only_on_initial_and_steady_boundaries(mocker) -> None:
    talker = _make_talker()
    talker._talker_emit_codec_chunks = True
    talker._talker_initial_codec_chunk_frames = 4
    talker._talker_codec_chunk_frames = 5
    samples = iter(range(1, 10))
    mocker.patch.object(talker, "_sample_audio_code", side_effect=lambda *_: torch.tensor(next(samples)))
    info = {
        "request_id": "req-sparse",
        "audio_state": {"step": 0, "min_tokens": 50, "max_tokens": 100},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    outputs = [
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        ).multimodal_outputs
        for _ in range(9)
    ]

    for index in (0, 1, 2, 4, 5, 6, 7):
        assert outputs[index]["meta"]["req_id"] == []
        assert outputs[index]["codes"]["audio"] == []
    assert outputs[3]["meta"]["req_id"] == ["req-sparse"]
    assert outputs[3]["meta"]["sparse_audio"] == ["1"]
    assert outputs[3]["codes"]["audio"][0].reshape(-1).tolist() == [1, 2, 3, 4]
    assert outputs[8]["meta"]["req_id"] == ["req-sparse"]
    assert outputs[8]["codes"]["audio"][0].reshape(-1).tolist() == [5, 6, 7, 8, 9]


def test_sparse_codec_output_flushes_tail_with_terminal_marker(mocker) -> None:
    talker = _make_talker()
    talker._talker_emit_codec_chunks = True
    talker._talker_initial_codec_chunk_frames = 4
    talker._talker_codec_chunk_frames = 5
    samples = iter([1, 2, 3, 4, 5, 6, 7])
    mocker.patch.object(talker, "_sample_audio_code", side_effect=lambda *_: torch.tensor(next(samples)))
    info = {
        "request_id": "req-terminal",
        "audio_state": {"step": 0, "min_tokens": 0, "max_tokens": 100},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    outputs = []
    for _ in range(7):
        outputs.append(
            talker.make_omni_output(
                torch.ones(1, 2),
                model_intermediate_buffer=[info],
                request_token_spans=[(0, 1)],
            ).multimodal_outputs
        )

    assert outputs[3]["codes"]["audio"][0].reshape(-1).tolist() == [1, 2, 3, 4]
    terminal = outputs[6]
    assert terminal["meta"]["req_id"] == ["req-terminal"]
    assert terminal["meta"]["finished"][0].item() is True
    assert terminal["codes"]["audio"][0].reshape(-1).tolist() == [5, 6]


def test_max_token_terminal_drops_unconsumed_codec_delta(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    info = {
        "request_id": "req-limit",
        "audio_state": {"step": 1, "max_tokens": 2},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    # MiniCPMTTS.generate_chunk samples once at the max-token boundary to
    # advance RNG state, but the sampled code is not fed into KV or returned.
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_request_local_state_survives_missing_runner_buffer_update(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    first_info = {
        "request_id": "req-local-state",
        "audio_state": {"step": 1, "max_tokens": 3},
        "audio_codes": {"accumulated": torch.tensor([4])},
    }

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[first_info],
        request_token_spans=[(0, 1)],
    )
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-local-state"}],
        request_token_spans=[(0, 1)],
    )

    assert second.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker._request_audio_states["req-local-state"]["step"] == 3


def test_missing_conditioning_fails_clearly() -> None:
    talker = _make_talker()

    with pytest.raises(ValueError, match="tts_token_ids and tts_hidden_states"):
        talker.preprocess(
            torch.tensor([0]),
            None,
            _omni_is_prefill=True,
            request_id="req-invalid",
        )


def test_empty_speech_segment_finishes_without_sampling_codes() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(8, 4)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])
    talker._text_eos_id = 5
    talker._tts_bos_id = 6

    _, embeds, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-empty",
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 4),
    )

    assert torch.equal(embeds, talker.emb_text(torch.tensor([5, 6])))
    assert updates["audio_state"]["finished"] is True

    # Stage 1's sampling min_tokens keeps scheduling decode steps until the stop
    # token becomes eligible, and those steps have no previous code to embed.
    _, decode_embeds, _ = talker.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        request_id="req-empty",
        audio_state=updates["audio_state"],
        audio_codes=updates["audio_codes"],
    )

    assert decode_embeds.shape == (1, 4)


def test_chunked_prefill_tail_aligns_condition_with_prompt_length(mocker) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    condition = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)

    _, embeds, _ = talker.preprocess(
        torch.zeros(9, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=59,
        _omni_prompt_len=68,
        request_id="req-chunked-prefill",
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    assert torch.equal(embeds, condition)
    state = talker._request_audio_states["req-chunked-prefill"]
    assert state["min_tokens"] == 50
    assert state["max_tokens"] == 64


@pytest.mark.parametrize(
    ("meta", "expected_min_tokens"),
    [
        ({"turn_start": True}, 0),
        ({}, 26),
        ({"turn_end": True}, 0),
    ],
)
def test_native_duplex_prefill_uses_official_chunk_limits(
    mocker,
    meta,
    expected_min_tokens,
) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    mocker.patch.object(
        talker,
        "_build_condition_embeddings",
        return_value=torch.ones(3, 2),
    )

    talker.preprocess(
        torch.zeros(3, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-duplex-chunk",
        native_duplex=True,
        meta=meta,
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    state = talker._request_audio_states["req-duplex-chunk"]
    assert state["min_tokens"] == expected_min_tokens
    assert state["max_tokens"] == 26


def test_native_duplex_condition_matches_official_text_plus_audio_bos() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(16, 2)
    talker.projector_semantic = nn.Identity()
    talker._normalize = False
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    with torch.no_grad():
        talker.emb_text.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(16, 2))

    token_ids = torch.tensor([2, 3])
    hidden_states = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    condition = talker._build_condition_embeddings(
        token_ids,
        hidden_states,
        native_duplex=True,
    )

    expected_text = talker.emb_text(token_ids) + hidden_states
    expected = torch.cat(
        [expected_text, talker.emb_text(torch.tensor([talker._tts_bos_id]))],
        dim=0,
    )
    assert torch.equal(condition, expected)
    assert condition.shape[0] == token_ids.shape[0] + 1


def test_request_cleanup_evicts_ar_rng_and_decode_state() -> None:
    talker = _make_talker()
    talker._request_generators["req-done"] = torch.Generator()
    talker._request_audio_states["req-done"] = {"step": 1}

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_generators
    assert "req-done" not in talker._request_audio_states
