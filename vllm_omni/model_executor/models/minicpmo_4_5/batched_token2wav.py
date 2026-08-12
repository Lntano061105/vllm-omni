# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

_SILENCE_TOKEN = 4218
logger = init_logger(__name__)


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    projected_speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


@dataclass
class _SteadyNPUGraph:
    """Fixed-shape recurrent Token2Wav graph owned by one backend instance."""

    graph: Any
    tokens: torch.Tensor
    features: PromptFeatures
    input_state: BatchedToken2WavState
    audio: torch.Tensor
    output_state: BatchedToken2WavState
    state_signature: tuple[Any, ...]
    prompt_mel_length: int


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(
        self,
        token2wav: Any,
        *,
        cfg_mode: str = "full",
        solver: str = "euler",
        prompt_bucket_frames: int = 0,
        prompt_encoder_device: str | torch.device | None = None,
    ):
        super().__init__()
        if cfg_mode not in {"full", "conditional", "conditional_scale"}:
            raise ValueError(
                "MiniCPM-o Token2Wav cfg_mode must be one of "
                "'full', 'conditional', or 'conditional_scale'"
            )
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        flow_parameter = next(self.flow.parameters(), None)
        self.flow_device = (
            flow_parameter.device if flow_parameter is not None else token2wav.speech_window.device
        )
        self.flow_dtype = flow_parameter.dtype if flow_parameter is not None else token2wav.speech_window.dtype
        self.cfg_mode = cfg_mode
        self.cfg_batch_multiplier = 2 if cfg_mode == "full" else 1
        if solver not in {"euler", "rk4"}:
            raise ValueError("MiniCPM-o Token2Wav solver must be 'euler' or 'rk4'")
        self.solver = solver
        self.prompt_bucket_frames = int(prompt_bucket_frames)
        if self.prompt_bucket_frames < 0:
            raise ValueError("prompt_bucket_frames must be >= 0")
        self.prompt_encoder: nn.Module | None = None
        if prompt_encoder_device is not None:
            prompt_device = torch.device(prompt_encoder_device)
            if prompt_device != self.flow_device:
                self.prompt_encoder = copy.deepcopy(self.flow.encoder).to(prompt_device)
        hift_parameter = next(self.hift.parameters(), None)
        self.hift_device = (
            hift_parameter.device if hift_parameter is not None else token2wav.speech_window.device
        )
        self.hift_dtype = hift_parameter.dtype if hift_parameter is not None else token2wav.speech_window.dtype
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self.n_timesteps = int(token2wav.n_timesteps)
        if self.n_timesteps < 1:
            raise ValueError("MiniCPM-o Token2Wav n_timesteps must be >= 1")
        self.num_evaluations = self.n_timesteps * (4 if self.solver == "rk4" else 1)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        timeline = torch.linspace(0, 1, self.n_timesteps + 1, dtype=torch.float32)
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        self.register_buffer(
            "cfm_timeline",
            timeline.to(device=self.flow_device, dtype=self.flow_dtype),
            persistent=False,
        )
        # Euler timesteps are fixed for the lifetime of the backend, and the
        # DiT timestep MLP is in inference mode.  Cache its output once instead
        # of rebuilding sinusoidal features (including device arange/exp/cos/
        # sin kernels) for every step of every audio chunk.
        if self.solver == "rk4":
            starts = timeline[:-1]
            ends = timeline[1:]
            middles = (starts + ends) * 0.5
            evaluation_times = torch.stack(
                (starts, middles, middles, ends),
                dim=1,
            ).reshape(-1)
        else:
            evaluation_times = timeline[:-1]
        self.register_buffer(
            "cfm_evaluation_times",
            evaluation_times.to(device=self.flow_device, dtype=self.flow_dtype),
            persistent=False,
        )
        # Use normal no-grad tensors rather than inference tensors. Stage-2
        # startup prewarm can be invoked from a context where autograd is
        # technically enabled, and PyTorch rejects inference tensors saved by
        # parameterized ops even though this model never calls backward.
        with torch.no_grad():
            time_embeddings = self.flow.decoder.estimator.t_embedder(self.cfm_evaluation_times).unsqueeze(1)
        self.register_buffer(
            "cfm_time_embeddings",
            time_embeddings.detach(),
            persistent=False,
        )
        # Every DiT block applies a large adaLN modulation projection to the
        # timestep embedding.  Both the embedding and model weights are fixed
        # during inference, so doing those projections in every block, Euler
        # evaluation, audio chunk, and request is pure duplicate work.  Cache
        # the exact projection outputs once and run a local block loop that
        # consumes them.  Unknown estimator implementations retain the
        # upstream blocks_forward_chunk fallback.
        estimator = self.flow.decoder.estimator
        final_layer = getattr(estimator, "final_layer", None)
        modulation_supported = final_layer is not None and all(
            all(
                hasattr(block, name)
                for name in ("adaLN_modulation", "norm1", "attn", "norm2", "mlp", "norm3", "conv")
            )
            for block in estimator.blocks
        ) and all(
            hasattr(final_layer, name)
            for name in ("adaLN_modulation", "norm_final", "linear")
        )
        block_modulations = None
        final_modulations = None
        if modulation_supported:
            with torch.no_grad():
                block_modulations = torch.stack(
                    [block.adaLN_modulation(time_embeddings) for block in estimator.blocks],
                    dim=1,
                ).detach()
                final_modulations = final_layer.adaLN_modulation(time_embeddings).detach()
        self.register_buffer(
            "cfm_block_modulations",
            block_modulations,
            persistent=False,
        )
        self.register_buffer(
            "cfm_final_modulations",
            final_modulations,
            persistent=False,
        )
        self.cached_dit_modulation = modulation_supported
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        self._initial_states: dict[tuple[str, str], BatchedToken2WavState] = {}
        self._steady_npugraphs: dict[tuple[Any, ...], _SteadyNPUGraph] = {}
        self._steady_npugraph_lock = Lock()
        self.steady_npugraph_replays = 0
        self.initial_state_cache_hits = 0

    @property
    def steady_npugraph_bucket_count(self) -> int:
        return len(self._steady_npugraphs)

    @staticmethod
    def _clone_state(state: BatchedToken2WavState) -> BatchedToken2WavState:
        return BatchedToken2WavState(
            flow_cache={name: tensor.clone() for name, tensor in state.flow_cache.items()},
            hift_cache={name: tensor.clone() for name, tensor in state.hift_cache.items()},
        )

    @staticmethod
    def _copy_state_(
        destination: BatchedToken2WavState,
        source: BatchedToken2WavState,
    ) -> None:
        for cache_name in ("flow_cache", "hift_cache"):
            destination_cache = getattr(destination, cache_name)
            source_cache = getattr(source, cache_name)
            if destination_cache.keys() != source_cache.keys():
                raise RuntimeError(
                    f"Token2Wav NPUGraph {cache_name} keys changed: "
                    f"{sorted(destination_cache)} != {sorted(source_cache)}"
                )
            for name, destination_tensor in destination_cache.items():
                source_tensor = source_cache[name]
                if (
                    destination_tensor.shape != source_tensor.shape
                    or destination_tensor.dtype != source_tensor.dtype
                    or destination_tensor.device != source_tensor.device
                ):
                    raise RuntimeError(
                        f"Token2Wav NPUGraph {cache_name}.{name} layout changed: "
                        f"{tensor_signature(destination_tensor)!r} != "
                        f"{tensor_signature(source_tensor)!r}"
                    )
                destination_tensor.copy_(source_tensor)

    def capture_steady_npugraph(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        state: BatchedToken2WavState,
    ) -> None:
        """Capture the exact-shape non-terminal single-stream decode path.

        The graph owns all recurrent input and output buffers. Live dispatch
        copies request state into those buffers and clones graph outputs back
        to request-owned storage, so sequential requests cannot overwrite one
        another. Shapes outside this one steady bucket retain eager execution.
        """
        if self.flow_device.type != "npu":
            raise ValueError("steady Token2Wav NPUGraph requires an NPU device")
        if int(tokens.shape[0]) != 1:
            raise ValueError("steady Token2Wav NPUGraph only supports batch size 1")
        graph_key = (
            tensor_signature(tokens),
            self._prompt_mel_length(features),
            state_shape_signature(state),
        )
        if graph_key in self._steady_npugraphs:
            return
        static_tokens = tokens.detach().clone()
        static_features = PromptFeatures(
            speech_tokens=features.speech_tokens.detach().clone(),
            projected_speaker_embedding=(
                features.projected_speaker_embedding.detach().clone()
            ),
            mels=features.mels.detach().clone(),
        )
        static_state = self._clone_state(state)
        pool = torch.npu.graph_pool_handle()
        graph = torch.npu.NPUGraph()
        with torch.inference_mode(), torch.npu.graph(graph, pool=pool):
            audio_rows, output_states = self._decode_batch_eager(
                static_tokens,
                static_features,
                [static_state],
                last_chunk=False,
            )
        output_state = output_states[0]
        input_signature = state_shape_signature(static_state)
        output_signature = state_shape_signature(output_state)
        if input_signature != output_signature:
            raise RuntimeError(
                "steady Token2Wav NPUGraph requires shape-stable recurrent "
                f"state, got input={input_signature!r} output={output_signature!r}"
            )
        self._steady_npugraphs[graph_key] = _SteadyNPUGraph(
            graph=graph,
            tokens=static_tokens,
            features=static_features,
            input_state=static_state,
            audio=audio_rows[0],
            output_state=output_state,
            state_signature=input_signature,
            prompt_mel_length=self._prompt_mel_length(features),
        )
        logger.info(
            "Captured steady MiniCPM-o Token2Wav NPUGraph bucket: "
            "tokens=%s prompt_mels=%d state=%s",
            tensor_signature(tokens),
            self._prompt_mel_length(features),
            input_signature,
        )

    def _decode_batch_npugraph(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures | Sequence[PromptFeatures],
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]] | None:
        if len(states) != 1 or not isinstance(features, PromptFeatures):
            return None
        graph_key = (
            tensor_signature(tokens),
            self._prompt_mel_length(features),
            state_shape_signature(states[0]),
        )
        captured = self._steady_npugraphs.get(graph_key)
        if (
            captured is None
            or last_chunk
            or flush_encoder
            or tokens.shape != captured.tokens.shape
            or tokens.dtype != captured.tokens.dtype
            or tokens.device != captured.tokens.device
            or self._prompt_mel_length(features) != captured.prompt_mel_length
            or state_shape_signature(states[0]) != captured.state_signature
        ):
            return None

        with self._steady_npugraph_lock:
            captured.tokens.copy_(tokens)
            captured.features.speech_tokens.copy_(features.speech_tokens)
            captured.features.projected_speaker_embedding.copy_(
                features.projected_speaker_embedding
            )
            captured.features.mels.copy_(features.mels)
            self._copy_state_(captured.input_state, states[0])
            captured.graph.replay()
            # The graph reuses its outputs on every replay. Clone them before
            # releasing the host lock so request state and connector audio stay
            # valid when the next stream reuses the graph buffers.
            audio = captured.audio.detach().clone()
            next_state = self._clone_state(captured.output_state)
            torch.accelerator.synchronize(self.flow_device)
            self.steady_npugraph_replays += 1
            if self.steady_npugraph_replays == 1:
                logger.info(
                    "Replayed steady MiniCPM-o Token2Wav NPUGraph: "
                    "tokens=%s prompt_mels=%d",
                    tensor_signature(tokens),
                    captured.prompt_mel_length,
                )
        return [audio], [next_state]

    def bucket_prompt_features(
        self,
        features: PromptFeatures,
        target_frames: int | None = None,
    ) -> PromptFeatures:
        """Pad prompt conditioning to a reusable NPU shape bucket.

        Ascend eager operators pay a multi-second first-use cost for every new
        prompt sequence length.  Rounding only the cached conditioning suffix
        lets a small set of shapes be warmed at startup.  Speaker extraction
        remains based on the original waveform; the suffix uses the model's
        silence codec token and repeats the final (normally silent) mel frame.
        """
        current_frames = int(features.speech_tokens.shape[1])
        if target_frames is None:
            bucket = self.prompt_bucket_frames
            if bucket <= 0:
                return features
            target_frames = ((current_frames + bucket - 1) // bucket) * bucket
        target_frames = int(target_frames)
        if target_frames < current_frames:
            raise ValueError(
                f"target prompt bucket {target_frames} is below prompt length {current_frames}"
            )
        if target_frames == current_frames:
            return features

        token_padding = features.speech_tokens.new_full(
            (features.speech_tokens.shape[0], target_frames - current_frames),
            _SILENCE_TOKEN,
        )
        speech_tokens = torch.cat((features.speech_tokens, token_padding), dim=1)
        up_rate = int(features.mels.shape[1]) // current_frames
        if up_rate <= 0 or int(features.mels.shape[1]) != current_frames * up_rate:
            raise ValueError(
                "prompt mel frames must be an integer multiple of speech token frames"
            )
        target_mels = target_frames * up_rate
        mel_padding = features.mels[:, -1:, :].expand(
            -1,
            target_mels - int(features.mels.shape[1]),
            -1,
        )
        return PromptFeatures(
            speech_tokens=speech_tokens,
            projected_speaker_embedding=features.projected_speaker_embedding,
            mels=torch.cat((features.mels, mel_padding), dim=1),
        )

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            speaker_embedding = values[2].to(
                device=self.flow_device,
                dtype=self.flow_dtype,
            )
            with self._autocast(speaker_embedding.device):
                projected_speaker_embedding = self.flow.spk_embed_affine_layer(
                    F.normalize(speaker_embedding, dim=1)
                )
            cached = self.bucket_prompt_features(PromptFeatures(
                speech_tokens=values[0],
                projected_speaker_embedding=projected_speaker_embedding,
                mels=values[3].to(device=self.flow_device, dtype=self.flow_dtype),
            ))
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        cache_key = (prompt_cache_id, prompt_wav)
        self._prompt_features.pop(cache_key, None)
        self._initial_states.pop(cache_key, None)

    def setup_cached_batch(
        self,
        prompt_cache_id: str,
        prompt_wav: str,
        batch_size: int,
    ) -> tuple[PromptFeatures, list[BatchedToken2WavState]]:
        """Reuse a deterministic, read-only prompt-initialized state.

        Prompt encoding and the initial CFM cache construction depend only on
        the reference prompt. Computing them for every request wastes most of
        the Code2Wav work before the first live codec frame arrives. The
        streaming encoder, DiT estimator, and HiFT vocoder treat their incoming
        caches as read-only and return new cache tensors, so all new requests
        can start from one immutable template without copying tens of MiB of
        prompt attention state. The first live decode returns request-owned
        states as usual.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        cache_key = (prompt_cache_id, prompt_wav)
        features = self.prepare_prompt(prompt_cache_id, prompt_wav)
        template = self._initial_states.get(cache_key)
        if template is None:
            template = self.setup_batch(features, 1)[0]
            self._initial_states[cache_key] = template
            logger.info(
                "Cached MiniCPM-o Token2Wav immutable initial state: prompt=%s",
                prompt_cache_id,
            )
        else:
            self.initial_state_cache_hits += 1
            if self.initial_state_cache_hits == 1:
                logger.info(
                    "Reused MiniCPM-o Token2Wav immutable initial state"
                )
        return features, [template] * batch_size

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    @staticmethod
    def _speaker_batch(
        features: PromptFeatures | Sequence[PromptFeatures],
        batch_size: int,
    ) -> torch.Tensor:
        if isinstance(features, PromptFeatures):
            return features.projected_speaker_embedding.expand(batch_size, -1)
        if len(features) != batch_size:
            raise ValueError(f"prompt feature batch {len(features)} != token batch {batch_size}")
        if batch_size == 1:
            return features[0].projected_speaker_embedding
        return torch.cat([feature.projected_speaker_embedding for feature in features], dim=0)

    @staticmethod
    def _prompt_mel_length(features: PromptFeatures | Sequence[PromptFeatures]) -> int:
        if isinstance(features, PromptFeatures):
            return int(features.mels.shape[1])
        lengths = {int(feature.mels.shape[1]) for feature in features}
        if len(lengths) != 1:
            raise ValueError(f"mixed prompt mel lengths cannot share one cache batch: {sorted(lengths)}")
        return lengths.pop()

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        prompt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder = self.prompt_encoder if prompt and self.prompt_encoder is not None else self.flow.encoder
        encoder_parameter = next(encoder.parameters(), None)
        encoder_device = encoder_parameter.device if encoder_parameter is not None else self.flow_device
        encoder_dtype = encoder_parameter.dtype if encoder_parameter is not None else self.flow_dtype
        embedded = self.flow.input_embedding(tokens).to(
            device=encoder_device,
            dtype=encoder_dtype,
        )
        if cnn_cache is not None:
            cnn_cache = cnn_cache.to(device=encoder_device, dtype=encoder_dtype)
        if att_cache is not None:
            att_cache = att_cache.to(device=encoder_device, dtype=encoder_dtype)
        hidden, new_cnn, new_att = encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        hidden = hidden.to(device=self.flow_device, dtype=self.flow_dtype)
        return (
            self.flow.encoder_proj(hidden),
            new_cnn.to(device=self.flow_device, dtype=self.flow_dtype),
            new_att.to(device=self.flow_device, dtype=self.flow_dtype),
        )

    @staticmethod
    def _estimator_buffer_shapes(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        return (
            (depth, batch_size, cnn_channels, cnn_width),
            (depth, batch_size, heads, old_att_len + chunk_size, att_width),
        )

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time_embedding: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        evaluation: int,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
    ) -> torch.Tensor:
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        if not self.cached_dit_modulation:
            return estimator.blocks_forward_chunk(
                estimator_input,
                time_embedding,
                None,
                old_cnn,
                old_att,
                cnn_out,
                att_out,
            )
        return self._estimator_blocks_forward_chunk_cached_modulation(
            estimator,
            estimator_input,
            evaluation=evaluation,
            cnn_cache=old_cnn,
            att_cache=old_att,
            cnn_out=cnn_out,
            att_out=att_out,
        )

    @staticmethod
    def _modulate(
        value: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return value * (1.0 + scale) + shift

    def _estimator_blocks_forward_chunk_cached_modulation(
        self,
        estimator: nn.Module,
        estimator_input: torch.Tensor,
        *,
        evaluation: int,
        cnn_cache: Any,
        att_cache: Any,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
    ) -> torch.Tensor:
        """Run the upstream DiT block loop with exact cached adaLN outputs."""
        if self.cfm_block_modulations is None or self.cfm_final_modulations is None:
            raise RuntimeError("cached DiT modulation buffers are unavailable")

        hidden = estimator.in_proj(estimator_input.transpose(1, 2))
        batch_size = int(hidden.shape[0])
        for block_index, block in enumerate(estimator.blocks):
            modulation = self.cfm_block_modulations[evaluation, block_index].expand(
                batch_size,
                -1,
                -1,
            )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                shift_conv,
                scale_conv,
                gate_conv,
            ) = modulation.chunk(9, dim=-1)

            attention, new_att = block.attn.forward_chunk(
                self._modulate(block.norm1(hidden), shift_msa, scale_msa),
                att_cache[block_index],
                None,
            )
            hidden = hidden + gate_msa * attention
            convolution, new_cnn = block.conv.forward_chunk(
                self._modulate(block.norm3(hidden), shift_conv, scale_conv),
                cnn_cache[block_index],
            )
            hidden = hidden + gate_conv * convolution
            hidden = hidden + gate_mlp * block.mlp(
                self._modulate(block.norm2(hidden), shift_mlp, scale_mlp)
            )

            cnn_out[block_index].copy_(new_cnn)
            att_out[block_index, :, :, : new_att.shape[2], :].copy_(new_att)

        final_modulation = self.cfm_final_modulations[evaluation].expand(
            batch_size,
            -1,
            -1,
        )
        shift, scale = final_modulation.chunk(2, dim=-1)
        hidden = self._modulate(estimator.final_layer.norm_final(hidden), shift, scale)
        return estimator.final_layer.linear(hidden).transpose(1, 2)

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = self.cfm_timeline
        if timeline.device != mu.device or timeline.dtype != mu.dtype:
            timeline = timeline.to(device=mu.device, dtype=mu.dtype)
        if self.cfg_mode == "full":
            model_mu = torch.cat((mu, torch.zeros_like(mu)), dim=0)
            model_speakers = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
            model_cond = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        else:
            model_mu = mu
            model_speakers = speakers
            model_cond = cond
        model_batch_size = self.cfg_batch_multiplier * batch_size
        first_old_att = att_cache[0] if att_cache is not None else None
        cnn_shape, att_shape = self._estimator_buffer_shapes(
            estimator,
            model_mu,
            first_old_att,
        )
        # Allocate the final stacked cache once and let every Euler step write
        # directly into its contiguous slice.  The previous implementation
        # allocated two tensors per step and copied all of them again through
        # torch.stack at the end of every audio chunk.
        next_cnn = mu.new_empty((self.num_evaluations, *cnn_shape))
        next_att = mu.new_empty((self.num_evaluations, *att_shape))

        def evaluate(latent: torch.Tensor, evaluation: int) -> torch.Tensor:
            old_cnn = cnn_cache[evaluation] if cnn_cache is not None else None
            old_att = att_cache[evaluation] if att_cache is not None else None
            # CFG evaluates the conditional and unconditional branches with
            # the same latent.  The score-critical single-request path can
            # expose two read-only rows through a stride-0 view instead of
            # allocating and copying ``x`` at every Euler step.  Keep the
            # existing materialized layout for real batches because flattening
            # a repeated multi-row batch cannot be represented as one view.
            if self.cfg_mode == "full":
                model_x = (
                    latent.expand(2, -1, -1)
                    if batch_size == 1
                    else torch.cat((latent, latent), dim=0)
                )
            else:
                model_x = latent
            estimate = self._estimator_step(
                estimator,
                x=model_x,
                mu=model_mu,
                time_embedding=self.cfm_time_embeddings[evaluation : evaluation + 1].expand(
                    model_batch_size,
                    -1,
                    -1,
                ),
                speakers=model_speakers,
                cond=model_cond,
                evaluation=evaluation,
                cnn_cache=old_cnn,
                att_cache=old_att,
                cnn_out=next_cnn[evaluation],
                att_out=next_att[evaluation],
            )
            if self.cfg_mode == "full":
                conditional, unconditional = estimate.split(batch_size, dim=0)
                return (
                    (1.0 + decoder.inference_cfg_rate) * conditional
                    - decoder.inference_cfg_rate * unconditional
                )
            if self.cfg_mode == "conditional_scale":
                return (1.0 + decoder.inference_cfg_rate) * estimate
            return estimate

        if self.solver == "rk4":
            for step in range(self.n_timesteps):
                dt = timeline[step + 1] - timeline[step]
                evaluation = step * 4
                k1 = evaluate(x, evaluation)
                k2 = evaluate(x + 0.5 * dt * k1, evaluation + 1)
                k3 = evaluate(x + 0.5 * dt * k2, evaluation + 2)
                k4 = evaluate(x + dt * k3, evaluation + 3)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            for step in range(self.n_timesteps):
                dt = timeline[step + 1] - timeline[step]
                x = x + dt * evaluate(x, step)
        return x, next_cnn, next_att

    def _split_flow_cache(
        self,
        cache: dict[str, torch.Tensor],
        batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        if batch_size == 1:
            return [
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"].detach(),
                    "conformer_att_cache": cache["conformer_att_cache"].detach(),
                    "estimator_cnn_cache": cache["estimator_cnn_cache"].detach(),
                    "estimator_att_cache": cache["estimator_att_cache"].detach(),
                }
            ]
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            if self.cfg_mode == "full":
                estimator_cnn = torch.cat(
                    (
                        cache["estimator_cnn_cache"][:, :, row : row + 1],
                        cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                    ),
                    dim=2,
                ).detach()
                estimator_att = torch.cat(
                    (
                        cache["estimator_att_cache"][:, :, row : row + 1],
                        cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                    ),
                    dim=2,
                ).detach()
            else:
                estimator_cnn = cache["estimator_cnn_cache"][:, :, row : row + 1].detach().clone()
                estimator_att = cache["estimator_att_cache"][:, :, row : row + 1].detach().clone()
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": estimator_cnn,
                    "estimator_att_cache": estimator_att,
                }
            )
        return result

    def _stack_flow_cache(self, states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        flows = [state.flow_cache for state in states]
        if len(flows) == 1:
            return dict(flows[0])
        if self.cfg_mode != "full":
            return {
                "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
                "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
                "estimator_cnn_cache": torch.cat([flow["estimator_cnn_cache"] for flow in flows], dim=2),
                "estimator_att_cache": torch.cat([flow["estimator_att_cache"] for flow in flows], dim=2),
            }
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tokens, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
                prompt=True,
            )
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                features.projected_speaker_embedding.expand(batch_size, -1),
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": torch.zeros(
                        (1, mel_channels, 0),
                        device=self.hift_device,
                        dtype=self.hift_dtype,
                    ),
                    "source": torch.zeros(
                        (1, 1, 0),
                        device=self.hift_device,
                        dtype=self.hift_dtype,
                    ),
                    "speech": torch.zeros(
                        (1, 0),
                        device=self.hift_device,
                        dtype=self.hift_dtype,
                    ),
                },
            )
            for row in split
        ]

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures | Sequence[PromptFeatures],
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        graph_result = self._decode_batch_npugraph(
            tokens,
            features,
            states,
            last_chunk=last_chunk,
            flush_encoder=flush_encoder,
        )
        if graph_result is not None:
            return graph_result
        return self._decode_batch_eager(
            tokens,
            features,
            states,
            last_chunk=last_chunk,
            flush_encoder=flush_encoder,
        )

    def _decode_batch_eager(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures | Sequence[PromptFeatures],
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        flow_cache = self._stack_flow_cache(states)
        speakers = self._speaker_batch(features, batch_size)
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )

        prompt_len = self._prompt_mel_length(features)
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        if batch_size == 1:
            old_mel = states[0].hift_cache["mel"]
            old_source = states[0].hift_cache["source"]
            old_speech = states[0].hift_cache["speech"]
        else:
            old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
            old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
            old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        chunk_mel = chunk_mel.to(device=old_mel.device, dtype=old_mel.dtype)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        speech, source = self.hift(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        if batch_size == 1:
            next_states = [
                BatchedToken2WavState(
                    flow_cache=new_flow[0],
                    hift_cache={name: value.detach() for name, value in next_hift.items()},
                )
            ]
        else:
            next_states = [
                BatchedToken2WavState(
                    flow_cache=new_flow[row],
                    hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
                )
                for row in range(batch_size)
            ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states
