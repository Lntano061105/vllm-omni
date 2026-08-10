#!/usr/bin/env python3
"""Standalone NPU profiler for MiniCPM-o 4.5 Flow and HiFT.

This avoids the five-minute three-stage service startup while screening
Stage-2 optimizations.  It reports asynchronous NPU-event time for the token
encoder, CFM/DiT solver, HiFT, and the complete steady streaming chunk.  The
optional weight-norm A/B uses the same loaded weights and prompt state.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
    MiniCPMO45Token2wav,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    _remove_weight_norm_for_inference,
)


def _event() -> Any:
    if hasattr(torch, "npu"):
        return torch.npu.Event(enable_timing=True)
    return torch.cuda.Event(enable_timing=True)


def _synchronize(device: torch.device) -> None:
    torch.accelerator.synchronize(device)


class EventProfiler:
    def __init__(self) -> None:
        self.events: dict[str, list[tuple[Any, Any]]] = defaultdict(list)

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            start, end = _event(), _event()
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            self.events[name].append((start, end))
            return result

        return timed

    def reset(self) -> None:
        self.events.clear()

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, pairs in self.events.items():
            values = []
            for start, end in pairs:
                try:
                    values.append(float(start.elapsed_time(end)))
                except RuntimeError:
                    # Some CPU-only module wrappers enqueue no accelerator
                    # work, and torch_npu then leaves their timing events
                    # unrecorded. They are not device bottlenecks.
                    continue
            if not values:
                continue
            result[name] = {
                "calls": len(values),
                "total_ms": sum(values),
                "mean_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        return result


def _instrument_encoder(adapter: BatchedToken2Wav, profiler: EventProfiler) -> None:
    encoder = adapter.flow.encoder
    encoder.embed.forward = profiler.wrap("encoder.embed", encoder.embed.forward)
    encoder.pre_lookahead_layer.forward_chunk = profiler.wrap(
        "encoder.pre_lookahead",
        encoder.pre_lookahead_layer.forward_chunk,
    )
    for index, layer in enumerate(encoder.encoders):
        layer.forward = profiler.wrap(f"encoder.block.{index}", layer.forward)
    encoder.up_layer.forward_chunk = profiler.wrap(
        "encoder.upsample",
        encoder.up_layer.forward_chunk,
    )
    encoder.up_embed.forward = profiler.wrap("encoder.up_embed", encoder.up_embed.forward)
    for index, layer in enumerate(encoder.up_encoders):
        layer.forward = profiler.wrap(f"encoder.up_block.{index}", layer.forward)
    encoder.after_norm.forward = profiler.wrap(
        "encoder.after_norm",
        encoder.after_norm.forward,
    )


def _run_case(
    adapter: BatchedToken2Wav,
    profiler: EventProfiler,
    *,
    prompt_id: str,
    prompt_wav: str,
    initial_frames: int,
    steady_frames: int,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    features, states = adapter.setup_cached_batch(prompt_id, prompt_wav, 1)
    device = features.speech_tokens.device
    lookahead = adapter._pre_lookahead_len() or 0

    def decode(frames: int) -> None:
        nonlocal states
        tokens = torch.full(
            (1, lookahead + frames),
            4218,
            dtype=torch.long,
            device=device,
        )
        _, states = adapter.decode_batch(
            tokens,
            features,
            states,
            last_chunk=False,
        )

    # Reach the same capped-attention-cache steady state used by the service.
    decode(initial_frames)
    for _ in range(max(3, warmups)):
        decode(steady_frames)
    _synchronize(device)
    profiler.reset()

    total_pairs: list[tuple[Any, Any]] = []
    for _ in range(iterations):
        start, end = _event(), _event()
        start.record()
        decode(steady_frames)
        end.record()
        total_pairs.append((start, end))
    _synchronize(device)

    totals = [float(start.elapsed_time(end)) for start, end in total_pairs]
    result = profiler.summary()
    result["total"] = {
        "calls": len(totals),
        "total_ms": sum(totals),
        "mean_ms": sum(totals) / len(totals),
        "min_ms": min(totals),
        "max_ms": max(totals),
    }
    audio_seconds = steady_frames / 25.0
    result["rtf"] = {
        "audio_seconds_per_chunk": audio_seconds,
        "mean": (sum(totals) / len(totals)) / (audio_seconds * 1000.0),
    }
    return result


def _profile_cold_prompt(
    adapter: BatchedToken2Wav,
    profiler: EventProfiler,
    *,
    prompt_id: str,
    prompt_wav: str,
) -> dict[str, Any]:
    """Measure the two cold-start components hidden by steady-state RTF.

    Prompt feature extraction mixes CPU audio/ONNX work with the NPU speech
    tokenizer.  Initial-state construction then runs the prompt through the
    streaming encoder and CFM.  Synchronizing around both phases makes their
    wall times directly comparable with request-observed TTFP.
    """
    device = adapter.flow_device
    _synchronize(device)
    started = time.perf_counter()
    features = adapter.prepare_prompt(prompt_id, prompt_wav)
    _synchronize(device)
    prepare_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    profiler.reset()
    states = adapter.setup_batch(features, 1)
    _synchronize(device)
    setup_ms = (time.perf_counter() - started) * 1000.0
    lookahead = adapter._pre_lookahead_len() or 0
    first_tokens = torch.full(
        (1, lookahead + 4),
        4218,
        dtype=torch.long,
        device=features.speech_tokens.device,
    )
    _synchronize(device)
    started = time.perf_counter()
    adapter.decode_batch(first_tokens, features, states, last_chunk=False)
    _synchronize(device)
    first_live_ms = (time.perf_counter() - started) * 1000.0
    return {
        "prompt_id": prompt_id,
        "prompt_wav": prompt_wav,
        "speech_token_frames": int(features.speech_tokens.shape[1]),
        "mel_frames": int(features.mels.shape[1]),
        "prepare_prompt_ms": prepare_ms,
        "setup_initial_state_ms": setup_ms,
        "first_live_chunk_ms": first_live_ms,
        "setup_device_components": profiler.summary(),
        "total_ms": prepare_ms + setup_ms + first_live_ms,
        "flow_cache_signature": str(
            {
                name: tuple(tensor.shape)
                for name, tensor in states[0].flow_cache.items()
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/MiniCPM-o-4_5")
    parser.add_argument("--prompt-wav")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cfg-mode", choices=("full", "conditional"), default="conditional")
    parser.add_argument("--solver", choices=("euler", "rk4"), default="rk4")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--initial-frames", type=int, default=4)
    parser.add_argument("--steady-frames", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compare-remove-weight-norm", action="store_true")
    parser.add_argument(
        "--disable-jit-compile",
        action="store_true",
        help="Use precompiled Ascend operators instead of per-shape JIT compilation.",
    )
    parser.add_argument("--fused-encoder-attention", action="store_true")
    parser.add_argument("--profile-encoder-components", action="store_true")
    parser.add_argument(
        "--compile-encoder-backend",
        choices=("npu", "npugraph_ex", "npugraphs"),
    )
    parser.add_argument("--prompt-encoder-device")
    parser.add_argument("--encoder-device")
    parser.add_argument("--encoder-dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument(
        "--cold-prompt-wav",
        action="append",
        default=[],
        help="Additional prompt WAV to profile from an empty cache; repeatable.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be >= 1")
        torch.set_num_threads(args.cpu_threads)

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
        if args.disable_jit_compile:
            torch.npu.set_compile_mode(jit_compile=False)
    model_path = Path(args.model_path)
    prompt_wav = args.prompt_wav or str(model_path / "assets" / "HT_ref_audio.wav")

    token2wav = MiniCPMO45Token2wav(
        str(model_path / "assets" / "token2wav"),
        float16=False,
        n_timesteps=args.steps,
        device=device,
    )
    if args.fused_encoder_attention:
        from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
            patch_minicpmo45_encoder_fused_attention,
        )

        patch_minicpmo45_encoder_fused_attention(token2wav.flow.encoder)
    if args.compile_encoder_backend:
        encoder = token2wav.flow.encoder
        compiled_forward = torch.compile(
            type(encoder).forward_chunk,
            backend=args.compile_encoder_backend,
            dynamic=True,
            fullgraph=False,
        )
        encoder.forward_chunk = MethodType(compiled_forward, encoder)
    adapter = BatchedToken2Wav(
        token2wav,
        cfg_mode=args.cfg_mode,
        solver=args.solver,
        prompt_encoder_device=args.prompt_encoder_device,
    )
    if args.encoder_device:
        encoder_dtype = (
            torch.float32
            if args.encoder_dtype in (None, "float32")
            else torch.bfloat16
        )
        adapter.flow.encoder.to(
            device=torch.device(args.encoder_device),
            dtype=encoder_dtype,
        )
    profiler = EventProfiler()
    if args.profile_encoder_components:
        _instrument_encoder(adapter, profiler)
    adapter._encode_chunk = profiler.wrap("encoder", adapter._encode_chunk)  # type: ignore[method-assign]
    adapter._decode_cfm = profiler.wrap("cfm", adapter._decode_cfm)  # type: ignore[method-assign]
    adapter.hift.forward = profiler.wrap("hift", adapter.hift.forward)  # type: ignore[method-assign]

    with torch.inference_mode():
        cold_prompts = [
            _profile_cold_prompt(
                adapter,
                profiler,
                prompt_id=f"cold-{index}",
                prompt_wav=path,
            )
            for index, path in enumerate(args.cold_prompt_wav)
        ]
        before = _run_case(
            adapter,
            profiler,
            prompt_id="HT_ref_audio",
            prompt_wav=prompt_wav,
            initial_frames=args.initial_frames,
            steady_frames=args.steady_frames,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        payload: dict[str, Any] = {
            "cold_prompts": cold_prompts,
            "before_remove_weight_norm": before,
        }
        if args.compare_remove_weight_norm:
            removed = _remove_weight_norm_for_inference(adapter.hift)
            if removed == 0:
                raise RuntimeError("HiFT contained no weight norm to remove")
            after = _run_case(
                adapter,
                profiler,
                prompt_id="HT_ref_audio",
                prompt_wav=prompt_wav,
                initial_frames=args.initial_frames,
                steady_frames=args.steady_frames,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            payload["after_remove_weight_norm"] = after
            payload["removed_weight_norm_modules"] = removed
            payload["speedup"] = {
                "total": before["total"]["mean_ms"] / after["total"]["mean_ms"],
                "hift": before["hift"]["mean_ms"] / after["hift"]["mean_ms"],
            }

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
