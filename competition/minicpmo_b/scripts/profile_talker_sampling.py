#!/usr/bin/env python3
"""Microbenchmark MiniCPM-o codec sampling on CPU or NPU."""

from __future__ import annotations

import argparse
import json
import time

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _apply_repetition_penalty,
    _apply_top_k_top_p,
)


def _dense_repetition_penalty(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    recent = history[-16:].to(device=logits.device, dtype=torch.long)
    frequencies = torch.bincount(recent, minlength=logits.shape[-1]).to(logits.dtype)
    alpha = torch.pow(torch.as_tensor(1.05, device=logits.device), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _true_sparse_repetition_penalty(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    recent = history[-16:].to(device=logits.device, dtype=torch.long)
    token_ids, counts = torch.unique(recent, return_counts=True)
    selected = logits.index_select(-1, token_ids)
    alpha = torch.pow(
        torch.as_tensor(1.05, device=logits.device, dtype=logits.dtype),
        counts.to(logits.dtype),
    )
    adjusted = torch.where(selected < 0, selected * alpha, selected / alpha)
    return logits.clone().scatter(-1, token_ids.expand(logits.shape[0], -1), adjusted)


def _scatter_repetition_penalty(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Exact dense penalty without torch.bincount's NPU scalar syncs."""
    recent = history[-16:].to(device=logits.device, dtype=torch.long)
    frequencies = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    frequencies.scatter_add_(
        0,
        recent,
        torch.ones_like(recent, dtype=logits.dtype),
    )
    alpha = torch.pow(torch.as_tensor(1.05, device=logits.device), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _full_sort_top_k_top_p(logits: torch.Tensor) -> torch.Tensor:
    filtered = logits.clone()
    sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative_probs <= 0.15
    remove[..., -3:] = False
    remove = remove.scatter(-1, sorted_indices, remove)
    filtered.masked_fill_(remove, float("-inf"))
    threshold = torch.topk(filtered, 25, dim=-1).values[..., -1, None]
    return filtered.masked_fill(filtered < threshold, float("-inf"))


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _measure(fn, iterations: int, device: torch.device) -> float:
    for _ in range(10):
        fn()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0" if hasattr(torch, "npu") else "cpu")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(42)
    logits_cpu = torch.randn(1, 6562, generator=generator, dtype=torch.float32)
    logits = logits_cpu.to(device)
    history = torch.tensor([3, 9, 3, 1024, 9, 9, 6550, 1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)

    def old_sampling() -> torch.Tensor:
        filtered = _full_sort_top_k_top_p(_dense_repetition_penalty(logits, history))
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1)

    def new_sampling() -> torch.Tensor:
        penalized = _apply_repetition_penalty(logits, history, penalty=1.05, window_size=16)
        filtered = _apply_top_k_top_p(penalized, top_k=args.top_k, top_p=0.85, min_tokens_to_keep=3)
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1)

    def compact_candidate_sampling() -> torch.Tensor:
        penalized = _apply_repetition_penalty(logits, history, penalty=1.05, window_size=16)
        # Preserve the upstream TopP -> TopK order, but run softmax and
        # multinomial only on the surviving 25 candidates instead of the
        # complete 6562-token vocabulary.
        top_p_filtered = _apply_top_k_top_p(
            penalized,
            top_k=None,
            top_p=0.85,
            min_tokens_to_keep=3,
        )
        candidate_logits, candidate_ids = torch.topk(top_p_filtered, args.top_k, dim=-1)
        sampled_candidate = torch.multinomial(torch.softmax(candidate_logits, dim=-1), 1)
        return candidate_ids.gather(-1, sampled_candidate)

    def true_sparse_sampling() -> torch.Tensor:
        penalized = _true_sparse_repetition_penalty(logits, history)
        filtered = _apply_top_k_top_p(
            penalized,
            top_k=args.top_k,
            top_p=0.85,
            min_tokens_to_keep=3,
        )
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1)

    def true_sparse_compact_sampling() -> torch.Tensor:
        penalized = _true_sparse_repetition_penalty(logits, history)
        top_p_filtered = _apply_top_k_top_p(
            penalized,
            top_k=None,
            top_p=0.85,
            min_tokens_to_keep=3,
        )
        candidate_logits, candidate_ids = torch.topk(top_p_filtered, args.top_k, dim=-1)
        sampled_candidate = torch.multinomial(torch.softmax(candidate_logits, dim=-1), 1)
        return candidate_ids.gather(-1, sampled_candidate)

    def scatter_sampling() -> torch.Tensor:
        penalized = _scatter_repetition_penalty(logits, history)
        filtered = _apply_top_k_top_p(
            penalized,
            top_k=args.top_k,
            top_p=0.85,
            min_tokens_to_keep=3,
        )
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1)

    def dense_rep_fast_warp_sampling() -> torch.Tensor:
        penalized = _dense_repetition_penalty(logits, history)
        filtered = _apply_top_k_top_p(penalized, top_k=25, top_p=0.85, min_tokens_to_keep=3)
        return torch.multinomial(torch.softmax(filtered, dim=-1), 1)

    old_ms = _measure(old_sampling, args.iterations, device)
    new_ms = _measure(new_sampling, args.iterations, device)
    dense_fast_ms = _measure(dense_rep_fast_warp_sampling, args.iterations, device)
    compact_ms = _measure(compact_candidate_sampling, args.iterations, device)
    true_sparse_ms = _measure(true_sparse_sampling, args.iterations, device)
    true_sparse_compact_ms = _measure(true_sparse_compact_sampling, args.iterations, device)
    scatter_ms = _measure(scatter_sampling, args.iterations, device)
    result = {
        "device": str(device),
        "iterations": args.iterations,
        "top_k": args.top_k,
        "old_sampling_ms": old_ms,
        "new_sampling_ms": new_ms,
        "speedup": old_ms / new_ms,
        "saved_ms_per_codec_token": old_ms - new_ms,
        "dense_rep_fast_warp_sampling_ms": dense_fast_ms,
        "dense_rep_fast_warp_speedup": old_ms / dense_fast_ms,
        "compact_candidate_sampling_ms": compact_ms,
        "compact_candidate_speedup_vs_current": new_ms / compact_ms,
        "compact_candidate_saved_ms_per_codec_token": new_ms - compact_ms,
        "true_sparse_sampling_ms": true_sparse_ms,
        "true_sparse_speedup_vs_current": new_ms / true_sparse_ms,
        "true_sparse_saved_ms_per_codec_token": new_ms - true_sparse_ms,
        "true_sparse_compact_sampling_ms": true_sparse_compact_ms,
        "true_sparse_compact_speedup_vs_current": new_ms / true_sparse_compact_ms,
        "scatter_sampling_ms": scatter_ms,
        "scatter_sampling_speedup_vs_current": new_ms / scatter_ms,
        "scatter_sampling_saved_ms_per_codec_token": new_ms - scatter_ms,
        "dense_repetition_ms": _measure(
            lambda: _dense_repetition_penalty(logits, history),
            args.iterations,
            device,
        ),
        "sparse_repetition_ms": _measure(
            lambda: _apply_repetition_penalty(logits, history, penalty=1.05, window_size=16),
            args.iterations,
            device,
        ),
        "true_sparse_repetition_ms": _measure(
            lambda: _true_sparse_repetition_penalty(logits, history),
            args.iterations,
            device,
        ),
        "scatter_repetition_ms": _measure(
            lambda: _scatter_repetition_penalty(logits, history),
            args.iterations,
            device,
        ),
        "full_sort_warper_ms": _measure(
            lambda: _full_sort_top_k_top_p(logits),
            args.iterations,
            device,
        ),
        "fast_warper_ms": _measure(
            lambda: _apply_top_k_top_p(logits, top_k=25, top_p=0.85, min_tokens_to_keep=3),
            args.iterations,
            device,
        ),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")


if __name__ == "__main__":
    main()
