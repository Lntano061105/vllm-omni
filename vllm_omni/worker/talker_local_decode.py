from __future__ import annotations

from collections.abc import Mapping

import torch


def talker_state_allows_local_step(state: Mapping[str, object]) -> bool:
    """Return whether the runner may execute one more Talker decode step.

    The runner only enters the local loop after the current step produced no
    sparse wire payload.  It re-checks that condition after every local step,
    so a codec chunk or terminal sample becomes the final scheduler-visible
    result of the loop.  Consequently EOS does not need to remain masked here;
    we only need to reject an already-finished or exhausted request.
    """
    try:
        step = int(state.get("step", 0))
        max_tokens = int(state.get("max_tokens", 0))
    except (TypeError, ValueError):
        return False
    return not bool(state.get("finished")) and step < max_tokens


def refresh_fixed_codec_history_row(
    history_row: torch.Tensor,
    *,
    codes: torch.Tensor | None,
    current: torch.Tensor | None,
    step: int,
    cached_step: int,
    same_request: bool,
) -> str:
    """Refresh one graph-sampler history row without recopying 16 tokens.

    Codec repetition counts are order-independent, so the fixed row can be a
    circular buffer.  During steady decode only the newly sampled scalar is
    copied.  Request compaction, prefill, or a skipped step rebuilds the row
    from the authoritative accumulated codes.
    """
    if history_row.ndim != 1 or history_row.numel() == 0:
        raise ValueError(f"codec history row must be a non-empty vector, got {history_row.shape}")
    if step < 0:
        raise ValueError(f"codec step must be non-negative, got {step}")
    window = int(history_row.numel())
    if same_request and step == cached_step:
        return "unchanged"
    if (
        same_request
        and step == cached_step + 1
        and isinstance(current, torch.Tensor)
        and current.numel() > 0
    ):
        slot = (step - 1) % window
        latest = current.reshape(-1)[-1:].to(
            device=history_row.device,
            dtype=history_row.dtype,
        )
        history_row[slot : slot + 1].copy_(latest)
        return "advanced"

    history_row.fill_(-1)
    if isinstance(codes, torch.Tensor) and codes.numel() > 0:
        recent = codes.reshape(-1)[-window:].to(
            device=history_row.device,
            dtype=history_row.dtype,
        )
        start = step - int(recent.numel())
        slots = torch.arange(
            start,
            step,
            device=history_row.device,
            dtype=torch.long,
        ).remainder(window)
        history_row.scatter_(0, slots, recent)
    return "rebuilt"
