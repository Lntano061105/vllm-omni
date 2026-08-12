from __future__ import annotations

import pytest
import torch

from vllm_omni.worker.talker_local_decode import (
    refresh_fixed_codec_history_row,
    talker_state_allows_local_step,
)

pytestmark = [pytest.mark.cpu]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"step": 24, "min_tokens": 26, "max_tokens": 26}, True),
        # The next sample may hit the request-local codec limit; it remains the
        # scheduler-visible final local step and is therefore safe.
        ({"step": 25, "min_tokens": 26, "max_tokens": 26}, True),
        ({"step": 26, "min_tokens": 26, "max_tokens": 26}, False),
        # EOS eligibility is safe because a terminal local result exits the
        # loop and is forwarded instead of being hidden.
        ({"step": 8, "min_tokens": 8, "max_tokens": 26}, True),
        ({"step": 7, "min_tokens": 8, "max_tokens": 26}, True),
        # Native duplex boundary units allow EOS immediately and can still use
        # runner-local decode.
        ({"step": 1, "min_tokens": 0, "max_tokens": 26}, True),
        ({"step": 1, "min_tokens": 0, "max_tokens": 26, "finished": True}, False),
        ({"step": "bad", "min_tokens": 8, "max_tokens": 26}, False),
    ],
)
def test_talker_state_allows_scheduler_visible_local_steps(
    state: dict[str, object],
    expected: bool,
) -> None:
    assert talker_state_allows_local_step(state) is expected


def test_fixed_codec_history_uses_single_token_circular_updates() -> None:
    row = torch.full((4,), -1, dtype=torch.long)

    assert refresh_fixed_codec_history_row(
        row,
        codes=torch.tensor([10, 11, 12]),
        current=torch.tensor([12]),
        step=3,
        cached_step=-1,
        same_request=False,
    ) == "rebuilt"
    assert row.tolist() == [10, 11, 12, -1]

    assert refresh_fixed_codec_history_row(
        row,
        codes=torch.tensor([10, 11, 12, 13]),
        current=torch.tensor([13]),
        step=4,
        cached_step=3,
        same_request=True,
    ) == "advanced"
    assert row.tolist() == [10, 11, 12, 13]

    assert refresh_fixed_codec_history_row(
        row,
        codes=torch.tensor([11, 12, 13, 14]),
        current=torch.tensor([14]),
        step=5,
        cached_step=4,
        same_request=True,
    ) == "advanced"
    assert row.tolist() == [14, 11, 12, 13]


def test_fixed_codec_history_rebuilds_after_compaction_or_step_jump() -> None:
    row = torch.tensor([90, 91, 92, 93], dtype=torch.long)
    assert refresh_fixed_codec_history_row(
        row,
        codes=torch.tensor([20, 21]),
        current=torch.tensor([21]),
        step=2,
        cached_step=8,
        same_request=False,
    ) == "rebuilt"
    assert sorted(value for value in row.tolist() if value >= 0) == [20, 21]

    previous = row.clone()
    assert refresh_fixed_codec_history_row(
        row,
        codes=torch.tensor([20, 21]),
        current=torch.tensor([21]),
        step=2,
        cached_step=2,
        same_request=True,
    ) == "unchanged"
    assert torch.equal(row, previous)
