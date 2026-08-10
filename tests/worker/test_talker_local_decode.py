from __future__ import annotations

import pytest

from vllm_omni.worker.talker_local_decode import talker_state_allows_local_step

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
