from __future__ import annotations

from collections.abc import Mapping


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
