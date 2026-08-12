from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/gate_multiturn_duplex.py"
)
_SPEC = importlib.util.spec_from_file_location("gate_multiturn_duplex", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _run() -> dict[str, object]:
    return {
        "ok": True,
        "done_count": 3,
        "cancelled_count": 0,
        "stale_audio_delta_count": 0,
        "truncate_count": 0,
        "error_count": 0,
        "lifecycle_counts_ok": True,
        "cross_turn_independent_ok": True,
    }


def test_accepts_strict_two_by_three_result() -> None:
    result = _MOD.evaluate({"sessions": 2, "audio_turns": 6, "runs": [_run(), _run()]})
    assert result["passed"] is True
    assert result["failures"] == []


def test_rejects_stale_or_incomplete_turn() -> None:
    bad = _run()
    bad["stale_audio_delta_count"] = 1
    bad["cross_turn_independent_ok"] = False
    result = _MOD.evaluate({"sessions": 2, "audio_turns": 6, "runs": [_run(), bad]})
    assert result["passed"] is False
    assert any("stale_audio_delta_count" in item for item in result["failures"])
    assert any("cross_turn_independent_ok" in item for item in result["failures"])
