from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "competition/minicpmo_b/scripts/analyze_paired_confirmation.py"
_SPEC = importlib.util.spec_from_file_location("analyze_paired_confirmation", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _result(ttft: float, ttfp: float, rtf: float) -> dict[str, object]:
    return {
        "sessions": 32,
        "audio_speak_generation_chunk_count": 64,
        "runs": [{"ok": True} for _ in range(32)],
        "ttft_ms": {"mean": ttft},
        "ttfp_ms": {"mean": ttfp},
        "speak_generation_rtf": {"mean": rtf},
    }


def test_paired_confirmation_passes_repeated_gain() -> None:
    result = _MOD.evaluate(
        _result(2000, 2000, 0.80),
        _result(1600, 1600, 0.60),
        _result(1640, 1640, 0.62),
        _result(2040, 2040, 0.82),
    )
    assert result["passed"] is True, result["failures"]


def test_paired_confirmation_rejects_order_only_gain() -> None:
    result = _MOD.evaluate(
        _result(2000, 2000, 0.80),
        _result(1600, 1600, 0.60),
        _result(2100, 2100, 0.90),
        _result(2000, 2000, 0.80),
    )
    assert result["passed"] is False
    assert any("A2→B2" in failure for failure in result["failures"])


def test_paired_confirmation_rejects_large_repeat_drift() -> None:
    result = _MOD.evaluate(
        _result(2000, 2000, 0.80),
        _result(1600, 1600, 0.60),
        _result(1200, 1200, 0.40),
        _result(2200, 2200, 0.88),
    )
    assert result["passed"] is False
    assert any("repeat drift" in failure for failure in result["failures"])


def test_paired_confirmation_requires_complete_runs() -> None:
    incomplete = _result(1600, 1600, 0.60)
    incomplete["runs"] = [{"ok": True}]
    result = _MOD.evaluate(
        _result(2000, 2000, 0.80),
        incomplete,
        _result(1640, 1640, 0.62),
        _result(2040, 2040, 0.82),
    )
    assert result["passed"] is False
    assert any("runs[]" in failure for failure in result["failures"])
