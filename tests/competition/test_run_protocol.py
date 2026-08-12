from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "competition/minicpmo_b/scripts/run_protocol.py"
_SPEC = importlib.util.spec_from_file_location("run_protocol", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_variants_may_differ_when_comparison_protocol_is_identical() -> None:
    base = _MOD.build_protocol(
        kind="duplex",
        fields={"warmups": 2, "sessions": 32},
        files={"input": {"sha256": "a"}},
        trees={},
        variant_fields={"label": "baseline"},
        variant_files={"deploy": {"sha256": "b"}},
    )
    optimized = _MOD.build_protocol(
        kind="duplex",
        fields={"warmups": 2, "sessions": 32},
        files={"input": {"sha256": "a"}},
        trees={},
        variant_fields={"label": "optimized"},
        variant_files={"deploy": {"sha256": "c"}},
    )

    assert _MOD.compare_protocols(base, optimized)["passed"] is True


def test_comparison_rejects_warmup_or_input_hash_change() -> None:
    base = _MOD.build_protocol(
        kind="performance",
        fields={"warmups": 2},
        files={"dataset": {"sha256": "a"}},
        trees={},
        variant_fields={},
        variant_files={},
    )
    changed = _MOD.build_protocol(
        kind="performance",
        fields={"warmups": 1},
        files={"dataset": {"sha256": "z"}},
        trees={},
        variant_fields={},
        variant_files={},
    )

    result = _MOD.compare_protocols(base, changed)
    assert result["passed"] is False
    assert "A/B comparison protocol differs" in result["failures"]


def test_comparison_rejects_tampered_fingerprint() -> None:
    protocol = _MOD.build_protocol(
        kind="accuracy",
        fields={"suite": "videomme"},
        files={},
        trees={},
        variant_fields={},
        variant_files={},
    )
    tampered = dict(protocol)
    tampered["comparison_sha256"] = "0" * 64

    result = _MOD.compare_protocols(protocol, tampered)
    assert result["passed"] is False
    assert any("comparison_sha256 is invalid" in item for item in result["failures"])


def test_required_fields_reject_jointly_wrong_ab_protocol() -> None:
    protocol = _MOD.build_protocol(
        kind="performance",
        fields={"benchmark_seed": 17, "num_warmups": 2},
        files={},
        trees={},
        variant_fields={},
        variant_files={},
    )

    failures = _MOD.require_fields(
        protocol, {"benchmark_seed": 0, "num_warmups": 2}
    )
    assert failures == ["comparison field 'benchmark_seed' is 17, expected 0"]
