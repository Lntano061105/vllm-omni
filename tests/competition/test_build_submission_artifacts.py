from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/build_submission_artifacts.py"
)
_SPEC = importlib.util.spec_from_file_location("build_submission_artifacts", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_sha256_and_atomic_write(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    _MOD._atomic_write(path, b"submission-data")

    assert path.read_bytes() == b"submission-data"
    assert _MOD._sha256(path) == hashlib.sha256(b"submission-data").hexdigest()
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_manifest_paths_cover_repository_wide_supporting_tests() -> None:
    paths = _MOD._manifest_paths(
        "vllm_omni/worker/talker_local_decode.py\n"
        "tests/competition/test_gate.py\n"
        "tests/benchmarks/test_accuracy_bench_utils.py\n"
        "tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py\n"
        "tests/benchmarks/test_accuracy_bench_utils.py\n"
    )

    assert paths == [
        "tests/benchmarks/test_accuracy_bench_utils.py",
        "tests/competition/test_gate.py",
        "tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py",
        "vllm_omni/worker/talker_local_decode.py",
    ]
