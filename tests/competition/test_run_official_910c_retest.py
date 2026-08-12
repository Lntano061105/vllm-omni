from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/run_official_910c_retest.py"
)
_SPEC = importlib.util.spec_from_file_location("run_official_910c_retest", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_phase_plan_is_symmetric_and_has_final_gates(tmp_path: Path) -> None:
    phases = _MOD.build_phases(result_root=tmp_path, device=0, port=8091)
    names = [phase.name for phase in phases]

    assert names[:2] == ["preflight", "environment"]
    for family in ("performance", "duplex"):
        assert f"{family}-baseline" in names
        assert f"{family}-optimized" in names
        assert f"protocol-{family}" in names
    for suite in ("daily-omni", "videomme", "seed-tts"):
        assert f"accuracy-{suite}-baseline" in names
        assert f"accuracy-{suite}-optimized" in names
        assert f"protocol-accuracy-{suite}" in names
        assert f"gate-accuracy-{suite}" in names
    for concurrency in (1, 4, 8):
        assert f"summary-performance-c{concurrency}" in names
        assert f"gate-duplex-c{concurrency}" in names
    assert names.index("gate-duplex-c8") < names.index("accuracy-daily-omni-baseline")
    for suite in ("daily-omni", "videomme", "seed-tts"):
        assert names.index(f"accuracy-{suite}-optimized") < names.index(
            f"protocol-accuracy-{suite}"
        )
        assert names.index(f"protocol-accuracy-{suite}") < names.index(
            f"gate-accuracy-{suite}"
        )
        if suite != "seed-tts":
            next_suite = ("daily-omni", "videomme", "seed-tts")[("daily-omni", "videomme", "seed-tts").index(suite) + 1]
            assert names.index(f"gate-accuracy-{suite}") < names.index(
                f"accuracy-{next_suite}-baseline"
            )


def test_explicit_paths_propagate_to_accuracy_and_preflight(tmp_path: Path) -> None:
    model = tmp_path / "model"
    daily = tmp_path / "daily"
    video = tmp_path / "video"
    seed = tmp_path / "seed"
    phases = _MOD.build_phases(
        result_root=tmp_path / "results",
        device=0,
        port=8091,
        model_path=model,
        daily_omni_root=daily,
        videomme_root=video,
        seed_tts_root=seed,
        image_digest="quay.io/ascend/vllm-omni@sha256:" + "a" * 64,
    )
    by_name = {phase.name: phase for phase in phases}

    assert str(model) in by_name["preflight"].argv
    assert "quay.io/ascend/vllm-omni@sha256:" + "a" * 64 in by_name["preflight"].argv
    daily_phase = by_name["accuracy-daily-omni-baseline"]
    assert daily_phase.env["MODEL_PATH"] == str(model)
    assert daily_phase.env["DAILY_OMNI_ROOT"] == str(daily)
    assert daily_phase.env["VIDEOMME_ROOT"] == str(video)
    assert daily_phase.env["SEED_TTS_ROOT"] == str(seed)
    environment = by_name["environment"]
    assert environment.env["DAILY_OMNI_ROOT"] == str(daily)
    assert environment.env["VIDEOMME_ROOT"] == str(video)
    assert environment.env["SEED_TTS_ROOT"] == str(seed)


def test_render_is_stable() -> None:
    phase = _MOD.Phase("x", ("tool", "--arg", "a b"), {"Z": "2", "A": "a b"})
    assert _MOD._render(phase) == "A='a b' Z=2 tool --arg 'a b'"


def test_plan_freezes_phase_order_commands_and_hashes(tmp_path: Path) -> None:
    phases = _MOD.build_phases(result_root=tmp_path, device=0, port=8091)
    inputs = {"result_root": str(tmp_path), "device": 0, "port": 8091}
    plan = _MOD.build_plan(phases, inputs=inputs)

    assert plan["phase_count"] == 26
    assert plan["phase_names"] == [phase.name for phase in phases]
    assert plan["inputs"] == inputs
    for phase, frozen in zip(phases, plan["phases"], strict=True):
        command = _MOD._render(phase)
        assert frozen["name"] == phase.name
        assert frozen["command"] == command
        assert frozen["command_sha256"] == __import__("hashlib").sha256(
            command.encode()
        ).hexdigest()


def test_execute_requires_image_digest() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--phase",
            "preflight",
            "--execute",
            "--confirm-single-910c",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode != 0
    assert "--execute requires --image-digest" in completed.stdout
