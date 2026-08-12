from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
    source = {"git_commit": "a" * 40, "git_tree": "b" * 40}
    plan = _MOD.build_plan(phases, inputs=inputs, source=source)

    assert plan["format_version"] == 2
    assert plan["phase_count"] == 26
    assert plan["phase_names"] == [phase.name for phase in phases]
    assert plan["inputs"] == inputs
    assert plan["source"] == source
    for phase, frozen in zip(phases, plan["phases"], strict=True):
        command = _MOD._render(phase)
        assert frozen["name"] == phase.name
        assert frozen["command"] == command
        assert frozen["command_sha256"] == __import__("hashlib").sha256(
            command.encode()
        ).hexdigest()
    assert _MOD._plan_sha256(plan) == hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _passed_marker(path: Path, phase: _MOD.Phase, plan: dict[str, object]) -> None:
    command = _MOD._render(phase)
    path.write_text(
        json.dumps(
            {
                "phase": phase.name,
                "command": command,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "plan_sha256": _MOD._plan_sha256(plan),
                "source": plan["source"],
                "started_utc": "2026-08-12T00:00:00Z",
                "finished_utc": "2026-08-12T00:00:01Z",
                "returncode": 0,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )


def test_marker_resume_is_bound_to_plan_and_archives_rerun_downstream(
    tmp_path: Path,
) -> None:
    phases = [
        _MOD.Phase("first", ("true",), {}),
        _MOD.Phase("second", ("true", "2"), {}),
        _MOD.Phase("third", ("true", "3"), {}),
    ]
    plan = _MOD.build_plan(
        phases,
        inputs={"result_root": str(tmp_path)},
        source={"git_commit": "a" * 40, "git_tree": "b" * 40},
    )
    state = tmp_path / "orchestrator_state"
    state.mkdir()
    for phase in phases:
        _passed_marker(state / f"{phase.name}.json", phase, plan)
    second_hash = hashlib.sha256(_MOD._render(phases[1]).encode()).hexdigest()
    assert _MOD._marker_passes(
        state / "second.json", second_hash, _MOD._plan_sha256(plan)
    )
    assert not _MOD._marker_passes(
        state / "second.json", second_hash, "0" * 64
    )

    archived = _MOD._archive_invalidated_markers(
        state, phases, first_rerun_index=1
    )
    assert archived == ["second", "third"]
    assert (state / "first.json").is_file()
    assert not (state / "second.json").exists()
    histories = list((state / "history").iterdir())
    assert len(histories) == 1
    assert (histories[0] / "second.json").is_file()
    assert (histories[0] / "third.json").is_file()
    invalidation = json.loads((histories[0] / "invalidation.json").read_text())
    assert invalidation["invalidated_from_phase"] == "second"


def test_existing_plan_drift_is_detectable_without_overwrite(tmp_path: Path) -> None:
    phases = [_MOD.Phase("x", ("true",), {})]
    source = {"git_commit": "a" * 40, "git_tree": "b" * 40}
    frozen = _MOD.build_plan(phases, inputs={"port": 8091}, source=source)
    changed = _MOD.build_plan(phases, inputs={"port": 8092}, source=source)
    path = tmp_path / "orchestrator_plan.json"
    _MOD._freeze_or_validate_plan(path, frozen)
    before = path.read_bytes()
    _MOD._freeze_or_validate_plan(path, frozen)
    with pytest.raises(RuntimeError, match="use a new --result-root"):
        _MOD._freeze_or_validate_plan(path, changed)
    assert path.read_bytes() == before


def test_selected_downstream_phase_requires_bound_predecessor_markers(
    tmp_path: Path,
) -> None:
    phases = [
        _MOD.Phase("preflight", ("true",), {}),
        _MOD.Phase("environment", ("true", "env"), {}),
        _MOD.Phase("performance", ("true", "perf"), {}),
    ]
    plan = _MOD.build_plan(
        phases,
        inputs={"result_root": str(tmp_path)},
        source={"git_commit": "a" * 40, "git_tree": "b" * 40},
    )
    state = tmp_path / "orchestrator_state"
    state.mkdir()
    missing = _MOD._missing_prerequisites(
        selected=[phases[2]],
        all_phases=phases,
        state_dir=state,
        plan_hash=_MOD._plan_sha256(plan),
    )
    assert missing == {"performance": ["preflight", "environment"]}

    _passed_marker(state / "preflight.json", phases[0], plan)
    _passed_marker(state / "environment.json", phases[1], plan)
    assert _MOD._missing_prerequisites(
        selected=[phases[2]],
        all_phases=phases,
        state_dir=state,
        plan_hash=_MOD._plan_sha256(plan),
    ) == {}
    assert _MOD._missing_prerequisites(
        selected=phases,
        all_phases=phases,
        state_dir=tmp_path / "empty-state",
        plan_hash=_MOD._plan_sha256(plan),
    ) == {}


def test_selection_cannot_cross_dependencies_invalidated_by_earlier_rerun() -> None:
    phases = [
        _MOD.Phase("first", ("true",), {}),
        _MOD.Phase("second", ("true", "2"), {}),
        _MOD.Phase("third", ("true", "3"), {}),
        _MOD.Phase("fourth", ("true", "4"), {}),
    ]
    assert _MOD._selection_gaps_after_invalidation(
        selected=[phases[1], phases[3]],
        all_phases=phases,
        first_execution_index=1,
    ) == ["third"]
    assert _MOD._selection_gaps_after_invalidation(
        selected=[phases[1]],
        all_phases=phases,
        first_execution_index=1,
    ) == []
    assert _MOD._selection_gaps_after_invalidation(
        selected=phases[1:],
        all_phases=phases,
        first_execution_index=1,
    ) == []


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
