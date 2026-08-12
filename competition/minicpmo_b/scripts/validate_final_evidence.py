#!/usr/bin/env python3
"""Audit all official-910C, Demo, report, and source evidence before submission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = REPO_ROOT / "competition/minicpmo_b/results/official_910c"
_PROTOCOL_SCRIPT = Path(__file__).with_name("run_protocol.py")
_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "minicpmo_b_run_protocol", _PROTOCOL_SCRIPT
)
if _PROTOCOL_SPEC is None or _PROTOCOL_SPEC.loader is None:
    raise RuntimeError(f"cannot load protocol verifier: {_PROTOCOL_SCRIPT}")
_PROTOCOL = importlib.util.module_from_spec(_PROTOCOL_SPEC)
_PROTOCOL_SPEC.loader.exec_module(_PROTOCOL)
_PAIRED_SCRIPT = Path(__file__).with_name("analyze_paired_confirmation.py")
_PAIRED_SPEC = importlib.util.spec_from_file_location(
    "minicpmo_b_paired_confirmation", _PAIRED_SCRIPT
)
if _PAIRED_SPEC is None or _PAIRED_SPEC.loader is None:
    raise RuntimeError(f"cannot load paired verifier: {_PAIRED_SCRIPT}")
_PAIRED = importlib.util.module_from_spec(_PAIRED_SPEC)
_PAIRED_SPEC.loader.exec_module(_PAIRED)
_DUPLEX_GATE_SCRIPT = Path(__file__).with_name("gate_duplex_candidate.py")
_DUPLEX_GATE_SPEC = importlib.util.spec_from_file_location(
    "minicpmo_b_duplex_gate", _DUPLEX_GATE_SCRIPT
)
if _DUPLEX_GATE_SPEC is None or _DUPLEX_GATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load duplex gate: {_DUPLEX_GATE_SCRIPT}")
_DUPLEX_GATE = importlib.util.module_from_spec(_DUPLEX_GATE_SPEC)
sys.modules[_DUPLEX_GATE_SPEC.name] = _DUPLEX_GATE
_DUPLEX_GATE_SPEC.loader.exec_module(_DUPLEX_GATE)
_ACCURACY_GATE_SCRIPT = Path(__file__).with_name("compare_accuracy_results.py")
_ACCURACY_GATE_SPEC = importlib.util.spec_from_file_location(
    "minicpmo_b_accuracy_gate", _ACCURACY_GATE_SCRIPT
)
if _ACCURACY_GATE_SPEC is None or _ACCURACY_GATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load accuracy gate: {_ACCURACY_GATE_SCRIPT}")
_ACCURACY_GATE = importlib.util.module_from_spec(_ACCURACY_GATE_SPEC)
_ACCURACY_GATE_SPEC.loader.exec_module(_ACCURACY_GATE)
_ORCHESTRATOR_SCRIPT = Path(__file__).with_name("run_official_910c_retest.py")
_ORCHESTRATOR_SPEC = importlib.util.spec_from_file_location(
    "minicpmo_b_official_orchestrator", _ORCHESTRATOR_SCRIPT
)
if _ORCHESTRATOR_SPEC is None or _ORCHESTRATOR_SPEC.loader is None:
    raise RuntimeError(f"cannot load official orchestrator: {_ORCHESTRATOR_SCRIPT}")
_ORCHESTRATOR = importlib.util.module_from_spec(_ORCHESTRATOR_SPEC)
sys.modules[_ORCHESTRATOR_SPEC.name] = _ORCHESTRATOR
_ORCHESTRATOR_SPEC.loader.exec_module(_ORCHESTRATOR)
_FINAL_PACKAGE_SCRIPT = Path(__file__).with_name("build_final_submission.py")
_FINAL_PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "minicpmo_b_final_package", _FINAL_PACKAGE_SCRIPT
)
if _FINAL_PACKAGE_SPEC is None or _FINAL_PACKAGE_SPEC.loader is None:
    raise RuntimeError(f"cannot load final package verifier: {_FINAL_PACKAGE_SCRIPT}")
_FINAL_PACKAGE = importlib.util.module_from_spec(_FINAL_PACKAGE_SPEC)
_FINAL_PACKAGE_SPEC.loader.exec_module(_FINAL_PACKAGE)
REQUIRED_ORCHESTRATOR_PHASES = (
    "preflight",
    "environment",
    "performance-baseline",
    "performance-optimized",
    "duplex-baseline",
    "duplex-optimized",
    "protocol-performance",
    "protocol-duplex",
    "summary-performance-c1",
    "gate-duplex-c1",
    "summary-performance-c4",
    "gate-duplex-c4",
    "summary-performance-c8",
    "gate-duplex-c8",
    "accuracy-daily-omni-baseline",
    "accuracy-daily-omni-optimized",
    "protocol-accuracy-daily-omni",
    "gate-accuracy-daily-omni",
    "accuracy-videomme-baseline",
    "accuracy-videomme-optimized",
    "protocol-accuracy-videomme",
    "gate-accuracy-videomme",
    "accuracy-seed-tts-baseline",
    "accuracy-seed-tts-optimized",
    "protocol-accuracy-seed-tts",
    "gate-accuracy-seed-tts",
)


def _json(path: Path, failures: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        failures.append(f"missing JSON: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        failures.append(f"JSON root is not an object: {path}")
        return None
    return value


def _passed_json(path: Path, failures: list[str]) -> dict[str, Any] | None:
    value = _json(path, failures)
    if value is not None and value.get("passed") is not True:
        failures.append(f"gate did not pass: {path}")
    return value


def _check_protocol_pair(
    *,
    expected_kind: str,
    baseline_path: Path,
    optimized_path: Path,
    gate_path: Path,
    failures: list[str],
    required_fields: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Recompute A/B provenance instead of trusting a secondary gate file."""
    baseline = _json(baseline_path, failures)
    optimized = _json(optimized_path, failures)
    gate = _passed_json(gate_path, failures)
    if baseline is None or optimized is None:
        return None
    result = _PROTOCOL.compare_protocols(baseline, optimized)
    if result.get("kind") != expected_kind:
        failures.append(
            f"A/B protocol kind is {result.get('kind')!r}, expected {expected_kind!r}: "
            f"{gate_path}"
        )
    comparison = baseline.get("comparison")
    fields = comparison.get("fields") if isinstance(comparison, dict) else None
    for key, expected in (required_fields or {}).items():
        actual = fields.get(key) if isinstance(fields, dict) else None
        if actual != expected:
            failures.append(
                f"A/B protocol field {key!r} is {actual!r}, expected {expected!r}: "
                f"{gate_path}"
            )
    if result.get("passed") is not True:
        details = "; ".join(str(item) for item in result.get("failures", []))
        failures.append(f"A/B protocol verification failed for {gate_path}: {details}")
        return None
    if gate is not None:
        for key in ("kind", "comparison_sha256"):
            if gate.get(key) != result.get(key):
                failures.append(
                    f"protocol gate {key} does not match recomputed result: {gate_path}"
                )
    return baseline


def _nonempty(path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        failures.append(f"missing or empty evidence file: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timezone_aware_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value or value == "REPLACE_ME":
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _exists(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing evidence file: {path}")


def _verify_local_sha256_manifest(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, raw_name = line.split(None, 1)
        except ValueError:
            failures.append(f"invalid SHA256 manifest line {path}:{line_number}")
            continue
        target = path.parent / raw_name.strip()
        if not target.is_file():
            failures.append(f"SHA256 manifest target is missing: {target}")
            continue
        actual = _sha256_file(target)
        if actual != expected:
            failures.append(f"SHA256 mismatch: {target}")


def _verify_source_snapshot(
    archive: Path, manifest_path: Path, failures: list[str]
) -> None:
    """Verify every regular file in the Git archive against source_sha256."""
    if not archive.is_file() or not manifest_path.is_file():
        return
    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            digest, raw_name = line.split(None, 1)
        except ValueError:
            failures.append(
                f"invalid source SHA256 manifest line {manifest_path}:{line_number}"
            )
            continue
        name = raw_name.strip()
        if name in expected:
            failures.append(f"duplicate source SHA256 manifest entry: {name}")
        expected[name] = digest
    observed: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar.getmembers():
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts:
                    failures.append(f"unsafe source snapshot path: {member.name}")
                    continue
                if not member.isfile():
                    if not member.isdir():
                        failures.append(
                            f"non-regular source snapshot member: {member.name}"
                        )
                    continue
                if member.name in observed:
                    failures.append(f"duplicate source snapshot member: {member.name}")
                    continue
                observed.add(member.name)
                extracted = tar.extractfile(member)
                if extracted is None:
                    failures.append(f"cannot read source snapshot member: {member.name}")
                    continue
                digest = hashlib.sha256(extracted.read()).hexdigest()
                if expected.get(member.name) != digest:
                    failures.append(
                        "source snapshot SHA256 mismatch or unmanifested member: "
                        + member.name
                    )
    except (OSError, tarfile.TarError) as exc:
        failures.append(f"cannot read source snapshot {archive}: {exc}")
        return
    missing = sorted(set(expected) - observed)
    if missing:
        failures.append(
            "source snapshot is missing manifest entries: " + ", ".join(missing[:20])
        )


def _check_environment(root: Path, failures: list[str]) -> None:
    env = root / "environment"
    preflight = _passed_json(env / "preflight.json", failures)
    if preflight is not None:
        if preflight.get("single_910c") is not True:
            failures.append("preflight does not prove a single visible 910C")
        selected = str(preflight.get("selected_device", 0))
        if preflight.get("npu_processes", {}).get(selected):
            failures.append("preflight selected NPU was occupied")
        assets = preflight.get("asset_preflight")
        if not isinstance(assets, dict) or assets.get("passed") is not True:
            failures.append("preflight does not prove all model/data/evaluator assets")
        probe = preflight.get("token2wav_asset_probe")
        if not isinstance(probe, dict) or probe.get("passed") is not True:
            failures.append("preflight does not prove S3Tokenizer/campplus compatibility")
    for name in (
        "timestamp_utc.txt",
        "uname.txt",
        "python_version.txt",
        "python_packages.txt",
        "npu_smi.txt",
        "container_image_digest.txt",
        "cann_version.txt",
        "git_commit.txt",
        "model_manifest.tsv",
        "model_metadata_sha256.txt",
        "model_weights_sha256.txt",
        "seed_tts_eval_model_manifest.tsv",
        "seed_tts_eval_model_sha256.txt",
        "dataset_manifest.tsv",
        "dataset_metadata_sha256.txt",
    ):
        _nonempty(env / name, failures)
    _exists(env / "git_status.txt", failures)
    for name in ("model_manifest.tsv", "seed_tts_eval_model_manifest.tsv", "dataset_manifest.tsv"):
        path = env / name
        if path.is_file() and "MISSING" in path.read_text(encoding="utf-8", errors="replace"):
            failures.append(f"environment manifest contains MISSING entry: {path}")
    digest_path = env / "container_image_digest.txt"
    if digest_path.is_file() and not re.search(
        r"@sha256:[0-9a-fA-F]{64}$", digest_path.read_text(encoding="utf-8").strip()
    ):
        failures.append("environment container image digest is not immutable sha256 form")
    cann_path = env / "cann_version.txt"
    if cann_path.is_file() and "not found" in cann_path.read_text(
        encoding="utf-8", errors="replace"
    ).lower():
        failures.append("environment CANN version was not captured")


def _check_orchestrator(root: Path, failures: list[str]) -> None:
    state = root / "orchestrator_state"
    plan = _json(state / "orchestrator_plan.json", failures)
    expected_commands: dict[str, str] = {}
    if plan is not None:
        if plan.get("format_version") != 2:
            failures.append("orchestrator plan format_version is not 2")
        source = plan.get("source")
        if not isinstance(source, dict):
            failures.append("orchestrator plan source identity is missing")
            source = {}
        git_commit = source.get("git_commit")
        git_tree = source.get("git_tree")
        if not isinstance(git_commit, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40,64}", git_commit
        ):
            failures.append("orchestrator plan git_commit is invalid")
        if not isinstance(git_tree, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40,64}", git_tree
        ):
            failures.append("orchestrator plan git_tree is invalid")
        env_commit_path = root / "environment/git_commit.txt"
        if env_commit_path.is_file() and isinstance(git_commit, str):
            env_commit = env_commit_path.read_text(encoding="utf-8").strip()
            if env_commit != git_commit:
                failures.append(
                    "orchestrator plan commit differs from environment git_commit.txt"
                )
        source_commit_path = root / "final_submission/source/git_commit.txt"
        if source_commit_path.is_file() and isinstance(git_commit, str):
            source_commit = source_commit_path.read_text(encoding="utf-8").strip()
            if source_commit != git_commit:
                failures.append(
                    "orchestrator plan commit differs from source artifact git_commit.txt"
                )
        source_tree_path = root / "final_submission/source/git_tree.txt"
        if source_tree_path.is_file() and isinstance(git_tree, str):
            source_tree = source_tree_path.read_text(encoding="utf-8").strip()
            if source_tree != git_tree:
                failures.append(
                    "orchestrator plan tree differs from source artifact git_tree.txt"
                )
        inputs = plan.get("inputs")
        try:
            if not isinstance(inputs, dict):
                raise TypeError("inputs is not an object")
            rebuilt = _ORCHESTRATOR.build_phases(
                result_root=Path(str(inputs["result_root"])),
                device=int(inputs["device"]),
                port=int(inputs["port"]),
                model_path=Path(str(inputs["model_path"])),
                daily_omni_root=Path(str(inputs["daily_omni_root"])),
                videomme_root=Path(str(inputs["videomme_root"])),
                seed_tts_root=Path(str(inputs["seed_tts_root"])),
                whisper_model=Path(str(inputs["whisper_model"])),
                wavlm_model=Path(str(inputs["wavlm_model"])),
                utmos_model=Path(str(inputs["utmos_model"])),
                image_digest=str(inputs["image_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"orchestrator plan inputs cannot rebuild phases: {exc}")
            rebuilt = []
        rebuilt_names = [phase.name for phase in rebuilt]
        if rebuilt_names != list(REQUIRED_ORCHESTRATOR_PHASES):
            failures.append("orchestrator rebuilt phase order/count differs from required 26 phases")
        expected_plan = (
            _ORCHESTRATOR.build_plan(rebuilt, inputs=inputs, source=source)
            if rebuilt
            else None
        )
        if expected_plan is not None and plan != expected_plan:
            failures.append("orchestrator_plan.json differs from current canonical build_phases output")
        expected_commands = {phase.name: _ORCHESTRATOR._render(phase) for phase in rebuilt}
    plan_hash = _ORCHESTRATOR._plan_sha256(plan) if plan is not None else None
    for phase in REQUIRED_ORCHESTRATOR_PHASES:
        marker = _json(state / f"{phase}.json", failures)
        if marker is None:
            continue
        if marker.get("phase") != phase:
            failures.append(f"orchestrator marker phase mismatch: {marker}")
        if marker.get("passed") is not True or marker.get("returncode") != 0:
            failures.append(f"orchestrator phase did not pass: {phase}")
        command = marker.get("command")
        digest = marker.get("command_sha256")
        if not isinstance(command, str) or not command:
            failures.append(f"orchestrator phase has no command: {phase}")
        elif not isinstance(digest, str) or digest != hashlib.sha256(command.encode()).hexdigest():
            failures.append(f"orchestrator command hash mismatch: {phase}")
        elif expected_commands and command != expected_commands.get(phase):
            failures.append(f"orchestrator marker command differs from canonical plan: {phase}")
        if plan_hash is not None and marker.get("plan_sha256") != plan_hash:
            failures.append(f"orchestrator marker plan hash mismatch: {phase}")
        if plan is not None and marker.get("source") != plan.get("source"):
            failures.append(f"orchestrator marker source identity mismatch: {phase}")
        started = _timezone_aware_datetime(marker.get("started_utc"))
        finished = _timezone_aware_datetime(marker.get("finished_utc"))
        if started is None or finished is None:
            failures.append(f"orchestrator marker timestamps are invalid: {phase}")
        elif finished <= started:
            failures.append(f"orchestrator marker finished before start: {phase}")


def _check_performance(root: Path, failures: list[str]) -> None:
    baseline = root / "official_910c_baseline"
    optimized = root / "official_910c_optimized"
    _check_protocol_pair(
        expected_kind="chat-completions-performance-matrix",
        baseline_path=baseline / "performance/run_protocol.json",
        optimized_path=optimized / "performance/run_protocol.json",
        gate_path=root / "protocol_gate_performance.json",
        failures=failures,
        required_fields={
            "benchmark_seed": 0,
            "num_warmups": 2,
            "no_oversample": True,
            "request_rate": "inf",
            "disable_shuffle": False,
            "c1_prompts": 32,
            "c4_prompts": 64,
            "c8_prompts": 128,
        },
    )
    _check_protocol_pair(
        expected_kind="realtime-duplex-speak-generation-matrix",
        baseline_path=baseline / "duplex_rtf/run_protocol.json",
        optimized_path=optimized / "duplex_rtf/run_protocol.json",
        gate_path=root / "protocol_gate_duplex.json",
        failures=failures,
        required_fields={
            "num_warmups": 2,
            "turns_per_session": 1,
            "input_chunk_ms": 200,
            "turn_duration_ms": 0,
            "c1_prompts": 32,
            "c4_prompts": 64,
            "c8_prompts": 128,
        },
    )
    for concurrency, prompts in ((1, 32), (4, 64), (8, 128)):
        suffix = f"c{concurrency}_n{prompts}"
        duplex_results: dict[str, dict[str, Any]] = {}
        for label, case_root in (("baseline", baseline), ("optimized", optimized)):
            _nonempty(case_root / "performance" / "server.log", failures)
            perf = _json(case_root / "performance" / f"seed_tts_{suffix}.json", failures)
            if perf is not None:
                completed = int(perf.get("completed", perf.get("successful", 0)) or 0)
                failed = int(perf.get("failed", 0) or 0)
                if completed != prompts or failed != 0:
                    failures.append(
                        f"{label} performance {suffix} incomplete: completed={completed}, failed={failed}"
                    )
                for metric in ("mean_ttft_ms", "mean_audio_ttfp_ms"):
                    if not isinstance(perf.get(metric), (int, float)):
                        failures.append(f"{label} performance {suffix} missing {metric}")
            duplex = _json(
                case_root / "duplex_rtf" / f"native_duplex_{suffix}.json", failures
            )
            if duplex is not None:
                duplex_results[label] = duplex
                if int(duplex.get("sessions", 0) or 0) != prompts:
                    failures.append(f"{label} duplex {suffix} session count mismatch")
                if int(duplex.get("audio_speak_generation_chunk_count", 0) or 0) <= 0:
                    failures.append(f"{label} duplex {suffix} has no SPEAK-generation chunks")
                for metric in ("ttft_ms", "ttfp_ms", "speak_generation_rtf"):
                    summary = duplex.get(metric)
                    if not isinstance(summary, dict) or not isinstance(summary.get("mean"), (int, float)):
                        failures.append(f"{label} duplex {suffix} missing {metric}.mean")
            _nonempty(case_root / "duplex_rtf" / "server.log", failures)
        gate_path = optimized / "duplex_rtf" / f"gate_c{concurrency}.json"
        saved_gate = _passed_json(gate_path, failures)
        if set(duplex_results) == {"baseline", "optimized"}:
            try:
                recomputed = _DUPLEX_GATE.evaluate_candidate(
                    duplex_results["baseline"],
                    duplex_results["optimized"],
                    _DUPLEX_GATE.PROFILES["combined"],
                )
            except (TypeError, ValueError) as exc:
                failures.append(f"cannot recompute duplex gate c{concurrency}: {exc}")
            else:
                if recomputed.get("passed") is not True:
                    failures.append(
                        f"recomputed duplex gate c{concurrency} failed: "
                        + "; ".join(str(item) for item in recomputed.get("failures", []))
                    )
                if saved_gate is not None:
                    for key in ("metrics", "baseline", "candidate", "failures"):
                        if saved_gate.get(key) != recomputed.get(key):
                            failures.append(
                                f"saved duplex gate c{concurrency} {key} differs from recomputation"
                            )
        _nonempty(root / f"performance_{suffix}.md", failures)

    _passed_json(optimized / "duplex_rtf" / "activation_gate.json", failures)
    multiturn_root = optimized / "duplex_rtf" / "multiturn_s2_t3"
    _passed_json(multiturn_root / "multiturn_gate.json", failures)
    multiturn = _json(multiturn_root / "native_duplex_rtf.json", failures)
    if multiturn is not None:
        if multiturn.get("sessions") != 2 or multiturn.get("audio_turns") != 6:
            failures.append("optimized multi-turn raw result is not 2 sessions x 3 turns")
        runs = multiturn.get("runs")
        if not isinstance(runs, list) or len(runs) != 2 or not all(
            isinstance(run, dict) and run.get("ok") is True for run in runs
        ):
            failures.append("optimized multi-turn raw runs are incomplete")


def _check_accuracy(root: Path, failures: list[str]) -> None:
    expected_absolute = {
        "daily-omni": ("daily_omni_accuracy", "higher", 0.78),
        "videomme": ("videomme_accuracy", "higher", 0.68),
        "seed-tts": ("seed_tts_content_error_mean", "lower", 0.05),
    }
    for suite in ("daily-omni", "videomme", "seed-tts"):
        suite_required_fields: dict[str, Any] = {
            "benchmark_seed": 0,
            "num_warmups": 0,
            "no_oversample": True,
            "request_rate": "inf",
            "temperature": 0,
        }
        if suite == "daily-omni":
            suite_required_fields.update(
                disable_shuffle=False,
                input_mode="all",
                pack_mode="minicpm-interleave",
                output_len=512,
            )
        elif suite == "videomme":
            suite_required_fields.update(
                num_prompts=2700,
                disable_shuffle=True,
                pack_mode="minicpm-frames",
                max_frames=96,
                duration="all",
                output_len=128,
            )
        else:
            suite_required_fields.update(
                num_prompts=1000,
                disable_shuffle=False,
                locale="en",
                turns_per_session=1,
                sim_eval=1,
                utmos_eval=1,
            )
        protocol = _check_protocol_pair(
            expected_kind=f"accuracy-{suite}",
            baseline_path=(
                root / f"official_910c_baseline/accuracy/{suite}/run_protocol.json"
            ),
            optimized_path=(
                root / f"official_910c_optimized/accuracy/{suite}/run_protocol.json"
            ),
            gate_path=root / f"protocol_gate_accuracy_{suite}.json",
            failures=failures,
            required_fields=suite_required_fields,
        )
        protocol_fields = (
            protocol.get("comparison", {}).get("fields", {})
            if isinstance(protocol, dict)
            else {}
        )
        expected_requests = protocol_fields.get("num_prompts")
        if not isinstance(expected_requests, int) or expected_requests <= 0:
            failures.append(f"{suite} protocol has no positive integer num_prompts")
            expected_requests = None
        protocol_trees = (
            protocol.get("comparison", {}).get("trees", {})
            if isinstance(protocol, dict)
            else {}
        )
        protocol_files = (
            protocol.get("comparison", {}).get("files", {})
            if isinstance(protocol, dict)
            else {}
        )
        if suite == "seed-tts":
            for key in ("whisper_model", "wavlm_model"):
                if key not in protocol_trees:
                    failures.append(f"Seed-TTS protocol is missing evaluator tree: {key}")
            if "utmos_model" not in protocol_files:
                failures.append("Seed-TTS protocol is missing evaluator file: utmos_model")
        gate_path = root / f"accuracy_gate_{suite}.json"
        gate = _passed_json(gate_path, failures)
        if gate is not None and gate.get("max_regression") != 0.02:
            failures.append(f"{suite} gate does not use the required 2pp threshold")
        if gate is not None:
            metric, direction, threshold = expected_absolute[suite]
            absolute = gate.get("absolute_gate")
            if absolute != {"metric": metric, "direction": direction, "threshold": threshold}:
                failures.append(f"{suite} gate does not prove the absolute accuracy threshold")
        accuracy_results: dict[str, dict[str, Any]] = {}
        for label in ("baseline", "optimized"):
            case = root / f"official_910c_{label}" / "accuracy" / suite
            _passed_json(case / "activation_gate.json", failures)
            matches = sorted(case.glob("seed_tts_quality_resumed*.json")) or sorted(
                case.glob("qwen_omni_acc_*.json")
            ) + sorted(case.glob("omni_acc_videomme_*.json"))
            if len(matches) != 1:
                failures.append(
                    f"expected exactly one final {suite} result for {label}, found {len(matches)} under {case}"
                )
            else:
                result = _json(matches[0], failures)
                if result is None:
                    continue
                accuracy_results[label] = result
                if suite == "seed-tts":
                    if result.get("seed_tts_quality_complete") is not True:
                        failures.append(f"{label} Seed-TTS quality result is incomplete")
                    if expected_requests is not None:
                        for key in (
                            "seed_tts_session_count",
                            "seed_tts_turn_count",
                            "seed_tts_content_evaluated",
                            "seed_tts_sim_evaluated",
                            "seed_tts_utmos_evaluated",
                        ):
                            if int(result.get(key, 0) or 0) != expected_requests:
                                failures.append(
                                    f"{label} Seed-TTS {key}={result.get(key)!r}, "
                                    f"expected={expected_requests} from protocol"
                                )
                else:
                    prefix = "daily_omni" if suite == "daily-omni" else "videomme"
                    completed = int(result.get("completed", 0) or 0)
                    evaluated = int(result.get(f"{prefix}_evaluated", 0) or 0)
                    evaluated_ok = int(result.get(f"{prefix}_evaluated_ok", 0) or 0)
                    expected = expected_requests
                    if expected is None:
                        continue
                    if expected <= 0 or completed != expected:
                        failures.append(
                            f"{label} {suite} completed={completed}, expected={expected}"
                        )
                    if evaluated != expected or evaluated_ok != expected:
                        failures.append(
                            f"{label} {suite} evaluated={evaluated}, "
                            f"evaluated_ok={evaluated_ok}, expected={expected}"
                        )
                    for key in (
                        "failed",
                        f"{prefix}_request_failed",
                        f"{prefix}_parse_failed",
                    ):
                        if int(result.get(key, 0) or 0) != 0:
                            failures.append(f"{label} {suite} {key} is not zero")
        if set(accuracy_results) == {"baseline", "optimized"} and expected_requests is not None:
            recomputed = _ACCURACY_GATE.compare_accuracy(
                suite,
                accuracy_results["baseline"],
                accuracy_results["optimized"],
                max_regression=0.02,
                expected_requests=expected_requests,
            )
            if recomputed.get("passed") is not True:
                failures.append(
                    f"recomputed accuracy gate {suite} failed: "
                    + "; ".join(str(item) for item in recomputed.get("failures", []))
                )
            if gate is not None:
                for key in (
                    "suite",
                    "metrics",
                    "max_regression",
                    "expected_requests",
                    "absolute_gate",
                    "failures",
                ):
                    if gate.get(key) != recomputed.get(key):
                        failures.append(
                            f"saved accuracy gate {suite} {key} differs from recomputation"
                        )


def _check_demo(demo_root: Path, failures: list[str]) -> None:
    manifest = _json(demo_root / "demo_evidence.json", failures)
    if manifest is None:
        return
    if manifest.get("official_910c") is not True:
        failures.append("Demo evidence is not marked official_910c=true")
    for key in (
        "service_exit_clean",
    ):
        if manifest.get(key) is not True:
            failures.append(f"Demo {key} is not true")
    for key in (
        "unexpected_error_count",
        "audio_interruption_count",
        "empty_audio_packet_count",
        "audio_underrun_count",
    ):
        if manifest.get(key) != 0:
            failures.append(f"Demo {key}={manifest.get(key)!r}, expected 0")
    timestamps: dict[str, datetime] = {}
    for key in ("started_utc", "finished_utc"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value or value == "REPLACE_ME":
            failures.append(f"Demo {key} is not populated")
        else:
            parsed = _timezone_aware_datetime(value)
            if parsed is not None:
                timestamps[key] = parsed
            else:
                failures.append(
                    f"Demo {key} is not a valid timezone-aware ISO-8601 timestamp"
                )
    continuous_minutes = manifest.get("continuous_run_minutes")
    if (
        isinstance(continuous_minutes, bool)
        or not isinstance(continuous_minutes, (int, float))
        or continuous_minutes <= 0
    ):
        failures.append("Demo continuous_run_minutes must be positive")
    elif set(timestamps) == {"started_utc", "finished_utc"}:
        elapsed_minutes = (
            timestamps["finished_utc"] - timestamps["started_utc"]
        ).total_seconds() / 60
        if elapsed_minutes <= 0:
            failures.append("Demo finished_utc must be after started_utc")
        elif continuous_minutes > elapsed_minutes + 0.1:
            failures.append(
                "Demo continuous_run_minutes exceeds timestamp elapsed duration"
            )
    raw_metadata = manifest.get("metadata_file")
    if not isinstance(raw_metadata, str) or raw_metadata.startswith("REPLACE_"):
        failures.append("Demo metadata_file is not populated")
    else:
        raw_metadata_path = demo_root / raw_metadata
        metadata_path = raw_metadata_path.resolve()
        try:
            metadata_path.relative_to(demo_root.resolve())
        except ValueError:
            failures.append("Demo metadata_file escapes demo root")
        else:
            if raw_metadata_path.is_symlink():
                failures.append("Demo metadata_file is a symlink")
            _nonempty(metadata_path, failures)
            expected_metadata_sha = manifest.get("metadata_sha256")
            if not isinstance(expected_metadata_sha, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", expected_metadata_sha
            ):
                failures.append("Demo metadata_sha256 is not a valid SHA256")
            elif metadata_path.is_file() and _sha256_file(
                metadata_path
            ) != expected_metadata_sha.lower():
                failures.append("Demo metadata_file SHA256 mismatch")
            metadata = _json(metadata_path, failures)
            if metadata is not None:
                for key in (
                    "official_910c",
                    "demo_name",
                    "started_utc",
                    "finished_utc",
                    "continuous_run_minutes",
                    "service_exit_clean",
                    "unexpected_error_count",
                    "audio_interruption_count",
                    "empty_audio_packet_count",
                    "audio_underrun_count",
                ):
                    if metadata.get(key) != manifest.get(key):
                        failures.append(
                            f"Demo metadata {key} differs from generated manifest"
                        )
    scenarios = manifest.get("scenarios")
    scenario_evidence_paths: set[Path] = set()
    for name in ("text", "audio", "video", "text_audio"):
        if not isinstance(scenarios, dict) or not isinstance(scenarios.get(name), dict):
            failures.append(f"Demo scenario is missing: {name}")
            continue
        scenario = scenarios[name]
        if scenario.get("passed") is not True:
            failures.append(f"Demo scenario did not pass: {name}")
        request_count = scenario.get("request_count")
        completed = scenario.get("completed_response_count")
        if (
            isinstance(request_count, bool)
            or not isinstance(request_count, int)
            or request_count <= 0
        ):
            failures.append(f"Demo scenario {name} request_count must be positive")
        elif (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed != request_count
        ):
            failures.append(
                f"Demo scenario {name} completed_response_count={completed!r}, "
                f"expected request_count={request_count}"
            )
        if name != "text":
            packets = scenario.get("audio_packet_count")
            if (
                isinstance(packets, bool)
                or not isinstance(packets, int)
                or packets <= 0
            ):
                failures.append(
                    f"Demo scenario {name} audio_packet_count must be positive"
                )
        raw_evidence = scenario.get("evidence_file")
        if not isinstance(raw_evidence, str) or raw_evidence.startswith("REPLACE_"):
            failures.append(f"Demo scenario {name} evidence_file is not populated")
        else:
            raw_evidence_path = demo_root / raw_evidence
            evidence_path = raw_evidence_path.resolve()
            try:
                evidence_path.relative_to(demo_root.resolve())
            except ValueError:
                failures.append(
                    f"Demo scenario {name} evidence_file escapes demo root"
                )
            else:
                if raw_evidence_path.is_symlink():
                    failures.append(
                        f"Demo scenario {name} evidence_file is a symlink"
                    )
                if evidence_path in scenario_evidence_paths:
                    failures.append(
                        f"Demo scenario {name} evidence_file is not independent"
                    )
                scenario_evidence_paths.add(evidence_path)
                _nonempty(evidence_path, failures)
                expected_evidence_sha = scenario.get("evidence_sha256")
                if not isinstance(expected_evidence_sha, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", expected_evidence_sha
                ):
                    failures.append(
                        f"Demo scenario {name} evidence_sha256 is not a valid SHA256"
                    )
                elif evidence_path.is_file() and _sha256_file(
                    evidence_path
                ) != expected_evidence_sha.lower():
                    failures.append(
                        f"Demo scenario {name} evidence_file SHA256 mismatch"
                    )
                evidence = _json(evidence_path, failures)
                if evidence is not None:
                    if evidence.get("scenario") != name:
                        failures.append(
                            f"Demo scenario {name} evidence declares "
                            f"scenario={evidence.get('scenario')!r}"
                        )
                    requests = evidence.get("requests")
                    if not isinstance(requests, list) or not requests:
                        failures.append(
                            f"Demo scenario {name} evidence requests must be non-empty"
                        )
                    else:
                        completed_from_evidence = 0
                        packets_from_evidence = 0
                        request_ids: set[str] = set()
                        for index, request in enumerate(requests):
                            prefix = f"Demo scenario {name} request[{index}]"
                            if not isinstance(request, dict):
                                failures.append(f"{prefix} is not an object")
                                continue
                            request_id = request.get("request_id")
                            if not isinstance(request_id, str) or not request_id:
                                failures.append(f"{prefix} request_id is not populated")
                            elif request_id in request_ids:
                                failures.append(f"{prefix} request_id is duplicated")
                            else:
                                request_ids.add(request_id)
                            request_started = _timezone_aware_datetime(
                                request.get("started_utc")
                            )
                            request_finished = _timezone_aware_datetime(
                                request.get("finished_utc")
                            )
                            if request_started is None or request_finished is None:
                                failures.append(
                                    f"{prefix} timestamps must be timezone-aware ISO-8601"
                                )
                            elif request_finished <= request_started:
                                failures.append(
                                    f"{prefix} finished_utc must be after started_utc"
                                )
                            elif set(timestamps) == {"started_utc", "finished_utc"} and (
                                request_started < timestamps["started_utc"]
                                or request_finished > timestamps["finished_utc"]
                            ):
                                failures.append(
                                    f"{prefix} timestamps fall outside Demo run interval"
                                )
                            if request.get("completed") is True:
                                completed_from_evidence += 1
                            else:
                                failures.append(f"{prefix} did not complete")
                            packet_count = request.get("audio_packet_count", 0)
                            if name != "text" and (
                                isinstance(packet_count, bool)
                                or not isinstance(packet_count, int)
                                or packet_count <= 0
                            ):
                                failures.append(
                                    f"{prefix} audio_packet_count must be positive"
                                )
                            elif isinstance(packet_count, int) and not isinstance(
                                packet_count, bool
                            ):
                                packets_from_evidence += packet_count
                            raw_output = request.get("output_file")
                            if not isinstance(raw_output, str) or raw_output.startswith(
                                "REPLACE_"
                            ):
                                failures.append(f"{prefix} output_file is not populated")
                            else:
                                raw_output_path = demo_root / raw_output
                                output_path = raw_output_path.resolve()
                                try:
                                    output_path.relative_to(demo_root.resolve())
                                except ValueError:
                                    failures.append(
                                        f"{prefix} output_file escapes demo root"
                                    )
                                else:
                                    if raw_output_path.is_symlink():
                                        failures.append(
                                            f"{prefix} output_file is a symlink"
                                        )
                                    _nonempty(output_path, failures)
                                    expected_output_sha = request.get("output_sha256")
                                    if not isinstance(
                                        expected_output_sha, str
                                    ) or not re.fullmatch(
                                        r"[0-9a-fA-F]{64}", expected_output_sha
                                    ):
                                        failures.append(
                                            f"{prefix} output_sha256 is not a valid SHA256"
                                        )
                                    elif output_path.is_file() and _sha256_file(
                                        output_path
                                    ) != expected_output_sha.lower():
                                        failures.append(
                                            f"{prefix} output_file SHA256 mismatch"
                                        )
                        if request_count != len(requests):
                            failures.append(
                                f"Demo scenario {name} request_count={request_count!r}, "
                                f"evidence has {len(requests)} request(s)"
                            )
                        if completed != completed_from_evidence:
                            failures.append(
                                f"Demo scenario {name} completed_response_count={completed!r}, "
                                f"evidence has {completed_from_evidence} completed request(s)"
                            )
                        if name != "text" and packets != packets_from_evidence:
                            failures.append(
                                f"Demo scenario {name} audio_packet_count={packets!r}, "
                                f"evidence has {packets_from_evidence} packet(s)"
                            )
    for key in ("service_log", "video_file"):
        raw = manifest.get(key)
        if not isinstance(raw, str) or not raw or raw == "REPLACE_WITH_PATH_RELATIVE_TO_DEMO_ROOT":
            failures.append(f"Demo {key} is not populated")
        else:
            raw_path = demo_root / raw
            path = raw_path.resolve()
            try:
                path.relative_to(demo_root.resolve())
            except ValueError:
                failures.append(f"Demo {key} escapes demo root: {raw}")
            else:
                if raw_path.is_symlink():
                    failures.append(f"Demo {key} is a symlink")
                _nonempty(path, failures)
                sha_key = (
                    "service_log_sha256" if key == "service_log" else "video_sha256"
                )
                expected_sha = manifest.get(sha_key)
                if not isinstance(expected_sha, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", expected_sha
                ):
                    failures.append(f"Demo {sha_key} is not a valid SHA256")
                elif path.is_file() and _sha256_file(path) != expected_sha.lower():
                    failures.append(f"Demo {key} SHA256 mismatch")
                if key == "video_file" and path.is_file():
                    with path.open("rb") as handle:
                        header = handle.read(32)
                    if b"ftyp" not in header and not header.startswith(b"\x1aE\xdf\xa3"):
                        failures.append(
                            "Demo video_file is not recognizable MP4/WebM media"
                        )
                if key == "service_log" and path.is_file():
                    log_text = path.read_text(encoding="utf-8", errors="replace")
                    fatal_patterns = (
                        r"(?m)^.*Traceback \(most recent call last\):",
                        r"(?m)^.*\bERROR\b",
                        r"ERR99999",
                        r"Segmentation fault",
                        r"Engine core initialization failed",
                    )
                    if any(re.search(pattern, log_text) for pattern in fatal_patterns):
                        failures.append("Demo service_log contains fatal error markers")


def audit_demo(demo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    _check_demo(demo_root.expanduser().resolve(), failures)
    return {
        "passed": not failures,
        "demo_root": str(demo_root.expanduser().resolve()),
        "failure_count": len(failures),
        "failures": failures,
    }


def _check_source(source_root: Path, failures: list[str]) -> None:
    for name in (
        "source_snapshot.tar.gz",
        "optimization.patch",
        "git_commit.txt",
        "git_base_commit.txt",
        "git_branch.txt",
        "git_tree.txt",
        "changed_files.txt",
        "source_sha256.txt",
        "artifact_metadata.json",
        "artifact_sha256.txt",
        "submission_package_audit.json",
    ):
        _nonempty(source_root / name, failures)
    _exists(source_root / "git_status.txt", failures)
    metadata = _json(source_root / "artifact_metadata.json", failures)
    if metadata is not None:
        if metadata.get("final_candidate") is not True or metadata.get("dirty_worktree") is not False:
            failures.append("source artifact metadata is not a clean final candidate")
        for key, filename in (
            ("head_commit", "git_commit.txt"),
            ("base_commit", "git_base_commit.txt"),
            ("branch", "git_branch.txt"),
        ):
            path = source_root / filename
            if path.is_file() and metadata.get(key) != path.read_text(
                encoding="utf-8"
            ).strip():
                failures.append(f"source metadata {key} does not match {filename}")
        manifest_path = source_root / "source_sha256.txt"
        if manifest_path.is_file():
            manifest_count = sum(
                1
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line
            )
            if metadata.get("source_manifest_file_count") != manifest_count:
                failures.append(
                    "source metadata manifest count does not match source_sha256.txt"
                )
    audit = _passed_json(source_root / "submission_package_audit.json", failures)
    if audit is not None and audit.get("warnings"):
        failures.append("source submission package audit contains warnings")
    _verify_local_sha256_manifest(source_root / "artifact_sha256.txt", failures)
    _verify_source_snapshot(
        source_root / "source_snapshot.tar.gz",
        source_root / "source_sha256.txt",
        failures,
    )
    environment_commit = source_root.parents[1] / "environment/git_commit.txt"
    source_commit = source_root / "git_commit.txt"
    if environment_commit.is_file() and source_commit.is_file():
        if environment_commit.read_text(encoding="utf-8").strip() != source_commit.read_text(
            encoding="utf-8"
        ).strip():
            failures.append(
                "official environment git commit does not match source artifact commit"
            )


def _check_package(package_root: Path, failures: list[str]) -> None:
    archive = package_root / "minicpmo_b_official_910c.tar.gz"
    verification = _passed_json(package_root / "archive_verification.json", failures)
    _nonempty(archive, failures)
    _nonempty(package_root / "archive_sha256.txt", failures)
    recomputed: dict[str, Any] | None = None
    if archive.is_file():
        try:
            authoritative_files = _FINAL_PACKAGE.collect_evidence_files(
                package_root.parents[1], package_root
            )
        except ValueError as exc:
            failures.append(f"cannot collect authoritative package evidence: {exc}")
            authoritative_files = None
        recomputed = _FINAL_PACKAGE.verify_archive(
            archive, expected_files=authoritative_files
        )
        if recomputed.get("passed") is not True:
            failures.append(
                "final submission archive internal verification failed: "
                + "; ".join(str(item) for item in recomputed.get("failures", []))
            )
    if verification is not None:
        actual = _sha256_file(archive) if archive.is_file() else None
        if actual != verification.get("archive_sha256"):
            failures.append("final submission archive SHA256 does not match verification JSON")
        if not isinstance(verification.get("evidence_file_count"), int) or verification[
            "evidence_file_count"
        ] <= 0:
            failures.append("final submission archive contains no evidence files")
        if recomputed is not None:
            for key in (
                "archive",
                "passed",
                "archive_sha256",
                "evidence_file_count",
                "failures",
            ):
                if verification.get(key) != recomputed.get(key):
                    failures.append(
                        f"saved archive verification {key} differs from recomputation"
                    )
    manifest = package_root / "archive_sha256.txt"
    if manifest.is_file() and archive.is_file():
        expected_line = f"{_sha256_file(archive)}  {archive.name}"
        if manifest.read_text(encoding="utf-8").strip() != expected_line:
            failures.append("archive_sha256.txt does not match final submission archive")


def _check_optional_paired_confirmation(root: Path, failures: list[str]) -> None:
    paired = root / "paired_confirmation"
    if not paired.exists():
        return
    protocol_fields = {
        "num_warmups": 2,
        "turns_per_session": 1,
        "input_chunk_ms": 200,
        "turn_duration_ms": 0,
        "c1_prompts": 32,
        "c4_prompts": 64,
        "c8_prompts": 128,
    }
    a1 = root / "official_910c_baseline/duplex_rtf/run_protocol.json"
    b1 = root / "official_910c_optimized/duplex_rtf/run_protocol.json"
    a2 = paired / "baseline_a2/duplex_rtf/run_protocol.json"
    b2 = paired / "optimized_b2/duplex_rtf/run_protocol.json"
    for baseline_path, optimized_path, gate_name in (
        (a1, a2, "protocol_gate_baseline_repeat.json"),
        (b1, b2, "protocol_gate_optimized_repeat.json"),
        (a2, b2, "protocol_gate_second_pair.json"),
    ):
        _check_protocol_pair(
            expected_kind="realtime-duplex-speak-generation-matrix",
            baseline_path=baseline_path,
            optimized_path=optimized_path,
            gate_path=paired / gate_name,
            failures=failures,
            required_fields=protocol_fields,
        )
    saved_gate = _passed_json(paired / "paired_confirmation_gate.json", failures)
    for variant in ("baseline_a2", "optimized_b2"):
        case = paired / variant / "duplex_rtf"
        _json(case / "run_protocol.json", failures)
        _nonempty(case / "server.log", failures)
        result = _json(case / "native_duplex_c1_n32.json", failures)
        if result is not None:
            if int(result.get("sessions", 0) or 0) != 32:
                failures.append(f"paired confirmation {variant} is not 32 sessions")
            runs = result.get("runs")
            if not isinstance(runs, list) or len(runs) != 32 or not all(
                isinstance(run, dict) and run.get("ok") is True for run in runs
            ):
                failures.append(f"paired confirmation {variant} runs are incomplete")
    raw_paths = (
        root / "official_910c_baseline/duplex_rtf/native_duplex_c1_n32.json",
        root / "official_910c_optimized/duplex_rtf/native_duplex_c1_n32.json",
        paired / "optimized_b2/duplex_rtf/native_duplex_c1_n32.json",
        paired / "baseline_a2/duplex_rtf/native_duplex_c1_n32.json",
    )
    raw = [_json(path, failures) for path in raw_paths]
    if all(isinstance(item, dict) for item in raw):
        recomputed = _PAIRED.evaluate(*raw)
        if recomputed.get("passed") is not True:
            failures.append(
                "paired confirmation recomputation failed: "
                + "; ".join(str(item) for item in recomputed.get("failures", []))
            )
        if saved_gate is not None:
            for key in ("method", "max_repeat_drift_pct", "metrics"):
                if saved_gate.get(key) != recomputed.get(key):
                    failures.append(
                        f"paired confirmation saved gate {key} differs from recomputation"
                    )


def audit(
    *,
    result_root: Path,
    demo_root: Path,
    source_root: Path,
    report: Path,
    require_package: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    _check_environment(result_root, failures)
    _check_orchestrator(result_root, failures)
    _check_performance(result_root, failures)
    _check_accuracy(result_root, failures)
    _check_optional_paired_confirmation(result_root, failures)
    _check_demo(demo_root, failures)
    _check_source(source_root, failures)
    if require_package:
        _check_package(result_root / "final_submission/package", failures)
    _nonempty(report, failures)
    if report.is_file():
        report_text = report.read_text(encoding="utf-8", errors="replace")
        if "待填写" in report_text:
            failures.append(f"final report still contains 待填写 placeholders: {report}")
        for heading in (
            "## 环境与版本",
            "## 原始性能瓶颈分析",
            "## 最终优化方法",
            "## TTFT / TTFP（Chat Completions 辅助矩阵）",
            "## 官方目标口径：Realtime SPEAK 生成阶段",
            "## 精度准入",
            "## 稳定性、Activation 与 Demo",
            "## 资源使用与异常说明",
            "## 完整复现步骤",
            "## 复现与制品",
        ):
            if heading not in report_text:
                failures.append(f"final report is missing required section: {heading}")
    return {
        "passed": not failures,
        "official_910c_evidence": True,
        "result_root": str(result_root),
        "demo_root": str(demo_root),
        "source_root": str(source_root),
        "report": str(report),
        "package_required": require_package,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--demo-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-package",
        action="store_true",
        help="Also verify the already-built final submission archive.",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only validate the structured Demo evidence directory.",
    )
    args = parser.parse_args()
    result_root = args.result_root.expanduser().resolve()
    demo_root = (args.demo_root or result_root / "demo").expanduser().resolve()
    if args.require_package and args.output is not None:
        output = args.output.expanduser().resolve()
        try:
            output.relative_to(result_root)
        except ValueError:
            pass
        else:
            parser.error(
                "--require-package --output must be outside result-root; writing into "
                "the authoritative evidence tree would invalidate the verified archive"
            )
    if args.demo_only:
        if args.require_package:
            parser.error("--demo-only cannot be combined with --require-package")
        result = audit_demo(demo_root)
    else:
        result = audit(
            result_root=result_root,
            demo_root=demo_root,
            source_root=(args.source_root or result_root / "final_submission/source").expanduser().resolve(),
            report=(args.report or result_root / "final_submission/FINAL_REPORT.md").expanduser().resolve(),
            require_package=args.require_package,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
