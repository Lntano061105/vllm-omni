#!/usr/bin/env python3
"""Dry-run-first, resumable official single-910C retest orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "vllm_omni/deploy/minicpmo_4_5.yaml"
BASELINE_ACC = REPO_ROOT / "competition/minicpmo_b/config/ablations/minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml"
OPTIMIZED = REPO_ROOT / "competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml"


@dataclass(frozen=True)
class Phase:
    name: str
    argv: tuple[str, ...]
    env: dict[str, str]


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def source_identity() -> dict[str, str]:
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
    }


def require_clean_source() -> None:
    status = _git_value("status", "--porcelain", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise RuntimeError(
            "official execution requires a clean Git worktree; commit or remove "
            f"all changes first:\n{preview}"
        )


def _phase(
    name: str,
    script: str,
    env: dict[str, str] | None = None,
    *args: str,
) -> Phase:
    return Phase(name, (str(REPO_ROOT / script), *args), env or {})


def build_phases(
    *,
    result_root: Path,
    device: int,
    port: int,
    model_path: Path = Path("/workspace/MiniCPM-o-4_5"),
    daily_omni_root: Path = Path("/tmp/minicpmo_b_daily_omni"),
    videomme_root: Path = Path("/tmp/minicpmo_b_videomme"),
    seed_tts_root: Path = Path("/tmp/minicpmo_b_seedtts"),
    whisper_model: Path = Path("/workspace/whisper-large-v3"),
    wavlm_model: Path = Path("/workspace/wavlm-base-plus"),
    utmos_model: Path = Path("/workspace/utmos/utmos.jit"),
    image_digest: str = "",
) -> list[Phase]:
    common = {
        "NPU_DEVICE": str(device),
        "PORT": str(port),
        "MODEL_PATH": str(model_path),
    }
    baseline_root = result_root / "official_910c_baseline"
    optimized_root = result_root / "official_910c_optimized"
    phases = [
        _phase(
            "preflight",
            "competition/minicpmo_b/scripts/validate_official_910c_host.py",
            {},
            "--device",
            str(device),
            "--model-path",
            str(model_path),
            "--daily-omni-root",
            str(daily_omni_root),
            "--videomme-root",
            str(videomme_root),
            "--seed-tts-root",
            str(seed_tts_root),
            "--whisper-model",
            str(whisper_model),
            "--wavlm-model",
            str(wavlm_model),
            "--utmos-model",
            str(utmos_model),
            "--result-root",
            str(result_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--output",
            str(result_root / "environment" / "preflight.json"),
            "--image-digest",
            image_digest,
        ),
        _phase(
            "environment",
            "competition/minicpmo_b/scripts/collect_environment.sh",
            {
                "OUTPUT_DIR": str(result_root / "environment"),
                "HASH_MODEL_WEIGHTS": "1",
                "MODEL_PATH": str(model_path),
                "CONTAINER_IMAGE_DIGEST": image_digest,
                "DAILY_OMNI_ROOT": str(daily_omni_root),
                "VIDEOMME_ROOT": str(videomme_root),
                "SEED_TTS_ROOT": str(seed_tts_root),
                "SEED_TTS_WHISPER_MODEL": str(whisper_model),
                "SEED_TTS_WAVLM_MODEL": str(wavlm_model),
                "SEED_TTS_UTMOS_FILE": str(utmos_model),
            },
        ),
        _phase(
            "performance-baseline",
            "competition/minicpmo_b/scripts/run_perf_matrix.sh",
            {
                **common,
                "CASE_NAME": "official_910c_baseline",
                "RESULT_DIR": str(baseline_root / "performance"),
                "DEPLOY_CONFIG": str(BASELINE),
                "REQUIRE_STAGE1_FULL_DECODE": "0",
            },
        ),
        _phase(
            "performance-optimized",
            "competition/minicpmo_b/scripts/run_perf_matrix.sh",
            {
                **common,
                "CASE_NAME": "official_910c_optimized",
                "RESULT_DIR": str(optimized_root / "performance"),
                "DEPLOY_CONFIG": str(OPTIMIZED),
                "REQUIRE_STAGE1_FULL_DECODE": "1",
            },
        ),
        _phase(
            "duplex-baseline",
            "competition/minicpmo_b/scripts/run_duplex_matrix.sh",
            {
                **common,
                "RESULT_DIR": str(baseline_root / "duplex_rtf"),
                "DEPLOY_CONFIG": str(BASELINE),
                "REQUIRE_STAGE1_FULL_DECODE": "0",
                "REQUIRE_ACTIVATION_GATE": "0",
            },
        ),
        _phase(
            "duplex-optimized",
            "competition/minicpmo_b/scripts/run_duplex_matrix.sh",
            {
                **common,
                "RESULT_DIR": str(optimized_root / "duplex_rtf"),
                "DEPLOY_CONFIG": str(OPTIMIZED),
                "REQUIRE_STAGE1_FULL_DECODE": "1",
                "REQUIRE_ACTIVATION_GATE": "1",
            },
        ),
        _phase(
            "protocol-performance",
            "competition/minicpmo_b/scripts/run_protocol.py",
            {},
            "compare",
            str(baseline_root / "performance/run_protocol.json"),
            str(optimized_root / "performance/run_protocol.json"),
            "--output",
            str(result_root / "protocol_gate_performance.json"),
            "--require-field",
            "benchmark_seed=0",
            "--require-field",
            "num_warmups=2",
            "--require-field",
            "no_oversample=true",
            "--require-field",
            "request_rate=inf",
            "--require-field",
            "c1_prompts=32",
            "--require-field",
            "c4_prompts=64",
            "--require-field",
            "c8_prompts=128",
        ),
        _phase(
            "protocol-duplex",
            "competition/minicpmo_b/scripts/run_protocol.py",
            {},
            "compare",
            str(baseline_root / "duplex_rtf/run_protocol.json"),
            str(optimized_root / "duplex_rtf/run_protocol.json"),
            "--output",
            str(result_root / "protocol_gate_duplex.json"),
            "--require-field",
            "num_warmups=2",
            "--require-field",
            "turns_per_session=1",
            "--require-field",
            "input_chunk_ms=200",
            "--require-field",
            "turn_duration_ms=0",
            "--require-field",
            "c1_prompts=32",
            "--require-field",
            "c4_prompts=64",
            "--require-field",
            "c8_prompts=128",
        ),
    ]
    for concurrency, prompts in ((1, 32), (4, 64), (8, 128)):
        suffix = f"c{concurrency}_n{prompts}"
        phases.append(
            _phase(
                f"summary-performance-c{concurrency}",
                "competition/minicpmo_b/scripts/summarize_performance.py",
                {},
                f"baseline={baseline_root / 'performance' / ('seed_tts_' + suffix + '.json')}",
                f"optimized={optimized_root / 'performance' / ('seed_tts_' + suffix + '.json')}",
                "--output",
                str(result_root / f"performance_{suffix}.md"),
            )
        )
        phases.append(
            _phase(
                f"gate-duplex-c{concurrency}",
                "competition/minicpmo_b/scripts/gate_duplex_candidate.py",
                {},
                str(baseline_root / "duplex_rtf" / f"native_duplex_{suffix}.json"),
                str(optimized_root / "duplex_rtf" / f"native_duplex_{suffix}.json"),
                "--profile",
                "combined",
                "--output",
                str(optimized_root / "duplex_rtf" / f"gate_c{concurrency}.json"),
            )
        )

    for suite in ("daily-omni", "videomme", "seed-tts"):
        protocol_requirements = [
            "--require-field",
            "benchmark_seed=0",
            "--require-field",
            "num_warmups=0",
            "--require-field",
            "no_oversample=true",
            "--require-field",
            "request_rate=inf",
            "--require-field",
            "temperature=0",
        ]
        if suite == "daily-omni":
            protocol_requirements.extend(
                (
                    "--require-field",
                    "disable_shuffle=false",
                    "--require-field",
                    "input_mode=all",
                    "--require-field",
                    "pack_mode=minicpm-interleave",
                    "--require-field",
                    "output_len=512",
                )
            )
        elif suite == "videomme":
            protocol_requirements.extend(
                (
                    "--require-field",
                    "num_prompts=2700",
                    "--require-field",
                    "disable_shuffle=true",
                    "--require-field",
                    "pack_mode=minicpm-frames",
                    "--require-field",
                    "max_frames=96",
                    "--require-field",
                    "output_len=128",
                )
            )
        else:
            protocol_requirements.extend(
                (
                    "--require-field",
                    "num_prompts=1000",
                    "--require-field",
                    "disable_shuffle=false",
                    "--require-field",
                    "locale=en",
                    "--require-field",
                    "sim_eval=1",
                    "--require-field",
                    "utmos_eval=1",
                )
            )
        for label, config, root in (
            ("baseline", BASELINE_ACC, baseline_root),
            ("optimized", OPTIMIZED, optimized_root),
        ):
            env = {
                **common,
                "SUITE": suite,
                "CASE_NAME": f"official_910c_{label}",
                "RESULT_DIR": str(root / "accuracy" / suite),
                "DEPLOY_CONFIG": str(config),
                "REQUIRE_STAGE0_MM_CACHE_DISABLED": "1",
                "DAILY_OMNI_ROOT": str(daily_omni_root),
                "VIDEOMME_ROOT": str(videomme_root),
                "SEED_TTS_ROOT": str(seed_tts_root),
            }
            if suite == "videomme":
                env.update(NUM_PROMPTS="2700", MAX_CONCURRENCY="4")
            elif suite == "seed-tts":
                env.update(
                    NUM_PROMPTS="1000",
                    MAX_CONCURRENCY="4",
                    SEED_TTS_SIM_EVAL="1",
                    SEED_TTS_UTMOS_EVAL="1",
                    MIN_SEED_TTS_MEAN_SIM="-1",
                    MIN_SEED_TTS_MEAN_UTMOS="0",
                    SEED_TTS_HF_WHISPER_MODEL=str(whisper_model),
                    SEED_TTS_WAVLM_MODEL=str(wavlm_model),
                    SEED_TTS_UTMOS_JIT_FILE=str(utmos_model),
                )
            else:
                env.update(MAX_CONCURRENCY="1")
            phases.append(
                _phase(
                    f"accuracy-{suite}-{label}",
                    "competition/minicpmo_b/scripts/run_accuracy_case.sh",
                    env,
                )
            )
        phases.append(
            _phase(
                f"protocol-accuracy-{suite}",
                "competition/minicpmo_b/scripts/run_protocol.py",
                {},
                "compare",
                str(baseline_root / "accuracy" / suite / "run_protocol.json"),
                str(optimized_root / "accuracy" / suite / "run_protocol.json"),
                "--output",
                str(result_root / f"protocol_gate_accuracy_{suite}.json"),
                *protocol_requirements,
            )
        )
        phases.append(
            _phase(
                f"gate-accuracy-{suite}",
                "competition/minicpmo_b/scripts/compare_accuracy_results.py",
                {},
                suite,
                str(baseline_root / "accuracy" / suite),
                str(optimized_root / "accuracy" / suite),
                "--output",
                str(result_root / f"accuracy_gate_{suite}.json"),
            )
        )
    return phases


def _render(phase: Phase) -> str:
    env = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(phase.env.items()))
    argv = " ".join(shlex.quote(arg) for arg in phase.argv)
    return f"{env} {argv}".strip()


def build_plan(
    phases: list[Phase],
    *,
    inputs: dict[str, object],
    source: dict[str, str] | None = None,
) -> dict[str, object]:
    """Freeze the canonical phase order and commands for later re-audit."""
    return {
        "format_version": 2,
        "phase_count": len(phases),
        "phase_names": [phase.name for phase in phases],
        "inputs": inputs,
        "source": source or source_identity(),
        "phases": [
            {
                "name": phase.name,
                "command": _render(phase),
                "command_sha256": hashlib.sha256(_render(phase).encode()).hexdigest(),
            }
            for phase in phases
        ],
    }


def _plan_sha256(plan: dict[str, object]) -> str:
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read orchestrator state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"orchestrator state is not a JSON object: {path}")
    return value


def _freeze_or_validate_plan(path: Path, plan: dict[str, object]) -> None:
    if path.is_file():
        frozen = _load_json_object(path)
        if frozen != plan:
            raise RuntimeError(
                "existing orchestrator plan differs from current inputs/source; "
                "use a new --result-root instead of mixing official evidence"
            )
        return
    _atomic_json(path, plan)


def _marker_passes(marker: Path, command_hash: str, plan_hash: str) -> bool:
    if not marker.is_file():
        return False
    old = _load_json_object(marker)
    return (
        old.get("passed") is True
        and old.get("returncode") == 0
        and old.get("command_sha256") == command_hash
        and old.get("plan_sha256") == plan_hash
    )


def _archive_invalidated_markers(
    state_dir: Path,
    all_phases: list[Phase],
    *,
    first_rerun_index: int,
) -> list[str]:
    invalidated = [
        state_dir / f"{phase.name}.json"
        for phase in all_phases[first_rerun_index:]
        if (state_dir / f"{phase.name}.json").is_file()
    ]
    if not invalidated:
        return []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = state_dir / "history" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived: list[str] = []
    for marker in invalidated:
        target = archive / marker.name
        os.replace(marker, target)
        archived.append(marker.stem)
    _atomic_json(
        archive / "invalidation.json",
        {
            "invalidated_from_phase": all_phases[first_rerun_index].name,
            "invalidated_utc": datetime.now(timezone.utc).isoformat(),
            "archived_markers": archived,
        },
    )
    return archived


def _missing_prerequisites(
    *,
    selected: list[Phase],
    all_phases: list[Phase],
    state_dir: Path,
    plan_hash: str,
) -> dict[str, list[str]]:
    indexes = {phase.name: index for index, phase in enumerate(all_phases)}
    selected_names = {phase.name for phase in selected}
    missing: dict[str, list[str]] = {}
    for phase in selected:
        required: list[str] = []
        for prerequisite in all_phases[: indexes[phase.name]]:
            if prerequisite.name in selected_names:
                continue
            command_hash = hashlib.sha256(
                _render(prerequisite).encode()
            ).hexdigest()
            if not _marker_passes(
                state_dir / f"{prerequisite.name}.json",
                command_hash,
                plan_hash,
            ):
                required.append(prerequisite.name)
        if required:
            missing[phase.name] = required
    return missing


def _selection_gaps_after_invalidation(
    *,
    selected: list[Phase],
    all_phases: list[Phase],
    first_execution_index: int,
) -> list[str]:
    selected_names = {phase.name for phase in selected}
    selected_indexes = [
        index
        for index, phase in enumerate(all_phases)
        if phase.name in selected_names and index >= first_execution_index
    ]
    if not selected_indexes:
        return []
    last_selected = max(selected_indexes)
    return [
        phase.name
        for phase in all_phases[first_execution_index : last_selected + 1]
        if phase.name not in selected_names
    ]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "competition/minicpmo_b/results/official_910c",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/MiniCPM-o-4_5"))
    parser.add_argument("--daily-omni-root", type=Path, default=Path("/tmp/minicpmo_b_daily_omni"))
    parser.add_argument("--videomme-root", type=Path, default=Path("/tmp/minicpmo_b_videomme"))
    parser.add_argument("--seed-tts-root", type=Path, default=Path("/tmp/minicpmo_b_seedtts"))
    parser.add_argument("--whisper-model", type=Path, default=Path("/workspace/whisper-large-v3"))
    parser.add_argument("--wavlm-model", type=Path, default=Path("/workspace/wavlm-base-plus"))
    parser.add_argument("--utmos-model", type=Path, default=Path("/workspace/utmos/utmos.jit"))
    parser.add_argument(
        "--image-digest",
        default=os.environ.get("CONTAINER_IMAGE_DIGEST", ""),
        help="Required for execution, e.g. quay.io/...@sha256:...",
    )
    parser.add_argument("--phase", action="append", help="Run/print only named phase; repeatable.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-single-910c", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()
    inputs = {
        "result_root": str(result_root),
        "device": args.device,
        "port": args.port,
        "model_path": str(args.model_path.expanduser().resolve()),
        "daily_omni_root": str(args.daily_omni_root.expanduser().resolve()),
        "videomme_root": str(args.videomme_root.expanduser().resolve()),
        "seed_tts_root": str(args.seed_tts_root.expanduser().resolve()),
        "whisper_model": str(args.whisper_model.expanduser().resolve()),
        "wavlm_model": str(args.wavlm_model.expanduser().resolve()),
        "utmos_model": str(args.utmos_model.expanduser().resolve()),
        "image_digest": args.image_digest,
    }
    all_phases = build_phases(
        result_root=result_root,
        device=args.device,
        port=args.port,
        model_path=args.model_path.expanduser().resolve(),
        daily_omni_root=args.daily_omni_root.expanduser().resolve(),
        videomme_root=args.videomme_root.expanduser().resolve(),
        seed_tts_root=args.seed_tts_root.expanduser().resolve(),
        whisper_model=args.whisper_model.expanduser().resolve(),
        wavlm_model=args.wavlm_model.expanduser().resolve(),
        utmos_model=args.utmos_model.expanduser().resolve(),
        image_digest=args.image_digest,
    )
    phases = all_phases
    if args.phase:
        wanted = set(args.phase)
        unknown = wanted - {phase.name for phase in phases}
        if unknown:
            parser.error("unknown phase(s): " + ", ".join(sorted(unknown)))
        phases = [phase for phase in phases if phase.name in wanted]

    print("Official 910C retest plan:")
    for phase in phases:
        print(f"[{phase.name}] {_render(phase)}")
    if not args.execute:
        print("Dry run only. Add --execute --confirm-single-910c on the official idle 910C host.")
        return 0
    if not args.confirm_single_910c:
        parser.error("--execute requires --confirm-single-910c")
    if not args.image_digest:
        parser.error("--execute requires --image-digest or CONTAINER_IMAGE_DIGEST")

    try:
        require_clean_source()
        plan = build_plan(all_phases, inputs=inputs)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    state_dir = result_root / "orchestrator_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    plan_path = state_dir / "orchestrator_plan.json"
    try:
        _freeze_or_validate_plan(plan_path, plan)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    plan_hash = _plan_sha256(plan)

    missing_prerequisites = _missing_prerequisites(
        selected=phases,
        all_phases=all_phases,
        state_dir=state_dir,
        plan_hash=plan_hash,
    )
    if missing_prerequisites:
        details = "; ".join(
            f"{phase}: {', '.join(required)}"
            for phase, required in missing_prerequisites.items()
        )
        raise SystemExit(
            "selected phase(s) have incomplete prerequisites; include them in this "
            f"run or resume the full plan: {details}"
        )

    phase_indexes = {phase.name: index for index, phase in enumerate(all_phases)}
    execution_indexes = [
        phase_indexes[phase.name]
        for phase in phases
        if args.rerun_completed
        or not _marker_passes(
            state_dir / f"{phase.name}.json",
            hashlib.sha256(_render(phase).encode()).hexdigest(),
            plan_hash,
        )
    ]
    if execution_indexes:
        first_rerun_index = min(execution_indexes)
        gaps = _selection_gaps_after_invalidation(
            selected=phases,
            all_phases=all_phases,
            first_execution_index=first_rerun_index,
        )
        if gaps:
            raise SystemExit(
                "selected phases cross dependencies that will be invalidated; "
                "include every intermediate phase or rerun only the earliest phase: "
                + ", ".join(gaps)
            )
        archived = _archive_invalidated_markers(
            state_dir,
            all_phases,
            first_rerun_index=first_rerun_index,
        )
        if archived:
            print(
                "Archived invalidated markers from "
                f"{all_phases[first_rerun_index].name}: {', '.join(archived)}"
            )
    for phase in phases:
        command = _render(phase)
        command_hash = hashlib.sha256(command.encode()).hexdigest()
        marker = state_dir / f"{phase.name}.json"
        if marker.is_file() and not args.rerun_completed:
            if _marker_passes(marker, command_hash, plan_hash):
                print(f"Skipping completed phase: {phase.name}")
                continue
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(
            phase.argv,
            cwd=REPO_ROOT,
            env={**os.environ, **phase.env},
            check=False,
        )
        record = {
            "phase": phase.name,
            "command": command,
            "command_sha256": command_hash,
            "plan_sha256": plan_hash,
            "source": plan["source"],
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
        }
        _atomic_json(marker, record)
        if completed.returncode:
            raise SystemExit(f"phase failed: {phase.name} (returncode={completed.returncode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
