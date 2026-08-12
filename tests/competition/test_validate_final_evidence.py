from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import tarfile
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/validate_final_evidence.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_final_evidence", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_REPORT_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/render_final_report.py"
)
_REPORT_SPEC = importlib.util.spec_from_file_location("render_final_report", _REPORT_SCRIPT)
assert _REPORT_SPEC and _REPORT_SPEC.loader
_REPORT_MOD = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(_REPORT_MOD)


def _write(path: Path, content: str = "evidence\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload))


def _build_complete_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "official_910c"
    env = root / "environment"
    _write_json(
        env / "preflight.json",
        {
            "passed": True,
            "single_910c": True,
            "selected_device": 0,
            "visible_devices": {"0": "Ascend 910C"},
            "npu_processes": {},
            "asset_preflight": {"passed": True},
            "token2wav_asset_probe": {"passed": True},
        },
    )
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
        _write(env / name)
    _write(
        env / "container_image_digest.txt",
        "quay.io/ascend/vllm-omni@sha256:" + "a" * 64 + "\n",
    )
    _write(env / "cann_version.txt", "Version=9.1.0\n")
    _write(env / "git_status.txt", "")
    orchestrator_inputs = {
        "result_root": str(root),
        "device": 0,
        "port": 8091,
        "model_path": "/workspace/MiniCPM-o-4_5",
        "daily_omni_root": "/tmp/minicpmo_b_daily_omni",
        "videomme_root": "/tmp/minicpmo_b_videomme",
        "seed_tts_root": "/tmp/minicpmo_b_seedtts",
        "whisper_model": "/workspace/whisper-large-v3",
        "wavlm_model": "/workspace/wavlm-base-plus",
        "utmos_model": "/workspace/utmos/utmos.jit",
        "image_digest": "quay.io/ascend/vllm-omni@sha256:" + "a" * 64,
    }
    orchestrator_phases = _MOD._ORCHESTRATOR.build_phases(
        result_root=root,
        device=0,
        port=8091,
        image_digest=orchestrator_inputs["image_digest"],
    )
    _write_json(
        root / "orchestrator_state/orchestrator_plan.json",
        _MOD._ORCHESTRATOR.build_plan(
            orchestrator_phases, inputs=orchestrator_inputs
        ),
    )
    for phase in orchestrator_phases:
        command = _MOD._ORCHESTRATOR._render(phase)
        _write_json(
            root / "orchestrator_state" / f"{phase.name}.json",
            {
                "phase": phase.name,
                "command": command,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "returncode": 0,
                "passed": True,
            },
        )

    for family in ("performance", "duplex_rtf"):
        kind = {
            "performance": "chat-completions-performance-matrix",
            "duplex_rtf": "realtime-duplex-speak-generation-matrix",
        }[family]
        protocols = {}
        for label in ("baseline", "optimized"):
            protocols[label] = _MOD._PROTOCOL.build_protocol(
                kind=kind,
                fields={
                    "benchmark_seed": 0,
                    "num_warmups": 2,
                    "no_oversample": True,
                    "request_rate": "inf",
                    "turns_per_session": 1,
                    "input_chunk_ms": 200,
                    "turn_duration_ms": 0,
                    "c1_prompts": 32,
                    "c4_prompts": 64,
                    "c8_prompts": 128,
                    "disable_shuffle": False,
                },
                files={},
                trees={},
                variant_fields={"label": label},
                variant_files={},
            )
            _write_json(
                root / f"official_910c_{label}/{family}/run_protocol.json",
                protocols[label],
            )
        gate_name = "performance" if family == "performance" else "duplex"
        _write_json(
            root / f"protocol_gate_{gate_name}.json",
            _MOD._PROTOCOL.compare_protocols(protocols["baseline"], protocols["optimized"]),
        )

    for concurrency, prompts in ((1, 32), (4, 64), (8, 128)):
        suffix = f"c{concurrency}_n{prompts}"
        duplex_payloads = {}
        for label in ("baseline", "optimized"):
            case = root / f"official_910c_{label}"
            _write(case / "performance/server.log")
            _write(case / "duplex_rtf/server.log")
            _write_json(
                case / "performance" / f"seed_tts_{suffix}.json",
                {
                    "completed": prompts,
                    "failed": 0,
                    "mean_ttft_ms": 100.0,
                    "mean_audio_ttfp_ms": 200.0,
                },
            )
            optimized_case = label == "optimized"
            duplex_payloads[label] = {
                "sessions": prompts,
                "audio_turns": prompts,
                "audio_speak_generation_chunk_count": prompts,
                "runs": [{"ok": True} for _ in range(prompts)],
                "ttft_ms": {"mean": 80.0 if optimized_case else 100.0},
                "ttfp_ms": {"mean": 160.0 if optimized_case else 200.0},
                "speak_generation_rtf": {
                    "mean": 0.35 if optimized_case else 0.50,
                    "median": 0.35 if optimized_case else 0.50,
                    "p99": 0.40 if optimized_case else 0.55,
                },
            }
            _write_json(
                case / "duplex_rtf" / f"native_duplex_{suffix}.json",
                duplex_payloads[label],
            )
        _write_json(
            root / "official_910c_optimized/duplex_rtf" / f"gate_c{concurrency}.json",
            _MOD._DUPLEX_GATE.evaluate_candidate(
                duplex_payloads["baseline"],
                duplex_payloads["optimized"],
                _MOD._DUPLEX_GATE.PROFILES["combined"],
            ),
        )
        _write(root / f"performance_{suffix}.md")
    _write_json(
        root / "official_910c_optimized/duplex_rtf/activation_gate.json",
        {"passed": True},
    )
    _write_json(
        root
        / "official_910c_optimized/duplex_rtf/multiturn_s2_t3/multiturn_gate.json",
        {"passed": True},
    )
    _write_json(
        root
        / "official_910c_optimized/duplex_rtf/multiturn_s2_t3/native_duplex_rtf.json",
        {"sessions": 2, "audio_turns": 6, "runs": [{"ok": True}, {"ok": True}]},
    )

    for suite in ("daily-omni", "videomme", "seed-tts"):
        suite_protocols = {
            label: _MOD._PROTOCOL.build_protocol(
                kind=f"accuracy-{suite}",
                fields={
                    "suite": suite,
                    "num_prompts": 1000 if suite == "seed-tts" else (2700 if suite == "videomme" else 1),
                    "benchmark_seed": 0,
                    "num_warmups": 0,
                    "no_oversample": True,
                    "request_rate": "inf",
                    "temperature": 0,
                },
                files={"utmos_model": {"sha256": "u"}} if suite == "seed-tts" else {},
                trees=(
                    {
                        "whisper_model": {"inventory_sha256": "w"},
                        "wavlm_model": {"inventory_sha256": "v"},
                    }
                    if suite == "seed-tts"
                    else {}
                ),
                variant_fields={"label": label},
                variant_files={},
            )
            for label in ("baseline", "optimized")
        }
        suite_fields = {
            "daily-omni": {
                "disable_shuffle": False,
                "input_mode": "all",
                "pack_mode": "minicpm-interleave",
                "output_len": 512,
            },
            "videomme": {
                "disable_shuffle": True,
                "pack_mode": "minicpm-frames",
                "max_frames": 96,
                "duration": "all",
                "output_len": 128,
            },
            "seed-tts": {
                "disable_shuffle": False,
                "locale": "en",
                "turns_per_session": 1,
                "sim_eval": 1,
                "utmos_eval": 1,
            },
        }[suite]
        for protocol in suite_protocols.values():
            protocol["comparison"]["fields"].update(suite_fields)
            protocol["comparison_sha256"] = _MOD._PROTOCOL.fingerprint(
                protocol["kind"], protocol["comparison"]
            )
        _write_json(
            root / f"protocol_gate_accuracy_{suite}.json",
            _MOD._PROTOCOL.compare_protocols(
                suite_protocols["baseline"], suite_protocols["optimized"]
            ),
        )
        absolute = {
            "daily-omni": ("daily_omni_accuracy", "higher", 0.78),
            "videomme": ("videomme_accuracy", "higher", 0.68),
            "seed-tts": ("seed_tts_content_error_mean", "lower", 0.05),
        }[suite]
        accuracy_payloads = {}
        for label in ("baseline", "optimized"):
            case = root / f"official_910c_{label}/accuracy/{suite}"
            _write_json(case / "run_protocol.json", suite_protocols[label])
            _write_json(case / "activation_gate.json", {"passed": True})
            if suite == "seed-tts":
                accuracy_payloads[label] = {
                    "completed": 1000,
                    "failed": 0,
                    "seed_tts_quality_complete": True,
                    "seed_tts_session_count": 1000,
                    "seed_tts_turn_count": 1000,
                    "seed_tts_content_evaluated": 1000,
                    "seed_tts_sim_evaluated": 1000,
                    "seed_tts_utmos_evaluated": 1000,
                    "seed_tts_request_failed": 0,
                    "seed_tts_no_pcm": 0,
                    "seed_tts_asr_failed": 0,
                    "seed_tts_save_audio_failed": 0,
                    "seed_tts_sim_failed": 0,
                    "seed_tts_sim_skipped_no_ref": 0,
                    "seed_tts_utmos_failed": 0,
                    "seed_tts_content_error_mean": 0.02,
                    "seed_tts_sim_mean": 0.80,
                    "seed_tts_utmos_mean": 4.0,
                }
                _write_json(
                    case / "seed_tts_quality_resumed.json",
                    accuracy_payloads[label],
                )
            else:
                filename = (
                    "omni_acc_videomme_result.json"
                    if suite == "videomme"
                    else f"qwen_omni_acc_{suite}.json"
                )
                prefix = "videomme" if suite == "videomme" else "daily_omni"
                expected = 2700 if suite == "videomme" else 1
                metric = "videomme_accuracy" if suite == "videomme" else "daily_omni_accuracy"
                accuracy_payloads[label] = {
                    "completed": expected,
                    "failed": 0,
                    metric: 0.80,
                    f"{prefix}_evaluated": expected,
                    f"{prefix}_evaluated_ok": expected,
                    f"{prefix}_request_failed": 0,
                    f"{prefix}_parse_failed": 0,
                }
                _write_json(case / filename, accuracy_payloads[label])
        expected_requests = 1000 if suite == "seed-tts" else (2700 if suite == "videomme" else 1)
        _write_json(
            root / f"accuracy_gate_{suite}.json",
            _MOD._ACCURACY_GATE.compare_accuracy(
                suite,
                accuracy_payloads["baseline"],
                accuracy_payloads["optimized"],
                max_regression=0.02,
                expected_requests=expected_requests,
            ),
        )

    demo = root / "demo"
    _write(demo / "server.log")
    _write(demo / "demo.mp4")
    _write_json(
        demo / "demo_evidence.json",
        {
            "official_910c": True,
            "started_utc": "2026-08-12T00:00:00Z",
            "finished_utc": "2026-08-12T00:30:00Z",
            "continuous_run_minutes": 30,
            "service_log": "server.log",
            "video_file": "demo.mp4",
            "service_exit_clean": True,
            "unexpected_error_count": 0,
            "audio_interruption_count": 0,
            "empty_audio_packet_count": 0,
            "scenarios": {
                name: {"passed": True} for name in ("text", "audio", "video", "text_audio")
            },
        },
    )

    source = root / "final_submission/source"
    source_members = {
        "competition/minicpmo_b/README.md": b"submission source\n",
        "vllm_omni/config/stage_config.py": b"source code\n",
    }
    source_archive = source / "source_snapshot.tar.gz"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, "w:gz") as tar:
        for name, data in sorted(source_members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    source_manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(source_members.items())
    )
    files = (
        "optimization.patch",
        "git_commit.txt",
        "git_base_commit.txt",
        "git_branch.txt",
        "git_tree.txt",
        "changed_files.txt",
    )
    for name in files:
        _write(source / name)
    _write(source / "source_sha256.txt", source_manifest)
    _write(source / "git_commit.txt", (env / "git_commit.txt").read_text())
    _write(source / "git_base_commit.txt", "base-commit\n")
    _write(source / "git_branch.txt", "minicpm-challenge-optimized\n")
    _write(source / "git_status.txt", "")
    _write_json(
        source / "artifact_metadata.json",
        {
            "final_candidate": True,
            "dirty_worktree": False,
            "head_commit": (source / "git_commit.txt").read_text().strip(),
            "base_commit": "base-commit",
            "branch": "minicpm-challenge-optimized",
            "source_manifest_file_count": len(source_members),
        },
    )
    _write_json(
        source / "submission_package_audit.json",
        {"passed": True, "warnings": []},
    )
    hashed_names = (
        "source_snapshot.tar.gz",
        *files,
        "source_sha256.txt",
        "git_status.txt",
        "artifact_metadata.json",
        "submission_package_audit.json",
    )
    manifest = "".join(
        f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
        for name in hashed_names
    )
    _write(source / "artifact_sha256.txt", manifest)
    report = root / "final_submission/FINAL_REPORT.md"
    headings = (
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
    )
    _write(report, "# Final official 910C report\n\n" + "\n\n".join(headings) + "\n")
    return root, demo, source, report


def test_complete_official_evidence_passes(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is True, result["failures"]


def test_rejects_910b_style_or_incomplete_demo_evidence(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    _write_json(root / "environment/preflight.json", {"passed": True, "single_910c": False})
    manifest = json.loads((demo / "demo_evidence.json").read_text(encoding="utf-8"))
    manifest["scenarios"]["video"]["passed"] = False
    _write_json(demo / "demo_evidence.json", manifest)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("single visible 910C" in item for item in result["failures"])
    assert any("scenario did not pass: video" in item for item in result["failures"])


def test_rejects_report_placeholder_and_bad_artifact_hash(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    _write(report, "待填写\n")
    _write(source / "optimization.patch", "changed after manifest\n")

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("待填写" in item for item in result["failures"])
    assert any("SHA256 mismatch" in item for item in result["failures"])


def _refresh_artifact_manifest(source: Path) -> None:
    names = [
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name != "artifact_sha256.txt"
    ]
    payload = "".join(
        f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(names)
    )
    _write(source / "artifact_sha256.txt", payload)


def test_rejects_replaced_source_archive_even_with_refreshed_outer_hash(
    tmp_path: Path,
) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    with tarfile.open(source / "source_snapshot.tar.gz", "w:gz") as tar:
        data = b"different source\n"
        info = tarfile.TarInfo("competition/minicpmo_b/README.md")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    _refresh_artifact_manifest(source)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("source snapshot SHA256 mismatch" in item for item in result["failures"])
    assert any("source snapshot is missing manifest entries" in item for item in result["failures"])


def test_rejects_source_commit_different_from_official_environment(
    tmp_path: Path,
) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    _write(source / "git_commit.txt", "different-commit\n")
    metadata = json.loads((source / "artifact_metadata.json").read_text())
    metadata["head_commit"] = "different-commit"
    _write_json(source / "artifact_metadata.json", metadata)
    _refresh_artifact_manifest(source)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any(
        "official environment git commit does not match source artifact commit" in item
        for item in result["failures"]
    )


def test_rejects_self_consistent_but_noncanonical_orchestrator_command(
    tmp_path: Path,
) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    marker_path = root / "orchestrator_state/performance-baseline.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["command"] += " EXTRA_UNOFFICIAL_FLAG=1"
    marker["command_sha256"] = hashlib.sha256(marker["command"].encode()).hexdigest()
    _write_json(marker_path, marker)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any(
        "marker command differs from canonical plan: performance-baseline" in item
        for item in result["failures"]
    )


def test_rejects_tampered_orchestrator_plan_even_when_markers_are_unchanged(
    tmp_path: Path,
) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    plan_path = root / "orchestrator_state/orchestrator_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["phase_names"] = list(reversed(plan["phase_names"]))
    _write_json(plan_path, plan)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any(
        "orchestrator_plan.json differs from current canonical" in item
        for item in result["failures"]
    )


def test_require_package_rejects_missing_final_archive(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    result = _MOD.audit(
        result_root=root,
        demo_root=demo,
        source_root=source,
        report=report,
        require_package=True,
    )
    assert result["passed"] is False
    assert any("archive_verification.json" in item for item in result["failures"])


def test_render_final_report_uses_official_speak_metric(tmp_path: Path) -> None:
    root, _, _, _ = _build_complete_tree(tmp_path)
    rendered = _REPORT_MOD.render(root)

    assert "官方目标口径：Realtime SPEAK 生成阶段" in rendered
    assert "c1 / n32" in rendered
    assert "PASS" in rendered
    assert "待填写" not in rendered
    assert "## 原始性能瓶颈分析" in rendered
    assert "## 最终优化方法" in rendered
    assert "## 完整复现步骤" in rendered


def test_rejects_partial_videomme_raw_result(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    path = root / "official_910c_optimized/accuracy/videomme/omni_acc_videomme_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed"] = 2699
    _write_json(path, payload)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("videomme completed=2699" in item for item in result["failures"])


def test_recomputes_and_rejects_tampered_protocol_despite_passed_gate(
    tmp_path: Path,
) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    path = root / "official_910c_optimized/performance/run_protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparison"]["fields"]["num_warmups"] = 1
    _write_json(path, payload)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("A/B protocol verification failed" in item for item in result["failures"])


def test_rejects_identical_but_wrong_protocol_kind(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    paths = [
        root / f"official_910c_{label}/performance/run_protocol.json"
        for label in ("baseline", "optimized")
    ]
    protocols = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["kind"] = "accuracy-daily-omni"
        payload["comparison_sha256"] = _MOD._PROTOCOL.fingerprint(
            payload["kind"], payload["comparison"]
        )
        _write_json(path, payload)
        protocols.append(payload)
    _write_json(
        root / "protocol_gate_performance.json",
        _MOD._PROTOCOL.compare_protocols(protocols[0], protocols[1]),
    )

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("expected 'chat-completions-performance-matrix'" in item for item in result["failures"])


def test_rejects_identical_protocol_with_wrong_benchmark_seed(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    paths = [
        root / f"official_910c_{label}/performance/run_protocol.json"
        for label in ("baseline", "optimized")
    ]
    protocols = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["comparison"]["fields"]["benchmark_seed"] = 17
        payload["comparison_sha256"] = _MOD._PROTOCOL.fingerprint(
            payload["kind"], payload["comparison"]
        )
        _write_json(path, payload)
        protocols.append(payload)
    _write_json(
        root / "protocol_gate_performance.json",
        _MOD._PROTOCOL.compare_protocols(protocols[0], protocols[1]),
    )

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("benchmark_seed" in item and "expected 0" in item for item in result["failures"])


def test_optional_paired_confirmation_is_recomputed(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    paired = root / "paired_confirmation"
    a1_protocol = json.loads(
        (root / "official_910c_baseline/duplex_rtf/run_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    b1_protocol = json.loads(
        (root / "official_910c_optimized/duplex_rtf/run_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    _write_json(paired / "baseline_a2/duplex_rtf/run_protocol.json", a1_protocol)
    _write_json(paired / "optimized_b2/duplex_rtf/run_protocol.json", b1_protocol)
    for left, right, name in (
        (a1_protocol, a1_protocol, "protocol_gate_baseline_repeat.json"),
        (b1_protocol, b1_protocol, "protocol_gate_optimized_repeat.json"),
        (a1_protocol, b1_protocol, "protocol_gate_second_pair.json"),
    ):
        _write_json(paired / name, _MOD._PROTOCOL.compare_protocols(left, right))

    def perf(ttft: float, ttfp: float, rtf: float) -> dict[str, object]:
        return {
            "sessions": 32,
            "audio_turns": 32,
            "audio_speak_generation_chunk_count": 64,
            "runs": [{"ok": True} for _ in range(32)],
            "ttft_ms": {"mean": ttft},
            "ttfp_ms": {"mean": ttfp},
            "speak_generation_rtf": {
                "mean": rtf,
                "median": rtf,
                "p99": rtf * 1.05,
            },
        }

    paths_and_payloads = (
        (root / "official_910c_baseline/duplex_rtf/native_duplex_c1_n32.json", perf(2000, 2000, 0.8)),
        (root / "official_910c_optimized/duplex_rtf/native_duplex_c1_n32.json", perf(1600, 1600, 0.6)),
        (paired / "optimized_b2/duplex_rtf/native_duplex_c1_n32.json", perf(1640, 1640, 0.62)),
        (paired / "baseline_a2/duplex_rtf/native_duplex_c1_n32.json", perf(2040, 2040, 0.82)),
    )
    for path, payload in paths_and_payloads:
        _write_json(path, payload)
    _write_json(
        root / "official_910c_optimized/duplex_rtf/gate_c1.json",
        _MOD._DUPLEX_GATE.evaluate_candidate(
            paths_and_payloads[0][1],
            paths_and_payloads[1][1],
            _MOD._DUPLEX_GATE.PROFILES["combined"],
        ),
    )
    _write(paired / "optimized_b2/duplex_rtf/server.log")
    _write(paired / "baseline_a2/duplex_rtf/server.log")
    gate = _MOD._PAIRED.evaluate(*(payload for _, payload in paths_and_payloads))
    _write_json(paired / "paired_confirmation_gate.json", gate)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is True, result["failures"]

    tampered = dict(gate)
    tampered["metrics"] = {}
    _write_json(paired / "paired_confirmation_gate.json", tampered)
    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("differs from recomputation" in item for item in result["failures"])


def test_core_duplex_and_accuracy_gates_are_recomputed(tmp_path: Path) -> None:
    root, demo, source, report = _build_complete_tree(tmp_path)
    duplex_gate = root / "official_910c_optimized/duplex_rtf/gate_c1.json"
    saved = json.loads(duplex_gate.read_text(encoding="utf-8"))
    saved["metrics"] = {}
    _write_json(duplex_gate, saved)

    accuracy_gate = root / "accuracy_gate_daily-omni.json"
    saved_accuracy = json.loads(accuracy_gate.read_text(encoding="utf-8"))
    saved_accuracy["metrics"] = {}
    _write_json(accuracy_gate, saved_accuracy)

    result = _MOD.audit(result_root=root, demo_root=demo, source_root=source, report=report)
    assert result["passed"] is False
    assert any("saved duplex gate c1 metrics differs" in item for item in result["failures"])
    assert any(
        "saved accuracy gate daily-omni metrics differs" in item
        for item in result["failures"]
    )
