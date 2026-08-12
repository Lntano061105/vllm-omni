from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "competition/minicpmo_b/scripts/finalize_demo_evidence.py"
VALIDATOR = REPO / "competition/minicpmo_b/scripts/validate_final_evidence.py"


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_demo(tmp_path: Path) -> Path:
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "server.log").write_text(
        "Application startup complete.\nclean shutdown\n", encoding="utf-8"
    )
    (demo / "demo.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypmp42" + b"recording" * 8
    )
    _json(
        demo / "demo_run_metadata.json",
        {
            "official_910c": True,
            "demo_name": "official Demo",
            "started_utc": "2026-08-12T00:00:00Z",
            "finished_utc": "2026-08-12T00:30:00Z",
            "continuous_run_minutes": 30,
            "service_exit_clean": True,
            "unexpected_error_count": 0,
            "audio_interruption_count": 0,
            "empty_audio_packet_count": 0,
            "audio_underrun_count": 0,
        },
    )
    for name in ("text", "audio", "video", "text_audio"):
        output = demo / f"output_{name}.bin"
        output.write_bytes(f"output-{name}".encode())
        request = {
            "request_id": f"request-{name}-1",
            "started_utc": "2026-08-12T00:01:00Z",
            "finished_utc": "2026-08-12T00:02:00Z",
            "completed": True,
            "audio_packet_count": 0 if name == "text" else 7,
            "output_file": output.name,
        }
        _json(
            demo / f"scenario_{name}.json",
            {"scenario": name, "requests": [request]},
        )
    return demo


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_finalizer_builds_recomputed_manifest_and_demo_only_passes(
    tmp_path: Path,
) -> None:
    demo = _build_demo(tmp_path)
    completed = _run(SCRIPT, "--demo-root", str(demo))
    assert completed.returncode == 0, completed.stdout
    result = json.loads(completed.stdout)
    assert result["passed"] is True
    manifest = json.loads((demo / "demo_evidence.json").read_text())
    assert manifest["scenarios"]["text"]["request_count"] == 1
    assert manifest["scenarios"]["audio"]["audio_packet_count"] == 7
    assert len(manifest["service_log_sha256"]) == 64
    assert len(manifest["video_sha256"]) == 64

    checked = _run(
        VALIDATOR, "--demo-only", "--demo-root", str(demo)
    )
    assert checked.returncode == 0, checked.stdout


def test_finalizer_rejects_incomplete_request_and_zero_audio_packets(
    tmp_path: Path,
) -> None:
    demo = _build_demo(tmp_path)
    audio_path = demo / "scenario_audio.json"
    audio = json.loads(audio_path.read_text())
    audio["requests"][0]["completed"] = False
    audio["requests"][0]["audio_packet_count"] = 0
    _json(audio_path, audio)

    completed = _run(SCRIPT, "--demo-root", str(demo))
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["passed"] is False
    assert any("audio_packet_count must be positive" in item for item in result["failures"])
    assert any("did not complete" in item for item in result["failures"])


def test_demo_only_recomputes_manifest_counts_from_raw_requests(
    tmp_path: Path,
) -> None:
    demo = _build_demo(tmp_path)
    assert _run(SCRIPT, "--demo-root", str(demo)).returncode == 0
    manifest_path = demo / "demo_evidence.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scenarios"]["video"]["request_count"] = 99
    manifest["scenarios"]["video"]["audio_packet_count"] = 999
    _json(manifest_path, manifest)

    completed = _run(
        VALIDATOR, "--demo-only", "--demo-root", str(demo)
    )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert any("evidence has 1 request(s)" in item for item in result["failures"])
    assert any("evidence has 7 packet(s)" in item for item in result["failures"])


def test_demo_only_rejects_tampered_output_and_scenario_hashes(
    tmp_path: Path,
) -> None:
    demo = _build_demo(tmp_path)
    assert _run(SCRIPT, "--demo-root", str(demo)).returncode == 0
    (demo / "output_audio.bin").write_bytes(b"tampered output")
    scenario = json.loads((demo / "scenario_video.json").read_text())
    scenario["requests"][0]["request_id"] = "tampered-request"
    _json(demo / "scenario_video.json", scenario)

    completed = _run(
        VALIDATOR, "--demo-only", "--demo-root", str(demo)
    )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert any("output_file SHA256 mismatch" in item for item in result["failures"])
    assert any("evidence_file SHA256 mismatch" in item for item in result["failures"])


def test_demo_only_rejects_metadata_tamper_duplicate_id_and_out_of_run_time(
    tmp_path: Path,
) -> None:
    demo = _build_demo(tmp_path)
    text_path = demo / "scenario_text.json"
    text = json.loads(text_path.read_text())
    duplicate = dict(text["requests"][0])
    duplicate["started_utc"] = "2026-08-11T23:58:00Z"
    duplicate["finished_utc"] = "2026-08-11T23:59:00Z"
    text["requests"].append(duplicate)
    _json(text_path, text)
    assert _run(SCRIPT, "--demo-root", str(demo)).returncode == 1

    metadata_path = demo / "demo_run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["unexpected_error_count"] = 1
    _json(metadata_path, metadata)

    completed = _run(
        VALIDATOR, "--demo-only", "--demo-root", str(demo)
    )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert any("metadata_file SHA256 mismatch" in item for item in result["failures"])
    assert any("request_id is duplicated" in item for item in result["failures"])
    assert any("outside Demo run interval" in item for item in result["failures"])


def test_finalizer_rejects_input_collision_and_refuses_to_overwrite_source(
    tmp_path: Path,
) -> None:
    demo = _build_demo(tmp_path)
    source = demo / "scenario_audio.json"
    before = source.read_bytes()
    completed = _run(
        SCRIPT,
        "--demo-root",
        str(demo),
        "--scenario-video",
        "scenario_audio.json",
        "--output",
        "scenario_audio.json",
    )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert any("inputs must be distinct" in item for item in result["failures"])
    assert source.read_bytes() == before

    completed = _run(
        SCRIPT,
        "--demo-root",
        str(demo),
        "--output",
        "scenario_audio.json",
    )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert any("output collides" in item for item in result["failures"])
    assert source.read_bytes() == before
