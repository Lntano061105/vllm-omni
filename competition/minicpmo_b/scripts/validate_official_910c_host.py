#!/usr/bin/env python3
"""Fail-fast guard before an official single-card 910C retest.

This script is intentionally read-only. It rejects development 910B hosts,
multiple visible NPUs, an occupied selected NPU, and stale vLLM-Omni workers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


_DEVICE_ROW = re.compile(r"^\|\s*(\d+)\s+([^|]*?910C[^|]*?)\s+", re.IGNORECASE)
_ANY_910_ROW = re.compile(r"^\|\s*(\d+)\s+([^|]*?910[A-Za-z0-9_-]*[^|]*?)\s+", re.IGNORECASE)
_NPU_PROCESS_ROW = re.compile(r"^\|\s*(\d+)\s+\d+\s+\|\s*(\d+)\s+\|")
_STALE_PROCESS = re.compile(
    r"(?:vllm\s+serve|StageEngineCoreProc|APIServer|run_perf_matrix\.sh|"
    r"run_duplex_matrix\.sh|run_accuracy_case\.sh)",
    re.IGNORECASE,
)


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _nonempty(path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        failures.append(f"missing or empty required asset: {path}")


def inspect_assets(
    *,
    model_path: Path,
    daily_omni_root: Path,
    videomme_root: Path,
    seed_tts_root: Path,
    whisper_model: Path,
    wavlm_model: Path,
    utmos_model: Path,
    result_root: Path,
    host: str,
    port: int,
    check_port: bool = True,
    min_free_gb: float,
    package_versions: dict[str, str | None],
    check_media_files: bool = True,
    video_parquet_rows: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    for name in (
        "config.json",
        "model.safetensors.index.json",
        "assets/system_ref_audio.wav",
        "assets/token2wav/speech_tokenizer_v2_25hz.onnx",
        "assets/token2wav/campplus.onnx",
        "assets/token2wav/flow.pt",
        "assets/token2wav/flow.yaml",
        "assets/token2wav/hift.pt",
    ):
        _nonempty(model_path / name, failures)
    index = model_path / "model.safetensors.index.json"
    checkpoint_shards: list[str] = []
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            checkpoint_shards = sorted(set(payload["weight_map"].values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid model weight index {index}: {exc}")
        for name in checkpoint_shards:
            _nonempty(model_path / name, failures)

    qa = daily_omni_root / "qa.json"
    _nonempty(qa, failures)
    daily_rows = 0
    if qa.is_file():
        try:
            value = json.loads(qa.read_text(encoding="utf-8"))
            daily_rows = len(value) if isinstance(value, list) else 0
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid Daily-Omni qa.json: {exc}")
        if daily_rows <= 0:
            failures.append("Daily-Omni qa.json contains no rows")
        elif check_media_files:
            daily_video_root = daily_omni_root / "Videos"
            missing_daily: list[str] = []
            for row in value:
                if not isinstance(row, dict):
                    missing_daily.append("<invalid-row>")
                    continue
                video_id = str(row.get("video_id") or row.get("video") or "").strip()
                if not video_id:
                    missing_daily.append("<missing-video-id>")
                    continue
                video_candidates = (
                    daily_video_root / video_id / f"{video_id}_video.mp4",
                    daily_video_root / f"{video_id}.mp4",
                )
                audio_candidates = (
                    daily_video_root / video_id / f"{video_id}_audio.wav",
                    daily_video_root / f"{video_id}.wav",
                )
                if not any(path.is_file() and path.stat().st_size > 0 for path in video_candidates):
                    missing_daily.append(f"{video_id}:video")
                if not any(path.is_file() and path.stat().st_size > 0 for path in audio_candidates):
                    missing_daily.append(f"{video_id}:audio")
            if missing_daily:
                failures.append(
                    f"Daily-Omni is missing {len(missing_daily)} referenced media file(s); "
                    f"examples={missing_daily[:8]}"
                )

    video_parquet = videomme_root / "videomme/test-00000-of-00001.parquet"
    _nonempty(video_parquet, failures)
    observed_video_rows = video_parquet_rows
    video_ids: list[str] = []
    if video_parquet.is_file() and observed_video_rows is None:
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(video_parquet)
            observed_video_rows = parquet.metadata.num_rows
            if check_media_files:
                table = pq.read_table(video_parquet, columns=["videoID"])
                video_ids = [str(value).strip() for value in table.column(0).to_pylist()]
        except Exception as exc:
            failures.append(f"cannot inspect Video-MME parquet metadata: {exc}")
    if observed_video_rows != 2700:
        failures.append(f"Video-MME parquet has {observed_video_rows} rows, expected 2700")
    if check_media_files and video_ids:
        video_root = videomme_root / "video"
        missing_video_ids: list[str] = []
        for video_id in sorted(set(video_ids)):
            candidates = (
                video_root / f"{video_id}.mp4",
                video_root / f"{video_id}.MP4",
                video_root / f"{video_id}.mkv",
                video_root / video_id / f"{video_id}.mp4",
            )
            if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
                missing_video_ids.append(video_id)
        if missing_video_ids:
            failures.append(
                f"Video-MME is missing {len(missing_video_ids)} referenced videos; "
                f"examples={missing_video_ids[:8]}"
            )
    seed_meta = seed_tts_root / "en/meta.lst"
    _nonempty(seed_meta, failures)
    seed_rows = 0
    if seed_meta.is_file():
        seed_lines = [line.strip() for line in seed_meta.open(encoding="utf-8") if line.strip()]
        seed_rows = len(seed_lines)
        if seed_rows < 1000:
            failures.append(f"Seed-TTS en/meta.lst has {seed_rows} rows, expected at least 1000")
        if check_media_files:
            missing_seed: list[str] = []
            malformed_seed = 0
            for line in seed_lines[:1000]:
                parts = line.split("|")
                if len(parts) < 4:
                    malformed_seed += 1
                    continue
                reference = seed_tts_root / "en" / parts[2]
                if not reference.is_file() or reference.stat().st_size <= 0:
                    missing_seed.append(parts[2])
            if malformed_seed:
                failures.append(f"Seed-TTS first 1000 rows contain {malformed_seed} malformed lines")
            if missing_seed:
                failures.append(
                    f"Seed-TTS is missing {len(missing_seed)} reference WAV(s) in first 1000 rows; "
                    f"examples={missing_seed[:8]}"
                )

    for label, root in (("Whisper", whisper_model), ("WavLM", wavlm_model)):
        if not root.is_dir():
            failures.append(f"{label} model directory is missing: {root}")
            continue
        _nonempty(root / "config.json", failures)
        weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
        if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
            failures.append(f"{label} model weights are missing under {root}")
    _nonempty(utmos_model, failures)

    required_versions = {
        "s3tokenizer": "0.3.0",
        "vllm-omni": "0.25.0",
    }
    for package, expected in required_versions.items():
        actual = package_versions.get(package)
        if actual != expected:
            failures.append(f"package {package}={actual!r}, expected {expected!r}")
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers", "onnxruntime"):
        if not package_versions.get(package):
            failures.append(f"required package is missing: {package}")

    if check_port:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
        except OSError as exc:
            failures.append(f"service port {host}:{port} is not available: {exc}")

    result_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(result_root).free / (1024**3)
    if free_gb < min_free_gb:
        failures.append(
            f"insufficient free disk under {result_root}: {free_gb:.2f} GiB < {min_free_gb:.2f} GiB"
        )

    return {
        "passed": not failures,
        "model_checkpoint_shards": checkpoint_shards,
        "daily_omni_rows": daily_rows,
        "seed_tts_rows": seed_rows,
        "video_parquet_bytes": video_parquet.stat().st_size if video_parquet.is_file() else 0,
        "video_parquet_rows": observed_video_rows,
        "package_versions": package_versions,
        "service_address": f"{host}:{port}",
        "free_disk_gb": free_gb,
        "minimum_free_disk_gb": min_free_gb,
        "failures": failures,
        "warnings": warnings,
    }


def run_token2wav_asset_probe(model_path: Path, timeout_s: int) -> dict[str, Any]:
    token2wav = model_path / "assets/token2wav"
    code = (
        "import json, onnxruntime, s3tokenizer; "
        f"p={str(token2wav)!r}; "
        "m=s3tokenizer.load_model(p+'/speech_tokenizer_v2_25hz.onnx'); "
        "s=onnxruntime.InferenceSession(p+'/campplus.onnx', providers=['CPUExecutionProvider']); "
        "print(json.dumps({'s3tokenizer_model':type(m).__name__,'campplus_inputs':len(s.get_inputs())}))"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "returncode": None,
            "output": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "failure": f"Token2Wav asset probe timed out after {timeout_s}s",
        }
    output = completed.stdout.strip()
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output[-8000:],
        "failure": None
        if completed.returncode == 0
        else "S3Tokenizer/campplus asset compatibility probe failed",
    }


def inspect_host(
    *,
    device: int,
    model_path: Path,
    daily_omni_root: Path,
    videomme_root: Path,
    seed_tts_root: Path,
    image_digest: str,
    npu_smi_text: str,
    process_text: str,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if not re.search(r"@sha256:[0-9a-fA-F]{64}$", image_digest.strip()):
        failures.append("official container image digest must end with @sha256:<64 hex>")
    rows: dict[int, str] = {}
    for line in npu_smi_text.splitlines():
        match = _ANY_910_ROW.match(line)
        if match:
            rows[int(match.group(1))] = match.group(2).strip()

    selected = rows.get(device)
    if len(rows) != 1:
        failures.append(f"expected exactly one visible NPU, found {len(rows)}: {rows}")
    if selected is None:
        failures.append(f"selected NPU {device} is not visible: {rows}")
    elif "910c" not in selected.lower():
        failures.append(f"selected NPU {device} is not a 910C: {selected}")

    npu_processes: dict[int, list[int]] = {}
    for line in npu_smi_text.splitlines():
        match = _NPU_PROCESS_ROW.match(line)
        if match:
            npu_processes.setdefault(int(match.group(1)), []).append(int(match.group(2)))
    selected_processes = npu_processes.get(device, [])
    if selected_processes:
        failures.append(
            f"selected NPU {device} is occupied by process(es): {selected_processes}"
        )

    stale = [line for line in process_text.splitlines() if _STALE_PROCESS.search(line)]
    if stale:
        failures.append("stale vLLM-Omni process(es): " + " | ".join(stale))

    required_model_entries = ("config.json",)
    if not model_path.is_dir():
        failures.append(f"model directory is missing: {model_path}")
    else:
        for name in required_model_entries:
            if not (model_path / name).is_file():
                failures.append(f"model file is missing: {model_path / name}")

    return {
        "passed": not failures,
        "single_910c": len(rows) == 1 and selected is not None and "910c" in selected.lower(),
        "selected_device": device,
        "visible_devices": rows,
        "npu_processes": npu_processes,
        "model_path": str(model_path),
        "container_image_digest": image_digest,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--daily-omni-root", type=Path, required=True)
    parser.add_argument("--videomme-root", type=Path, required=True)
    parser.add_argument("--seed-tts-root", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--wavlm-model", type=Path, required=True)
    parser.add_argument("--utmos-model", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--asset-probe-timeout", type=int, default=180)
    parser.add_argument("--skip-token2wav-probe", action="store_true")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    npu = _run("npu-smi", "info")
    processes = _run("ps", "-eo", "pid,ppid,stat,cmd")
    if npu.returncode:
        result: dict[str, Any] = {
            "passed": False,
            "failures": [f"npu-smi info failed ({npu.returncode}): {npu.stdout.strip()}"],
            "warnings": [],
        }
    elif processes.returncode:
        result = {
            "passed": False,
            "failures": [f"process inspection failed ({processes.returncode}): {processes.stdout.strip()}"],
            "warnings": [],
        }
    else:
        result = inspect_host(
            device=args.device,
            model_path=args.model_path.resolve(),
            daily_omni_root=args.daily_omni_root.resolve(),
            videomme_root=args.videomme_root.resolve(),
            seed_tts_root=args.seed_tts_root.resolve(),
            image_digest=args.image_digest,
            npu_smi_text=npu.stdout,
            process_text=processes.stdout,
        )
        package_names = (
            "s3tokenizer",
            "onnxruntime",
            "torch",
            "torch-npu",
            "vllm",
            "vllm-omni",
            "vllm-ascend",
            "transformers",
        )
        versions: dict[str, str | None] = {}
        for name in package_names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        assets = inspect_assets(
            model_path=args.model_path.resolve(),
            daily_omni_root=args.daily_omni_root.resolve(),
            videomme_root=args.videomme_root.resolve(),
            seed_tts_root=args.seed_tts_root.resolve(),
            whisper_model=args.whisper_model.resolve(),
            wavlm_model=args.wavlm_model.resolve(),
            utmos_model=args.utmos_model.resolve(),
            result_root=args.result_root.resolve(),
            host=args.host,
            port=args.port,
            min_free_gb=args.min_free_gb,
            package_versions=versions,
        )
        result["asset_preflight"] = assets
        result["failures"].extend(assets["failures"])
        result["warnings"].extend(assets["warnings"])
        if not args.skip_token2wav_probe:
            probe = run_token2wav_asset_probe(
                args.model_path.resolve(), args.asset_probe_timeout
            )
            result["token2wav_asset_probe"] = probe
            if not probe["passed"]:
                result["failures"].append(
                    f"{probe['failure']}: {probe.get('output', '')[-2000:]}"
                )
        result["passed"] = not result["failures"]

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
