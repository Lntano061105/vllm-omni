#!/usr/bin/env python3
"""Build and verify Demo evidence from per-request scenario records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import validate_final_evidence


SCENARIOS = ("text", "audio", "video", "text_audio")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(path: Path, demo_root: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"symlink evidence is not allowed: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(demo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"evidence path escapes demo root: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"missing or empty evidence file: {path}")
    return relative.as_posix()


def build_manifest(
    *,
    demo_root: Path,
    metadata_path: Path,
    service_log: Path,
    video_file: Path,
    scenario_paths: dict[str, Path],
) -> dict[str, Any]:
    metadata_relative = _relative_file(metadata_path, demo_root)
    metadata = _load_object(metadata_path)
    manifest = dict(metadata)
    manifest["metadata_file"] = metadata_relative
    manifest["metadata_sha256"] = _sha256_file(metadata_path)
    manifest["service_log"] = _relative_file(service_log, demo_root)
    manifest["service_log_sha256"] = _sha256_file(service_log)
    manifest["video_file"] = _relative_file(video_file, demo_root)
    manifest["video_sha256"] = _sha256_file(video_file)
    scenarios: dict[str, dict[str, Any]] = {}
    for name in SCENARIOS:
        path = scenario_paths[name]
        evidence_relative = _relative_file(path, demo_root)
        raw = _load_object(path)
        if raw.get("scenario") != name:
            raise ValueError(
                f"scenario file {path} declares {raw.get('scenario')!r}, expected {name!r}"
            )
        requests = raw.get("requests")
        if not isinstance(requests, list):
            raise ValueError(f"scenario {name} requests is not a list: {path}")
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                raise ValueError(f"scenario {name} request[{index}] is not an object")
            output_file = request.get("output_file")
            if not isinstance(output_file, str) or output_file.startswith("REPLACE_"):
                raise ValueError(
                    f"scenario {name} request[{index}] output_file is not populated"
                )
            output_path = demo_root / output_file
            request["output_file"] = _relative_file(output_path, demo_root)
            request["output_sha256"] = _sha256_file(output_path)
        _atomic_json(path, raw)
        completed = sum(
            1 for request in requests
            if isinstance(request, dict) and request.get("completed") is True
        )
        packets = sum(
            request.get("audio_packet_count", 0)
            for request in requests
            if isinstance(request, dict)
            and isinstance(request.get("audio_packet_count", 0), int)
            and not isinstance(request.get("audio_packet_count", 0), bool)
        )
        scenario = {
            "passed": bool(requests) and completed == len(requests),
            "evidence_file": evidence_relative,
            "evidence_sha256": _sha256_file(path),
            "request_count": len(requests),
            "completed_response_count": completed,
        }
        if name != "text":
            scenario["audio_packet_count"] = packets
        scenarios[name] = scenario
    manifest["scenarios"] = scenarios
    return manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--metadata", default="demo_run_metadata.json")
    parser.add_argument("--service-log", default="server.log")
    parser.add_argument("--video-file", default="demo.mp4")
    parser.add_argument("--output", default="demo_evidence.json")
    for name in SCENARIOS:
        parser.add_argument(f"--scenario-{name.replace('_', '-')}", default=f"scenario_{name}.json")
    args = parser.parse_args()
    demo_root = args.demo_root.expanduser().resolve()

    def inside(raw: str) -> Path:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else demo_root / path

    try:
        metadata_path = inside(args.metadata)
        service_log = inside(args.service_log)
        video_file = inside(args.video_file)
        scenario_paths = {
            name: inside(getattr(args, f"scenario_{name}")) for name in SCENARIOS
        }
        source_paths = {
            "metadata": metadata_path.resolve(),
            "service_log": service_log.resolve(),
            "video_file": video_file.resolve(),
            **{
                f"scenario_{name}": path.resolve()
                for name, path in scenario_paths.items()
            },
        }
        reverse: dict[Path, list[str]] = {}
        for label, path in source_paths.items():
            reverse.setdefault(path, []).append(label)
        collisions = [labels for labels in reverse.values() if len(labels) > 1]
        if collisions:
            raise ValueError(
                "Demo evidence inputs must be distinct: "
                + "; ".join(", ".join(labels) for labels in collisions)
            )
        output = inside(args.output).resolve()
        if output in reverse:
            raise ValueError(
                f"output collides with Demo evidence input: {', '.join(reverse[output])}"
            )
        try:
            output.relative_to(demo_root)
        except ValueError as exc:
            raise ValueError("output escapes demo root") from exc
        manifest = build_manifest(
            demo_root=demo_root,
            metadata_path=metadata_path,
            service_log=service_log,
            video_file=video_file,
            scenario_paths=scenario_paths,
        )
        _atomic_json(output, manifest)
        result = validate_final_evidence.audit_demo(demo_root)
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "failures": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
