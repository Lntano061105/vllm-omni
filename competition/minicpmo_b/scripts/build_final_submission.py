#!/usr/bin/env python3
"""Build and verify the complete official-910C submission evidence archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "competition/minicpmo_b/results/official_910c"
ARCHIVE_PREFIX = "minicpmo_b_official_910c"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_evidence_files(result_root: Path, output_dir: Path) -> list[tuple[str, Path]]:
    """Return stable archive names and files, excluding the package output itself."""
    required_roots = (
        result_root / "environment",
        result_root / "official_910c_baseline",
        result_root / "official_910c_optimized",
        result_root / "demo",
        result_root / "orchestrator_state",
        result_root / "final_submission/source",
    )
    optional_roots = (result_root / "paired_confirmation",)
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    output_dir = output_dir.resolve()
    for root in required_roots:
        if not root.exists():
            raise ValueError(f"missing evidence directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlinks are not allowed in final evidence: {path}")
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(output_dir)
            except ValueError:
                pass
            else:
                continue
            relative = path.relative_to(result_root).as_posix()
            if relative in seen:
                raise ValueError(f"duplicate archive path: {relative}")
            seen.add(relative)
            files.append((relative, path))
    for root in optional_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlinks are not allowed in final evidence: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(result_root).as_posix()
            if relative in seen:
                raise ValueError(f"duplicate archive path: {relative}")
            seen.add(relative)
            files.append((relative, path))

    explicit = (
        result_root / "final_submission/FINAL_REPORT.md",
        result_root / "final_evidence_audit.json",
        *sorted(result_root.glob("performance_c*_n*.md")),
        *sorted(result_root.glob("accuracy_gate_*.json")),
    )
    for path in explicit:
        if not path.is_file():
            raise ValueError(f"missing final evidence file: {path}")
        relative = path.relative_to(result_root).as_posix()
        if relative not in seen:
            seen.add(relative)
            files.append((relative, path))
    return sorted(files)


def build_manifest(files: Iterable[tuple[str, Path]]) -> bytes:
    lines = [f"{_sha256_file(path)}  {name}" for name, path in files]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _tar_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def write_deterministic_archive(
    archive: Path,
    files: list[tuple[str, Path]],
    *,
    metadata: dict[str, object],
) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    manifest = build_manifest(files)
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with tmp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    for name, path in files:
                        info = _tar_info(name, path.stat().st_size)
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    tar.addfile(
                        _tar_info("PACKAGE_FILE_SHA256.txt", len(manifest)),
                        io.BytesIO(manifest),
                    )
                    tar.addfile(
                        _tar_info("PACKAGE_METADATA.json", len(metadata_bytes)),
                        io.BytesIO(metadata_bytes),
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, archive)
    finally:
        tmp.unlink(missing_ok=True)


def verify_archive(archive: Path) -> dict[str, object]:
    failures: list[str] = []
    content: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                failures.append(f"unsafe archive path: {member.name}")
                continue
            if not member.isfile():
                failures.append(f"non-regular archive member: {member.name}")
                continue
            if member.name in content:
                failures.append(f"duplicate archive member: {member.name}")
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                failures.append(f"cannot read archive member: {member.name}")
                continue
            content[member.name] = extracted.read()

    manifest_raw = content.get("PACKAGE_FILE_SHA256.txt")
    metadata_raw = content.get("PACKAGE_METADATA.json")
    if manifest_raw is None:
        failures.append("archive is missing PACKAGE_FILE_SHA256.txt")
    if metadata_raw is None:
        failures.append("archive is missing PACKAGE_METADATA.json")
    expected_names: set[str] = set()
    if manifest_raw is not None:
        for line_number, line in enumerate(manifest_raw.decode("utf-8").splitlines(), 1):
            try:
                expected, name = line.split(None, 1)
            except ValueError:
                failures.append(f"invalid package manifest line {line_number}")
                continue
            name = name.strip()
            expected_names.add(name)
            payload = content.get(name)
            if payload is None:
                failures.append(f"manifest target missing from archive: {name}")
            elif _sha256_bytes(payload) != expected:
                failures.append(f"archive member SHA256 mismatch: {name}")
    actual_evidence = set(content) - {"PACKAGE_FILE_SHA256.txt", "PACKAGE_METADATA.json"}
    extras = sorted(actual_evidence - expected_names)
    if extras:
        failures.append("archive members absent from manifest: " + ", ".join(extras))
    return {
        "passed": not failures,
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "evidence_file_count": len(actual_evidence),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir or result_root / "final_submission/package"
    ).expanduser().resolve()

    audit_path = result_root / "final_evidence_audit.json"
    audit_cmd = [
        sys.executable,
        str(REPO_ROOT / "competition/minicpmo_b/scripts/validate_final_evidence.py"),
        "--result-root",
        str(result_root),
        "--output",
        str(audit_path),
    ]
    audited = subprocess.run(
        audit_cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if audited.returncode:
        raise SystemExit("final evidence audit failed; archive not built:\n" + audited.stdout)

    files = collect_evidence_files(result_root, output_dir)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metadata = {
        "format_version": 1,
        "environment_timestamp_utc": (
            result_root / "environment/timestamp_utc.txt"
        ).read_text(encoding="utf-8").strip(),
        "official_910c_evidence": True,
        "final_evidence_passed": audit.get("passed") is True,
        "evidence_file_count": len(files),
    }
    archive = output_dir / f"{ARCHIVE_PREFIX}.tar.gz"
    write_deterministic_archive(archive, files, metadata=metadata)
    verification = verify_archive(archive)
    if not verification["passed"]:
        archive.unlink(missing_ok=True)
        raise SystemExit(
            "final archive verification failed:\n"
            + json.dumps(verification, ensure_ascii=False, indent=2)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "archive_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "archive_sha256.txt").write_text(
        f"{verification['archive_sha256']}  {archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
