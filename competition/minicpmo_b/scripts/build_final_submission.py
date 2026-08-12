#!/usr/bin/env python3
"""Build and verify the complete official-910C submission evidence archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "competition/minicpmo_b/results/official_910c"
ARCHIVE_PREFIX = "minicpmo_b_official_910c"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _read_bounded(handle: BinaryIO, size: int, *, limit: int, label: str) -> bytes:
    if size > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return payload


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    publish_succeeded = False
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backup_for_rollback(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


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


def verify_archive(
    archive: Path,
    *,
    expected_files: Iterable[tuple[str, Path]] | None = None,
) -> dict[str, object]:
    failures: list[str] = []
    actual: dict[str, str] = {}
    manifest_raw: bytes | None = None
    metadata_raw: bytes | None = None
    reserved = {"PACKAGE_FILE_SHA256.txt", "PACKAGE_METADATA.json"}
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts:
                    failures.append(f"unsafe archive path: {member.name}")
                    continue
                if not member.isfile():
                    failures.append(f"non-regular archive member: {member.name}")
                    continue
                if member.name in actual or (
                    member.name == "PACKAGE_FILE_SHA256.txt" and manifest_raw is not None
                ) or (
                    member.name == "PACKAGE_METADATA.json" and metadata_raw is not None
                ):
                    failures.append(f"duplicate archive member: {member.name}")
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    failures.append(f"cannot read archive member: {member.name}")
                    continue
                try:
                    if member.name == "PACKAGE_FILE_SHA256.txt":
                        manifest_raw = _read_bounded(
                            extracted,
                            member.size,
                            limit=16 * 1024 * 1024,
                            label=member.name,
                        )
                    elif member.name == "PACKAGE_METADATA.json":
                        metadata_raw = _read_bounded(
                            extracted,
                            member.size,
                            limit=1024 * 1024,
                            label=member.name,
                        )
                    else:
                        actual[member.name] = _sha256_stream(extracted)
                except ValueError as exc:
                    failures.append(str(exc))
    except (OSError, tarfile.TarError, EOFError) as exc:
        failures.append(f"cannot read final archive: {exc}")

    if manifest_raw is None:
        failures.append("archive is missing PACKAGE_FILE_SHA256.txt")
    if metadata_raw is None:
        failures.append("archive is missing PACKAGE_METADATA.json")
    expected: dict[str, str] = {}
    if manifest_raw is not None:
        try:
            manifest_text = manifest_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"package manifest is not UTF-8: {exc}")
            manifest_text = ""
        for line_number, line in enumerate(manifest_text.splitlines(), 1):
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in parts[0]
            ):
                failures.append(f"invalid package manifest line {line_number}")
                continue
            digest, name = parts[0].lower(), parts[1].strip()
            pure = PurePosixPath(name)
            if not name or pure.is_absolute() or ".." in pure.parts or name in reserved:
                failures.append(f"unsafe package manifest target: {name!r}")
                continue
            if name in expected:
                failures.append(f"duplicate package manifest entry: {name}")
                continue
            expected[name] = digest
        for name, digest in expected.items():
            if name not in actual:
                failures.append(f"manifest target missing from archive: {name}")
            elif actual[name] != digest:
                failures.append(f"archive member SHA256 mismatch: {name}")
    extras = sorted(set(actual) - set(expected))
    if extras:
        failures.append("archive members absent from manifest: " + ", ".join(extras))
    if expected_files is not None:
        authoritative: dict[str, str] = {}
        for name, path in expected_files:
            if name in authoritative:
                failures.append(f"duplicate authoritative evidence path: {name}")
                continue
            authoritative[name] = _sha256_file(path)
        missing_from_archive = sorted(set(authoritative) - set(actual))
        unexpected_in_archive = sorted(set(actual) - set(authoritative))
        if missing_from_archive:
            failures.append(
                "archive is missing authoritative evidence: "
                + ", ".join(missing_from_archive)
            )
        if unexpected_in_archive:
            failures.append(
                "archive contains non-authoritative evidence: "
                + ", ".join(unexpected_in_archive)
            )
        for name in sorted(set(authoritative) & set(actual)):
            if authoritative[name] != actual[name]:
                failures.append(
                    f"archive evidence differs from current authoritative file: {name}"
                )
    metadata: dict[str, object] | None = None
    if metadata_raw is not None:
        try:
            value = json.loads(metadata_raw)
            if not isinstance(value, dict):
                raise ValueError("root is not an object")
            metadata = value
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            failures.append(f"invalid package metadata: {exc}")
    if metadata is not None:
        if metadata.get("format_version") != 1:
            failures.append("package metadata format_version is not 1")
        if metadata.get("official_910c_evidence") is not True:
            failures.append("package metadata is not official_910c_evidence=true")
        if metadata.get("final_evidence_passed") is not True:
            failures.append("package metadata final_evidence_passed is not true")
        if metadata.get("evidence_file_count") != len(actual):
            failures.append("package metadata evidence_file_count mismatch")
    try:
        archive_sha256 = _sha256_file(archive)
    except OSError as exc:
        failures.append(f"cannot hash final archive: {exc}")
        archive_sha256 = None
    return {
        "passed": not failures,
        "archive": archive.name,
        "archive_sha256": archive_sha256,
        "evidence_file_count": len(actual),
        "failures": failures,
    }


def publish_package_set(
    *,
    candidate: Path,
    archive: Path,
    verification_path: Path,
    sha256_path: Path,
    expected_files: Iterable[tuple[str, Path]],
) -> dict[str, object]:
    verification = verify_archive(candidate, expected_files=expected_files)
    if verification.get("passed") is not True:
        candidate.unlink(missing_ok=True)
        raise ValueError(
            "final archive verification failed:\n"
            + json.dumps(verification, ensure_ascii=False, indent=2)
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    rollback = archive.parent / f".rollback-{os.getpid()}"
    rollback.mkdir(parents=True, exist_ok=False)
    published = (archive, verification_path, sha256_path)
    existed = {path: path.is_file() for path in published}
    rollback_incomplete = False
    try:
        for path in published:
            if existed[path]:
                _backup_for_rollback(path, rollback / path.name)
        os.replace(candidate, archive)
        _fsync_directory(archive.parent)
        verification["archive"] = archive.name
        _atomic_bytes(
            verification_path,
            (
                json.dumps(
                    verification, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8"),
        )
        _atomic_bytes(
            sha256_path,
            f"{verification['archive_sha256']}  {archive.name}\n".encode("utf-8"),
        )
        _fsync_directory(archive.parent)
    except BaseException as publish_exc:
        restore_failures: list[str] = []
        for path in published:
            backup = rollback / path.name
            try:
                if existed[path] and backup.is_file():
                    os.replace(backup, path)
                elif not existed[path]:
                    path.unlink(missing_ok=True)
            except OSError as restore_exc:
                restore_failures.append(f"{path}: {restore_exc}")
        candidate.unlink(missing_ok=True)
        try:
            _fsync_directory(archive.parent)
        except OSError as restore_exc:
            restore_failures.append(
                f"fsync {archive.parent}: {restore_exc}"
            )
        if restore_failures:
            rollback_incomplete = True
            raise RuntimeError(
                "package publication failed and rollback was incomplete; preserve "
                f"{rollback} for manual recovery: " + "; ".join(restore_failures)
            ) from publish_exc
        raise
    finally:
        if not rollback_incomplete:
            shutil.rmtree(rollback, ignore_errors=True)
    return verification


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
    candidate = output_dir / f".{archive.name}.candidate-{os.getpid()}"
    candidate.unlink(missing_ok=True)
    write_deterministic_archive(candidate, files, metadata=metadata)
    verification_path = output_dir / "archive_verification.json"
    sha256_path = output_dir / "archive_sha256.txt"
    try:
        verification = publish_package_set(
            candidate=candidate,
            archive=archive,
            verification_path=verification_path,
            sha256_path=sha256_path,
            expected_files=files,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
