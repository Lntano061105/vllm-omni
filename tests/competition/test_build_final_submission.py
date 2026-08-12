from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "competition/minicpmo_b/scripts/build_final_submission.py"
)
_SPEC = importlib.util.spec_from_file_location("build_final_submission", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _file(path: Path, data: bytes = b"evidence") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_deterministic_archive_and_embedded_hash_verification(tmp_path: Path) -> None:
    first = _file(tmp_path / "a.json", b'{"passed":true}\n')
    second = _file(tmp_path / "nested/b.log", b"server evidence\n")
    files = [("a.json", first), ("nested/b.log", second)]
    metadata = {
        "format_version": 1,
        "environment_timestamp_utc": "2026-08-12T00:00:00Z",
        "official_910c_evidence": True,
        "final_evidence_passed": True,
        "evidence_file_count": 2,
    }
    archive_a = tmp_path / "a.tar.gz"
    archive_b = tmp_path / "b.tar.gz"
    _MOD.write_deterministic_archive(archive_a, files, metadata=metadata)
    _MOD.write_deterministic_archive(archive_b, files, metadata=metadata)

    assert archive_a.read_bytes() == archive_b.read_bytes()
    result = _MOD.verify_archive(archive_a)
    assert result["passed"] is True
    assert result["archive"] == archive_a.name
    assert result["evidence_file_count"] == 2
    with tarfile.open(archive_a, "r:gz") as tar:
        stored = json.load(tar.extractfile("PACKAGE_METADATA.json"))
    assert stored == metadata


def test_verifier_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    payload = _file(tmp_path / "payload.bin", b"real")
    archive = tmp_path / "bad.tar.gz"
    _MOD.write_deterministic_archive(
        archive,
        [("payload.bin", payload)],
        metadata={"format_version": 1},
    )
    # Rebuild with a deliberately inconsistent embedded manifest using the
    # same safe tar writer primitives.
    import gzip
    import io

    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                data = b"real"
                tar.addfile(_MOD._tar_info("payload.bin", len(data)), io.BytesIO(data))
                manifest = ("0" * 64 + "  payload.bin\n").encode()
                tar.addfile(
                    _MOD._tar_info("PACKAGE_FILE_SHA256.txt", len(manifest)),
                    io.BytesIO(manifest),
                )
                metadata = json.dumps(
                    {
                        "format_version": 1,
                        "official_910c_evidence": True,
                        "final_evidence_passed": True,
                        "evidence_file_count": 1,
                    }
                ).encode()
                tar.addfile(
                    _MOD._tar_info("PACKAGE_METADATA.json", len(metadata)),
                    io.BytesIO(metadata),
                )

    result = _MOD.verify_archive(archive)
    assert result["passed"] is False
    assert any("SHA256 mismatch" in failure for failure in result["failures"])


def test_verifier_streams_evidence_members_without_unbounded_read(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _file(tmp_path / "large.bin", b"x" * (5 * 1024 * 1024))
    archive = tmp_path / "streamed.tar.gz"
    _MOD.write_deterministic_archive(
        archive,
        [("large.bin", payload)],
        metadata={
            "format_version": 1,
            "official_910c_evidence": True,
            "final_evidence_passed": True,
            "evidence_file_count": 1,
        },
    )
    original_read = tarfile.ExFileObject.read

    def bounded_read(self, size=-1):
        assert size >= 0, "archive verifier attempted an unbounded member read"
        return original_read(self, size)

    monkeypatch.setattr(tarfile.ExFileObject, "read", bounded_read)
    assert _MOD.verify_archive(archive)["passed"] is True


def test_failed_candidate_verification_preserves_existing_archive(
    tmp_path: Path,
) -> None:
    archive = _file(tmp_path / "final.tar.gz", b"previous valid archive")
    verification_path = _file(tmp_path / "archive_verification.json", b"old json")
    sha_path = _file(tmp_path / "archive_sha256.txt", b"old sha")
    candidate = _file(tmp_path / ".candidate", b"not a tar archive")
    before = {path: path.read_bytes() for path in (archive, verification_path, sha_path)}

    try:
        _MOD.publish_package_set(
            candidate=candidate,
            archive=archive,
            verification_path=verification_path,
            sha256_path=sha_path,
            expected_files=[],
        )
    except ValueError as exc:
        assert "verification failed" in str(exc)
    else:
        raise AssertionError("invalid candidate unexpectedly published")
    for path, expected in before.items():
        assert path.read_bytes() == expected
    assert not candidate.exists()


def test_package_set_publish_rolls_back_all_files_on_sidecar_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source = _file(tmp_path / "source.bin", b"new evidence")
    candidate = tmp_path / ".candidate.tar.gz"
    _MOD.write_deterministic_archive(
        candidate,
        [("source.bin", source)],
        metadata={
            "format_version": 1,
            "official_910c_evidence": True,
            "final_evidence_passed": True,
            "evidence_file_count": 1,
        },
    )
    archive = _file(tmp_path / "final.tar.gz", b"old archive")
    verification_path = _file(tmp_path / "archive_verification.json", b"old json")
    sha_path = _file(tmp_path / "archive_sha256.txt", b"old sha")
    before = {path: path.read_bytes() for path in (archive, verification_path, sha_path)}
    original_atomic = _MOD._atomic_bytes

    def fail_on_sha(path: Path, payload: bytes) -> None:
        if path == sha_path:
            raise OSError("injected sidecar failure")
        original_atomic(path, payload)

    monkeypatch.setattr(_MOD, "_atomic_bytes", fail_on_sha)
    try:
        _MOD.publish_package_set(
            candidate=candidate,
            archive=archive,
            verification_path=verification_path,
            sha256_path=sha_path,
            expected_files=[("source.bin", source)],
        )
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("injected sidecar failure did not abort publication")
    for path, expected in before.items():
        assert path.read_bytes() == expected
    assert not candidate.exists()
    assert not list(tmp_path.glob(".rollback-*"))


def test_package_set_successfully_publishes_consistent_three_file_set(
    tmp_path: Path,
) -> None:
    source = _file(tmp_path / "source.bin", b"new evidence")
    candidate = tmp_path / ".candidate.tar.gz"
    _MOD.write_deterministic_archive(
        candidate,
        [("source.bin", source)],
        metadata={
            "format_version": 1,
            "official_910c_evidence": True,
            "final_evidence_passed": True,
            "evidence_file_count": 1,
        },
    )
    archive = tmp_path / "final.tar.gz"
    verification_path = tmp_path / "archive_verification.json"
    sha_path = tmp_path / "archive_sha256.txt"
    result = _MOD.publish_package_set(
        candidate=candidate,
        archive=archive,
        verification_path=verification_path,
        sha256_path=sha_path,
        expected_files=[("source.bin", source)],
    )
    assert result["passed"] is True
    assert result["archive"] == archive.name
    assert archive.is_file()
    assert not candidate.exists()
    assert json.loads(verification_path.read_text()) == result
    assert sha_path.read_text().strip() == (
        f"{result['archive_sha256']}  {archive.name}"
    )
    assert not list(tmp_path.glob(".rollback-*"))


def test_incomplete_rollback_preserves_backup_directory(
    tmp_path: Path, monkeypatch
) -> None:
    source = _file(tmp_path / "source.bin", b"new evidence")
    candidate = tmp_path / ".candidate.tar.gz"
    _MOD.write_deterministic_archive(
        candidate,
        [("source.bin", source)],
        metadata={
            "format_version": 1,
            "official_910c_evidence": True,
            "final_evidence_passed": True,
            "evidence_file_count": 1,
        },
    )
    archive = _file(tmp_path / "final.tar.gz", b"old archive")
    verification_path = _file(tmp_path / "archive_verification.json", b"old json")
    sha_path = _file(tmp_path / "archive_sha256.txt", b"old sha")
    original_atomic = _MOD._atomic_bytes
    original_replace = _MOD.os.replace

    def fail_sidecar(path: Path, payload: bytes) -> None:
        if path == sha_path:
            raise OSError("injected publish failure")
        original_atomic(path, payload)

    def fail_archive_restore(source_path, target_path) -> None:
        source_path = Path(source_path)
        target_path = Path(target_path)
        if source_path.parent.name.startswith(".rollback-") and target_path == archive:
            raise OSError("injected rollback failure")
        original_replace(source_path, target_path)

    monkeypatch.setattr(_MOD, "_atomic_bytes", fail_sidecar)
    monkeypatch.setattr(_MOD.os, "replace", fail_archive_restore)
    try:
        _MOD.publish_package_set(
            candidate=candidate,
            archive=archive,
            verification_path=verification_path,
            sha256_path=sha_path,
            expected_files=[("source.bin", source)],
        )
    except RuntimeError as exc:
        assert "rollback was incomplete" in str(exc)
        assert "manual recovery" in str(exc)
    else:
        raise AssertionError("incomplete rollback was not surfaced")
    rollbacks = list(tmp_path.glob(".rollback-*"))
    assert len(rollbacks) == 1
    assert (rollbacks[0] / archive.name).read_bytes() == b"old archive"


def test_missing_archive_returns_failure_instead_of_raising(tmp_path: Path) -> None:
    result = _MOD.verify_archive(tmp_path / "missing.tar.gz")
    assert result["passed"] is False
    assert result["archive_sha256"] is None
    assert any("cannot read final archive" in item for item in result["failures"])


def test_verifier_rejects_duplicate_manifest_entries_and_unmanifested_member(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "bad-structure.tar.gz"
    import gzip

    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name, data in (("payload.bin", b"real"), ("extra.bin", b"extra")):
                    tar.addfile(_MOD._tar_info(name, len(data)), io.BytesIO(data))
                digest = __import__("hashlib").sha256(b"real").hexdigest()
                manifest = f"{digest}  payload.bin\n{digest}  payload.bin\n".encode()
                tar.addfile(
                    _MOD._tar_info("PACKAGE_FILE_SHA256.txt", len(manifest)),
                    io.BytesIO(manifest),
                )
                metadata = json.dumps(
                    {
                        "format_version": 1,
                        "official_910c_evidence": True,
                        "final_evidence_passed": True,
                        "evidence_file_count": 2,
                    }
                ).encode()
                tar.addfile(
                    _MOD._tar_info("PACKAGE_METADATA.json", len(metadata)),
                    io.BytesIO(metadata),
                )

    result = _MOD.verify_archive(archive)
    assert result["passed"] is False
    assert any("duplicate package manifest entry" in item for item in result["failures"])
    assert any("absent from manifest" in item for item in result["failures"])


def test_verifier_compares_archive_to_current_authoritative_files(
    tmp_path: Path,
) -> None:
    first = _file(tmp_path / "a.bin", b"version-one")
    archive = tmp_path / "authoritative.tar.gz"
    metadata = {
        "format_version": 1,
        "official_910c_evidence": True,
        "final_evidence_passed": True,
        "evidence_file_count": 1,
    }
    _MOD.write_deterministic_archive(
        archive, [("a.bin", first)], metadata=metadata
    )
    second = _file(tmp_path / "b.bin", b"new evidence")
    first.write_bytes(b"version-two")

    result = _MOD.verify_archive(
        archive,
        expected_files=[("a.bin", first), ("b.bin", second)],
    )
    assert result["passed"] is False
    assert any("missing authoritative evidence: b.bin" in item for item in result["failures"])
    assert any("differs from current authoritative file: a.bin" in item for item in result["failures"])


def test_collect_evidence_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "official"
    for relative in (
        "environment",
        "official_910c_baseline",
        "official_910c_optimized",
        "demo",
        "orchestrator_state",
        "final_submission/source",
    ):
        (root / relative).mkdir(parents=True)
    _file(root / "final_submission/FINAL_REPORT.md")
    _file(root / "final_evidence_audit.json")
    target = _file(root / "outside.txt")
    (root / "environment/link").symlink_to(target)

    try:
        _MOD.collect_evidence_files(root, root / "final_submission/package")
    except ValueError as exc:
        assert "symlinks are not allowed" in str(exc)
    else:
        raise AssertionError("expected symlink rejection")


def test_collect_evidence_includes_optional_paired_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "official"
    for relative in (
        "environment",
        "official_910c_baseline",
        "official_910c_optimized",
        "demo",
        "orchestrator_state",
        "final_submission/source",
    ):
        _file(root / relative / "evidence.txt")
    paired = _file(root / "paired_confirmation/paired_confirmation_gate.json")
    _file(root / "final_submission/FINAL_REPORT.md")
    _file(root / "final_evidence_audit.json")

    files = _MOD.collect_evidence_files(
        root, root / "final_submission/package"
    )
    names = {name for name, _ in files}

    assert "paired_confirmation/paired_confirmation_gate.json" in names
    assert dict(files)["paired_confirmation/paired_confirmation_gate.json"] == paired
