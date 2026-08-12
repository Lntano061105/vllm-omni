from __future__ import annotations

import importlib.util
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
    }
    archive_a = tmp_path / "a.tar.gz"
    archive_b = tmp_path / "b.tar.gz"
    _MOD.write_deterministic_archive(archive_a, files, metadata=metadata)
    _MOD.write_deterministic_archive(archive_b, files, metadata=metadata)

    assert archive_a.read_bytes() == archive_b.read_bytes()
    result = _MOD.verify_archive(archive_a)
    assert result["passed"] is True
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
                metadata = b"{}\n"
                tar.addfile(
                    _MOD._tar_info("PACKAGE_METADATA.json", len(metadata)),
                    io.BytesIO(metadata),
                )

    result = _MOD.verify_archive(archive)
    assert result["passed"] is False
    assert any("SHA256 mismatch" in failure for failure in result["failures"])


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
