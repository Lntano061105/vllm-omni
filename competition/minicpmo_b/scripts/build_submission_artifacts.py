#!/usr/bin/env python3
"""Build reproducible source artifacts for the MiniCPM-o challenge submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "competition/minicpmo_b"


def _run(*argv: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result


def _text(*argv: str, check: bool = True) -> str:
    return _run(*argv, check=check).stdout.decode("utf-8", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _manifest_paths(ls_files_output: str) -> list[str]:
    """Return every tracked regular-file candidate in stable order.

    The source archive contains the complete HEAD tree, so its checksum manifest
    must have the same repository-wide scope.  Keeping only the competition,
    runtime, and competition-test directories silently omitted supporting
    benchmark/E2E tests changed by the submission.
    """
    return sorted({line for line in ls_files_output.splitlines() if line})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Commit/ref for the official baseline used to generate optimization.patch.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only: include dirty status metadata but never uncommitted files.",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Skip the full git archive (useful for a fast development audit).",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    head = _text("git", "rev-parse", "HEAD").strip()
    base = _text("git", "rev-parse", "--verify", f"{args.base_ref}^{{commit}}").strip()
    branch = _text("git", "branch", "--show-current").strip()
    status = _text("git", "status", "--porcelain=v1", "--untracked-files=all")
    if status and not args.allow_dirty:
        raise SystemExit(
            "refusing to build a final artifact from a dirty worktree; commit/stash all "
            "changes or pass --allow-dirty for a non-final development snapshot"
        )

    audit_path = output / "submission_package_audit.json"
    audit_cmd = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts/validate_submission_package.py"),
        "--require-tracked",
        "--output",
        str(audit_path),
    ]
    audit = subprocess.run(
        audit_cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if audit.returncode:
        raise SystemExit(
            "submission package audit failed; no final artifact was built:\n"
            + audit.stdout.decode("utf-8", errors="replace")
        )

    _atomic_write(output / "git_commit.txt", (head + "\n").encode())
    _atomic_write(output / "git_base_commit.txt", (base + "\n").encode())
    _atomic_write(output / "git_branch.txt", (branch + "\n").encode())
    _atomic_write(output / "git_status.txt", status.encode())
    _atomic_write(
        output / "git_tree.txt",
        _run("git", "ls-tree", "-r", "--full-tree", "HEAD").stdout,
    )
    _atomic_write(
        output / "changed_files.txt",
        _run("git", "diff", "--name-status", f"{base}..{head}").stdout,
    )
    _atomic_write(
        output / "optimization.patch",
        _run("git", "diff", "--binary", "--full-index", f"{base}..{head}").stdout,
    )

    tracked = _manifest_paths(_text("git", "ls-files"))
    manifest_lines: list[str] = []
    for raw in sorted(tracked):
        path = REPO_ROOT / raw
        if path.is_file():
            manifest_lines.append(f"{_sha256(path)}  {raw}")
    _atomic_write(
        output / "source_sha256.txt",
        ("\n".join(manifest_lines) + "\n").encode(),
    )

    archive_path = output / "source_snapshot.tar.gz"
    if not args.skip_archive:
        archive_tmp = output / f".{archive_path.name}.tmp-{os.getpid()}"
        archived = _run(
            "git",
            "archive",
            "--format=tar.gz",
            f"--output={archive_tmp}",
            "HEAD",
        )
        if archived.returncode:
            raise SystemExit("git archive failed")
        os.replace(archive_tmp, archive_path)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head_commit": head,
        "base_commit": base,
        "base_ref": args.base_ref,
        "branch": branch,
        "dirty_worktree": bool(status),
        "final_candidate": not bool(status),
        "source_archive_included": not args.skip_archive,
        "source_manifest_file_count": len(manifest_lines),
    }
    _atomic_write(
        output / "artifact_metadata.json",
        (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )

    artifact_lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_sha256.txt":
            artifact_lines.append(f"{_sha256(path)}  {path.name}")
    _atomic_write(
        output / "artifact_sha256.txt",
        ("\n".join(artifact_lines) + "\n").encode(),
    )
    print(f"Submission source artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
