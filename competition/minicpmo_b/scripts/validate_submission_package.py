#!/usr/bin/env python3
"""Static, NPU-free audit for the MiniCPM-o challenge submission package."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "competition/minicpmo_b"
DOCS = (
    PACKAGE_ROOT / "README.md",
    PACKAGE_ROOT / "OFFICIAL_910C_RETEST.md",
    PACKAGE_ROOT / "SUBMISSION_CHECKLIST.md",
)
REQUIRED = (
    PACKAGE_ROOT / "config/minicpmo_4_5_910c_low_latency.yaml",
    PACKAGE_ROOT / "config/ablations/minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml",
    PACKAGE_ROOT / "scripts/start_server.sh",
    PACKAGE_ROOT / "scripts/benchmark_seed_tts.sh",
    PACKAGE_ROOT / "scripts/benchmark_duplex_rtf.sh",
    PACKAGE_ROOT / "scripts/benchmark_accuracy.sh",
    PACKAGE_ROOT / "scripts/run_perf_matrix.sh",
    PACKAGE_ROOT / "scripts/run_duplex_matrix.sh",
    PACKAGE_ROOT / "scripts/run_official_910c_retest.py",
    PACKAGE_ROOT / "scripts/run_paired_confirmation.sh",
    PACKAGE_ROOT / "scripts/analyze_paired_confirmation.py",
    PACKAGE_ROOT / "scripts/validate_official_910c_host.py",
    PACKAGE_ROOT / "scripts/validate_final_evidence.py",
    PACKAGE_ROOT / "scripts/render_final_report.py",
    PACKAGE_ROOT / "scripts/build_final_submission.py",
    PACKAGE_ROOT / "scripts/run_protocol.py",
    PACKAGE_ROOT / "scripts/run_accuracy_case.sh",
    PACKAGE_ROOT / "scripts/collect_environment.sh",
    PACKAGE_ROOT / "scripts/compare_accuracy_results.py",
    PACKAGE_ROOT / "scripts/gate_duplex_candidate.py",
    PACKAGE_ROOT / "scripts/gate_multiturn_duplex.py",
    PACKAGE_ROOT / "scripts/validate_candidate_log.py",
    PACKAGE_ROOT / "scripts/resume_seed_tts_quality.py",
    PACKAGE_ROOT / "scripts/summarize_performance.py",
    PACKAGE_ROOT / "scripts/validate_submission_package.py",
    PACKAGE_ROOT / "scripts/build_submission_artifacts.py",
    PACKAGE_ROOT / "DEMO_EVIDENCE_TEMPLATE.json",
)
FORBIDDEN_TRACKED_PREFIXES = (
    "competition/minicpmo_b/results/",
    "extra-info/",
    "kernel_meta/",
)
MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024
PLANNED_NEW_SOURCE_PREFIXES = (
    "competition/minicpmo_b/",
    "tests/competition/",
)


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _is_forbidden(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES)


def _is_planned_new_source(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in PLANNED_NEW_SOURCE_PREFIXES) and not _is_forbidden(relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="Fail if a required package file has not been added to git.",
    )
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for path in REQUIRED + DOCS:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(REPO_ROOT)}")

    path_pattern = re.compile(
        r"competition/minicpmo_b/[A-Za-z0-9_./-]+\.(?:sh|py|yaml|md)"
    )
    referenced: set[Path] = set()
    for doc in DOCS:
        if not doc.is_file():
            continue
        for raw in path_pattern.findall(doc.read_text(encoding="utf-8")):
            referenced.add(REPO_ROOT / raw)
    for path in sorted(referenced):
        if not path.is_file():
            errors.append(f"document references missing file: {path.relative_to(REPO_ROOT)}")

    shell_scripts = sorted((PACKAGE_ROOT / "scripts").glob("*.sh"))
    for script in shell_scripts:
        checked = _run("bash", "-n", str(script))
        if checked.returncode:
            errors.append(
                f"shell syntax failed: {script.relative_to(REPO_ROOT)}: {checked.stdout.strip()}"
            )
        if script.name != "server_utils.sh" and not os.access(script, os.X_OK):
            errors.append(f"shell script is not executable: {script.relative_to(REPO_ROOT)}")

    python_scripts = sorted((PACKAGE_ROOT / "scripts").glob("*.py"))
    compiled = _run(sys.executable, "-m", "py_compile", *(str(p) for p in python_scripts))
    if compiled.returncode:
        errors.append(f"Python syntax failed: {compiled.stdout.strip()}")

    # Parse the two production-facing configs through the real inheritance and
    # validation path, then ensure the portable 910C YAML does not retain the
    # development machine's physical NPU 5 binding.
    config_check = _run(
        sys.executable,
        "-c",
        "from vllm_omni.config.stage_config import load_deploy_config; "
        "from pathlib import Path; "
        "a=load_deploy_config(Path('competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml')); "
        "b=load_deploy_config(Path('competition/minicpmo_b/config/ablations/minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml')); "
        "assert [str(s.devices) for s in a.stages] == ['0','0','0']; "
        "assert a.stages[0].mm_processor_cache_gb == 0; "
        "assert b.stages[0].mm_processor_cache_gb == 0",
    )
    if config_check.returncode:
        errors.append(f"deploy config validation failed: {config_check.stdout.strip()}")

    required_untracked: list[str] = []
    required_ignored: list[str] = []
    for path in REQUIRED + DOCS:
        if not path.is_file():
            continue
        tracked = _run("git", "ls-files", "--error-unmatch", str(path.relative_to(REPO_ROOT)))
        if tracked.returncode:
            relative = str(path.relative_to(REPO_ROOT))
            required_untracked.append(relative)
            ignored = _run("git", "check-ignore", "-q", relative)
            if ignored.returncode == 0:
                required_ignored.append(relative)
    if required_untracked:
        message = "required files not tracked by git: " + ", ".join(required_untracked)
        (errors if args.require_tracked else warnings).append(message)
    if required_ignored:
        message = (
            "required files are ignored by git and need an explicit .gitignore exception "
            "or git add -f: " + ", ".join(required_ignored)
        )
        (errors if args.require_tracked else warnings).append(message)

    # Guard the Git submission boundary.  Result trees and profiler caches are
    # evidence/runtime products, not source; accidentally tracking them can add
    # hundreds of MiB and make the official checkout irreproducible.  The size
    # gate is intentionally above the largest legitimate file in this repo.
    tracked_files = sorted(
        line for line in _run("git", "ls-files").stdout.splitlines() if line
    )
    forbidden_tracked = [path for path in tracked_files if _is_forbidden(path)]
    if forbidden_tracked:
        errors.append(
            "runtime/foreign artifacts tracked by git: " + ", ".join(forbidden_tracked)
        )
    oversized_tracked: list[str] = []
    for relative in tracked_files:
        path = REPO_ROOT / relative
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            oversized_tracked.append(f"{relative} ({path.stat().st_size} bytes)")
    if oversized_tracked:
        errors.append(
            "tracked files exceed 10 MiB source limit: " + ", ".join(oversized_tracked)
        )

    # Produce an exact, machine-readable staging plan from the current tree.
    # Existing tracked changes are always part of the submission.  New files
    # are accepted only from the competition package and its dedicated tests;
    # known profiler/exception caches are explicitly excluded, and everything
    # else is surfaced for human review rather than silently staged.
    changed_tracked = sorted(
        line
        for line in _run("git", "diff", "--name-only", "HEAD", "--").stdout.splitlines()
        if line
    )
    untracked = sorted(
        line
        for line in _run(
            "git", "ls-files", "--others", "--exclude-standard"
        ).stdout.splitlines()
        if line
    )
    planned_untracked = [path for path in untracked if _is_planned_new_source(path)]
    excluded_untracked = [path for path in untracked if _is_forbidden(path)]
    unclassified_untracked = sorted(
        set(untracked) - set(planned_untracked) - set(excluded_untracked)
    )
    planned_source_files = sorted(set(changed_tracked) | set(planned_untracked))
    if unclassified_untracked:
        warnings.append(
            "unclassified untracked files require manual review: "
            + ", ".join(unclassified_untracked)
        )
    if planned_untracked:
        message = (
            "planned submission source files not tracked by git: "
            + ", ".join(planned_untracked)
        )
        if args.require_tracked:
            errors.append(message)

    # Every checked-in ablation must keep a resolvable inheritance chain.  The
    # portable production YAML is intentionally self-contained, while NPU5
    # provenance configs may inherit from their sibling experiments.
    missing_config_bases: list[str] = []
    for config in sorted((PACKAGE_ROOT / "config").rglob("*.yaml")):
        for line in config.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^base_config:\s*['\"]?([^'\"#]+)", line.strip())
            if not match:
                continue
            raw_base = match.group(1).strip()
            base = (config.parent / raw_base).resolve()
            if not base.is_file():
                missing_config_bases.append(
                    f"{config.relative_to(REPO_ROOT)} -> {raw_base}"
                )
    if missing_config_bases:
        errors.append(
            "deploy config base_config targets are missing: "
            + ", ".join(missing_config_bases)
        )

    report = {
        "passed": not errors,
        "npu_free_static_audit": True,
        "required_file_count": len(REQUIRED),
        "document_reference_count": len(referenced),
        "shell_script_count": len(shell_scripts),
        "python_script_count": len(python_scripts),
        "tracked_file_count": len(tracked_files),
        "forbidden_tracked_prefixes": list(FORBIDDEN_TRACKED_PREFIXES),
        "max_tracked_file_bytes": MAX_TRACKED_FILE_BYTES,
        "required_untracked": required_untracked,
        "required_ignored": required_ignored,
        "forbidden_tracked": forbidden_tracked,
        "oversized_tracked": oversized_tracked,
        "planned_new_source_prefixes": list(PLANNED_NEW_SOURCE_PREFIXES),
        "changed_tracked": changed_tracked,
        "planned_untracked": planned_untracked,
        "planned_source_files": planned_source_files,
        "excluded_untracked": excluded_untracked,
        "unclassified_untracked": unclassified_untracked,
        "missing_config_bases": missing_config_bases,
        "errors": errors,
        "warnings": warnings,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
