from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_static_submission_package_audit_passes(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "competition/minicpmo_b/scripts/validate_submission_package.py"),
            "--output",
            str(output),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["npu_free_static_audit"] is True
    assert result["errors"] == []
    assert result["forbidden_tracked"] == []
    assert result["oversized_tracked"] == []
    assert result["missing_config_bases"] == []
    assert "competition/minicpmo_b/results/" in result["forbidden_tracked_prefixes"]
    assert "competition/minicpmo_b/" in result["planned_new_source_prefixes"]
    assert "tests/competition/" in result["planned_new_source_prefixes"]
    assert result["required_ignored"] == []
    demo_template = "competition/minicpmo_b/DEMO_EVIDENCE_TEMPLATE.json"
    assert subprocess.run(
        ["git", "ls-files", "--error-unmatch", demo_template],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "check-ignore", "-q", demo_template],
        cwd=repo,
        check=False,
    ).returncode != 0
    assert all(
        not path.startswith("extra-info/")
        and not path.startswith("kernel_meta/")
        and not path.startswith("competition/minicpmo_b/results/")
        for path in result["planned_source_files"]
    )
