#!/usr/bin/env python3
"""Write and compare normalized A/B benchmark provenance protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _pairs(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not key or key in output:
            raise ValueError(f"invalid or duplicate key: {key!r}")
        output[key] = _value(value)
    return output


def _files(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected KEY=PATH, got {raw!r}")
        key, path_raw = raw.split("=", 1)
        path = Path(path_raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"protocol input file is missing: {path}")
        output[key] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return output


def _trees(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected KEY=PATH, got {raw!r}")
        key, path_raw = raw.split("=", 1)
        root = Path(path_raw).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"protocol input directory is missing: {root}")
        lines: list[str] = []
        total_bytes = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"protocol input tree contains symlink: {path}")
            if path.is_file():
                size = path.stat().st_size
                total_bytes += size
                lines.append(f"{path.relative_to(root).as_posix()}\t{size}")
        inventory = ("\n".join(lines) + "\n").encode("utf-8")
        output[key] = {
            "path": str(root),
            "file_count": len(lines),
            "total_bytes": total_bytes,
            "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        }
    return output


def fingerprint(kind: str, comparison: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"kind": kind, "comparison": comparison},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_protocol(
    *,
    kind: str,
    fields: dict[str, Any],
    files: dict[str, Any],
    trees: dict[str, Any],
    variant_fields: dict[str, Any],
    variant_files: dict[str, Any],
) -> dict[str, Any]:
    comparison = {"fields": fields, "files": files, "trees": trees}
    return {
        "protocol_version": 1,
        "kind": kind,
        "comparison": comparison,
        "variant": {"fields": variant_fields, "files": variant_files},
        "comparison_sha256": fingerprint(kind, comparison),
    }


def compare_protocols(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for label, value in (("baseline", baseline), ("optimized", optimized)):
        comparison = value.get("comparison")
        kind = value.get("kind")
        if value.get("protocol_version") != 1:
            failures.append(f"{label} protocol_version is not 1")
        if not isinstance(kind, str) or not isinstance(comparison, dict):
            failures.append(f"{label} protocol is malformed")
        elif value.get("comparison_sha256") != fingerprint(kind, comparison):
            failures.append(f"{label} comparison_sha256 is invalid")
    if baseline.get("kind") != optimized.get("kind"):
        failures.append(
            f"protocol kind changed: baseline={baseline.get('kind')!r}, "
            f"optimized={optimized.get('kind')!r}"
        )
    if baseline.get("comparison") != optimized.get("comparison"):
        failures.append("A/B comparison protocol differs")
    if baseline.get("comparison_sha256") != optimized.get("comparison_sha256"):
        failures.append("A/B comparison fingerprint differs")
    return {
        "passed": not failures,
        "kind": baseline.get("kind"),
        "comparison_sha256": baseline.get("comparison_sha256"),
        "baseline_variant": baseline.get("variant"),
        "optimized_variant": optimized.get("variant"),
        "failures": failures,
    }


def require_fields(protocol: dict[str, Any], required: dict[str, Any]) -> list[str]:
    comparison = protocol.get("comparison")
    fields = comparison.get("fields") if isinstance(comparison, dict) else None
    failures: list[str] = []
    for key, expected in required.items():
        actual = fields.get(key) if isinstance(fields, dict) else None
        if actual != expected:
            failures.append(
                f"comparison field {key!r} is {actual!r}, expected {expected!r}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--kind", required=True)
    write.add_argument("--field", action="append", default=[])
    write.add_argument("--file", action="append", default=[])
    write.add_argument("--tree", action="append", default=[])
    write.add_argument("--variant-field", action="append", default=[])
    write.add_argument("--variant-file", action="append", default=[])
    compare = sub.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("optimized", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument(
        "--require-field",
        action="append",
        default=[],
        help="Require an exact comparison KEY=JSON_VALUE in addition to A/B equality.",
    )
    args = parser.parse_args()

    if args.command == "write":
        result = build_protocol(
            kind=args.kind,
            fields=_pairs(args.field),
            files=_files(args.file),
            trees=_trees(args.tree),
            variant_fields=_pairs(args.variant_field),
            variant_files=_files(args.variant_file),
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    optimized = json.loads(args.optimized.read_text(encoding="utf-8"))
    result = compare_protocols(baseline, optimized)
    required_failures = require_fields(baseline, _pairs(args.require_field))
    if required_failures:
        result["failures"].extend(required_failures)
        result["passed"] = False
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
