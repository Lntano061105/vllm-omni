#!/usr/bin/env python3
"""Summarize Ascend op_statistic.csv into challenge-relevant hot paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUPS = {
    "matmul": {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"},
    "attention": {"FusedInferAttentionScore", "FlashAttentionScore"},
    "slot_mapping": {"_compute_slot_mapping_kernel"},
    "sampler_control": {
        "Bincount",
        "ScatterElements",
        "Sort",
        "Cumsum",
        "MaskedFill",
        "MaskedFill_",
        "TopKV2",
        "SoftmaxV2",
    },
}


def summarize(paths: list[Path], *, top: int = 20) -> dict[str, Any]:
    operators: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "total_time_us": 0.0}
    )
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                op_type = str(row.get("OP Type", "")).strip()
                if not op_type:
                    continue
                operators[op_type]["count"] += float(row.get("Count", 0) or 0)
                operators[op_type]["total_time_us"] += float(
                    row.get("Total Time(us)", 0) or 0
                )

    device_total_us = sum(item["total_time_us"] for item in operators.values())
    ranked = sorted(
        operators.items(),
        key=lambda item: item[1]["total_time_us"],
        reverse=True,
    )

    def render_op(name: str, values: dict[str, float]) -> dict[str, float | str]:
        total_us = values["total_time_us"]
        count = values["count"]
        return {
            "op_type": name,
            "count": int(count),
            "total_time_us": total_us,
            "avg_time_us": total_us / count if count else 0.0,
            "device_share_pct": 100.0 * total_us / device_total_us if device_total_us else 0.0,
        }

    groups: dict[str, dict[str, float | list[str]]] = {}
    for group_name, members in GROUPS.items():
        matched = [name for name in operators if name in members]
        group_us = sum(operators[name]["total_time_us"] for name in matched)
        share = group_us / device_total_us if device_total_us else 0.0
        groups[group_name] = {
            "operators": sorted(matched),
            "total_time_us": group_us,
            "device_share_pct": share * 100.0,
            "device_only_max_speedup_if_eliminated": (
                1.0 / (1.0 - share) if 0.0 <= share < 1.0 else float("inf")
            ),
        }

    removable = (
        float(groups["slot_mapping"]["total_time_us"])
        + float(groups["sampler_control"]["total_time_us"])
    )
    removable_share = removable / device_total_us if device_total_us else 0.0
    return {
        "files": [str(path) for path in paths],
        "device_total_time_us": device_total_us,
        "top_operators": [render_op(name, values) for name, values in ranked[:top]],
        "groups": groups,
        "slot_plus_sampler": {
            "total_time_us": removable,
            "device_share_pct": removable_share * 100.0,
            "device_only_max_speedup_if_eliminated": (
                1.0 / (1.0 - removable_share)
                if 0.0 <= removable_share < 1.0
                else float("inf")
            ),
            "warning": (
                "Device-only Amdahl bound; end-to-end gain also depends on host wait, "
                "overlap, graph replay and other stages."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be >= 1")
    result = summarize(args.csv, top=args.top)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
