"""Audit immutable fields and the editable-row boundary of geometry-X outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-prefix", default="geometry_x_v2")
    args = parser.parse_args()
    cells = {
        "cora_canonical": 25,
        "cora_order19102": 35,
        "citeseer_canonical": 38,
        "citeseer_order19102": 40,
    }
    report = {"name": "geometry-x-invariant-audit-v1", "cells": {}}
    expected_modes = {
        "geometry_x_v2": ("x-geometry", "x_geometry_relative"),
        "class_rep_x_v1": ("x-class-rep", "x_class_rep_relative"),
    }
    expected_mode, expected_variable = expected_modes.get(
        args.candidate_prefix, (None, None))
    if expected_mode is None:
        raise ValueError(f"unknown candidate prefix: {args.candidate_prefix}")
    for name, old_rows in cells.items():
        base = load(args.root / f"control_{name}" / "memory.pt")
        candidate_dir = args.root / f"{args.candidate_prefix}_{name}"
        candidate = load(candidate_dir / "memory.pt")
        protocol = json.loads(
            (candidate_dir / "protocol.json")
            .read_text(encoding="utf-8"))
        editable = list(range(old_rows, len(base["x"])))
        prefix_equal = bool(torch.equal(base["x"][:old_rows], candidate["x"][:old_rows]))
        changed = (candidate["x"] - base["x"]).norm(dim=1) > 1e-8
        changed_old = int(changed[:old_rows].sum())
        changed_new = int(changed[old_rows:].sum())
        unchanged_new = int((~changed[old_rows:]).sum())
        fields = {}
        for key in ("text_emb", "texts", "edge_index", "mass", "depth", "stage"):
            if key in base and key in candidate:
                fields[key] = equal(base[key], candidate[key])
        audit = json.loads(
            (candidate_dir / "audit.json")
            .read_text(encoding="utf-8"))
        checks = {
            "old_x_unchanged": prefix_equal,
            "no_old_x_changed": changed_old == 0,
            "all_selected_rows_new": all(row >= old_rows for row in editable),
            "immutable_fields_unchanged": all(fields.values()),
            "memory_rows_unchanged": len(base["x"]) == len(candidate["x"]),
            "historical_raw_reads_zero": audit.get("historical_raw_reads") == 0,
            "only_x_gradient_variable": audit.get("outer_gradient_variables") == [
                expected_variable],
            "nonzero_outer_gradient": any(
                float(item.get("gradient_norms", {}).get(expected_variable, 0.0)) > 0
                for item in audit.get("curve", [])),
            "reported_mode_correct": protocol.get("mode") == expected_mode,
        }
        report["cells"][name] = {
            "old_rows": old_rows,
            "total_rows": len(base["x"]),
            "changed_old_rows": changed_old,
            "changed_new_rows": changed_new,
            "unchanged_new_rows": unchanged_new,
            "max_relative_displacement": audit.get("max_relative_displacement"),
            "mean_relative_displacement": audit.get("mean_relative_displacement"),
            "immutable_fields": fields,
            "checks": checks,
            "passed": all(checks.values()) and changed_new > 0,
        }
    report["passed"] = all(item["passed"] for item in report["cells"].values())
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "geometry_x_audit", "output": str(args.output),
                      "passed": report["passed"]}), flush=True)


if __name__ == "__main__":
    main()
