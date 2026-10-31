"""Audit real construction cost for sealed edge-aware X+P/XPA runs.

The public-baseline runner records only load/audit time for sealed memories.
This script reads the construction audit emitted by the X+P runner and joins
it with the actual memory sizes and Student timing in results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_rows(root: Path) -> list[dict]:
    result = read(root / "results.json")
    rows = []
    for row in result["rows"]:
        stage = int(row["stage"])
        audit_path = root / f"stage{stage}" / "audit.json"
        audit = read(audit_path) if audit_path.exists() else {
            "config": {}, "curve": [], "selected_outer_step": None,
        }
        curve = audit.get("curve") or [{}]
        first_grad = curve[0].get("gradient_norms", {})
        rows.append({
            "dataset": root.name.removesuffix("_base4"),
            "stage": stage,
            "memory_rows": int(row["memory_rows"]),
            "new_rows": int(row["memory_rows"] - (0 if stage == 0 else
                                                    result["rows"][stage - 1]["memory_rows"])),
            "construction_seconds": float(audit.get("construction_seconds_wall", 0.0)),
            "outer_steps": int(audit.get("config", {}).get("outer_steps", 0)),
            "inner_steps": int(audit.get("config", {}).get("inner_steps", 0)),
            "grad_x": float(first_grad.get("x_class_relative", 0.0)),
            "grad_p": float(first_grad.get("p_logits", 0.0)),
            "selected_outer_step": audit.get("selected_outer_step"),
            "student_seconds": float(row.get("student_wall_seconds", 0.0)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing <dataset>_base4 result folders")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*_base4")):
        if (path / "results.json").exists():
            rows.extend(stage_rows(path))
    if not rows:
        raise SystemExit("no X+P result folders found")

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    summary = []
    for dataset, values in sorted(by_dataset.items()):
        summary.append({
            "dataset": dataset,
            "stages": len(values),
            "construction_seconds_total": sum(v["construction_seconds"] for v in values),
            "construction_seconds_mean_stage": sum(v["construction_seconds"] for v in values) / len(values),
            "student_seconds_total": sum(v["student_seconds"] for v in values),
            "outer_steps_total": sum(v["outer_steps"] for v in values),
            "inner_steps_total": sum(v["inner_steps"] for v in values),
            "max_first_step_grad_x": max(v["grad_x"] for v in values),
            "max_first_step_grad_p": max(v["grad_p"] for v in values),
        })
    payload = {
        "protocol": "edge-aware X+P real construction audit 2026-10-01",
        "source_root": str(args.root),
        "rows": rows,
        "summary": summary,
        "limitations": [
            "Student wall time is inherited from the X+P runner and includes report-seed retraining.",
            "The audit covers X+P/XPA construction, not the sealed-source loader time in the public baseline runner.",
            "A topology revision is included when the source run used x-class+p+a; the current main candidate is X+P with A reported separately.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# X+P/XPA real construction cost audit",
        "",
        "| Dataset | Construction s | Mean stage s | Student s | Outer steps | Inner steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append("| {dataset} | {construction_seconds_total:.2f} | {construction_seconds_mean_stage:.2f} | {student_seconds_total:.2f} | {outer_steps_total} | {inner_steps_total} |".format(**row))
    md = args.output.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
