"""Summarize the paired P3-A response-off/on experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else None


def arm_rows(payload, arm):
    return [row for row in payload["results"] if row["arm"] == arm]


def summarize_order(order_dir: Path):
    manifest = read(order_dir / "manifest.json")
    payload = read(order_dir / "results.json")
    rows = {}
    for arm in ("dispersion", "v4"):
        stages = sorted(arm_rows(payload, arm), key=lambda x: x["stage"])
        if len(stages) != 3:
            raise ValueError(f"{order_dir}: {arm} has {len(stages)} stages")
        final = stages[-1]["metrics"]["all"]["cumulative_f1"]["mean"]
        stage_avg = mean([s["metrics"]["all"]["cumulative_f1"]["mean"] for s in stages])
        old = stages[-1]["metrics"]["all"]["old_f1"]["mean"]
        new = stages[-1]["metrics"]["all"]["new_f1"]["mean"]
        costs = payload["costs"][arm]
        audits = payload["audits"].get(arm, [])
        selected = []
        for audit in audits:
            selection = audit.get("selection") or {}
            selected.append({
                "stage": audit.get("stage"),
                "base_allocation": selection.get("base_allocation"),
                "selected_class_rows": selection.get("selected_class_rows"),
                "decisions": len(selection.get("decisions", [])),
                "tensor_sha256": audit.get("tensor_sha256"),
            })
        rows[arm] = {
            "final_f1": final,
            "stage_average_f1": stage_avg,
            "old_f1": old,
            "new_f1": new,
            "final_seed_std": stages[-1]["metrics"]["all"]["cumulative_f1"]["std"],
            "budgets": manifest["budgets"],
            "cost": {
                "build_seconds": costs.get("build_chain_seconds_with_io"),
                "selector_seconds": costs.get("selector_seconds", 0.0),
                "selector_runs": costs.get("selector_runs", 0),
                "selector_steps": costs.get("selector_steps", 0),
                "student_seconds_one_seed": costs.get("student_wall_seconds", 0.0) / 6.0,
            },
            "selected": selected,
        }
    rows["delta_v4_minus_dispersion"] = {
        key: rows["v4"][key] - rows["dispersion"][key]
        for key in ("final_f1", "stage_average_f1", "old_f1", "new_f1")
    }
    rows["cost_ratio_selector_to_dispersion_build"] = (
        rows["v4"]["cost"]["selector_seconds"] /
        max(rows["dispersion"]["cost"]["build_seconds"], 1e-9)
    )
    return {
        "dataset": manifest["dataset"],
        "order": manifest["order"],
        "enumeration_protocol": manifest.get("enumeration_protocol"),
        "class_order_policy": manifest.get("class_order_policy"),
        "stage_ids_sha256": manifest.get("stage_ids_sha256"),
        "rows": rows,
    }


def summarize(root: Path):
    entries = []
    for dataset in ("cora", "citeseer"):
        for order_dir in sorted((root / dataset).glob("*/")):
            if (order_dir / "manifest.json").exists() and (order_dir / "results.json").exists():
                entries.append(summarize_order(order_dir))
    if len(entries) != 6:
        raise ValueError(f"expected six completed dataset/order entries, got {len(entries)}")
    aggregate = {}
    for dataset in ("cora", "citeseer"):
        subset = [x for x in entries if x["dataset"] == dataset]
        aggregate[dataset] = {
            "orders": [x["order"] for x in subset],
            "mean_final_delta": mean([x["rows"]["delta_v4_minus_dispersion"]["final_f1"] for x in subset]),
            "mean_stage_average_delta": mean([x["rows"]["delta_v4_minus_dispersion"]["stage_average_f1"] for x in subset]),
            "min_final_delta": min(x["rows"]["delta_v4_minus_dispersion"]["final_f1"] for x in subset),
            "max_final_delta": max(x["rows"]["delta_v4_minus_dispersion"]["final_f1"] for x in subset),
            "mean_v4_selector_seconds": mean([x["rows"]["v4"]["cost"]["selector_seconds"] for x in subset]),
            "mean_dispersion_build_seconds": mean([x["rows"]["dispersion"]["cost"]["build_seconds"] for x in subset]),
            "mean_selector_to_dispersion_build_ratio": mean([x["rows"]["cost_ratio_selector_to_dispersion_build"] for x in subset]),
            "changed_stage_count": sum(
                1 for x in subset for item in x["rows"]["v4"]["selected"]
                if item.get("tensor_sha256") != next(
                    y["tensor_sha256"] for y in x["rows"]["dispersion"]["selected"]
                    if y.get("stage") == item.get("stage"))
            ) if all(x["rows"]["dispersion"]["selected"] for x in subset) else None,
        }
    return {"entries": entries, "aggregate": aggregate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    value = summarize(args.root)
    serialized = json.dumps(value, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
