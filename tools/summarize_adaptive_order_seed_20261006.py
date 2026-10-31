"""Aggregate the Adaptive coherent order/seed confirmation outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


MAIN = "cat_mmd_tag_x_conditioned_routing_coherent_adaptive"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def parse_name(name: str) -> tuple[str, int, int] | None:
    prefix, suffix = "adaptive_order_", "_20261006"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    body = name[len(prefix):-len(suffix)]
    dataset, order, construction = body.rsplit("_o", 1)[0], None, None
    if "_c" not in body or "_o" not in body:
        return None
    dataset, rest = body.split("_o", 1)
    order_text, construction_text = rest.split("_c", 1)
    try:
        return dataset, int(order_text), int(construction_text)
    except ValueError:
        return None


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for directory in sorted(root.glob("adaptive_order_*_20261006")):
        identity = parse_name(directory.name)
        if identity is None or not (directory / "summary.json").exists():
            continue
        dataset, order_seed, construction_seed = identity
        payload = load(directory / "summary.json")
        overall = payload.get("overall", {})
        if isinstance(overall, list):
            items = ((value.get("arm"), value) for value in overall)
        else:
            items = overall.items()
        for arm, value in items:
            rows.append({
                "dataset": dataset,
                "order_seed": order_seed,
                "construction_seed": construction_seed,
                "arm": arm,
                **{key: value.get(key) for key in (
                    "final_f1", "final_f1_std", "final_accuracy",
                    "final_accuracy_std", "old_f1", "new_f1",
                    "stage_avg_f1", "stage_avg_accuracy",
                    "worst_stage_f1", "construction_seconds",
                    "student_seconds")},
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["dataset"], r["arm"]) for r in rows})
    output = []
    for dataset, arm in keys:
        subset = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
        value = {"dataset": dataset, "arm": arm, "cells": len(subset)}
        for key in ("final_f1", "final_accuracy", "old_f1", "new_f1",
                    "stage_avg_f1", "stage_avg_accuracy", "worst_stage_f1",
                    "construction_seconds", "student_seconds"):
            values = [float(r[key]) for r in subset if r.get(key) is not None]
            value[key] = mean(values) if values else None
            value[key + "_std"] = sample_std(values) if values else None
        output.append(value)
    return output


def paired(rows: list[dict], reference: str = "cat_mmd_tag_cctp_top5") -> list[dict]:
    index = {(r["dataset"], r["order_seed"], r["construction_seed"], r["arm"]): r
             for r in rows}
    arms = sorted({r["arm"] for r in rows if r["arm"] != reference})
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        cells = [(o, c) for d, o, c, a in index if d == dataset and a == reference]
        for arm in arms:
            diffs = []
            for order_seed, construction_seed in cells:
                left = index.get((dataset, order_seed, construction_seed, arm))
                base = index.get((dataset, order_seed, construction_seed, reference))
                if left and base:
                    diffs.append(float(left["final_f1"]) - float(base["final_f1"]))
            if diffs:
                out.append({
                    "dataset": dataset,
                    "arm": arm,
                    "reference": reference,
                    "cells": len(diffs),
                    "mean_delta_f1": mean(diffs),
                    "std_delta_f1": sample_std(diffs),
                    "positive_cells": sum(v > 0 for v in diffs),
                    "positive_ratio": sum(v > 0 for v in diffs) / len(diffs),
                    "worst_delta_f1": min(diffs),
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cells.csv", rows)
    summary = summarize(rows)
    pairs = paired(rows)
    (args.output / "summary.json").write_text(
        json.dumps({"cells": rows, "summary": summary, "paired": pairs},
                   indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# Adaptive coherent order/seed confirmation",
        "",
        "All cells use the same Student, memory budget, current-delta-only access and base+4 group sizes.",
        "",
        "| Dataset | Arm | Cells | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(key: str) -> str:
            value, spread = row.get(key), row.get(key + "_std")
            return "-" if value is None else f"{value:.2f} +/- {spread:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {row['cells']} | "
            f"{fmt('final_f1')} | {fmt('final_accuracy')} | "
            f"{fmt('old_f1')} | {fmt('new_f1')} | {fmt('stage_avg_f1')} | "
            f"{fmt('worst_stage_f1')} |")
    lines += ["", "## Paired comparison to CCTP top-5", "",
              "| Dataset | Arm | Cells | Mean delta F1 | Positive cells | Worst delta |",
              "|---|---|---:|---:|---:|---:|"]
    for row in pairs:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {row['cells']} | "
            f"{row['mean_delta_f1']:+.2f} | "
            f"{row['positive_cells']}/{row['cells']} "
            f"({100*row['positive_ratio']:.1f}%) | "
            f"{row['worst_delta_f1']:+.2f} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"cells": len(rows), "summary": len(summary),
                      "paired": len(pairs)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
