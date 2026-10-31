"""Summarize the small-graph multimodal gradient-matching experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from(path: Path, orders: list[str], variant: str) -> list[dict]:
    payload = load(path)
    datasets = payload["protocol"]["datasets"]
    if len(datasets) != 1:
        raise ValueError(f"expected one dataset in {path}")
    dataset = datasets[0]
    allowed_orders = set(orders)
    rows = []
    for row in payload["rows"]:
        order = str(row["order"])
        if order not in allowed_orders:
            continue
        aggregate = row["aggregate"]
        rows.append({
            "dataset": dataset,
            "order": order,
            "variant": variant,
            "arm": row["arm"],
            "stage": int(row["stage"]),
            "f1": float(aggregate["mean_f1"]),
            "accuracy": float(aggregate["mean_accuracy"]),
        })
    return rows


def final(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["stage"] == 2]


def aggregate(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["dataset"], r["variant"], r["arm"]) for r in final(rows)})
    output = []
    for dataset, variant, arm in keys:
        subset = [r for r in final(rows)
                  if (r["dataset"], r["variant"], r["arm"])
                  == (dataset, variant, arm)]
        f1 = [r["f1"] for r in subset]
        acc = [r["accuracy"] for r in subset]
        output.append({
            "dataset": dataset,
            "variant": variant,
            "arm": arm,
            "cells": len(subset),
            "f1_mean": mean(f1),
            "f1_std": stdev(f1) if len(f1) > 1 else 0.0,
            "accuracy_mean": mean(acc),
            "accuracy_std": stdev(acc) if len(acc) > 1 else 0.0,
        })
    return output


def paired(rows: list[dict], variant: str) -> list[dict]:
    subset = [r for r in final(rows) if r["variant"] == variant]
    index = {(r["dataset"], r["order"], r["arm"]): r for r in subset}
    output = []
    for dataset in sorted({r["dataset"] for r in subset}):
        orders = sorted({r["order"] for r in subset if r["dataset"] == dataset})
        for arm in ("cat_gm_text", "cat_gm_decoupled"):
            diffs = []
            acc_diffs = []
            for order in orders:
                base = index[(dataset, order, "cat_x")]
                candidate = index.get((dataset, order, arm))
                if candidate is not None:
                    diffs.append(candidate["f1"] - base["f1"])
                    acc_diffs.append(candidate["accuracy"] - base["accuracy"])
            if diffs:
                output.append({
                    "dataset": dataset,
                    "variant": variant,
                    "arm": arm,
                    "cells": len(diffs),
                    "delta_f1": mean(diffs),
                    "delta_accuracy": mean(acc_diffs),
                    "positive_f1": sum(v > 0 for v in diffs),
                    "worst_f1": min(diffs),
                })
    return output


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.parent
    rows: list[dict] = []
    for dataset in ("cora", "citeseer"):
        rows.extend(rows_from(
            root / f"cat_gm_route_anchor_{dataset}_results.json",
            ["order19611", "order19612"], "route_anchor"))
        rows.extend(rows_from(
            root / f"cat_gm_route_anchor_order19613_{dataset}_results.json",
            ["order19613"], "route_anchor"))
        rows.extend(rows_from(
            root / f"cat_gm_decoupled_{dataset}_results.json",
            ["order19611"], "plain_decoupled"))
        rows.extend(rows_from(
            root / f"cat_gm_decoupled_order19612_{dataset}_results.json",
            ["order19612"], "plain_decoupled"))
        rows.extend(rows_from(
            root / f"cat_gm_multiseed_order19612_{dataset}_results.json",
            ["order19612"], "multi_proxy"))
    summary = aggregate(rows)
    paired_rows = paired(rows, "route_anchor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cells": rows, "summary": summary,
                                       "paired": paired_rows},
                                      indent=2, ensure_ascii=True),
                            encoding="utf-8")
    markdown = [
        "# Multimodal gradient matching small-graph summary",
        "",
        "All rows use the same prompt-only Student and current-delta-only protocol.",
        "Final-stage Macro-F1 and Accuracy are reported for each completed cell.",
        "",
        "| Dataset | Variant | Arm | Cells | Final F1 | Accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary:
        markdown.append(
            f"| {row['dataset']} | {row['variant']} | {row['arm']} | "
            f"{row['cells']} | {fmt(row['f1_mean'])} +/- {fmt(row['f1_std'])} | "
            f"{fmt(row['accuracy_mean'])} +/- {fmt(row['accuracy_std'])} |"
        )
    markdown += [
        "", "## Paired delta against CaT",
        "", "| Dataset | Variant | Arm | Cells | Delta F1 | Delta Acc | Positive F1 cells | Worst F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("route_anchor", "plain_decoupled", "multi_proxy"):
        for row in paired(rows, variant):
            markdown.append(
                f"| {row['dataset']} | {variant} | {row['arm']} | {row['cells']} | "
                f"{row['delta_f1']:+.2f} | {row['delta_accuracy']:+.2f} | "
                f"{row['positive_f1']}/{row['cells']} | {row['worst_f1']:+.2f} |"
            )
    (args.output.with_suffix(".md")).write_text("\n".join(markdown) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"cells": len(rows), "summary": len(summary)},
                     ensure_ascii=True))


if __name__ == "__main__":
    main()
