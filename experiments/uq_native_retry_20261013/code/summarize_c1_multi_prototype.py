"""Summarize and audit the C1 multi-prototype small-graph experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


DATASETS = ("cora", "citeseer")
ORDERS = ("canonical", "order1701", "order1702")
ARMS = ("dispersion", "subcluster_mean", "subcluster_weighted", "subcluster_moment")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_rows(payload: dict, dataset: str, order: str) -> list[dict]:
    by_arm = {arm: [] for arm in ARMS}
    for result in payload["results"]:
        if result["arm"] not in by_arm:
            continue
        metrics = result["metrics"]["all"]
        by_arm[result["arm"]].append({
            "dataset": dataset,
            "order": order,
            "arm": result["arm"],
            "stage": int(result["stage"]),
            "rows": int(result["rows"]),
            "f1": float(metrics["cumulative_f1"]["mean"]),
            "f1_std": float(metrics["cumulative_f1"]["std"]),
            "old_f1": None if metrics["old_f1"] is None else float(metrics["old_f1"]["mean"]),
            "new_f1": None if metrics["new_f1"] is None else float(metrics["new_f1"]["mean"]),
            "seed_count": len(result["per_seed"]),
        })
    rows = []
    for arm, stages in by_arm.items():
        stages = sorted(stages, key=lambda x: x["stage"])
        if len(stages) != 3:
            raise ValueError(f"{dataset}/{order}/{arm}: expected 3 stage results")
        final = stages[-1]
        rows.append({
            "dataset": dataset,
            "order": order,
            "arm": arm,
            "rows": final["rows"],
            "stage_f1": [x["f1"] for x in stages],
            "stage_f1_std": [x["f1_std"] for x in stages],
            "stage_average_f1": mean(x["f1"] for x in stages),
            "final_f1": final["f1"],
            "final_f1_std": final["f1_std"],
            "old_f1": final["old_f1"],
            "new_f1": final["new_f1"],
            "seed_count": min(x["seed_count"] for x in stages),
        })
    return rows


def construction_summary(root: Path, dataset: str, order: str, arm: str) -> dict:
    order_dir = root / dataset / order
    rows, subclusters, build_seconds = [], [], []
    for stage in range(3):
        audit = read_json(order_dir / f"{arm}_s{stage}_construction.json")
        rows.append(int(audit["payload"]["rows"]))
        build_seconds.append(float(audit["build_seconds"]))
        if audit.get("subcluster_count") is not None:
            subclusters.append(int(audit["subcluster_count"]))
    return {
        "rows_by_stage": rows,
        "build_seconds": sum(build_seconds),
        "subcluster_count_by_stage": subclusters,
        "subcluster_count_total": sum(subclusters) if subclusters else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, output = args.root, args.output
    rows, construction, audits = [], [], []
    for dataset in DATASETS:
        acceptance = read_json(root / dataset / "acceptance.json")
        audits.append({"dataset": dataset, "acceptance": acceptance})
        for order in ORDERS:
            payload = read_json(root / dataset / order / "results.json")
            rows.extend(result_rows(payload, dataset, order))
            for arm in ARMS:
                record = construction_summary(root, dataset, order, arm)
                construction.append({"dataset": dataset, "order": order, "arm": arm, **record})

    lookup = {(r["dataset"], r["order"], r["arm"]): r for r in rows}
    for row in rows:
        control = lookup[(row["dataset"], row["order"], "dispersion")]
        row["delta_final_vs_c0"] = row["final_f1"] - control["final_f1"]
        row["delta_stage_average_vs_c0"] = row["stage_average_f1"] - control["stage_average_f1"]
        row["delta_old_vs_c0"] = None if row["old_f1"] is None else row["old_f1"] - control["old_f1"]
        row["delta_new_vs_c0"] = None if row["new_f1"] is None else row["new_f1"] - control["new_f1"]

    aggregates = []
    for dataset in DATASETS:
        for arm in ARMS:
            subset = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            aggregates.append({
                "dataset": dataset,
                "arm": arm,
                "final_f1_mean": mean(r["final_f1"] for r in subset),
                "final_f1_order_sd": pstdev(r["final_f1"] for r in subset),
                "stage_average_f1_mean": mean(r["stage_average_f1"] for r in subset),
                "old_f1_mean": mean(r["old_f1"] for r in subset),
                "new_f1_mean": mean(r["new_f1"] for r in subset),
                "delta_final_vs_c0": mean(r["delta_final_vs_c0"] for r in subset),
                "delta_stage_average_vs_c0": mean(r["delta_stage_average_vs_c0"] for r in subset),
                "delta_old_vs_c0": mean(r["delta_old_vs_c0"] for r in subset),
                "delta_new_vs_c0": mean(r["delta_new_vs_c0"] for r in subset),
                "wins_final_vs_c0": sum(r["delta_final_vs_c0"] > 0 for r in subset),
                "wins_stage_average_vs_c0": sum(r["delta_stage_average_vs_c0"] > 0 for r in subset),
                "seed_count": min(r["seed_count"] for r in subset),
            })

    costs = []
    for dataset in DATASETS:
        for order in ORDERS:
            payload = read_json(root / dataset / order / "results.json")
            for arm in ARMS:
                c = payload["costs"][arm]
                costs.append({
                    "dataset": dataset,
                    "order": order,
                    "arm": arm,
                    "build_chain_seconds": float(c["build_chain_seconds_with_io"]),
                    "stage_build_seconds_sum": float(c["stage_build_seconds_sum"]),
                    "student_seconds": float(c["student_seconds"]),
                    "student_fits": int(c["student_fits"]),
                    "student_steps": int(c["student_steps"]),
                })

    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "c1-multi-prototype-small-v1",
        "datasets": list(DATASETS),
        "orders": list(ORDERS),
        "arms": list(ARMS),
        "rows": rows,
        "aggregates": aggregates,
        "construction": construction,
        "costs": costs,
        "acceptance": audits,
        "external_reference": {
            "k_center_final_order_mean": {"cora": 49.40, "citeseer": 57.73},
            "kmeans_medoid_final_order_mean": {"cora": 54.24, "citeseer": 51.08},
            "source": "P1X_SYNTHETIC_X_COMPLETION_ZH_2026-09-18.md",
        },
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(aggregates[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)
    with (output / "per_order.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [k for k, v in rows[0].items() if not isinstance(v, list)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    print(json.dumps({"aggregates": aggregates, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
