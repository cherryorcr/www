"""Aggregate matched splits/seeds without treating model seeds as datasets."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    methods, differences, stages, scopes, costs, cosine, topology, audit = [], [], [], [], [], [], [], []
    for dataset in ("cora", "citeseer"):
        folder = root / dataset
        manifest = json.loads((folder / "manifest.json").read_text())
        records = json.loads((folder / "results.json").read_text())
        audited = json.loads((folder / "audit.json").read_text())
        assert audited["passed"] and audited["fits"] == len(records) == 270
        audit.append({k: v for k, v in audited.items() if k not in ("topology", "float64_records")})
        topology.extend(dict(dataset=dataset, **r) for r in audited["topology"])
        keyed = {(r["split"], r["seed"], r["arm"], r["stage"]): r for r in records}
        for arm in manifest["arms"]:
            for stage in range(3):
                rows = [r for r in records if r["arm"] == arm and r["stage"] == stage]
                for scope in ("all", "old", "new", "cross", "within_new"):
                    if rows[0]["metrics"][scope]["auc"] is None:
                        continue
                    row = {"dataset": dataset, "arm": arm, "stage": stage, "scope": scope,
                           "positives": rows[0]["metrics"][scope]["positives"]}
                    for metric in ("auc", "ap"):
                        values = [r["metrics"][scope][metric] * 100 for r in rows]
                        row[metric] = float(np.mean(values))
                        row[metric + "_seed_split_sd"] = float(np.std(values, ddof=1))
                    if scope == "all":
                        stages.append(row)
                    scopes.append(row)
            row = {"dataset": dataset, "arm": arm}
            for metric in ("auc", "ap"):
                split_means = [np.mean([keyed[(split, seed, arm, 2)]["metrics"]["all"][metric] * 100
                                       for seed in manifest["student_seeds"]]) for split in manifest["split_seeds"]]
                row["final_" + metric] = float(np.mean(split_means))
                row["final_" + metric + "_float64_reload"] = float(np.mean([
                    r["metrics"]["all"][metric] * 100 for r in audited["float64_records"]
                    if r["arm"] == arm and r["stage"] == 2]))
                row["final_" + metric + "_split_mean_sd"] = float(np.std(split_means, ddof=1))
                row["stage_mean_" + metric] = float(np.mean([r["metrics"]["all"][metric] * 100 for r in records if r["arm"] == arm]))
                for split in manifest["split_seeds"]:
                    delta = [(keyed[(split, seed, arm, 2)]["metrics"]["all"][metric] -
                              keyed[(split, seed, "c0", 2)]["metrics"]["all"][metric]) * 100 for seed in manifest["student_seeds"]]
                    differences.append({"dataset": dataset, "arm": arm, "metric": metric, "split": split,
                                        "delta_c0_pp": float(np.mean(delta)), "paired_seed_sd": float(np.std(delta, ddof=1)),
                                        "positive_seed_pairs": sum(v > 0 for v in delta), "seed_pairs": len(delta)})
            methods.append(row)
            train_time = [sum(keyed[(split, seed, arm, stage)]["cost"]["train_seconds"] for stage in range(3))
                          for split in manifest["split_seeds"] for seed in manifest["student_seeds"]]
            final = keyed[(manifest["split_seeds"][0], manifest["student_seeds"][0], arm, 2)]
            costs.append({"dataset": dataset, "arm": arm, "train_three_stages_seconds": float(np.mean(train_time)),
                          "final_rows": final["cost"]["train_rows"], "final_pairs": final["cost"]["train_positive_pairs"],
                          "final_edge_batch": final["cost"]["edge_batch"], "logical_bytes": None,
                          "cached_c0_three_stage_seconds": None, "shared_rebuild_all_arms_seconds": None})
        for split in manifest["split_seeds"]:
            construction = json.loads((folder / f"split_{split}/construction.json").read_text())
            for row in costs:
                if row["dataset"] == dataset and row["arm"] != "full_raw_equal" and row["logical_bytes"] is None:
                    row["logical_bytes"] = construction["stage_records"][-1]["arms"][row["arm"]]["payload"]["logical_bytes"]
                    row["cached_c0_three_stage_seconds"] = sum(r["cached_c0_build_seconds"] for r in construction["stage_records"])
                    row["shared_rebuild_all_arms_seconds"] = construction["shared_rebuild_seconds"]
            cos = json.loads((folder / f"split_{split}/cosine_s2.json").read_text())
            for method, scores in cos.items():
                cosine.append({"dataset": dataset, "split": split, "method": method,
                               "auc": scores["all"]["auc"] * 100, "ap": scores["all"]["ap"] * 100})
    summary = dict(methods=methods, paired_final_differences=differences, stages=stages, scopes=scopes,
                   costs=costs, cosine=cosine, topology=topology, audit=audit,
                   scope="development transfer on masked pool edges; 3 splits x 6 student seeds; AP uses balanced negatives")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for name in ("methods", "paired_final_differences", "stages", "scopes", "costs", "cosine", "topology"):
        write_csv(root / (name + ".csv"), summary[name])
    print(json.dumps({"methods": methods, "costs": costs, "audit": audit}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    summarize(Path(p.parse_args().root))
