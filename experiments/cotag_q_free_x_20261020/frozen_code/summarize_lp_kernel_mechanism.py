"""Aggregate mechanism-control LP results against the parent C0/XTG runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader(); out.writerows(rows)


def summarize(root):
    all_methods, diffs, stage_rows, audits = [], [], [], []
    for dataset in ("cora", "citeseer"):
        parent = json.loads((root.parent / "link_prediction_20260920" / "summary.json").read_text())
        pmethods = {(r["arm"], r["final_auc"], r["final_ap"]): r
                    for r in parent["methods"] if r["dataset"] == dataset}
        folder = root / dataset
        manifest = json.loads((folder / "manifest.json").read_text())
        records = json.loads((folder / "results.json").read_text())
        audits.append(json.loads((folder / "audit.json").read_text()))
        keyed = {(r["split"], r["seed"], r["arm"], r["stage"]): r for r in records}
        c0 = {(s, seed, 2): keyed[(s, seed, "c0_replay", 2)] for s in manifest["splits"] for seed in manifest["seeds"]}
        for arm in manifest["arms"]:
            values = [keyed[(s, seed, arm, 2)]["metrics"]["all"] for s in manifest["splits"] for seed in manifest["seeds"]]
            row = {"dataset": dataset, "arm": arm,
                   "final_auc": float(np.mean([v["auc"] for v in values]) * 100),
                   "final_ap": float(np.mean([v["ap"] for v in values]) * 100),
                   "final_auc_split_sd": float(np.std([np.mean([keyed[(s, seed, arm, 2)]["metrics"]["all"]["auc"] for seed in manifest["seeds"]]) * 100 for s in manifest["splits"]], ddof=1)),
                   "final_ap_split_sd": float(np.std([np.mean([keyed[(s, seed, arm, 2)]["metrics"]["all"]["ap"] for seed in manifest["seeds"]]) * 100 for s in manifest["splits"]], ddof=1))}
            if arm == "c0_replay":
                parent_c0 = pmethods[next(k for k in pmethods if k[0] == "c0")]
                row["parent_c0_auc"] = parent_c0["final_auc"]
                row["parent_c0_ap"] = parent_c0["final_ap"]
            else:
                row["parent_c0_auc"] = pmethods[next(k for k in pmethods if k[0] == "c0")]["final_auc"]
                row["parent_c0_ap"] = pmethods[next(k for k in pmethods if k[0] == "c0")]["final_ap"]
            row["delta_parent_c0_auc"] = row["final_auc"] - row["parent_c0_auc"]
            row["delta_parent_c0_ap"] = row["final_ap"] - row["parent_c0_ap"]
            all_methods.append(row)
            for metric in ("auc", "ap"):
                for split in manifest["splits"]:
                    delta = [(keyed[(split, seed, arm, 2)]["metrics"]["all"][metric] -
                              keyed[(split, seed, "c0_replay", 2)]["metrics"]["all"][metric]) * 100 for seed in manifest["seeds"]]
                    diffs.append({"dataset": dataset, "arm": arm, "metric": metric, "split": split,
                                  "delta_c0_replay_pp": float(np.mean(delta)), "paired_seed_sd": float(np.std(delta, ddof=1)),
                                  "positive_seed_pairs": sum(v > 0 for v in delta), "seed_pairs": len(delta)})
            for stage in range(3):
                for metric in ("auc", "ap"):
                    vals = [keyed[(s, seed, arm, stage)]["metrics"]["all"][metric] * 100 for s in manifest["splits"] for seed in manifest["seeds"]]
                    stage_rows.append({"dataset": dataset, "arm": arm, "stage": stage, "metric": metric,
                                       "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))})
    result = {"methods": all_methods, "paired_differences": diffs, "stages": stage_rows, "audits": audits,
              "interpretation": "Mechanism development controls reuse parent edge splits; not independent generalization."}
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for name in ("methods", "paired_differences", "stages"):
        write(root / (name + ".csv"), result[name])
    print(json.dumps({"methods": all_methods, "audits": audits}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    summarize(Path(p.parse_args().root))
