"""Summarize GraphGPS calibration JSON artifacts into a compact table."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*.json")):
        if path.name == "summary.json" or not ("_synthetic" in path.name or
                                                 "_raw" in path.name):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") != "complete":
            continue
        for run in payload.get("runs", []):
            summary = run.get("summary", {})
            rows.append({
                "dataset": payload.get("dataset"),
                "order": payload.get("order"),
                "stage": int(payload.get("stage", run.get("stage", -1))),
                "arm": run.get("arm"),
                "seed": int(run.get("seed")),
                "steps": int(run.get("steps", 0)),
                "rows": int(run.get("train_nodes", 0)),
                "final_f1": float(summary.get("final_all_cumulative_f1")),
                "stage_avg_f1": float(summary.get("stage_average_all_cumulative_f1")),
                "seen_final_f1": float(summary.get("final_seen_cumulative_f1")),
                "seconds": float(run.get("seconds", 0.0)),
                "loss_first": float(run.get("loss_first")),
                "loss_last": float(run.get("loss_last")),
            })
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["order"], row["stage"], row["arm"])].append(row)
    summary = []
    for key, values in sorted(groups.items()):
        out = {"dataset": key[0], "order": key[1], "stage": key[2], "arm": key[3],
               "n": len(values), "rows": values[0]["rows"], "steps": values[0]["steps"]}
        for metric in ("final_f1", "stage_avg_f1", "seen_final_f1", "seconds", "loss_first", "loss_last"):
            vals = np.asarray([v[metric] for v in values], dtype=float)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std())
        summary.append(out)
    payload = {"status": "complete", "rows": rows, "summary": summary}
    out_path = args.output or args.root / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "groups": len(summary), "output": str(out_path)}))
    print("dataset\tstage\tarm\tn\trows\tfinal_mean\tstage_avg_mean\tseconds_mean")
    for row in summary:
        print("\t".join(str(row.get(k, "")) for k in (
            "dataset", "stage", "arm", "n", "rows", "final_f1_mean", "stage_avg_f1_mean", "seconds_mean")))


if __name__ == "__main__":
    main()
