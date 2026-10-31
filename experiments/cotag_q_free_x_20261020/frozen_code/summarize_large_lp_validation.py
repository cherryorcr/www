import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    root = Path(root)
    records = json.loads((root / "results.json").read_text())
    rows = []
    for arm in sorted({r["arm"] for r in records}):
        values = [r["metrics"]["all"] for r in records
                  if r["arm"] == arm and r["stage"] == 4]
        rows.append({
            "dataset": records[0]["dataset"], "arm": arm,
            "final_auc": float(np.mean([v["auc"] for v in values]) * 100),
            "final_ap": float(np.mean([v["ap"] for v in values]) * 100),
            "final_auc_sd": float(np.std([v["auc"] for v in values], ddof=1) * 100),
            "final_ap_sd": float(np.std([v["ap"] for v in values], ddof=1) * 100)})
    stages = []
    for arm in sorted({r["arm"] for r in records}):
        for stage in range(5):
            values = [r["metrics"]["all"] for r in records
                      if r["arm"] == arm and r["stage"] == stage]
            stages.append({
                "dataset": records[0]["dataset"], "arm": arm, "stage": stage,
                "auc": float(np.mean([v["auc"] for v in values]) * 100),
                "ap": float(np.mean([v["ap"] for v in values]) * 100)})
    result = {
        "methods": rows, "stages": stages,
        "scope": "large-graph screening; frozen base+4 C0 chain; independent edge splits"}
    (root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
