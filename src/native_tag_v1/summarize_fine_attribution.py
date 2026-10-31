"""Summarize fine attribution JSON runs into compact tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def read_results(path: Path):
    rows = json.loads((path / "results.json").read_text(encoding="utf-8"))["results"]
    out = {}
    for row in rows:
        vals = [m["checkpoint"]["macro_f1"] for m in row["model_runs"]]
        stages = [[x["macro_f1"] for x in m["checkpoint"]["per_stage"]] for m in row["model_runs"]]
        out[row["arm"]] = {
            "f1": mean(vals), "student_f1": vals,
            "per_stage": [mean(x) for x in zip(*stages)],
            "student_steps": [m["selected_step"] for m in row["model_runs"]],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-runs", nargs="+", required=True)
    p.add_argument("--quality-runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    old = {Path(x).name: read_results(Path(x)) for x in args.old_runs}
    quality = {Path(x).name: read_results(Path(x)) for x in args.quality_runs}
    payload = {"old_update": old, "new_quality": quality}
    # Cross-seed summaries are intentionally descriptive; only two seeds here.
    for family, runs in list(payload.items()):
        arms = sorted({a for result in runs.values() for a in result})
        payload[family + "_aggregate"] = {}
        for arm in arms:
            vals = [result[arm]["f1"] for result in runs.values() if arm in result]
            stages = [result[arm]["per_stage"] for result in runs.values() if arm in result]
            payload[family + "_aggregate"][arm] = {
                "seed_values": vals,
                "mean": mean(vals),
                "population_sd": pstdev(vals) if len(vals) > 1 else 0.0,
                "per_stage_mean": [mean(x) for x in zip(*stages)] if stages else [],
            }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
