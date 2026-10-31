"""Dev-only configuration launcher for the unified v3 protocol.

The launcher creates an explicit finite grid and delegates each point to the
unified runner.  It never changes the test split and writes a manifest that
records every searched configuration.  Selection should be performed from the
generated results before one configuration is frozen for final evaluation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--text-operators", nargs="+",
                    default=["concat", "mean_pool", "weighted_pool", "set_pool"])
    ap.add_argument("--refresh-modes", nargs="+",
                    default=["frozen_old", "bounded_refresh"])
    ap.add_argument("--dynamic-rows", action="store_true")
    ap.add_argument("--student-seeds", type=int, nargs="+", default=[200, 201, 202])
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(args.rates, args.text_operators,
                                  args.refresh_modes))
    if args.max_runs is not None:
        grid = grid[:max(0, int(args.max_runs))]
    manifest = {"dataset": args.dataset, "grid": [], "test_accessed": False,
                "selection_rule": "choose using dev-only results, then freeze one configuration"}
    runner = Path(__file__).with_name("run_unified_v3_continual.py")
    for index, (rate, operator, refresh) in enumerate(grid):
        name = f"rate{rate:g}_{operator}_{refresh}"
        run_out = out / name
        cmd = [sys.executable, str(runner), "--dataset", args.dataset,
               "--output", str(run_out), "--device", args.device,
               "--rate", str(rate), "--arms", "append_fixed",
               "--text-operator", operator, "--refresh-mode", refresh,
               "--student-seeds", *[str(v) for v in args.student_seeds]]
        if args.data is not None:
            cmd.extend(["--data", args.data])
        if args.minilm is not None:
            cmd.extend(["--minilm", args.minilm])
        if args.student_steps is not None:
            cmd.extend(["--student-steps", str(args.student_steps)])
        if args.dynamic_rows:
            cmd.append("--dynamic-rows")
        subprocess.run(cmd, check=True)
        manifest["grid"].append({"index": index, "rate": rate,
                                 "text_operator": operator,
                                 "refresh_mode": refresh,
                                 "dynamic_rows": bool(args.dynamic_rows),
                                 "output": str(run_out)})
        (out / "calibration_manifest.json").write_text(
            json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"runs": len(grid), "output": str(out),
                      "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
