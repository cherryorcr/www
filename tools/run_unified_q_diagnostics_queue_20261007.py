"""Resource-bounded remote experiment queue with durable exit codes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    p.add_argument("--phase", choices=("diagnostics", "confirmation", "transfer"), required=True)
    p.add_argument("--workers-per-gpu", type=int, default=2)
    args = p.parse_args()
    root = args.project
    output = root / "results/unified_q_diagnostics_20261007"
    snapshot = root / "experiments/unified_q_diagnostics_20261007"
    output.mkdir(parents=True, exist_ok=True)
    minilm = "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"
    arms = ["cat_cctp", "uq_30", "uq_64", "uq_128", "uq_strict", "uq_init",
            "uq_q_fixed", "uq_r_fixed", "uq_feature_only", "uq_edge_free",
            "uq_proxy384", "uq_prop384", "uq_v3", "uq_old_bounded", "uq_old_unbounded"]
    tasks = []
    if args.phase == "diagnostics":
        for dataset in ("wikics", "bookhis"):
            tasks.append({"kind": "ab", "dataset": dataset, "tag": f"ab_{dataset}_fixed_schedule"})
        for arm in arms:
            for dataset in ("wikics", "bookhis"):
                tasks.append({"kind": "diag", "dataset": dataset, "arms": [arm],
                              "tag": f"diag_{dataset}_{arm}", "seed": 271006})
    elif args.phase == "confirmation":
        for order in (9101, 9102, 9103):
            for seed in (271006, 271007, 271008):
                if (order, seed) == (9101, 271006):
                    continue
                for dataset in ("wikics", "bookhis"):
                    tasks.append({"kind": "diag", "dataset": dataset, "arms": ["cat_cctp", "uq_v3"],
                        "tag": f"confirm_{dataset}_o{order}_c{seed}", "seed": seed,
                        "order": order})
    else:
        for arm in ("cat_cctp", "uq_30", "uq_v3"):
            tasks.append({"kind": "diag", "dataset": "sportsfit", "arms": [arm],
                "seed": 271006, "tag": f"transfer_sportsfit_{arm}"})
    plan = {"phase": args.phase, "tasks": tasks, "primary": "uq_v3", "budget_mode": "fixed_rate",
            "student_seeds": [150, 151, 152], "student_steps": 2048,
            "student_batch": 24, "workers_per_gpu": args.workers_per_gpu,
            "timing_note": "shared GPUs; not an exclusive wall-time benchmark"}
    plan_path = output / f"{args.phase}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("queue plan changed")
    plan_path.write_text(json.dumps(plan, indent=2))

    def execute(task, gpu):
        dataset, tag = task["dataset"], task["tag"]
        cache = root / f"results/public_baselines_text_cache_20261001/{dataset}"
        environment = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        if task["kind"] == "ab":
            environment["PYTHONPATH"] = str(root / "experiments/unified_q_revision_20261007/code") + ":" + str(root / "code/v2_llm_fusion")
            source = root / f"results/unified_q_revision_verified_20261007/{dataset}_fixed_order9101_seed271006"
            cmd = [sys.executable, "-u", str(snapshot / "ab/run_unified_q_student_ab_20261007.py"),
                   "--source", str(source)]
        else:
            environment["PYTHONPATH"] = str(snapshot / "code") + ":" + str(root / "code/v2_llm_fusion")
            reference = (root / f"results/unified_q_large_full_20261007/{dataset}_unified_q_smoke.json"
                         if dataset != "sportsfit" else snapshot / "sportsfit_reference.json")
            cmd = [sys.executable, "-u", str(snapshot / "code/run_unified_q_diagnostics_20261007.py"),
                   "--dataset", dataset, "--reference", str(reference),
                   "--arms", *task["arms"], "--construction-seed", str(task["seed"])]
            if task.get("order", 9101) != 9101:
                cmd += ["--order-seed", str(task["order"])]
        cmd += ["--output", str(output / tag), "--text-cache", str(cache),
                "--minilm", minilm, "--device", f"cuda:{gpu}"]
        if shutil.disk_usage(root).free < 2 * 2**30:
            return {"tag": tag, "exit_code": -2, "error": "less than 2 GiB free; no files deleted"}
        start = time.time()
        print(json.dumps({"event": "start", "tag": tag, "gpu": gpu}), flush=True)
        with (output / f"{tag}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = {"tag": tag, "exit_code": result.returncode, "gpu": gpu,
                  "seconds": time.time() - start}
        print(json.dumps({"event": "finish", **record}), flush=True)
        return record

    lanes = [[] for _ in range(2 * args.workers_per_gpu)]
    for i, task in enumerate(tasks):
        lanes[i % len(lanes)].append(task)

    def lane(index):
        records = []
        for task in lanes[index]:
            records.append(execute(task, index % 2))
            (output / f"{args.phase}_lane{index}.json").write_text(json.dumps(records, indent=2))
        return records

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        records = [r for records in pool.map(lane, range(len(lanes))) for r in records]
    (output / f"{args.phase}_exit_codes.json").write_text(json.dumps(records, indent=2))
    if any(r["exit_code"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
