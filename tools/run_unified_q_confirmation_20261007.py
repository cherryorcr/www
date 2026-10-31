"""Bounded two-GPU confirmation queue, with no parameter selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-pids", nargs="*", type=int, default=[])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers_per_gpu <= 3:
        parser.error("workers-per-gpu must be in [1, 3]")
    args.output.mkdir(parents=True, exist_ok=True)
    for pid in args.wait_pids:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
    runner = args.project / "experiments/unified_q_revision_20261007/code/run_unified_q_revision_20261007.py"
    tasks = []
    for dataset in ("wikics", "bookhis"):
        for order in (9101, 9102, 9103):
            for seed in (271006, 271007, 271008):
                if order == 9101 and seed == 271006:
                    continue
                tasks.append((dataset, order, seed, "fixed_schedule", ["cat_cctp", "uq_semantic"]))
        tasks.append((dataset, 9101, 271006, "fixed_rate", ["cat_cctp", "uq_balanced", "uq_semantic"]))
    plan = {"tasks": tasks, "primary": "uq_semantic", "student_seeds": [150, 151, 152],
            "worker_count": 2 * args.workers_per_gpu,
            "timing_note": "confirmation jobs share GPUs; not exclusive efficiency benchmarks"}
    (args.output / "confirmation_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def run_task(task, gpu):
        dataset, order, seed, policy, arms = task
        tag = f"{dataset}_{policy}_order{order}_seed{seed}"
        output = args.output / tag
        command = [sys.executable, "-u", str(runner), "--dataset", dataset,
                   "--reference", str(args.project / f"results/unified_q_large_full_20261007/{dataset}_unified_q_smoke.json"),
                   "--text-cache", str(args.project / f"results/public_baselines_text_cache_20261001/{dataset}"),
                   "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                   "--output", str(output), "--device", f"cuda:{gpu}",
                   "--construction-seed", str(seed), "--budget-mode", policy,
                   "--arms", *arms]
        if order != 9101:
            command.extend(("--order-seed", str(order)))
        environment = dict(os.environ, PYTHONPATH=str(args.project / "code/v2_llm_fusion"),
                           TOKENIZERS_PARALLELISM="false")
        print(json.dumps({"event": "start", "cell": tag, "gpu": gpu}), flush=True)
        with (args.output / f"{tag}.log").open("w", encoding="utf-8") as log:
            value = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = {"cell": tag, "gpu": gpu, "exit_code": value.returncode}
        print(json.dumps(dict(event="finish", **record)), flush=True)
        return record

    # Fixed lanes avoid oversubscribing a device when shorter jobs complete.
    lanes = [[] for _ in range(2 * args.workers_per_gpu)]
    for i, task in enumerate(tasks):
        lanes[i % len(lanes)].append(task)

    def run_lane(index):
        return [run_task(task, index % 2) for task in lanes[index]]

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        results = [record for records in pool.map(run_lane, range(len(lanes))) for record in records]
    (args.output / "confirmation_exit_codes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(r["exit_code"] != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
