"""Frozen confirmation queue. Only current-run artifacts are written."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    p.add_argument("--wait-pids", nargs="*", type=int, default=[])
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = p.parse_args()
    root = args.project
    output = root / "results/unified_q_inloop_validation_20261007"
    code = root / "experiments/unified_q_inloop_cat_20261007/code"
    runner = code / "run_unified_q_inloop_cat_20261007.py"
    arms = ["uq_dm_edgefree", "uq_dm_edgefree_no_gm", "cat_cctp"]
    tasks = [dict(dataset=d, order=o, seed=c, tag=f"confirm_{d}_o{o}_c{c}")
             for o in (9101, 9102, 9103) for c in (271006, 271007, 271008)
             for d in ("wikics", "bookhis", "sportsfit")
             if (o, c) != (9101, 271006)]
    references = {d: root / f"results/unified_q_inloop_cat_20261007/{d}/protocol.json"
                  for d in ("wikics", "bookhis", "sportsfit")}
    plan = dict(tasks=tasks, arms=arms, student_seeds=[150, 151, 152], student_steps=2048,
        student_batch=24, budget_mode="fixed_schedule", candidate="uq_dm_edgefree",
        no_gm_is_ablation=True, parameter_search=False, evaluation="report-dev",
        canonical_reuse="pilot reference order and seed, verify exact inputs before pooling",
        references={d: json.loads(path.read_text()) for d, path in references.items()},
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(), minimum_free_gib=1.5)
    path = output / "confirmation_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("existing preregistration differs")
    write(path, plan)
    for pid in args.wait_pids:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)

    def lane(gpu):
        records = []
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            target = output / task["tag"]
            result = target / "results.json"
            if result.exists() and json.loads(result.read_text()).get("status") == "complete":
                records.append(dict(task=task, reused=True, exit_code=0))
                continue
            if shutil.disk_usage(root).free < 1.5 * 2**30:
                records.append(dict(task=task, exit_code=-2, error="disk guard; no deletions"))
                break
            command = [sys.executable, "-u", str(runner), "--dataset", task["dataset"],
                "--reference", str(references[task["dataset"]]), "--text-cache",
                str(root / f"results/public_baselines_text_cache_20261001/{task['dataset']}"),
                "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                "--output", str(target), "--device", f"cuda:{gpu}", "--construction-seed",
                str(task["seed"]), "--budget-mode", "fixed_schedule", "--arms", *arms]
            if task["order"] != 9101:
                command += ["--order-seed", str(task["order"])]
            env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                PYTHONPATH=f"{code}:{root / 'code/v2_llm_fusion'}")
            print(json.dumps(dict(event="start", gpu=gpu, **task)), flush=True)
            started = time.time()
            with (output / f"{task['tag']}.log").open("a", encoding="utf-8") as log:
                child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = dict(task=task, exit_code=child.returncode, seconds=time.time()-started, gpu=gpu)
            records.append(record)
            write(output / f"confirmation_gpu{gpu}.json", records)
            print(json.dumps(dict(event="finish", **record)), flush=True)
        return records

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        records = [r for items in pool.map(lane, args.gpus) for r in items]
    complete = len(records) == len(tasks) and all(r["exit_code"] == 0 for r in records)
    write(output / "confirmation_status.json", dict(status="complete" if complete else "incomplete", records=records))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
