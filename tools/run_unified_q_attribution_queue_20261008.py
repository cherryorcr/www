"""Frozen P0/P1 queue. No algorithm selection and no old-file deletion."""
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


ROOT = Path("/home/pc/research/TAG_condensation")
CODE = ROOT / "experiments/unified_q_attribution_20261008/code"
OUTPUT = ROOT / "results/unified_q_attribution_20261008"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("reference", "orders"), required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[0,1])
    args = p.parse_args()
    orders = [9101] if args.phase == "reference" else [9102,9103]
    tasks = [dict(dataset=d, order=o, tag=f"{d}_o{o}")
             for o in orders for d in ("wikics", "bookhis", "sportsfit")]
    runner = CODE / "run_unified_q_attribution_20261008.py"
    plan = dict(phase=args.phase, tasks=tasks, construction_seed=271006,
        student_seeds=[150,151,152], student_steps=2048, batch=24, q_steps=30,
        text_controls=args.phase == "reference", legacy_final=args.phase == "reference",
        no_parameter_search=True, no_deletion=True,
        core="unchanged uq_dm_edgefree; same protocol as 20261008 matrix",
        source_hash=hashlib.sha256(runner.read_bytes()).hexdigest())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{args.phase}_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("registered plan changed")
    write(path, plan)
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    def lane(gpu):
        records = []
        while not pending.empty():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            output = OUTPUT / task["tag"]
            command = [sys.executable, "-u", str(runner),
                "--dataset", task["dataset"], "--order-seed", str(task["order"]),
                "--reference", str(ROOT/f"results/unified_q_inloop_cat_20261007/{task['dataset']}/protocol.json"),
                "--text-cache", str(ROOT/f"results/public_baselines_text_cache_20261001/{task['dataset']}"),
                "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                "--output", str(output), "--device", f"cuda:{gpu}"]
            if args.phase == "reference":
                command += ["--text-controls", "--legacy-final"]
            if shutil.disk_usage(ROOT).free < 1.0 * 2**30:
                records.append(dict(task=task, exit_code=-2, error="disk guard; nothing deleted"))
                break
            env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                PYTHONPATH=f"{CODE}:{ROOT/'code/v2_llm_fusion'}:{ROOT/'code/frozen_v1_2026_09_09'}")
            print(json.dumps(dict(event="start", gpu=gpu, task=task)), flush=True)
            started = time.time()
            with (OUTPUT/f"{task['tag']}.log").open("a") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(task=task, gpu=gpu, exit_code=process.returncode,
                                seconds=time.time()-started))
            write(OUTPUT/f"{args.phase}_gpu{gpu}.json", records)
            print(json.dumps(dict(event="finish", **records[-1])), flush=True)
            if process.returncode:
                break
        return records
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        records = [r for output in executor.map(lane, args.gpus) for r in output]
    done = len(records) == len(tasks) and all(r["exit_code"] == 0 for r in records)
    write(OUTPUT/f"{args.phase}_status.json", dict(status="complete" if done else "incomplete", records=records))
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
