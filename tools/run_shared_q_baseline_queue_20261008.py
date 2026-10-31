"""Bounded two-GPU queue for the registered comparisons."""
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
EXPERIMENT = ROOT / "experiments/shared_q_baseline_20261008"
CODE = EXPERIMENT / "code"
OUTPUT = ROOT / "results/shared_q_baseline_20261008"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("shared", "handoff", "smoke", "handoff_smoke"), required=True)
    args = p.parse_args()
    if args.phase == "shared":
        tasks = [dict(dataset=d, order=o, tag=f"shared_{d}_o{o}")
                 for o in (9101,9102,9103) for d in ("wikics","bookhis","sportsfit")]
    elif args.phase == "handoff":
        tasks = [dict(dataset=d, tag=f"handoff_{d}") for d in ("wikics","bookhis","sportsfit")]
    elif args.phase == "handoff_smoke":
        tasks = [dict(dataset="wikics", tag="smoke_handoff_wikics")]
    else:
        tasks = [dict(dataset="wikics", order=9101, tag="smoke_shared_wikics")]
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    handoff = args.phase.startswith("handoff")
    runner = CODE / ("run_handoff_student_transfer_20261008.py" if handoff else "run_shared_q_attribution_20261008.py")
    plan = dict(tasks=tasks, phase=args.phase, student_seeds=[150,151,152], construction_seed=271006,
        no_parameter_search=True, no_deletion=True, runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    path = OUTPUT / f"{args.phase}_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("plan changed")
    write(path, plan)

    def lane(gpu):
        records = []
        while not pending.empty():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            output = OUTPUT / task["tag"]
            existing = output / "results.json"
            if existing.exists() and json.loads(existing.read_text()).get("status") == "complete":
                records.append(dict(task=task, exit_code=0, reused=True))
                continue
            if shutil.disk_usage(ROOT).free < .85 * 2**30:
                records.append(dict(task=task, exit_code=-2, error="disk guard; nothing deleted"))
                break
            cmd = [sys.executable, "-u", str(runner), "--dataset", task["dataset"],
                   "--device", f"cuda:{gpu}", "--output", str(output),
                   "--text-cache", str(ROOT / f"results/public_baselines_text_cache_20261001/{task['dataset']}"),
                   "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"]
            if handoff:
                cmd += ["--manifest", str(EXPERIMENT / f"manifests/{task['dataset']}_manifest.json")]
            else:
                cmd += ["--order-seed", str(task["order"]),
                        "--reference", str(ROOT / f"results/unified_q_inloop_cat_20261007/{task['dataset']}/protocol.json"),
                        "--source", str(ROOT / f"results/unified_q_attribution_20261008/{task['dataset']}_o{task['order']}")]
            if args.phase in ("smoke", "handoff_smoke"):
                cmd += ["--max-stages", "1", "--q-steps", "2", "--student-steps", "8"]
            env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", PYTHONPATH=f"{CODE}:{ROOT/'code/v2_llm_fusion'}:{ROOT/'code/frozen_v1_2026_09_09'}")
            print(json.dumps(dict(event="start", task=task, gpu=gpu)), flush=True)
            started = time.time()
            with (OUTPUT / f"{task['tag']}.log").open("a") as log:
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(task=task, gpu=gpu, exit_code=proc.returncode, seconds=time.time()-started))
            write(OUTPUT / f"{args.phase}_gpu{gpu}.json", records)
            print(json.dumps(dict(event="finish", **records[-1])), flush=True)
            if proc.returncode:
                break
        return records

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = [r for lane_records in executor.map(lane, (0,1)) for r in lane_records]
    complete = len(records) == len(tasks) and all(r["exit_code"] == 0 for r in records)
    write(OUTPUT / f"{args.phase}_status.json", dict(status="complete" if complete else "incomplete", records=records))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
