"""Matched baselines and attribution, with explicit resource accounting."""
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


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("baselines", "diagnostics"), required=True)
    p.add_argument("--wait-pids", nargs="*", type=int, default=[])
    p.add_argument("--gpus", nargs="+", type=int, default=[0,1])
    args = p.parse_args()
    root = Path("/home/pc/research/TAG_condensation")
    output = root / "results/unified_q_inloop_validation_20261007"
    code = root / "experiments/unified_q_inloop_validation_20261007/code"
    original = root / "experiments/unified_q_inloop_cat_20261007/code"
    name = ("run_unified_q_matched_baselines_20261007.py" if args.phase == "baselines"
            else "run_unified_q_fixed_content_20261007.py")
    runner = code / name
    tasks = [dict(dataset=d, tag=f"{args.phase}_{d}") for d in ("wikics", "bookhis", "sportsfit")]
    plan = dict(phase=args.phase, tasks=tasks, gpus=args.gpus,
        student_seeds=[150,151,152], student_steps=2048, student_batch=24,
        seed=271006, order=9101, budget="same canonical schedule as pilot",
        efficiency="possibly shared GPU; not exclusive efficiency measurements",
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        reference_root="results/unified_q_inloop_cat_20261007", no_parameter_search=True)
    plan_path = output / f"{args.phase}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise AssertionError("preregistered plan differs")
    write(plan_path, plan)
    for pid in args.wait_pids:
        while True:
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                break
            time.sleep(15)
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    def lane(gpu):
        records=[]
        while True:
            try:
                task=pending.get_nowait()
            except queue.Empty:
                break
            result_path = output / task["tag"] / "results.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("status") == "complete":
                records.append(dict(task=task, exit_code=0, reused=True))
                continue
            if shutil.disk_usage(root).free < 1.5 * 2**30:
                records.append(dict(task=task, exit_code=-2, error="disk guard; no files deleted"))
                break
            dataset=task["dataset"]
            cmd=[sys.executable,"-u",str(runner),"--dataset",dataset,
                "--reference",str(root/f"results/unified_q_inloop_cat_20261007/{dataset}/protocol.json"),
                "--text-cache",str(root/f"results/public_baselines_text_cache_20261001/{dataset}"),
                "--minilm","/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                "--output",str(output/task["tag"]),"--device",f"cuda:{gpu}"]
            env=dict(os.environ,TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONPATH=f"{code}:{original}:{root/'code/v2_llm_fusion'}")
            print(json.dumps(dict(event="start",gpu=gpu,**task)),flush=True)
            started=time.time()
            with (output/f"{task['tag']}.log").open("a") as log:
                process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(task=task,exit_code=process.returncode,gpu=gpu,seconds=time.time()-started)
            records.append(record)
            write(output/f"{args.phase}_gpu{gpu}.json",records)
            print(json.dumps(dict(event="finish",**record)),flush=True)
        return records
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        records=[r for items in pool.map(lane,args.gpus) for r in items]
    complete=len(records)==len(tasks) and all(r["exit_code"]==0 for r in records)
    write(output/f"{args.phase}_status.json",dict(status="complete" if complete else "incomplete",records=records))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
