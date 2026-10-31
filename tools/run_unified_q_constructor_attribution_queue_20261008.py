"""Run the preregistered constructor controls, without changing the primary."""
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
CODE = ROOT/"experiments/unified_q_constructor_attribution_20261008/code"
OUTPUT = ROOT/"results/unified_q_constructor_attribution_20261008_v2"


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",choices=("reference","orders","all"),default="all")
    p.add_argument("--gpus",nargs="+",type=int,default=[0,1])
    args = p.parse_args()
    orders = [9101] if args.phase == "reference" else [9102,9103] if args.phase == "orders" else [9101,9102,9103]
    tasks = [dict(dataset=d,order=o,tag=f"{d}_o{o}") for o in orders for d in ("wikics","bookhis","sportsfit")]
    runner = CODE/"run_unified_q_constructor_attribution_20261008.py"
    plan = dict(tasks=tasks,phase=args.phase,construction_seed=271006,student_seeds=[150,151,152],
        steps=2048,q_steps=30,no_parameter_search=True,no_deletion=True,
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        source="unified_q_attribution_20261008",common_payload_readout="final-stage only")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path = OUTPUT/f"{args.phase}_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("registered plan changed")
    write(path,plan)
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
            if shutil.disk_usage(ROOT).free < 1.0*2**30:
                records.append(dict(task=task,exit_code=-2,error="disk guard; nothing deleted"))
                break
            command = [sys.executable,"-u",str(runner),"--dataset",task["dataset"],
                "--order-seed",str(task["order"]),"--device",f"cuda:{gpu}",
                "--reference",str(ROOT/f"results/unified_q_inloop_cat_20261007/{task['dataset']}/protocol.json"),
                "--source",str(ROOT/f"results/unified_q_attribution_20261008/{task['tag']}"),
                "--text-cache",str(ROOT/f"results/public_baselines_text_cache_20261001/{task['dataset']}"),
                "--minilm","/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                "--output",str(OUTPUT/task["tag"])]
            env = dict(os.environ,TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",PYTHONPATH=f"{CODE}:{ROOT/'code/v2_llm_fusion'}:{ROOT/'code/frozen_v1_2026_09_09'}")
            started = time.time()
            print(json.dumps(dict(event="start",gpu=gpu,task=task)),flush=True)
            with (OUTPUT/f"{task['tag']}.log").open("a") as log:
                proc = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            records.append(dict(task=task,gpu=gpu,exit_code=proc.returncode,seconds=time.time()-started))
            write(OUTPUT/f"{args.phase}_gpu{gpu}.json",records)
            print(json.dumps(dict(event="finish",**records[-1])),flush=True)
            if proc.returncode:
                break
        return records
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        records = [record for lane_records in executor.map(lane,args.gpus) for record in lane_records]
    done = len(records)==len(tasks) and all(r["exit_code"]==0 for r in records)
    write(OUTPUT/f"{args.phase}_status.json",dict(status="complete" if done else "incomplete",records=records))
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
