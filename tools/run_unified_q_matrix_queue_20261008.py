"""Predeclared two-GPU matrix, no parameter selection and no data deletion."""
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
OUTPUT = ROOT / "results/unified_q_matrix_20261008"
CODE = ROOT / "experiments/unified_q_matrix_20261008/code"
NEW_ABLATIONS = ["uq_dm_edgefree", "uq_no_dm", "uq_no_prop", "uq_no_residual",
                 "uq_frozen_text", "uq_detached_text", "uq_no_text_init",
                 "uq_no_q_anchor", "uq_no_x_anchor", "uq_dm_only"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("ablations", "ablation_completion", "baselines", "tagsam"), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0,1])
    parser.add_argument("--wait-for-phase", choices=("baselines", "ablation_completion"))
    args = parser.parse_args()
    tasks = []
    for dataset in ("wikics", "bookhis", "sportsfit"):
        if args.phase in ("ablations", "ablation_completion"):
            tail = args.phase == "ablation_completion" and dataset == "wikics"
            tasks.append(dict(dataset=dataset, order=9101,
                arms=NEW_ABLATIONS[7:] if tail else NEW_ABLATIONS,
                tag=f"ablation_{dataset}_tail" if tail else f"ablation_{dataset}"))
        elif args.phase == "tagsam":
            tasks.append(dict(dataset=dataset, order=9101,
                              arms=["tagsam_response_adapter"], tag=f"tagsam_{dataset}"))
        else:
            for order in (9102, 9103):
                tasks.append(dict(dataset=dataset, order=order,
                    arms=["random_coreset", "kcenter_coreset", "gecc_hard", "puma_tag"],
                    tag=f"baseline_{dataset}_o{order}"))
    runner = CODE / ("run_unified_q_tagsam_20261008.py" if args.phase == "tagsam"
                     else "run_unified_q_matrix_20261008.py")
    plan = dict(phase=args.phase, tasks=tasks, construction_seed=271006,
                student_seeds=[150,151,152], student_steps=2048, q_steps=30,
                budget="fixed reference per-stage schedule", no_parameter_search=True,
                reuse="prior full/no-GM/fixed-Q and reference-order external cells are separate archived evidence",
                runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / f"{args.phase}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("registered plan changed")
    write(plan_path, plan)
    if args.wait_for_phase:
        status_path = OUTPUT / f"{args.wait_for_phase}_status.json"
        deadline = time.time() + 12 * 3600
        while not status_path.exists():
            if time.time() > deadline:
                raise TimeoutError("dependency queue did not finish within 12 hours")
            time.sleep(20)
        if json.loads(status_path.read_text()).get("status") != "complete":
            raise RuntimeError("dependency queue failed; do not start another GPU workload")
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
            output = OUTPUT / task["tag"]
            existing = output / "results.json"
            if existing.exists() and json.loads(existing.read_text()).get("status") == "complete":
                records.append(dict(task=task, exit_code=0, reused=True))
                continue
            if shutil.disk_usage(ROOT).free < 1.0 * 2**30:
                records.append(dict(task=task, exit_code=-2, error="disk guard; nothing deleted"))
                break
            command = [sys.executable, "-u", str(runner), "--dataset", task["dataset"],
                "--reference", str(ROOT/f"results/unified_q_inloop_cat_20261007/{task['dataset']}/protocol.json"),
                "--text-cache", str(ROOT/f"results/public_baselines_text_cache_20261001/{task['dataset']}"),
                "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                "--output", str(output), "--device", f"cuda:{gpu}", "--order-seed", str(task["order"])]
            if args.phase != "tagsam":
                command += ["--arms", *task["arms"]]
            environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                PYTHONPATH=f"{CODE}:{ROOT/'code/v2_llm_fusion'}")
            print(json.dumps(dict(event="start", gpu=gpu, task=task)), flush=True)
            started = time.time()
            with (OUTPUT/f"{task['tag']}.log").open("a") as log:
                process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            record = dict(task=task, gpu=gpu, exit_code=process.returncode, seconds=time.time()-started)
            records.append(record)
            write(OUTPUT/f"{args.phase}_gpu{gpu}.json", records)
            print(json.dumps(dict(event="finish", **record)), flush=True)
            if process.returncode:
                break
        return records
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        records = [r for result in executor.map(lane, args.gpus) for r in result]
    done = len(records) == len(tasks) and all(r["exit_code"] == 0 for r in records)
    write(OUTPUT/f"{args.phase}_status.json", dict(status="complete" if done else "incomplete", records=records))
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
