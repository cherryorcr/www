"""Run the preregistered frozen-memory SAGE matrix after confirmation."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import time

ROOT = Path("/home/pc/research/TAG_condensation")
CODE = ROOT / "experiments/uq_text_path_confirmation_20261008/code"
EXTRA = Path("/dev/shm/uq_sage_code_20261009")
EXTENDED = Path("/dev/shm/uq_confirmation_extended_20261009")
OLD = ROOT / "results/uq_text_path_20261008"
OUT = Path("/dev/shm/uq_sage_extended_20261009")
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
ORDERS = dict(wikics=[9201, 9202, 9203], sportsfit=[9201, 9202, 9203], bookhis=[9202, 9203, 9205])
SEEDS = [271108, 271109, 271110]


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temp.replace(path)


def tasks():
    return [dict(dataset=d, order=o, seed=s) for d, orders in ORDERS.items() for o in orders for s in SEEDS]


def run_one(task, gpu):
    tag = f"{task['dataset']}_o{task['order']}_c{task['seed']}"
    source = OLD / ("confirm_" + tag)
    if not source.exists():
        source = EXTENDED / tag
    if not (source / "results.json").exists():
        raise FileNotFoundError(source)
    output = OUT / tag
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=":".join(map(str, (CODE, EXTRA, ROOT / "code/v2_llm_fusion", ROOT / "code/frozen_v1_2026_09_09"))))
    command = [PYTHON, "-u", str(EXTRA / "run_uq_frozen_sage_20261009.py"),
        "--source", str(source), "--output", str(output), "--device", f"cuda:{gpu}",
        "--text-cache", str(ROOT / f"results/public_baselines_text_cache_20261001/{task['dataset']}"),
        "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"]
    with (OUT / (tag + ".log")).open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(**task, gpu=gpu, exit_code=process.returncode, status="complete" if process.returncode == 0 else "failed")
    print(json.dumps(record), flush=True)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(tasks=tasks(), arms=["uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard"],
        students=["prompt_sage"], steps=2048, student_seeds=[150, 151, 152], student_batch=24,
        no_parameter_search=True, selection_used_downstream_scores=False,
        architecture="two-layer mean SAGE, explicit self-loop neighborhood plus independent root; not canonical loop-free SAGE",
        frozen_memory_only=True, dependency=str(EXTENDED / "status.json"))
    plan_path = OUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("registered plan differs")
    if not plan_path.exists():
        write(plan_path, plan)
    lock = OUT / "launcher.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        write(OUT / "status.json", dict(status="waiting_for_confirmation", records=[]))
        deadline = time.monotonic() + 4 * 3600
        while True:
            status = EXTENDED / "status.json"
            if status.exists():
                dependency = json.loads(status.read_text())
                if dependency["status"] != "complete":
                    raise RuntimeError("confirmation failed; inspect before readout")
                break
            if time.monotonic() > deadline:
                raise TimeoutError("confirmation wait exceeded four hours")
            time.sleep(30)
        pending = queue.Queue()
        for task in tasks():
            pending.put(task)
        write(OUT / "status.json", dict(status="running", records=[]))
        def lane(gpu):
            records = []
            while True:
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    break
                records.append(run_one(task, gpu))
                write(OUT / f"gpu{gpu}.json", records)
                if records[-1]["exit_code"]:
                    break
            return records
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            records = [r for group in pool.map(lane, (0, 1)) for r in group]
        success = len(records) == len(tasks()) and all(r["exit_code"] == 0 for r in records)
        write(OUT / "status.json", dict(status="complete" if success else "incomplete", records=records))
        if not success:
            raise RuntimeError("incomplete SAGE matrix; failed outputs retained")
    except BaseException as error:
        write(OUT / "launcher_error.json", dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
