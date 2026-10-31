"""Durably resume the five missing BookHis confirmation cells.

This is intentionally separate from the original launcher. It never removes
or rewrites a completed cell and records every subprocess outcome atomically,
so an interrupted parent process cannot make the queue look complete.
"""
import json
import concurrent.futures
import os
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path("/home/pc/research/TAG_condensation")
CODE = ROOT / "experiments/uq_text_path_confirmation_20261008/code"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
OUT = Path("/dev/shm/uq_confirmation_extended_20261009")
MINILM = "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"
TEXT_CACHE = ROOT / "results/public_baselines_text_cache_20261001/bookhis"
REFERENCE = ROOT / "results/unified_q_inloop_cat_20261007/bookhis/protocol.json"
TASKS = [
    ("bookhis", 9202, 271110),
    ("bookhis", 9203, 271110),
    ("bookhis", 9205, 271108),
    ("bookhis", 9205, 271109),
    ("bookhis", 9205, 271110),
]
STATUS_LOCK = threading.Lock()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def existing_complete(output):
    path = output / "results.json"
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("status") == "complete" and len(value.get("stages", [])) == 20 and len(value.get("rows", [])) == 8
    except Exception:
        return False


def run_one(dataset, order, seed, gpu):
    tag = f"{dataset}_o{order}_c{seed}"
    output = OUT / tag
    log = OUT / f"{tag}.log"
    if existing_complete(output):
        return dict(dataset=dataset, order=order, seed=seed, gpu=gpu, status="already_complete", exit_code=0)
    if output.exists():
        return dict(dataset=dataset, order=order, seed=seed, gpu=gpu,
                    status="refused_existing_incomplete", exit_code=None)
    output.mkdir(parents=True)
    command = [PYTHON, "-u", str(CODE / "run_unified_q_text_confirmation_20261008.py"),
        "--dataset", dataset, "--device", f"cuda:{gpu}", "--output", str(output),
        "--reference", str(REFERENCE), "--text-cache", str(TEXT_CACHE),
        "--minilm", MINILM, "--order-seed", str(order), "--construction-seed", str(seed)]
    started = time.time()
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(dict(event="start", dataset=dataset, order=order,
                                     seed=seed, gpu=gpu, pid=os.getpid())) + "\n")
        try:
            process = subprocess.run(command, env={**os.environ,
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": ":".join((str(CODE), str(ROOT / "code/v2_llm_fusion"),
                                         str(ROOT / "code/frozen_v1_2026_09_09")))},
                stdout=stream, stderr=subprocess.STDOUT)
            status = "complete" if process.returncode == 0 and existing_complete(output) else "failed"
            result = dict(dataset=dataset, order=order, seed=seed, gpu=gpu,
                          status=status, exit_code=process.returncode,
                          seconds=time.time() - started)
        except BaseException as error:
            result = dict(dataset=dataset, order=order, seed=seed, gpu=gpu,
                          status="exception", exit_code=None, error=repr(error),
                          seconds=time.time() - started)
        stream.write(json.dumps(dict(event="finish", **result)) + "\n")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "bookhis_resume_20261009.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit("resume lock already exists")
    started = time.time()
    records = []
    plan = dict(tasks=[dict(dataset=d, order=o, seed=s) for d, o, s in TASKS],
                frozen_protocol="same 2048-step confirmation, four arms, GCN/SGC, no parameter search",
                parent_plan=str(OUT / "plan.json"), started=started)
    write_json(OUT / "bookhis_resume_plan_20261009.json", plan)
    write_json(OUT / "bookhis_resume_status.json", dict(status="running", records=[]))
    try:
        def lane(gpu):
            lane_records = []
            for index in range(gpu, len(TASKS), 2):
                dataset, order, seed = TASKS[index]
                record = run_one(dataset, order, seed, gpu)
                lane_records.append(record)
                with STATUS_LOCK:
                    records.append(record)
                    write_json(OUT / "bookhis_resume_status.json", dict(status="running", records=records))
            return lane_records

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lane, (0, 1)))
        success = all(r["status"] in ("complete", "already_complete") for r in records)
        write_json(OUT / "bookhis_resume_status.json", dict(status="complete" if success else "incomplete", records=records))
        if not success:
            raise SystemExit(1)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
