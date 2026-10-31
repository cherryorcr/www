"""Run the missing cells for the 3-order x 3-construction confirmation.

The original 2x2 cells are kept as-is. This launcher only fills the missing
combinations in an isolated /dev/shm directory and writes JSON/log metadata.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading


ROOT = Path("/home/pc/research/TAG_condensation")
CODE = ROOT / "experiments/uq_text_path_confirmation_20261008/code"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
OUT = Path("/dev/shm/uq_confirmation_extended_20261009")
ORDERS = {
    "wikics": [9201, 9202, 9203],
    "sportsfit": [9201, 9202, 9203],
    "bookhis": [9202, 9203, 9205],
}
SEEDS = [271108, 271109, 271110]
OLD = {
    "wikics": {9201, 9202},
    "sportsfit": {9201, 9202},
    "bookhis": {9202, 9203},
}
COMMON_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONPATH": ":".join((str(CODE), str(ROOT / "code/v2_llm_fusion"),
                             str(ROOT / "code/frozen_v1_2026_09_09"))),
}
LOCK = threading.Lock()


def tasks():
    return [
        (dataset, order, seed)
        for dataset, orders in ORDERS.items()
        for order in orders
        for seed in SEEDS
        if not (order in OLD[dataset] and seed in (271108, 271109))
    ]


def run_one(item, gpu):
    dataset, order, seed = item
    tag = f"{dataset}_o{order}_c{seed}"
    output = OUT / tag
    log = OUT / f"{tag}.log"
    if (output / "results.json").exists():
        try:
            if json.loads((output / "results.json").read_text())["status"] == "complete":
                return dict(dataset=dataset, order=order, seed=seed, gpu=gpu,
                            status="already_complete", exit_code=0)
        except Exception:
            pass
    if output.exists():
        raise RuntimeError(f"Existing incomplete output preserved; inspect before resuming: {output}")
    reference = ROOT / f"results/unified_q_inloop_cat_20261007/{dataset}/protocol.json"
    cache = ROOT / f"results/public_baselines_text_cache_20261001/{dataset}"
    command = [PYTHON, "-u", str(CODE / "run_unified_q_text_confirmation_20261008.py"),
        "--dataset", dataset, "--device", f"cuda:{gpu}", "--output", str(output),
        "--reference", str(reference), "--text-cache", str(cache),
        "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        "--order-seed", str(order), "--construction-seed", str(seed)]
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event="start", dataset=dataset, order=order,
                                     seed=seed, gpu=gpu), ensure_ascii=True) + "\n")
        stream.flush()
        result = subprocess.run(command, env={**__import__("os").environ, **COMMON_ENV},
                                stdout=stream, stderr=subprocess.STDOUT)
    record = dict(dataset=dataset, order=order, seed=seed, gpu=gpu,
                  status="complete" if result.returncode == 0 else "failed",
                  exit_code=result.returncode)
    with LOCK:
        print(json.dumps(record, ensure_ascii=True), flush=True)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(orders=ORDERS, construction_seeds=SEEDS,
                skipped_existing={d: sorted(v) for d, v in OLD.items()},
                tasks=[dict(dataset=d, order=o, seed=s)
                for d, o, s in tasks()], no_score_selection=True,
                protocol="same reference fixed schedule, 2048 Student steps, seeds 150/151/152")
    plan_path = OUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Refusing to overwrite a different registered plan")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    pending = tasks()
    def lane(gpu):
        return [run_one(item, gpu) for item in pending[ gpu::2 ]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [v for group in pool.map(lane, (0, 1)) for v in group]
    status = "complete" if len(records) == len(pending) and all(v["exit_code"] == 0 for v in records) else "incomplete"
    (OUT / "status.json").write_text(json.dumps(dict(status=status, records=records), indent=2), encoding="utf-8")
    print(json.dumps(dict(status=status, count=len(records), expected=len(pending))), flush=True)
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
