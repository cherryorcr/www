"""Deploy and run the registered shared-Q attribution matrix remotely."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
HOST = "pc@100.126.175.112"
BASE = "/home/pc/research/TAG_condensation"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
EXP = BASE + "/experiments/uq_q_seed_attribution_20261010"
OUT = "/dev/shm/uq_q_seed_attribution_20261010"
LOCAL = ROOT / "results/uq_q_seed_attribution_20261010"
SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", HOST]

DATASETS = ("wikics", "sportsfit", "bookhis")
ORDERS = (9101, 9102, 9103)
SEEDS = (271006, 271108, 271109)


def remote(command: str, *, capture=False, timeout=300):
    return subprocess.run(SSH + [command], check=True, capture_output=capture, timeout=timeout)


def deploy():
    archive = ROOT / "tmp_remote_results/uq_q_seed_attribution_20261010.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        for path in (ROOT / "code/v2_llm_fusion").glob("*.py"):
            stream.add(path, arcname="code/" + path.name)
        stream.add(ROOT / "code/v2_llm_fusion/run_uq_q_seed_matrix_20261010.py",
                   arcname="code/run_uq_q_seed_matrix_20261010.py")
    target = BASE + "/tmp/" + archive.name
    subprocess.run(["scp", "-C", "-o", "BatchMode=yes", str(archive), HOST + ":" + target],
                   check=True, timeout=300)
    remote("mkdir -p " + shlex.quote(EXP + "/code") + " " + shlex.quote(OUT))
    remote("tar -xzf " + shlex.quote(target) + " -C " + shlex.quote(EXP))
    print(json.dumps({"event": "deployed", "experiment": EXP,
                      "archive": str(archive), "bytes": archive.stat().st_size}), flush=True)


def tasks():
    return [(d, o, s) for d in DATASETS for o in ORDERS for s in SEEDS]


def command(dataset: str, order: int, seed: int, gpu: int) -> tuple[str, str]:
    tag = f"{dataset}_o{order}_c{seed}"
    output = f"{OUT}/{tag}"
    ref = f"{BASE}/results/unified_q_inloop_cat_20261007/{dataset}/protocol.json"
    cache = f"{BASE}/results/public_baselines_text_cache_20261001/{dataset}"
    env = ("env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
           "TOKENIZERS_PARALLELISM=false PYTHONPATH=" +
           shlex.quote(EXP + "/code:" + BASE + "/code/v2_llm_fusion:" +
                       BASE + "/code/frozen_v1_2026_09_09"))
    args = [PYTHON, "-u", EXP + "/code/run_uq_q_seed_matrix_20261010.py",
            "--dataset", dataset, "--reference", ref, "--text-cache", cache,
            "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
            "--output", output, "--device", f"cuda:{gpu}",
            "--order-seed", str(order), "--construction-seed", str(seed)]
    log = f"{OUT}/{tag}.log"
    return tag, "mkdir -p " + shlex.quote(output) + "; " + env + " " + \
        " ".join(map(shlex.quote, args)) + " > " + shlex.quote(log) + " 2>&1"


def run_matrix():
    plan = {"datasets": list(DATASETS), "orders": list(ORDERS),
            "construction_seeds": list(SEEDS), "arms": ["uq_shared_q", "uq_split_q", "uq_no_residual"],
            "student_seeds": [150, 151, 152], "student_steps": 2048,
            "no_parameter_search": True, "output": OUT}
    remote("mkdir -p " + shlex.quote(OUT))
    remote("cat > " + shlex.quote(OUT + "/plan.json") + " <<'EOF'\n" +
           json.dumps(plan, indent=2) + "\nEOF")
    pending = tasks()

    def lane(gpu: int):
        records = []
        for dataset, order, seed in pending[gpu::2]:
            tag, cmd = command(dataset, order, seed, gpu)
            print(json.dumps({"event": "start", "gpu": gpu, "dataset": dataset,
                              "order": order, "seed": seed}), flush=True)
            result = subprocess.run(SSH + [cmd], timeout=6 * 3600)
            record = {"tag": tag, "gpu": gpu, "dataset": dataset, "order": order,
                      "seed": seed, "exit_code": result.returncode}
            records.append(record)
            print(json.dumps({"event": "finish", **record}), flush=True)
            if result.returncode:
                break
        return records

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [v for group in pool.map(lane, (0, 1)) for v in group]
    status = {"expected": len(pending), "completed_processes": len(records),
              "records": records,
              "status": "complete" if len(records) == len(pending)
              and all(v["exit_code"] == 0 for v in records) else "incomplete"}
    (LOCAL / "remote_status.json").parent.mkdir(parents=True, exist_ok=True)
    (LOCAL / "remote_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if status["status"] != "complete":
        raise SystemExit(1)


def recover_matrix_v2():
    """Retry only incomplete cells; never delete a previous partial output."""
    expected = tasks()
    pending = []
    probes = []
    for dataset, order, seed in expected:
        tag = f"{dataset}_o{order}_c{seed}"
        local = LOCAL / tag / "results.json"
        try:
            item = json.loads(local.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            item = {}
        if item.get("status") != "complete":
            pending.append((dataset, order, seed))
    # The remote tmpfs was cleared by the reboot.  Start only cells that do
    # not have a complete local archive; no old remote directory is reused.
    remote("mkdir -p " + shlex.quote(OUT))
    print(json.dumps({"event": "recovery_plan", "pending": pending}), flush=True)

    def lane(gpu: int):
        records = []
        for dataset, order, seed in pending[gpu::2]:
            tag, cmd = command(dataset, order, seed, gpu)
            print(json.dumps({"event": "recovery_start", "gpu": gpu,
                              "dataset": dataset, "order": order, "seed": seed}), flush=True)
            result = subprocess.run(SSH + [cmd], timeout=6 * 3600)
            record = {"tag": tag, "gpu": gpu, "dataset": dataset, "order": order,
                      "seed": seed, "exit_code": result.returncode}
            records.append(record)
            print(json.dumps({"event": "recovery_finish", **record}), flush=True)
        return records

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [v for group in pool.map(lane, (0, 1)) for v in group]
    status = {"pending": len(pending), "records": records,
              "status": "complete" if len(records) == len(pending)
              and all(v["exit_code"] == 0 for v in records) else "incomplete"}
    (LOCAL / "recovery_status.json").parent.mkdir(parents=True, exist_ok=True)
    (LOCAL / "recovery_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if status["status"] != "complete":
        raise SystemExit(1)


def collect():
    LOCAL.mkdir(parents=True, exist_ok=True)
    response = subprocess.run(
        SSH + ["tar -czf - -C " + shlex.quote(OUT) +
               " --exclude='*.log' --exclude='*.pt' --exclude='*.pt.gz' ."],
        capture_output=True, timeout=600,
    )
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors="replace"))
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode="r:gz") as stream:
        for item in stream:
            if item.isdir():
                continue
            target = (LOCAL / item.name).resolve()
            if not item.isfile() or not target.is_relative_to(LOCAL.resolve()):
                raise ValueError("unsafe remote archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.extractfile(item).read())
            count += 1
    print(json.dumps({"event": "collected", "files": count,
                      "compressed_bytes": len(response.stdout)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "run", "recover", "collect", "status"))
    args = parser.parse_args()
    if args.action == "deploy":
        deploy()
    elif args.action == "run":
        run_matrix()
    elif args.action == "recover":
        recover_matrix_v2()
    elif args.action == "collect":
        collect()
    else:
        print(json.dumps({"remote_output": OUT,
                          "local_output": str(LOCAL),
                          "status": "use ssh find /dev/shm/uq_q_seed_attribution_20261010"}))


if __name__ == "__main__":
    main()
