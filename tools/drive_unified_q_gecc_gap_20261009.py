"""Isolated, resumable development cells; never deletes previous experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "/home/pc/research/TAG_condensation"
HOST = "pc@100.126.175.112"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", HOST]
RESULT = BASE + "/results/uq_gecc_gap_20261009"
LOCAL = ROOT / "results/uq_gecc_gap_20261009"
TASKS = [dict(dataset=d, order=9202 if d == "bookhis" else 9201, seed=s)
         for s in (271108, 271109) for d in ("wikics", "sportsfit", "bookhis")]
P2 = ("p2-paired", "p2-marginal", "p2-shuffled")


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def snapshot(phase):
    return BASE + "/experiments/uq_gecc_gap_" + phase.replace("-", "_") + "_v3_20261009"


def deploy(phase):
    dest = snapshot(phase)
    remote("test ! -e " + shlex.quote(dest))
    archive = ROOT / ("tmp_remote_results/" + dest.rsplit("/", 1)[-1] + ".tar.gz")
    with tarfile.open(archive, "w:gz") as out:
        for path in (ROOT / "code/v2_llm_fusion").glob("*.py"):
            out.add(path, arcname="code/" + path.name)
    subprocess.run(["scp", "-C", str(archive), HOST + ":" + BASE + "/tmp/" + archive.name], check=True)
    remote("mkdir " + shlex.quote(dest))
    remote("tar -xzf " + shlex.quote(BASE + "/tmp/" + archive.name) + " -C " + shlex.quote(dest))
    print(json.dumps(dict(snapshot=dest, bytes=archive.stat().st_size)), flush=True)


def collect(tensors):
    exclude = " --exclude='*.tmp' --exclude='*.log' --exclude='*.png' --exclude='*.md'"
    if not tensors:
        exclude += " --exclude='*.pt.gz'"
    response = subprocess.run(SSH + ["tar -czf - -C " + RESULT + exclude + " ."],
                              capture_output=True, timeout=600)
    warning = response.stderr.decode(errors="replace")
    if response.returncode and not (response.returncode == 1 and warning.strip()
            and all("file changed as we read it" in line for line in warning.splitlines() if line.strip())):
        raise RuntimeError(f"collection failed ({response.returncode}): {warning}")
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode="r:gz") as archive:
        for entry in archive:
            if entry.isdir():
                continue
            target = (LOCAL / entry.name).resolve()
            if not entry.isfile() or not target.is_relative_to(LOCAL.resolve()) or target.suffix not in (".json", ".gz"):
                raise ValueError("unsafe archive entry")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(entry).read())
            count += 1
    print(json.dumps(dict(collected=count, bytes=len(response.stdout), live_snapshot_warning=warning)), flush=True)


def command(phase, task, gpu):
    dest = snapshot(phase)
    tag = f"{phase}_v3_{task['dataset']}_o{task['order']}_c{task['seed']}"
    source = BASE + f"/results/uq_text_path_20261008/confirm_{task['dataset']}_o{task['order']}_c{task['seed']}"
    env = "env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH=" + shlex.quote(dest + "/code:" + BASE + "/code/v2_llm_fusion:" + BASE + "/code/frozen_v1_2026_09_09")
    args = [PYTHON, "-u", dest + "/code/run_unified_q_gecc_gap_20261009.py",
        "--phase", phase, "--dataset", task["dataset"], "--device", f"cuda:{gpu}",
        "--order-seed", str(task["order"]), "--construction-seed", str(task["seed"]),
        "--source", source, "--output", RESULT + "/" + tag,
        "--reference", BASE + f"/results/unified_q_inloop_cat_20261007/{task['dataset']}/protocol.json",
        "--text-cache", BASE + f"/results/public_baselines_text_cache_20261001/{task['dataset']}",
        "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"]
    return tag, env + " " + " ".join(map(shlex.quote, args))


def run(phase, first):
    LOCAL.mkdir(parents=True, exist_ok=True)
    if phase == "p1":
        gate = json.loads((LOCAL / "p1_gate.json").read_text())
        if gate["p0_completed_cells"] != 6 or gate["decision"] != "run_single_factor_development_test_not_promote":
            raise ValueError("P0 gate not satisfied")
    if phase in P2:
        gate = json.loads((LOCAL / "p2_gate.json").read_text())
        if gate["decision"] != "test_pairing_hypothesis_from_original_not_message_pooling" or gate["probe_passed_stages"] != 15:
            raise ValueError("pairing information gate not satisfied")
    tasks = TASKS[:1] if first else TASKS
    plan = dict(phase=phase, tasks=TASKS, development=True, independent_confirmation=False,
        student_seeds=[150, 151, 152], student_steps=2048, students=["GCN", "SGC"],
        no_sealed_test=True, single_change="feature_pooling only" if phase == "p1" else phase,
        conditional_gates="P1 after representation diagnostics; P2 only after separate decision; P3 after candidate freeze")
    plan_path = LOCAL / f"{phase}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("predeclared plan changed")
    plan_path.write_text(json.dumps(plan, indent=2))
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
            tag, cmd = command(phase, task, gpu)
            print(json.dumps(dict(event="cell_start", phase=phase, gpu=gpu, **task)), flush=True)
            with (LOCAL / (tag + ".log")).open("a", encoding="utf-8") as log:
                result = subprocess.run(SSH + [cmd], stdout=log, stderr=subprocess.STDOUT)
            record = dict(**task, gpu=gpu, phase=phase, exit_code=result.returncode)
            records.append(record)
            print(json.dumps(dict(event="cell_finish", **record)), flush=True)
            if result.returncode:
                break
        return records
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = [r for group in executor.map(lane, (0, 1)) for r in group]
    status = "complete" if len(records) == len(tasks) and all(r["exit_code"] == 0 for r in records) else "incomplete"
    (LOCAL / f"{phase}_{'first' if first else 'batch'}_status.json").write_text(json.dumps(dict(status=status, records=records), indent=2))
    if status != "complete":
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("deploy", "run", "collect"))
    p.add_argument("--phase", choices=("p0", "p1", "p1-no-prop", *P2, "p2-all"), default="p0")
    p.add_argument("--first", action="store_true")
    p.add_argument("--tensors", action="store_true")
    args = p.parse_args()
    if args.phase == "p2-all" and args.action != "collect":
        for phase in P2:
            if args.action == "deploy":
                deploy(phase)
            else:
                run(phase, args.first)
        return
    if args.action == "deploy":
        deploy(args.phase)
    elif args.action == "collect":
        collect(args.tensors)
    else:
        run(args.phase, args.first)


if __name__ == "__main__":
    main()
