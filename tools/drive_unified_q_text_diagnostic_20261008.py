"""Deploy, run and collect the controlled text-path investigation without deleting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shlex
import subprocess
import queue
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
HOST = "pc@100.126.175.112"
BASE = "/home/pc/research/TAG_condensation"
EXPERIMENT = BASE + "/experiments/uq_text_path_20261008"
RESULT = BASE + "/results/uq_text_path_20261008"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", HOST]


def remote(command, capture=False, timeout=14400):
    return subprocess.run(SSH + [command], check=True, capture_output=capture, timeout=timeout)


def deploy():
    archive = ROOT / ("tmp_remote_results/" + EXPERIMENT.rsplit("/", 1)[-1] + ".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as out:
        for path in (ROOT / "code/v2_llm_fusion").glob("*.py"):
            out.add(path, arcname="code/" + path.name)
    subprocess.run(["scp", "-C", "-o", "BatchMode=yes", str(archive),
                    HOST + ":" + BASE + "/tmp/" + archive.name], check=True, timeout=180)
    remote("mkdir -p " + shlex.quote(EXPERIMENT))
    remote("tar -xzf " + BASE + "/tmp/" + archive.name + " -C " + shlex.quote(EXPERIMENT))
    print(json.dumps(dict(deployed=EXPERIMENT, bytes=archive.stat().st_size)), flush=True)


def collect(tensors=False):
    exclusions = " --exclude='*.log' --exclude='*.tmp' --exclude='*.md' --exclude='*.png'"
    command = "tar -czf - -C " + RESULT + exclusions + " ."
    if not tensors:
        command = "tar -czf - -C " + RESULT + exclusions + " --exclude='*.pt' --exclude='*.pt.gz' ."
    response = subprocess.run(SSH + [command], capture_output=True, timeout=300)
    warning = response.stderr.decode(errors="replace")
    if response.returncode and not (response.returncode == 1 and warning.strip()
            and all("file changed as we read it" in line for line in warning.splitlines() if line.strip())):
        raise RuntimeError(f"archive collection failed ({response.returncode}): {warning}")
    dest = (ROOT / "results/uq_text_path_20261008").resolve()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode="r:gz") as archive:
        for entry in archive:
            if entry.isdir():
                continue
            path = (dest / entry.name).resolve()
            if not entry.isfile() or not path.is_relative_to(dest) or path.suffix not in (".json", ".pt", ".gz"):
                raise ValueError(f"unexpected archive member: {entry.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".sync")
            temp.write_bytes(archive.extractfile(entry).read())
            temp.replace(path)
            count += 1
    print(json.dumps(dict(files=count, compressed_bytes=len(response.stdout), live_snapshot_warning=warning)), flush=True)


def main():
    global EXPERIMENT
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("deploy", "deploy-confirmation", "diagnose", "diagnose-all", "confirm", "confirm-all", "confirm-feasible", "preflight-orders", "collect", "test"))
    p.add_argument("--dataset", default="wikics")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--tag")
    p.add_argument("--order", type=int, default=9101)
    p.add_argument("--seed", type=int, default=271006)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--tensors", action="store_true")
    args = p.parse_args()
    if args.action in ("deploy-confirmation", "confirm", "confirm-all", "confirm-feasible"):
        EXPERIMENT = BASE + "/experiments/uq_text_path_confirmation_20261008"
    if args.action in ("deploy", "deploy-confirmation"):
        return deploy()
    if args.action == "collect":
        return collect(args.tensors)
    if args.action == "preflight-orders":
        script = ROOT / "tools/preflight_uq_text_orders_20261008.py"
        target = BASE + "/tmp/" + script.name
        subprocess.run(["scp", "-C", "-o", "BatchMode=yes", str(script), HOST + ":" + target], check=True, timeout=180)
        remote("env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false "
               + PYTHON + " " + shlex.quote(target))
        return
    if args.action == "diagnose-all":
        def lane(gpu):
            for dataset in (("wikics", "bookhis") if gpu == 0 else ("sportsfit",)):
                subprocess.run([__import__("sys").executable, __file__, "diagnose",
                    "--dataset", dataset, "--gpu", str(gpu), "--tag", "diagnostic_v2_" + dataset], check=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lane, (0, 1)))
        return
    if args.action in ("confirm-all", "confirm-feasible"):
        pending = queue.Queue()
        if args.action == "confirm-feasible":
            feasibility = json.loads((ROOT / "results/uq_text_path_20261008/order_feasibility.json").read_text())
            tasks = [dict(dataset=d, order=feasibility["datasets"][d]["selected_orders"][i], seed=s)
                     for i in (0,1) for s in (271108,271109) for d in ("wikics","sportsfit","bookhis")]
        else:
            tasks = [dict(dataset=d, order=o, seed=s) for o in (9201, 9202)
                     for s in (271108, 271109) for d in ("wikics", "sportsfit", "bookhis")]
        suffix = "confirmation_feasible" if args.action == "confirm-feasible" else "confirmation"
        plan_path = ROOT / f"results/uq_text_path_20261008/{suffix}_plan.json"
        plan = dict(tasks=tasks, students=["prompt_gcn", "prompt_sgc"], student_seeds=[150,151,152],
                    steps=2048, arms=["uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard"],
                    selection_used_downstream_scores=False, no_parameter_search=True)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("plan changed")
        plan_path.write_text(json.dumps(plan, indent=2))
        if args.plan_only:
            print(json.dumps(dict(plan=str(plan_path), tasks=tasks)), flush=True)
            return
        for task in tasks:
            pending.put(task)
        def lane(gpu):
            records = []
            while not pending.empty():
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    break
                tag = f"confirm_{task['dataset']}_o{task['order']}_c{task['seed']}"
                print(json.dumps(dict(event="cell_start", gpu=gpu, **task)), flush=True)
                log_path = plan_path.parent / (tag + ".log")
                with log_path.open("a", encoding="utf-8") as log:
                    proc = subprocess.run([sys.executable, __file__, "confirm", "--dataset", task["dataset"],
                        "--gpu", str(gpu), "--tag", tag, "--order", str(task["order"]), "--seed", str(task["seed"])],
                        stdout=log, stderr=subprocess.STDOUT)
                records.append(dict(**task, exit_code=proc.returncode, gpu=gpu))
                (plan_path.parent / f"{suffix}_gpu{gpu}.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(event="cell_finish", **records[-1])), flush=True)
                if proc.returncode:
                    break
            return records
        with ThreadPoolExecutor(max_workers=2) as executor:
            records = [r for group in executor.map(lane, (0, 1)) for r in group]
        status = "complete" if len(records) == len(tasks) and all(r["exit_code"] == 0 for r in records) else "incomplete"
        (plan_path.parent / f"{suffix}_status.json").write_text(json.dumps(dict(status=status, records=records), indent=2))
        if status != "complete":
            raise SystemExit(1)
        return
    env = "env TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=" + shlex.quote(EXPERIMENT + "/code:" + BASE + "/code/v2_llm_fusion:" + BASE + "/code/frozen_v1_2026_09_09")
    if args.action == "test":
        cmd = [PYTHON, "-m", "pytest", EXPERIMENT + "/code/test_unified_q_text_diagnostics.py", "-q"]
    else:
        name = "run_unified_q_text_diagnostic_20261008.py" if args.action == "diagnose" else "run_unified_q_text_confirmation_20261008.py"
        tag = args.tag or f"{args.action}_{args.dataset}_o{args.order}_c{args.seed}"
        cmd = [PYTHON, "-u", EXPERIMENT + "/code/" + name,
            "--dataset", args.dataset, "--device", f"cuda:{args.gpu}", "--output", RESULT + "/" + tag,
            "--reference", BASE + f"/results/unified_q_inloop_cat_20261007/{args.dataset}/protocol.json",
            "--text-cache", BASE + f"/results/public_baselines_text_cache_20261001/{args.dataset}",
            "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
            "--order-seed", str(args.order), "--construction-seed", str(args.seed)]
        if args.action == "diagnose" and args.dataset == "wikics":
            cmd += ["--check-observer"]
        if args.smoke:
            cmd += ["--max-stages", "1", "--q-steps", "2"]
            if args.action == "confirm":
                cmd += ["--student-steps", "8", "--student-seeds", "150"]
    remote(env + " " + " ".join(map(shlex.quote, cmd)))


if __name__ == "__main__":
    main()
