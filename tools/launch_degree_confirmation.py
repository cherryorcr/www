"""Supervise two GPU workers, preserving logs and nonzero exit status."""
import json
from pathlib import Path
import subprocess
import sys
import time


root = Path(__file__).resolve().parents[1]
out = Path("/home/pc/research/TAG_condensation/results/degree_confirmation_20260920")
out.mkdir(parents=True, exist_ok=True)
workers, handles = {}, []
for dataset, device in (("arxiv", "cuda:0"), ("bookchild", "cuda:1")):
    log = (out / (dataset + ".log")).open("a", buffering=1)
    handles.append(log)
    workers[dataset] = subprocess.Popen(
        [sys.executable, "-u", str(root / "tools/degree_confirmation_worker.py"),
         "--dataset", dataset, "--device", device], cwd=root, stdout=log, stderr=subprocess.STDOUT)
print(json.dumps({"worker_pids": {k: p.pid for k, p in workers.items()}}), flush=True)
last = None
while True:
    status = {}
    for dataset, process in workers.items():
        fits = list((out / dataset).glob("order_*/*/*_seed*.json"))
        status[dataset] = {"fits": len(fits), "exit": process.poll(),
                           "complete": (out / dataset / "pipeline_complete.json").exists()}
    if status != last:
        print(json.dumps(status), flush=True)
        last = status
    if all(p.poll() is not None for p in workers.values()):
        break
    time.sleep(30)
for handle in handles:
    handle.close()
raise SystemExit(0 if all(p.returncode == 0 for p in workers.values()) else 1)
