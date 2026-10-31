"""Sequential per-GPU extension; one dataset per invocation, no detached jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cora", "citeseer"])
    ap.add_argument("--device", required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).parent
    commands = [
        ("supervised", ["run_supervised_upper_bound_small.py"]),
        ("static", ["run_tagsam_graphclip_static.py"]),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    status = {"dataset": args.dataset, "device": args.device, "jobs": []}
    status_path = root / f"{args.dataset}_status.json"
    for name, runner in commands:
        output = root / name / args.dataset
        command = [sys.executable, "-u", str(code / runner[0]), "--dataset", args.dataset,
                   "--device", args.device, "--output", str(output)]
        started = time.time()
        row = {"name": name, "command": command, "started": started, "state": "running"}
        status["jobs"].append(row)
        status_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(row), flush=True)
        with (root / f"{args.dataset}_{name}.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        row.update(exit_code=result.returncode, seconds=time.time() - started,
                   state="complete" if result.returncode == 0 else "failed")
        status_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(row), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
