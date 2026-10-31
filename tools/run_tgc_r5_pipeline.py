"""Prepare the entire matrix before scoring; fail closed on any worker error."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code = root / "code/v2_llm_fusion"
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    started = time.perf_counter()
    completed = []

    def status(**value):
        value.update(elapsed_seconds=time.perf_counter() - started, completed=completed)
        (args.output / "pipeline_status.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

    for phase in ("prepare", "seal", "evaluate", "audit"):
        jobs = []
        for index, dataset in enumerate((None,) if phase == "seal" else ("cora", "citeseer")):
            name = f"{dataset or 'all'}_{phase}"
            command = [sys.executable, "-u", args.runner, "--phase", phase,
                       "--output", str(args.output), "--device", f"cuda:{index}"]
            if dataset:
                command += ["--dataset", dataset]
            handle = (args.output / f"{name}.log").open("w")
            process = subprocess.Popen(command, cwd=code, stdout=handle, stderr=subprocess.STDOUT)
            jobs.append((name, process, handle))
        while any(p.poll() is None for _, p, _ in jobs):
            status(status="running", phase=phase, jobs={n: p.poll() for n, p, _ in jobs})
            time.sleep(10)
        failed = []
        for name, process, handle in jobs:
            handle.close()
            completed.append({"job": name, "exit_code": process.returncode})
            if process.returncode:
                failed.append(name)
                print((args.output / f"{name}.log").read_text()[-6000:], flush=True)
        if failed:
            status(status="failed", phase=phase, failed=failed)
            raise RuntimeError(f"failed jobs: {failed}")
    status(status="complete")
    print("TGC_R5_PIPELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
