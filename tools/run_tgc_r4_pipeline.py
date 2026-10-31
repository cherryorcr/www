"""Run the frozen remote matrix and retain logs, status, and numerical preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "v2_llm_fusion"
sys.path.insert(0, str(CODE))


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    import torch
    import tgc_relational_response as M
    import run_tgc_r4_relational as B
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    logs = args.output / "preflight.log"
    with logs.open("w") as handle:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "test_tgc_relational_response.py",
                        "test_tgc_student_revision.py", "test_tgc_revision_gate.py"],
                       cwd=CODE, stdout=handle, stderr=subprocess.STDOUT, check=True)
    numeric = []
    for dataset in B.DATASETS:
        for order in B.ORDERS:
            for stage in (1, 2):
                path = args.pilot / dataset / order / f"c0_s{stage}_memory.pt"
                memory = torch.load(path, map_location="cpu", weights_only=False)
                result = M.numerical_check(memory, 8100, "cuda:0")
                numeric.append({"dataset": dataset, "order": order, "stage": stage, "comparisons": result})
    write(args.output / "actual_graph_numerical_preflight.json", {"strict_passed": True, "records": numeric})
    print("PREFLIGHT_PASSED", flush=True)
    statuses = []
    for phase in ("prepare", "seal", "evaluate", "audit"):
        jobs = []
        datasets = (None,) if phase == "seal" else B.DATASETS
        for index, dataset in enumerate(datasets):
            name = f"{dataset or 'all'}_{phase}"
            command = [sys.executable, "-u", "run_tgc_r4_relational.py", "--phase", phase,
                       "--output", str(args.output), "--device", f"cuda:{index}"]
            if dataset:
                command += ["--dataset", dataset]
            handle = (args.output / f"{name}.log").open("w")
            process = subprocess.Popen(command, cwd=CODE, stdout=handle, stderr=subprocess.STDOUT)
            jobs.append((name, process, handle))
        while any(process.poll() is None for _, process, _ in jobs):
            state = {name: process.poll() for name, process, _ in jobs}
            write(args.output / "pipeline_status.json", {"phase": phase, "jobs": state,
                                                         "elapsed_seconds": time.perf_counter() - t0})
            print(json.dumps({"phase": phase, "jobs": state}), flush=True)
            time.sleep(30)
        errors = []
        for name, process, handle in jobs:
            handle.close()
            statuses.append({"job": name, "exit_code": process.returncode})
            if process.returncode:
                errors.append(name)
                print((args.output / f"{name}.log").read_text()[-9000:], flush=True)
        if errors:
            write(args.output / "pipeline_status.json", {"status": "failed", "phase": phase, "jobs": statuses})
            raise RuntimeError(f"R4 jobs failed: {errors}")
    write(args.output / "pipeline_status.json", {"status": "complete", "jobs": statuses,
                                                 "elapsed_seconds": time.perf_counter() - t0})
    print("TGC_R4_MATRIX_AND_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
