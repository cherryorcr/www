"""Read-only SSH status and small-artifact sync for the frozen UQ experiment."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


REMOTE = r'''
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
files, cells = [], []
for path in sorted(root.rglob("*.json")):
    files.append(str(path.relative_to(root)))
    if path.name != "results.json":
        continue
    result = json.loads(path.read_text())
    arms = {}
    for row in result["rows"]:
        arms[row["arm"]] = {"stages": row["stage"] + 1,
            "final_f1": row["score"]["final_f1"],
            "accuracy": row["score"]["accuracy"]}
    cells.append({"cell": str(path.parent.relative_to(root)),
        "status": result["status"], "arms": arms})
processes = subprocess.check_output(
    ["ps", "-eo", "pid,etime,args"], text=True).splitlines()
processes = [p for p in processes if "python" in p and any(name in p for name in (
    "run_unified_q_revision_20261007.py", "run_unified_q_confirmation_20261007.py",
    "run_unified_q_diagnostics_20261007.py", "run_unified_q_diagnostics_followup_20261007.py",
    "run_unified_q_matched_baselines_20261007.py", "run_unified_q_fixed_content_20261007.py",
    "run_aux.py", "run_unified_q_inloop_cat_20261007.py"))]
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
    "--format=csv,noheader"], text=True).strip()
print(json.dumps({"files": files, "cells": cells, "processes": processes,
    "gpu": gpu, "free_gib": shutil.disk_usage(root).free / 2**30}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="pc@100.126.175.112")
    parser.add_argument("--remote", default=(
        "/home/pc/research/TAG_condensation/results/unified_q_revision_verified_20261007"))
    parser.add_argument("--local", type=Path, default=Path(
        "results/unified_q_revision_verified_20261007"))
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        args.host, "/home/pc/miniconda3/envs/tagsam/bin/python", "-", args.remote],
        input=REMOTE, text=True, encoding="utf-8", capture_output=True, check=True)
    status = json.loads(response.stdout)
    if not args.status_only:
        root = args.local.resolve()
        command = " ".join(shlex.quote(v) for v in
            ["tar", "-czf", "-", "-C", args.remote, "--", *status["files"]])
        archive = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, command],
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            for member in bundle:
                destination = (root / member.name).resolve()
                if (not destination.is_relative_to(root) or not member.isfile()
                        or member.name not in status["files"]):
                    raise ValueError("unexpected remote archive member")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".sync")
                temporary.write_bytes(bundle.extractfile(member).read())
                temporary.replace(destination)
        (root / "remote_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if args.compact:
        status = {"complete_run_cells": sum(c["status"] == "complete" for c in status["cells"]),
            "incomplete": [{"cell": c["cell"], "stages": {a: v["stages"] for a, v in c["arms"].items()}}
                for c in status["cells"] if c["status"] != "complete"],
            "active_processes": len(status["processes"]), "gpu": status["gpu"],
            "free_gib": status["free_gib"]}
    print(json.dumps({k: v for k, v in status.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
