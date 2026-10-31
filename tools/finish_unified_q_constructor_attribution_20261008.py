"""Collect this active queue to completion; no scheduling or remote edits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/unified_q_constructor_attribution_20261008_v2"
REMOTE = "/home/pc/research/TAG_condensation/results/unified_q_constructor_attribution_20261008_v2"


def run(script, *args):
    process = subprocess.run([sys.executable, str(ROOT / "tools" / script), *args],
        cwd=ROOT, text=True, capture_output=True, timeout=240)
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=4.)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_hours * 3600
    last_progress, previous_complete, failures = None, set(), 0
    sync_args = ("--remote", REMOTE, "--local", str(RESULT))
    while time.monotonic() < deadline:
        try:
            state = run("sync_unified_q_attribution_20261008.py", *sync_args)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            failures += 1
            print(json.dumps(dict(event="sync_retry", attempt=failures, error=str(exc)[-1200:])), flush=True)
            if failures >= 3:
                raise
            time.sleep(args.interval)
            continue
        failures = 0
        complete = {c["cell"] for c in state["cells"] if c["status"] == "complete"}
        missing = []
        for name in complete:
            value = json.loads((RESULT / name / "results.json").read_text())
            if any(not (RESULT / name / f"block_{arm}_s{stage}.pt").exists()
                   for arm in value["protocol"]["arms"] for stage in range(5)):
                missing.append(name)
        if missing:
            run("sync_unified_q_attribution_20261008.py", *sync_args,
                "--archives", "--cells", *sorted(missing))
        if complete and complete != previous_complete:
            audit = run("audit_unified_q_constructor_attribution_20261008.py")
            summary = run("summarize_unified_q_constructor_attribution_20261008.py")
            print(json.dumps(dict(event="audited", **audit, summary=summary)), flush=True)
            previous_complete = complete
        progress = [{k: c[k] for k in ("cell", "status", "prompt_stages")} for c in state["cells"]]
        if progress != last_progress:
            print(json.dumps(dict(event="progress", complete=len(complete), cells=progress)), flush=True)
            last_progress = progress
        status_path = RESULT / "all_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["status"] != "complete":
                raise RuntimeError(f"Remote queue stopped incomplete: {status}")
            if len(complete) != 9:
                raise RuntimeError("Queue complete but expected cells are missing")
            audit = json.loads((RESULT / "archive_audit.json").read_text())
            if (not audit["formal_matrix_complete"] or audit["evaluated_memories"] != 324
                    or audit["fixed_payload_pairs"] != 54):
                raise RuntimeError("Final audit counts do not match the registered matrix")
            print(json.dumps(dict(event="complete", **audit)), flush=True)
            return
        time.sleep(args.interval)
    raise TimeoutError("Active collection timeout; remote jobs were not modified")


if __name__ == "__main__":
    main()
