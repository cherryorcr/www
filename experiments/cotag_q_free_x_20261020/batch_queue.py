"""Run a sequential batch on one visible GPU, preserving every attempt."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUEUES = {
    0: [("bookhis", 271108), ("sportsfit", 271006), ("bookhis", 271109)],
    1: [("arxiv", 271108), ("sportsfit", 271108), ("arxiv", 271109),
        ("sportsfit", 271109)],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    hashes = {name: digest(HERE / name) for name in ("run.py", "independent_x.py")}
    args.output_root.mkdir(parents=True, exist_ok=True)
    events = args.output_root / f"queue_{args.group}_{time.time_ns()}.jsonl"
    failed = False
    with events.open("x", encoding="utf-8") as ledger:
        def emit(**record):
            value = json.dumps(dict(time=time.time(), **record))
            ledger.write(value + "\n")
            ledger.flush()
            print(value, flush=True)

        for dataset, seed in QUEUES[args.group]:
            output = args.output_root / f"{dataset}_o9101_c{seed}"
            result_path = output / "results.json"
            if result_path.exists():
                record = json.loads(result_path.read_text())
                if (record.get("status") == "complete"
                        and record["protocol"]["runner_hash"] == hashes["run.py"]
                        and record["protocol"]["control_hash"] == hashes["independent_x.py"]):
                    emit(event="skip_verified_complete", dataset=dataset, seed=seed)
                    continue
                raise RuntimeError(f"preserve existing incomplete or different result: {output}")
            if shutil.disk_usage(ROOT).free < 8 * 2**30:
                raise RuntimeError("less than 8 GiB available; queue stopped")
            if any(digest(HERE / name) != value for name, value in hashes.items()):
                raise RuntimeError("source changed during queue")
            output.mkdir(parents=True, exist_ok=True)
            emit(event="start", dataset=dataset, seed=seed, source_hashes=hashes)
            with (output / "console.log").open("x", encoding="utf-8") as log:
                child = subprocess.run([sys.executable, str(HERE / "run.py"),
                    "--dataset", dataset, "--construction-seed", str(seed),
                    "--output", str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            complete = result_path.exists() and json.loads(result_path.read_text()).get("status") == "complete"
            failed |= child.returncode != 0 or not complete
            emit(event="finish", dataset=dataset, seed=seed,
                 returncode=child.returncode, complete=complete)
        emit(event="queue_complete", failures=failed)
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=tuple(QUEUES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
