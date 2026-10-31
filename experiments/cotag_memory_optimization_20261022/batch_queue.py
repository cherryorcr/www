"""One sequential experiment queue per GPU with immutable sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    sources = [HERE / "run.py", HERE / "README.md", Path(__file__)] + list((HERE / "implementation").glob("*.py"))
    hashes = {str(p): digest(p) for p in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"queue_{args.dataset}_{time.time_ns()}.jsonl").open("x") as ledger:
        def emit(**value):
            line = json.dumps(dict(time=time.time(), **value))
            ledger.write(line + "\n")
            ledger.flush()
            print(line, flush=True)

        def launch(variant, seed, limit):
            if hashes != {str(p): digest(p) for p in sources}:
                raise RuntimeError("source changed during batch")
            if shutil.disk_usage(ROOT).free < 8 * 2**30:
                raise RuntimeError("disk floor")
            output = args.output / f"{args.dataset}_{variant}_c{seed}"
            path = output / "results.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if existing["status"] == "complete" or (limit == 1 and existing["status"] == "pilot_complete"):
                    emit(event="skip", variant=variant, seed=seed, limit=limit)
                    return
                if existing["status"] == "failed":
                    raise RuntimeError(f"preserve failed run {path}")
            output.mkdir(parents=True, exist_ok=True)
            log = output / f"console_{limit}_{time.time_ns()}.log"
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false")
            cmd = [sys.executable, str(HERE / "run.py"), "--dataset", args.dataset,
                   "--variant", variant, "--seed", str(seed), "--stage-limit", str(limit),
                   "--output", str(output)]
            emit(event="start", variant=variant, seed=seed, limit=limit)
            with log.open("x") as stream:
                child = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            emit(event="exit", variant=variant, seed=seed, limit=limit, returncode=child.returncode)
            if child.returncode:
                raise RuntimeError(f"failed run; see {log}")

        for variant in ("original", "cache_offload", "class_blocks", "combined"):
            launch(variant, 271006, 1)
        emit(event="pilot_complete", source_hashes=hashes)
        if not args.pilot_only:
            for seed in (271006, 271108, 271109):
                for variant in ("original", "combined"):
                    launch(variant, seed, 0)
        emit(event="queue_complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "sportsfit"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-only", action="store_true")
    main(p.parse_args())
