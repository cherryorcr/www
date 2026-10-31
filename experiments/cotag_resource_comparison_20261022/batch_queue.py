"""Sequential registered attempts. Preserve failures and continue other attempts."""
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


def main(args):
    sources = [HERE / n for n in ("README.md", "run.py", "protocol_utils.py", "batch_queue.py")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for seed in (271006, 271108, 271109):
        for arm in ("full_history", "cat_cctp", "gecc_hard"):
            assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
            if shutil.disk_usage(ROOT).free < 8 * 2**30:
                raise RuntimeError("disk floor")
            out = args.output / f"{args.dataset}_{arm}_c{seed}"
            if out.exists():
                result = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
                outcomes.append(dict(arm=arm, seed=seed, status=result.get("status", "existing_incomplete"), skipped=True))
                continue
            log = args.output / f"{out.name}_{time.time_ns()}.log"
            print(json.dumps(dict(event="start", dataset=args.dataset, arm=arm, seed=seed)), flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            with log.open("x") as stream:
                process = subprocess.run([sys.executable, str(HERE / "run.py"), "--dataset", args.dataset,
                    "--arm", arm, "--seed", str(seed), "--output", str(out)], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT)
            row = dict(arm=arm, seed=seed, returncode=process.returncode, log=log.name)
            outcomes.append(row)
            print(json.dumps(dict(event="exit", dataset=args.dataset, **row)), flush=True)
    (args.output / f"queue_{args.dataset}_{time.time_ns()}.json").write_text(json.dumps(outcomes, indent=2))
    print(json.dumps(dict(event="queue_complete", dataset=args.dataset)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "sportsfit"), required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
