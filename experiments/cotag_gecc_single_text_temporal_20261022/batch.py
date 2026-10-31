"""Bounded serial queue, with no reruns of finished or failed sequences."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "results/cotag_gecc_single_text_temporal_20261022"


def main(args):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    history = []
    for dataset in args.datasets:
        for seed in (271006, 271108, 271109):
            result_path = OUT / "runs" / dataset / f"gecc_nearest1_c{seed}/results.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved["status"] == "complete":
                    print(json.dumps(dict(event="skip_complete_for_later_verification", dataset=dataset, seed=seed)), flush=True)
                    continue
                raise RuntimeError(f"Existing incomplete/failed run preserved: {result_path}")
            log = logs / f"{dataset}_c{seed}_{args.attempt}.log"
            started = time.time()
            with log.open("x") as stream:
                child = subprocess.run([sys.executable, "-u", str(HERE / "run_single.py"),
                    "--dataset", dataset, "--seed", str(seed)], env=environment,
                    stdout=stream, stderr=subprocess.STDOUT)
            history.append(dict(dataset=dataset, seed=seed, returncode=child.returncode,
                                seconds=time.time()-started, log=str(log)))
            status = logs / f"batch_{args.attempt}.json"
            temporary = status.with_suffix(".tmp")
            temporary.write_text(json.dumps(history, indent=2))
            temporary.replace(status)
            print(json.dumps(history[-1]), flush=True)
            if child.returncode:
                raise SystemExit(child.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("acm", "dblp", "arxiv"), default=["acm", "dblp", "arxiv"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--attempt", default="v1")
    main(parser.parse_args())
