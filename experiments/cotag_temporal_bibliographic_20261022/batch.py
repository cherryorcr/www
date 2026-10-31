"""One isolated queue per GPU; preserve logs and stop on a failed job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "results/cotag_temporal_bibliographic_20261022"


def main(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    folder = OUT / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [("encode", ["encode_data.py", "--dataset", args.dataset])]
    jobs += [(f"{arm}_c{seed}", ["run_temporal.py", "--dataset", args.dataset, "--arm", arm, "--seed", str(seed)])
             for seed in (271006, 271108, 271109) for arm in ("uq_dm_edgefree", "cat_cctp", "gecc_hard")]
    status = []
    for name, argv in jobs:
        logfile = folder / f"{args.dataset}_{name}_{args.attempt}.log"
        start = time.time()
        with logfile.open("x") as stream:
            child = subprocess.run([sys.executable, "-u", str(HERE / argv[0]), *argv[1:]], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        status.append(dict(job=name, dataset=args.dataset, returncode=child.returncode,
                           seconds=time.time()-start, log=str(logfile)))
        path = folder / f"{args.dataset}_batch_{args.attempt}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(path)
        print(json.dumps(status[-1]), flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("acm", "dblp"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--attempt", default="v1")
    main(parser.parse_args())
