"""Serialized jobs per GPU, with remote-file logs and explicit failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/pc/research/TAG_condensation')
OUT = ROOT / 'results/cotag_fidelity_temporal_20261021'


def main(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.track == 'temporal':
        jobs = [(f'{arm}_c{seed}', ['run_temporal.py', '--arm', arm, '--seed', str(seed)])
                for seed in (271006, 271108, 271109)
                for arm in ('uq_dm_edgefree', 'cat_cctp', 'gecc_hard')]
    elif args.track == 'tagsam':
        jobs = [(dataset, ['run_tagsam.py', '--dataset', dataset])
                for dataset in ('bookhis', 'arxiv', 'sportsfit')]
    elif args.track == 'precision':
        jobs = [(f'uq_dm_edgefree_c{seed}', ['run_temporal_precise.py', '--seed', str(seed)])
                for seed in (271006, 271108, 271109)]
    elif args.track == 'native_teacher':
        jobs = [('cora', ['run_native_teacher_budget.py', '--dataset', 'cora', '--outer', '5000'])]
    else:
        jobs = [('cora', ['run_native_small.py', '--dataset', 'cora', '--outer', '5000'])]
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, script in jobs:
        log = logs / f'{args.track}_{name}_{args.attempt}.log'
        if log.exists():
            raise FileExistsError('Preserve existing logs; inspect before restarting a batch')
        start = time.time()
        with log.open('x') as stream:
            result = subprocess.run([sys.executable, '-u', str(HERE/script[0]), *script[1:]],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        summary.append(dict(job=name, returncode=result.returncode,
                            seconds=time.time()-start, log=str(log)))
        path = logs / f'{args.track}_batch_{args.attempt}.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(path)
        print(json.dumps(summary[-1]), flush=True)
    if any(row['returncode'] for row in summary):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', choices=('temporal', 'tagsam', 'precision', 'native', 'native_teacher'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--attempt', default='v1')
    main(parser.parse_args())
