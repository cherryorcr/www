"""Use an idle GPU for a reserved suffix of the unchanged experiment queue."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import queue as registered_queue


def worker(lane):
    q = registered_queue
    with (q.OUT / f'lane{lane}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((q.OUT / f'lane{lane}_status.json').read_text())
        if old['status'] != 'queue_exhausted':
            raise RuntimeError('The target GPU queue has not finished.')
        record = json.loads((q.EXP / 'registration.json').read_text())
        assert all(q.sha(q.EXP / p) == digest for p, digest in record['source_files'].items())
        pending = [job for job in record['jobs'] if job['lane'] != lane
                   and not (Path(job['output']) / 'queue_receipt.json').exists()]
        if len(pending) < 20:
            raise RuntimeError('Too few unclaimed jobs for a disjoint suffix reservation.')
        # The original worker moves forward; reserve the far end before running any job.
        selected = pending[-(len(pending) // 2):]
        manifest = q.OUT / f'lane{lane}_balanced_reservation.json'
        with manifest.open('x') as stream:
            json.dump(dict(time=time.time(), gpu=lane,
                           scheduling_only=True, jobs=[j['id'] for j in selected]), stream, indent=2)
        for job in selected:
            output = Path(job['output'])
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise RuntimeError(f'Output already claimed: {output}')
            with (output / 'queue_receipt.json').open('x') as stream:
                json.dump(dict(status='reserved', job=job['id'], lane=lane,
                               registered_lane=job['lane'], scheduling_only=True), stream)
        for job in selected:
            if shutil.disk_usage(q.ROOT).free < 8 * 2**30:
                q.save(q.OUT / f'lane{lane}_status.json', dict(status='blocked_disk', job=job['id']))
                return
            output = Path(job['output'])
            receipt = output / 'queue_receipt.json'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=job['pythonpath'])
            state = dict(status='running', job=job['id'], lane=lane, registered_lane=job['lane'],
                         scheduling_only=True, pid=os.getpid(), start=time.time(), command=job['command'])
            q.save(receipt, state)
            q.save(q.OUT / f'lane{lane}_status.json', state)
            try:
                with (output / 'run.log').open('w') as log:
                    proc = subprocess.run(job['command'], env=env, cwd=q.ROOT,
                                          stdout=log, stderr=subprocess.STDOUT, timeout=6*3600)
                if proc.returncode:
                    raise RuntimeError(f'runner exit {proc.returncode}; see run.log')
                result = output / 'results.json'
                q.validate(result, job)
                state.update(status='complete', result_sha256=q.sha(result))
            except Exception as error:
                state.update(status='failed', error=str(error))
            state['end'] = time.time()
            q.save(receipt, state)
            q.save(q.OUT / f'lane{lane}_status.json', state)
            print(json.dumps(state), flush=True)
        q.save(q.OUT / f'lane{lane}_status.json', dict(status='balanced_queue_exhausted', time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=int, choices=[0, 1], required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        log_path = registered_queue.OUT / f'lane{args.lane}_balanced_worker.log'
        with log_path.open('x') as log:
            proc = subprocess.Popen([registered_queue.PYTHON, '-u', __file__, '--lane', str(args.lane)],
                                    cwd=registered_queue.ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, gpu=args.lane, log=str(log_path))))
    else:
        worker(args.lane)
