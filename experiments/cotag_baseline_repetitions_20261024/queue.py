"""Registered repetitions of existing table adapters, without outcome-based changes."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/pc/research/TAG_condensation')
EXP = ROOT / 'experiments/cotag_baseline_repetitions_20261024'
OUT = ROOT / 'results/cotag_baseline_repetitions_20261024'
PYTHON = '/home/pc/miniconda3/envs/tagsam/bin/python'
ENCODER = '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip'
ORDERS = [9101, 9102, 9103]
SEEDS = [271006, 271108, 271109]
STUDENTS = [150, 151, 152]
DATASETS = ['cora', 'citeseer', 'bookhis', 'arxiv', 'sportsfit']
ARMS = ['tagsam', 'random_coreset', 'cluster_medoid', 'herding_coreset',
        'kcenter_coreset', 'gcond_x_core', 'sfgc_core', 'stad_native_local']
STAGES = dict(cora=3, citeseer=3, bookhis=5, arxiv=3, sportsfit=5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def arm_name(dataset, arm):
    if arm == 'tagsam':
        return 'tagsam_native_bounded' if dataset in DATASETS[:2] else 'tagsam_selector_response'
    return arm


def validate(path, job):
    data = json.loads(path.read_text())
    p = data['protocol']
    assert data['status'] == 'complete', 'chain incomplete'
    assert p['dataset'] == job['dataset']
    assert p['order_seed'] == job['order'] and p['construction_seed'] == job['seed']
    assert p['student_seeds'] == STUDENTS
    assert p['student_steps'] == 2048 and p['student_batch'] == 24
    assert p['memory_rate'] == .05
    rows = data['rows']
    assert len(rows) == STAGES[job['dataset']]
    assert [r['stage'] for r in rows] == list(range(len(rows)))
    assert all(r['arm'] == job['native_arm'] for r in rows)
    for row in rows:
        score = row['scores']['sealed_test']
        assert sorted(v['seed'] for v in score['per_student']) == STUDENTS
        for key in ['final_f1', 'accuracy']:
            values = [v[key] for v in score['per_student']]
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in values)
            assert abs(statistics.mean(values) - score[key]) < 1e-8
        assert row['memory_rows'] == p['budgets'][row['stage']]
    if 'expected_masks' in job:
        assert all(p[k] == v for k, v in job['expected_masks'].items()), 'mask differs'
    if 'expected_source_hashes' in job:
        assert p['source_hashes'] == job['expected_source_hashes'], 'implementation differs'
    return data


def original_candidates(dataset, arm):
    native = arm_name(dataset, arm)
    if arm in ARMS[1:-1]:
        suffix = '_v2' if arm in ['gcond_x_core', 'sfgc_core'] else ''
        return [ROOT / f'results/uq_baseline_expansion_20261013/cells/{dataset}_{arm}_o9101_c271006{suffix}/results.json']
    if dataset in DATASETS[:2]:
        return [ROOT / f'results/cotag_q_missing_baselines_20261019_repair_v2/{dataset}/{native}/results.json']
    category = 'baselines' if arm == 'tagsam' else 'stad'
    prefix = ROOT / f'results/uq_final_evidence_20261012/{category}'
    return [prefix / f'{native}_{dataset}_o9101_c271006{suffix}/results.json'
            for suffix in ['', '_retry20261013']]


def command(job):
    ds, native = job['dataset'], job['native_arm']
    code = EXP / 'code'
    if job['arm'] in ARMS[1:-1]:
        runner = code / 'expansion/run_uq_baseline_expansion_20261013.py'
        flags = ['--arms', native, '--root', str(ROOT)]
        module_path = code / 'expansion'
    elif ds in DATASETS[:2]:
        runner = code / 'run_cotag_q_missing_small_baselines_20261019.py'
        flags = ['--arm', native, '--root', str(ROOT)]
        module_path = ROOT / 'tools'
    else:
        runner = code / 'native/run_uq_native_baseline_batch_20261012.py'
        flags = ['--arm', native]
        module_path = code / 'native'
    cache = (ROOT / 'results/uq_baseline_expansion_20261013/text_cache' if ds in DATASETS[:2]
             else ROOT / 'results/public_baselines_text_cache_20261001') / ds
    args = [PYTHON, '-u', str(runner), '--dataset', ds, *flags,
            '--text-cache', str(cache), '--minilm', ENCODER,
            '--output', job['output'], '--order-seed', str(job['order']),
            '--construction-seed', str(job['seed']), '--device', 'cuda:0']
    if ds not in DATASETS[:2]:
        args.extend(['--reference', str(ROOT / f'experiments/uq_final_evidence_20261012/{ds}_reference.json')])
    return args, ':'.join(map(str, [module_path, ROOT / 'code/v2_llm_fusion', ROOT / 'tools', ROOT]))


def register():
    target = EXP / 'registration.json'
    if target.exists():
        raise RuntimeError('Registration already exists. Use status or worker; do not overwrite.')
    code = EXP / 'code'
    code.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'experiments/uq_baseline_expansion_20261013/code_v2', code / 'expansion')
    shutil.copytree(ROOT / 'experiments/uq_native_retry_20261013/code', code / 'native')
    shutil.copy2(ROOT / 'tools/run_cotag_q_missing_small_baselines_20261019.py', code)
    originals, jobs, masks_by_dataset = [], [], {}
    for ds, arm in itertools.product(DATASETS, ARMS):
        job = dict(dataset=ds, arm=arm, native_arm=arm_name(ds, arm), order=ORDERS[0], seed=SEEDS[0])
        selected = None
        for p in original_candidates(ds, arm):
            if p.exists() and json.loads(p.read_text()).get('status') == 'complete':
                selected = p
                break
        if selected is None:
            raise RuntimeError(f'No completed original: {ds}/{arm}')
        data = validate(selected, job)
        masks = {k: data['protocol'][k] for k in ['pool_mask_hash', 'report_mask_hash', 'test_mask_hash']}
        assert masks_by_dataset.setdefault(ds, masks) == masks, 'main-table masks differ'
        job.update(source=str(selected), source_sha256=sha(selected), expected_masks=masks,
                   expected_source_hashes=data['protocol']['source_hashes'])
        originals.append(job)
    for order, seed, ds, arm in itertools.product(ORDERS, SEEDS, DATASETS, ARMS):
        if (order, seed) == (ORDERS[0], SEEDS[0]):
            continue
        original = next(j for j in originals if (j['dataset'], j['arm']) == (ds, arm))
        name = f'{ds}_{arm}_o{order}_c{seed}'
        job = {k: original[k] for k in ['dataset', 'arm', 'native_arm', 'expected_masks', 'expected_source_hashes']}
        job.update(id=name, order=order, seed=seed, lane=len(jobs) % 2,
                   output=str(OUT / 'runs' / name))
        job['command'], job['pythonpath'] = command(job)
        jobs.append(job)
    record = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        purpose='Repeat existing eight main-table adapters; no native-core TAGSAM expansion.',
        orders=ORDERS, construction_seeds=SEEDS, student_seeds=STUDENTS,
        student_updates=2048, student_batch=24, memory_rate=.05, stages=STAGES,
        access='current increment plus frozen historical memory',
        test_status='Previously inspected historical masks, not newly blind test sets.',
        statistics='Average students within each order/construction, then mean and sample SD across nine means. Orders and seeds are crossed factors.',
        safeguards='No test-based tuning; no automatic failed-job retries; disk floor 8 GiB; one process per GPU.',
        metadata_note='Small-adapter inherited construction_seed_rule text mentions 271006. Numeric construction_seed and CLI determine the actual seed.',
        source_files={str(p.relative_to(EXP)): sha(p) for p in code.rglob('*.py')},
        originals=originals, jobs=jobs)
    save(target, record)
    print(json.dumps(dict(reused=len(originals), planned=len(jobs), total=len(originals)+len(jobs))))


def worker(lane):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'lane{lane}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((EXP / 'registration.json').read_text())
        assert all(sha(EXP / p) == digest for p, digest in record['source_files'].items())
        for job in record['jobs']:
            if job['lane'] != lane:
                continue
            output = Path(job['output'])
            receipt = output / 'queue_receipt.json'
            result = output / 'results.json'
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old['status'] == 'complete':
                    validate(result, job)
                continue
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f'Unregistered partial output requires inspection: {output}')
            if shutil.disk_usage(ROOT).free < 8 * 2**30:
                save(OUT / f'lane{lane}_status.json', dict(status='blocked_disk', job=job['id']))
                return
            output.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=job['pythonpath'])
            state = dict(status='running', job=job['id'], pid=os.getpid(), start=time.time(),
                         command=job['command'], lane=lane)
            save(receipt, state)
            save(OUT / f'lane{lane}_status.json', state)
            try:
                with (output / 'run.log').open('w') as log:
                    proc = subprocess.run(job['command'], env=env, cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT, timeout=6*3600)
                if proc.returncode:
                    raise RuntimeError(f'runner exit {proc.returncode}; see run.log')
                validate(result, job)
                state.update(status='complete', result_sha256=sha(result))
            except Exception as error:
                state.update(status='failed', error=str(error))
            state['end'] = time.time()
            save(receipt, state)
            save(OUT / f'lane{lane}_status.json', state)
            print(json.dumps(state), flush=True)
        save(OUT / f'lane{lane}_status.json', dict(status='queue_exhausted', time=time.time()))


def status():
    record = json.loads((EXP / 'registration.json').read_text())
    rows, pending, failed, running = [], [], [], []
    for job in record['originals'] + record['jobs']:
        old = 'source' in job
        p = Path(job['source']) if old else Path(job['output']) / 'results.json'
        if old:
            assert sha(p) == job['source_sha256'], 'original result changed'
        receipt = {} if old else Path(job['output']) / 'queue_receipt.json'
        receipt = json.loads(receipt.read_text()) if not old and receipt.exists() else {}
        if not old and receipt.get('status') != 'complete':
            dest = failed if receipt.get('status') == 'failed' else running if receipt.get('status') == 'running' else pending
            dest.append(dict(job=job['id'], error=receipt.get('error')))
            continue
        data = validate(p, job)
        score = data['rows'][-1]['scores']['sealed_test']
        rows.append(dict(dataset=job['dataset'], arm=job['arm'], order=job['order'],
            seed=job['seed'], final_f1=score['final_f1'], accuracy=score['accuracy'], source=str(p), sha256=sha(p)))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'completed_constructions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    state = dict(complete=len(rows), reused=len(record['originals']), planned_total=360,
                 new_complete=len(rows)-len(record['originals']), pending=len(pending),
                 failed=failed, running=running, free_GiB=shutil.disk_usage(ROOT).free/2**30)
    save(OUT / 'status.json', state)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'worker', 'status'])
    parser.add_argument('--lane', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'worker':
        if args.lane is None:
            parser.error('--lane is required')
        worker(args.lane)
    else:
        status()
