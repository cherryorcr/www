"""Five-stage ArXiv comparisons using the frozen main-table implementations."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path('/home/pc/research/TAG_condensation')
EXP = ROOT / 'experiments/cotag_arxiv_five_stage_20261024'
OUT = ROOT / 'results/cotag_arxiv_five_stage_20261024'
PREVIOUS = ROOT / 'experiments/cotag_baseline_repetitions_20261024'
LOCKS = ROOT / 'results/cotag_baseline_repetitions_20261024'
PYTHON = '/home/pc/miniconda3/envs/tagsam/bin/python'
ARMS = ['uq_dm_edgefree', 'cat_cctp', 'gecc_hard', 'random_coreset',
        'kcenter_coreset', 'cluster_medoid', 'herding_coreset', 'gcond_x_core',
        'sfgc_core', 'tagsam', 'stad_native_local']
ORDERS = [9101, 9102, 9103]
SEEDS = [271006, 271108, 271109]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def previous_validator():
    spec = importlib.util.spec_from_file_location('previous_repetitions', PREVIOUS / 'queue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STAGES = dict(module.STAGES, arxiv=5)
    return module


def validate(path, job):
    data = previous_validator().validate(path, job)
    p = data['protocol']
    assert [len(group) for group in p['groups']] == [8] * 5
    assert sorted(itertools.chain.from_iterable(p['groups'])) == list(range(40))
    assert sum(p['stage_raw_rows']) == 118539 and p['budgets'][-1] == 5927
    for row in data['rows']:
        assert all(value for name, value in row['checks'].items()
                   if isinstance(value, bool)), row['checks']
    return data


def register():
    target = EXP / 'registration.json'
    if target.exists():
        raise RuntimeError('Registration is immutable; use status or launch.')
    previous = json.loads((PREVIOUS / 'registration.json').read_text())
    originals = {j['arm']: j for j in previous['originals'] if j['dataset'] == 'arxiv'}
    for folder in ['expansion', 'native']:
        shutil.copytree(PREVIOUS / 'code' / folder, EXP / 'code' / folder)
    reference = dict(name='arxiv-five-stage-main-comparison-20261024', dataset='arxiv',
        groups=[list(range(i, i + 8)) for i in range(0, 40, 8)], report_dev_mod=3,
        budgets=[1631, 3107, 3718, 4198, 5927],
        budget_note='Reference order only. Recompute cumulative 5% budgets from each order legal pool.')
    save(EXP / 'reference.json', reference)
    jobs = []
    for order, seed, arm in itertools.product(ORDERS, SEEDS, ARMS):
        native = 'tagsam_selector_response' if arm == 'tagsam' else arm
        original = originals.get(arm, originals['cluster_medoid'])
        name = f'arxiv_{arm}_o{order}_c{seed}'
        output = OUT / 'runs' / name
        if arm in ['tagsam', 'stad_native_local']:
            folder = EXP / 'code/native'
            runner = folder / 'run_uq_native_baseline_batch_20261012.py'
            flags = ['--arm', native]
        else:
            folder = EXP / 'code/expansion'
            runner = folder / 'run_uq_baseline_expansion_20261013.py'
            flags = ['--arms', native, '--root', str(ROOT)]
        command = [PYTHON, '-u', str(runner), '--dataset', 'arxiv', *flags,
            '--reference', str(EXP / 'reference.json'),
            '--text-cache', str(ROOT / 'results/public_baselines_text_cache_20261001/arxiv'),
            '--minilm', '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip',
            '--output', str(output), '--order-seed', str(order),
            '--construction-seed', str(seed), '--device', 'cuda:0']
        jobs.append(dict(id=name, dataset='arxiv', arm=arm, native_arm=native,
            order=order, seed=seed, output=str(output), command=command,
            pythonpath=':'.join(map(str, [folder, ROOT / 'code/v2_llm_fusion', ROOT / 'tools', ROOT])),
            expected_masks=original['expected_masks'],
            expected_source_hashes=original['expected_source_hashes']))
    files = [EXP / 'queue.py', EXP / 'reference.json', PREVIOUS / 'queue.py']
    for directory in [EXP / 'code', ROOT / 'experiments/uq_final_evidence_20261012/code']:
        files.extend(directory.rglob('*.py'))
    record = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        purpose='Replace three-stage ArXiv in the main table with a verified five-stage comparison.',
        orders=ORDERS, construction_seeds=SEEDS, student_seeds=[150, 151, 152],
        stages=5, classes_per_stage=[8]*5, memory_rate=.05, student_updates=2048,
        student_batch=24, q_updates=30, q_sinkhorn_iteration_cap=4096,
        q_numerical_policy='Unchanged frozen main implementation; no automatic numerical repair or failed-job retry.',
        access='Current increment and own frozen memory; common transductive evaluation graph.',
        test_status='Existing historical test mask, already inspected in earlier studies; not a newly blind test.',
        prior_five_stage='20261018 development scores cannot be reused as test scores; memories were not retained.',
        provenance='Same adapters and objectives as the current main table. No additional TAGSAM native-budget study.',
        statistics='Average three students within each memory; mean and sample SD across 3 orders x 3 construction seeds.',
        scheduling='Reuse GPU lane locks after previous repetitions finish; one job per GPU; 8 GiB free-disk floor.',
        source_files={str(p): sha(p) for p in files}, jobs=jobs)
    save(target, record)
    print(json.dumps(dict(registered=len(jobs), reference=reference)))


def worker(lane):
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / f'lane{lane}_status.json'
    with (OUT / f'worker{lane}.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (LOCKS / f'lane{lane}.lock').open('a') as gpu_lock:
            save(state_path, dict(status='waiting_for_previous_queue', lane=lane, pid=os.getpid()))
            # Sharing the existing lock prevents overlap with the original or balanced worker.
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            record = json.loads((EXP / 'registration.json').read_text())
            for job in record['jobs']:
                output = Path(job['output'])
                receipt = output / 'queue_receipt.json'
                with (OUT / 'claims.lock').open('a') as claim_lock:
                    fcntl.flock(claim_lock, fcntl.LOCK_EX)
                    if receipt.exists():
                        continue
                    if output.exists() and any(output.iterdir()):
                        raise RuntimeError(f'Unregistered partial output: {output}')
                    if shutil.disk_usage(ROOT).free < 8 * 2**30:
                        save(state_path, dict(status='blocked_disk', job=job['id']))
                        return
                    assert all(sha(p) == digest for p, digest in record['source_files'].items())
                    state = dict(status='running', job=job['id'], lane=lane, pid=os.getpid(),
                                 start=time.time(), command=job['command'])
                    save(receipt, state)
                save(state_path, state)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=job['pythonpath'])
                try:
                    with (output / 'run.log').open('w') as log:
                        process = subprocess.run(job['command'], env=env, cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, timeout=6*3600)
                    if process.returncode:
                        raise RuntimeError(f'runner exit {process.returncode}; see run.log')
                    validate(output / 'results.json', job)
                    state.update(status='complete', result_sha256=sha(output / 'results.json'))
                except Exception as error:
                    state.update(status='failed', error=str(error))
                state['end'] = time.time()
                save(receipt, state)
                save(state_path, state)
                print(json.dumps(state), flush=True)
                if state['status'] == 'failed':
                    return
            save(state_path, dict(status='queue_exhausted', time=time.time()))


def status():
    record = json.loads((EXP / 'registration.json').read_text())
    counts, rows, problems = {}, [], []
    for job in record['jobs']:
        receipt = Path(job['output']) / 'queue_receipt.json'
        state = json.loads(receipt.read_text()) if receipt.exists() else dict(status='pending')
        counts[state['status']] = counts.get(state['status'], 0) + 1
        if state['status'] == 'complete':
            result = Path(job['output']) / 'results.json'
            data = validate(result, job)
            score = data['rows'][-1]['scores']['sealed_test']
            rows.append(dict(arm=job['arm'], order=job['order'], seed=job['seed'],
                f1=score['final_f1'], accuracy=score['accuracy'], source=str(result), sha256=sha(result)))
        elif state['status'] == 'failed':
            problems.append(state)
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUT / 'completed_constructions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = []
    for arm in ARMS:
        group = [row for row in rows if row['arm'] == arm]
        if len(group) == 9:
            groups.append(dict(arm=arm, memories=9,
                f1=statistics.mean(r['f1'] for r in group),
                f1_sd=statistics.stdev(r['f1'] for r in group),
                accuracy=statistics.mean(r['accuracy'] for r in group),
                accuracy_sd=statistics.stdev(r['accuracy'] for r in group)))
    state = dict(counts=counts, total=99, complete_methods=groups, failures=problems,
                 free_GiB=shutil.disk_usage(ROOT).free/2**30)
    save(OUT / 'status.json', state)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'worker', 'launch', 'status'])
    parser.add_argument('--lane', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.action == 'launch':
        assert args.lane is not None
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / f'lane{args.lane}_worker.log').open('x') as log:
            process = subprocess.Popen([PYTHON, '-u', __file__, 'worker', '--lane', str(args.lane)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps(dict(pid=process.pid, lane=args.lane)))
    elif args.action == 'worker':
        assert args.lane is not None
        worker(args.lane)
    else:
        globals()[args.action]()
