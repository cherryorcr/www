"""Larger-base ArXiv comparison using the registered main-table adapters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


ROOT = Path('/home/pc/research/TAG_condensation')
EXP = ROOT / 'experiments/cotag_arxiv_base4_20261026'
OUT = ROOT / 'results/cotag_arxiv_base4_20261026'
PREVIOUS = ROOT / 'experiments/cotag_arxiv_five_stage_20261024'
LOCKS = ROOT / 'results/cotag_baseline_repetitions_20261024'
PYTHON = '/home/pc/miniconda3/envs/tagsam/bin/python'
CLASS_COUNTS = [16, 6, 6, 6, 6]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def previous_module():
    spec = importlib.util.spec_from_file_location('equal_stage_queue', PREVIOUS / 'queue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_checks(checks):
    for key in ['budget_ok', 'prefix_ok', 'finite_ok']:
        assert checks[key] is True, (key, checks)
    # A warning flag has the opposite polarity to a successful invariant.
    if 'numerical_warning' in checks:
        assert checks['numerical_warning'] is False, checks
    if 'sinkhorn_feasible' in checks:
        assert checks['sinkhorn_feasible'] is True, checks
        tolerance = checks['sinkhorn_solver_tolerance']
        allowance = checks['full_matrix_rounding_allowance']
        assert math.isfinite(tolerance) and tolerance > 0, checks
        assert math.isfinite(allowance) and allowance >= 0, checks
        for key, limit in [('sinkhorn_block_error', tolerance),
                           ('sinkhorn_full_error', tolerance + allowance)]:
            assert math.isfinite(checks[key]) and 0 <= checks[key] <= limit, checks


def validate(path, job, class_counts=CLASS_COUNTS):
    data = previous_module().previous_validator().validate(path, job)
    protocol = data['protocol']
    assert [len(group) for group in protocol['groups']] == class_counts
    assert sorted(c for group in protocol['groups'] for c in group) == list(range(40))
    assert sum(protocol['stage_raw_rows']) == 118539
    assert protocol['budgets'][-1] == 5927
    for row in data['rows']:
        validate_checks(row['checks'])
    return data


def register():
    target = EXP / 'registration.json'
    assert not target.exists(), 'Registration is immutable.'
    previous = json.loads((PREVIOUS / 'registration.json').read_text())
    reference = dict(name='arxiv-base4-main-comparison-20261026', dataset='arxiv',
        groups=[list(range(a, b)) for a, b in [(0, 16), (16, 22), (22, 28), (28, 34), (34, 40)]],
        report_dev_mod=3, budgets=[], base_fraction=.4,
        budget_note='Recompute cumulative 5% budgets from each order legal pool.')
    save(EXP / 'reference.json', reference)
    jobs = []
    for original in previous['jobs']:
        job = dict(original, output=str(OUT / 'runs' / original['id']))
        command = list(original['command'])
        command[command.index('--reference') + 1] = str(EXP / 'reference.json')
        command[command.index('--output') + 1] = job['output']
        job['command'] = command
        jobs.append(job)
    sources = dict(previous['source_files'])
    assert all(sha(path) == digest for path, digest in sources.items()), 'Frozen source changed.'
    sources.update({str(path): sha(path) for path in [EXP / 'queue.py', EXP / 'reference.json']})
    record = dict(previous, created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        purpose='Complete the main comparison with a 40% initial base followed by four increments.',
        classes_per_stage=CLASS_COUNTS, base_fraction=.4, jobs=jobs, source_files=sources,
        supersedes_schedule=str(PREVIOUS / 'registration.json'),
        superseded_results='Preserved as equal-stage evidence; never reused as larger-base scores.',
        split_basis='Existing derive_stage_classes base_incremental default base_fraction=0.4.',
        validator_correction='Check numerical_warning=False as success, independently of positive invariant flags.',
        q_numerical_policy='Frozen solver, tolerances, and iteration cap unchanged. Stop on real failure.')
    save(target, record)
    print(json.dumps(dict(registered=len(jobs), classes_per_stage=CLASS_COUNTS)))


def audit_previous():
    old = json.loads((PREVIOUS / 'registration.json').read_text())
    rows = []
    for job in old['jobs']:
        receipt = Path(job['output']) / 'queue_receipt.json'
        if not receipt.exists():
            continue
        state = json.loads(receipt.read_text())
        if state['status'] not in ['complete', 'failed']:
            continue
        item = dict(job=job['id'], original_status=state['status'])
        try:
            path = Path(job['output']) / 'results.json'
            validate(path, job, [8] * 5)
            item.update(valid=True, result_sha256=sha(path))
        except Exception as error:
            item.update(valid=False, error=str(error))
        rows.append(item)
    report = dict(note='Read-only revalidation; original receipts, logs, and results are unchanged.', rows=rows)
    save(EXP / 'previous_validation_audit.json', report)
    print(json.dumps(report, indent=2))


def worker(lane):
    import fcntl

    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / f'lane{lane}_status.json'
    with (OUT / f'worker{lane}.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (LOCKS / f'lane{lane}.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record = json.loads((EXP / 'registration.json').read_text())
            assert all(sha(path) == digest for path, digest in record['source_files'].items())
            for job in record['jobs']:
                output = Path(job['output'])
                receipt = output / 'queue_receipt.json'
                with (OUT / 'claims.lock').open('a') as claim_lock:
                    fcntl.flock(claim_lock, fcntl.LOCK_EX)
                    if receipt.exists():
                        continue
                    assert not output.exists() or not any(output.iterdir()), output
                    if shutil.disk_usage(ROOT).free < 8 * 2**30:
                        save(state_path, dict(status='blocked_disk', job=job['id']))
                        return
                    state = dict(status='running', job=job['id'], lane=lane,
                        pid=os.getpid(), start=time.time(), command=job['command'])
                    save(receipt, state)
                save(state_path, state)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=job['pythonpath'])
                try:
                    with (output / 'run.log').open('x') as log:
                        process = subprocess.run(job['command'], env=env, cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, timeout=6*3600)
                    assert process.returncode == 0, f'runner exit {process.returncode}; see run.log'
                    result = output / 'results.json'
                    validate(result, job)
                    state.update(status='complete', result_sha256=sha(result))
                except Exception as error:
                    state.update(status='failed', error=str(error))
                state['end'] = time.time()
                save(receipt, state)
                save(state_path, state)
                print(json.dumps(state), flush=True)
                if state['status'] == 'failed':
                    return
            save(state_path, dict(status='queue_exhausted', time=time.time()))


def launch(lane):
    assert lane in [0, 1]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'lane{lane}_launch_{time.time_ns()}.log').open('x') as log:
        process = subprocess.Popen([PYTHON, '-u', __file__, 'worker', '--lane', str(lane)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    print(json.dumps(dict(pid=process.pid, lane=lane)))


def status():
    record = json.loads((EXP / 'registration.json').read_text())
    counts, rows, failures = {}, [], []
    for job in record['jobs']:
        receipt = Path(job['output']) / 'queue_receipt.json'
        state = json.loads(receipt.read_text()) if receipt.exists() else dict(status='pending')
        counts[state['status']] = counts.get(state['status'], 0) + 1
        if state['status'] == 'complete':
            path = Path(job['output']) / 'results.json'
            data = validate(path, job)
            score = data['rows'][-1]['scores']['sealed_test']
            rows.append(dict(arm=job['arm'], order=job['order'], seed=job['seed'],
                f1=score['final_f1'], accuracy=score['accuracy'], source=str(path), sha256=sha(path)))
        elif state['status'] == 'failed':
            failures.append(state)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'completed_constructions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'order', 'seed', 'f1', 'accuracy', 'source', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    expected = {(order, seed) for order in record['orders'] for seed in record['construction_seeds']}
    for arm in dict.fromkeys(job['arm'] for job in record['jobs']):
        group = [row for row in rows if row['arm'] == arm]
        assert len(group) == len({(r['order'], r['seed']) for r in group})
        if {(r['order'], r['seed']) for r in group} == expected:
            groups.append(dict(arm=arm, memories=9,
                f1=statistics.mean(r['f1'] for r in group),
                f1_sd=statistics.stdev(r['f1'] for r in group),
                accuracy=statistics.mean(r['accuracy'] for r in group),
                accuracy_sd=statistics.stdev(r['accuracy'] for r in group)))
    report = dict(counts=counts, total=len(record['jobs']), classes_per_stage=CLASS_COUNTS,
        complete_methods=groups, failures=failures, free_GiB=shutil.disk_usage(ROOT).free/2**30)
    save(OUT / 'status.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'audit_previous', 'worker', 'launch', 'status'])
    parser.add_argument('--lane', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.action in ['launch', 'worker']:
        globals()[args.action](args.lane)
    else:
        globals()[args.action]()
