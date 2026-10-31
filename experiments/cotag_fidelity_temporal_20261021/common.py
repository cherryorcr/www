"""Immutable experiment inputs and checkpoint IO for the new evidence batch."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(os.environ.get('COTAG_ROOT', '/home/pc/research/TAG_condensation'))
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / 'experiments/uq_final_evidence_20261012/code'
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(ROOT / 'code/v2_llm_fusion'))
import run_uq_final_confirmation_20261012 as E
import torch

R = E.R
MINILM = '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip'
STUDENT_SEEDS = [150, 151, 152]
RESULTS = ROOT / 'results/cotag_fidelity_temporal_20261021'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def disk_check():
    free = shutil.disk_usage(ROOT).free
    if free < 8 * 2**30:
        raise RuntimeError('Preserve an 8 GiB free disk floor')
    return free / 2**30


def archive(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    disk_check()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wb', compresslevel=3) as stream:
        torch.save(value, stream)
    tmp.replace(path)


def load_memory(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def source_hashes():
    modules = (R, R.UQ, R.M, R.P, R.L, R.V, R.I, R.I.D, E)
    hashes = {Path(m.__file__).name: sha(m.__file__) for m in modules}
    hashes['common.py'] = sha(__file__)
    return hashes


def class_data(dataset, device):
    ref = R.L.read(ROOT / f'experiments/uq_final_evidence_20261012/{dataset}_reference.json')
    groups = R.V.choose_groups(ref, 9101)
    data = R.L.load_large(dataset, MINILM,
        ROOT / f'results/public_baselines_text_cache_20261001/{dataset}', device,
        dict(groups=groups, report_dev_mod=ref.get('report_dev_mod', 3)))
    ids = R.L.stage_ids(data)
    budgets, floors = R.V.choose_budgets(ref, list(map(len, ids)), groups, 'fixed_rate', .05)
    checks = dict(groups=groups, stage_raw_rows=list(map(len, ids)), budgets=budgets,
        class_floor_adjustments=floors, stage_id_hashes=[R.V.hash_tensor(x) for x in ids],
        pool_mask_hash=R.V.hash_tensor(data['pool_mask']),
        report_mask_hash=R.V.hash_tensor(data['report_dev_mask']),
        test_mask_hash=R.V.hash_tensor(data['test_mask']))
    primary = read(ROOT / ('results/uq_final_evidence_20261012/confirmation/'
                          f'{dataset}_o9101_c271006/results.json'))
    for key, value in checks.items():
        if primary['protocol'][key] != value:
            raise AssertionError(f'primary protocol differs: {key}')
    return data, ids, checks


def start_result(path, protocol):
    protocol = json.loads(json.dumps(protocol))
    if path.exists():
        old = read(path)
        if old['protocol'] != protocol:
            raise ValueError('Existing protocol differs; preserve output and use a new version')
        if old['status'] == 'failed':
            raise ValueError('Failed attempt is immutable; do not silently retry')
        return old
    result = dict(protocol=protocol, status='running', rows=[])
    save(path, result)
    save(path.parent / 'protocol.json', protocol)
    return result
