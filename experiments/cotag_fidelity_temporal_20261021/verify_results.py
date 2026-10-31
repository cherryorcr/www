"""Independent checkpoint and aggregation checks without importing constructors."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory_hash(memory):
    result = hashlib.sha256(json.dumps(memory['texts'], ensure_ascii=True).encode())
    for key in ('x', 'text_emb', 'labels', 'mass', 'edge_index'):
        result.update(tensor_hash(memory[key]).encode())
    return result.hexdigest()


def near(left, right):
    if not math.isfinite(float(left)) or not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f'Aggregation mismatch: {left} != {right}')


def check_scores(scores, temporal):
    checked = 0
    if temporal:
        for mode, splits in scores['prediction'].items():
            for aggregate in splits.values():
                rows = aggregate['per_student']
                assert sorted(row['seed'] for row in rows) == [150, 151, 152]
                if mode == 'current_snapshot':
                    assert aggregate['historical'] is None
                    assert aggregate['cumulative'] == aggregate['current']
                for cohort in ('current', 'historical', 'cumulative'):
                    values = [row[cohort] for row in rows if row[cohort] is not None]
                    if not values:
                        assert aggregate[cohort] is None
                        continue
                    assert len(values) == 3
                    assert len({v['nodes'] for v in values}) == 1
                    for metric in ('f1', 'accuracy'):
                        numbers = [v[metric] for v in values]
                        near(aggregate[cohort][metric], statistics.mean(numbers))
                        near(aggregate[cohort][metric+'_sd'], statistics.stdev(numbers))
                        checked += 2
    else:
        for aggregate in scores.values():
            rows = aggregate['per_student']
            assert sorted(row['seed'] for row in rows) == [150, 151, 152]
            for metric in ('final_f1', 'accuracy', 'old_f1', 'new_f1'):
                numbers = [row[metric] for row in rows if row[metric] is not None]
                if numbers:
                    near(aggregate[metric], statistics.mean(numbers))
                    if len(numbers) > 1:
                        near(aggregate[metric+'_std'], statistics.stdev(numbers))
                    checked += 2
    return checked


def expected_temporal(source):
    raw = torch.load(source, map_location='cpu', weights_only=False)
    years = torch.as_tensor(raw['node_year']).flatten()
    original = raw['original_node_ids'].long()
    pool = ~torch.isin(original % 10, torch.tensor([1, 2, 3]))
    ids = []
    low = -math.inf
    for high in (2015, 2016, 2017, 2018, 2020):
        ids.append(torch.where(pool & (years > low) & (years <= high))[0])
        low = high
    assert sum(map(len, ids)) == int(pool.sum())
    return dict(stage_id_hashes=[tensor_hash(idx) for idx in ids],
                pool_mask_hash=tensor_hash(pool),
                report_mask_hash=tensor_hash(original % 10 == 3),
                test_mask_hash=tensor_hash(original % 10 == 2),
                budgets=[round(.05*sum(map(len, ids[:s+1]))) for s in range(5)])


def verify_one(path, expected):
    result = read(path)
    protocol = result['protocol']
    temporal = 'arxiv-publication-year' in protocol['name']
    assert read(path.with_name('protocol.json')) == protocol
    assert protocol['student_seeds'] == [150, 151, 152]
    assert protocol['student_steps'] == 2048 and protocol['student_batch'] == 24
    assert protocol['memory_rate'] == .05
    assert protocol['method_selection_from_test'] is False
    if temporal:
        for key, value in expected.items():
            assert protocol[key] == value, key
        if 'numerical_repair_hash' in protocol:
            assert protocol['q_config']['sinkhorn_max_steps'] == 16384
    nstage = len(protocol['budgets'])
    stages = [row['stage'] for row in result['rows']]
    assert stages == list(range(len(stages))), 'Noncontiguous or duplicate completed stages'
    if result['status'] == 'complete':
        assert stages == list(range(nstage))
    if result['status'] == 'failed':
        assert result.get('error') and result.get('traceback')
    parent, count, metrics = None, 0, 0
    for row in result['rows']:
        stage = row['stage']
        directory = path.parent/f's{stage}'
        audit = read(directory/'construction.json')
        assert audit == row['construction']
        with gzip.open(directory/'memory.pt.gz', 'rb') as stream:
            memory = torch.load(stream, map_location='cpu', weights_only=False)
        budget = protocol['budgets'][stage]
        for key in ('x', 'text_emb', 'labels', 'mass', 'texts'):
            assert len(memory[key]) == budget, (stage, key, budget)
        assert memory['edge_index'].numel() == 0
        for key in ('x', 'text_emb', 'mass'):
            assert bool(torch.isfinite(memory[key]).all())
        assert memory_hash(memory) == audit['memory_hash']
        if parent is not None:
            keep = len(parent['x'])
            for key in ('x', 'text_emb', 'labels', 'mass'):
                assert torch.equal(parent[key], memory[key][:keep]), (stage, key)
            assert parent['texts'] == memory['texts'][:keep]
        if temporal and 'numerical_repair_hash' in protocol:
            assert audit['audit']['precision_verified_calls'] > 0
        metrics += check_scores(row['scores'], temporal)
        parent = memory
        count += 1
    return dict(status=result['status'], verified_stages=count, scalar_aggregation_checks=metrics,
                protocol_equal=True, budgets_equal=True, frozen_prefix_equal=True,
                checkpoints_hash_equal=True, finite_memory=True,
                temporal_source_masks_equal=True if temporal else None, stage_count=nstage)


def verify_native_readout(path):
    result = read(path)
    assert result['status'] == 'complete'
    assert sorted(row['seed'] for row in result['rows']) == [150, 151, 152]
    source = path.parent.parent/'results.json'
    original = read(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['protocol']['source_result_hash']
    assert original['rows'][-1]['construction']['memory_hash'] == result['protocol']['source_memory_hash']
    count = 0
    for row in result['rows']:
        assert [v['epoch'] for v in row['losses']] == list(range(1, 16))
    for split, summary in result['summary'].items():
        for metric in ('macro_f1', 'accuracy'):
            values = [row['scores'][split][metric] for row in result['rows']]
            near(summary[metric], statistics.mean(values))
            near(summary[metric+'_sd'], statistics.stdev(values))
            count += 2
    teacher = result['teacher_check']
    assert teacher['training']['updates'] == original['rows'][-1]['construction']['teacher']['updates']
    return dict(status='complete', scalar_aggregation_checks=count,
                source_hash_equal=True, same_constructed_memory=True,
                fixed_fifteen_epochs=True,
                teacher_loss_max_abs_difference=teacher['original_loss_max_abs_difference'])


def run(root, source):
    expected = expected_temporal(source)
    output = []
    for path in sorted(root.rglob('results.json')):
        relative = path.relative_to(root)
        if 'native_student_diagnostic' in relative.parts:
            try:
                output.append(dict(path=str(relative), passed=True, **verify_native_readout(path)))
            except Exception as error:
                output.append(dict(path=str(relative), passed=False, error=repr(error)))
            continue
        if 'precision_repair' in relative.parts and (root/'precision_repair/INVALID_ATTEMPT.json').exists():
            output.append(dict(path=str(relative), excluded='invalid intervention, retained as failure evidence'))
            continue
        try:
            output.append(dict(path=str(relative), passed=True, **verify_one(path, expected)))
        except Exception as error:
            output.append(dict(path=str(relative), passed=False, error=repr(error)))
    report = dict(results=output, passed=all(r.get('passed', True) for r in output),
                  scope='Saved completed stages only. Prediction metrics checked against stored per-student scores, not recomputed from logits.',
                  temporal_expected=expected)
    target = root/'analysis/verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021'))
    parser.add_argument('--source', type=Path, default=Path('/home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm.pt'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.root, args.source)
