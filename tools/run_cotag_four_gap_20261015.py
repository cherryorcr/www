"""Registered, additive evidence; all construction imports come from the frozen snapshot."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path('/home/pc/research/TAG_condensation')
FROZEN = ROOT / 'experiments/uq_final_evidence_20261012/code'
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(ROOT / 'code/v2_llm_fusion'))
import run_uq_final_confirmation_20261012 as E

R = E.R
SEEDS = [150, 151, 152]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def emit(**value):
    print(json.dumps(value), flush=True)


def read_full_json(path):
    """Read the envelope without the repository loader's protocol unwrapping."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(args):
    ref = R.L.read(args.reference)
    groups = R.V.choose_groups(ref, args.order_seed)
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                         dict(groups=groups, report_dev_mod=ref.get('report_dev_mod', 3)))
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(ref, sizes, groups, 'fixed_rate', .05)
    protocol = dict(name='cotag-four-gap-evidence-20261015-v1', task=args.task,
        arm=args.arm, dataset=args.dataset, order_seed=args.order_seed,
        construction_seed=args.construction_seed, groups=groups, stage_raw_rows=sizes,
        budgets=budgets, class_floor_adjustments=floors, memory_rate=.05,
        actual_final_rate=budgets[-1] / sum(sizes),
        student=R.P.model_config(), student_steps=2048, student_batch=24,
        student_seeds=SEEDS, method_selection_from_test=False,
        evaluation='report-dev and previously inspected test; post-hoc evidence, not a new blind test',
        official_reproduction=False,
        source_hashes={Path(m.__file__).name: E.file_hash(m.__file__)
                       for m in (R, R.L, R.M, R.P, R.UQ, R.V, E)},
        runner_hash=E.file_hash(__file__),
        pool_mask_hash=R.V.hash_tensor(data['pool_mask']),
        report_mask_hash=R.V.hash_tensor(data['report_dev_mask']),
        test_mask_hash=R.V.hash_tensor(data['test_mask']),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids])
    primary = read_full_json(ROOT / 'results/uq_final_evidence_20261012/confirmation' /
                             f'{args.dataset}_o{args.order_seed}_c{args.construction_seed}' / 'results.json')
    keys = ('groups', 'stage_raw_rows', 'budgets', 'student', 'student_steps',
            'student_batch', 'student_seeds', 'pool_mask_hash', 'report_mask_hash',
            'test_mask_hash', 'stage_id_hashes')
    checks = {key: protocol[key] == primary['protocol'][key] for key in keys}
    if not all(checks.values()):
        raise AssertionError(dict(protocol_mismatch=checks))
    protocol['primary_protocol_checks'] = checks
    return data, ids, protocol


def new_result(args, protocol):
    path = args.output / 'results.json'
    if path.exists():
        old = read_full_json(path)
        if old['protocol'] == protocol and old['status'] == 'complete':
            emit(event='skip_verified_complete', output=str(path))
            return None
        raise RuntimeError('Preserve partial or different output; select a fresh directory')
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise RuntimeError('Disk floor: less than 8 GiB available')
    save(args.output / 'protocol.json', protocol)
    result = dict(protocol=protocol, rows=[], status='running')
    save(path, result)
    return result


def text_factorial(args):
    data, ids, protocol = load(args)
    base = R.q_config('uq_dm_edgefree', args.construction_seed, 30, 4096)
    cfg = (replace(base, text_weight=0.) if args.arm == 'no_text_init'
           else replace(base, pair_weight=0.))
    changes = {k: [asdict(base)[k], asdict(cfg)[k]] for k in asdict(base)
               if asdict(base)[k] != asdict(cfg)[k]}
    expected = 'text_weight' if args.arm == 'no_text_init' else 'pair_weight'
    if set(changes) != {expected}:
        raise AssertionError(changes)
    protocol.update(q_config=asdict(cfg), changes=changes,
        access='current-delta-only; own frozen prefix; no historical raw reads')
    result = new_result(args, protocol)
    if result is None:
        return
    parent = None
    for stage, raw_ids in enumerate(ids):
        delta = R.L.delta_view(data, raw_ids, stage)
        new_n = protocol['budgets'][stage] - (0 if parent is None else len(parent['x']))
        labels = R.M.row_labels(R.M.ratio_allocation(delta['y'], new_n))
        R.V.sync(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        start = time.perf_counter()
        block, audit = R.UQ.optimize_stage(delta, parent, labels, data['encoder'],
            data['prompt_emb'], cfg, args.device, use_boundary=False, use_topology=False)
        R.V.sync(args.device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(args.device) / 2**20
        memory = R.L.append_edge_free(parent, block)
        checks = R.V.validate_memory(memory, parent, protocol['budgets'][stage])
        checks.update(R.assignment_checks(audit))
        if not checks.get('sinkhorn_feasible') or memory['edge_index'].numel():
            raise AssertionError(checks)
        scores = E.evaluate(memory, data, stage, SEEDS, args.device)
        result['rows'].append(dict(stage=stage, arm=args.arm,
            raw_rows=len(raw_ids), new_rows=new_n, memory_rows=len(memory['x']),
            memory_hash=R.V.memory_hash(memory), checks=checks,
            construction_seconds=elapsed, construction_peak_mb=peak,
            scores=scores, score=scores['report_dev'], audit=R.compact_audit(audit)))
        save(args.output / 'results.json', result)
        emit(event='evaluated', dataset=args.dataset, arm=args.arm, stage=stage,
             report_f1=scores['report_dev']['final_f1'])
        parent = memory
        torch.cuda.empty_cache()
    result['status'] = 'complete'
    save(args.output / 'results.json', result)


def induced_edges(edges, ids, n):
    mapping = torch.full((n,), -1, dtype=torch.long)
    mapping[ids] = torch.arange(len(ids))
    edges = edges.long().cpu()
    keep = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
    return mapping[edges[:, keep]]


def sampling_coverage(n, seeds, steps=2048, batch=24):
    rows = []
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed * 7 + 1)
        seen = torch.zeros(n, dtype=torch.bool)
        for _ in range(steps):
            seen[torch.randperm(n, generator=gen)[:min(batch, n)]] = True
        rows.append(dict(seed=seed, unique_rows=int(seen.sum()), fraction=float(seen.float().mean()),
                         draws=steps * min(batch, n)))
    return rows


def full_history(args):
    data, stages, protocol = load(args)
    ids = torch.sort(torch.cat(stages)).values
    stage = len(stages) - 1
    texts = [R.UQ._append_text(data['texts'][int(i)], 320) for i in ids]
    emb = torch.as_tensor(data['encoder'].encode(texts)).float().cpu()
    memory = dict(x=data['x'][ids].float().cpu(), text_emb=emb,
        labels=data['y'][ids].long().cpu(), texts=texts,
        mass=torch.ones(len(ids)), depth=torch.zeros(len(ids), dtype=torch.long),
        stage=torch.full((len(ids),), stage, dtype=torch.long),
        edge_index=(torch.empty((2, 0), dtype=torch.long) if args.arm == 'full_ef'
                    else induced_edges(data['edge_index'], ids, len(data['x']))))
    protocol.update(reference='full legal training pool; historical raw access permitted only for this reference',
        construction_seed_relevant=False, evaluated_stage=stage,
        text_preprocess='same whitespace cleaning and 320 character cap as CoTAG-Q',
        topology=('self loops only' if args.arm == 'full_ef' else 'induced edges within full legal pool'),
        reference_rows=len(ids), reference_edges=memory['edge_index'].shape[1],
        reference_id_hash=R.V.hash_tensor(ids),
        memory_hash=R.V.memory_hash(memory), training_budget='fixed 2048 updates, 24 sampled loss rows',
        not_an_optimized_performance_upper_bound=True)
    result = new_result(args, protocol)
    if result is None:
        return
    result['sampling_coverage'] = sampling_coverage(len(ids), SEEDS)
    save(args.output / 'results.json', result)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    scores = E.evaluate(memory, data, stage, SEEDS, args.device)
    R.V.sync(args.device)
    result['rows'].append(dict(stage=stage, arm=args.arm, memory_rows=len(ids),
        scores=scores, score=scores['report_dev'], student_seconds=time.perf_counter()-start,
        student_peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20))
    result['status'] = 'complete'
    save(args.output / 'results.json', result)
    emit(event='full_history_complete', dataset=args.dataset, arm=args.arm,
         report_f1=scores['report_dev']['final_f1'])


@contextlib.contextmanager
def sparse_edgefree():
    original = R.UQ._hard_adjacency
    def sparse(edges, parent, n, device):
        old = (torch.empty((2, 0), dtype=torch.long) if parent is None else
               parent.get('edge_index', torch.empty((2, 0), dtype=torch.long)))
        if edges.numel() or old.numel():
            raise AssertionError('Sparse identity intervention is edge-free only')
        return R.P.normalized_adjacency(edges, n, str(device))
    R.UQ._hard_adjacency = sparse
    try:
        yield
    finally:
        R.UQ._hard_adjacency = original


def equivalence(device):
    torch.manual_seed(20261015)
    logits = torch.randn(16, 4, device=device, requires_grad=True)
    x = torch.randn(16, 8, device=device)
    target = torch.randn(4, 6, device=device)
    model = R.P.GraphTextPromptStudent(8, hidden=12, out=6, text_dim=6).to(device)
    adj = R.P.normalized_adjacency(torch.empty((2, 0), dtype=torch.long), 4, device)
    records = []
    for a in (adj.to_dense(), adj):
        q = logits.softmax(1)
        pooled = q.T @ x / q.sum(0)[:, None]
        encoded = R.UQ._encode_graph(model, pooled, a)
        loss = (encoded-target).square().mean()
        grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True,
                                    allow_unused=True)
        active = [g for g in grads if g is not None]
        gm = sum(g.square().sum() for g in active)
        second = torch.autograd.grad(gm, logits)[0]
        records.append((encoded.detach(), torch.cat([g.detach().flatten() for g in active]), second))
    errors = {key: float((records[0][i]-records[1][i]).abs().max())
              for i,key in enumerate(('forward_max_error', 'first_gradient_max_error', 'second_gradient_max_error'))}
    if max(errors.values()) > 1e-5:
        raise AssertionError(errors)
    return dict(passed=True, **errors)


def cost(args):
    data, stages, protocol = load(args)
    protocol.update(stage=0, benchmark='isolated GPU; same GPU order dense then sparse; one run each',
        q_config=asdict(R.q_config('uq_dm_edgefree', args.construction_seed, 30, 4096)),
        change='replace EF dense identity with sparse identity; objective and solver unchanged')
    result = new_result(args, protocol)
    if result is None:
        return
    result['operator_equivalence'] = equivalence(args.device)
    delta = R.L.delta_view(data, stages[0], 0)
    labels = R.M.row_labels(R.M.ratio_allocation(delta['y'], protocol['budgets'][0]))
    cfg = R.q_config('uq_dm_edgefree', args.construction_seed, 30, 4096)
    blocks = []
    for variant in ('dense_identity', 'sparse_identity'):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        R.V.sync(args.device)
        start = time.perf_counter()
        with sparse_edgefree() if variant == 'sparse_identity' else contextlib.nullcontext():
            block, audit = R.UQ.optimize_stage(delta, None, labels, data['encoder'], data['prompt_emb'],
                cfg, args.device, use_boundary=False, use_topology=False)
        R.V.sync(args.device)
        result['rows'].append(dict(arm=variant, stage=0,
            construction_seconds=time.perf_counter()-start,
            construction_peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20,
            construction_reserved_mb=torch.cuda.max_memory_reserved(args.device)/2**20,
            memory_hash=R.V.memory_hash(block), audit=R.compact_audit(audit)))
        blocks.append(block)
        save(args.output / 'results.json', result)
        emit(event='cost_variant_complete', dataset=args.dataset, **{k:v for k,v in result['rows'][-1].items() if k != 'audit'})
    a, b = blocks
    result['construction_equivalence'] = dict(x_max_error=float((a['x']-b['x']).abs().max()),
        x_relative_norm_error=float((a['x']-b['x']).norm()/a['x'].norm().clamp_min(1e-12)),
        text_emb_max_error=float((a['text_emb']-b['text_emb']).abs().max()),
        text_source_identity=a['text_source_ids'] == b['text_source_ids'],
        text_identity=a['texts'] == b['texts'])
    n, m = len(stages[0]), len(labels)
    valid = sum(int((delta['y'] == c).sum()) * int((labels == c).sum()) for c in torch.unique(labels))
    result['allocation_accounting'] = dict(raw_rows=n, slots=m, dense_entries=n*m,
        class_legal_entries=valid, illegal_fraction=1-valid/(n*m),
        dense_logits_fp32_mib=n*m*4/2**20,
        logits_plus_gradient_and_two_adam_states_mib=n*m*4*4/2**20,
        synthetic_dense_identity_mib=m*m*4/2**20,
        synthetic_sparse_identity_approx_mib=m*(2*8+4)/2**20,
        note='Tensor byte accounting excludes activations, Q copies, text encoder, allocator and proxy models')
    result['status'] = 'complete'
    save(args.output / 'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('text', 'fullhistory', 'cost'), required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--dataset', choices=('arxiv','sportsfit','bookhis'), required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--text-cache', type=Path, required=True)
    p.add_argument('--minilm', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--order-seed', type=int, default=9101)
    p.add_argument('--construction-seed', type=int, default=271006)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    {'text':text_factorial, 'fullhistory':full_history, 'cost':cost}[args.task](args)
