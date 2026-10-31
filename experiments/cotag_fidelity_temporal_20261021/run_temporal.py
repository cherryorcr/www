"""Year-based TAG condensation, with explicitly stage-limited inference graphs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
import traceback

import common as C
from common import R, E, torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from temporal_protocol import BOUNDS, arrival_stages, validate_delta, snapshot_ids, cohort_masks

ARMS = ('uq_dm_edgefree', 'cat_cctp', 'gecc_hard')


def load(device):
    path = Path(R.L.U.DATA_CFG['arxiv']['path'])
    raw = torch.load(path, map_location='cpu', weights_only=False)
    years = torch.as_tensor(raw['node_year']).flatten().long()
    arrival = arrival_stages(years)
    provenance = dict(path=str(path), sha256=C.sha(path),
                      feature_protocol=raw.get('feature_protocol'),
                      years_hash=R.V.hash_tensor(years), boundary_years=BOUNDS)
    data = R.L.U.load_unified('arxiv', str(path), C.MINILM, device,
                             n_stages=5, split_mode='base_incremental', report_dev_mod=3)
    if not torch.equal(raw['original_node_ids'], data['original_ids']):
        raise AssertionError('Year-to-node mapping differs')
    del raw
    data['text_emb'] = torch.load(C.ROOT / 'results/public_baselines_text_cache_20261001/arxiv/text_emb_pool_only.pt',
                                 map_location='cpu', weights_only=False).float()
    if bool(data['text_emb'][~data['pool_mask']].any()):
        raise AssertionError('Cache contains held-out text')
    data['prompt_emb'] = data['encoder'].encode(data['prompts'])
    ids = [torch.where(data['pool_mask'] & (arrival == s))[0] for s in range(5)]
    groups = [torch.unique(data['y'][idx]).tolist() for idx in ids]
    budgets, floors = R.V.choose_budgets({}, list(map(len, ids)), groups, 'fixed_rate', .05)
    stats = []
    for stage, idx in enumerate(ids):
        visible = snapshot_ids(arrival, stage, 'arrived_snapshot')
        edges = R.L.induced_edges(data['edge_index'], visible, len(arrival))
        delta_edges = R.L.induced_edges(data['edge_index'], idx, len(arrival))
        stats.append(dict(stage=stage, year_end=BOUNDS[stage],
            arrival_nodes=int((arrival == stage).sum()), pool_nodes=len(idx),
            current_class_counts=torch.bincount(data['y'][idx], minlength=40).tolist(),
            arrived_nodes=len(visible), arrived_edges=edges.shape[1], delta_pool_edges=delta_edges.shape[1],
            report_rows=int((data['report_dev_mask'] & (arrival == stage)).sum()),
            test_rows=int((data['test_mask'] & (arrival == stage)).sum())))
    return data, arrival, ids, budgets, floors, provenance, stats


def evaluate(memory, data, arrival, stage, device):
    prompt = data['prompt_emb'].float().to(device)
    reports = {'report_dev': data['report_dev_mask'], 'historical_test': data['test_mask']}
    output = {mode: {split: [] for split in reports}
              for mode in ('arrived_snapshot', 'current_snapshot')}
    resources = []
    for seed in C.STUDENT_SEEDS:
        R.V.sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model = R.P.train_prompt_student(memory, prompt, seed, device, 2048, lr=.001, batch=24)
        R.V.sync(device)
        cost = dict(seed=seed, train_seconds=time.perf_counter()-start,
                    train_peak_mb=torch.cuda.max_memory_allocated(device)/2**20)
        for mode, splits in output.items():
            idx = snapshot_ids(arrival, stage, mode)
            if bool((arrival[idx] > stage).any()):
                raise AssertionError('Future node in inference snapshot')
            edge = R.L.induced_edges(data['edge_index'], idx, len(arrival))
            adj = R.P.normalized_adjacency(edge, len(idx), device)
            with torch.no_grad():
                pred = model.logits(data['x'][idx].to(device), adj, prompt).argmax(1).cpu()
            for split, mask in reports.items():
                metrics = dict(seed=seed)
                for cohort, allowed in cohort_masks(mask, arrival, stage).items():
                    local = allowed[idx]
                    count = int(local.sum())
                    if count == 0:
                        metrics[cohort] = None
                        continue
                    yy, pp = data['y'][idx][local].numpy(), pred[local].numpy()
                    metrics[cohort] = dict(nodes=count,
                        f1=float(100*f1_score(yy, pp, labels=list(range(40)), average='macro', zero_division=0)),
                        accuracy=float(100*accuracy_score(yy, pp)))
                splits[split].append(metrics)
            del adj, pred
        resources.append(cost)
        del model
        torch.cuda.empty_cache()
    for mode, splits in output.items():
        for split, items in list(splits.items()):
            summary = dict(per_student=items)
            for cohort in ('cumulative', 'current', 'historical'):
                values = [x[cohort] for x in items if x[cohort] is not None]
                summary[cohort] = None if not values else dict(nodes=values[0]['nodes'], **{
                    key: float(np.mean([r[key] for r in values])) for key in ('f1', 'accuracy')}, **{
                    key+'_sd': float(np.std([r[key] for r in values], ddof=1))
                    for key in ('f1', 'accuracy')})
            splits[split] = summary
    return dict(prediction=output, training_resources=resources)


def run(args):
    C.disk_check()
    out = C.RESULTS / 'temporal' / f'{args.arm}_c{args.seed}'
    if (out/'results.json').exists() and C.read(out/'results.json')['status'] == 'complete':
        C.emit(event='skip_complete', path=str(out))
        return
    data, arrival, ids, budgets, floors, source, stats = load(args.device)
    cfg = R.q_config('uq_dm_edgefree', args.seed, 30, 4096)
    protocol = dict(name='arxiv-publication-year-base4-v1', dataset='arxiv', arm=args.arm,
        construction_seed=args.seed, order_seed=None, student_seeds=C.STUDENT_SEEDS,
        student_steps=2048, student_batch=24, memory_rate=.05, budgets=budgets,
        class_floor_adjustments=floors, q_config=asdict(cfg), source=source, stages=stats,
        access='current-stage legal training nodes and own frozen synthetic prefix',
        inference_modes=['arrived_snapshot', 'current_snapshot'],
        edge_time_assumption='An observed final-graph edge becomes available when both endpoints have arrived; no edge-event timestamps are available.',
        stream='publication-year arrivals with recurring classes; not disjoint new-class tasks',
        prompt_taxonomy='fixed known 40-category metadata; no future node labels in constructors',
        cohort_definition='arrival cohorts, not old/new classes',
        full_cpu_source='offline harness holds data for slicing; not a physical deletion implementation',
        future_raw_on_gpu=False, method_selection_from_test=False,
        evaluation='predeclared chronological extension on previously inspected masks; not new blind data',
        stage_id_hashes=[R.V.hash_tensor(idx) for idx in ids],
        pool_mask_hash=R.V.hash_tensor(data['pool_mask']),
        report_mask_hash=R.V.hash_tensor(data['report_dev_mask']),
        test_mask_hash=R.V.hash_tensor(data['test_mask']),
        source_hashes=C.source_hashes(), runner_hash=C.sha(__file__),
        protocol_code_hash=C.sha(Path(__file__).with_name('temporal_protocol.py')))
    result = C.start_result(out/'results.json', protocol)
    parent = None
    try:
        for stage, idx in enumerate(ids):
            stage_dir = out / f's{stage}'
            memory_path = stage_dir/'memory.pt.gz'
            audit_path = stage_dir/'construction.json'
            saved = next((r for r in result['rows'] if r['stage'] == stage), None)
            if memory_path.exists() and audit_path.exists():
                memory = C.load_memory(memory_path)
                construction = C.read(audit_path)
                R.V.validate_memory(memory, parent, budgets[stage])
                if R.V.memory_hash(memory) != construction['memory_hash']:
                    raise AssertionError('Checkpoint hash changed')
            else:
                legal = validate_delta(idx, arrival, data['pool_mask'], stage)
                delta = R.L.delta_view(data, idx, stage)
                labels = R.M.row_labels(R.M.ratio_allocation(delta['y'],
                    budgets[stage]-(0 if parent is None else len(parent['x']))))
                R.V.sync(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                start = time.perf_counter()
                memory, audit = E.construct(args.arm, delta, parent, labels, cfg,
                    data['encoder'], data['prompt_emb'], args.seed, args.device)
                R.V.sync(args.device)
                checks = R.V.validate_memory(memory, parent, budgets[stage])
                if 'q_row_sum_max_error' in audit:
                    checks.update(R.assignment_checks(audit))
                    if not checks['sinkhorn_feasible']:
                        C.save(stage_dir/'failed_assignment.json', audit)
                        raise AssertionError('Assignment feasibility failed')
                construction = dict(stage=stage, access=legal, checks=checks,
                    seconds=time.perf_counter()-start,
                    peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20,
                    memory_rows=len(memory['x']), memory_hash=R.V.memory_hash(memory),
                    audit=R.compact_audit(audit))
                C.archive(memory_path, memory)
                C.save(audit_path, construction)
                del delta
                C.emit(event='temporal_constructed', arm=args.arm, seed=args.seed,
                       stage=stage, seconds=construction['seconds'])
            if saved is None:
                scores = evaluate(memory, data, arrival, stage, args.device)
                result['rows'].append(dict(stage=stage, construction=construction, scores=scores))
                C.save(out/'results.json', result)
                C.emit(event='temporal_evaluated', arm=args.arm, seed=args.seed, stage=stage,
                       scores=scores['prediction']['arrived_snapshot']['report_dev']['cumulative'])
            elif saved['construction']['memory_hash'] != construction['memory_hash']:
                raise AssertionError('Metrics and checkpoint mismatch')
            parent = memory
            torch.cuda.empty_cache()
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        C.save(out/'results.json', result)
        raise
    C.save(out/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, default=271006)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
