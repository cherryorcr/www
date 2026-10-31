"""Isolated training-only resource comparison; never selects on test scores."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/home/pc/research/TAG_condensation')
sys.path.insert(0, str(ROOT / 'tools'))
import run_four_priority_experiments_20261017 as A
import run_full_raw_prompt_student_20261011 as F


def cpu_only(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_only(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_only(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=A.DATASETS, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    out = ROOT / 'results/cotag_training_resources_20261019' / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('status') == 'complete':
            print(json.dumps({'event': 'reuse_complete', 'dataset': args.dataset}))
            return
        if saved.get('rows'):
            raise RuntimeError('Preserve incomplete audit with measured repeats; inspect before restarting.')
        path.rename(out / f'preflight_{time.time_ns()}.json')
    source = ROOT / 'results/uq_final_evidence_20261012/confirmation' / f'{args.dataset}_o9101_c271006'
    reference = A.R.L.read(source / 'results.json')
    groups = A.R.V.choose_groups(reference, 9101)
    cache = (ROOT / 'results/public_baselines_text_cache_20261001/bookhis' if args.dataset == 'bookhis'
             else ROOT / 'results/tgc_large_full_history_20260926' / f'{args.dataset}_base4')
    data = A.L.load_large(args.dataset,
        '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip',
        cache, args.device, {'groups': groups, 'report_dev_mod': 3})
    loaded_mb = torch.cuda.memory_allocated(args.device) / 2**20
    data.pop('x_device', None)
    data.pop('real_adj', None)
    gc.collect()
    torch.cuda.empty_cache()
    evaluation_cache_mb = loaded_mb - torch.cuda.memory_allocated(args.device) / 2**20
    ids = A.L.stage_ids(data)
    budgets, _ = A.R.V.choose_budgets(reference, list(map(len, ids)), groups, 'fixed_rate', .05)
    archive = source / 'final_uq_dm_edgefree.pt.gz'
    if archive.exists():
        memory = A.load_gzip(archive)
        memory_source, expected_hash = 'archived_final_memory', None
    else:
        memory, memory_source, expected_hash = A.reconstruct_primary_memory(
            reference, data, ids, args.dataset, 271006, args.device)
    memory = cpu_only(memory)
    full = cpu_only(F.raw_memory(data, torch.cat(ids), len(ids) - 1))
    prompt = data['prompt_emb'].detach().cpu()
    data['encoder'].model.cpu()
    data['encoder'].device = 'cpu'
    del data
    gc.collect()
    torch.cuda.empty_cache()
    after_purge_mb = torch.cuda.memory_allocated(args.device) / 2**20
    # Record the small runtime baseline separately from graph/model tensors.
    if after_purge_mb > 32:
        raise AssertionError(f'Unaccounted GPU residency after purge: {after_purge_mb} MiB')
    previous = 0
    dimensions = []
    for stage, (raw_ids, budget) in enumerate(zip(ids, budgets)):
        new_rows = int(budget) - previous
        dimensions.append({'stage': stage, 'raw_nodes': len(raw_ids),
            'new_slots': new_rows, 'frozen_prefix_rows': previous,
            'dense_logit_elements': len(raw_ids) * new_rows})
        previous = int(budget)
    report = {'status': 'running', 'dataset': args.dataset,
        'protocol': {'order_seed': 9101, 'construction_seed': 271006,
            'student_seeds': list(A.STUDENT_SEEDS), 'student_steps': 2048,
            'student_batch': 24, 'lr': .001, 'evaluation_performed': False,
            'text_embeddings': 'same frozen MiniLM; precomputed before timing',
            'measurement': 'absolute torch.cuda.max_memory_allocated and synchronized wall time; training only',
            'evaluation_cache_resident': False, 'text_encoder_resident': False,
            'full_raw_reference': 'all final legal pool nodes and induced edges',
            'full_raw_edgefree': 'same raw nodes, labels and texts; self-loop adjacency control',
            'construction_in_timing': False, 'warmup_updates': 2,
            'student_code': str(Path(A.P.__file__).resolve()),
            'student_sha256': A.sha_file(Path(A.P.__file__)),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'memory_source': memory_source, 'memory_hash': A.R.V.memory_hash(memory),
        'expected_memory_hash': expected_hash, 'stage_dimensions': dimensions,
        'full_dense_logit_elements': len(full['x']) * len(memory['x']),
        'loader_gpu_mb': loaded_mb, 'evaluation_cache_gpu_mb': evaluation_cache_mb,
        'idle_gpu_mb_after_purge': after_purge_mb, 'rows': []}
    A.jwrite(path, report)
    conditions = [('memory_5pct', memory), ('full_raw_graph', full),
        ('full_raw_edgefree', dict(full, edge_index=torch.empty((2, 0), dtype=torch.long)))]
    for name, training_memory in conditions:
        warm = A.P.train_prompt_student(training_memory, prompt, 149, args.device, 2, batch=24)
        del warm
        for seed in A.STUDENT_SEEDS:
            gc.collect()
            torch.cuda.empty_cache()
            A.sync(args.device)
            baseline_mb = torch.cuda.memory_allocated(args.device) / 2**20
            if baseline_mb > 32:
                raise AssertionError(f'Previous training state retained: {baseline_mb}')
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            model = A.P.train_prompt_student(training_memory, prompt, seed,
                args.device, 2048, lr=.001, batch=24)
            A.sync(args.device)
            elapsed = time.perf_counter() - started
            row = {'condition': name, 'student_seed': seed,
                'nodes': len(training_memory['x']),
                'edges': int(training_memory['edge_index'].shape[1]),
                'seconds': elapsed,
                'peak_allocated_mb': torch.cuda.max_memory_allocated(args.device) / 2**20,
                'peak_reserved_mb': torch.cuda.max_memory_reserved(args.device) / 2**20,
                'baseline_mb': baseline_mb}
            report['rows'].append(row)
            A.jwrite(path, report)
            print(json.dumps({'event': 'training_complete', 'dataset': args.dataset, **row}), flush=True)
            del model
    report['status'] = 'complete'
    A.jwrite(path, report)


if __name__ == '__main__':
    main()
