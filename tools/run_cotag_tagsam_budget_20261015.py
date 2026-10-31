"""Fixed-prefix TAGSAM response-budget diagnostic; no model selection."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
from pathlib import Path
import time

import run_cotag_four_gap_20261015 as B

R, E, torch = B.R, B.E, B.torch
import uq_tagsam_native_selector_20261012 as N
import run_unified_q_tagsam_20261008 as A


def restore_native_runtime(source):
    manifest = B.read_full_json(source / 'runtime_manifest.json')
    recorded = manifest['source_files']
    name = 'run_student_compatible_append.py'
    path = Path(next(p for p in recorded if Path(p).name == name))
    if E.file_hash(path) != recorded[str(path)]:
        raise AssertionError('Archived TAGSAM append source hash differs')
    spec = importlib.util.spec_from_file_location('cotag_tagsam_archived_append', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    A.T.SCA = module
    A.T.C.SCA = module
    checks = {}
    for current in (N, A, A.T, A.T.C, A.T.G, A.T.SCA):
        filename = Path(current.__file__).name
        expected = next(value for key, value in recorded.items()
                        if Path(key).name == filename)
        actual = E.file_hash(current.__file__)
        if actual != expected:
            raise AssertionError(f'TAGSAM source differs: {filename}')
        checks[filename] = dict(path=current.__file__, sha256=actual, matches_archive=True)
    return checks


def run(args):
    data, stages, protocol = B.load(args)
    source = (B.ROOT / 'results/uq_final_evidence_20261012/baselines' /
              f'tagsam_selector_response_{args.dataset}_o9101_c271006_retry20261013')
    native_runtime = restore_native_runtime(source)
    previous = B.read_full_json(source / 'results.json')
    for key in ('groups', 'budgets', 'stage_id_hashes', 'student', 'student_steps',
                'student_batch', 'student_seeds', 'pool_mask_hash', 'report_mask_hash', 'test_mask_hash'):
        if protocol[key] != previous['protocol'][key]:
            raise AssertionError(f'TAGSAM source protocol differs: {key}')
    archive = source / 'final_tagsam_selector_response.pt.gz'
    with gzip.open(archive, 'rb') as stream:
        completed = torch.load(stream, map_location='cpu', weights_only=False)
    stage = len(stages) - 1
    old_n = protocol['budgets'][-2]
    tensor_keys = ('x', 'text_emb', 'labels', 'mass', 'depth', 'stage')
    parent = {key: completed[key][:old_n].clone() for key in tensor_keys}
    parent.update(texts=list(completed['texts'][:old_n]),
                  edge_index=torch.empty((2, 0), dtype=torch.long))
    delta = R.L.delta_view(data, stages[stage], stage)
    loss, official = A.T.official_loss(N.REPO)
    protocol.update(
        diagnostic='100 vs 500 outer updates at final stage with the same frozen TAGSAM prefix',
        arms=[100, 500], evaluated_stage=stage, prefix_rows=old_n,
        prefix_hash=R.V.memory_hash(parent), source_result=str(source / 'results.json'),
        source_result_sha256=E.file_hash(source / 'results.json'),
        source_memory_sha256=E.file_hash(archive), outer_steps=[100, 500], inner_steps=10,
        teacher_steps=2048, reference_cap=256, official_source=official,
        access='current delta and previously saved synthetic prefix only',
        runner_hash=E.file_hash(__file__),
        diagnostic_module_hashes={Path(m.__file__).name: E.file_hash(m.__file__)
            for m in (B, N, A, A.T, A.T.C, A.T.G, A.T.SCA)},
        archived_runtime_checks=native_runtime,
        caveat='Not all-stage 500 updates or native 5000x15; no distilled-rate downstream training')
    result = B.new_result(args, protocol)
    if result is None:
        return
    begin = time.perf_counter()
    initial, selector = N.selector_block(delta, protocol['budgets'][-1]-old_n,
        data['encoder'], args.construction_seed, args.device)
    R.V.sync(args.device)
    result['selector'] = selector
    result['selector_seconds'] = time.perf_counter()-begin
    result['initial_memory_hash'] = R.V.memory_hash(initial)
    raw = A.reference_memory(delta, parent)
    teacher = A.T.G.train_student(raw, 9000+stage, args.device, 2048, lr=.001, batch=24)
    B.save(args.output / 'results.json', result)
    for steps in (100, 500):
        R.V.sync(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        begin = time.perf_counter()
        memory, audit = A.T.optimize(initial, parent, raw, teacher, stage,
            'tagsam_selfloop', args.device, loss, steps)
        memory['labels'] = torch.cat((parent['labels'], initial['labels']))
        checks = R.V.validate_memory(memory, parent, protocol['budgets'][-1])
        if memory['edge_index'].numel():
            raise AssertionError('Unexpected synthetic edges')
        R.V.sync(args.device)
        seconds = time.perf_counter()-begin
        peak = torch.cuda.max_memory_allocated(args.device)/2**20
        if steps == 100:
            result['archived_100_replay'] = dict(
                x_max_error=float((memory['x']-completed['x']).abs().max()),
                x_relative_norm_error=float((memory['x']-completed['x']).norm()/completed['x'].norm().clamp_min(1e-12)),
                text_identity=memory['texts'] == completed['texts'],
                label_identity=bool(torch.equal(memory['labels'], completed['labels'])),
                archived_memory_hash=R.V.memory_hash(completed),
                reconstructed_memory_hash=R.V.memory_hash(memory))
        scores = E.evaluate(memory, data, stage, B.SEEDS, args.device)
        result['rows'].append(dict(stage=stage, arm=f'tagsam_outer_{steps}',
            outer_steps=steps, memory_rows=len(memory['x']),
            memory_hash=R.V.memory_hash(memory), checks=checks, scores=scores,
            score=scores['report_dev'], response_seconds=seconds,
            response_peak_mb=peak, audit=audit))
        B.save(args.output / 'results.json', result)
        B.emit(event='tagsam_budget_evaluated', dataset=args.dataset,
               steps=steps, report_f1=scores['report_dev']['final_f1'],
               test_f1=scores['sealed_test']['final_f1'])
        del memory
        torch.cuda.empty_cache()
    result['status'] = 'complete'
    B.save(args.output / 'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=('arxiv', 'sportsfit', 'bookhis'), required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--text-cache', type=Path, required=True)
    p.add_argument('--minilm', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--order-seed', type=int, default=9101)
    p.add_argument('--construction-seed', type=int, default=271006)
    args = p.parse_args()
    args.task, args.arm = 'tagsam_budget', 'outer_100_500'
    if args.order_seed != 9101 or args.construction_seed != 271006:
        raise ValueError('Only the registered fixed-prefix source cell is supported')
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
