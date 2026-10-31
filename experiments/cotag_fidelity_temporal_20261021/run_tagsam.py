"""Full independent stage chains for predeclared TAGSAM response budgets."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import common as C
from common import E, R, torch
import tagsam_response as T
import uq_tagsam_native_selector_20261012 as N

BUDGETS = ((100, 10), (500, 10), (5000, 15))


def initial_block(data, delta, rows, seed, device, cache):
    protocol = dict(source_id_hash=R.V.hash_tensor(delta['ids']), rows=rows, seed=seed,
                    native_selector_hash=C.sha(N.__file__), source_hashes=C.source_hashes())
    if (cache/'block.pt.gz').exists():
        audit = C.read(cache/'audit.json')
        if audit['protocol'] != protocol:
            raise AssertionError('Selector cache protocol differs')
        block = C.load_memory(cache/'block.pt.gz')
        if R.V.memory_hash(block) != audit['memory_hash']:
            raise AssertionError('Selector cache corrupted')
        return block, audit
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    block, selector = N.selector_block(delta, rows, data['encoder'], seed, device)
    R.V.sync(device)
    audit = dict(protocol=protocol, selector=selector, seconds=time.perf_counter()-start,
                 peak_mb=torch.cuda.max_memory_allocated(device)/2**20,
                 memory_hash=R.V.memory_hash(block))
    C.archive(cache/'block.pt.gz', block)
    C.save(cache/'audit.json', audit)
    return block, audit


def run(args):
    C.disk_check()
    data, ids, base = C.class_data(args.dataset, args.device)
    loss_fn, official = T.T.official_loss(N.REPO)
    root = C.RESULTS / 'tagsam' / f'{args.dataset}_o9101_c271006'
    for outer, inner in BUDGETS:
        out = root / f'outer{outer}_updates{inner}'
        protocol = dict(name='tagsam-full-chain-budget-v1', dataset=args.dataset,
            order_seed=9101, construction_seed=271006, **base, outer_steps=outer,
            inner_mode='minibatch updates, not complete synthetic-data epochs',
            inner_steps=inner, inner_batch=16, reference_cap=256, reference_resampled=False,
            teacher_steps=2048, student_steps=2048, student_batch=24,
            student_seeds=C.STUDENT_SEEDS, student=R.P.model_config(), memory_rate=.05,
            access='current legal delta plus this arm own frozen prefix',
            initial_reuse='prefix-independent native PMI selector blocks only',
            official_source=official, official_reproduction=False,
            remaining_changes=['common encoders and proxy', '320-character text cap',
                'minibatch unroll not full native epochs', 'fixed response subset 256 not resampled 2000',
                'common downstream optimizer not distilled learning rates'],
            method_selection_from_test=False, evaluation='previously inspected historical masks',
            source_hashes=C.source_hashes(), runner_hash=C.sha(__file__),
            response_hash=C.sha(T.__file__), selector_hash=C.sha(N.__file__))
        result = C.start_result(out/'results.json', protocol)
        if result['status'] == 'complete':
            continue
        parent = None
        try:
            for stage, idx in enumerate(ids):
                stage_dir = out/f's{stage}'
                path, audit_path = stage_dir/'memory.pt.gz', stage_dir/'construction.json'
                completed = next((r for r in result['rows'] if r['stage'] == stage), None)
                if path.exists() and audit_path.exists():
                    memory, audit = C.load_memory(path), C.read(audit_path)
                    R.V.validate_memory(memory, parent, base['budgets'][stage])
                    if R.V.memory_hash(memory) != audit['memory_hash']:
                        raise AssertionError('Stage checkpoint corrupted')
                else:
                    delta = R.L.delta_view(data, idx, stage)
                    new_rows = base['budgets'][stage]-(0 if parent is None else len(parent['x']))
                    initial, selection = initial_block(data, delta, new_rows, 271006, args.device,
                                                       root/f'selector_s{stage}')
                    raw = T.A.reference_memory(delta, parent)
                    start = time.perf_counter()
                    teacher = T.T.G.train_student(raw, 9000+stage, args.device, 2048, lr=.001, batch=24)
                    R.V.sync(args.device)
                    teacher_seconds = time.perf_counter()-start
                    torch.cuda.reset_peak_memory_stats(args.device)
                    start = time.perf_counter()
                    memory, response = T.optimize(initial, parent, raw, teacher, stage,
                        args.device, loss_fn, outer, loops=inner, trace_path=stage_dir/'trace.json')
                    R.V.sync(args.device)
                    audit = dict(stage=stage, memory_hash=R.V.memory_hash(memory),
                        prefix_hash=None if parent is None else R.V.memory_hash(parent),
                        initial_hash=selection['memory_hash'], selector_seconds=selection['seconds'],
                        selector_peak_mb=selection['peak_mb'], teacher_seconds=teacher_seconds,
                        response_seconds=time.perf_counter()-start,
                        response_peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20,
                        response=response, checks=R.V.validate_memory(memory, parent, base['budgets'][stage]))
                    C.archive(path, memory)
                    C.save(audit_path, audit)
                    del teacher, raw, delta
                if completed is None:
                    scores = E.evaluate(memory, data, stage, C.STUDENT_SEEDS, args.device)
                    result['rows'].append(dict(stage=stage, memory_rows=len(memory['x']),
                        memory_hash=R.V.memory_hash(memory), construction=audit, scores=scores))
                    C.save(out/'results.json', result)
                    C.emit(event='tagsam_evaluated', dataset=args.dataset, outer=outer, inner=inner,
                        stage=stage, report_f1=scores['report_dev']['final_f1'])
                elif completed['memory_hash'] != R.V.memory_hash(memory):
                    raise AssertionError('Completed metrics differ from stage memory')
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
    parser.add_argument('--dataset', choices=('arxiv', 'sportsfit', 'bookhis'), required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
