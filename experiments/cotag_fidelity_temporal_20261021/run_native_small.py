"""Higher-fidelity complete Cora chain, common student and native internal core."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import sys

import common as C
from common import R, E, torch
sys.path.append(str(C.ROOT/'tools'))
import run_uq_baseline_expansion_20261013 as B
import run_tagsam as T
import native_tagsam as N


def run(args):
    root = C.RESULTS / 'native_epochs' / f'{args.dataset}_outer{args.outer}'
    loader = SimpleNamespace(dataset=args.dataset, minilm=C.MINILM, device=args.device,
        order_seed=9101, text_cache=C.ROOT/f'results/uq_baseline_expansion_20261013/text_cache/{args.dataset}')
    ref = B.read_reference(None, args.dataset)
    data = B.load_data(loader, ref, R)
    ids = R.L.stage_ids(data)
    budgets, floors = R.V.choose_budgets(ref, list(map(len, ids)), data['stage_classes'], 'fixed_rate', .05)
    native = N.components(args.device, data['x'].shape[1])
    protocol = dict(name='native-tagsam-full-epoch-continual-v1', dataset=args.dataset,
        order_seed=9101, construction_seed=271006, groups=data['stage_classes'], budgets=budgets,
        memory_rate=.05, class_floor_adjustments=floors, outer_steps=args.outer,
        inner_epochs=15, inner_batch=20, native_source=native.provenance,
        internal_bert=str(N.BERT), final_student=R.P.model_config(),
        student_steps=2048, student_batch=24, student_seeds=C.STUDENT_SEEDS,
        retained=['native GPT2 PMI selector', 'native CLIP/GCN and residual TextProjection',
            'frozen BERT CLS inside construction', 'native full-epoch unroll function',
            'native wBCELoss', 'native SGD with learned rates and StepLR', 'fresh response subset up to 2000'],
        adaptations=['current legal delta and frozen own prefix', '5% cumulative row budget',
            '320-character common stored text cap and documented selector numerical repairs',
            'teacher fixed final 3-epoch model, not official evaluation-accuracy selection',
            'full legal-source graph in place of NeighborSampler; same GCN operator',
            'common MiniLM/prompt student for comparison; native distilled rates not used downstream'],
        official_reproduction=False, method_selection_from_test=False,
        resource_isolation=False, resource_note='May share GPU with bounded adapter diagnostic; runtime is not an isolated efficiency comparison',
        evaluation='previously inspected historical masks; not new blind data',
        source_hashes=C.source_hashes(), runner_hash=C.sha(__file__), native_wrapper_hash=C.sha(N.__file__),
        data_loader_hash=C.sha(B.__file__),
        stage_id_hashes=[R.V.hash_tensor(idx) for idx in ids],
        pool_mask_hash=R.V.hash_tensor(data['pool_mask']),
        report_mask_hash=R.V.hash_tensor(data['report_dev_mask']),
        test_mask_hash=R.V.hash_tensor(data['test_mask']))
    result = C.start_result(root/'results.json', protocol)
    if result['status'] == 'complete':
        return
    bert = N.BertFeatures(args.device)
    parent = None
    try:
        for stage, idx in enumerate(ids):
            stage_dir = root/f's{stage}'
            memory_path, audit_path = stage_dir/'memory.pt.gz', stage_dir/'construction.json'
            completed = next((row for row in result['rows'] if row['stage'] == stage), None)
            if memory_path.exists() and audit_path.exists():
                memory, audit = C.load_memory(memory_path), C.read(audit_path)
                if R.V.memory_hash(memory) != audit['memory_hash']:
                    raise AssertionError('Native stage memory changed')
                R.V.validate_memory(memory, parent, budgets[stage])
            else:
                delta = R.L.delta_view(data, idx, stage)
                rows = budgets[stage]-(0 if parent is None else len(parent['x']))
                initial, selector = T.initial_block(data, delta, rows, 271006, args.device,
                    C.RESULTS/'native_epochs'/f'{args.dataset}_selector_s{stage}')
                raw = T.T.A.reference_memory(delta, parent)
                synthetic = R.L.append_edge_free(parent, initial)
                raw_text, syn_text = bert.encode(raw['texts']), bert.encode(synthetic['texts'])
                start = time.perf_counter()
                teacher, teacher_audit = N.train_teacher(native, raw, raw_text, stage, args.device)
                R.V.sync(args.device)
                teacher_seconds = time.perf_counter()-start
                torch.cuda.reset_peak_memory_stats(args.device)
                start = time.perf_counter()
                memory, response = N.optimize(native, initial, parent, syn_text, raw, raw_text,
                                               teacher, stage, args.device, args.outer, stage_dir)
                R.V.sync(args.device)
                audit = dict(stage=stage, memory_hash=R.V.memory_hash(memory),
                    memory_rows=len(memory['x']), initial_hash=selector['memory_hash'],
                    selector_seconds=selector['seconds'], teacher=teacher_audit,
                    teacher_seconds=teacher_seconds, response=response,
                    response_seconds=time.perf_counter()-start,
                    response_peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20,
                    checks=R.V.validate_memory(memory, parent, budgets[stage]))
                C.archive(memory_path, memory)
                C.save(audit_path, audit)
                del teacher, raw, raw_text, syn_text, synthetic, delta
            if completed is None:
                scores = E.evaluate(memory, data, stage, C.STUDENT_SEEDS, args.device)
                result['rows'].append(dict(stage=stage, construction=audit, scores=scores))
                C.save(root/'results.json', result)
                C.emit(event='native_epoch_evaluated', stage=stage,
                       f1=scores['report_dev']['final_f1'], outer=args.outer)
            parent = memory
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        C.save(root/'results.json', result)
        raise
    C.save(root/'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('cora', 'citeseer'), default='cora')
    parser.add_argument('--outer', type=int, choices=(100, 5000), default=5000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
