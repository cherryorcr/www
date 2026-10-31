"""Actual Student permutation controls on exact previously archived memories."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

import unified_q_pairing_readout as D
import run_unified_q_gecc_gap_20261009 as G

C, A, R, L, V, P, S = G.C, G.A, G.R, G.L, G.V, G.P, G.S
ARMS = ('uq_dm_edgefree', 'cat_cctp', 'gecc_hard')
PERMUTATIONS = (271011, 271012)


def run(args):
    source_path = args.source / 'results.json'
    source = json.loads(source_path.read_text())
    if source['status'] != 'complete':
        raise ValueError('source incomplete')
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                        dict(groups=groups, report_dev_mod=reference.get('report_dev_mod', 3)))
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, 'fixed_schedule', .05)
    cfg = R.q_config('uq_dm_edgefree', args.construction_seed, 30, 2048)
    common = G.check_source(source['protocol'], args, data, ids, budgets, cfg)
    protocol = dict(common, name='uq-pairing-readout-v1', source=str(args.source),
        source_results_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        code_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                     for m in (D, P, S, G)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arms=list(ARMS), permutation_seeds=list(PERMUTATIONS), groups=groups,
        control='post-construction permutation within stage/class; all text objects and provenance move together',
        constructor_changed=False, student_changed=False, development_only=True,
        evaluation='final memory report-dev; transductive inference; not sealed test',
        gradients='initialization-only; 32 batch-24 paired minibatches; no optimization',
        storage='permutation indices and hashes only; base tensor archive reused')
    path = args.output / 'results.json'
    args.output.mkdir(parents=True, exist_ok=True)
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol,
        rows=[], diagnostics=[], geometry=[], controls=[], status='running')
    if result['protocol'] != protocol:
        raise ValueError('changed source or protocol on resume')
    V.write_json(args.output / 'protocol.json', protocol)
    V.write_json(path, result)
    for arm in ARMS:
        archive = args.source / f'final_{arm}.pt.gz'
        memory = C.load_memory(archive)
        digest = V.memory_hash(memory)
        recorded = [r for r in source['stages'] if r['arm'] == arm][-1]
        if digest != recorded['memory_hash'] or memory['edge_index'].numel():
            raise AssertionError('invalid base archive')
        if not any(r['arm'] == arm for r in result['geometry']):
            result['geometry'].append(dict(arm=arm, memory_hash=digest, **D.text_geometry(memory)))
        for row in source['rows']:
            if row['arm'] == arm and not any(r['arm'] == arm and r['student'] == row['student']
                                             and r['permutation_seed'] is None for r in result['rows']):
                if row['memory_hash'] != digest:
                    raise AssertionError('original readout hash differs')
                result['rows'].append(dict(row, permutation_seed=None, reused=True))
        V.write_json(path, result)
        for permutation_seed in PERMUTATIONS:
            permutation = D.within_stage_class_permutation(memory, permutation_seed)
            variant = D.permute_payload(memory, permutation)
            vdigest = V.memory_hash(variant)
            control = dict(arm=arm, permutation_seed=permutation_seed, permutation=permutation.tolist(),
                permutation_sha256=V.hash_tensor(permutation), base_memory_hash=digest, memory_hash=vdigest,
                base_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                base_archive=str(archive),
                fixed_hashes={k: V.hash_tensor(memory[k]) for k in A.FIXED_KEYS},
                changed_texts=sum(a != b for a, b in zip(memory['texts'], variant['texts'])),
                moved_rows=int((permutation != torch.arange(len(permutation))).sum()),
                text_multiset_preserved=True, class_stage_preserved=True,
                source_ids_sha256=hashlib.sha256(json.dumps(variant['text_source_ids']).encode()).hexdigest())
            key = f'{arm}_p{permutation_seed}'
            sidecar = args.output / f'{key}.json'
            if sidecar.exists() and json.loads(sidecar.read_text()) != control:
                raise AssertionError('permutation reconstruction changed')
            V.write_json(sidecar, control)
            if not any(r['arm'] == arm and r['permutation_seed'] == permutation_seed for r in result['controls']):
                result['controls'].append({k: v for k, v in control.items() if k != 'permutation'})
            for name, trainer, factory in (
                    ('prompt_gcn', None, P.GraphTextPromptStudent),
                    ('prompt_sgc', S.train_sgc_student, S.GraphTextSGCStudent)):
                if permutation_seed == PERMUTATIONS[0] and not any(
                        r['arm'] == arm and r['student'] == name for r in result['diagnostics']):
                    with A.preserve_rng():
                        probe = D.gradient_probe(memory, data['prompt_emb'], factory, args.device,
                            args.construction_seed + 2711, permutation)
                    result['diagnostics'].append(dict(arm=arm, student=name, **probe))
                    V.write_json(path, result)
                    print(json.dumps(dict(event='gradient_probe', arm=arm, student=name,
                        full_batch_cos=probe['full_group_to_minibatch_gradient_cosine'])), flush=True)
                if any(r['arm'] == arm and r['student'] == name and
                       r['permutation_seed'] == permutation_seed for r in result['rows']):
                    continue
                V.sync(args.device)
                started = time.perf_counter()
                with A.preserve_rng():
                    score = L.score(variant, data, 4, args.student_seeds, args.device,
                                    args.student_steps, 24, trainer=trainer)
                V.sync(args.device)
                if V.memory_hash(variant) != vdigest or V.memory_hash(memory) != digest:
                    raise AssertionError('evaluation mutated frozen inputs')
                result['rows'].append(dict(arm=arm, student=name, permutation_seed=permutation_seed,
                    score=score, memory_hash=vdigest, base_memory_hash=digest, reused=False,
                    student_seconds=time.perf_counter() - started))
                V.write_json(path, result)
                print(json.dumps(dict(event='evaluated', dataset=args.dataset, arm=arm, student=name,
                    permutation_seed=permutation_seed, f1=score['final_f1'], accuracy=score['accuracy'])), flush=True)
            del variant
        del memory
    if len(result['rows']) != 18 or len(result['controls']) != 6 or len(result['diagnostics']) != 6:
        raise AssertionError('incomplete pairing experiment')
    result['status'] = 'complete'
    V.write_json(path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'output', 'reference', 'text-cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dataset', required=True, choices=('wikics', 'sportsfit', 'bookhis'))
    parser.add_argument('--minilm', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--order-seed', type=int, required=True)
    parser.add_argument('--construction-seed', type=int, required=True)
    parser.add_argument('--student-steps', type=int, default=2048)
    parser.add_argument('--student-seeds', type=int, nargs='+', default=[150, 151, 152])
    torch.set_num_threads(2)
    run(parser.parse_args())
