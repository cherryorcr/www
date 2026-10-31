"""Evaluate saved cap=1/cap=0 topology candidates under the report protocol.

Construction and topology selection are already sealed in the source run. This
script only reads the saved candidate memories and performs downstream Student
evaluation, so it never feeds report/test outcomes back into the gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import run_tgc_semantic_recursive_large_20260926 as L
import run_unified_v3_continual as U


def load(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding='utf-8')


def run_students_with_saved_edges(memory: dict, stage: int, data: dict,
                                  args, device: str) -> dict:
    """Run the original Student while honoring ``memory.edge_index``.

    The legacy adapter reconstructs edges from text for historical controls.
    That is useful as a legacy protocol, but it erases the A/topology
    intervention. This path keeps the same GCN, optimizer, seeds, steps and
    report masks while passing the saved candidate edges.
    """
    syn = {
        'x': memory['x'], 'text_emb': memory['text_emb'],
        'texts': memory['texts'], 'edge_index': memory['edge_index'],
        'weights': memory.get('mass'),
    }
    rows = []
    for seed in args.student_seeds:
        steps = U.protocol_v3.student_steps(
            len(syn['x']), override=args.student_steps)
        model = U.G.train_student(
            syn, int(seed), device, steps, batch=args.student_batch)
        rows.append(U.G.eval_stagewise(
            model, data['x_device'], data['real_adj'], data['prompt_emb'],
            data['y'], data['report_dev_mask'], stage))
        del model
    return {
        'roots': int(len(syn['x'])),
        'student_steps': int(steps),
        'student_batch': int(args.student_batch),
        'student_seeds': list(args.student_seeds),
        'student_training_runs': int(len(args.student_seeds)),
        'student_optimizer_steps_total': int(steps * len(args.student_seeds)),
        'cum_dev_f1_mean': float(sum(row['cumulative_macro_f1']
                                     for row in rows) / len(rows)),
        'cum_dev_f1_std': 0.0,
        'current_f1_mean': float(sum(row['current_f1'] for row in rows)
                                 / len(rows)),
        'old_mean_f1_mean': (
            float(sum(row['old_mean_f1'] for row in rows) / len(rows))
            if stage > 0 else None),
        'per_stage_mean': [
            float(sum(row['per_stage'][s]['macro_f1'] for row in rows)
                  / len(rows)) for s in range(stage + 1)],
        'per_student': rows,
    }


def prepare_data(dataset: str, source: Path, out: Path, minilm: str,
                 device: str) -> tuple[dict, list[list[int]]]:
    manifest = json.loads((source / 'results.json').read_text())
    nested = manifest.get('protocol')
    protocol = nested if isinstance(nested, dict) else manifest
    groups = [[int(v) for v in group]
              for group in protocol.get('stage_classes', protocol.get('groups', []))]
    data = U.load_unified(
        dataset, U.DATA_CFG[dataset]['path'], minilm, device,
        n_stages=5, split_mode='base_incremental', base_fraction=0.4,
        report_dev_mod=int(protocol.get('report_dev_mod', 3)))
    cumulative, seen = [], []
    for group in groups:
        seen.extend(group)
        cumulative.append(list(seen))
    data['stage_classes'] = groups
    U.G.STAGE_CLASSES = cumulative
    try:
        import run_gate1_continual as legacy
        legacy.STAGE_CLASSES = cumulative
    except ImportError:
        pass
    data = L.set_pool_text_embeddings(data, source, out)
    data['prompt_emb'] = data['encoder'].encode(data['prompts'])
    data['real_adj'] = U.G.norm_adj(data['edge_index'], len(data['x']), device)
    data['x_device'] = data['x'].float().to(device)
    return data, groups


def evaluate(dataset: str, experiment_root: Path, source: Path,
             minilm: str, device: str, seeds: list[int], stage: int,
             graph_mode: str) -> dict:
    data, groups = prepare_data(dataset, source, experiment_root, minilm, device)
    base = experiment_root / f'{dataset}_base4'
    args = SimpleNamespace(student_seeds=seeds, student_batch=16,
                           student_steps=None)
    arms = {
        'cap1_revision': base / f'stage{stage}/candidate_cap1_revision.pt',
        'cap0_control': base / f'stage{stage}/candidate_cap0_control.pt',
        'gate_selected': base / f'stage{stage}/memory.pt',
    }
    result = {
        'dataset': dataset, 'stage': int(stage), 'student_seeds': seeds,
        'evaluation': ('downstream report-dev only; no construction decision; '
                       + graph_mode),
        'arms': {},
    }
    for name, path in arms.items():
        memory = load(path)
        if graph_mode == 'edge-aware':
            summary = run_students_with_saved_edges(
                memory, stage, data, args, device)
        else:
            summary = U.run_students(memory, stage, data, args, device)
        result['arms'][name] = summary
        print(json.dumps({'event': 'candidate_eval_complete',
                          'dataset': dataset, 'arm': name,
                          'macro_f1': summary['cum_dev_f1_mean'],
                          'accuracy': sum(
                              row.get('cumulative_accuracy', 0.0)
                              for row in summary['per_student'])
                          / len(summary['per_student'])}), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--experiment-root', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--minilm', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--student-seeds', nargs='+', type=int,
                    default=[100, 101, 102, 103, 104, 105])
    ap.add_argument('--graph-mode', choices=('legacy', 'edge-aware'),
                    default='edge-aware')
    args = ap.parse_args()
    root = args.experiment_root / f'{args.dataset}_base4'
    result = evaluate(
        args.dataset, args.experiment_root,
        args.source_root / f'{args.dataset}_base4', args.minilm, args.device,
        args.student_seeds, args.stage, args.graph_mode)
    output_name = ('candidate_evaluation_edge_aware.json'
                   if args.graph_mode == 'edge-aware'
                   else 'candidate_evaluation.json')
    save(root / output_name, result)
    print(json.dumps({'event': 'candidate_eval_saved',
                      'path': str(root / output_name)}), flush=True)


if __name__ == '__main__':
    main()
