"""Migrate the frozen Bounded X-P-A operator to the large base+4 protocol."""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

import direct_student_outer as DSO
import prepare_class_aware_x_support as CLASS
import run_semantic_recursive_unified_baselines_20260926 as B
import run_tgc_semantic_recursive_large_20260926 as L
import run_unified_v3_continual as U


def load(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def save_json(path: Path, value: dict) -> None:
    def default(item):
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu()
            if value.numel() <= 64:
                return value.tolist()
            return {'tensor_shape': list(value.shape),
                    'tensor_mean': float(value.float().mean()),
                    'tensor_norm': float(value.float().norm())}
        raise TypeError(f'not JSON serializable: {type(item).__name__}')
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True,
                               default=default), encoding='utf-8')


def clone_memory(memory: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor)
            else copy.deepcopy(value) for key, value in memory.items()}


def compose_with_parent(c0: dict, parent: dict) -> dict:
    old_rows = len(parent['x'])
    if len(c0['x']) < old_rows:
        raise ValueError('current C0 memory is shorter than recursive parent')
    out = clone_memory(c0)
    for key in ('x', 'text_emb', 'mass', 'depth', 'stage'):
        out[key][:old_rows] = parent[key]
    out['texts'] = list(parent['texts']) + list(c0['texts'])[old_rows:]
    # Preserve the parent's old-old topology; only current-C0 edges touching
    # newly appended rows are allowed to enter the recursive candidate.
    parent_edges = parent['edge_index'].long()
    current_edges = c0['edge_index'].long()
    if current_edges.numel():
        keep = ((current_edges[0] >= old_rows)
                | (current_edges[1] >= old_rows))
        current_edges = current_edges[:, keep]
    if parent_edges.numel() and current_edges.numel():
        merged = torch.cat((parent_edges, current_edges), dim=1)
    elif parent_edges.numel():
        merged = parent_edges
    else:
        merged = current_edges
    out['edge_index'] = merged.contiguous()
    return out


def frozen_config(plan_seed: int, old_graph_anchor_weight: float = 0.0,
                  old_graph_anchor_fraction: float = 1.0,
                  new_loss_scale: float = 1.0,
                  a_max_revisions_per_row: int = 1) -> DSO.Config:
    return DSO.Config(
        inner_steps=8, outer_steps=4, outer_lr=0.02,
        old_weight=0.25, new_weight=1.0,
        new_loss_scale=float(new_loss_scale),
        x_old_radius=0.05, x_new_radius=0.10,
        p_mix=0.25, p_temperature=0.20, p_sinkhorn_iters=24,
        a_pool=4, a_temperature=1.0, a_existing_logit=0.1,
        a_max_revisions_per_row=int(a_max_revisions_per_row),
        candidate_prompt_margin_weight=1.0,
        candidate_prompt_margin=0.20,
        class_representation_weight=1.0,
        old_graph_anchor_weight=float(old_graph_anchor_weight),
        old_graph_anchor_fraction=float(old_graph_anchor_fraction),
        train_seeds=(9300,), validation_seeds=(9310,), plan_seed=int(plan_seed))


def score_summary(summary: dict, stage: int) -> dict:
    students = summary['per_student']
    f1 = [row['cumulative_macro_f1'] for row in students]
    acc = [row.get('cumulative_accuracy', row.get('accuracy', 0.0))
           for row in students]
    stage_f1 = summary['per_stage_mean']
    stage_acc = []
    for index in range(stage + 1):
        values = [row['per_stage'][index].get('accuracy', 0.0)
                  for row in students]
        stage_acc.append(sum(values) / len(values))
    return {
        'mean_f1': float(summary['cum_dev_f1_mean']),
        'std_f1': float(summary['cum_dev_f1_std']),
        'mean_accuracy': float(sum(acc) / len(acc)),
        'std_accuracy': float(torch.tensor(acc).float().std(unbiased=False)),
        'old_f1': summary.get('old_mean_f1_mean'),
        'new_f1': summary.get('current_f1_mean'),
        'per_stage_f1': stage_f1,
        'per_stage_accuracy': stage_acc,
    }


def run_students_with_saved_edges(memory: dict, stage: int, data: dict,
                                  args: argparse.Namespace,
                                  device: str) -> dict:
    """Run the original Student with the candidate memory topology.

    The legacy unified adapter reconstructs synthetic edges from text.  That
    remains available as a historical control, but it would erase the A
    intervention.  This path changes only the graph passed to the same GCN
    Student, optimizer, seeds, steps and report masks.
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


def sealed_validation_response(candidate: dict, support: dict, old_rows: int,
                               cfg: DSO.Config, device: str,
                               support_class_id: torch.Tensor,
                               prompt_emb: torch.Tensor,
                               prompt_ids: torch.Tensor,
                               base_slot_class_id: torch.Tensor) -> dict:
    """Measure a topology candidate without using report/test outcomes.

    The gate uses the same sealed support split as the outer construction, but
    evaluates the candidate with the original Student from a fresh validation
    seed.  It reports the weighted support response, its old/new components,
    and the candidate-row class margin.  This is deliberately separate from
    ``U.run_students``: the latter is the downstream report evaluation and is
    never consulted by the construction gate.
    """
    plan = DSO.make_support_plan(
        support, old_rows, cfg.support_batch, cfg.plan_seed,
        support_class_id=support_class_id, stratify_new=True)
    support_adj = DSO.fixed_norm_adj(
        support['edge_index'], len(support['x']), device)
    candidate_adj = DSO.fixed_norm_adj(
        candidate['edge_index'], len(candidate['x']), device)
    selected = torch.arange(
        int(old_rows), len(candidate['x']), dtype=torch.long, device=device)
    rows = []
    for seed in cfg.validation_seeds:
        unrolled = DSO.unroll_student(
            candidate['x'].to(device), candidate['text_emb'].to(device),
            candidate_adj,
            int(seed), cfg, False)
        values = DSO.support_losses(
            unrolled['params'], support, support_adj, plan, 'validation', cfg,
            support_class_id=support_class_id, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids)
        margin = DSO.candidate_class_margin_loss(
            unrolled['params'], candidate, candidate_adj, selected,
            base_slot_class_id, prompt_emb, prompt_ids,
            float(cfg.candidate_prompt_margin))
        rows.append({
            'seed': int(seed),
            'support_loss': float(values['total'].detach()),
            'old_loss': float(values.get('old', values['total']).detach()),
            'new_loss': float(values.get('new', values['total']).detach()),
            'new_margin': float(margin['candidate_prompt_margin'].detach()),
        })
    if not rows:
        raise AssertionError('topology gate has no validation seeds')
    return {
        'seeds': rows,
        'support_loss': float(sum(row['support_loss'] for row in rows)
                              / len(rows)),
        'old_loss': float(sum(row['old_loss'] for row in rows) / len(rows)),
        'new_loss': float(sum(row['new_loss'] for row in rows) / len(rows)),
        'new_margin': float(sum(row['new_margin'] for row in rows)
                            / len(rows)),
    }


def select_topology_candidate(revised: dict, frozen: dict, support: dict,
                              old_rows: int, cfg: DSO.Config, device: str,
                              support_class_id: torch.Tensor,
                              prompt_emb: torch.Tensor,
                              prompt_ids: torch.Tensor,
                              base_slot_class_id: torch.Tensor,
                              min_support_gain: float,
                              min_margin_gain: float,
                              support_tolerance: float,
                              margin_tolerance: float,
                              old_tolerance: float) -> tuple[dict, dict]:
    """Choose revision or frozen topology using only sealed response probes."""
    revision = sealed_validation_response(
        revised, support, old_rows, cfg, device, support_class_id,
        prompt_emb, prompt_ids, base_slot_class_id)
    control = sealed_validation_response(
        frozen, support, old_rows, cfg, device, support_class_id,
        prompt_emb, prompt_ids, base_slot_class_id)
    delta = {
        'support_loss': revision['support_loss'] - control['support_loss'],
        'old_loss': revision['old_loss'] - control['old_loss'],
        'new_loss': revision['new_loss'] - control['new_loss'],
        'new_margin': revision['new_margin'] - control['new_margin'],
    }
    old_safe = delta['old_loss'] <= float(old_tolerance)
    support_win = (
        delta['support_loss'] <= -float(min_support_gain)
        and delta['new_margin'] <= float(margin_tolerance))
    margin_win = (
        delta['new_margin'] <= -float(min_margin_gain)
        and delta['support_loss'] <= float(support_tolerance))
    accepted = bool(old_safe and (support_win or margin_win))
    selected = revised if accepted else frozen
    decision = {
        'name': 'sealed-validation-topology-fallback-v1',
        'source': 'sealed_validation_support_response_only',
        'uses_report_or_test_labels': False,
        'accepted_revision': accepted,
        'selected_topology': 'cap=1' if accepted else 'cap=0',
        'revision': revision,
        'cap0_control': control,
        'delta_revision_minus_cap0': delta,
        'rules': {
            'old_loss_tolerance': float(old_tolerance),
            'min_support_gain': float(min_support_gain),
            'min_new_margin_gain': float(min_margin_gain),
            'support_loss_tolerance': float(support_tolerance),
            'new_margin_tolerance': float(margin_tolerance),
            'old_response_safe': old_safe,
            'support_win': support_win,
            'new_margin_win': margin_win,
        },
    }
    return clone_memory(selected), decision


def run_dataset(dataset: str, args: argparse.Namespace) -> None:
    source = args.source_root / f'{dataset}_base4'
    # Accept both a shared source root and a dataset-specific source root.
    # This keeps the runner compatible with the immutable per-dataset source
    # artifacts used by the large-graph experiments.
    if not (source / 'results.json').exists():
        source = args.source_root
    source_manifest = json.loads((source / 'results.json').read_text())
    # Some historical results put a string protocol identifier at the top
    # level, while newer artifacts may put a dict under ``protocol``.  Only
    # unwrap the latter; a string is metadata, not the manifest itself.
    nested_protocol = source_manifest.get('protocol')
    if isinstance(nested_protocol, dict):
        source_manifest = nested_protocol
    protocol_dict = nested_protocol if isinstance(nested_protocol, dict) else {}
    groups = [[int(v) for v in group]
              for group in source_manifest.get(
                  'stage_classes', source_manifest.get('groups', []))]
    budgets = [int(v) for v in source_manifest.get('budgets', [])]
    if not budgets:
        budgets = [len(load(source / f'xpa_s{s}_memory.pt')['x'])
                   for s in range(5)]

    def source_memory(stage: int) -> dict:
        append = source / f'append_fixed_s{stage}_memory.pt'
        xpa = source / f'xpa_s{stage}_memory.pt'
        return load(append if append.exists() else xpa)
    data_path = U.DATA_CFG[dataset]['path']
    data = U.load_unified(dataset, data_path, args.minilm, args.device,
                          n_stages=5, split_mode='base_incremental',
                          base_fraction=0.4,
                          report_dev_mod=int(source_manifest.get(
                              'report_dev_mod',
                              protocol_dict.get(
                                  'report_dev_mod', 3))))
    data['stage_classes'] = groups
    cumulative_groups = []
    seen = []
    for group in groups:
        seen.extend(group)
        cumulative_groups.append(list(seen))
    U.G.STAGE_CLASSES = cumulative_groups
    # ``eval_stagewise`` reads the legacy module-level table directly.
    # Keep it synchronized after replacing the loader's default split.
    try:
        import run_gate1_continual as LEGACY
        LEGACY.STAGE_CLASSES = cumulative_groups
    except ImportError:
        pass
    data = L.set_pool_text_embeddings(data, source, args.output_root)
    data['prompt_emb'] = data['encoder'].encode(data['prompts'])
    data['real_adj'] = U.G.norm_adj(data['edge_index'], len(data['x']),
                                     args.device)
    data['x_device'] = data['x'].float().to(args.device)
    pool = data['pool_mask'].cpu()
    ids_by_stage = []
    for classes in groups:
        ids = torch.where(pool & torch.isin(
            data['y'].cpu(), torch.tensor(classes, dtype=torch.long)))[0]
        ids_by_stage.append(ids)
    expected = source_manifest.get('stage_train_counts')
    if expected is None:
        expected = [int(v) for v in source_manifest.get('stage_counts', [])]
    if not expected:
        expected = [len(ids) for ids in ids_by_stage]
    if [len(ids) for ids in ids_by_stage] != [int(v) for v in expected]:
        raise AssertionError('large source stage counts do not match loader')

    out = args.output_root / f'{dataset}_base4'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = frozen_config(args.plan_seed, args.old_graph_anchor_weight,
                        args.old_graph_anchor_fraction, args.new_loss_scale,
                        args.a_max_revisions_per_row)
    gate_cfg = (replace(
        cfg,
        inner_steps=int(args.gate_inner_steps),
        validation_seeds=tuple(int(seed)
                               for seed in args.gate_validation_seeds))
        if args.topology_gate else cfg)
    protocol = {
        'name': ('bounded-student-compatible-xpa-topology-gated-large-v1'
                 if args.topology_gate
                 else 'bounded-student-compatible-xpa-large-v1'),
        'dataset': dataset, 'stage_protocol': 'base+4',
        'groups': groups, 'budgets': budgets,
        'student': 'original two-layer GCN + trainable text projection + bidirectional InfoNCE',
        'student_seeds': args.student_seeds,
        'student_graph_mode': args.student_graph_mode,
        'construction': 'current pool delta only; no historical raw reads',
        'topology': 'current-stage C0 topology plus bounded A revision',
        'method': args.mode, 'frozen_config': cfg.dict(),
        'memory_rate': 0.05, 'report_dev_mod': int(source_manifest.get(
            'report_dev_mod', protocol_dict.get(
                'report_dev_mod', 3))),
        'construction_seed': int(args.plan_seed),
        'deterministic_torch': bool(args.deterministic),
        'topology_gate': {
            'enabled': bool(args.topology_gate),
            'validation_seeds': list(gate_cfg.validation_seeds),
            'inner_steps': int(gate_cfg.inner_steps),
            'selection': 'sealed support response; no report/test labels',
            'min_support_gain': float(args.gate_min_support_gain),
            'min_new_margin_gain': float(args.gate_min_margin_gain),
            'support_loss_tolerance': float(args.gate_support_tolerance),
            'new_margin_tolerance': float(args.gate_margin_tolerance),
            'old_loss_tolerance': float(args.gate_old_tolerance),
        },
    }
    save_json(out / 'protocol.json', protocol)
    parent = None
    rows = []

    def persist_partial() -> None:
        save_json(out / 'results.partial.json',
                  {'protocol': protocol, 'rows': rows,
                   'complete': False})

    for stage in range(args.max_stage + 1):
        c0 = source_memory(stage)
        if len(c0['x']) != budgets[stage]:
            raise AssertionError('large C0 budget mismatch')
        if stage == 0:
            candidate = clone_memory(c0)
            audit = {'stage': 0, 'mode': 'sealed_c0_base',
                     'historical_raw_reads': 0, 'report_labels_used': False}
        else:
            old_rows = len(parent['x'])
            delta = B.raw_support(data, ids_by_stage[stage], stage)
            support = B.concat_support(parent, delta, stage)
            base = compose_with_parent(c0, parent)
            class_id = torch.full((len(support['x']),), -1, dtype=torch.long)
            class_id[old_rows:] = data['y'].cpu()[ids_by_stage[stage]].long()
            slot_class, slot_audit = CLASS.build_slot_classes(
                base, support, old_rows, class_id)
            prompt_emb = data['encoder'].encode(data['prompts']).cpu()
            prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)
            stage_out = out / f'stage{stage}'
            stage_out.mkdir(parents=True, exist_ok=False)
            torch.save(base, stage_out / 'base.pt')
            torch.save(support, stage_out / 'support.pt')
            torch.save(class_id, stage_out / 'support_class_id.pt')
            torch.save(slot_class, stage_out / 'base_slot_class_id.pt')
            save_json(stage_out / 'slot_audit.json', slot_audit)
            started = time.perf_counter()
            revised_candidate, audit = DSO.optimize(
                base, support, old_rows,
                list(range(old_rows, len(base['x']))),
                mode=args.mode, device=args.device, cfg=cfg,
                support_class_id=class_id, prompt_emb=prompt_emb,
                prompt_ids=prompt_ids, base_slot_class_id=slot_class)
            candidate = revised_candidate
            topology_gate = None
            if args.topology_gate:
                # Keep the joint X/P result fixed and change only the
                # materialized topology.  The control therefore isolates A
                # instead of re-optimizing X/P under a second objective.
                frozen_candidate = clone_memory(revised_candidate)
                frozen_candidate['edge_index'] = base['edge_index'].clone()
                candidate, topology_gate = select_topology_candidate(
                    revised_candidate, frozen_candidate, support, old_rows,
                    gate_cfg, args.device, class_id, prompt_emb, prompt_ids,
                    slot_class, args.gate_min_support_gain,
                    args.gate_min_margin_gain, args.gate_support_tolerance,
                    args.gate_margin_tolerance, args.gate_old_tolerance)
                torch.save(revised_candidate,
                           stage_out / 'candidate_cap1_revision.pt')
                torch.save(frozen_candidate,
                           stage_out / 'candidate_cap0_control.pt')
                save_json(stage_out / 'topology_gate.json', topology_gate)
            audit.update({'stage': stage,
                          'construction_seconds_wall': time.perf_counter() - started,
                          'historical_raw_reads': 0,
                          'report_labels_used': False,
                          'slot_audit': slot_audit,
                          'topology_gate': topology_gate,
                          'selected_output_sha256': DSO.memory_hash(candidate)})
            save_json(stage_out / 'audit.json', audit)
            torch.save(candidate, stage_out / 'memory.pt')
        parent = candidate
        torch.save(candidate, out / f'xpa_s{stage}_memory.pt')
        student_args = SimpleNamespace(student_seeds=args.student_seeds,
                                        student_batch=16,
                                        student_steps=args.student_steps)
        started = time.perf_counter()
        if args.student_graph_mode == 'edge-aware':
            summary = run_students_with_saved_edges(
                candidate, stage, data, student_args, args.device)
        else:
            summary = U.run_students(candidate, stage, data, student_args,
                                     args.device)
        scored = score_summary(summary, stage)
        summary_for_row = summary
        del summary
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
        row = {'dataset': dataset, 'stage': stage,
               'memory_rows': len(candidate['x']),
               'aggregate': scored, 'summary': summary_for_row,
               'student_wall_seconds': time.perf_counter() - started,
               'construction_audit': audit}
        rows.append(row)
        persist_partial()
        print(json.dumps({'event': 'large_stage_complete',
                          'dataset': dataset, 'stage': stage,
                          'final_f1': scored['mean_f1'],
                          'student_seconds': row['student_wall_seconds']}),
              flush=True)
    save_json(out / 'results.json', {'protocol': protocol, 'rows': rows,
                                     'complete': True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['arxiv', 'bookchild', 'sportsfit'])
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--minilm', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-stage', type=int, default=1)
    ap.add_argument('--student-steps', type=int, default=None)
    ap.add_argument('--student-seeds', nargs='+', type=int, default=[100, 101])
    ap.add_argument('--student-graph-mode', choices=('legacy', 'edge-aware'),
                    default='legacy')
    ap.add_argument('--plan-seed', type=int, default=9320)
    ap.add_argument('--old-graph-anchor-weight', type=float, default=0.0)
    ap.add_argument('--old-graph-anchor-fraction', type=float, default=1.0)
    ap.add_argument('--new-loss-scale', type=float, default=1.0)
    ap.add_argument('--mode', default='x-class+p+a',
                    choices=('x-class', 'x-class+p', 'x-class+p+a'))
    ap.add_argument('--a-max-revisions-per-row', type=int, default=1)
    ap.add_argument('--topology-gate', action='store_true',
                    help='select cap=1 versus cap=0 with sealed validation response')
    ap.add_argument('--gate-min-support-gain', type=float, default=0.001)
    ap.add_argument('--gate-min-margin-gain', type=float, default=0.001)
    ap.add_argument('--gate-support-tolerance', type=float, default=0.002)
    ap.add_argument('--gate-margin-tolerance', type=float, default=0.002)
    ap.add_argument('--gate-old-tolerance', type=float, default=0.0)
    ap.add_argument('--gate-inner-steps', type=int, default=8)
    ap.add_argument('--gate-validation-seeds', nargs='+', type=int,
                    default=[9310])
    ap.add_argument('--deterministic', action='store_true',
                    help='enable deterministic PyTorch algorithms for audit runs')
    args = ap.parse_args()
    if args.deterministic:
        # Keep this opt-in: historical runs used the original CUDA kernels.
        # The flag makes repeated baseline/candidate comparisons auditable.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for dataset in args.datasets:
        run_dataset(dataset, args)
    print(json.dumps({'event': 'complete', 'datasets': args.datasets,
                      'max_stage': args.max_stage,
                      'output': str(args.output_root)}), flush=True)


if __name__ == '__main__':
    main()
