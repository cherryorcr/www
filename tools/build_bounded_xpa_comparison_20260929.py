"""Merge the frozen unified small-graph baselines with Bounded X-P-A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--cora', type=Path, required=True)
    ap.add_argument('--citeseer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    roots = {'cora': args.cora, 'citeseer': args.citeseer}
    rows = []
    for dataset in ('cora', 'citeseer'):
        for order in ('order19611', 'order19612'):
            old = [row for row in baseline['rows']
                   if row['dataset'] == dataset and row['order'] == order]
            by_arm = {}
            for arm in baseline['protocol']['arms']:
                stages = sorted([row for row in old if row['arm'] == arm],
                                key=lambda row: row['stage'])
                if not stages:
                    continue
                final = stages[-1]['aggregate']
                by_arm[arm] = {
                    'dataset': dataset, 'order': order, 'arm': arm,
                    'final_f1': final['mean_f1'],
                    'final_acc': final['mean_accuracy'],
                    'stage_avg_f1': sum(row['aggregate']['mean_f1']
                                        for row in stages) / len(stages),
                    'stage_avg_acc': sum(row['aggregate']['mean_accuracy']
                                         for row in stages) / len(stages),
                    'old_f1': final.get('old_f1'),
                    'new_f1': final.get('new_f1'),
                    'final_rows': stages[-1]['memory_rows'],
                    'student_seconds': sum(row['student_wall_seconds']
                                           for row in stages),
                    'construction_seconds': None,
                }
            fresh = json.loads(
                (roots[dataset] / f'{dataset}_{order}' / 'results.json')
                .read_text(encoding='utf-8'))
            stages = sorted(fresh['rows'], key=lambda row: row['stage'])
            final = stages[-1]['aggregate']
            construction = sum(float(row.get('construction_audit', {})
                                    .get('construction_seconds_wall', 0.0))
                               for row in stages)
            by_arm['bounded_xpa'] = {
                'dataset': dataset, 'order': order, 'arm': 'bounded_xpa',
                'final_f1': final['mean_f1'],
                'final_acc': final['mean_accuracy'],
                'stage_avg_f1': sum(row['aggregate']['mean_f1']
                                    for row in stages) / len(stages),
                'stage_avg_acc': sum(row['aggregate']['mean_accuracy']
                                     for row in stages) / len(stages),
                'old_f1': final.get('old_f1'),
                'new_f1': final.get('new_f1'),
                'final_rows': stages[-1]['memory_rows'],
                'student_seconds': sum(row['student_wall_seconds']
                                       for row in stages),
                'construction_seconds': construction,
            }
            c0 = by_arm['c0']
            for row in by_arm.values():
                row['delta_final_vs_c0_pp'] = row['final_f1'] - c0['final_f1']
                row['delta_stage_avg_vs_c0_pp'] = (
                    row['stage_avg_f1'] - c0['stage_avg_f1'])
                row['delta_final_acc_vs_c0_pp'] = (
                    row['final_acc'] - c0['final_acc'])
                rows.append(row)

    summary = {}
    for arm in sorted(set(row['arm'] for row in rows)):
        group = [row for row in rows if row['arm'] == arm]
        keys = ('final_f1', 'final_acc', 'stage_avg_f1', 'stage_avg_acc',
                'delta_final_vs_c0_pp', 'delta_stage_avg_vs_c0_pp',
                'delta_final_acc_vs_c0_pp')
        summary[arm] = {
            key: sum(float(row[key]) for row in group) / len(group)
            for key in keys if all(row.get(key) is not None for row in group)
        }
        summary[arm]['positive_final_cells'] = sum(
            row['delta_final_vs_c0_pp'] > 0 for row in group)
        summary[arm]['cells'] = len(group)
    xpa = [row for row in rows if row['arm'] == 'bounded_xpa']
    summary['bounded_xpa']['min_final_delta_vs_c0_pp'] = min(
        row['delta_final_vs_c0_pp'] for row in xpa)
    summary['bounded_xpa']['max_final_delta_vs_c0_pp'] = max(
        row['delta_final_vs_c0_pp'] for row in xpa)

    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        'name': 'bounded-xpa-unified-comparison-v1',
        'protocol_source': str(args.baseline),
        'new_protocol_roots': {key: str(value) for key, value in roots.items()},
        'rows': rows,
        'summary_by_arm': summary,
        'interpretation': {
            'same_student_seeds': True,
            'same_memory_budgets': True,
            'c0_internal_anchor': True,
            'stateful_arms_not_direct_memory_only_comparators': True,
            'static_tagsam_not_in_this_table': True,
        },
    }
    (args.output / 'comparison.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding='utf-8')
    lines = [
        '# P1 Unified Small-Graph Comparison', '',
        '| Dataset | Order | Arm | Final F1 | Final Acc | Stage-avg F1 | Stage-avg Acc | Old F1 | New F1 | Delta F1 vs C0 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"|{row['dataset']}|{row['order']}|{row['arm']}|"
            f"{row['final_f1']:.2f}|{row['final_acc']:.2f}|"
            f"{row['stage_avg_f1']:.2f}|{row['stage_avg_acc']:.2f}|"
            f"{row['old_f1'] if row['old_f1'] is not None else 0:.2f}|"
            f"{row['new_f1'] if row['new_f1'] is not None else 0:.2f}|"
            f"{row['delta_final_vs_c0_pp']:+.2f}|")
    lines.extend(['', '## Bounded X-P-A Summary', '',
                  '| Mean over four cells | Value |', '|---|---:|'])
    for key, value in summary['bounded_xpa'].items():
        if isinstance(value, (int, float)):
            lines.append(f'|{key}|{value:.3f}|')
    lines.extend(['', 'The X-P-A arm uses the same Student, seeds, budgets, class orders, and report-dev rule as the existing unified baseline. Stateful Student arms retain model parameters and are separate controls. Static TAG baselines are not included in this initial unified table.'])
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n',
                                               encoding='utf-8')
    print(json.dumps({'output': str(args.output),
                      'rows': len(rows),
                      'xpa_summary': summary['bounded_xpa']}, indent=2))


if __name__ == '__main__':
    main()
