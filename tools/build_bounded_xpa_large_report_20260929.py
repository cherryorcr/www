"""Build the P2/P3 large-graph report for Bounded X-P-A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def xpa_row(path: Path) -> dict:
    rows = read(path)['rows']
    final = rows[-1]['aggregate']
    return {
        'final_f1': final['mean_f1'],
        'final_accuracy': final['mean_accuracy'],
        'stage_average_f1': sum(row['aggregate']['mean_f1'] for row in rows) / len(rows),
        'stage_average_accuracy': sum(row['aggregate']['mean_accuracy'] for row in rows) / len(rows),
        'old_f1': final.get('old_f1'),
        'new_f1': final.get('new_f1'),
        'construction_seconds': sum(row['construction_audit'].get('construction_seconds_wall', 0.0) for row in rows),
        'stage_f1': [row['aggregate']['mean_f1'] for row in rows],
        'stage_accuracy': [row['aggregate']['mean_accuracy'] for row in rows],
    }


def reference(path: Path) -> dict:
    rows = read(path)['summary']['rows']
    final = rows[-1]
    out = {}
    for name in ('c0', 'semantic_recursive'):
        value = final[name]
        out[name] = {
            'final_f1': value['cum_dev_f1_mean'],
            'final_accuracy': sum(row['cumulative_accuracy'] for row in value['per_student']) / len(value['per_student']),
            # Average the cumulative checkpoint score at each arrival stage.
            # value['per_stage_mean'] is the final-checkpoint score broken
            # down by task stage, which is a different statistic.
            'stage_average_f1': sum(row[name]['cum_dev_f1_mean'] for row in rows) / len(rows),
            'old_f1': value.get('old_mean_f1_mean'),
            'new_f1': value.get('current_f1_mean'),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root9320', type=Path, required=True)
    ap.add_argument('--root-seeds', nargs='+', type=Path, required=True)
    ap.add_argument('--reference-root', type=Path, required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for dataset in args.datasets:
        baseline = reference(args.reference_root / f'{dataset}_base4/results.json')
        dataset_roots = [root for root in [args.root9320, *args.root_seeds]
                         if (root / f'{dataset}_base4/results.json').exists()]
        if not dataset_roots:
            raise FileNotFoundError(f'no XPA results found for {dataset}')
        base_path = dataset_roots[0] / f'{dataset}_base4/results.json'
        base = xpa_row(base_path)
        for arm, value in baseline.items():
            rows.append({'dataset': dataset, 'arm': arm, **value,
                         'delta_final_vs_xpa_pp': value['final_f1'] - base['final_f1']})
        rows.append({'dataset': dataset, 'arm': 'bounded_xpa_seed9320', **base,
                     'delta_final_vs_xpa_pp': 0.0})
        for path in dataset_roots[1:]:
            candidate = path / f'{dataset}_base4/results.json'
            if candidate.exists():
                value = xpa_row(candidate)
                rows.append({'dataset': dataset,
                             'arm': f'bounded_xpa_{path.name}',
                             **value,
                             'delta_final_vs_xpa_pp': value['final_f1'] - base['final_f1']})
    output = {'name': 'bounded-xpa-large-p2-p3-v2', 'rows': rows,
              'protocol': {'base4': True, 'same_student': True,
                            'same_memory_budget': True,
                            'stage_average_definition': 'mean cumulative checkpoint score over arrival stages',
                            'construction_seeds_by_dataset': {
                                dataset: [
                                    int(read(path / f'{dataset}_base4/protocol.json')['construction_seed'])
                                    for path in dataset_roots
                                    if (path / f'{dataset}_base4/protocol.json').exists()
                                ]
                                for dataset in args.datasets
                            },
                            'no_test_selection': True}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding='utf-8')
    lines = ['# Bounded X-P-A P2/P3 Large-Graph Report', '',
             '|Dataset|Arm|Final F1|Final Accuracy|Stage-average F1|Old F1|New F1|Build seconds|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['dataset']}|{row['arm']}|{row['final_f1']:.2f}|{row['final_accuracy']:.2f}|{row['stage_average_f1']:.2f}|{(row['old_f1'] or 0):.2f}|{(row['new_f1'] or 0):.2f}|{row.get('construction_seconds', 0):.2f}|")
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
