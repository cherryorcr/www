"""Matched reference extraction, inference-boundary contrasts and resource tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from summarize import collect, interval, read, write_csv


def compare_protocols(left, right):
    fields = ('groups', 'budgets', 'stage_id_hashes', 'pool_mask_hash',
              'report_mask_hash', 'test_mask_hash', 'student_seeds',
              'student_steps', 'student_batch', 'order_seed', 'construction_seed')
    for field in fields:
        assert left[field] == right[field], field
    return fields


def extract_reference(path, comparator, output):
    source, other = read(path), read(comparator)
    checks = compare_protocols(source['protocol'], other['protocol'])
    assert source['status'] == 'complete'
    compact = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   protocol=source['protocol'], comparator=str(comparator),
                   equality_checked=list(checks), rows=[])
    final = len(source['protocol']['budgets'])-1
    for row in source['rows']:
        if row['stage'] != final:
            continue
        compact['rows'].append(dict(arm=row['arm'], stage=final, scores=row['scores']))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2), encoding='utf-8')
    return compact


def main(root):
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    refs = []
    for dataset in ('arxiv', 'sportsfit', 'bookhis'):
        reference = root.parent/f'uq_final_evidence_20261012/confirmation/{dataset}_o9101_c271006/results.json'
        comparator = root/f'tagsam/{dataset}_o9101_c271006/outer100_updates10/results.json'
        if comparator.exists():
            refs.append(extract_reference(reference, comparator, root/f'references/{dataset}_matched.json'))
    comparator = root/'native_epochs/cora_outer5000/results.json'
    if comparator.exists():
        reference = root.parent/'cotag_q_missing_baselines_20261019_repair_v2/cora/tagsam_native_bounded/results.json'
        refs.append(extract_reference(reference, comparator, root/'references/cora_bounded.json'))
    reference_rows, work = [], []
    for ref in refs:
        for stage, size in enumerate(ref['protocol']['budgets']):
            work.append(dict(dataset=ref['protocol']['dataset'], stage=stage, memory_rows=size,
                native_inner_updates_per_outer=15*(size//20),
                native_updates_for_5000_outer=5000*15*(size//20),
                proxy_inner_updates_per_outer=15, proxy_updates_for_5000_outer=75000,
                note='Analytical update counts, not measured runtime or OOM evidence'))
        for row in ref['rows']:
            for split, score in row['scores'].items():
                reference_rows.append(dict(dataset=ref['protocol']['dataset'], arm=row['arm'],
                    split=split, f1=score['final_f1'], f1_student_sd=score['final_f1_std'],
                    accuracy=score['accuracy'], accuracy_student_sd=score['accuracy_std'],
                    source=ref['source'], comparison_hashes_equal=True))
    write_csv(out/'matched_references.csv', reference_rows)
    write_csv(out/'native_budget_work.csv', work)

    _, temporal, _, _ = collect(root)
    differences, group = [], {}
    selected = [row for row in temporal if row['run_complete'] and row['stage']==4 and row['cohort']=='current']
    by_key = {(row['variant'], row['construction_seed'], row['split'], row['mode']): row for row in selected}
    for (variant, seed, split, mode), row in by_key.items():
        if mode != 'arrived_snapshot':
            continue
        current = by_key[(variant, seed, split, 'current_snapshot')]
        item = dict(variant=variant, seed=seed, split=split,
            f1_current_minus_arrived=current['f1']-row['f1'],
            accuracy_current_minus_arrived=current['accuracy']-row['accuracy'],
            cohort='same final-stage queries, same trained students')
        differences.append(item)
        group.setdefault((variant, split), []).append(item)
    aggregated = []
    for (variant, split), rows in group.items():
        changes = [row['f1_current_minus_arrived'] for row in rows]
        aggregated.append(dict(variant=variant, split=split, n_constructions=len(rows),
            f1_current_minus_arrived=statistics.mean(changes), ci95=interval(changes)))
    write_csv(out/'inference_boundary_pairs.csv', differences)
    (out/'inference_boundary_summary.json').write_text(json.dumps(aggregated, indent=2))

    costs, stage_means = [], []
    for path in sorted(root.rglob('results.json')):
        relative = path.relative_to(root)
        if 'precision_repair' in relative.parts or 'native_student_diagnostic' in relative.parts:
            continue
        result = read(path)
        if result['status'] != 'complete':
            continue
        is_temporal = 'arxiv-publication-year' in result['protocol']['name']
        if is_temporal:
            rows = result['rows']
            costs.append(dict(source=str(relative), dataset=result['protocol']['dataset'],
                construction_seconds=sum(row['construction']['seconds'] for row in rows),
                construction_peak_mib=max(row['construction']['peak_mb'] for row in rows),
                student_train_mean_seconds=statistics.mean(v['train_seconds'] for row in rows for v in row['scores']['training_resources']),
                student_train_peak_mib=max(v['train_peak_mb'] for row in rows for v in row['scores']['training_resources']),
                memory_rows=rows[-1]['construction']['memory_rows'],
                resource_isolation=False))
            for split in ('report_dev', 'historical_test'):
                for mode, cohort in (('arrived_snapshot', 'cumulative'), ('current_snapshot', 'current')):
                    values = [row['scores']['prediction'][mode][split][cohort] for row in rows]
                    stage_means.append(dict(source=str(relative), split=split, mode=mode, cohort=cohort,
                        f1_mean_over_stages=statistics.mean(v['f1'] for v in values),
                        accuracy_mean_over_stages=statistics.mean(v['accuracy'] for v in values)))
        else:
            rows = result['rows']
            costs.append(dict(source=str(relative), dataset=result['protocol']['dataset'],
                response_seconds=sum(row['construction']['response_seconds'] for row in rows),
                selector_seconds=sum(row['construction']['selector_seconds'] for row in rows),
                teacher_seconds=sum(row['construction']['teacher_seconds'] for row in rows),
                response_peak_mib=max(row['construction']['response_peak_mb'] for row in rows),
                memory_rows=result['protocol']['budgets'][-1], resource_isolation=False))
    write_csv(out/'resource_descriptive.csv', costs)
    write_csv(out/'temporal_stage_means.csv', stage_means)
    print(json.dumps(dict(matched_reference_files=len(refs), inference_pairs=len(differences),
                         completed_resource_chains=len(costs))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021'))
    main(parser.parse_args().root)
