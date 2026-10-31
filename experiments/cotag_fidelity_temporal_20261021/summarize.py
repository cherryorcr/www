"""Source-backed tables; never mix seeds, masks, or implementation variants."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def read(path):
    return json.loads(path.read_text())


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def interval(values):
    n = len(values)
    if n < 2:
        return None
    mean = float(np.mean(values))
    half = float(student_t.ppf(.975, n-1)*np.std(values, ddof=1)/np.sqrt(n))
    return [mean-half, mean+half]


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root):
    status, temporal, tagsam, native = [], [], [], []
    for path in sorted(root.rglob('results.json')):
        if 'native_student_diagnostic' in path.relative_to(root).parts:
            continue
        value = read(path)
        protocol = value['protocol']
        invalid = (root/'precision_repair'/'INVALID_ATTEMPT.json').exists() and 'precision_repair' in path.relative_to(root).parts
        status.append(dict(path=str(path.relative_to(root)), status=value['status'],
                           completed_stages=len(value['rows']), error=value.get('error'), invalid_intervention=invalid))
        if invalid:
            continue
        if 'arxiv-publication-year' in protocol['name']:
            variant = protocol['arm']
            if 'numerical_repair_hash' in protocol:
                variant += '_float64'
            for row in value['rows']:
                for mode, splits in row['scores']['prediction'].items():
                    for split, score in splits.items():
                        for cohort in ('cumulative', 'current', 'historical'):
                            metric = score[cohort]
                            if metric is None:
                                continue
                            temporal.append(dict(variant=variant, construction_seed=protocol['construction_seed'],
                                run_complete=value['status']=='complete', stage=row['stage'],
                                mode=mode, split=split, cohort=cohort, **metric,
                                construction_seconds=row['construction']['seconds'],
                                construction_peak_mb=row['construction']['peak_mb'],
                                memory_rows=row['construction']['memory_rows'],
                                source=str(path.relative_to(root))))
        else:
            for row in value['rows']:
                for split, score in row['scores'].items():
                    result = dict(dataset=protocol['dataset'], outer_steps=protocol['outer_steps'],
                        teacher_budget=protocol.get('teacher_updates', '3_epochs' if 'native-tagsam' in protocol['name'] else '2048_proxy_updates'),
                        stage=row['stage'], split=split, run_complete=value['status']=='complete',
                        f1=score['final_f1'], f1_student_sd=score['final_f1_std'],
                        accuracy=score['accuracy'], accuracy_student_sd=score['accuracy_std'],
                        old_f1=score['old_f1'], new_f1=score['new_f1'],
                        source=str(path.relative_to(root)))
                    (native if 'native-tagsam' in protocol['name'] else tagsam).append(result)
    return status, temporal, tagsam, native


def group_temporal(rows):
    groups = {}
    for row in rows:
        if not row['run_complete']:
            continue
        key = tuple(row[k] for k in ('variant', 'stage', 'mode', 'split', 'cohort'))
        groups.setdefault(key, []).append(row)
    out = []
    for key, values in groups.items():
        if len({v['construction_seed'] for v in values}) != len(values):
            raise AssertionError('A construction seed was counted twice')
        out.append(dict(zip(('variant', 'stage', 'mode', 'split', 'cohort'), key),
            constructions=len(values), f1=float(np.mean([r['f1'] for r in values])),
            f1_construction_sd=sample_sd([r['f1'] for r in values]),
            accuracy=float(np.mean([r['accuracy'] for r in values])),
            accuracy_construction_sd=sample_sd([r['accuracy'] for r in values])))
    return out


def stage_average_temporal(rows):
    groups = {}
    keys = ('variant', 'construction_seed', 'mode', 'split', 'cohort')
    for row in rows:
        if not row['run_complete'] or (row['mode'], row['cohort']) not in (
                ('arrived_snapshot', 'cumulative'), ('current_snapshot', 'current')):
            continue
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, values in groups.items():
        if sorted(row['stage'] for row in values) != list(range(5)):
            raise AssertionError('Stage average needs five unique completed stages')
        out.append(dict(zip(keys, key), stage='five_stage_mean', run_complete=True,
            f1=float(np.mean([row['f1'] for row in values])),
            accuracy=float(np.mean([row['accuracy'] for row in values]))))
    return out


def paired_temporal(rows, stage=4):
    pairs = []
    for mode in ('arrived_snapshot', 'current_snapshot'):
        for cohort in ('cumulative', 'current', 'historical'):
            for split in ('report_dev', 'historical_test'):
                select = [r for r in rows if r['stage'] == stage and r['run_complete'] and
                          (r['mode'], r['cohort'], r['split']) == (mode, cohort, split)]
                for main in ('uq_dm_edgefree', 'uq_dm_edgefree_float64'):
                    left = {r['construction_seed']: r for r in select if r['variant'] == main}
                    for baseline in ('cat_cctp', 'gecc_hard'):
                        right = {r['construction_seed']: r for r in select if r['variant'] == baseline}
                        seeds = sorted(left.keys() & right.keys())
                        if not seeds:
                            continue
                        differences = [left[s]['f1']-right[s]['f1'] for s in seeds]
                        pairs.append(dict(main=main, baseline=baseline, mode=mode, cohort=cohort,
                            split=split, seeds=seeds, n=len(seeds),
                            difference_pp=float(np.mean(differences)), ci95=interval(differences),
                            positive_count=sum(v > 0 for v in differences), worst_difference_pp=min(differences),
                            per_construction_differences=differences))
    return pairs


def fmt(value, sd=None):
    if value is None:
        return 'NA'
    return f'{value:.2f}' if sd is None else f'{value:.2f} +/- {sd:.2f}'


def split_label(value):
    return 'historical_test' if value == 'sealed_test' else value


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    status, temporal, tagsam, native = collect(root)
    grouped, paired = group_temporal(temporal), paired_temporal(temporal)
    stage_runs = stage_average_temporal(temporal)
    stage_summary = group_temporal(stage_runs)
    stage_pairs = paired_temporal(stage_runs, stage='five_stage_mean')
    readouts = []
    for path in sorted(root.rglob('native_student_diagnostic/results.json')):
        value = read(path)
        if value['status'] != 'complete':
            continue
        protocol = value['protocol']
        for split, score in value['summary'].items():
            readouts.append(dict(dataset=protocol['dataset'],
                teacher_budget=protocol['construction_protocol'].get('teacher_updates', '3_epochs'),
                split=split, **score, source=str(path.relative_to(root)),
                teacher_f1=value['teacher_check']['scores'][split]['macro_f1'],
                teacher_updates=value['teacher_check']['training']['updates']))
    for name, rows in (('status', status), ('temporal_runs', temporal),
                       ('temporal_summary', grouped), ('tagsam_budgets', tagsam),
                       ('tagsam_native_epochs', native), ('native_readouts', readouts),
                       ('temporal_stage_average_runs', stage_runs),
                       ('temporal_stage_average_summary', stage_summary)):
        write_csv(out/f'{name}.csv', rows)
    (out/'paired_intervals.json').write_text(json.dumps(paired, indent=2))
    (out/'temporal_stage_average_pairs.json').write_text(json.dumps(stage_pairs, indent=2))
    (out/'summary.json').write_text(json.dumps(dict(status=status, temporal=grouped,
        paired=paired, stage_average=stage_summary, stage_average_pairs=stage_pairs,
        tagsam=tagsam, native=native, native_readouts=readouts), indent=2))
    lines = ['# Baseline fidelity and chronological evaluation', '',
        'This report is generated from result files. Incomplete chains are excluded from final comparisons.', '',
        '## Coverage', '', '| Output | Status | Completed stages |', '|---|---|---|']
    lines += [f"| {r['path']} | {'invalid/excluded' if r['invalid_intervention'] else r['status']} | {r['completed_stages']} |" for r in status]
    lines += ['', '## Chronological ArXiv, Final Stage', '',
        'Construction means average three student seeds. SD is across completed construction seeds.',
        'Float64 denotes verified float64 normalization plus a solver cap of 16384, not the original implementation.', '',
        '| Variant | Inference | Cohort | Split | n constructions | Macro-F1 | Accuracy |',
        '|---|---|---|---|---:|---:|---:|']
    for r in grouped:
        if r['stage'] != 4 or (r['mode']=='current_snapshot' and r['cohort']!='current'):
            continue
        lines.append(f"| {r['variant']} | {r['mode']} | {r['cohort']} | {r['split']} | {r['constructions']} | "
                     f"{fmt(r['f1'], r['f1_construction_sd'])} | {fmt(r['accuracy'], r['accuracy_construction_sd'])} |")
    lines += ['', '## Chronological ArXiv, Five-Stage Average', '',
        'Unweighted mean of five stage scores within each construction, then mean and SD across constructions.',
        'This descriptive trajectory summary is separate from final-stage scores.', '',
        '| Variant | Inference | Cohort | Split | n constructions | Macro-F1 | Accuracy |',
        '|---|---|---|---|---:|---:|---:|']
    for r in stage_summary:
        lines.append(f"| {r['variant']} | {r['mode']} | {r['cohort']} | {r['split']} | {r['constructions']} | "
                     f"{fmt(r['f1'], r['f1_construction_sd'])} | {fmt(r['accuracy'], r['accuracy_construction_sd'])} |")
    lines += ['', '## TAGSAM Complete-Chain Budget Diagnostic', '',
        'These are minibatch-unroll adapters, not official full-epoch runs. SD is over students for one construction.', '',
        '| Dataset | Outer updates | Split | Macro-F1 | Accuracy |', '|---|---:|---|---:|---:|']
    for r in tagsam:
        last = 2 if r['dataset']=='arxiv' else 4
        if not r['run_complete'] or r['stage'] != last:
            continue
        lines.append(f"| {r['dataset']} | {r['outer_steps']} | {split_label(r['split'])} | {fmt(r['f1'], r['f1_student_sd'])} | {fmt(r['accuracy'], r['accuracy_student_sd'])} |")
    lines += ['', '## Native-Epoch Construction Adaptation', '',
        'Native inner-epoch function, model, BERT, weighted BCE and schedule; common downstream student.',
        'This is not an unchanged official end-to-end reproduction.', '',
        '| Dataset | Teacher budget | Outer updates | Stage | Split | Macro-F1 | Accuracy |', '|---|---|---:|---:|---|---:|---:|']
    for r in native:
        lines.append(f"| {r['dataset']} | {r['teacher_budget']} | {r['outer_steps']} | {r['stage']} | {split_label(r['split'])} | {fmt(r['f1'], r['f1_student_sd'])} | {fmt(r['accuracy'], r['accuracy_student_sd'])} |")
    lines += ['', '## Separate Native-Student Diagnostic', '',
        'Same final memory, BERT/native model, learned rates, fixed 15 epochs; not the common-student budget.', '',
        '| Dataset | Teacher budget | Split | Macro-F1 | Accuracy | Reconstructed teacher F1 |',
        '|---|---|---|---:|---:|---:|']
    for r in readouts:
        lines.append(f"| {r['dataset']} | {r['teacher_budget']} | {split_label(r['split'])} | {fmt(r['macro_f1'], r['macro_f1_sd'])} | {fmt(r['accuracy'], r['accuracy_sd'])} | {r['teacher_f1']:.2f} |")
    lines += ['', '## Interpretation Boundaries', '',
        '- Publication time is observed; edge availability is assumed at the later endpoint arrival.',
        '- The chronological stream has recurring classes. Arrival-cohort retention is not new-class forgetting.',
        '- Arrived-snapshot inference can read historical nodes; constructors cannot. Current-snapshot inference cannot score old queries.',
        '- Original numerical failures remain part of the evidence. No failed run is treated as a zero score.',
        '- Historical masks have been inspected previously. These are protocol-frozen diagnostics, not new blind benchmarks.',
        '- Co-located run timings are descriptive, not an isolated efficiency comparison.',
        '- A small-graph native-core result does not close the large-graph official baseline fidelity gap.', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(results=len(status), complete=sum(r['status']=='complete' for r in status),
                         failed=sum(r['status']=='failed' for r in status), output=str(out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.root, args.output or args.root/'analysis')
