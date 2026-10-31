"""Reproducible summaries of the registered four-gap evidence batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

DATASETS = ('arxiv', 'sportsfit', 'bookhis')
NAMES = dict(arxiv='ArXiv', sportsfit='SportsFit', bookhis='BookHis')
SEEDS = (271006, 271108, 271109)
SPLITS = ('report_dev', 'sealed_test')
METRICS = ('final_f1', 'accuracy', 'old_f1', 'new_f1', 'stage_f1')
MATCH = ('groups', 'budgets', 'stage_raw_rows', 'student', 'student_steps',
         'student_batch', 'student_seeds', 'pool_mask_hash', 'report_mask_hash',
         'test_mask_hash', 'stage_id_hashes')


def stats(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise AssertionError('Statistics require a nonempty finite vector')
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(.975, n-1)*sd/np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, sd=sd, ci_low=mean-half if half is not None else None,
        ci_high=mean+half if half is not None else None,
        positive=int((values > 0).sum()), worst=float(values.min()), values=values.tolist())


def validate_protocol(result, primary):
    failures = [key for key in MATCH
                if key not in result['protocol'] or
                result['protocol'][key] != primary['protocol'][key]]
    for key in ('dataset', 'order_seed', 'construction_seed'):
        if result['protocol'].get(key) != primary['protocol'].get(key):
            failures.append(key)
    if failures:
        raise AssertionError(dict(protocol_mismatch=failures))


def validate_score(score, protocol, stage):
    expected = protocol['student_seeds']
    seeds = [row['seed'] for row in score['per_student']]
    if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(expected):
        raise AssertionError('Student seeds are missing, duplicated, or unmatched')
    for key in ('student_steps', 'student_batch', 'student_seeds'):
        if score[key] != protocol[key]:
            raise AssertionError(f'Score protocol differs: {key}')
    for metric in METRICS[:-1]:
        values = [row[metric] for row in score['per_student']]
        if metric == 'old_f1' and stage == 0 and score[metric] is None:
            if any(value is not None for value in values):
                raise AssertionError('Inconsistent missing first-stage old F1')
            continue
        if any(value is None for value in values):
            raise AssertionError(f'Missing metric: {metric}')
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
            raise AssertionError(f'Invalid percentage: {metric}')
        if score[metric] is None or not np.isfinite(score[metric]) or not np.isclose(
                float(values.mean()), score[metric], rtol=0, atol=1e-7):
            raise AssertionError(f'Student mean differs: {metric}')


def validate_row(row, protocol, expected_rows, require_checks=True):
    if row['memory_rows'] != expected_rows:
        raise AssertionError('Memory row budget differs')
    for key in ('budget_ok', 'prefix_ok', 'finite_ok'):
        if require_checks and row.get('checks', {}).get(key) is not True:
            raise AssertionError(f'Missing or failed construction check: {key}')
    if row.get('checks', {}).get('sinkhorn_feasible') is False:
        raise AssertionError('Failed assignment feasibility check')
    for split in SPLITS:
        score = row['scores'][split]
        if score['split'] != split:
            raise AssertionError('Evaluation split identity differs')
        validate_score(score, protocol, row['stage'])


def validate_rows(result, arm=None, final_only=False):
    protocol = result['protocol']
    rows = sorted([row for row in result['rows'] if arm is None or row.get('arm') == arm],
                  key=lambda row: row['stage'])
    stages = [row['stage'] for row in rows]
    expected = ([len(protocol['budgets'])-1] if final_only else
                list(range(len(protocol['budgets']))))
    if stages != expected:
        raise AssertionError(f'Incomplete or duplicated stage sequence: {stages}')
    for row in rows:
        stage = row['stage']
        expected_rows = (protocol['reference_rows'] if final_only else protocol['budgets'][stage])
        validate_row(row, protocol, expected_rows, require_checks=not final_only)
    return rows


def paired_retention(ours, reference, metric, seeds):
    values = []
    for seed in seeds:
        a = float(np.mean([next(row[metric] for row in score['per_student']
                               if row['seed'] == seed) for score in ours]))
        b = next(row[metric] for row in reference['per_student'] if row['seed'] == seed)
        if b <= 0:
            raise AssertionError('Retention reference must be positive')
        values.append(dict(student_seed=seed, ours_mean_over_constructions=a,
                           reference_value=b, retention=100*a/b, difference=a-b))
    return values


def export(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def fmt(x):
    return '--' if x is None else f'{x:.2f}'


def summarize(args):
    root = args.root
    batch = root / 'results/cotag_q_four_gap_20261015_v1'
    sources, pending = [], []
    cache = {}

    def read(path, expected=True):
        if path in cache:
            return cache[path]
        if not path.exists():
            if expected:
                pending.append(dict(path=str(path.relative_to(root)), status='missing'))
            return None
        raw = path.read_bytes()
        result = json.loads(raw)
        sources.append(dict(path=str(path.relative_to(root)),
            sha256=hashlib.sha256(raw).hexdigest(), status=result.get('status')))
        if result.get('status') != 'complete':
            pending.append(dict(path=str(path.relative_to(root)), status=result.get('status')))
            return None
        cache[path] = result
        return result

    text_cells, text_students, summary, contrasts, interactions = [], [], [], [], []
    primary_map = {}
    for ds in DATASETS:
        cells = {}
        for seed in SEEDS:
            primary = read(root / 'results/uq_final_evidence_20261012/confirmation' /
                f'{ds}_o9101_c{seed}/results.json')
            if primary is None:
                raise RuntimeError('Completed primary evidence is missing')
            primary_map[ds, seed] = primary
            inputs = dict(full=(primary, 'uq_dm_edgefree'),
                no_text_init=(read(batch / 'text' / f'{ds}_o9101_c{seed}_no_text_init/results.json'), None),
                no_pair_feedback=(read(batch / 'text' / f'{ds}_o9101_c{seed}_no_pair_feedback/results.json'), None),
                both_off=(read(root / 'results/uq_exact5_no_node_text_20261015' /
                    f'{ds}_o9101_c{seed}/results.json'), None))
            for arm, (result, row_arm) in inputs.items():
                if result is None:
                    continue
                validate_protocol(result, primary)
                rows = validate_rows(result, row_arm)
                for split in SPLITS:
                    score = rows[-1]['scores'][split]
                    row = dict(dataset=ds, construction_seed=seed, arm=arm, split=split,
                        **{key: score[key] for key in METRICS if key != 'stage_f1'},
                        stage_f1=float(np.mean([r['scores'][split]['final_f1'] for r in rows])),
                        construction_seconds=sum(r['construction_seconds'] for r in rows),
                        construction_peak_mb=max(r['construction_peak_mb'] for r in rows))
                    cells[arm, seed, split] = row
                    text_cells.append(row)
                    for s in score['per_student']:
                        text_students.append(dict(dataset=ds, construction_seed=seed,
                            arm=arm, split=split, **s))
        for split in SPLITS:
            for metric in METRICS:
                for arm in ('full', 'no_text_init', 'no_pair_feedback', 'both_off'):
                    selected = [cells[arm, seed, split][metric] for seed in SEEDS
                                if (arm, seed, split) in cells]
                    if len(selected) == 3:
                        summary.append(dict(dataset=ds, split=split, arm=arm, metric=metric, **stats(selected)))
                    if arm != 'full' and len(selected) == 3:
                        differences = [cells['full', seed, split][metric]-cells[arm, seed, split][metric]
                                       for seed in SEEDS]
                        contrasts.append(dict(dataset=ds, split=split, control=arm,
                            metric=metric, orientation='full minus control', **stats(differences)))
                if all((arm, seed, split) in cells for arm in ('full','no_text_init','no_pair_feedback','both_off') for seed in SEEDS):
                    values = [cells['full',s,split][metric]-cells['no_text_init',s,split][metric]
                              -cells['no_pair_feedback',s,split][metric]+cells['both_off',s,split][metric]
                              for s in SEEDS]
                    interactions.append(dict(dataset=ds, split=split, metric=metric,
                        orientation='full - no-init - no-pair + both-off', **stats(values)))

    retention, retention_students = [], []
    for ds in DATASETS:
        for arm in ('full_ef','full_topology'):
            result = read(batch / 'fullhistory' / f'{ds}_o9101_c271006_{arm}/results.json')
            if result is None:
                continue
            validate_protocol(result, primary_map[ds,SEEDS[0]])
            reference_rows = validate_rows(result, final_only=True)
            for split in SPLITS:
                real = reference_rows[0]['scores'][split]
                ours = [validate_rows(primary_map[ds,s], 'uq_dm_edgefree')[-1]['scores'][split] for s in SEEDS]
                for metric in ('final_f1','accuracy','old_f1','new_f1'):
                    pairs = paired_retention(ours, real, metric, result['protocol']['student_seeds'])
                    ratios = [pair['retention'] for pair in pairs]
                    paired = [pair['difference'] for pair in pairs]
                    retention_students.extend(dict(dataset=ds, split=split, reference=arm,
                                                   metric=metric, **pair) for pair in pairs)
                    ours_mean = float(np.mean([o[metric] for o in ours]))
                    retention.append(dict(dataset=ds, split=split, reference=arm, metric=metric,
                        ours=ours_mean, reference_value=real[metric],
                        ratio_of_means=100*ours_mean/real[metric],
                        mean_paired_student_ratio=float(np.mean(ratios)),
                        paired_student_ratio_sd=float(np.std(ratios,ddof=1)),
                        paired_student_difference_mean=float(np.mean(paired)),
                        full_rows=result['protocol']['reference_rows'],
                        synthetic_rows=result['protocol']['budgets'][-1],
                        full_edges=result['protocol']['reference_edges'],
                        actual_rate=result['protocol']['actual_final_rate'],
                        sampled_loss_row_coverage=float(np.mean([r['fraction'] for r in result['sampling_coverage']])),
                        note='3 Student pairs after averaging 3 construction seeds; full-history is not triplicated'))

    baseline, replay, cost = [], [], []
    for ds in DATASETS:
        result = read(batch / 'tagsam_budget' / f'{ds}_o9101_c271006/results.json')
        if result is not None:
            validate_protocol(result, primary_map[ds,SEEDS[0]])
            if sorted(row['outer_steps'] for row in result['rows']) != [100,500]:
                raise AssertionError('TAGSAM response budget arms differ')
            for row in result['rows']:
                if row['stage'] != len(result['protocol']['budgets'])-1:
                    raise AssertionError('TAGSAM budget diagnostic must be final-stage only')
                validate_row(row, result['protocol'], result['protocol']['budgets'][-1])
            replay.append(dict(dataset=ds, **result['archived_100_replay']))
            for split in SPLITS:
                for r in result['rows']:
                    score = r['scores'][split]
                    baseline.append(dict(dataset=ds, split=split, outer_steps=r['outer_steps'],
                        **{k:score[k] for k in ('final_f1','accuracy','old_f1','new_f1')},
                        student_f1_sd=score['final_f1_std'], response_seconds=r['response_seconds'],
                        response_peak_mb=r['response_peak_mb']))
        result = read(batch / 'cost' / f'{ds}_o9101_c271006_dense_vs_sparse/results.json')
        if result is not None:
            validate_protocol(result, primary_map[ds,SEEDS[0]])
            if sorted(row['arm'] for row in result['rows']) != ['dense_identity','sparse_identity']:
                raise AssertionError('Identity benchmark arms differ')
            if not result['operator_equivalence']['passed']:
                raise AssertionError('Identity operator equivalence failed')
            for r in result['rows']:
                cost.append(dict(dataset=ds, arm=r['arm'], stage=0,
                    **{k:r[k] for k in ('construction_seconds','construction_peak_mb','construction_reserved_mb')},
                    **result['operator_equivalence'], **result['construction_equivalence'],
                    **result['allocation_accounting']))

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    outputs = dict(text_cells=text_cells,text_students=text_students,text_summary=summary,
        text_contrasts=contrasts,text_interactions=interactions,fullhistory_retention=retention,
        fullhistory_student_pairs=retention_students,tagsam_budget=baseline,tagsam_replay=replay,cost=cost)
    for name, rows in outputs.items():
        export(out / f'{name}.csv', rows)
    manifest = dict(sources=sources, pending=pending, status='complete' if not pending else 'partial',
        text_inference='t intervals over 3 construction-cell means at order 9101; Students averaged first',
        fullhistory_inference='Ratio of means and mean Student-paired ratio; no invented construction replicates',
        splits='Both masks previously inspected; post-hoc diagnostics, not fresh blind confirmation',
        summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'data.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8')
    text_report(out, outputs, manifest)
    if not args.skip_figures:
        figures(out, outputs)
    print(json.dumps(dict(output=str(out),status=manifest['status'],pending=len(pending),
                         rows={k:len(v) for k,v in outputs.items()})))


def text_report(out, data, manifest):
    lines = ['# CoTAG-Q four-gap evidence', '', f"Batch status: {manifest['status']}; pending cells: {len(manifest['pending'])}.",
        '', 'All current test-mask metrics below are post-hoc. No method is selected from these results.',
        '', '## Text initialization and pair feedback', '',
        'Three construction seeds, one order, three Student seeds per construction; F1 is a percentage.', '']
    rows = [r for r in data['text_summary'] if r['split']=='sealed_test' and r['metric']=='final_f1']
    lines += [table(['Dataset','Arm','F1 mean','Construction SD'],
        [[r['dataset'],r['arm'],fmt(r['mean']),fmt(r['sd'])] for r in rows]),'']
    rows = [r for r in data['text_contrasts'] if r['split']=='sealed_test' and r['metric']=='final_f1']
    lines += [table(['Dataset','Full minus','F1 difference','95% paired CI','Positive','Worst'],
        [[r['dataset'],r['control'],fmt(r['mean']),f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]",
          f"{r['positive']}/{r['n']}",fmt(r['worst'])] for r in rows]),'']
    lines += ['Initialization intervention removes the text term from both prototype selection and initial logits. ',
        'Pair intervention removes the node-payload pair term on both real and synthetic proxy losses; prompt supervision remains.',
        'The interaction is full minus no-init minus no-pair plus both-off. All effects are conditional on the frozen constructor.', '',
        '## Full-history matched reference', '',
        'Same legal pool, order, masks, text preprocessing, encoder, Student and 2048-update budget. ',
        'Full topology means edges induced by legal training nodes, not a fully supervised entire-graph upper bound.', '']
    rows = [r for r in data['fullhistory_retention'] if r['split']=='sealed_test' and r['metric'] in ('final_f1','accuracy')]
    lines += [table(['Dataset','Reference','Metric','Ours','Full','Ratio of means (%)','Mean paired ratio (%)','Loss-row coverage (%)'],
        [[r['dataset'],r['reference'],r['metric'],fmt(r['ours']),fmt(r['reference_value']),
          fmt(r['ratio_of_means']),fmt(r['mean_paired_student_ratio']),fmt(100*r['sampled_loss_row_coverage'])] for r in rows]),'',
        'Loss-row coverage counts rows sampled for supervision, not graph-forward exposure. ',
        'All nodes participate in the full graph forward; training-pool neighbors can receive indirect gradients.', '',
        '## TAGSAM last-stage response budget', '',
        'Same fixed prefix, PMI initialization and teacher; 100 versus 500 outer updates, 10 inner steps. ',
        'This is a last-stage diagnostic, not all-stage retraining or official 5000x15 reproduction.', '']
    rows = [r for r in data['tagsam_budget'] if r['split']=='sealed_test']
    lines += [table(['Dataset','Outer steps','F1','Accuracy','Response seconds'],
        [[r['dataset'],r['outer_steps'],fmt(r['final_f1']),fmt(r['accuracy']),fmt(r['response_seconds'])] for r in rows]),'',
        '## Identity-operator resource diagnostic', '',
        'Stage zero, one seed, one dense then sparse run per graph. Timing is descriptive, without repeated timing inference.', '']
    lines += [table(['Dataset','Operator','Seconds','Allocated GiB','Reserved GiB','X max error','Same text'],
        [[r['dataset'],r['arm'],fmt(r['construction_seconds']),fmt(r['construction_peak_mb']/1024),
          fmt(r['construction_reserved_mb']/1024),f"{r['x_max_error']:.3g}",r['text_identity']] for r in data['cost']]),'']
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')


def figures(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'figure.dpi':160})
    colors = ['#177e89','#cb6040','#56606b']

    def save(fig, name, records, description):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', bbox_inches='tight')
        provenance = dict(description=description, records=records,
            data_sha256=hashlib.sha256((out/'data.json').read_bytes()).hexdigest(),
            renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out/f'{name}.source.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        plt.close(fig)

    selected = [r for r in data['text_contrasts'] if r['split']=='sealed_test' and r['metric']=='final_f1']
    if len(selected) == 9:
        fig, axes = plt.subplots(1,3,figsize=(7.1,2.5),sharex=True,sharey=True)
        fig.subplots_adjust(left=.18,right=.98,bottom=.27,top=.83,wspace=.16)
        names = {'no_text_init':'No text init.','no_pair_feedback':'No pair loss','both_off':'Both removed'}
        low = min(0., min(r['ci_low'] for r in selected))
        high = max(0., max(r['ci_high'] for r in selected))
        pad = .08 * max(high-low, 1.)
        for i,(ax,ds) in enumerate(zip(axes,DATASETS)):
            rows=[next(r for r in selected if r['dataset']==ds and r['control']==arm) for arm in names]
            for y,r,col,marker in zip(range(3),rows,colors,('o','s','^')):
                ax.errorbar(r['mean'],y,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt=marker,color=col,capsize=3,markersize=4)
            ax.axvline(0,color='#999999',linewidth=.8)
            ax.set_yticks(range(3),list(names.values()))
            ax.set_ylim(2.6,-.6); ax.set_xlim(low-pad,high+pad)
            ax.tick_params(axis='y',length=0,labelleft=i==0)
            ax.set_title(NAMES[ds])
            ax.grid(axis='x',alpha=.2)
        fig.supxlabel('Full minus control (Macro-F1 points; paired 95% CI)',fontsize=8.5,y=.04)
        save(fig,'text_factorial',selected,
             'One order; three construction means, each averaging three Students. Common x scale. Previously inspected test mask.')
    selected=[r for r in data['fullhistory_retention'] if r['split']=='sealed_test' and r['metric'] in ('final_f1','accuracy')]
    if len(selected)==12:
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True)
        fig.subplots_adjust(left=.12,right=.96,bottom=.21,top=.77,wspace=.35)
        for ax,metric in zip(axes,('final_f1','accuracy')):
            for offset,arm,col,marker in zip((-.19,.19),('full_ef','full_topology'),colors,('o','^')):
                values=[next(r['ratio_of_means'] for r in selected if r['dataset']==ds and r['reference']==arm and r['metric']==metric) for ds in DATASETS]
                ax.scatter(values,np.arange(3)+offset,color=col,marker=marker,label='Edge-free full history' if arm=='full_ef' else 'Topology full history',s=28,zorder=3)
                for y,v in enumerate(values):
                    ax.annotate(f'{v:.1f}',(v,y+offset),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
            ax.axvline(100,color='#777777',linestyle='--',linewidth=1)
            ax.set_yticks(range(3),[NAMES[ds] for ds in DATASETS]); ax.set_ylim(2.5,-.5)
            ax.set_xlabel(('Macro-F1' if metric=='final_f1' else 'Accuracy')+' retained (%)')
            ax.grid(axis='x',alpha=.2); ax.margins(x=.28)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=8)
        save(fig,'fullhistory_matched',selected,
             'Ratios of means at order 9101; common finite 2048-update Student budget; no uncertainty bars inferred. Test mask previously inspected.')
    if len(data['cost'])==6:
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
        fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.78,wspace=.32)
        for ax,metric,scale,label in zip(axes,('construction_peak_mb','construction_seconds'),(1024,1),('Peak allocation (GiB)','Construction time (s)')):
            for offset,arm,col,hatch in zip((-.17,.17),('dense_identity','sparse_identity'),colors,('','//')):
                values=[next(r[metric]/scale for r in data['cost'] if r['dataset']==ds and r['arm']==arm) for ds in DATASETS]
                bars=ax.bar(np.arange(3)+offset,values,width=.32,color=col,hatch=hatch,edgecolor='white',linewidth=.6,label=arm.replace('_',' '))
                ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=7.5)
            ax.set_xticks(range(3),[NAMES[ds] for ds in DATASETS]); ax.set_ylabel(label)
            ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.margins(y=.22)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=8)
        save(fig,'identity_cost',data['cost'],
             'Stage zero, one seed, dense then sparse on the same isolated GPU. Descriptive single-run timings; frozen primary results unchanged.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-figures',action='store_true')
    summarize(p.parse_args())
