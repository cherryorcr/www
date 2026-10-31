"""Pairing effects are aggregated within construction cells before graphs."""
import argparse
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'uq_dm_edgefree': 'In-loop CaT-Q', 'cat_cctp': 'CaT + nearest5', 'gecc_hard': 'GECC + TAG'}
DATASETS = ('wikics', 'sportsfit', 'bookhis')
STUDENTS = ('prompt_gcn', 'prompt_sgc')
METRICS = ('final_f1', 'accuracy', 'old_f1', 'new_f1')


def summarize(root):
    cells, effects, geometry, probes = [], [], [], []
    for path in sorted(root.glob('*/results.json')):
        obj = json.loads(path.read_text())
        if obj['status'] != 'complete':
            raise ValueError('incomplete cell: ' + str(path.parent))
        p = obj['protocol']
        identity = dict(dataset=p['dataset'], construction_seed=p['construction_seed'], order_seed=p['order_seed'])
        cells.append(dict(**identity, rows=len(obj['rows']), controls=len(obj['controls'])))
        for arm in LABELS:
            for student in STUDENTS:
                rows = [r for r in obj['rows'] if r['arm'] == arm and r['student'] == student]
                original = [r for r in rows if r['permutation_seed'] is None]
                permuted = [r for r in rows if r['permutation_seed'] is not None]
                if len(original) != 1 or len(permuted) != 2:
                    raise ValueError('missing paired rows')
                base = original[0]['score']
                shuffled = {m: st.mean(r['score'][m] for r in permuted) for m in METRICS}
                effects.append(dict(**identity, arm=arm, student=student,
                    native={m: base[m] for m in METRICS}, shuffled=shuffled,
                    advantage={m: base[m] - shuffled[m] for m in METRICS},
                    per_permutation=[dict(permutation_seed=r['permutation_seed'],
                        **{m: base[m] - r['score'][m] for m in METRICS}) for r in permuted]))
        geometry.extend(dict(**identity, **g) for g in obj['geometry'])
        probes.extend(dict(**identity, **g) for g in obj['diagnostics'])
    expected = {(d, s) for d in DATASETS for s in (271108, 271109)}
    if {(r['dataset'], r['construction_seed']) for r in cells} != expected or len(cells) != 6:
        raise ValueError('complete six-cell matrix required')
    aggregates = []
    for dataset in DATASETS:
        for student in STUDENTS:
            for arm in LABELS:
                group = [e for e in effects if e['dataset'] == dataset and e['student'] == student and e['arm'] == arm]
                aggregates.append(dict(dataset=dataset, student=student, arm=arm, cells=len(group),
                    native={m: st.mean(e['native'][m] for e in group) for m in METRICS},
                    shuffled={m: st.mean(e['shuffled'][m] for e in group) for m in METRICS},
                    advantage={m: st.mean(e['advantage'][m] for e in group) for m in METRICS},
                    advantage_std={m: st.stdev(e['advantage'][m] for e in group) for m in METRICS},
                    positive_cells=sum(e['advantage']['final_f1'] > 0 for e in group),
                    worst_cell=min(e['advantage']['final_f1'] for e in group)))
    main = [a for a in aggregates if a['arm'] == 'uq_dm_edgefree']
    mean_advantage = {s: st.mean(a['advantage']['final_f1'] for a in main if a['student'] == s) for s in STUDENTS}
    consistent_graphs = sum(all(next(a for a in main if a['dataset'] == d and a['student'] == s)
                                ['advantage']['final_f1'] > 0 for s in STUDENTS) for d in DATASETS)
    gate = dict(native_minus_permuted_mean_f1=mean_advantage,
        positive_graphs_on_both_architectures=consistent_graphs,
        prioritize_slot_pairing=all(v >= .5 for v in mean_advantage.values()) and consistent_graphs >= 2,
        interpretation='mechanism-priority screen, not a promotion or statistical significance test')
    out = dict(scope='post-construction attribution; one development order, two construction seeds; report-dev',
        execution=dict(cells=6, new_student_trainings=216, reused_student_trainings=108,
                       permutation_sidecars=36, gradient_probes=36),
        gate=gate, aggregates=aggregates, paired_cells=effects, text_geometry=geometry, gradient_probes=probes)
    (root / 'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    lines = ['# Fixed-memory Pairing and Readout Diagnosis', '',
        'Native minus shuffled; positive values favor the learned original pairing. '
        'Two permutations and three Student seeds are averaged inside each construction cell. '
        'One development order and two construction seeds per graph; report-dev only.', '',
        'External adaptations are CaT + nearest5 and GECC + TAG. Shuffling is an internal attribution '
        'control on each memory, not an additional external baseline.', '']
    for dataset in DATASETS:
        lines += ['## ' + dataset, '',
            '| Method | Student | Native F1 / Acc | Shuffled F1 / Acc | F1 advantage | Acc advantage | Old F1 advantage | New F1 advantage | Positive cells |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for a in aggregates:
            if a['dataset'] != dataset:
                continue
            n, s, d = a['native'], a['shuffled'], a['advantage']
            lines.append(f"| {LABELS[a['arm']]} | {a['student'][7:].upper()} | {n['final_f1']:.2f} / {n['accuracy']:.2f} | "
                f"{s['final_f1']:.2f} / {s['accuracy']:.2f} | {d['final_f1']:+.2f} | {d['accuracy']:+.2f} | "
                f"{d['old_f1']:+.2f} | {d['new_f1']:+.2f} | {a['positive_cells']}/2 |")
        lines += ['']
    lines += ['## Gradient Diagnostics', '',
        'Same model initialization for native/shuffled; gradients are not trained-model trajectories.', '',
        '| Dataset | Method | Student | Group/full vs minibatch cosine | Node/full vs minibatch cosine | Shuffle gradient relative difference | MC half-split relative difference |',
        '|---|---|---|---:|---:|---:|---:|']
    for a in aggregates:
        group = [g for g in probes if all(g[k] == a[k] for k in ('dataset', 'arm', 'student'))]
        means = [st.mean(g[k] for g in group) for k in (
            'full_group_to_minibatch_gradient_cosine', 'full_node_to_minibatch_gradient_cosine',
            'minibatch_permutation_gradient_relative', 'minibatch_sampling_half_difference_relative')]
        lines.append(f"| {a['dataset']} | {LABELS[a['arm']]} | {a['student'][7:].upper()} | "
                     + ' | '.join(f'{v:.4f}' for v in means) + ' |')
    lines += ['', '## Gate', '', '```json', json.dumps(gate, indent=2), '```', '']
    (root / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    plt.rcParams.update({'font.size': 10})
    for metric, title, filename in (('final_f1', 'Macro-F1 advantage of native pairing (pp)', 'pairing_f1.png'),
                                   ('accuracy', 'Accuracy advantage of native pairing (pp)', 'pairing_accuracy.png')):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, student in zip(axes, STUDENTS):
            x = np.arange(3)
            for i, (arm, color) in enumerate(zip(LABELS, ('#207f70', '#477d9c', '#95619b'))):
                rows = [next(a for a in aggregates if a['dataset'] == d and a['student'] == student and a['arm'] == arm) for d in DATASETS]
                values = [a['advantage'][metric] for a in rows]
                bars = ax.bar(x + (i - 1) * .25, values, .23, label=LABELS[arm], color=color,
                    yerr=[a['advantage_std'][metric] for a in rows], capsize=3)
                ax.bar_label(bars, labels=[f'{v:+.2f}' for v in values], padding=4, fontsize=9)
            ax.axhline(0, color='#555555', linewidth=.8)
            ax.set_xticks(x, ['WikiCS', 'SportsFit', 'BookHis'])
            ax.set_title(student[7:].upper())
            ax.margins(y=.25)
        axes[0].set_ylabel(title)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
        fig.suptitle('Fixed memory, within-stage/class text permutations; report-dev')
        fig.tight_layout(rect=(0, .10, 1, .94))
        fig.savefig(root / filename, dpi=160)
        plt.close(fig)
    print(json.dumps(dict(execution=out['execution'], gate=gate)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'results/uq_pairing_readout_20261009')
    summarize(parser.parse_args().root)
