"""Generate tables only from complete, validated construction groups."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAPER = ROOT / 'paper_www'
SOURCE = PAPER / 'sections/06_results.tex'
DATASETS = ['arxiv', 'sportsfit', 'bookhis', 'cora', 'citeseer']
METHODS = [
    ('cat_cctp', 'CaT', 'Continual graph condensation'),
    ('gecc_hard', 'GECC', None),
    ('random_coreset', 'Random', 'Selection of real nodes'),
    ('kcenter_coreset', 'K-center', None),
    ('cluster_medoid', 'Cluster medoid', None),
    ('herding_coreset', 'Herding', None),
    ('gcond_x_core', 'GCond-x', 'Adaptations of graph condensation'),
    ('sfgc_core', 'SFGC', None),
    ('tagsam', 'TAGSAM', 'Adaptations of TAG condensation'),
    ('stad_native_local', 'STAD', None),
    ('uq_dm_edgefree', r'\method{}', 'Proposed method'),
]
EXPECTED = {(str(o), str(c)) for o in [9101, 9102, 9103] for c in [271006, 271108, 271109]}


def capture():
    target = HERE / 'previous_table_snapshot.json'
    assert not target.exists(), 'Keep the original table snapshot immutable.'
    text = SOURCE.read_text(encoding='utf-8')
    lines = [line for line in text.splitlines() if re.search(r'& [13]/[13]/3 &', line)]
    assert len(lines) == len(METHODS)
    values = {}
    for (arm, name, _), line in zip(METHODS, lines):
        fields = line.split('&')
        assert len(fields) == 7
        repetitions = fields[1].strip()
        values[arm] = {}
        for ds, field in zip(DATASETS, fields[2:]):
            numbers = [float(v) for v in re.findall(r'\d+(?:\.\d+)?', field)]
            assert len(numbers) == 4, field
            values[arm][ds] = dict(zip(['f1', 'f1_sd', 'accuracy', 'accuracy_sd'], numbers))
            values[arm][ds]['memories'] = 9 if repetitions == '3/3/3' else 1
    target.write_text(json.dumps(dict(source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source='Local Table 2 before the five-stage ArXiv migration', values=values), indent=2), encoding='utf-8')


def load_complete(path, old_format):
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for arm, _, _ in METHODS:
        for ds in DATASETS if old_format else ['arxiv']:
            group = [r for r in rows if r['arm'] == arm and (not old_format or r['dataset'] == ds)]
            keys = {(r['order'], r['seed']) for r in group}
            assert len(group) == len(keys), (arm, ds, 'duplicate construction')
            assert keys <= EXPECTED
            if keys != EXPECTED:
                continue
            f1 = [float(r['final_f1' if old_format else 'f1']) for r in group]
            acc = [float(r['accuracy']) for r in group]
            result[arm, ds] = dict(f1=statistics.mean(f1), f1_sd=statistics.stdev(f1),
                accuracy=statistics.mean(acc), accuracy_sd=statistics.stdev(acc), memories=9)
    return result


def metric(value, mark=False):
    if value is None:
        return '--'
    f1 = f"{value['f1']:.2f}"
    if mark == 'best':
        f1 = r'\textbf{' + f1 + '}'
    elif mark == 'second':
        f1 = r'\underline{' + f1 + '}'
    text = f"{f1}$\\pm${value['f1_sd']:.2f} / {value['accuracy']:.2f}$\\pm${value['accuracy_sd']:.2f}"
    return text + (r'$^{\dagger}$' if value['memories'] == 1 else '')


def refresh():
    original = json.loads((HERE / 'previous_table_snapshot.json').read_text())['values']
    old_path = ROOT / 'experiments/cotag_baseline_repetitions_20261024/completed_constructions.csv'
    new_path = HERE / 'completed_constructions.csv'
    old = load_complete(old_path, True)
    new = load_complete(new_path, False)
    values, archive = {}, {}
    for arm, _, _ in METHODS:
        archive[arm] = old.get((arm, 'arxiv'), original[arm]['arxiv'])
        values[arm] = {ds: (new.get((arm, ds)) if ds == 'arxiv'
                           else old.get((arm, ds), original[arm][ds])) for ds in DATASETS}
    ranking = {ds: sorted({values[a][ds]['f1'] for a, _, _ in METHODS if values[a][ds]}, reverse=True)
               for ds in DATASETS}
    caption = (r'Node classification with adapted baselines and a common student. '
        r'Entries are Macro-F1 / Accuracy in percent, reported as mean $\pm$ SD '
        r'across nine memories from three class orders and three construction seeds. '
        r'Each memory averages three student seeds.')
    if any(v and v['memories'] == 1 for group in values.values() for v in group.values()):
        caption += r' $\dagger$ marks a provisional single-memory result whose SD is across students.'
    if any(group['arxiv'] is None for group in values.values()):
        caption += r' Dashes mark five-stage ArXiv results pending completion.'
    caption += r' Bold and underlined F1 means are the best and second best available results.'
    lines = [r'\begin{table*}[t]', r'\centering', r'\caption{' + caption + '}',
        r'\label{tab:main_comparison}', r'\label{tab:matched_main}',
        r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lccccc}', r'\toprule',
        r'Method & ArXiv & SportsFit & BookHis & Cora & Citeseer\\']
    for arm, name, group in METHODS:
        if group:
            lines.extend([r'\midrule', r'\multicolumn{6}{c}{\emph{' + group + r'}}\\'])
        cells = []
        for ds in DATASETS:
            v = values[arm][ds]
            position = ranking[ds].index(v['f1']) if v else -1
            cells.append(metric(v, 'best' if position == 0 else 'second' if position == 1 else False))
        lines.append(name + ' & ' + ' & '.join(cells) + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])
    (PAPER / 'tables/main_comparison.tex').write_text('\n'.join(lines), encoding='utf-8')
    archived = [r'\begin{table}[H]', r'\centering', r'\footnotesize',
        r'\caption{Archived ArXiv comparison with three stages (13/14/13 classes) on the historical test partition. Entries are mean $\pm$ SD. $n$ is the number of memories, each evaluated with three student seeds. With $n=1$, SD is across students. These scores do not represent the five-stage protocol.}',
        r'\label{tab:arxiv_three_stage}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcr@{}}',
        r'\toprule', r'Method & $n$ & Macro-F1 / Accuracy\\', r'\midrule']
    for arm, name, _ in METHODS:
        v = archive[arm]
        archived.append(name + f" & {v['memories']} & " + metric(dict(v, memories=9)) + r'\\')
    archived.extend([r'\bottomrule', r'\end{tabular*}', r'\end{table}', ''])
    (PAPER / 'tables/arxiv_three_stage_archive.tex').write_text('\n'.join(archived), encoding='utf-8')
    receipts = dict(inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [HERE / 'previous_table_snapshot.json', old_path, new_path] if p.exists()},
        complete_five_stage_methods=[a for a, _, _ in METHODS if values[a]['arxiv']],
        provisional=[f'{a}/{ds}' for a, _, _ in METHODS for ds in DATASETS
                     if values[a][ds] and values[a][ds]['memories'] == 1])
    (HERE / 'table_receipt.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    print(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    capture() if args.capture else refresh()
