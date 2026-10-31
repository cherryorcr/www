"""Summarize validated queue snapshots without treating partial groups as final."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
source = ROOT / 'completed_constructions.csv'
groups = defaultdict(list)
with source.open(newline='', encoding='utf-8') as stream:
    for row in csv.DictReader(stream):
        groups[row['arm'], row['dataset']].append(row)
summary = []
expected = {(str(order), str(seed)) for order in [9101, 9102, 9103]
            for seed in [271006, 271108, 271109]}
for (arm, dataset), rows in sorted(groups.items()):
    keys = {(r['order'], r['seed']) for r in rows}
    assert len(keys) == len(rows), (arm, dataset, 'duplicate replicate')
    assert keys <= expected
    f1 = [float(r['final_f1']) for r in rows]
    acc = [float(r['accuracy']) for r in rows]
    summary.append(dict(arm=arm, dataset=dataset, memories=len(rows), complete=keys == expected,
                        f1_mean=statistics.mean(f1), f1_sd=statistics.stdev(f1) if len(f1) > 1 else '',
                        accuracy_mean=statistics.mean(acc),
                        accuracy_sd=statistics.stdev(acc) if len(acc) > 1 else ''))
with (ROOT / 'summary_by_method.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
complete_arms = sorted({r['arm'] for r in summary
                        if all(g['complete'] for g in summary if g['arm'] == r['arm'])
                        and sum(g['arm'] == r['arm'] for g in summary) == 5})
receipt = dict(source=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               construction_count=sum(len(rows) for rows in groups.values()),
               complete_methods=complete_arms, statistic='Student mean per memory, then sample SD across nine memories.',
               partial_groups_are_final=False)
(ROOT / 'summary_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
for row in summary:
    if row['arm'] in complete_arms:
        print(f"{row['arm']:18} {row['dataset']:10} "
              f"{row['f1_mean']:.2f} +/- {row['f1_sd']:.2f} / "
              f"{row['accuracy_mean']:.2f} +/- {row['accuracy_sd']:.2f}")
