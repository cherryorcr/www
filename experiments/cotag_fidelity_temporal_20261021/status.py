"""Compact status without loading memory artifacts."""
import json
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/pc/research/TAG_condensation/results/cotag_fidelity_temporal_20261021')


def main(compact=False):
    rows = []
    for path in sorted(ROOT.rglob('results.json')):
        if 'native_student_diagnostic' in path.relative_to(ROOT).parts:
            continue
        value = json.loads(path.read_text())
        record = dict(path=str(path.relative_to(ROOT)), status=value['status'],
                      completed_stages=len(value['rows']), error=value.get('error'))
        if 'precision_repair' in path.relative_to(ROOT).parts:
            record['status'] = 'invalid/excluded'
        if value['rows']:
            row = value['rows'][-1]
            record['stage'] = row['stage']
            if 'prediction' in row['scores']:
                record['dev'] = row['scores']['prediction']['arrived_snapshot']['report_dev']['cumulative']
            else:
                score = row['scores']['report_dev']
                record['dev'] = {k: score[k] for k in ('final_f1', 'accuracy')}
        traces = sorted(path.parent.glob('s*/trace.json'))
        if traces:
            last = traces[-1]
            curve = json.loads(last.read_text())
            record['trace'] = dict(stage=last.parent.name, **curve[-1]) if curve else None
        rows.append(record)
    jobs = subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,pcpu,rss,args'], text=True)
    jobs = [line for line in jobs.splitlines() if 'cotag_fidelity_temporal_20261021' in line
            and 'status.py' not in line and 'ps -eo' not in line]
    output = dict(free_gib=shutil.disk_usage(ROOT).free/2**30, jobs=jobs, results=rows,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv'], text=True))
    if compact:
        print(f"Free disk: {output['free_gib']:.2f} GiB")
        print(output['gpu'])
        for row in rows:
            trace = row.get('trace') or {}
            print(row['path'], row['status'], f"stages={row['completed_stages']}",
                  f"trace={trace.get('stage')}/{trace.get('step')}", row.get('dev', {}))
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact', action='store_true')
    main(parser.parse_args().compact)
