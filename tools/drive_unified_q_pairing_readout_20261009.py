"""Two-GPU frozen-memory experiment, with compact local/remote progress."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import queue
import shlex
import subprocess
import tarfile

import drive_unified_q_gecc_gap_20261009 as B

ROOT = B.ROOT
NAME = 'uq_pairing_readout_20261009'
DEST = B.BASE + '/experiments/' + NAME + '_v1'
RESULT = B.BASE + '/results/' + NAME
LOCAL = ROOT / 'results' / NAME
DEPENDENCY = B.BASE + '/experiments/uq_gecc_gap_p2_paired_v3_20261009/code'
FILES = ('unified_q_pairing_readout.py', 'run_unified_q_pairing_readout_20261009.py')


def deploy():
    B.remote('test ! -e ' + shlex.quote(DEST))
    B.remote('mkdir -p ' + shlex.quote(DEST + '/code'))
    for filename in FILES:
        subprocess.run(['scp', '-C', str(ROOT / 'code/v2_llm_fusion' / filename),
                        B.HOST + ':' + DEST + '/code/' + filename], check=True)
    print(json.dumps(dict(snapshot=DEST)), flush=True)


def command(task, gpu):
    tag = f"{task['dataset']}_o{task['order']}_c{task['seed']}"
    source = B.BASE + '/results/uq_text_path_20261008/confirm_' + tag
    env = 'env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH='
    env += shlex.quote(DEST + '/code:' + DEPENDENCY + ':' + B.BASE + '/code/v2_llm_fusion:'
                       + B.BASE + '/code/frozen_v1_2026_09_09')
    args = [B.PYTHON, '-u', DEST + '/code/' + FILES[1], '--dataset', task['dataset'],
        '--order-seed', str(task['order']), '--construction-seed', str(task['seed']),
        '--device', f'cuda:{gpu}', '--source', source, '--output', RESULT + '/' + tag,
        '--reference', B.BASE + f"/results/unified_q_inloop_cat_20261007/{task['dataset']}/protocol.json",
        '--text-cache', B.BASE + f"/results/public_baselines_text_cache_20261001/{task['dataset']}",
        '--minilm', '/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip']
    return tag, env + ' ' + ' '.join(map(shlex.quote, args))


def run(first):
    LOCAL.mkdir(parents=True, exist_ok=True)
    tasks = B.TASKS[:1] if first else B.TASKS
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    plan = dict(tasks=B.TASKS, permutation_seeds=[271011, 271012], student_seeds=[150, 151, 152],
                student_steps=2048, new_trainings=216, reused_trainings=108, constructor_changed=False)
    plan_path = LOCAL / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('plan changed')
    plan_path.write_text(json.dumps(plan, indent=2))

    def lane(gpu):
        records = []
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            tag, cmd = command(task, gpu)
            print(json.dumps(dict(event='start', gpu=gpu, **task)), flush=True)
            with (LOCAL / (tag + '.log')).open('a', encoding='utf-8') as log:
                process = subprocess.run(B.SSH + [cmd], stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(**task, gpu=gpu, exit_code=process.returncode))
            print(json.dumps(dict(event='finish', **records[-1])), flush=True)
            if process.returncode:
                break
        return records

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [r for group in pool.map(lane, (0, 1)) for r in group]
    success = len(records) == len(tasks) and all(r['exit_code'] == 0 for r in records)
    (LOCAL / ('first_status.json' if first else 'batch_status.json')).write_text(
        json.dumps(dict(status='complete' if success else 'incomplete', records=records), indent=2))
    if not success:
        raise SystemExit(1)


def collect():
    response = subprocess.run(B.SSH + ['tar -czf - -C ' + shlex.quote(RESULT)
        + " --exclude='*.tmp' ."], capture_output=True, timeout=180)
    warning = response.stderr.decode(errors='replace')
    if response.returncode and not (response.returncode == 1 and warning.strip() and all(
            'file changed as we read it' in line for line in warning.splitlines() if line.strip())):
        raise RuntimeError(warning)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode='r:gz') as archive:
        for entry in archive:
            if entry.isdir():
                continue
            target = (LOCAL / entry.name).resolve()
            if not entry.isfile() or target.suffix != '.json' or not target.is_relative_to(LOCAL.resolve()):
                raise ValueError('unexpected archive entry')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(entry).read())
            count += 1
    print(json.dumps(dict(collected=count, bytes=len(response.stdout))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy', 'run', 'collect'))
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    if args.action == 'deploy':
        deploy()
    elif args.action == 'run':
        run(args.first)
    else:
        collect()
