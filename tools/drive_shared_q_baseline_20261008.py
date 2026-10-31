"""Deploy a private frozen snapshot and collect it, without deleting data."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
HOST = "pc@100.126.175.112"
REMOTE_ROOT = "/home/pc/research/TAG_condensation"
EXPERIMENT = REMOTE_ROOT + "/experiments/shared_q_baseline_20261008"
RESULT = REMOTE_ROOT + "/results/shared_q_baseline_20261008"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", HOST]


def remote(command, timeout=240, capture=False):
    return subprocess.run(SSH + [command], check=True, timeout=timeout, capture_output=capture)


def deploy():
    archive = ROOT / "tmp_remote_results/shared_q_baseline_frozen_20261008.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for path in (ROOT / "code/v2_llm_fusion").glob("*.py"):
            out.add(path, arcname="code/" + path.name)
        for path in (ROOT / "results/shared_q_baseline_20261008/handoff_audit").glob("*_manifest.json"):
            out.add(path, arcname="manifests/" + path.name)
        for name in ("preflight_shared_q_handoff_20261008.py", "run_shared_q_baseline_queue_20261008.py"):
            path = ROOT / "tools" / name
            if path.exists():
                out.add(path, arcname=name)
    target = REMOTE_ROOT + "/tmp/shared_q_baseline_frozen_20261008.tar.gz"
    subprocess.run(["scp", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    str(archive), HOST + ":" + target], check=True, timeout=180)
    remote("mkdir -p " + shlex.quote(EXPERIMENT))
    remote("tar -xzf " + shlex.quote(target) + " -C " + shlex.quote(EXPERIMENT))
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, remote=EXPERIMENT)), flush=True)


def collect(tensors):
    command = ["tar", "-czf", "-", "-C", RESULT, "--exclude=*.log", "--exclude=*.tmp"]
    if not tensors:
        command += ["--exclude=*.pt", "--exclude=*.pt.gz"]
    command += ["."]
    response = remote(" ".join(map(shlex.quote, command)), capture=True, timeout=300)
    dest = (ROOT / "results/shared_q_baseline_20261008").resolve()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode="r:gz") as stream:
        for item in stream:
            if item.isdir():
                continue
            path = (dest / item.name).resolve()
            if not item.isfile() or not path.is_relative_to(dest) or path.suffix not in (".json", ".pt", ".gz"):
                raise ValueError("unexpected archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".sync")
            temp.write_bytes(stream.extractfile(item).read())
            temp.replace(path)
            count += 1
    print(json.dumps(dict(collected_files=count, compressed_bytes=len(response.stdout))), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("deploy", "preflight", "collect", "run"))
    p.add_argument("--phase", default="shared")
    p.add_argument("--tensors", action="store_true")
    args = p.parse_args()
    if args.action == "deploy":
        deploy()
    elif args.action == "collect":
        collect(args.tensors)
    elif args.action == "preflight":
        remote(" ".join(map(shlex.quote, [PYTHON, EXPERIMENT + "/preflight_shared_q_handoff_20261008.py",
            "--manifests", EXPERIMENT + "/manifests", "--output", RESULT + "/preflight.json"])))
    else:
        remote(" ".join(map(shlex.quote, [PYTHON, "-u", EXPERIMENT + "/run_shared_q_baseline_queue_20261008.py",
                                         "--phase", args.phase])), timeout=6 * 3600)


if __name__ == "__main__":
    main()
