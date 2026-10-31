"""Deploy and run the CaT-conditioned chunk-payload smoke on the GPU host."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import posixpath
import shutil
import stat

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/pc/research/TAG_condensation"
SNAPSHOT = REMOTE_ROOT + "/work_runs/cat_payload_20261002_snapshot"
CODE_DIR = SNAPSHOT + "/code/v2_llm_fusion"
FILES = [
    "code/v2_llm_fusion/cat_multimodal_minimal.py",
    "code/v2_llm_fusion/run_cat_multimodal_minimal_20261002.py",
    "code/v2_llm_fusion/external_continual_mechanisms.py",
]


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("100.126.175.112", username="pc",
                   password=os.environ["TAG_REMOTE_PASSWORD"], timeout=30)
    return client


def mkdirs(sftp, path: str):
    try:
        sftp.stat(path)
        return
    except IOError:
        pass
    parent = posixpath.dirname(path)
    if parent and parent != path:
        mkdirs(sftp, parent)
    try:
        sftp.mkdir(path)
    except IOError:
        pass


def deploy(client, sftp):
    command = (
        f"rm -rf {SNAPSHOT} && mkdir -p {CODE_DIR} && "
        f"cp -a {REMOTE_ROOT}/code/v2_llm_fusion/. {CODE_DIR}/"
    )
    _, stdout, stderr = client.exec_command(command, timeout=600)
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(stderr.read().decode("utf-8", errors="replace"))
    hashes = {}
    for name in FILES:
        local = ROOT / name
        remote = SNAPSHOT + "/" + name.replace("\\", "/")
        mkdirs(sftp, posixpath.dirname(remote))
        sftp.put(str(local), remote)
        hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()
    return hashes


def exec_remote(client, command: str):
    _, stdout, stderr = client.exec_command(command, timeout=3600)
    for line in stdout:
        print(line, end="", flush=True)
    errors = stderr.read().decode("utf-8", errors="replace")
    if errors:
        print(errors, end="", flush=True)
    code = stdout.channel.recv_exit_status()
    if code:
        raise SystemExit(code)


def fetch_json(sftp, remote: str, local: Path, artifacts: bool = False):
    count = 0

    def walk(src: str, dst: Path):
        nonlocal count
        dst.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(src):
            rpath = posixpath.join(src, entry.filename)
            lpath = dst / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                walk(rpath, lpath)
            elif (entry.filename.endswith((".json", ".log")) or
                  artifacts and entry.filename.endswith(".pt")):
                sftp.get(rpath, str(lpath))
                count += 1

    walk(remote, local)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "exec", "fetch"))
    parser.add_argument("--command")
    parser.add_argument("--remote")
    parser.add_argument("--local")
    parser.add_argument("--artifacts", action="store_true")
    args = parser.parse_args()
    client = connect()
    try:
        sftp = client.open_sftp()
        if args.action == "deploy":
            print(deploy(client, sftp))
        elif args.action == "exec":
            if not args.command:
                raise ValueError("--command is required")
            exec_remote(client, args.command)
        else:
            if not args.remote or not args.local:
                raise ValueError("--remote and --local are required")
            print({"files": fetch_json(sftp, args.remote, Path(args.local),
                                        args.artifacts)})
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
