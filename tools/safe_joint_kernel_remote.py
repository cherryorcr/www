"""Upload/run/fetch the isolated safe joint-kernel follow-up snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shlex
import stat
import tarfile
import uuid

import paramiko

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/pc/research/TAG_condensation"
SNAPSHOT = REMOTE_ROOT + "/work_runs/safe_joint_kernel_v2_20260920_snapshot"


def mkdirs(sftp, path):
    try:
        sftp.stat(path)
    except IOError:
        parent = posixpath.dirname(path)
        if parent and parent != path:
            mkdirs(sftp, parent)
        try:
            sftp.mkdir(path)
        except IOError:
            pass


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("100.126.175.112", username="pc",
                   password=os.environ["TAG_REMOTE_PASSWORD"], timeout=30)
    return client


def upload_files(sftp, names):
    hashes = {}
    for name in names:
        source = ROOT / name
        destination = SNAPSHOT + "/" + name.replace("\\", "/")
        mkdirs(sftp, posixpath.dirname(destination))
        sftp.put(str(source), destination)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    with sftp.file(SNAPSHOT + "/safe_v2_sources.json", "w") as handle:
        handle.write(json.dumps(hashes, indent=2))
    return hashes


def fetch_tree(sftp, remote, local, artifacts=False):
    count = 0
    def walk(src, dst):
        nonlocal count
        dst.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(src):
            rpath, lpath = posixpath.join(src, entry.filename), dst / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                walk(rpath, lpath)
            elif (entry.filename.endswith((".json", ".csv", ".md", ".log"))
                  or artifacts and entry.filename.endswith(("_memory.pt", ".pt"))):
                sftp.get(rpath, str(lpath))
                count += 1
    walk(remote, Path(local))
    return count


def fetch_archive(client, sftp, remote, local):
    local = Path(local).resolve()
    local.mkdir(parents=True, exist_ok=True)
    archive = local.parent / (local.name + "_text_evidence.tar.gz")
    temporary = "/tmp/tag-text-evidence-" + uuid.uuid4().hex + ".tar.gz"
    command = ("tar --exclude='*.pt' --exclude='*.pyc' --exclude='__pycache__' -czf "
               + shlex.quote(temporary) + " -C " + shlex.quote(remote) + " .")
    _, stdout, stderr = client.exec_command(command, timeout=600)
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(stderr.read().decode(errors="replace"))
    try:
        sftp.get(temporary, str(archive))
    finally:
        sftp.remove(temporary)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            (local / member.name).resolve().relative_to(local)
            if not (member.isfile() or member.isdir()):
                raise ValueError("non-file entry in text evidence archive")
            if member.isfile() and not member.name.endswith((".json", ".csv", ".md", ".log")):
                raise ValueError("unexpected evidence extension: " + member.name)
        handle.extractall(local, members=members, filter="data")
    return {"files": sum(m.isfile() for m in members), "local": str(local),
            "archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("clone", "upload", "exec", "fetch"))
    parser.add_argument("--command")
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--only", nargs="+", help="Upload analysis/test files only; preserve frozen source manifest")
    parser.add_argument("--local")
    parser.add_argument("--remote")
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--archive", action="store_true", help="Fetch text evidence as one verified local archive")
    args = parser.parse_args()
    client = connect()
    try:
        sftp = client.open_sftp()
        if args.action == "clone":
            sftp.close()
            command = (f"mkdir -p {SNAPSHOT} && cp -a {REMOTE_ROOT}/work_runs/"
                       f"phase12_20260920_snapshot/. {SNAPSHOT}/")
            _, stdout, stderr = client.exec_command(command, timeout=600)
            code = stdout.channel.recv_exit_status()
            if code:
                raise SystemExit(stderr.read().decode(errors="replace"))
            print(json.dumps({"snapshot": SNAPSHOT, "cloned": True}))
            return
        if args.action == "upload":
            if args.only:
                hashes = {}
                for name in args.only:
                    source = (ROOT / name).resolve()
                    source.relative_to(ROOT)
                    destination = SNAPSHOT + "/" + name.replace("\\", "/")
                    mkdirs(sftp, posixpath.dirname(destination))
                    sftp.put(str(source), destination)
                    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
                print(json.dumps(hashes, indent=2))
                return
            base = [
                "code/v2_llm_fusion/safe_joint_kernel_append.py",
                "code/v2_llm_fusion/run_safe_joint_kernel_experiments.py",
                "code/v2_llm_fusion/diagnose_joint_kernel_topology.py",
                "code/v2_llm_fusion/summarize_safe_joint_kernel.py",
                "code/v2_llm_fusion/audit_safe_joint_kernel_v2.py",
                "code/v2_llm_fusion/test_safe_joint_kernel_append.py",
                "docs/current/SAFE_JOINT_KERNEL_V2_FROZEN_PROTOCOL_ZH_2026-09-20.md",
            ]
            print(json.dumps(upload_files(sftp, base + args.extra), indent=2))
        elif args.action == "exec":
            sftp.close()
            _, stdout, stderr = client.exec_command(args.command, timeout=1200)
            for line in stdout:
                print(line, end="", flush=True)
            errors = stderr.read().decode("utf-8", errors="replace")
            if errors:
                print(errors, end="", flush=True)
            raise SystemExit(stdout.channel.recv_exit_status())
        elif args.action == "fetch":
            if args.archive:
                if args.artifacts:
                    raise ValueError("--archive is text-only")
                print(json.dumps(fetch_archive(client, sftp, args.remote, args.local)))
            else:
                count = fetch_tree(sftp, args.remote, args.local, args.artifacts)
                print(json.dumps({"files": count, "local": args.local}))
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
