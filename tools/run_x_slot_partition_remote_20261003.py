"""Deploy and launch the X-conditioned slot-partition experiment remotely."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import posixpath
import shlex
import stat

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/pc/research/TAG_condensation"
SNAPSHOT = REMOTE_ROOT + "/work_runs/x_slot_partition_20261003_snapshot"
REMOTE_CODE = SNAPSHOT + "/code/v2_llm_fusion"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
SOURCE_ROOT = REMOTE_ROOT + "/results/p5_large_base4_20260919"
CACHE_ROOT = REMOTE_ROOT + "/results/public_baselines_text_cache_20261001"
MINILM = "/home/pc/models/all-MiniLM-L6-v2"
FILES = (
    "code/v2_llm_fusion/x_conditioned_payload_partition.py",
    "code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py",
)


def connect() -> paramiko.SSHClient:
    password = os.environ.get("TAG_REMOTE_PASSWORD")
    if not password:
        raise RuntimeError("TAG_REMOTE_PASSWORD is not set")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("100.126.175.112", username="pc", password=password,
                   timeout=30)
    return client


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
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


def remote_exec(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=3600)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def deploy(client: paramiko.SSHClient) -> dict[str, str]:
    status, _, err = remote_exec(
        client,
        f"rm -rf {shlex.quote(SNAPSHOT)} && mkdir -p {shlex.quote(REMOTE_CODE)} "
        f"&& cp -a {shlex.quote(REMOTE_ROOT + '/code/v2_llm_fusion')}/. "
        f"{shlex.quote(REMOTE_CODE)}/",
    )
    if status:
        raise RuntimeError(err)
    hashes: dict[str, str] = {}
    sftp = client.open_sftp()
    try:
        for name in FILES:
            local = ROOT / name
            remote = SNAPSHOT + "/" + name.replace("\\", "/")
            mkdirs(sftp, posixpath.dirname(remote))
            sftp.put(str(local), remote)
            hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()
    finally:
        sftp.close()
    return hashes


def launch(client: paramiko.SSHClient, datasets: list[str], gpu: int,
           output_name: str, student_steps: int | None,
           student_seeds: list[int], cat_steps: int,
           minilm: str, cache_root: str, arms: list[str]) -> dict[str, str]:
    output = REMOTE_ROOT + "/results/" + output_name
    log = REMOTE_ROOT + "/results/" + output_name + ".log"
    args = [
        PYTHON,
        "-u",
        "code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py",
        "--datasets", *datasets,
        "--source-root", SOURCE_ROOT,
        "--xpa-root", SOURCE_ROOT,
        "--output-root", output,
        "--minilm", minilm,
        "--text-cache-root", cache_root,
        "--device", f"cuda:{gpu}",
        "--arms", *arms,
        "--student-seeds", *(str(v) for v in student_seeds),
        "--cat-steps", str(cat_steps),
        "--construction-seed", "270927",
    ]
    if student_steps is not None:
        args.extend(("--student-steps", str(student_steps)))
    command = (
        f"cd {shlex.quote(SNAPSHOT)} && "
        f"export PYTHONPATH={shlex.quote(REMOTE_CODE)}:$PYTHONPATH && "
        f"mkdir -p {shlex.quote(REMOTE_ROOT + '/results')} && "
        f"nohup env PYTHONUNBUFFERED=1 {' '.join(shlex.quote(v) for v in args)} "
        f"> {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
    )
    status, out, err = remote_exec(client, command)
    if status:
        raise RuntimeError(err or out)
    return {"output": output, "log": log, "pid": out.strip(),
            "datasets": ",".join(datasets), "gpu": str(gpu)}


def fetch_tree(client: paramiko.SSHClient, remote: str, local: Path) -> int:
    count = 0
    sftp = client.open_sftp()

    def walk(src: str, dst: Path) -> None:
        nonlocal count
        dst.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(src):
            rpath = posixpath.join(src, entry.filename)
            lpath = dst / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                walk(rpath, lpath)
            elif entry.filename.endswith((".json", ".log", ".md")):
                sftp.get(rpath, str(lpath))
                count += 1

    try:
        walk(remote, local)
    finally:
        sftp.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "launch", "fetch", "status"))
    parser.add_argument("--datasets", nargs="+", default=["arxiv"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-name", default="x_slot_partition_large_20261003_gpu0")
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--minilm", default=MINILM)
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument(
        "--arms", nargs="+",
        default=["cat_mmd_tag_x_slot_partition"],
        choices=[
            "cat_mmd_tag_x_slot_partition",
            "cat_mmd_tag_x_slot_partition_st",
            "cat_mmd_tag_x_slot_partition_anchor",
            "cat_mmd_tag_x_slot_partition_full",
        ],
    )
    parser.add_argument("--local", type=Path)
    args = parser.parse_args()

    client = connect()
    try:
        if args.action == "deploy":
            print(deploy(client))
        elif args.action == "launch":
            print(launch(client, args.datasets, args.gpu, args.output_name,
                         args.student_steps, args.student_seeds,
                         args.cat_steps, args.minilm, args.cache_root,
                         args.arms))
        elif args.action == "fetch":
            if args.local is None:
                raise ValueError("--local is required for fetch")
            remote = REMOTE_ROOT + "/results/" + args.output_name
            print({"files": fetch_tree(client, remote, args.local)})
        else:
            remote_log = REMOTE_ROOT + "/results/" + args.output_name + ".log"
            status, out, err = remote_exec(
                client,
                f"ps -ef | grep -F {shlex.quote('run_public_continual_baselines_large_20261001.py')} "
                f"| grep -F {shlex.quote(args.output_name)} | grep -v grep || true; "
                f"tail -30 {shlex.quote(remote_log)} 2>/dev/null || true",
            )
            print(out, end="")
            print(err, end="")
    finally:
        client.close()


if __name__ == "__main__":
    main()
