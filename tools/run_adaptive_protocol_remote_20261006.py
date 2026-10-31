"""Deploy and launch the frozen Adaptive coherent protocol remotely.

The helper keeps credentials out of source code.  Set TAG_REMOTE_PASSWORD in
the calling environment before using it.
"""
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
SNAPSHOT = REMOTE_ROOT + "/work_runs/adaptive_protocol_20261006_snapshot"
PYTHON = "/home/pc/miniconda3/envs/tagsam/bin/python"
FILES = (
    "code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py",
    "code/v2_llm_fusion/run_prompt_zero_shot_small_20261006.py",
    "code/v2_llm_fusion/x_conditioned_payload_routing.py",
    "tools/adaptive_large_batch_20261006.sh",
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
    command = (
        f"rm -rf {shlex.quote(SNAPSHOT)} && "
        f"mkdir -p {shlex.quote(SNAPSHOT + '/code/v2_llm_fusion')} && "
        f"cp -a {shlex.quote(REMOTE_ROOT + '/code/v2_llm_fusion')}/. "
        f"{shlex.quote(SNAPSHOT + '/code/v2_llm_fusion')}/"
    )
    status, out, err = remote_exec(client, command)
    if status:
        raise RuntimeError(err or out)
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
    return {"snapshot": SNAPSHOT, **hashes}


def launch(client: paramiko.SSHClient, command_args: list[str],
           output_name: str) -> dict[str, str]:
    output = REMOTE_ROOT + "/results/" + output_name
    log = REMOTE_ROOT + "/results/" + output_name + ".log"
    args = [PYTHON, "-u", *command_args]
    command = (
        f"cd {shlex.quote(SNAPSHOT)} && "
        f"export PYTHONPATH={shlex.quote(SNAPSHOT + '/code/v2_llm_fusion')}:$PYTHONPATH && "
        f"mkdir -p {shlex.quote(REMOTE_ROOT + '/results')} && "
        f"nohup env PYTHONUNBUFFERED=1 "
        f"{' '.join(shlex.quote(v) for v in args)} "
        f"> {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
    )
    status, out, err = remote_exec(client, command)
    if status:
        raise RuntimeError(err or out)
    return {"output": output, "log": log, "pid": out.strip()}


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
            elif entry.filename.endswith((".json", ".csv", ".md", ".log")):
                sftp.get(rpath, str(lpath))
                count += 1

    try:
        walk(remote, local)
    finally:
        sftp.close()
    return count


def fetch_adaptive_summaries(client: paramiko.SSHClient, local: Path) -> int:
    """Fetch only compact summary files from the completed large-graph cells."""
    remote_root = REMOTE_ROOT + "/results"
    sftp = client.open_sftp()
    count = 0
    try:
        for name in sorted(sftp.listdir(remote_root)):
            if not (name.startswith("adaptive_order_")
                    and name.endswith("_20261006")):
                continue
            source = posixpath.join(remote_root, name, "summary.json")
            try:
                sftp.stat(source)
            except IOError:
                continue
            target = local / name / "summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(source, str(target))
            count += 1
    finally:
        sftp.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "upload", "launch-large",
                                             "launch-small", "launch-batch",
                                             "status", "fetch",
                                             "fetch-summaries"))
    parser.add_argument("--dataset", default="arxiv")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-name", required=False)
    parser.add_argument("--source-root", required=False)
    parser.add_argument("--cache-root", required=False)
    parser.add_argument(
        "--minilm",
        default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
    )
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--construction-seed", type=int, default=270927)
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--arms", nargs="+", required=False)
    parser.add_argument("--orders", nargs="+",
                        default=["order19611", "order19612", "order19613"])
    parser.add_argument("--local", type=Path)
    parser.add_argument("--remote-output")
    parser.add_argument("--student-steps", type=int)
    args = parser.parse_args()

    client = connect()
    try:
        if args.action == "deploy":
            print(deploy(client))
            return
        if args.action == "upload":
            local = ROOT / "tools/adaptive_large_batch_20261006.sh"
            remote = SNAPSHOT + "/tools/adaptive_large_batch_20261006.sh"
            sftp = client.open_sftp()
            try:
                mkdirs(sftp, posixpath.dirname(remote))
                sftp.put(str(local), remote)
            finally:
                sftp.close()
            print({"remote": remote,
                   "sha256": hashlib.sha256(local.read_bytes()).hexdigest()})
            return
        if args.action == "fetch":
            if not args.local or not args.remote_output:
                raise ValueError("fetch requires --local and --remote-output")
            remote = REMOTE_ROOT + "/results/" + args.remote_output
            print({"files": fetch_tree(client, remote, args.local)})
            return
        if args.action == "fetch-summaries":
            if not args.local:
                raise ValueError("fetch-summaries requires --local")
            print({"files": fetch_adaptive_summaries(client, args.local)})
            return
        if args.action == "status":
            name = args.output_name or "adaptive_protocol"
            log = REMOTE_ROOT + "/results/" + name + ".log"
            command = (
                f"ps -ef | grep -F {shlex.quote(name)} | grep -v grep || true; "
                f"tail -40 {shlex.quote(log)} 2>/dev/null || true"
            )
            _, out, err = remote_exec(client, command)
            print(out, end="")
            print(err, end="")
            return

        if not args.output_name:
            raise ValueError("launch requires --output-name")
        if args.action == "launch-large":
            if not args.source_root or not args.cache_root:
                raise ValueError("large launch requires --source-root and --cache-root")
            arms = args.arms or [
                "cat_mmd_tag_cctp_top5",
                "cat_mmd_tag_x_conditioned_routing_coherent",
                "cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
                "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor",
                "gecc_hard_tag", "k_center", "medoid", "random_real",
            ]
            command_args = [
                "code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py",
                "--datasets", args.dataset,
                "--source-root", args.source_root,
                "--xpa-root", args.source_root,
                "--output-root", REMOTE_ROOT + "/results/" + args.output_name,
                "--minilm", args.minilm,
                "--text-cache-root", args.cache_root,
                "--device", f"cuda:{args.gpu}",
                "--arms", *arms,
                "--student-seeds", *(str(v) for v in args.student_seeds),
                "--construction-seed", str(args.construction_seed),
            ]
            if args.order_seed is not None:
                command_args.extend(("--order-seed", str(args.order_seed)))
        elif args.action == "launch-small":
            arms = args.arms or [
                "cat_nearest",
                "formal_prompt_margin",
                "x_conditioned_routing_coherent",
                "x_conditioned_routing_coherent_adaptive",
                "x_conditioned_routing_coherent_anchor",
            ]
            cache_root = args.cache_root or (
                REMOTE_ROOT + "/results/p5_rate_curve_small_20260919")
            command_args = [
                "code/v2_llm_fusion/run_prompt_zero_shot_small_20261006.py",
                "--output-root", REMOTE_ROOT + "/results/" + args.output_name,
                "--cache-root", cache_root,
                "--minilm", args.minilm,
                "--device", f"cuda:{args.gpu}",
                "--datasets", "cora", "citeseer",
                "--orders", *args.orders,
                "--arms", *arms,
                "--report-seeds", *(str(v) for v in args.student_seeds),
                "--construction-seed", str(args.construction_seed),
            ]
            if args.student_steps is not None:
                command_args.extend(("--student-steps", str(args.student_steps)))
        else:
            datasets = args.datasets or ["arxiv", "sportsfit"]
            batch_log = REMOTE_ROOT + "/results/" + args.output_name + ".log"
            command = (
                f"cd {shlex.quote(SNAPSHOT)} && "
                f"chmod +x tools/adaptive_large_batch_20261006.sh && "
                f"nohup tools/adaptive_large_batch_20261006.sh {int(args.gpu)} "
                f"{' '.join(shlex.quote(v) for v in datasets)} "
                f"> {shlex.quote(batch_log)} 2>&1 < /dev/null & echo $!"
            )
            status, out, err = remote_exec(client, command)
            if status:
                raise RuntimeError(err or out)
            print({"batch_log": batch_log, "pid": out.strip(),
                   "datasets": datasets, "gpu": int(args.gpu)})
            return
        print(launch(client, command_args, args.output_name))
    finally:
        client.close()


if __name__ == "__main__":
    main()
