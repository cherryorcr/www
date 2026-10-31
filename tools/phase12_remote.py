"""Deploy and retrieve the isolated Phase-1/2 experiment, without saved credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import stat

import paramiko

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/pc/research/TAG_condensation"
SNAPSHOT = REMOTE + "/work_runs/phase12_20260920_snapshot"
FILES = ["code/v2_llm_fusion/" + name for name in (
    "protocol_v3.py", "run_multiscale_asm.py", "run_unified_v3_continual.py",
    "run_student_compatible_append.py", "run_small_comparison_audited.py",
    "small_comparison_metrics.py", "joint_kernel_append.py",
    "run_p0_random_synthetic.py", "run_frozen_baseline_joint_kernel.py",
    "test_joint_kernel_append.py")]
FILES += ["code/frozen_v1_2026_09_09/run_gate1_continual.py",
          "docs/current/PHASE12_C0_JOINT_KERNEL_FROZEN_PROTOCOL_ZH_2026-09-20.md"]


def mkdirs(sftp, path):
    try:
        sftp.stat(path)
    except FileNotFoundError:
        mkdirs(sftp, posixpath.dirname(path))
        sftp.mkdir(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["deploy", "upload", "exec", "fetch"])
    parser.add_argument("--command")
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--remote")
    parser.add_argument("--local")
    parser.add_argument("--artifacts", action="store_true")
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("100.126.175.112", username="pc",
                   password=os.environ["TAG_REMOTE_PASSWORD"], timeout=20)
    try:
        if args.action == "exec":
            _, stdout, stderr = client.exec_command(args.command, timeout=600)
            for line in stdout:
                print(line, end="", flush=True)
            errors = stderr.read().decode("utf-8", errors="replace")
            if errors:
                print(errors, flush=True)
            raise SystemExit(stdout.channel.recv_exit_status())
        sftp = client.open_sftp()
        if args.action in ("deploy", "upload"):
            hashes = {}
            for name in (FILES if args.action == "deploy" else []) + args.extra:
                source, destination = ROOT / name, SNAPSHOT + "/" + name
                mkdirs(sftp, posixpath.dirname(destination))
                sftp.put(str(source), destination)
                hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            with sftp.file(SNAPSHOT + ("/snapshot_sources.json" if args.action == "deploy" else "/audit_sources.json"), "w") as handle:
                handle.write(json.dumps(hashes, indent=2))
            print(json.dumps({"snapshot": SNAPSHOT, "files": len(hashes), "hashes": hashes}))
        else:
            target = Path(args.local)
            count = 0
            def walk(remote, local):
                nonlocal count
                local.mkdir(parents=True, exist_ok=True)
                for entry in sftp.listdir_attr(remote):
                    src, dst = posixpath.join(remote, entry.filename), local / entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        walk(src, dst)
                    elif (entry.filename.endswith((".json", ".csv", ".md", ".log"))
                          or args.artifacts and entry.filename.endswith("_memory.pt")):
                        if not dst.exists() or dst.stat().st_size != entry.st_size:
                            sftp.get(src, str(dst))
                        count += 1
            walk(args.remote, target)
            print(json.dumps({"files": count, "local": str(target)}))
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
