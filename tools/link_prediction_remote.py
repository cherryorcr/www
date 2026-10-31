"""Isolated LP snapshot transport; credentials are environment-only."""
import argparse
import json
from pathlib import Path
import shlex

import safe_joint_kernel_remote as R

SNAPSHOT = R.REMOTE_ROOT + "/work_runs/link_prediction_20260920_snapshot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "upload", "exec", "fetch"))
    parser.add_argument("--files", nargs="+", default=[])
    parser.add_argument("--command")
    parser.add_argument("--remote")
    parser.add_argument("--local")
    args = parser.parse_args()
    client = R.connect()
    try:
        sftp = client.open_sftp()
        if args.action in ("prepare", "exec"):
            command = args.command
            if args.action == "prepare":
                script = ("import shutil; from pathlib import Path; "
                          f"p=Path({SNAPSHOT!r}); "
                          f"shutil.copytree({R.SNAPSHOT!r},p) if not p.exists() else None")
                command = "/home/pc/miniconda3/envs/tagsam/bin/python -c " + shlex.quote(script)
            _, stdout, stderr = client.exec_command(command, timeout=7200)
            for line in stdout:
                print(line, end="", flush=True)
            error = stderr.read().decode(errors="replace")
            if error:
                print(error, end="", flush=True)
            raise SystemExit(stdout.channel.recv_exit_status())
        if args.action == "upload":
            hashes = {}
            for name in args.files:
                source = (R.ROOT / name).resolve()
                source.relative_to(R.ROOT)
                target = SNAPSHOT + "/" + name.replace("\\", "/")
                R.mkdirs(sftp, str(Path(target).parent).replace("\\", "/"))
                sftp.put(str(source), target)
                hashes[name] = R.hashlib.sha256(source.read_bytes()).hexdigest()
            print(json.dumps(hashes, indent=2))
        if args.action == "fetch":
            print(json.dumps(R.fetch_archive(client, sftp, args.remote, args.local)))
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
