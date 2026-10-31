"""Synchronize only this experiment's artifacts; never delete source results."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archives", action="store_true")
    p.add_argument("--cells", nargs="+", choices=[f"{d}_o{o}"
        for d in ("wikics", "sportsfit", "bookhis") for o in (9101,9102,9103)])
    p.add_argument("--host", default="pc@100.126.175.112")
    p.add_argument("--remote", default="/home/pc/research/TAG_condensation/results/unified_q_attribution_20261008")
    p.add_argument("--local", type=Path, default=Path("results/unified_q_attribution_20261008"))
    args = p.parse_args()
    command = ["tar", "-czf", "-", "-C", args.remote,
               "--exclude=*smoke*", "--exclude=*.log", "--exclude=*.tmp"]
    if not args.archives:
        command.append("--exclude=*.pt")
    command.extend(args.cells or ["."])
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        args.host, " ".join(shlex.quote(v) for v in command)], capture_output=True, check=True)
    root = args.local.resolve()
    count = total = 0
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode="r:gz") as bundle:
        for member in bundle:
            if member.isdir():
                continue
            target = (root/member.name).resolve()
            if (not target.is_relative_to(root) or not member.isfile()
                    or target.suffix not in (".json", ".pt")):
                raise ValueError(f"unexpected artifact {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".sync")
            temp.write_bytes(bundle.extractfile(member).read())
            temp.replace(target)
            count += 1
            total += member.size
    cells = []
    for path in sorted(root.glob("*_o*/results.json")):
        record = json.loads(path.read_text())
        current = [v for v in record["rows"] if v.get("student_family") == "prompt" or v.get("readout") == "native"]
        latest = {arm: max(v["stage"]+1 for v in current if v["arm"]==arm)
                  for arm in dict.fromkeys(v["arm"] for v in current)}
        cells.append(dict(cell=path.parent.name,status=record["status"],prompt_stages=latest))
    print(json.dumps(dict(files=count, transferred_mib=total/2**20, cells=cells),indent=2))


if __name__ == "__main__":
    main()
