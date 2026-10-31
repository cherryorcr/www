"""Take a read-only evidence snapshot; never copy dense memories or alter runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=25)
    return dict(args=args, exit_code=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def collect(root):
    expansion = "results/uq_baseline_expansion_20261013"
    final = "results/uq_final_evidence_20261012"
    directories = [expansion + "/cells", final + "/confirmation",
                   final + "/baselines", final + "/stad"]
    entries, errors = [], []
    for directory in directories:
        for path in sorted((root / directory).glob("*/results.json")):
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
                entries.append(dict(path=path.relative_to(root).as_posix(),
                    bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
                    canonical_sha256=canonical_sha(value), result=value,
                    runtime_manifest=json.loads((path.parent / "runtime_manifest.json").read_text())
                        if (path.parent / "runtime_manifest.json").exists() else None))
            except (ValueError, OSError) as error:
                errors.append(dict(path=str(path), error=str(error)))
    status_paths = [expansion + "/" + name for name in
        ("status.txt", "native_status.txt", "controls.status", "native.status")]
    status_paths += [final + "/" + name for name in
        ("baseline_status.txt", "native_baseline_queue.status", "native_retry_status.txt",
         "native_retry_queue.status", "native_parallel_status.txt",
         "native_parallel_references.status", "native_parallel_gecc.status")]
    statuses = {name: (root / name).read_text(errors="replace")
                for name in status_paths if (root / name).exists()}
    processes = command(["ps", "-eo", "pid,etime,pcpu,pmem,args"])
    processes["stdout"] = "\n".join(line for line in processes["stdout"].splitlines()
        if any(name in line for name in ("run_uq", "uq_expansion", "gecc_soft")))
    logs = sorted((root / expansion / "logs").glob("*.log"),
                  key=lambda path: path.stat().st_mtime, reverse=True)
    tails = {}
    for path in logs[:8]:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 2500))
            tails[path.relative_to(root).as_posix()] = stream.read().decode(errors="replace")
    return dict(collected_at=datetime.now(timezone.utc).isoformat(), root=str(root),
        entries=entries, errors=errors, statuses=statuses, log_tails=tails,
        processes=processes, disk=command(["df", "-h", str(root)]),
        gpu=command(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                     "--format=csv"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Snapshots are immutable; use a timestamped output")
    data = collect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output), sources=len(data["entries"]),
                          errors=len(data["errors"]))))


if __name__ == "__main__":
    main()
