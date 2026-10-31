"""Record Linux high-water RSS of active R4 workers; not a construction profiler."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "run_tgc_r4_relational.py" not in command or str(args.root) not in command:
                continue
            fields = dict(line.split(":", 1) for line in (folder / "status").read_text().splitlines() if ":" in line)
            records.append({"pid": int(folder.name), "command": command,
                            "vm_hwm_kib": int(fields["VmHWM"].split()[0]),
                            "vm_rss_kib": int(fields["VmRSS"].split()[0])})
        except (OSError, KeyError, ValueError):
            continue
    result = {"observed_at_unix": time.time(), "scope": "active worker high-water RSS up to this instant; not whole-run construction peak",
              "records": records}
    path = args.root / f"worker_rss_{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
