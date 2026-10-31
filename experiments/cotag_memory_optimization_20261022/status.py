import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
for path in sorted(root.glob("*/results.json")):
    r = json.loads(path.read_text())
    last = r["rows"][-1] if r["rows"] else None
    print(json.dumps(dict(run=path.parent.name, status=r["status"], stages=len(r["rows"]),
        last_peak_gib=None if last is None else last["peak_bytes"]/2**30,
        last_f1=None if last is None or last["scores"] is None else last["scores"]["f1"],
        failure=r.get("failure"))))
print(json.dumps(dict(free_gib=shutil.disk_usage(root).free/2**30)))
