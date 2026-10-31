"""Record a diagnostic-only provenance fix before any diagnostic execution."""
import hashlib
import json
from pathlib import Path
import time

root=Path("/home/pc/research/TAG_condensation")
output=root/"results/unified_q_inloop_validation_20261007"
if list(output.glob("diagnostics_*/results.json")):
    raise SystemExit("cannot amend after diagnostics start")
path=output/"diagnostics_plan.json"
plan=json.loads(path.read_text())
source=root/"experiments/unified_q_inloop_validation_20261007/code/run_unified_q_fixed_content_20261007.py"
current=hashlib.sha256(source.read_bytes()).hexdigest()
if current != plan["runner_sha256"]:
    plan.setdefault("pre_execution_amendments",[]).append(dict(time=time.time(),
        old_sha256=plan["runner_sha256"],new_sha256=current,
        reason="Keep source IDs aligned with CCTP/blank/permuted diagnostic payloads; candidate and objectives unchanged"))
    plan["runner_sha256"]=current
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan,indent=2))
    temporary.replace(path)
print(json.dumps(dict(status="amended-before-execution",runner_sha256=current)))
