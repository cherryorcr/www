"""Fresh-interpreter tests avoid module-cache contamination of the legacy check."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
    PYTHONPATH=os.pathsep.join(map(str, (HERE / "implementation",
        ROOT / "experiments/uq_final_evidence_20261012/code", ROOT / "code/v2_llm_fusion"))))
tests = [[HERE / "implementation/test_unified_q_blocks.py", HERE / "implementation/test_unified_q_multimodal.py"],
         [HERE / "test_legacy_regression.py"]]
rows = []
for paths in tests:
    p = subprocess.run([sys.executable, "-m", "pytest", *map(str, paths), "-q", "--tb=short"],
                       cwd=HERE, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, flush=True)
    rows.append(dict(tests=list(map(str, paths)), returncode=p.returncode, output=p.stdout))
(HERE / "verification.json").write_text(json.dumps(rows, indent=2))
raise SystemExit(int(any(row["returncode"] for row in rows)))
