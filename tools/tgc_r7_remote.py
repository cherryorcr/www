"""R7 isolated deployment; credentials are read only from the process environment."""
from pathlib import Path
import os
import phase12_remote as remote

ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/" + os.environ.get("TGC_R7_SNAPSHOT", "tgc_r7_stability_20260923_snapshot")
remote.FILES = [str(p.relative_to(ROOT)).replace("\\", "/")
                for folder in (ROOT / "code/v2_llm_fusion", ROOT / "code/frozen_v1_2026_09_09")
                for p in sorted(folder.glob("*.py"))]
remote.FILES += ["tools/run_tgc_r5_pipeline.py", "docs/current/TGC_R7_EXECUTION_PROTOCOL_ZH_2026-09-23.md"]
if (ROOT / "results/tgc_r7_preparation_20260923/history_selection.json").exists():
    remote.FILES += ["results/tgc_r7_preparation_20260923/history_selection.json"]

if __name__ == "__main__":
    remote.main()
