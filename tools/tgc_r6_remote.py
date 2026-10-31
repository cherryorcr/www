"""Isolated R6 deployment; does not modify prior snapshots."""
from pathlib import Path
import os
import phase12_remote as remote

ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/" + os.environ.get("TGC_R6_SNAPSHOT", "tgc_r6_confirmation_20260923_snapshot")
remote.FILES = [str(p.relative_to(ROOT)).replace("\\", "/")
                for folder in (ROOT / "code/v2_llm_fusion", ROOT / "code/frozen_v1_2026_09_09")
                for p in sorted(folder.glob("*.py"))]
remote.FILES += ["tools/run_tgc_r5_pipeline.py", "results/tgc_r6_preparation_20260923/history_selection.json",
                 "docs/current/TGC_R6_FROZEN_EXPERIMENTS_ZH_2026-09-23.md"]

if __name__ == "__main__":
    remote.main()
