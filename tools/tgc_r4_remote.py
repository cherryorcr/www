"""Deploy and run the isolated TGC-R4 relational-response experiment."""
from pathlib import Path
import os

import phase12_remote as remote

ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_r4_relational_20260923c_snapshot"
remote.FILES = [
    str(path.relative_to(ROOT)).replace("\\", "/")
    for folder in (ROOT / "code" / "v2_llm_fusion", ROOT / "code" / "frozen_v1_2026_09_09")
    for path in sorted(folder.glob("*.py"))
]
remote.FILES += [
    "docs/current/TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md",
]


if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
