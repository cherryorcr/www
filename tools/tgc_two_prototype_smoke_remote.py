"""Deploy the controlled shared two-prototype smoke experiment."""
from pathlib import Path
import os

import phase12_remote as remote


ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_two_prototype_smoke_20260924"
remote.FILES = [
    str(path.relative_to(ROOT)).replace("\\", "/")
    for folder in (ROOT / "code" / "v2_llm_fusion", ROOT / "code" / "frozen_v1_2026_09_09")
    for path in sorted(folder.glob("*.py"))
]
remote.FILES += [
    "code/v2_llm_fusion/run_tgc_two_prototype_smoke.py",
    "docs/current/TGC_P0_PLASTICITY_ATTRIBUTION_ZH_2026-09-24.md",
]


if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
