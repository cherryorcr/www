"""Deploy the isolated R17 teacher-message topology experiment snapshot."""
from pathlib import Path
import os

import phase12_remote as remote

ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/" + os.environ.get(
    "TGC_R17_SNAPSHOT", "tgc_r17_message_topology_20260924_snapshot")
remote.FILES = [
    str(path.relative_to(ROOT)).replace("\\", "/")
    for folder in (ROOT / "code" / "v2_llm_fusion", ROOT / "code" / "frozen_v1_2026_09_09")
    for path in sorted(folder.glob("*.py"))
]
remote.FILES += [
    "tools/run_tgc_r5_pipeline.py",
    "docs/current/TGC_R17_MESSAGE_TOPOLOGY_PROTOCOL_ZH_2026-09-24.md",
]

if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
