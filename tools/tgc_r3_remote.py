"""Deploy and run the independent TGC selector confirmation snapshot."""
from pathlib import Path
import os

import phase12_remote as remote


remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_r3_selector_20260922_snapshot"
ROOT = Path(__file__).resolve().parents[1]
remote.FILES = [
    str(path.relative_to(ROOT)).replace("\\", "/")
    for folder in (ROOT / "code" / "v2_llm_fusion", ROOT / "code" / "frozen_v1_2026_09_09")
    for path in sorted(folder.glob("*.py"))
]
remote.FILES.append("code/v2_llm_fusion/launch_tgc_r3_selector_20260922.sh")
remote.FILES.append("docs/current/TGC_R3_SELECTOR_CONFIRMATION_PROTOCOL_ZH_2026-09-22.md")


if __name__ == "__main__":
    # Paramiko tries the configured SSH keys before the optional password.
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
