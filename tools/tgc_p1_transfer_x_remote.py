"""Upload/execute P1 transfer-aware X optimization in the R15 snapshot."""
from pathlib import Path
import os

import phase12_remote as remote


ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/" + os.environ.get(
    "TGC_P1_SNAPSHOT", "tgc_r15_r11_cross_20260924_snapshot")
remote.FILES = ["code/v2_llm_fusion/run_tgc_p1_transfer_x.py"]


if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
