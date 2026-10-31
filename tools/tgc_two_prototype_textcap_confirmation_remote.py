"""Deploy and run the matched 320/324-character confirmation."""
from pathlib import Path
import os

import phase12_remote as remote


ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_two_prototype_smoke_20260924"
remote.FILES = []


if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
