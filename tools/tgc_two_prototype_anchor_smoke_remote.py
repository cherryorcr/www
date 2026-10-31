"""Deploy and run the C0-anchor two-prototype smoke experiment."""
from pathlib import Path
import os

import phase12_remote as remote


ROOT = Path(__file__).resolve().parents[1]
# Reuse the already deployed two-prototype snapshot.  Only the new runner is
# uploaded for this follow-up, so deployment does not resync the full v2 tree.
remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_two_prototype_smoke_20260924"
remote.FILES = []


if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
