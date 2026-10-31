"""Reuse the SSH transport for the isolated fixed-prefix experiment."""
import phase12_remote as remote


if __name__ == "__main__":
    remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_r2b2_20260922_snapshot"
    remote.main()
