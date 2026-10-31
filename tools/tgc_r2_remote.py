"""Use the existing SSH helper against an isolated TGC-R2 snapshot."""
import phase12_remote as remote


if __name__ == "__main__":
    remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_r2_20260922_snapshot"
    remote.main()
