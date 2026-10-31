"""Transport into a separate large-LP snapshot, leaving old experiments intact."""
import link_prediction_remote as L

L.R.SNAPSHOT = L.SNAPSHOT
L.SNAPSHOT = L.R.REMOTE_ROOT + "/work_runs/large_lp_20260920_snapshot"

if __name__ == "__main__":
    L.main()
