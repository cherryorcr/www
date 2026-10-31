"""Isolated transport for the prospective cross-task confirmation."""
import link_prediction_remote as L

L.R.SNAPSHOT = L.R.REMOTE_ROOT + "/work_runs/large_lp_20260920_snapshot"
L.SNAPSHOT = L.R.REMOTE_ROOT + "/work_runs/degree_confirmation_20260920_snapshot"

if __name__ == "__main__":
    L.main()
