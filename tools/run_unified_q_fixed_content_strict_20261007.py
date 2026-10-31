"""Retired full replay launcher; preserve provenance instead of rerunning."""

if __name__ == "__main__":
    raise SystemExit(
        "Full callback-RNG replay was not counted: startup attempts failed "
        "before construction. Use probe_unified_q_replay_20261007.py for the "
        "numerical audit; the frozen 117-cell matrix is reported separately."
    )
