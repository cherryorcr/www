"""Audit stage-class pool counts for the frozen public protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--groups", type=str, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.runner_dir))
    import run_unified_v3_continual as unified  # pylint: disable=import-outside-toplevel
    import run_public_continual_baselines_large_20261001 as runner  # pylint: disable=import-outside-toplevel

    groups = [[int(value) for value in group.split(",")] for group in args.groups.split(";")]
    data = unified.load_unified(
        args.dataset,
        str(args.data),
        str(args.minilm),
        "cpu",
        n_stages=5,
        split_mode="base_incremental",
        base_fraction=0.4,
        report_dev_mod=3,
    )
    counts = [len(ids) for ids in runner.stage_ids(data, groups)]
    print(f"dataset={args.dataset} counts={counts} total={sum(counts)}", flush=True)


if __name__ == "__main__":
    main()
