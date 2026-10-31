"""Predeclared in-loop CaT distribution matching, without CaT warm-start."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

import run_unified_q_diagnostics_20261007 as D


ARMS = ("cat_cctp", "uq_v3", "uq_dm_joint", "uq_edgefree384", "uq_dm_edgefree",
        "uq_dm_edgefree_no_gm", "uq_dm_edgefree_fixed_q")


def config(arm, seed, steps=30, max_sinkhorn=2048):
    if arm not in ARMS:
        raise ValueError(arm)
    base = D.config("uq_v3", seed, steps, max_sinkhorn)
    if "edgefree" in arm:
        base = replace(base, topology_refine_swaps=0)
    if arm.startswith("uq_dm"):
        base = replace(base, lambda_distribution=1.0)
    if arm.endswith("no_gm"):
        base = replace(base, lambda_gradient=0.0)
    if arm.endswith("fixed_q"):
        base = replace(base, train_q=False)
    return base


def use_topology(arm):
    return arm in ("uq_v3", "uq_dm_joint")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis", "arxiv"), required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--budget-mode", choices=("fixed_rate", "fixed_schedule"), default="fixed_rate")
    p.add_argument("--order-seed", type=int)
    p.add_argument("--construction-seed", type=int, default=271006)
    p.add_argument("--student-seeds", nargs="+", type=int, default=[150, 151, 152])
    p.add_argument("--student-steps", type=int, default=2048)
    p.add_argument("--q-steps", type=int, default=30)
    p.add_argument("--sinkhorn-max-steps", type=int, default=2048)
    p.add_argument("--max-stages", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(2)
    D.run(args, config_fn=config, topology_fn=use_topology,
          protocol_name="unified-q-inloop-cat-distribution-v1", primary="uq_dm_edgefree",
          extra_modules=(sys.modules[__name__],))


if __name__ == "__main__":
    main()
