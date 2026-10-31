"""Registered, one-factor TGC-R capacity diagnostics on four development cells."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import audit_tgc_revision_gate as AUD
import run_tgc_revision_gate as RUN
import tgc_revision_gate as TGC


# Protection-only controls separate eligible-set changes from quota changes.
CONFIGS = {
    "radius002": (0.80, 0.10, 0.02),
    "radius005": (0.80, 0.10, 0.05),
    "radius020": (0.80, 0.10, 0.20),
    "protect060": (0.60, 0.10, 0.10),
    "quota025": (0.60, 0.25, 0.10),
    "protect040": (0.40, 0.10, 0.10),
    "quota050": (0.40, 0.50, 0.10),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cora", "citeseer"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(RUN.P.STUDENT_SEEDS))
    ap.add_argument("--order", nargs="+", choices=RUN.ORDERS,
                    default=["canonical", "order19101"])
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--teacher-steps", type=int, default=128)
    ap.add_argument("--minilm", default=(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    ap.add_argument("--text-cache-root", default=(
        "/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    args.rate, args.arms = 0.05, ["c0", "tgc_r"]
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("duplicate report seeds")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cache = Path(args.text_cache_root) / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    data = RUN.R.load_data(args.dataset, args.minilm, args.device, cache)
    for name in args.configs:
        protected, quota, radius = CONFIGS[name]
        cfg = TGC.RevisionConfig(protected_fraction=protected,
                                 revision_fraction_of_old=quota,
                                 revision_radius=radius,
                                 teacher_steps=args.teacher_steps)
        args.output = str(root / f"{args.dataset}_{name}")
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        protocol = RUN.protocol(args, cfg)
        protocol.update(name="tgc-r2-capacity-v1", capacity_config=name,
                        result_scope="development, not independent confirmation",
                        round1_gate={"mean_final_min_pp": 0.5,
                                     "cells_at_least_minus1": 3,
                                     "mean_new_min_pp": 0.0,
                                     "build_cost_max_ratio": 3.0})
        RUN.stable_lock(out / "protocol.json", protocol)
        for order in args.order:
            RUN.run_cell(data, order, args, cfg, protocol)
        reports = [AUD.audit_cell(out / args.dataset / order) for order in args.order]
        result = {"cells": reports, "fits": sum(r["fit_count"] for r in reports),
                  "passed": len(reports) == len(args.order) and all(r["passed"] for r in reports)}
        RUN.save_json(out / "audit.json", result)
        if not result["passed"]:
            raise RuntimeError(f"capacity audit failed: {out}")
    print("TGC_R2_CAPACITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
