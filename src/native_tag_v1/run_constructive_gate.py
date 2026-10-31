"""Constructive gate: deterministic initializer vs seeded frozen compact.

Goal is NOT a new SOTA. The gate asks for a *stable* static TAG condensation
starting point: the deterministic constructor has no condensation-seed
randomness, so its cross-seed variance is zero by construction. The gate then
verifies that determinism was not bought with quality:

pass condition:
  mean(dev over student seeds) of the deterministic artifact
     >= mean over condensation seeds of the audited frozen compact,
  and >= the WORST per-seed frozen compact + comfortable margin.

Stage-0 static setting, rate 0.05, dev/checkpoint only, no test access.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from constructive_init import deterministic_select, knn_topology
from run_native_experiments import Experiment, dump

DEFAULT_CONFIG = dict(
    phase="static", stages=1, rate=0.05, seed=44, threads=4, hidden=384,
    lr=1e-4, student_steps=512, teacher_steps=800, validation_steps=512,
    check_every=32, batch_size=20, link_steps=160, degree=1.5,
    outer_steps=0, outer_lr=0.01, inner_steps=10, inner_lr=0.005,
    inner_lr_mode="fixed", reference_size=128, outer_eval_every=20,
    norm_cap=None, confirm=False, arms=[],
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seeds", type=int, nargs="+",
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--degree", type=float, default=1.5)
    parser.add_argument("--x-modes", nargs="+", default=["centroid", "real"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update(prepared=args.prepared, output=args.output, device=args.device,
                  model_seeds=args.model_seeds, rate=args.rate, degree=args.degree)
    exp = Experiment(argparse.Namespace(**config))
    source = exp.load_stage(0)
    k = math.floor(args.rate * len(source["x"]))
    if k < 2:
        raise ValueError("budget too small")

    started = time.time()
    for x_mode in args.x_modes:
        artifact = knn_topology(deterministic_select(source, k, x_mode=x_mode),
                                degree=args.degree)
        # Determinism audit: rebuilding must give a byte-identical object.
        again = knn_topology(deterministic_select(source, k, x_mode=x_mode),
                             degree=args.degree)
        for key in ("x", "text_emb", "edge_index", "weights", "origin_ids", "stage"):
            if not torch.equal(artifact[key], again[key]):
                raise RuntimeError(f"constructor is not deterministic in field {key}")
        exp.record(artifact, 0, f"det_construct_{x_mode}", "current_delta")

    dump(exp.output / "diagnostic_inputs.json", {
        "mode": "constructive_gate_stage0",
        "confirmation_accessed": False,
        "budget_k": k,
        "x_modes": args.x_modes,
        "determinism_audit": "rebuild equality checked for all fixed fields",
        "elapsed_seconds": time.time() - started,
    })
    dump(exp.output / "completed.json", {"status": "complete", "confirmation_accessed": False})


if __name__ == "__main__":
    main()
