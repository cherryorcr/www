"""Controlled C1 multi-prototype allocation experiment on small graphs.

This wrapper reuses the audited multi-order runner and changes only the
construction component under test:

* dispersion + Mean-X: frozen C0 control;
* subcluster-complexity + Mean-X;
* subcluster-complexity + Weighted-X;
* subcluster-complexity + Moment-X.

The student protocol, budgets, topology policy, text recipe, class orders and
student seeds remain those of ``run_small_order_cost_confirmation``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import run_small_order_cost_confirmation as B
import run_unified_v3_continual as U


B.__file__ = __file__
B.ARMS = [
    "dispersion",
    "subcluster_mean",
    "subcluster_weighted",
    "subcluster_moment",
]
B.P1X_ARMS = ["subcluster_mean", "subcluster_weighted", "subcluster_moment"]


def _arm_config(arm: str) -> tuple[str, str]:
    if arm == "dispersion":
        return "dispersion", "mean"
    if arm == "subcluster_mean":
        return "subcluster_complexity", "mean"
    if arm == "subcluster_weighted":
        return "subcluster_complexity", "weighted"
    if arm == "subcluster_moment":
        return "subcluster_complexity", "moment"
    raise ValueError(f"unknown C1 arm: {arm}")


def construct_synthetic(data, arm, stages, budgets, out, device, canonical):
    """Construct and audit one arm with immutable old memory."""
    allocation_mode, x_operator = _arm_config(arm)
    memories, audits, old = [], [], None
    for stage, ids in enumerate(stages):
        path = out / f"{arm}_s{stage}_memory.pt"
        audit_path = out / f"{arm}_s{stage}_construction.json"
        if path.exists() and audit_path.exists():
            mem = torch.load(path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if B.C.tensor_hash(mem) != audit["tensor_sha256"]:
                raise ValueError(f"stale C1 memory artifact: {path}")
        else:
            keep = len(old["x"]) if old is not None else 0
            count = int(budgets[stage]) - int(keep)
            if count <= 0:
                raise ValueError("C1 requires a growing fixed-rate budget")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            block = U.build_stage_block(
                data, ids, count, data["encoder"], stage,
                "joint_g5_c320", "concat", device, allocation_mode,
                data["stage_classes"][stage], 0.5, 0.5,
                x_operator=x_operator)
            mem = U.SCA.append_memory(
                old, block, keep, topology_mode="frozen_old", cross_degree=1,
                refresh_mode="frozen_old", device=device)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            allocation = block.get("allocation")
            audit = {
                "arm": arm,
                "stage": stage,
                "keep": int(keep),
                "new": int(count),
                "allocation_mode": allocation_mode,
                "x_operator": x_operator,
                "allocation": allocation,
                "subcluster_count": block.get("subcluster_count"),
                "actual_rows": int(len(block["x"])),
                "build_seconds": float(elapsed),
                "selector_seconds": 0.0,
                "selector_runs": 0,
                "selector_steps": 0,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "tensor_sha256": B.C.tensor_hash(mem),
                "payload": B.payload_stats(mem),
                "synthetic_only": True,
            }
            if canonical and arm == "dispersion":
                audit["canonical_regression"] = B.check_canonical_memory(
                    data["dataset"], arm, stage, mem)
            torch.save(mem, path)
            B.C.save_json(audit_path, audit)
        B.C.assert_frozen(old, mem)
        if len(mem["x"]) != int(budgets[stage]):
            raise AssertionError("C1 synthetic budget mismatch")
        memories.append(mem)
        audits.append(audit)
        old = mem
    return memories, audits


B.construct_synthetic = construct_synthetic


def main() -> None:
    B.main()
    # The imported runner owns argument parsing.  Read its output path from
    # argv so the C1 descriptor sits beside the immutable runner artifacts.
    import sys
    output = None
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        descriptor = {
            "protocol": "c1-multi-prototype-small-v1",
            "control": "dispersion + Mean-X",
            "candidates": {
                "subcluster_mean": "subcluster_complexity + Mean-X",
                "subcluster_weighted": "subcluster_complexity + Weighted-X",
                "subcluster_moment": "subcluster_complexity + Moment-X",
            },
            "changed_components": ["within-class subcluster allocation", "synthetic X operator"],
            "fixed_components": [
                "5% fixed-rate growing budget", "joint_g5_c320 concat",
                "frozen-old topology", "student GCN384", "InfoNCE",
                "batch=16", "student seeds=100..105",
            ],
            "source_labels": "class labels define protocol groups only; construction uses pool rows",
            "test_labels_evaluated": False,
        }
        (output / "c1_descriptor.json").write_text(
            json.dumps(descriptor, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
