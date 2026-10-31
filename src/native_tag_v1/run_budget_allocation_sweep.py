"""P3 budget allocation sweep (diagnostic, not a new method).

Fixed cumulative stage-2 budget (rate * cumulative train roots). We sweep how
that budget is split between the old stage-1 memory and a fresh stage-2
compact block:

- old memory is SHRUNK by per-stage medoid-style selection on its own rows
  (keeps original X/text/pairing; only drops rows), never re-optimized;
- the freed rows go to a larger new compact block built by the audited
  initializer (joint_medoid + link predictor), no ASM;
- arms include the status-quo split (old 67 / new 27) as anchor and a
  class-balanced-ish variant (equal per stage).

This isolates "budget squeeze on the new classes" from all other continual
losses. Dev/checkpoint only; no confirmation partition is read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from native_core import clean_data
from protocol_data import concat_graphs, subset_graph
from run_native_experiments import Experiment, dump, file_sha


def load_graph(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def check_memory(memory: dict, manifest_sha: str, max_stage: int) -> dict:
    metadata = memory.get("metadata", {})
    if metadata.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("memory was made from a different prepared manifest")
    if metadata.get("training_labels_used"):
        raise ValueError("memory metadata claims labels were used")
    if int(memory["stage"].max()) > max_stage:
        raise ValueError("memory contains a future stage")
    cleaned = clean_data(memory, "cpu")
    cleaned["metadata"] = metadata
    return cleaned


def shrink_memory(old: dict, keep: int, seed: int) -> dict:
    """Keep ``keep`` rows of the old memory, proportional per stage.

    Selection is norm-central per stage (rows whose X is closest to the
    stage mean X), a deterministic, label-free heuristic. X/text/pairing of
    kept rows are unchanged.
    """
    n = len(old["x"])
    if keep >= n:
        return old
    if keep < 2:
        raise ValueError("shrunk memory needs at least two rows")
    stages = old["stage"]
    unique = sorted(set(int(s) for s in stages.tolist()))
    quota: dict[int, int] = {}
    remaining = keep
    for i, s in enumerate(unique):
        count = int((stages == s).sum())
        if i == len(unique) - 1:
            q = remaining
        else:
            q = max(1, round(keep * count / n))
            q = min(q, count, remaining - (len(unique) - 1 - i))
        quota[s] = q
        remaining -= q
    chosen: list[int] = []
    for s in unique:
        rows = torch.where(stages == s)[0]
        mean = old["x"][rows].mean(dim=0, keepdim=True)
        dist = (old["x"][rows] - mean).norm(dim=1)
        order = torch.argsort(dist)
        chosen.extend(rows[order[: quota[s]]].tolist())
    return subset_graph(old, sorted(chosen))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--old-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--old-keep", type=int, nargs="+", default=[67, 55, 47, 32])
    args = parser.parse_args()

    base_config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    if args.device is not None:
        base_config["device"] = args.device
    base_config["prepared"] = args.prepared
    config = dict(base_config)
    config.update(output=args.output, model_seeds=args.model_seeds,
                  check_every=args.check_every, confirm=False, phase="continual",
                  stages=3, diagnostic_mode="budget_allocation_sweep", arms=[])
    exp = Experiment(argparse.Namespace(**config))

    stages = [exp.load_stage(i) for i in range(3)]
    manifest_sha = file_sha(exp.prepared / "manifest.json")
    old = check_memory(load_graph(Path(args.old_artifact)), manifest_sha, 1)
    delta = stages[2]
    full = concat_graphs(stages)
    cumulative_budget = math.floor(exp.args.rate * len(full["x"]))

    report: dict[str, dict] = {}
    for keep in sorted(set(args.old_keep), reverse=True):
        if keep > len(old["x"]):
            raise ValueError(f"old-keep {keep} exceeds old memory rows {len(old['x'])}")
        new_budget = cumulative_budget - keep
        if new_budget < 2:
            raise ValueError(f"old-keep {keep} leaves no room for a new block")
        shrunk = shrink_memory(old, keep, exp.args.seed)
        teacher = exp.teacher(delta, 2, f"alloc_keep{keep}")
        new_block = exp.initializer(delta, new_budget, 2, teacher)
        arm = f"alloc_old{keep}_new{new_budget}"
        graph = concat_graphs((shrunk, new_block))
        exp.record(graph, 2, arm, "budget_allocation_diagnostic")
        report[arm] = {"old_keep": keep, "new_budget": new_budget,
                       "old_rows_kept_origin_ids": shrunk["origin_ids"].tolist()}
        del teacher

    dump(exp.output / "diagnostic_inputs.json", {
        "mode": "budget_allocation_sweep",
        "confirmation_accessed": False,
        "cumulative_budget": cumulative_budget,
        "old_artifact": args.old_artifact,
        "old_artifact_sha256": file_sha(Path(args.old_artifact)),
        "old_rows_total": len(old["x"]),
        "stage2_raw_roots": len(delta["x"]),
        "allocations": report,
        "interpretation": "same total budget, different old/new split; old memory only shrunk, never re-optimized",
    })
    dump(exp.output / "completed.json", {"status": "complete", "confirmation_accessed": False})


if __name__ == "__main__":
    main()
