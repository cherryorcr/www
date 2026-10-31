"""Stage-2 source/composition attribution for a fixed stage-1 memory.

This script is deliberately a diagnostic, not a new continual algorithm.  It
keeps one previously exported stage-1 memory fixed and compares raw stage-2
data, a matched-size raw subset, and an already condensed stage-2 block under
the same independent-student evaluator.  No confirmation partition is read.
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
    if "training_labels_used" in metadata and metadata["training_labels_used"]:
        raise ValueError("memory metadata claims labels were used")
    if len(memory["x"]) < 2:
        raise ValueError("memory must contain at least two rows")
    if int(memory["stage"].max()) > max_stage:
        raise ValueError("memory contains a future stage")
    cleaned = clean_data(memory, "cpu")
    cleaned["metadata"] = metadata
    return cleaned


def matched_raw(delta: dict, count: int, seed: int) -> dict:
    if count < 2 or count > len(delta["x"]):
        raise ValueError(f"matched raw count must be in [2, {len(delta['x'])}]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(delta["x"]), generator=generator)[:count]
    return subset_graph(delta, indices)


def build_args(base_config: dict, output: str, model_seeds: list[int], check_every: int):
    config = dict(base_config)
    config.update(
        output=output,
        model_seeds=model_seeds,
        check_every=check_every,
        confirm=False,
        phase="continual",
        stages=3,
        diagnostic_mode="stage2_source_composition",
        arms=[],
    )
    return argparse.Namespace(**config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True,
                        help="completed preserve run_config.json")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--old-artifact", required=True,
                        help="fixed cumulative stage-1 memory artifact")
    parser.add_argument("--append-artifact", required=True,
                        help="cumulative stage-2 append artifact for comparison")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--raw-seed", type=int, default=4402)
    args = parser.parse_args()

    base_config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    if args.device is not None:
        base_config["device"] = args.device
    base_config["prepared"] = args.prepared
    exp_args = build_args(base_config, args.output, args.model_seeds, args.check_every)
    exp = Experiment(exp_args)

    stages = [exp.load_stage(i) for i in range(3)]
    manifest_sha = file_sha(exp.prepared / "manifest.json")
    old_path = Path(args.old_artifact)
    append_path = Path(args.append_artifact)
    old = check_memory(load_graph(old_path), manifest_sha, 1)
    append_final = check_memory(load_graph(append_path), manifest_sha, 2)
    if int(append_final["stage"].max()) != 2:
        raise ValueError("append artifact must contain a stage-2 block")

    delta = stages[2]
    full = concat_graphs(stages)
    source = concat_graphs((old, delta))
    cumulative_budget = math.floor(exp.args.rate * len(full["x"]))
    new_budget = cumulative_budget - len(old["x"])
    if new_budget < 2:
        raise ValueError("stage-2 increment budget cannot support contrastive training")

    # The stage-2 rows of an append artifact are re-indexed before concatenating
    # them with the fixed old memory.  This removes any accidental dependence on
    # the old rows exported by the recursive run.
    append_new = subset_graph(append_final,
                              torch.where(append_final["stage"] == 2)[0])
    if len(append_new["x"]) != new_budget:
        raise ValueError("append artifact has a different stage-2 row budget")

    raw_match = matched_raw(delta, new_budget, args.raw_seed)
    arms = {
        "full_history": full,
        "old_memory_only": old,
        "new_raw_over_budget": delta,
        "old_plus_new_raw_over_budget": source,
        "old_plus_new_raw_matched": concat_graphs((old, raw_match)),
        "old_plus_new_compact": concat_graphs((old, append_new)),
    }
    for name, graph in arms.items():
        exp.record(graph, 2, name, {
            "full_history": "full_history_reference",
            "old_memory_only": "stage1_memory_only_diagnostic",
            "new_raw_over_budget": "stage2_delta_over_budget_diagnostic",
            "old_plus_new_raw_over_budget": "memory_plus_raw_delta_over_budget",
            "old_plus_new_raw_matched": "same_budget_raw_delta_control",
            "old_plus_new_compact": "same_budget_condensed_delta",
        }[name])

    dump(exp.output / "diagnostic_inputs.json", {
        "mode": "stage2_source_composition",
        "confirmation_accessed": False,
        "base_config": str(Path(args.base_config)),
        "base_config_sha256": file_sha(Path(args.base_config)),
        "old_artifact": str(old_path),
        "old_artifact_sha256": file_sha(old_path),
        "append_artifact": str(append_path),
        "append_artifact_sha256": file_sha(append_path),
        "stage2_raw_roots": len(delta["x"]),
        "cumulative_budget": cumulative_budget,
        "old_roots": len(old["x"]),
        "new_budget": new_budget,
        "matched_raw_seed": args.raw_seed,
        "matched_raw_origin_ids": raw_match["origin_ids"].tolist(),
        "interpretation": "paired composition diagnostics; differences are not additive causal effects",
    })
    dump(exp.output / "completed.json", {"status": "complete", "confirmation_accessed": False})


if __name__ == "__main__":
    main()
