"""Fine-grained paired attribution for the native continual TAG prototype.

The script is deliberately a development diagnostic.  It never reads the
confirmation partition and never passes labels to condensation.  It creates
one shared initializer/teacher per contrast and changes one declared factor
at a time:

* ``old_update``: the same old memory, new block, initial X, text and topology
  are used for an all-X update and a new-X-only update.  Row drift snapshots
  are exported for the old and new rows.
* ``new_quality``: one fixed old memory and one fixed 27-row budget are used
  for matched raw rows, frozen compact rows, new-only ASM and joint ASM.  A
  protected-joint arm is included as a useful interaction control.
* ``text_pairing``: one fixed old memory and one fixed compact initialization
  are compared with paired new-block text versus the same text multiset
  shuffled across new synthetic rows.

The output is an immutable run directory with artifacts, independent student
curves, hashes and a machine-readable attribution summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from native_core import DATA_KEYS, _fingerprint, clean_data
from protocol_data import concat_graphs, subset_graph
from run_native_experiments import Experiment, dump, file_sha


def load_graph(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_sha(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        payload = str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_memory(memory: dict, manifest_sha: str, max_stage: int) -> dict:
    metadata = memory.get("metadata", {})
    if metadata.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("memory belongs to a different prepared manifest")
    if metadata.get("training_labels_used"):
        raise ValueError("memory metadata claims labels were used")
    if int(memory["stage"].max()) > max_stage:
        raise ValueError("memory contains a future stage")
    return clean_data(memory, "cpu") | {"metadata": metadata}


def select_matched_raw(delta: dict, count: int, seed: int) -> dict:
    if count < 2 or count > len(delta["x"]):
        raise ValueError("matched raw count is outside the available delta")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = torch.randperm(len(delta["x"]), generator=gen)[:count]
    return subset_graph(delta, rows)


def drift(before: dict, after: dict, mask: torch.Tensor) -> dict:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    b = before["x"][mask].double()
    a = after["x"][mask].double()
    d = (a - b).norm(dim=1)
    bn = b.norm(dim=1).clamp_min(1e-12)
    return {
        "rows": int(mask.sum()),
        "mean_l2": float(d.mean()) if mask.any() else 0.0,
        "median_l2": float(d.median()) if mask.any() else 0.0,
        "max_l2": float(d.max()) if mask.any() else 0.0,
        "mean_relative_l2": float((d / bn).mean()) if mask.any() else 0.0,
        "mean_norm_before": float(bn.mean()) if mask.any() else 0.0,
        "mean_norm_after": float(a.norm(dim=1).mean()) if mask.any() else 0.0,
    }


def snapshot_drift(initial: dict, candidate: dict, stage: int) -> dict:
    old = initial["stage"] != stage
    new = initial["stage"] == stage
    return {
        "old": drift(initial, candidate, old),
        "new": drift(initial, candidate, new),
        "selected_outer_step": candidate.get("metadata", {}).get("selected_outer_step"),
        "last_outer_step": candidate.get("metadata", {}).get("last_outer_step"),
        "x_norm_mean": candidate.get("metadata", {}).get("x_norm_mean"),
        "x_norm_max": candidate.get("metadata", {}).get("x_norm_max"),
        "x_l2": candidate.get("metadata", {}).get("x_l2"),
    }


def load_base_args(base_config: Path, output: str, prepared: str, device: str | None,
                   seed: int, model_seeds: list[int], check_every: int) -> argparse.Namespace:
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config.update({
        "output": output,
        "prepared": prepared,
        "seed": int(seed),
        "model_seeds": model_seeds,
        "device": device or config.get("device", "cuda:0"),
        "check_every": int(check_every),
        "confirm": False,
        "phase": "continual",
        "stages": 3,
        "arms": [],
        "diagnostic_mode": "fine_attribution",
    })
    return argparse.Namespace(**config)


def check_same_initial(a: dict, b: dict) -> None:
    for key in DATA_KEYS:
        if tensor_sha(a[key]) != tensor_sha(b[key]):
            raise RuntimeError(f"paired initial field differs: {key}")


def record_student_summary(row: dict) -> dict:
    """Keep the student-level evidence needed by paired diagnostics."""
    return {
        "arm": row["arm"],
        "roots": row["roots"],
        "mean_checkpoint_f1": row["mean_checkpoint_f1"],
        "model_runs": [
            {
                "seed": run["seed"],
                "selected_step": run["selected_step"],
                "best_checkpoint_f1": run["best_checkpoint_f1"],
                "best_minus_step0": run["best_minus_step0"],
                "checkpoint": run["checkpoint"],
                "endpoint_checkpoint": run["endpoint_checkpoint"],
                "step0_checkpoint": run["step0_checkpoint"],
            }
            for run in row["model_runs"]
        ],
    }


def run_old_update(exp: Experiment, old: dict, delta: dict, stage: int) -> dict:
    source = concat_graphs((old, delta))
    # One delta teacher and one joint teacher are shared by their paired calls.
    delta_teacher = exp.teacher(delta, stage, "fine_old_update_delta")
    # The protocol budget is based on cumulative *raw training roots*, not
    # the number of rows in the compact-memory-plus-delta source.
    cumulative_raw = sum(exp.counts[: stage + 1])
    new_k = math.floor(exp.args.rate * cumulative_raw) - len(old["x"])
    new_initial = exp.initializer(delta, new_k, stage, delta_teacher)
    initial = concat_graphs((old, new_initial))
    joint_teacher = exp.teacher(source, stage, "fine_old_update_joint")
    all_x = exp.optimize(source, initial, joint_teacher, stage, "old_update_all", extra_config={
        "optimize_rows": torch.ones(len(initial["x"]), dtype=torch.bool),
    })
    new_only = exp.optimize(source, initial, joint_teacher, stage, "old_update_new_only", extra_config={
        "optimize_rows": initial["stage"] == stage,
    })
    check_same_initial(initial, concat_graphs((old, new_initial)))
    exp.record(all_x, stage, "old_update_all", "same_source_same_initial_all_x")
    exp.record(new_only, stage, "old_update_new_only", "same_source_same_initial_new_x_only")
    # The initial artifact is recorded once for a direct no-update response
    # reference; its rows are otherwise identical to both optimizer inputs.
    exp.record(initial, stage, "old_update_initial", "paired_initial_no_update")
    all_last = load_graph(exp.output / f"old_update_all_s{stage}_last.pt")
    new_last = load_graph(exp.output / f"old_update_new_only_s{stage}_last.pt")
    all_best = load_graph(exp.output / f"old_update_all_s{stage}_best.pt")
    new_best = load_graph(exp.output / f"old_update_new_only_s{stage}_best.pt")
    return {
        "stage": stage,
        "source_rows": len(source["x"]),
        "old_rows": len(old["x"]),
        "new_rows": len(new_initial["x"]),
        "initial_sha256": {key: tensor_sha(initial[key]) for key in DATA_KEYS},
        "all_x_selected_drift": snapshot_drift(initial, all_x, stage),
        "new_x_only_selected_drift": snapshot_drift(initial, new_only, stage),
        "all_x_best_drift": snapshot_drift(initial, all_best, stage),
        "new_x_only_best_drift": snapshot_drift(initial, new_best, stage),
        "all_x_last_drift": snapshot_drift(initial, all_last, stage),
        "new_x_only_last_drift": snapshot_drift(initial, new_last, stage),
        "fixed_fields_equal": {key: tensor_sha(all_x[key]) == tensor_sha(new_only[key])
                               for key in DATA_KEYS if key != "x"},
        "training_labels_used": False,
    }


def run_new_quality(exp: Experiment, old: dict, delta: dict, stage: int, raw_seed: int) -> dict:
    source = concat_graphs((old, delta))
    total_budget = math.floor(exp.args.rate * sum(exp.counts[: stage + 1]))
    new_k = total_budget - len(old["x"])
    if new_k < 2:
        raise ValueError("new block budget is too small")
    delta_teacher = exp.teacher(delta, stage, "fine_new_quality_delta")
    compact_initial = exp.initializer(delta, new_k, stage, delta_teacher)
    raw = select_matched_raw(delta, new_k, raw_seed)
    raw_combined = concat_graphs((old, raw))
    compact_frozen = concat_graphs((old, compact_initial))
    exp.record(raw_combined, stage, "quality_raw_matched", "fixed_old_plus_matched_raw")
    exp.record(compact_frozen, stage, "quality_compact_frozen", "fixed_old_plus_frozen_compact")
    new_only = exp.optimize(delta, compact_initial, delta_teacher, stage, "quality_new_only_asm",
                            append_old=old)
    exp.record(concat_graphs((old, new_only)), stage, "quality_new_only_asm", "fixed_old_new_only_asm")

    joint_teacher = exp.teacher(source, stage, "fine_new_quality_joint")
    joint_initial = concat_graphs((old, compact_initial))
    joint = exp.optimize(source, joint_initial, joint_teacher, stage, "quality_joint_asm")
    protected = exp.optimize(source, joint_initial, joint_teacher, stage, "quality_protected_joint",
                             extra_config={"optimize_rows": joint_initial["stage"] == stage})
    exp.record(joint, stage, "quality_joint_asm", "fixed_old_joint_asm")
    exp.record(protected, stage, "quality_protected_joint", "fixed_old_protected_joint")
    return {
        "stage": stage,
        "old_rows": len(old["x"]),
        "new_budget": new_k,
        "raw_seed": int(raw_seed),
        "raw_origin_ids": raw["origin_ids"].tolist(),
        "compact_origin_ids": compact_initial["origin_ids"].tolist(),
        "compact_initial_sha256": {key: tensor_sha(compact_initial[key]) for key in DATA_KEYS},
        "same_old_memory": True,
        "same_new_row_count": True,
        "joint_vs_protected_initial_identical": True,
        "joint_old_drift": snapshot_drift(joint_initial, joint, stage),
        "protected_old_drift": snapshot_drift(joint_initial, protected, stage),
        "joint_last_drift": snapshot_drift(joint_initial,
                                            load_graph(exp.output / f"quality_joint_asm_s{stage}_last.pt"), stage),
        "protected_last_drift": snapshot_drift(joint_initial,
                                                load_graph(exp.output / f"quality_protected_joint_s{stage}_last.pt"), stage),
        "joint_vs_protected_fixed_fields_equal": {key: tensor_sha(joint[key]) == tensor_sha(protected[key])
                                                   for key in DATA_KEYS if key != "x"},
        "training_labels_used": False,
    }


def real_x_from_centroid(source: dict, centroid: dict) -> dict:
    """Replace only compact X with the selected source rows' real features.

    ``centroid`` is produced by one initializer call.  Reusing its non-X
    fields makes the comparison a field-level ablation: representative IDs,
    text embeddings/strings, masses and the post-decoder topology stay fixed.
    """
    result = clean_data(centroid, "cpu")
    lookup = {int(value): index for index, value in enumerate(source["origin_ids"].tolist())}
    selected = centroid["origin_ids"].detach().cpu().tolist()
    missing = [value for value in selected if value not in lookup]
    if missing:
        raise ValueError(f"centroid representative is absent from source: {missing[:3]}")
    result["x"] = torch.stack([source["x"][lookup[value]].detach().cpu() for value in selected])
    result["metadata"] = {**centroid.get("metadata", {}), "real_x": True,
                           "x_ablation": "selected_source_row_features"}
    return result


def run_representation_x(exp: Experiment, old: dict, delta: dict, stage: int) -> dict:
    """Compare centroid X and real representative X under identical fields."""
    total_budget = math.floor(exp.args.rate * sum(exp.counts[: stage + 1]))
    new_k = total_budget - len(old["x"])
    if new_k < 2:
        raise ValueError("new block budget is too small")
    delta_teacher = exp.teacher(delta, stage, "fine_representation_delta")
    centroid = exp.initializer(delta, new_k, stage, delta_teacher,
                               mode="joint_medoid", real_x=False)
    real = real_x_from_centroid(delta, centroid)
    paired = concat_graphs((old, centroid))
    paired_real = concat_graphs((old, real))
    for key in DATA_KEYS:
        if key == "x":
            continue
        if tensor_sha(paired[key]) != tensor_sha(paired_real[key]):
            raise RuntimeError(f"representation-X pair differs in fixed field: {key}")
    exp.record(paired, stage, "centroid_x_frozen", "fixed_representatives_centroid_features")
    exp.record(paired_real, stage, "real_x_frozen", "fixed_representatives_real_features")

    # ``record`` retrains independent students after exporting each artifact.
    # Keep the complete per-seed checkpoint records here instead of reducing
    # the field ablation to an artifact-only hash comparison.
    centroid_row = exp.rows[-2]
    real_row = exp.rows[-1]
    return {
        "stage": stage,
        "old_rows": len(old["x"]),
        "new_budget": new_k,
        "same_representative_ids": True,
        "same_text": tensor_sha(paired["text_emb"]) == tensor_sha(paired_real["text_emb"]),
        "same_topology": tensor_sha(paired["edge_index"]) == tensor_sha(paired_real["edge_index"]),
        "same_weights": tensor_sha(paired["weights"]) == tensor_sha(paired_real["weights"]),
        "representative_origin_ids": centroid["origin_ids"].tolist(),
        "centroid_x_sha256": tensor_sha(centroid["x"]),
        "real_x_sha256": tensor_sha(real["x"]),
        "x_equal": tensor_sha(centroid["x"]) == tensor_sha(real["x"]),
        "student_evaluation": {
            "centroid_x": record_student_summary(centroid_row),
            "real_x": record_student_summary(real_row),
            "delta_mean_checkpoint_f1": real_row["mean_checkpoint_f1"] - centroid_row["mean_checkpoint_f1"],
        },
        "training_labels_used": False,
    }


def shuffle_new_text(data: dict, stage: int, seed: int) -> tuple[dict, list[int]]:
    """Shuffle only new-block text while preserving every other field."""
    result = clean_data(data, "cpu")
    rows = torch.where(result["stage"] == stage)[0]
    if len(rows) < 2:
        raise ValueError("text pairing diagnostic needs at least two new rows")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(len(rows), generator=generator)
    result["text_emb"][rows] = result["text_emb"][rows][permutation]
    texts = list(result["texts"])
    old_texts = [texts[int(row)] for row in rows.tolist()]
    for row, text in zip(rows.tolist(), [old_texts[int(i)] for i in permutation.tolist()]):
        texts[row] = text
    result["texts"] = texts
    result["metadata"] = {**data.get("metadata", {}),
                           "text_pairing": "new_stage_rows_shuffled",
                           "text_shuffle_seed": int(seed),
                           "text_shuffle_permutation": permutation.tolist()}
    return result, permutation.tolist()


def run_text_pairing(exp: Experiment, old: dict, delta: dict, stage: int, text_seed: int) -> dict:
    """Compare paired and shuffled new-stage text with fixed X/topology."""
    total_budget = math.floor(exp.args.rate * sum(exp.counts[: stage + 1]))
    new_k = total_budget - len(old["x"])
    if new_k < 2:
        raise ValueError("new block budget is too small")
    delta_teacher = exp.teacher(delta, stage, "fine_text_pairing_delta")
    compact = exp.initializer(delta, new_k, stage, delta_teacher,
                              mode="joint_medoid", real_x=False)
    paired = concat_graphs((old, compact))
    shuffled, permutation = shuffle_new_text(paired, stage, text_seed)
    fixed_points = sum(index == value for index, value in enumerate(permutation))
    for key in ("x", "edge_index", "weights", "origin_ids", "stage"):
        if tensor_sha(paired[key]) != tensor_sha(shuffled[key]):
            raise RuntimeError(f"text pairing changed fixed field: {key}")
    if sorted(paired["texts"]) != sorted(shuffled["texts"]):
        raise RuntimeError("text pairing changed the text multiset")
    exp.record(paired, stage, "text_paired", "fixed_x_topology_paired_text")
    exp.record(shuffled, stage, "text_shuffled", "fixed_x_topology_new_text_shuffle")
    paired_row = exp.rows[-2]
    shuffled_row = exp.rows[-1]
    return {
        "stage": stage,
        "old_rows": len(old["x"]),
        "new_budget": new_k,
        "text_shuffle_seed": int(text_seed),
        "permutation": permutation,
        "moved_count": len(permutation) - fixed_points,
        "fixed_point_count": fixed_points,
        "same_x": True,
        "same_topology": True,
        "same_weights": True,
        "same_origin_ids": True,
        "same_stage": True,
        "same_text_multiset": True,
        "paired_field_sha256": {key: tensor_sha(paired[key]) for key in DATA_KEYS},
        "shuffled_field_sha256": {key: tensor_sha(shuffled[key]) for key in DATA_KEYS},
        "student_evaluation": {
            "paired": record_student_summary(paired_row),
            "shuffled": record_student_summary(shuffled_row),
            "delta_shuffled_minus_paired": shuffled_row["mean_checkpoint_f1"] - paired_row["mean_checkpoint_f1"],
        },
        "training_labels_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("old_update", "new_quality", "representation_x", "text_pairing"), required=True)
    parser.add_argument("--base-config", required=True,
                        help="run_config.json from a completed shared-teacher recursive run")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--old-artifact", required=True,
                        help="fixed stage-1 memory artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None,
                        help="condensation seed; defaults to base config seed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--raw-seed", type=int, default=4402)
    parser.add_argument("--text-seed", type=int, default=4403)
    args = parser.parse_args()
    base_config = Path(args.base_config)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    seed = int(base.get("seed", 44) if args.seed is None else args.seed)
    exp = Experiment(load_base_args(base_config, args.output, args.prepared, args.device,
                                    seed, args.model_seeds, args.check_every))
    stages = [exp.load_stage(i) for i in range(max(3, args.stage + 1))]
    manifest_sha = file_sha(exp.prepared / "manifest.json")
    old_path = Path(args.old_artifact)
    old = assert_memory(load_graph(old_path), manifest_sha, args.stage - 1)
    if int(old["stage"].max()) != args.stage - 1:
        raise ValueError("old artifact must end at stage-1")
    delta = stages[args.stage]
    started = time.time()
    if args.mode == "old_update":
        summary = run_old_update(exp, old, delta, args.stage)
    elif args.mode == "new_quality":
        summary = run_new_quality(exp, old, delta, args.stage, args.raw_seed)
    elif args.mode == "representation_x":
        summary = run_representation_x(exp, old, delta, args.stage)
    else:
        summary = run_text_pairing(exp, old, delta, args.stage, args.text_seed)
    dump(exp.output / "fine_attribution_summary.json", {
        "mode": args.mode,
        "confirmation_accessed": False,
        "base_config": str(base_config),
        "base_config_sha256": file_sha(base_config),
        "old_artifact": str(old_path),
        "old_artifact_sha256": file_sha(old_path),
        "prepared_manifest_sha256": manifest_sha,
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    })
    dump(exp.output / "diagnostic_inputs.json", {
        "mode": args.mode,
        "confirmation_accessed": False,
        "old_artifact": str(old_path),
        "old_artifact_sha256": file_sha(old_path),
        "stage": args.stage,
        "seed": seed,
        "model_seeds": args.model_seeds,
        "raw_seed": args.raw_seed,
        "text_seed": args.text_seed,
        "interpretation": "paired development attribution; not an additive causal decomposition",
    })
    dump(exp.output / "completed.json", {"status": "complete", "confirmation_accessed": False,
                                           "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    main()
