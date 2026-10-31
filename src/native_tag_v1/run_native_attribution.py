"""Paired development diagnostics using one immutable stage-0 memory.

All online variants start from exactly the same exported memory. Full raw
history is used only in explicitly named oracle arms; checkpoint labels stay
inside Experiment. No confirmation partition is accessed by this script.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from native_core import DATA_KEYS, _fingerprint, clean_data, select_reference_rows
from protocol_data import concat_graphs
from run_native_experiments import Experiment, dump, file_sha


def stage_audit(data):
    total = float(data["weights"].sum())
    return [{"stage": int(stage), "rows": int((data["stage"] == stage).sum()),
             "row_fraction": float((data["stage"] == stage).float().mean()),
             "mass": float(data["weights"][data["stage"] == stage].sum()),
             "mass_fraction": float(data["weights"][data["stage"] == stage].sum()) / total}
            for stage in sorted(data["stage"].unique().tolist())]


def check_anchor(anchor, stage0, manifest_hash, expected_roots):
    metadata = anchor.get("metadata", {})
    if metadata.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("anchor belongs to a different prepared dataset")
    if metadata.get("stage") != 0 or not bool((anchor["stage"] == 0).all()):
        raise ValueError("the paired anchor must contain only stage 0")
    if len(anchor["x"]) != expected_roots:
        raise ValueError("anchor root budget differs from this protocol")
    if not set(anchor["origin_ids"].tolist()) <= set(stage0["origin_ids"].tolist()):
        raise ValueError("anchor text representatives are not legal stage-0 training nodes")
    return clean_data(anchor)


def reference_audit(source, count, condensation_seed):
    audits = {}
    for policy in ("uniform", "mass_stratified"):
        generator = torch.Generator().manual_seed(condensation_seed + 3011)
        _, audits[policy] = select_reference_rows(source, count, generator, policy)
    old_counts = []
    for seed in range(200):
        rows, _ = select_reference_rows(source, count, torch.Generator().manual_seed(seed), "uniform")
        old_counts.append(int((source["stage"][rows] == 0).sum()))
    audits["uniform_sampling_sensitivity"] = {
        "sampling_seeds": list(range(200)), "old_row_counts": old_counts,
        "old_rows_mean": float(np.mean(old_counts)),
        "old_absence_fraction": float(np.mean(np.asarray(old_counts) == 0)),
        "training_runs": 0,
    }
    return audits


def run_preserved(exp, stage0, anchor, stages):
    """Compare recursive updates with identical stage-0 exported memory."""
    arms = ("union", "append", "append_joint", "protected_joint")
    previous = {arm: anchor for arm in arms}
    history = [stage0]
    exp.record(anchor, 0, "anchor", "shared_stage0_memory")
    audits = []
    for stage in range(1, stages):
        delta = exp.load_stage(stage)
        history.append(delta)
        full = concat_graphs(history)
        k = math.floor(exp.args.rate * len(full["x"]))
        exp.record(full, stage, "full", "full_history_reference")
        delta_teacher = exp.teacher(delta, stage, "delta")
        new_k = k - len(previous["append"]["x"])
        if new_k < 2:
            raise ValueError("the increment budget cannot support contrastive condensation")
        new_initial = exp.initializer(delta, new_k, stage, delta_teacher)
        teachers = {}
        for arm in arms:
            old = previous[arm]
            source = concat_graphs((old, delta))
            if arm == "append":
                new = exp.optimize(delta, new_initial, delta_teacher, stage, arm, append_old=old)
                compact = concat_graphs((old, new))
            else:
                source_signature = tuple(_fingerprint(source[key]) for key in DATA_KEYS)
                if source_signature not in teachers:
                    teachers[source_signature] = (exp.teacher(source, stage, arm), arm)
                teacher, teacher_owner = teachers[source_signature]
                initial = exp.initializer(source, k, stage, teacher) if arm == "union" else concat_graphs((old, new_initial))
                extra = {"optimize_rows": initial["stage"] == stage} if arm == "protected_joint" else None
                compact = exp.optimize(source, initial, teacher, stage, arm, extra_config=extra)
            previous[arm] = compact
            exp.record(compact, stage, arm, "memory_plus_current_delta")
            old_ids = set(old["origin_ids"].tolist())
            audits.append({"stage": stage, "arm": arm, "old_roots": len(old["x"]),
                           "new_root_budget": new_k, "cumulative_root_budget": k,
                           "old_text_ids_retained": len(old_ids & set(compact["origin_ids"].tolist())),
                           "selected_outer_step": compact.get("metadata", {}).get("selected_outer_step"),
                           "teacher_shared_with": teacher_owner if arm != "append" else "delta",
                           "source_representative_stages": stage_audit(source),
                           "mass_lineage_warning": "stage is text representative provenance, not raw cluster lineage"})
            dump(exp.output / "recursive_audit.json", audits)
        del delta_teacher
        del teachers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("static", "source", "recondense", "preserve"), required=True)
    parser.add_argument("--stages", type=int, choices=(2, 3), default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--outer-lr", type=float, default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--reference-probe", action="store_true")
    cli = parser.parse_args()
    base = Path(cli.base_run)
    if not (base / "completed.json").exists():
        raise ValueError("the anchor run must be complete")
    configuration = json.loads((base / "run_config.json").read_text())
    actual_arms = {
        "static": ["full", "random", "coreset", "frozen", "offline_asm", "offline_asm_last"],
        "source": ["full", "old_only", "delta_only", "union_no_recondense", "union_no_recondense_weighted"],
        "recondense": ["offline_frozen", "offline_asm", "union_frozen", "union", "append_frozen", "append"],
        "preserve": ["anchor", "full", "union", "append", "append_joint", "protected_joint"],
    }[cli.mode]
    if cli.mode == "recondense" and cli.reference_probe:
        actual_arms.append("reference_quota_union")
    configuration.update(output=cli.output, device=cli.device, confirm=False,
                         check_every=cli.check_every, model_seeds=cli.model_seeds,
                         phase="static" if cli.mode == "static" else "continual",
                         stages=1 if cli.mode == "static" else cli.stages if cli.mode == "preserve" else 2,
                         diagnostic_mode=cli.mode, anchor_run=str(base), arms=actual_arms,
                         reference_probe=cli.reference_probe)
    if cli.outer_lr is not None:
        configuration["outer_lr"] = cli.outer_lr
    exp = Experiment(argparse.Namespace(**configuration))
    stage0 = exp.load_stage(0)
    anchor_path = base / "offline_asm_s0_artifact.pt"
    registered = {row["arm"]: row for row in json.loads((base / "results.json").read_text())["results"]
                  if row["stage"] == 0}
    inputs = {}

    def load_registered(name):
        path = base / f"{name}_s0_artifact.pt"
        actual_hash = file_sha(path)
        if actual_hash != registered[name]["artifact_sha256"]:
            raise ValueError(f"registered artifact hash mismatch: {name}")
        inputs[name] = {"path": str(path), "sha256": actual_hash, "matches_registered_hash": True}
        data = torch.load(path, map_location="cpu", weights_only=True)
        check_anchor(data, stage0, file_sha(exp.prepared / "manifest.json"),
                     math.floor(exp.args.rate * len(stage0["x"])))
        return data

    old = check_anchor(load_registered("offline_asm"),
                       stage0, file_sha(exp.prepared / "manifest.json"),
                       math.floor(exp.args.rate * len(stage0["x"])))
    dump(exp.output / "diagnostic_inputs.json", {
        "anchor_artifact": str(anchor_path), "anchor_sha256": file_sha(anchor_path),
        "base_run_config_sha256": file_sha(base / "run_config.json"),
        "mode": cli.mode, "reference_probe": cli.reference_probe,
        "all_online_arms_share_identical_old_memory": True,
        "check_every": cli.check_every, "confirmation_accessed": False,
        "interpretation": "paired development contrasts, not an additive causal decomposition",
    })
    if cli.mode == "static":
        exp.record(stage0, 0, "full", "current_delta")
        for name in ("random", "coreset", "frozen", "offline_asm"):
            artifact = load_registered(name)
            exp.record(artifact, 0, name, "retraining_existing_stage0_artifact")
        last = torch.load(base / "offline_asm_s0_last.pt", map_location="cpu", weights_only=True)
        inputs["offline_asm_last"] = {"path": str(base / "offline_asm_s0_last.pt"),
                                      "sha256": file_sha(base / "offline_asm_s0_last.pt"),
                                      "matches_registered_hash": None,
                                      "reason": "unselected optimizer snapshot; not a results.json artifact"}
        exp.record(last, 0, "offline_asm_last", "unselected_last_iterate_diagnostic")
    elif cli.mode == "preserve":
        run_preserved(exp, stage0, old, cli.stages)
    else:
        delta = exp.load_stage(1)
        full = concat_graphs((stage0, delta))
        source = concat_graphs((old, delta))
        k = math.floor(exp.args.rate * len(full["x"]))
        dump(exp.output / "source_audit.json", {
            "full": stage_audit(full), "old_memory_plus_delta": stage_audit(source),
            "stage1_root_budget": k,
            "references": reference_audit(source, exp.args.reference_size, exp.args.seed + 1),
        })
        if cli.mode == "source":
            exp.record(full, 1, "full", "full_history_reference")
            exp.record(old, 1, "old_only", "old_memory_only_diagnostic")
            exp.record(delta, 1, "delta_only", "current_delta_over_budget_diagnostic")
            exp.record(source, 1, "union_no_recondense", "memory_plus_delta_over_budget_diagnostic")
            exp.record(source, 1, "union_no_recondense_weighted", "same_memory_plus_delta_query_weighted", weighted=True)
        else:
            full_teacher = exp.teacher(full, 1, "offline")
            offline_init = exp.initializer(full, k, 1, full_teacher)
            exp.record(offline_init, 1, "offline_frozen", "full_history_reference")
            offline = exp.optimize(full, offline_init, full_teacher, 1, "offline_asm")
            exp.record(offline, 1, "offline_asm", "full_history_reference")
            del full_teacher

            teacher = exp.teacher(source, 1, "union")
            initial = exp.initializer(source, k, 1, teacher)
            exp.record(initial, 1, "union_frozen", "memory_plus_current_delta")
            union = exp.optimize(source, initial, teacher, 1, "union")
            exp.record(union, 1, "union", "memory_plus_current_delta")
            if cli.reference_probe:
                # One changed component: reference sampling. Teacher, initial
                # graph/text, inner loss and downstream training are identical.
                variant = exp.optimize(source, initial, teacher, 1, "reference_quota_union",
                                       extra_config={"reference_policy": "mass_stratified"})
                exp.record(variant, 1, "reference_quota_union", "memory_plus_current_delta")
            del teacher

            new_k = k - len(old["x"])
            if new_k < 2:
                raise ValueError("Append needs at least two new roots for InfoNCE")
            delta_teacher = exp.teacher(delta, 1, "append")
            new_initial = exp.initializer(delta, new_k, 1, delta_teacher)
            exp.record(concat_graphs((old, new_initial)), 1, "append_frozen", "memory_plus_current_delta")
            new = exp.optimize(delta, new_initial, delta_teacher, 1, "append", append_old=old)
            exp.record(concat_graphs((old, new)), 1, "append", "memory_plus_current_delta")
            del delta_teacher
    dump(exp.output / "input_artifact_hashes.json", inputs)
    dump(exp.output / "completed.json", {"status": "complete", "elapsed_seconds": time.time() - exp.started})


if __name__ == "__main__":
    main()
