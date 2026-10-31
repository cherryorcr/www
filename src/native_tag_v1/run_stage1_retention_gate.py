"""Stage-1 response-retention gate (variant A, PROTOCOL V2 section 7 Gate 1).

For one condensation seed this runner:
1. loads the stage-0 synthetic memory artifact (old rows, never re-read raw);
2. loads the saved stage-0 teacher checkpoint and exports graph-to-class-prompt
   logits on the old synthetic rows only ("variant A" targets; this simulates
   the stage-end export that a live pipeline would perform before discarding
   the teacher -- no real node and no teacher parameter crosses the boundary);
3. builds the stage-1 compact block with the audited initializer;
4. runs joint ASM with text-anchor + response-retention, saving fixed outer
   snapshots 0/20/.../100 (selector disabled);
5. retrains independent students on every snapshot and reports the dev curve.

Gate pass condition (evaluated across seeds by the caller): dev-selected
outer > 0 and cumulative dev >= outer=0 for every condensation seed.
Confirmation/test is never read here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from native_core import clean_data, create_model
from protocol_data import concat_graphs
from run_native_experiments import Experiment, dump, file_sha
from run_outer_trajectory import (assert_memory, build_args, evaluate_snapshots,
                                  load_graph)


def load_teacher(path: Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("training_labels_used"):
        raise ValueError("teacher checkpoint claims label usage")
    model = create_model(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--stage0-artifact", required=True)
    parser.add_argument("--stage0-teacher", required=True,
                        help="saved stage-0 teacher checkpoint (.pt with state_dict)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument("--outer-eval-every", type=int, default=20)
    parser.add_argument("--text-anchor-weight", type=float, default=0.01)
    parser.add_argument("--response-retention-weight", type=float, default=0.1)
    args = parser.parse_args()

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    exp = Experiment(build_args(base, args.output, args.prepared, args.device,
                                args.seed, args.model_seeds, args.check_every,
                                args.outer_steps, args.outer_eval_every,
                                args.text_anchor_weight, "all"))
    stages = [exp.load_stage(i) for i in range(2)]
    manifest_sha = file_sha(exp.prepared / "manifest.json")
    old = assert_memory(load_graph(Path(args.stage0_artifact)), manifest_sha, 0)
    delta = stages[1]
    total_budget = math.floor(exp.args.rate * sum(exp.counts[:2]))
    new_k = total_budget - len(old["x"])
    if new_k < 2:
        raise ValueError("stage-1 new block budget is too small")

    # Variant-A targets for old rows from the (saved) stage-0 teacher on the
    # old synthetic rows of the *joint* synthetic graph topology. Message
    # passing must match later retention forwards, so targets are computed on
    # the concatenated initial graph, restricted to old rows.
    delta_teacher = exp.teacher(delta, 1, "gate_delta")
    compact = exp.initializer(delta, new_k, 1, delta_teacher)
    initial = concat_graphs((old, compact))
    old_rows = torch.arange(len(old["x"]))
    stage0_teacher = load_teacher(Path(args.stage0_teacher), exp.device)
    init_device = clean_data(initial, exp.device, with_adjacency=True)
    class_emb = exp.classes
    with torch.no_grad():
        old_targets = stage0_teacher(init_device, old_rows.to(exp.device),
                                     class_emb=torch.as_tensor(class_emb, device=exp.device,
                                                               dtype=init_device["x"].dtype))
    if not bool(torch.isfinite(old_targets).all()):
        raise FloatingPointError("nonfinite stage-0 response targets")
    del stage0_teacher, init_device

    source = concat_graphs((old, delta))
    teacher = exp.teacher(source, 1, "gate_joint")
    extra = {
        "text_anchor_weight": args.text_anchor_weight,
        "text_anchor_scope": "all",
        "response_retention_weight": args.response_retention_weight,
        "response_class_emb": class_emb,
        "response_targets": old_targets.detach().cpu(),
        "response_target_rows": old_rows,
    }
    name = "gate_retention"
    started = time.time()
    exp.optimize(source, initial, teacher, 1, name, extra_config=extra,
                 select_validation=False, retain_outer_steps=True)
    outer_steps = list(range(0, args.outer_steps + 1, args.outer_eval_every))
    curve = evaluate_snapshots(exp, exp.output, name, 1, outer_steps, args.model_seeds)
    curve.update({"text_anchor_weight": args.text_anchor_weight,
                  "response_retention_weight": args.response_retention_weight})
    # dev-based selection summary for the gate decision
    best = max(curve["trajectory"], key=lambda r: r["selected_student_mean_macro_f1"])
    base_row = [r for r in curve["trajectory"] if r["outer_step"] == 0][0]
    curve["gate"] = {
        "dev_selected_outer": best["outer_step"],
        "dev_selected_cum_f1": best["selected_student_mean_macro_f1"],
        "outer0_cum_f1": base_row["selected_student_mean_macro_f1"],
        "selected_minus_outer0": best["selected_student_mean_macro_f1"]
                                 - base_row["selected_student_mean_macro_f1"],
        "passes_outer_gt0": best["outer_step"] > 0,
    }
    dump(exp.output / "trajectory.json", curve)
    dump(exp.output / "diagnostic_inputs.json", {
        "mode": "stage1_response_retention_gate",
        "variant": "A_self_distillation_synthetic_rows",
        "confirmation_accessed": False,
        "stage0_artifact": args.stage0_artifact,
        "stage0_artifact_sha256": file_sha(Path(args.stage0_artifact)),
        "stage0_teacher": args.stage0_teacher,
        "stage0_teacher_sha256": file_sha(Path(args.stage0_teacher)),
        "old_rows": int(len(old["x"])),
        "new_budget": int(new_k),
        "response_target_aux_bytes": int(old_targets.numel() * old_targets.element_size()),
        "text_anchor_weight": args.text_anchor_weight,
        "response_retention_weight": args.response_retention_weight,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
    })
    dump(exp.output / "completed.json", {"status": "complete", "confirmation_accessed": False})
    print(json.dumps({"gate": curve["gate"], "seed": args.seed,
                      "lambda_r": args.response_retention_weight}), flush=True)


if __name__ == "__main__":
    main()
