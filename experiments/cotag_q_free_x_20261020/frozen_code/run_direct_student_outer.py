"""Run the direct Student outer construction on prepared legal artifacts.

The caller supplies:

* base: the synthetic memory to revise;
* support: previous synthetic memory concatenated with the current raw delta;
* old-rows: the prefix length in support and base.

No historical raw graph is loaded here. The output directory is immutable so
that each direct Student configuration remains auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import direct_student_outer as DSO


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(item):
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu()
            if value.numel() == 1:
                return float(value)
            return {"tensor_shape": list(value.shape),
                    "tensor_mean": float(value.float().mean()),
                    "tensor_norm": float(value.float().norm())}
        raise TypeError(f"not JSON serializable: {type(item).__name__}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True,
                               default=default),
                    encoding="utf-8")


def select_rows(base: dict, old_rows: int, mode: str, max_old: int,
                max_new: int) -> tuple[list[int], dict]:
    n = len(base["x"])
    if old_rows < 0 or old_rows > n:
        raise ValueError("old-rows is outside base")
    old = list(range(min(old_rows, int(max_old)))) if max_old >= 0 else []
    new_start = old_rows
    new = list(range(new_start, min(n, new_start + int(max_new)))) if max_new >= 0 else []
    if mode == "all":
        selected = list(range(n))
    elif mode == "new":
        selected = list(range(old_rows, n))
    elif mode == "old_new":
        selected = old + new
    elif mode == "first":
        selected = old + new
    else:
        raise ValueError(f"unknown row selection: {mode}")
    selected = sorted(set(selected))
    return selected, {"mode": mode, "selected_rows": selected,
                      "old_rows": old_rows, "max_old": max_old,
                      "max_new": max_new}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--old-rows", required=True, type=int)
    parser.add_argument("--mode", choices=DSO.MODES, default="x")
    parser.add_argument("--row-selection", choices=("all", "new", "old_new", "first"),
                        default="old_new")
    parser.add_argument("--max-old", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--inner-batch", type=int, default=16)
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--outer-lr", type=float, default=0.02)
    parser.add_argument("--support-batch", type=int, default=16)
    parser.add_argument("--old-weight", type=float, default=0.25)
    parser.add_argument("--new-weight", type=float, default=1.0)
    parser.add_argument("--prompt-weight", type=float, default=0.0)
    parser.add_argument("--prompt-margin-weight", type=float, default=0.0)
    parser.add_argument("--prompt-margin", type=float, default=0.20)
    parser.add_argument("--candidate-prompt-margin-weight", type=float,
                        default=0.0,
                        help="candidate-graph class margin weight for x-class+p+a")
    parser.add_argument("--candidate-prompt-margin", type=float, default=0.20,
                        help="candidate-graph class margin for x-class+p+a")
    parser.add_argument("--old-graph-anchor-weight", type=float, default=0.0,
                        help="candidate-vs-C0 old synthetic representation anchor")
    parser.add_argument("--class-center-weight", type=float, default=0.0)
    parser.add_argument("--support-class-id", type=Path, default=None,
                        help="optional construction-time class ids aligned to support rows")
    parser.add_argument("--base-slot-class-id", type=Path, default=None,
                        help="class ids aligned to candidate rows; -1 is reserved for old rows")
    parser.add_argument("--support-slot-id", type=Path, default=None,
                        help="provenance slot ids aligned to support rows; -1 is reserved for old rows")
    parser.add_argument("--base-slot-group-id", type=Path, default=None,
                        help="provenance group ids aligned to candidate rows; -1 is reserved for old rows")
    parser.add_argument("--prompt-emb", type=Path, default=None,
                        help="optional MiniLM prompt embedding tensor")
    parser.add_argument("--prompt-ids", type=Path, default=None,
                        help="optional prompt ids aligned to prompt_emb rows")
    parser.add_argument("--x-old-radius", type=float, default=0.05)
    parser.add_argument("--x-new-radius", type=float, default=0.20)
    parser.add_argument("--old-barrier-weight", type=float, default=0.0)
    parser.add_argument("--old-barrier-margin", type=float, default=0.0)
    parser.add_argument("--operator-aligned", action="store_true")
    parser.add_argument("--freeze-text-inner", action="store_true")
    parser.add_argument("--p-mix", type=float, default=0.25)
    parser.add_argument("--p-temperature", type=float, default=0.20)
    parser.add_argument("--class-constrained-p", action="store_true",
                        help="restrict P transport to matching construction-time classes")
    parser.add_argument("--a-pool", type=int, default=4)
    parser.add_argument("--a-temperature", type=float, default=0.20)
    parser.add_argument("--a-existing-logit", type=float, default=3.0)
    parser.add_argument("--a-max-revisions-per-row", type=int, default=None,
                        help="maximum number of original old/new cross edges that each new synthetic row may replace")
    parser.add_argument("--a-revision-margin", type=float, default=0.0,
                        help="minimum learned score advantage required to replace an original cross edge")
    parser.add_argument("--geometry-anchor-weight", type=float, default=0.50)
    parser.add_argument("--geometry-margin-weight", type=float, default=1.00)
    parser.add_argument("--geometry-smoothness-weight", type=float, default=0.25)
    parser.add_argument("--geometry-margin", type=float, default=0.20)
    parser.add_argument("--class-representation-weight", type=float, default=1.0)
    parser.add_argument("--slot-representation-weight", type=float, default=0.25)
    parser.add_argument("--slot-mass-weighted", action="store_true",
                        help="weight provenance-routed slot losses by frozen mass")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[9300])
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[9310])
    parser.add_argument("--plan-seed", type=int, default=9320)
    parser.add_argument("--stop-gradient-control", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    base, support = load(args.base), load(args.support)
    selected, selection_audit = select_rows(
        base, args.old_rows, args.row_selection, args.max_old, args.max_new)
    cfg = DSO.Config(
        inner_steps=args.inner_steps, inner_batch=args.inner_batch,
        outer_steps=args.outer_steps, outer_lr=args.outer_lr,
        support_batch=args.support_batch, old_weight=args.old_weight,
        new_weight=args.new_weight, prompt_weight=args.prompt_weight,
        prompt_margin_weight=args.prompt_margin_weight,
        prompt_margin=args.prompt_margin,
        candidate_prompt_margin_weight=args.candidate_prompt_margin_weight,
        candidate_prompt_margin=args.candidate_prompt_margin,
        old_graph_anchor_weight=args.old_graph_anchor_weight,
        class_center_weight=args.class_center_weight,
        x_old_radius=args.x_old_radius,
        x_new_radius=args.x_new_radius,
        old_barrier_weight=args.old_barrier_weight,
        old_barrier_margin=args.old_barrier_margin,
        operator_aligned=args.operator_aligned,
        freeze_text_inner=args.freeze_text_inner,
        p_mix=args.p_mix,
        p_temperature=args.p_temperature,
        p_class_constrained=args.class_constrained_p,
        a_pool=args.a_pool,
        a_temperature=args.a_temperature,
        a_existing_logit=args.a_existing_logit,
        a_max_revisions_per_row=args.a_max_revisions_per_row,
        a_revision_margin=args.a_revision_margin,
        geometry_anchor_weight=args.geometry_anchor_weight,
        geometry_margin_weight=args.geometry_margin_weight,
        geometry_smoothness_weight=args.geometry_smoothness_weight,
        geometry_margin=args.geometry_margin,
        class_representation_weight=args.class_representation_weight,
        slot_representation_weight=args.slot_representation_weight,
        slot_mass_weighted=args.slot_mass_weighted,
        train_seeds=tuple(args.train_seeds),
        validation_seeds=tuple(args.validation_seeds),
        plan_seed=args.plan_seed)
    if args.stop_gradient_control:
        result, audit = DSO.stop_gradient_control(
            base, support, args.old_rows, selected, args.mode, args.device, cfg)
    else:
        support_class_id = None
        prompt_emb = None
        prompt_ids = None
        base_slot_class_id = None
        support_slot_id = None
        base_slot_group_id = None
        if args.support_class_id is not None:
            support_class_id = load(args.support_class_id)
            if not isinstance(support_class_id, torch.Tensor):
                raise TypeError("support-class-id must contain a tensor")
        if args.prompt_emb is not None:
            prompt_emb = load(args.prompt_emb)
            if not isinstance(prompt_emb, torch.Tensor):
                raise TypeError("prompt-emb must contain a tensor")
        if args.prompt_ids is not None:
            prompt_ids = load(args.prompt_ids)
            if not isinstance(prompt_ids, torch.Tensor):
                raise TypeError("prompt-ids must contain a tensor")
        if args.base_slot_class_id is not None:
            base_slot_class_id = load(args.base_slot_class_id)
            if not isinstance(base_slot_class_id, torch.Tensor):
                raise TypeError("base-slot-class-id must contain a tensor")
        if args.support_slot_id is not None:
            support_slot_id = load(args.support_slot_id)
            if not isinstance(support_slot_id, torch.Tensor):
                raise TypeError("support-slot-id must contain a tensor")
        if args.base_slot_group_id is not None:
            base_slot_group_id = load(args.base_slot_group_id)
            if not isinstance(base_slot_group_id, torch.Tensor):
                raise TypeError("base-slot-group-id must contain a tensor")
        result, audit = DSO.optimize(
            base, support, args.old_rows, selected, args.mode, args.device, cfg,
            support_class_id=support_class_id, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids, base_slot_class_id=base_slot_class_id,
            support_slot_id=support_slot_id,
            base_slot_group_id=base_slot_group_id)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(result, args.output / "memory.pt")
    audit["selection"] = selection_audit
    audit["base_file"] = str(args.base)
    audit["support_file"] = str(args.support)
    save_json(args.output / "audit.json", audit)
    save_json(args.output / "protocol.json", {
        "name": "direct-student-outer-v2",
        "mode": args.mode,
        "stage": "prepared legal support",
        "old_rows": int(args.old_rows),
        "editable_rows": selected,
        "student": "canonical two-layer GCN + bidirectional InfoNCE",
        "inner_steps": int(args.inner_steps),
        "history_access": "previous synthetic memory + current raw delta support only",
        "report_labels_used": False,
        "prompt_loss": {
            "enabled": bool(args.prompt_weight > 0),
            "weight": float(args.prompt_weight),
            "margin_weight": float(args.prompt_margin_weight),
            "margin": float(args.prompt_margin),
            "class_ids_source": str(args.support_class_id) if args.support_class_id else None,
            "prompt_embedding_source": str(args.prompt_emb) if args.prompt_emb else None,
        },
        "candidate_prompt_margin": {
            "enabled": bool(args.mode == "x-class+p+a"
                             and args.candidate_prompt_margin_weight > 0),
            "weight": float(args.candidate_prompt_margin_weight),
            "margin": float(args.candidate_prompt_margin),
            "objective": "candidate synthetic graph after P/A propagation",
            "class_ids_source": (str(args.base_slot_class_id)
                                  if args.base_slot_class_id else None),
        },
        "old_graph_anchor": {
            "enabled": bool(args.old_graph_anchor_weight > 0),
            "weight": float(args.old_graph_anchor_weight),
            "reference": "C0 synthetic graph old-row representations",
            "uses_historical_raw": False,
        },
        "soft_assignments": {
            "p_mix": float(args.p_mix),
            "p_temperature": float(args.p_temperature),
            "p_class_constrained": bool(args.class_constrained_p),
            "a_pool": int(args.a_pool),
            "a_temperature": float(args.a_temperature),
            "a_existing_logit": float(args.a_existing_logit),
            "a_max_revisions_per_row": args.a_max_revisions_per_row,
            "a_revision_margin": float(args.a_revision_margin),
            "a_forward_mass": "unit mass per selected hard edge",
        },
        "class_aware": {
            "enabled": bool(DSO._x_mode(args.mode) == "x-class"),
            "base_slot_class_id_source": (str(args.base_slot_class_id)
                                           if args.base_slot_class_id else None),
            "class_center_weight": float(args.class_center_weight),
        },
        "geometry_x": {
            "enabled": bool(args.mode == "x-geometry"),
            "anchor_weight": float(args.geometry_anchor_weight),
            "margin_weight": float(args.geometry_margin_weight),
            "smoothness_weight": float(args.geometry_smoothness_weight),
            "margin": float(args.geometry_margin),
            "objective": "candidate graph prompt margin + parent anchor + within-class slot similarity preservation",
        },
        "class_representation_x": {
            "enabled": bool(args.mode == "x-class-rep"),
            "weight": float(args.class_representation_weight),
            "target": "mass-weighted candidate class graph response aligned to sealed current-delta class response",
            "source": (str(args.support_class_id)
                       if args.support_class_id else None),
            "objective": "raw current-delta class centers are detached targets; train/validation rows remain disjoint",
        },
        "slot_representation_x": {
            "enabled": bool(args.mode == "x-slot-rep"),
            "weight": float(args.slot_representation_weight),
            "target": "detached Student graph response of each provenance slot's current-delta raw subgroup",
            "objective": "slot-level response alignment on top of provenance-routed X updates",
        },
        "slot_mass_weighted": {
            "enabled": bool(args.slot_mass_weighted),
            "objective": "aggregate provenance-routed outer losses by frozen synthetic mass",
        },
        "provenance_aware": {
            "enabled": bool(args.mode == "x-provenance"),
            "support_slot_id_source": (str(args.support_slot_id)
                                        if args.support_slot_id else None),
            "base_slot_group_id_source": (str(args.base_slot_group_id)
                                           if args.base_slot_group_id else None),
            "assignment": "joint graph-text cosine with exact base.mass capacity"
            if args.mode == "x-provenance" else None,
        },
        "old_barrier": {
            "weight": float(args.old_barrier_weight),
            "margin": float(args.old_barrier_margin),
            "reference": "same sealed support and Student seeds as C0"
            if args.old_barrier_weight > 0 else None,
        },
        "operator_aligned": {
            "enabled": bool(args.operator_aligned),
            "graph_operator": "candidate synthetic adjacency for inner and outer"
            if args.operator_aligned else None,
            "new_text_targets": "current-delta mean text per provenance slot"
            if args.operator_aligned else None,
        },
        "freeze_text_inner": bool(args.freeze_text_inner),
        "historical_raw_reads": 0,
        "stop_gradient_control": bool(args.stop_gradient_control),
    })
    print(json.dumps({
        "event": "direct_student_outer_complete",
        "mode": args.mode,
        "output": str(args.output),
        "max_relative_displacement": audit.get("max_relative_displacement", 0.0),
        "selected_outer_step": audit.get("selected_outer_step"),
    }), flush=True)


if __name__ == "__main__":
    main()
