from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_content_gap_attribution as CAG
import run_teacher_guided_message_revision as BASE
import run_teacher_guided_payload_revision as PAYLOAD
import run_unified_v3_continual as U
import teacher_guided_message_revision as TGM
import teacher_guided_payload_revision as T
import tgr_role_conditioned_revision as R
import protocol_v3


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-cache", required=True, type=Path)
    parser.add_argument("--c0-root", required=True, type=Path)
    parser.add_argument("--c0-file", type=Path, default=None)
    parser.add_argument("--semantic-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default=None,
                        help="optional GraphCLIP MiniLM path for prompt probe")
    parser.add_argument("--teacher-steps", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--top-r", type=int, default=8)
    parser.add_argument("--old-class-mix", type=float, default=0.0)
    parser.add_argument("--old-radius", type=float, default=0.05)
    parser.add_argument("--new-radius", type=float, default=0.15)
    parser.add_argument("--core-radius", type=float, default=0.02)
    parser.add_argument("--boundary-radius", type=float, default=0.08)
    parser.add_argument("--core-fraction", type=float, default=0.50)
    parser.add_argument("--boundary-fraction", type=float, default=0.15)
    parser.add_argument("--old-center-weight", type=float, default=0.25)
    parser.add_argument("--old-relation-weight", type=float, default=0.20)
    parser.add_argument("--old-text-weight", type=float, default=0.10)
    parser.add_argument("--role-anchor-weight", type=float, default=0.05)
    parser.add_argument("--old-probe-weight", type=float, default=1.00)
    parser.add_argument("--new-probe-weight", type=float, default=0.50)
    parser.add_argument("--raw-new-probe-weight", type=float, default=1.00)
    parser.add_argument("--raw-class-weight", type=float, default=1.00)
    parser.add_argument("--raw-prompt-weight", type=float, default=1.00)
    parser.add_argument("--old-prompt-weight", type=float, default=2.00)
    parser.add_argument("--old-prompt-tolerance", type=float, default=0.0)
    parser.add_argument("--gradient-surgery", action="store_true")
    parser.add_argument("--old-probe-tolerance", type=float, default=0.0)
    parser.add_argument("--response-alignment-gate", action="store_true")
    parser.add_argument("--response-alignment-threshold", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.55)
    parser.add_argument("--gradient-weight", type=float, default=0.35)
    parser.add_argument("--degree-weight", type=float, default=0.10)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=[9450, 9451])
    parser.add_argument("--validation-teacher-seeds", nargs="+", type=int, default=[9460])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")

    raw = load(args.data)
    x, _, _, _ = PAYLOAD.raw_fields(raw)
    text_emb = load(args.text_cache)
    if len(text_emb) != len(x):
        raise ValueError("text cache is not node aligned")
    prompt_emb = None
    if args.minilm:
        unified = U.load_unified(args.dataset, str(args.data), args.minilm,
                                 args.device, n_stages=3,
                                 split_mode="canonical")
        prompt_emb = unified["encoder"].encode(unified["prompts"])
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    _, _, _, pool = protocol_v3.split_masks_v4(
        torch.arange(len(x)), protocol_v3.REPORT_DEV_MOD)
    y = PAYLOAD.raw_fields(raw)[2]
    ids = torch.where(pool & torch.isin(y, torch.tensor(groups[args.stage])))[0].long()
    current = PAYLOAD.make_delta(raw, text_emb, ids, args.stage)
    current_cell = args.semantic_root / f"{args.dataset}_{args.order}_s{args.stage}"
    parent_cell = args.semantic_root / f"{args.dataset}_{args.order}_s{args.stage - 1}"
    base = load(current_cell / "semantic_perm_memory.pt")
    if args.stage == 1:
        parent_path = args.c0_file
        if parent_path is None:
            parent_path = args.c0_root / args.dataset / args.order / "c0_s0_memory.pt"
        parent = load(parent_path)
    else:
        parent_path = parent_cell / "semantic_perm_memory.pt"
        parent = load(parent_path)
    old_rows = len(parent["x"])
    support = PAYLOAD.concat_support(parent, current)
    old_classes, provenance_source = BASE.infer_old_class_provenance(
        parent, args, old_rows)
    if old_classes is not None:
        support["class_id"][:old_rows] = old_classes
    if "class_id" not in support:
        raise ValueError("sealed class provenance is required for role-conditioned revision")

    teacher_cfg = T.Config(
        teacher_steps=args.teacher_steps, inner_steps=args.inner_steps,
        outer_steps=args.outer_steps, batch=args.batch, top_r=args.top_r,
        teacher_seeds=tuple(args.teacher_seeds),
        validation_teacher_seeds=tuple(args.validation_teacher_seeds),
        old_class_mix=float(args.old_class_mix),
        old_radius=float(args.old_radius), new_radius=float(args.new_radius),
        baseline_gate=True)
    role_cfg = R.Config(
        core_fraction=float(args.core_fraction),
        boundary_fraction=float(args.boundary_fraction),
        core_radius=float(args.core_radius),
        regular_radius=float(args.old_radius),
        boundary_radius=float(args.boundary_radius),
        new_radius=float(args.new_radius),
        old_center_weight=float(args.old_center_weight),
        old_relation_weight=float(args.old_relation_weight),
        old_text_weight=float(args.old_text_weight),
        role_anchor_weight=float(args.role_anchor_weight),
        old_probe_weight=float(args.old_probe_weight),
        new_probe_weight=float(args.new_probe_weight),
        raw_new_probe_weight=float(args.raw_new_probe_weight),
        raw_class_weight=float(args.raw_class_weight),
        raw_prompt_weight=float(args.raw_prompt_weight),
        old_prompt_weight=float(args.old_prompt_weight),
        old_prompt_tolerance=float(args.old_prompt_tolerance),
        gradient_surgery=bool(args.gradient_surgery),
        old_probe_tolerance=float(args.old_probe_tolerance),
        response_alignment_gate=bool(args.response_alignment_gate),
        response_alignment_threshold=float(args.response_alignment_threshold),
        boundary_weight=float(args.boundary_weight),
        gradient_weight=float(args.gradient_weight),
        degree_weight=float(args.degree_weight))
    result, audit, assignment = R.optimize(
        base, support, support["class_id"], old_rows, args.stage,
        args.device, teacher_cfg, role_cfg, prompt_emb=prompt_emb)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(support, args.output / "support.pt")
    torch.save(result, args.output / "tgr_role_memory.pt")
    (args.output / "assignment.json").write_text(
        json.dumps(assignment, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    protocol = {
        "name": "tgr-role-conditioned-bounded-revision-v1",
        "dataset": args.dataset, "order": args.order, "stage": int(args.stage),
        "base": str(current_cell / "semantic_perm_memory.pt"),
        "parent": str(parent_path), "current_delta_rows": int(len(ids)),
        "base_rows": int(len(base["x"])), "old_rows": int(old_rows),
        "teacher": "canonical two-layer GCN + bidirectional InfoNCE on legal support",
        "role_features": ["semantic_boundary_to_current_delta",
                           "teacher_gradient_sensitivity", "base_out_degree"],
        "optimized": ["X_old_role_bounded", "X_new"],
        "fixed": ["P_initial", "topology", "text_emb", "texts", "budget",
                  "Student architecture"],
        "old_memory_update": "allowed under role-specific trust regions",
        "old_class_objectives": ["class_centroid", "class_relation_matrix",
                                  "graph_text_consistency"],
        "old_class_provenance": provenance_source,
        "prompt_probe": bool(prompt_emb is not None),
        "historical_raw_reads": "current delta only",
        "report_labels_used": False,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "tgr_role_conditioned_revision_complete",
                      "dataset": args.dataset, "order": args.order,
                      "stage": int(args.stage), "output": str(args.output),
                      "selected_outer_step": audit["selected_outer_step"],
                      "gate_fallback": audit["gate_fallback"],
                      "role_counts": audit["roles"]["counts"]}), flush=True)


if __name__ == "__main__":
    main()
