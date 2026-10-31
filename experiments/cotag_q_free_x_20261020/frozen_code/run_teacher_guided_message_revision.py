"""Run projected-message Teacher/P/X revision on one Semantic-Recursive cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_content_gap_attribution as CAG
import run_teacher_guided_payload_revision as BASE
import teacher_guided_message_revision as TGM
import protocol_v3


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _classes_from_allocation(allocation, rows: int) -> torch.Tensor | None:
    """Expand sealed class-to-row counts into the stored row order."""
    if not isinstance(allocation, dict):
        return None
    values = []
    try:
        for class_id, count in sorted(allocation.items(), key=lambda item: int(item[0])):
            count = int(count)
            if count < 0:
                return None
            values.extend([int(class_id)] * count)
    except (TypeError, ValueError):
        return None
    if len(values) != int(rows):
        return None
    return torch.tensor(values, dtype=torch.long)


def infer_old_class_provenance(parent: dict, args: argparse.Namespace,
                               old_rows: int) -> tuple[torch.Tensor | None, str]:
    """Recover class provenance from sealed synthetic metadata only.

    Stage-0 memories retain their class allocation.  Later recursive parents
    are reconstructed from that allocation plus each prior semantic block's
    sealed ``c0_row_classes`` audit.  No historical raw graph is read here.
    """
    explicit = parent.get("class_id")
    if isinstance(explicit, torch.Tensor) and len(explicit) == int(old_rows):
        values = explicit.long().cpu()
        if bool((values >= 0).all()):
            return values, "parent.class_id"

    direct = _classes_from_allocation(parent.get("allocation"), old_rows)
    if direct is not None:
        return direct, "parent.allocation"

    if int(args.stage) <= 1:
        return None, "none"

    base_path = (args.c0_root / args.dataset / args.order / "c0_s0_memory.pt")
    if not base_path.exists():
        return None, "none"
    base = load(base_path)
    values = _classes_from_allocation(base.get("allocation"), len(base["x"]))
    if values is None:
        return None, "none"
    parts = [values]
    for prior_stage in range(1, int(args.stage)):
        audit_path = (args.semantic_root /
                      f"{args.dataset}_{args.order}_s{prior_stage}" /
                      "build_audit.json")
        if not audit_path.exists():
            return None, "none"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        block_classes = audit.get("structural_audit", {}).get("c0_row_classes")
        if not isinstance(block_classes, list):
            return None, "none"
        parts.append(torch.tensor(block_classes, dtype=torch.long))
    result = torch.cat(parts)
    if len(result) != int(old_rows) or not bool((result >= 0).all()):
        return None, "none"
    return result, "sealed.parent_allocation+semantic_block_audits"


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
    parser.add_argument("--mode", choices=("topology_only", "p_only", "x_only", "joint"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-steps", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--top-r", type=int, default=8)
    parser.add_argument("--message-mix", type=float, default=0.10)
    parser.add_argument("--swap-fraction", type=float, default=0.10)
    parser.add_argument("--old-class-mix", type=float, default=0.0)
    parser.add_argument("--old-radius", type=float, default=0.05)
    parser.add_argument("--new-radius", type=float, default=0.15)
    parser.add_argument("--ste-forward", action="store_true",
                        help="use the final hard topology in candidate forward passes")
    parser.add_argument("--plasticity-weight", type=float, default=0.0)
    parser.add_argument("--plasticity-margin", type=float, default=0.20)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=[9300, 9301])
    parser.add_argument("--validation-teacher-seeds", nargs="+", type=int, default=[9310])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    raw = load(args.data)
    x, _, _, _ = BASE.raw_fields(raw)
    text_emb = load(args.text_cache)
    if len(text_emb) != len(x):
        raise ValueError("text cache is not node aligned")
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    _, _, _, pool = protocol_v3.split_masks_v4(
        torch.arange(len(x)), protocol_v3.REPORT_DEV_MOD)
    ids = torch.where(pool & torch.isin(
        BASE.raw_fields(raw)[2], torch.tensor(groups[args.stage])))[0].long()
    current = BASE.make_delta(raw, text_emb, ids, args.stage)
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
    support = BASE.concat_support(parent, current)
    provenance_file = current_cell / "build_audit.json"
    provenance_source = "none"
    old_classes, provenance_source = infer_old_class_provenance(parent, args, old_rows)
    if old_classes is not None:
        support["class_id"][:old_rows] = old_classes
    cfg = TGM.T.Config(
        teacher_steps=args.teacher_steps, inner_steps=args.inner_steps,
        outer_steps=args.outer_steps, batch=args.batch, top_r=args.top_r,
        teacher_seeds=tuple(args.teacher_seeds),
        validation_teacher_seeds=tuple(args.validation_teacher_seeds),
        old_class_mix=float(args.old_class_mix),
        old_radius=float(args.old_radius),
        new_radius=float(args.new_radius),
        ste_forward=bool(args.ste_forward),
        plasticity_weight=float(args.plasticity_weight),
        plasticity_margin=float(args.plasticity_margin),
    )
    result, audit, assignment = TGM.optimize(
        base, support, support["class_id"], old_rows, args.stage,
        args.device, cfg, learn_x=args.mode in ("x_only", "joint"),
        learn_p=args.mode in ("p_only", "joint"),
        message_mix=args.message_mix, swap_fraction=args.swap_fraction)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(support, args.output / "support.pt")
    torch.save(result, args.output / "tgr_message_memory.pt")
    (args.output / "assignment.json").write_text(
        json.dumps(assignment, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "protocol.json").write_text(json.dumps({
        "name": "teacher-guided-message-revision-v1",
        "dataset": args.dataset, "order": args.order, "stage": args.stage,
        "mode": args.mode, "base": str(current_cell / "semantic_perm_memory.pt"),
        "parent": str(parent_path), "current_delta_rows": int(len(ids)),
        "base_rows": int(len(base["x"])), "old_rows": int(old_rows),
        "teacher": "canonical two-layer GCN + bidirectional InfoNCE on legal support",
        "assignment": "sparse top-r graph-text P",
        "message_kernel": "base anchor + bounded Q(P)^T A_support Q(P), row-degree materialized",
        "message_mix": float(args.message_mix),
        "swap_fraction": float(args.swap_fraction),
        "old_class_provenance": provenance_source,
        "old_class_provenance_rows": int((support["class_id"][:old_rows] >= 0).sum()),
        "old_class_mix": float(args.old_class_mix),
        "old_radius": float(args.old_radius),
        "new_radius": float(args.new_radius),
        "ste_forward": bool(args.ste_forward),
        "plasticity_weight": float(args.plasticity_weight),
        "plasticity_margin": float(args.plasticity_margin),
        "optimized": [value for value in (
            "X" if args.mode in ("x_only", "joint") else None,
            "P" if args.mode in ("p_only", "joint") else None) if value],
        "fixed": ["text_emb", "texts", "budget", "Student architecture"],
        "historical_raw_reads": "current delta only", "report_labels_used": False,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "teacher_guided_message_revision_complete",
                      "dataset": args.dataset, "order": args.order,
                      "stage": args.stage, "mode": args.mode,
                      "output": str(args.output),
                      "selected_outer_step": audit["selected_outer_step"],
                      "p_mean_abs_delta": audit["p_mean_abs_delta"],
                      "edge_count_after": audit["edge_count_after"]}), flush=True)


if __name__ == "__main__":
    main()
