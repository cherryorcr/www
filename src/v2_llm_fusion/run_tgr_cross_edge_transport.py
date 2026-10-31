"""Topology-only cross-edge transport experiment.

This runner keeps the deployed memory content and the initial assignment
fixed.  Only the support graph used by the message projection is changed.
The ``c0`` arm is the exact sealed Semantic-Recursive memory; other arms are
materialized by the existing hard topology operator after changing only the
support cross-edge block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import delta_cross_edge_transport as CE
import run_content_gap_attribution as CAG
import run_teacher_guided_message_revision as TGM_RUN
import run_teacher_guided_payload_revision as PAYLOAD
import teacher_guided_message_revision as TGM
import teacher_guided_payload_revision as T
import protocol_v3
from student_consistent_edge_swap import Config as StudentConsistentConfig
from student_consistent_edge_swap import evaluate_one_step, select_swaps


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_parent_provenance(parent_cell: Path, old_rows: int) -> tuple[dict[int, int] | None, str]:
    """Load only sealed per-row provenance, never reconstruct from raw data."""
    path = parent_cell / "build_audit.json"
    if not path.exists():
        return None, "missing_build_audit"
    audit = json.loads(path.read_text(encoding="utf-8"))
    records = audit.get("structural_audit", {}).get("records")
    if not isinstance(records, list) or not records:
        return None, "build_audit_without_records"
    # Recursive stage audits describe only the appended block.  Its rows are
    # the suffix of the parent memory; the earlier prefix may legitimately
    # have no sealed raw provenance.
    offset = int(old_rows) - len(records)
    if offset < 0:
        return None, "records_exceed_parent_rows"
    shifted = []
    for record in records:
        item = dict(record)
        item["source_original_ids"] = list(record.get("source_original_ids", []))
        shifted.append(item)
    mapping = CE.build_raw_to_old_slot(shifted, len(shifted))
    mapping = {raw_id: int(row + offset) for raw_id, row in mapping.items()}
    return mapping, str(path)


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def _cfg(args: argparse.Namespace) -> T.Config:
    return T.Config(
        teacher_steps=int(args.teacher_steps),
        inner_steps=int(args.inner_steps),
        outer_steps=max(1, int(args.outer_steps)),
        batch=int(args.batch), top_r=int(args.top_r),
        teacher_seeds=tuple(int(v) for v in args.teacher_seeds),
        validation_teacher_seeds=tuple(int(v) for v in args.validation_teacher_seeds),
    )


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
    parser.add_argument("--modes", nargs="+",
                        default=["c0", "none", "random", "semantic", "message"])
    parser.add_argument("--cross-degree", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=9821)
    parser.add_argument("--fill-incident", action="store_true")
    parser.add_argument("--message-mix", type=float, default=1.0)
    parser.add_argument("--swap-fraction", type=float, default=1.0)
    parser.add_argument("--teacher-steps", type=int, default=1)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--top-r", type=int, default=8)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=[9300])
    parser.add_argument("--validation-teacher-seeds", nargs="+", type=int, default=[9310])
    parser.add_argument("--student-consistent-seeds", nargs="+", type=int,
                        default=[17])
    parser.add_argument("--student-consistent-steps", type=int, default=1)
    parser.add_argument("--student-consistent-lr", type=float, default=1e-3)
    parser.add_argument("--student-consistent-batch", type=int, default=16)
    parser.add_argument("--student-consistent-max-swaps", type=int, default=1)
    parser.add_argument("--student-consistent-new-margin", type=float, default=0.0)
    parser.add_argument("--student-consistent-old-drop", type=float, default=0.0)
    parser.add_argument("--student-consistent-total-increase", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")

    raw = load(args.data)
    x, _, y, raw_edge = PAYLOAD.raw_fields(raw)
    text_emb = load(args.text_cache)
    if len(text_emb) != len(x):
        raise ValueError("text cache is not node aligned")
    groups = CAG.stage_groups(args.dataset, 3, args.order)
    _, _, _, pool = protocol_v3.split_masks_v4(
        torch.arange(len(x)), protocol_v3.REPORT_DEV_MOD)
    ids = torch.where(pool & torch.isin(
        y, torch.tensor(groups[args.stage])))[0].long()
    current = PAYLOAD.make_delta(raw, text_emb, ids, args.stage)
    current["current_delta_ids"] = ids.clone()
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

    old_classes, provenance_source = TGM_RUN.infer_old_class_provenance(
        parent, args, old_rows)
    if old_classes is None:
        old_classes = torch.full((old_rows,), -1, dtype=torch.long)
    parent_with_classes = T.clone_memory(parent)
    parent_with_classes["class_id"] = old_classes
    current["class_id"] = y[ids].long()
    raw_to_old, sealed_provenance_source = _load_parent_provenance(
        parent_cell, old_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "tgr-current-delta-cross-edge-transport-v1",
        "dataset": args.dataset, "order": args.order, "stage": int(args.stage),
        "arms": list(args.modes), "cross_degree": int(args.cross_degree),
        "support_only_intervention": True,
        "fixed": ["X", "text_emb", "texts", "budget", "initial_P",
                  "Student architecture", "hard materializer"],
        "student": "canonical two-layer GCN + bidirectional InfoNCE",
        "c0_arm": "exact sealed semantic_perm_memory",
        "historical_raw_reads": "current delta incident edges only; no historical raw features",
        "raw_provenance_source": sealed_provenance_source,
        "old_class_provenance": provenance_source,
        "selection_by_report": False,
        "student_consistent": {
            "enabled_if_requested": True,
            "seeds": [int(seed) for seed in args.student_consistent_seeds],
            "steps": int(args.student_consistent_steps),
            "lr": float(args.student_consistent_lr),
            "batch": int(args.student_consistent_batch),
            "max_swaps": int(args.student_consistent_max_swaps),
            "new_gain_margin": float(args.student_consistent_new_margin),
            "max_old_gain_drop": float(args.student_consistent_old_drop),
            "max_total_loss_increase": float(
                args.student_consistent_total_increase),
            "candidate_pool": "message_balanced",
            "selector_uses_report_or_test": False,
        },
    }
    _save_json(args.output / "protocol.json", protocol)

    for mode in args.modes:
        out = args.output / mode
        if out.exists():
            raise FileExistsError(f"immutable output exists: {out}")
        out.mkdir(parents=True, exist_ok=False)
        if mode == "c0":
            result = T.clone_memory(base)
            support = None
            support_audit = {
                "mode": "c0", "support_edge_count": None,
                "historical_raw_reads": 0, "report_labels_used": False,
            }
            revision_audit = {"mode": "c0", "topology_changed": False,
                              "edge_count_before": int(base["edge_index"].shape[1]),
                              "edge_count_after": int(base["edge_index"].shape[1])}
            assignment = {}
        elif mode == "none":
            # Strict no-op control: a zero cross-edge score must not trigger
            # an arbitrary endpoint permutation.  This arm is exactly C0.
            result = T.clone_memory(base)
            support, support_audit = CE.make_support(
                parent_with_classes, current, mode="none",
                cross_degree=int(args.cross_degree), seed=int(args.random_seed),
                raw_edge_index=raw_edge, current_ids=ids,
                raw_to_old_slot=raw_to_old, fill_incident=bool(args.fill_incident))
            revision_audit = {
                "mode": "strict_noop_control",
                "topology_changed": False,
                "edge_count_before": int(base["edge_index"].shape[1]),
                "edge_count_after": int(result["edge_index"].shape[1]),
                "cross_only": {
                    "old_old_preserved": True,
                    "new_new_preserved": True,
                    "per_new_cross_degree_preserved": True,
                    "exact_base_edge_index": True,
                },
                "p_updated": False, "x_updated": False,
                "historical_raw_reads": 0, "report_labels_used": False,
            }
            assignment = {}
        else:
            # Student-consistent uses the balanced message support only as a
            # legal candidate pool.  The final endpoint assignment is chosen
            # by the fixed short-horizon Student response below.
            support_mode = ("message_balanced" if mode in
                            ("message_global", "student_consistent") else mode)
            support, support_audit = CE.make_support(
                parent_with_classes, current, mode=support_mode,
                cross_degree=int(args.cross_degree), seed=int(args.random_seed),
                raw_edge_index=raw_edge, current_ids=ids,
                raw_to_old_slot=raw_to_old, fill_incident=bool(args.fill_incident))
            support_audit["arm_mode"] = mode
            support_audit["selection_mode"] = support_mode
            # Build the same deterministic initial P used by the message
            # pipeline, but do not optimize P or X.  Only the support graph
            # changes the projected cross-edge scores.
            assignment = T.build_sparse_assignment(
                base, support, old_rows, support["class_id"],
                _cfg(args), args.device)
            q, _ = TGM._assignment_matrix(
                assignment, old_rows, len(support["x"]), len(base["x"]),
                args.device)
            support_binary = TGM._binary_adjacency(
                support["edge_index"], len(support["x"]), args.device,
                base["x"].dtype)
            projected = TGM._projected_kernel(support_binary, q)
            if mode == "message_global":
                edge_index, edge_audit = CE.materialize_cross_only_global(
                    base["edge_index"], projected.detach().cpu(), old_rows)
            else:
                edge_index, edge_audit = CE.materialize_cross_only(
                    base["edge_index"], projected.detach().cpu(), old_rows)
            result = T.clone_memory(base)
            result["edge_index"] = edge_index
            student_selection = None
            if mode == "student_consistent":
                student_config = StudentConsistentConfig(
                    max_swaps=int(args.student_consistent_max_swaps),
                    student_steps=int(args.student_consistent_steps),
                    student_lr=float(args.student_consistent_lr),
                    batch=int(args.student_consistent_batch),
                    new_gain_margin=float(args.student_consistent_new_margin),
                    max_old_gain_drop=float(args.student_consistent_old_drop),
                    max_total_loss_increase=float(
                        args.student_consistent_total_increase),
                )
                selected, student_selection = select_swaps(
                    base, edge_index, old_rows, student_config,
                    lambda memory: evaluate_one_step(
                        memory, old_rows,
                        tuple(int(seed) for seed in args.student_consistent_seeds),
                        device=args.device, config=student_config))
                result = T.clone_memory(selected)
                support_audit["candidate_pool"] = "message_balanced"
                support_audit["student_consistent"] = True
            revision_audit = {
                "mode": ("student_consistent_bounded_cross_edge_swap"
                          if mode == "student_consistent"
                          else "cross_only_materializer"),
                "message_mix": float(args.message_mix),
                "swap_fraction": float(args.swap_fraction),
                "edge_count_before": int(base["edge_index"].shape[1]),
                "edge_count_after": int(result["edge_index"].shape[1]),
                "topology_changed": not torch.equal(
                    result["edge_index"], base["edge_index"]),
                "projected_edge_mass": float(projected.detach().sum()),
                "materializer": edge_audit.get("assignment_method", "greedy"),
                "cross_only": edge_audit,
                "p_updated": False, "x_updated": False,
                "historical_raw_reads": 0, "report_labels_used": False,
            }
            if student_selection is not None:
                revision_audit["student_selection"] = student_selection
            assignment = {
                "indices": assignment["indices"].tolist(),
                "weights": torch.softmax(assignment["logits"], dim=1).tolist(),
                "slot_class": assignment["slot_class"].tolist(),
                "raw_class": assignment["raw_class"].tolist(),
                "top_r": int(assignment["indices"].shape[1]),
            }
            torch.save(support, out / "support.pt")
        torch.save(result, out / "tgr_message_memory.pt")
        _save_json(out / "support_audit.json", support_audit)
        _save_json(out / "audit.json", {
            "protocol": protocol, "support": support_audit,
            "revision": revision_audit,
            "output_edge_count": int(result["edge_index"].shape[1]),
            "same_content_as_base": bool(torch.equal(result["x"], base["x"])
                                           and result["texts"] == base["texts"]
                                           and torch.equal(result["text_emb"], base["text_emb"])),
        })
        (out / "assignment.json").write_text(
            json.dumps(assignment, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps({"event": "cross_edge_arm_complete",
                          "dataset": args.dataset, "order": args.order,
                          "stage": int(args.stage), "mode": mode,
                          "output": str(out),
                          "cross_pairs": support_audit.get("cross_edge_pairs", 0),
                          "edge_count": int(result["edge_index"].shape[1])}),
              flush=True)


if __name__ == "__main__":
    main()
