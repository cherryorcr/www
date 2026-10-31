"""Run the frozen Bounded Student-Compatible X-P-A arm under the unified
small-graph baseline protocol.

The script is intentionally an evaluation runner.  It does not tune the
configuration, read historical raw rows for construction, or change the
canonical Student.  At each stage it replaces only the current C0 prefix
with the previously produced X-P-A memory, builds a legal support from that
prefix plus the current raw delta, and runs the frozen direct outer operator.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

import direct_student_outer as DSO
import prepare_class_aware_x_support as CLASS
import run_content_gap_attribution as CAG
import run_semantic_recursive_unified_baselines_20260926 as B


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def clone_memory(memory: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for key, value in memory.items()
    }


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.numel() <= 64:
                return tensor.tolist()
            return {"tensor_shape": list(tensor.shape),
                    "tensor_mean": float(tensor.float().mean()),
                    "tensor_norm": float(tensor.float().norm())}
        raise TypeError(f"not JSON serializable: {type(item).__name__}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True,
                               default=default),
                    encoding="utf-8")


def compose_with_parent(c0: dict, parent: dict) -> dict:
    """Use recursive parent payload and preserve its old-old topology.

    The current C0 artifact supplies edges incident to the newly appended
    rows.  Its old-old subgraph is discarded, otherwise a later stage would
    silently erase the parent's bounded A revisions.
    """
    old_rows = len(parent["x"])
    if len(c0["x"]) < old_rows:
        raise ValueError("current C0 memory is shorter than the parent")
    out = clone_memory(c0)
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        out[key][:old_rows] = parent[key]
    out["texts"] = list(parent["texts"]) + list(c0["texts"])[old_rows:]
    parent_edges = parent["edge_index"].long()
    current_edges = c0["edge_index"].long()
    if current_edges.numel():
        keep = ((current_edges[0] >= old_rows)
                | (current_edges[1] >= old_rows))
        current_edges = current_edges[:, keep]
    if parent_edges.numel() and current_edges.numel():
        merged = torch.cat((parent_edges, current_edges), dim=1)
    elif parent_edges.numel():
        merged = parent_edges
    else:
        merged = current_edges
    out["edge_index"] = merged.contiguous()
    return out


def support_class_ids(data: dict, ids: torch.Tensor, old_rows: int,
                      support: dict) -> torch.Tensor:
    result = torch.full((len(support["x"]),), -1, dtype=torch.long)
    result[old_rows:] = data["y"].cpu()[ids.cpu()].long()
    return result


def frozen_config(args: argparse.Namespace) -> DSO.Config:
    return DSO.Config(
        inner_steps=8,
        outer_steps=4,
        outer_lr=0.02,
        old_weight=0.25,
        new_weight=1.0,
        new_loss_scale=float(args.new_loss_scale),
        x_old_radius=0.05,
        x_new_radius=0.10,
        p_mix=0.25,
        p_temperature=0.20,
        p_sinkhorn_iters=24,
        operator_aligned=bool(args.operator_aligned),
        operator_aux_weight=float(args.operator_aux_weight),
        p_class_constrained=bool(args.class_constrained_p),
        a_pool=4,
        a_temperature=1.0,
        a_existing_logit=0.1,
        a_max_revisions_per_row=1,
        candidate_prompt_margin_weight=1.0,
        candidate_prompt_margin=0.20,
        class_representation_weight=(1.0 if args.class_representation else 0.0),
        p_old_context_ratio=float(args.p_old_context_ratio),
        p_old_context_cap=float(args.p_old_context_cap),
        p_response_gate=bool(args.p_response_gate),
        p_gate_margin=float(args.p_gate_margin),
        p_gate_old_tolerance=float(args.p_gate_old_tolerance),
        p_gate_per_class=bool(args.p_gate_per_class),
        a_message_weight=float(args.a_message_weight),
        a_marginal=bool(args.a_marginal),
        a_marginal_inner_steps=int(args.a_marginal_inner_steps),
        a_marginal_new_margin=float(args.a_marginal_new_margin),
        a_marginal_old_tolerance=float(args.a_marginal_old_tolerance),
        train_seeds=tuple(args.train_seeds),
        validation_seeds=tuple(args.validation_seeds),
        plan_seed=int(args.plan_seed),
    )


def run_order(dataset: str, order: str, args: argparse.Namespace) -> dict:
    source = args.source_root / dataset / order
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    ids_by_stage = [torch.tensor(v, dtype=torch.long)
                    for v in manifest["stage_source_ids"]]
    groups = [[int(v) for v in group] for group in manifest["stage_classes"]]
    budgets = [int(v) for v in manifest["budgets"]]

    data = B.load_data(dataset, args.minilm, args.device, args.cache_root)
    data["stage_classes"] = groups
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = B.G.norm_adj(data["edge_index"], len(data["x"]),
                                     args.device)
    data["x_device"] = data["x"].float().to(args.device)
    cfg = frozen_config(args)
    out = args.output_root / f"{dataset}_{order}"
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {out / 'results.json'}")
    out.mkdir(parents=True, exist_ok=False)

    topology = ("current-stage C0 topology with bounded one-revision-per-new-row A"
                if args.mode == "x-class+p+a"
                else "current-stage C0 topology; no topology revision")
    protocol = {
        "name": f"bounded-student-compatible-{args.mode}-unified-v1",
        "dataset": dataset,
        "order": order,
        "stage_classes": groups,
        "budgets": budgets,
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "report_seeds": args.report_seeds,
        "student_batch": 16,
        "student_steps": "max(50,min(2048,16*rows))",
        "construction": "current delta only; no historical raw reads; report/test labels not used",
        "topology": topology,
        "operator_aligned": bool(args.operator_aligned),
        "operator_aux_weight": float(args.operator_aux_weight),
        "class_constrained_p": bool(args.class_constrained_p),
        "method": args.mode,
        "frozen_config": cfg.dict(),
        "recursive_parent": True,
        "memory_budget": "same as source C0",
    }
    save_json(out / "protocol.json", protocol)

    parent = None
    rows = []
    for stage in range(3):
        c0 = load(source / f"c0_s{stage}_memory.pt")
        if len(c0["x"]) != budgets[stage]:
            raise AssertionError("C0 budget mismatch")
        if stage == 0:
            candidate = clone_memory(c0)
            audit = {
                "stage": 0,
                "mode": "sealed_c0_base",
                "historical_raw_reads": 0,
                "report_labels_used": False,
            }
        else:
            if parent is None:
                raise AssertionError("missing recursive parent")
            old_rows = len(parent["x"])
            delta = B.raw_support(data, ids_by_stage[stage], stage)
            support = B.concat_support(parent, delta, stage)
            base = compose_with_parent(c0, parent)
            class_id = support_class_ids(data, ids_by_stage[stage],
                                         old_rows, support)
            slot_class, slot_audit = CLASS.build_slot_classes(
                base, support, old_rows, class_id)
            prompt_emb = data["encoder"].encode(data["prompts"]).cpu()
            prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)
            stage_out = out / f"stage{stage}"
            stage_out.mkdir(parents=True, exist_ok=False)
            torch.save(support, stage_out / "support.pt")
            torch.save(class_id, stage_out / "support_class_id.pt")
            torch.save(slot_class, stage_out / "base_slot_class_id.pt")
            save_json(stage_out / "slot_audit.json", slot_audit)
            torch.save(base, stage_out / "base.pt")
            started = time.perf_counter()
            candidate, audit = DSO.optimize(
                base, support, old_rows,
                list(range(old_rows, len(base["x"]))),
                mode=args.mode, device=args.device, cfg=cfg,
                support_class_id=class_id,
                prompt_emb=prompt_emb,
                prompt_ids=prompt_ids,
                base_slot_class_id=slot_class)
            audit.update({
                "stage": stage,
                "construction_seconds_wall": float(time.perf_counter() - started),
                "source_c0": str(source / f"c0_s{stage}_memory.pt"),
                "historical_raw_reads": 0,
                "report_labels_used": False,
                "slot_audit": slot_audit,
            })
            save_json(stage_out / "audit.json", audit)
            torch.save(candidate, stage_out / "memory.pt")
        parent = candidate
        stage_path = out / f"stage{stage}_memory.pt"
        torch.save(candidate, stage_path)
        scores, elapsed = B.train_scratch(
            candidate, data, groups, stage, args.report_seeds, args.device)
        rows.append({
            "dataset": dataset,
            "order": order,
            "arm": "bounded_xp" if args.mode == "x-class+p" else "bounded_xpa",
            "stage": stage,
            "mode": "scratch",
            "aggregate": B.aggregate_stage_scores(scores),
            "per_student": scores,
            "memory_rows": int(len(candidate["x"])),
            "student_wall_seconds": float(elapsed),
            "construction_audit": audit,
        })
        print(json.dumps({"event": "stage_complete", "dataset": dataset,
                          "order": order, "stage": stage,
                          "final_f1": rows[-1]["aggregate"]["mean_f1"],
                          "accuracy": rows[-1]["aggregate"]["mean_accuracy"]}),
              flush=True)

    result = {"protocol": protocol, "rows": rows}
    save_json(out / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--mode", choices=("x-class+p", "x-class+p+a"),
                        default="x-class+p",
                        help="Student-compatible content/payload arm; A is optional")
    parser.add_argument("--orders", nargs="+", default=["order19611", "order19612"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[9300])
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[9310])
    parser.add_argument("--plan-seed", type=int, default=9320)
    parser.add_argument("--class-representation", action="store_true")
    parser.add_argument(
        "--operator-aligned", action="store_true",
        help="Use the candidate graph propagation operator in the outer loss.")
    parser.add_argument(
        "--operator-aux-weight", type=float, default=0.0,
        help="Add class-level operator alignment as an auxiliary outer loss.")
    parser.add_argument(
        "--class-constrained-p", action="store_true",
        help="Restrict Sinkhorn payload transport to matching incoming classes.")
    parser.add_argument("--p-old-context-ratio", type=float, default=0.0)
    parser.add_argument("--p-old-context-cap", type=float, default=0.0)
    parser.add_argument("--new-loss-scale", type=float, default=1.0)
    parser.add_argument("--p-response-gate", action="store_true")
    parser.add_argument("--p-gate-margin", type=float, default=0.0)
    parser.add_argument("--p-gate-old-tolerance", type=float, default=0.0)
    parser.add_argument("--p-gate-per-class", action="store_true")
    parser.add_argument("--a-message-weight", type=float, default=0.0)
    parser.add_argument("--a-marginal", action="store_true")
    parser.add_argument("--a-marginal-inner-steps", type=int, default=2)
    parser.add_argument("--a-marginal-new-margin", type=float, default=0.0)
    parser.add_argument("--a-marginal-old-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable output exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for order in args.orders:
        summaries.append(run_order(args.dataset, order, args))
    save_json(args.output_root / "summary.json", {
        "protocol": f"bounded-student-compatible-{args.mode}-unified-v1",
        "dataset": args.dataset,
        "orders": args.orders,
        "results": summaries,
    })
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "orders": args.orders, "output": str(args.output_root)}),
          flush=True)


if __name__ == "__main__":
    main()
