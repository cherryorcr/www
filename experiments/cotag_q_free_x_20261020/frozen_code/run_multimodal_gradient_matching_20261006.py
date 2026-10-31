"""Evaluate CaT warm-started one-step multimodal gradient matching.

The runner keeps the existing current-delta-only stream and evaluates every
arm with the same prompt-only zero-shot Student.  It is intentionally
independent from the older CaT multimodal feasibility runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import external_continual_mechanisms as M
import graphtext_prompt_student as P
import multimodal_gradient_matching as GM
import run_external_continual_adapters_20260927 as E
import run_semantic_recursive_unified_baselines_20260926 as BASE
import run_unified_v3_continual as U


ARMS = ("cat_x", "cat_gm_x", "cat_gm_text", "cat_gm_joint",
        "cat_gm_decoupled", "cat_gm_decoupled_topology",
        "cat_gm_decoupled_em",
        "cat_gm_joint_old")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                         encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clone_block(block: dict, labels: torch.Tensor) -> dict:
    result = {}
    for key, value in block.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif key == "texts":
            result[key] = list(value)
        else:
            result[key] = value
    result["labels"] = labels.detach().cpu().long().clone()
    return result


def protocol(args, cat_cfg: M.Config, gm_cfg: GM.Config) -> dict:
    return {
        "name": "cat-warm-start-one-step-multimodal-gradient-matching-20261006",
        "status": "independent feasibility and mechanism validation",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": list(args.arms),
        "report_seeds": args.report_seeds,
        "construction_access": "current raw delta plus already materialized synthetic parent",
        "historical_raw_reads": 0,
        "budget": "same cumulative 5% legal-pool budget as the existing suite",
        "topology": ("edge-free/self-loop-only except "
                     "cat_gm_decoupled_topology uses fixed Q^T A Q"),
        "student": P.model_config(),
        "construction": {
            "cat": asdict(cat_cfg),
            "gradient_matching": asdict(gm_cfg),
        },
        "arms_definition": {
            "cat_x": "CaT X plus current-delta nearest text attachment",
            "cat_gm_x": "CaT X plus GM; routing initialized but frozen",
            "cat_gm_text": "CaT X frozen plus trainable differentiable text routing",
            "cat_gm_joint": "CaT X plus trainable X and text routing",
            "cat_gm_decoupled": "joint GM with graph/text gradient blocks assigned to X/routing",
            "cat_gm_decoupled_topology": "decoupled GM plus fixed balanced slot coarsening topology",
            "cat_gm_decoupled_em": "decoupled GM plus EM-like message-aware Q reassignment",
            "cat_gm_joint_old": "joint GM with bounded old synthetic X revision",
        },
        "selection": "construction does not inspect report/test F1",
        "evaluation": "all-prompt report-dev Macro-F1 and Accuracy",
    }


def make_parent(parent: dict | None) -> dict | None:
    if parent is None:
        return None
    # The GM branch always writes labels.  Fail loudly instead of silently
    # reconstructing them from stage metadata if an incompatible parent is
    # accidentally supplied.
    if "labels" not in parent:
        raise ValueError("GM parent memory is missing construction labels")
    return parent


def check_memory(memory: dict, parent: dict | None, budget: int,
                 allow_old_update: bool, radius: float) -> None:
    """Validate the common memory contract, with one bounded old-X exception."""
    if len(memory["x"]) != int(budget) or len(memory["texts"]) != int(budget):
        raise AssertionError("memory budget or text count mismatch")
    for key in ("x", "text_emb", "labels", "mass", "depth", "stage"):
        if key not in memory or len(memory[key]) != int(budget):
            raise AssertionError(f"invalid memory field {key}")
        if isinstance(memory[key], torch.Tensor) and not bool(torch.isfinite(memory[key].float()).all()):
            raise AssertionError(f"nonfinite memory field {key}")
    edges = memory.get("edge_index",
                       torch.empty((2, 0), dtype=torch.long))
    if edges.shape[0] != 2 or (edges.numel() and
                               (int(edges.min()) < 0 or
                                int(edges.max()) >= int(budget))):
        raise AssertionError("invalid memory edge endpoint")
    if parent is None:
        return
    old_n = len(parent["x"])
    for key in ("text_emb", "labels", "mass", "depth", "stage"):
        if not torch.equal(memory[key][:old_n], parent[key]):
            raise AssertionError(f"parent changed outside X: {key}")
    if memory["texts"][:old_n] != parent["texts"]:
        raise AssertionError("parent text changed")
    parent_edges = parent.get("edge_index",
                              torch.empty((2, 0), dtype=torch.long))
    old_edges = edges[:, (edges < old_n).all(dim=0)]
    if not torch.equal(old_edges, parent_edges):
        raise AssertionError("parent topology changed")
    shift = memory["x"][:old_n] - parent["x"]
    limit = float(radius) * parent["x"].norm(dim=1).clamp_min(1e-3) + 1e-5
    if bool((shift.norm(dim=1) > limit).any()):
        raise AssertionError("old X update exceeded trust radius")


def construct_cell(data: dict, order: str, args, cat_cfg: M.Config,
                   gm_cfg: GM.Config) -> None:
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    construction_path = root / "construction.json"
    if construction_path.exists():
        return
    ids, budgets = BASE.set_stream(data, order)
    parents = {arm: None for arm in args.arms}
    records = []
    for stage in range(3):
        delta = E.delta_view(data, ids[stage], stage)
        old_rows = len(next(iter(parents.values()))["x"]) \
            if next(iter(parents.values())) is not None else 0
        new_rows = int(budgets[stage] - old_rows)
        if new_rows < 1:
            raise ValueError("stage has no new rows")
        labels = M.row_labels(M.ratio_allocation(delta["y"], new_rows))
        cat_stage = replace(cat_cfg, text_sources=5)
        started = time.perf_counter()
        base_block, cat_audit = M.cat_block(
            delta, new_rows, data["encoder"], cat_stage, args.device)
        x0 = base_block["x"].detach().cpu()
        for arm in args.arms:
            parent = make_parent(parents[arm])
            arm_started = time.perf_counter()
            if arm == "cat_x":
                block = clone_block(base_block, labels)
                block["edge_index"] = torch.empty((2, 0), dtype=torch.long)
                block_audit = {
                    "constructor": "cat_x_plus_cctp_text",
                    "caT_audit": cat_audit,
                    "x_updated_after_cat": False,
                    "text_routing_updated": False,
                }
                old_final = None
            else:
                cfg = replace(
                    gm_cfg,
                    update_x=arm in ("cat_gm_x", "cat_gm_joint",
                                     "cat_gm_decoupled", "cat_gm_decoupled_topology",
                                     "cat_gm_decoupled_em",
                                     "cat_gm_joint_old"),
                    update_route=arm in ("cat_gm_text", "cat_gm_joint",
                                         "cat_gm_decoupled", "cat_gm_decoupled_topology",
                                         "cat_gm_decoupled_em",
                                         "cat_gm_joint_old"),
                    allow_old_update=arm == "cat_gm_joint_old",
                    gradient_matching_mode=("decoupled" if arm in
                                            ("cat_gm_decoupled",
                                             "cat_gm_decoupled_topology",
                                             "cat_gm_decoupled_em")
                                            else gm_cfg.gradient_matching_mode),
                    topology_mode=("fixed_q" if arm == "cat_gm_decoupled_topology"
                                   else "em_q" if arm == "cat_gm_decoupled_em"
                                   else gm_cfg.topology_mode),
                )
                block, block_audit, old_final = GM.optimize_stage(
                    delta, parent, x0, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device)
                block_audit["caT_audit"] = cat_audit
            memory = GM.append_memory(parent, block, old_final)
            check_memory(memory, parent, int(budgets[stage]),
                         allow_old_update=(arm == "cat_gm_joint_old"),
                         radius=gm_cfg.old_trust_radius)
            memory_path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, memory_path)
            record = {
                "arm": arm,
                "stage": int(stage),
                "rows": int(budgets[stage]),
                "memory_file": memory_path.name,
                "memory_sha256": E.memory_hash(memory),
                "file_sha256": file_hash(memory_path),
                "delta_ids": ids[stage].tolist(),
                "parent_sha256": E.memory_hash(parent) if parent else None,
                "construction_seconds": float(time.perf_counter() - arm_started),
                "cat_seconds_shared": float(time.perf_counter() - started),
                "audit": block_audit,
                "resources": E.resource_stats(memory, data["encoder"]),
            }
            records.append(record)
            parents[arm] = memory
            print(json.dumps({"event": "constructed", "dataset": data["dataset"],
                              "order": order, "stage": stage, "arm": arm,
                              "seconds": record["construction_seconds"]}),
                  flush=True)
    save(construction_path, {"budgets": budgets, "records": records})


def evaluate_cell(data: dict, order: str, args) -> None:
    BASE.set_stream(data, order)
    root = args.output_root / data["dataset"] / order
    construction = read(root / "construction.json")
    rows = []
    for item in construction["records"]:
        memory = torch.load(root / item["memory_file"],
                            map_location="cpu", weights_only=False)
        steps = int(args.student_steps or U.protocol_v3.student_steps(
            len(memory["x"])))
        scores = []
        for seed in args.report_seeds:
            model = P.train_prompt_student(
                memory, data["prompt_emb"], int(seed), args.device, steps,
                lr=0.001, batch=24)
            scores.append({
                "seed": int(seed),
                "steps": steps,
                **BASE.score_model(model, data, data["stage_classes"],
                                   int(item["stage"])),
            })
            del model
        aggregate = BASE.aggregate_stage_scores(scores)
        rows.append({
            "dataset": data["dataset"],
            "order": order,
            "arm": item["arm"],
            "stage": int(item["stage"]),
            "memory_rows": int(item["rows"]),
            "aggregate": aggregate,
            "per_student": scores,
            "construction_seconds": item["construction_seconds"],
        })
        save(root / "progress.json", {"rows": rows})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                          "order": order, "stage": item["stage"],
                          "arm": item["arm"],
                          "f1": aggregate["mean_f1"],
                          "accuracy": aggregate["mean_accuracy"]}), flush=True)
    save(root / "results.json", {"rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int, default=[150, 151])
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--gm-steps", type=int, default=40)
    parser.add_argument("--candidate-cap", type=int, default=0)
    parser.add_argument("--new-trust-radius", type=float, default=0.05)
    parser.add_argument("--lambda-cat-anchor", type=float, default=0.5)
    parser.add_argument("--lambda-route-anchor", type=float, default=1.0)
    parser.add_argument("--straight-through-route", action="store_true")
    parser.add_argument("--topology-mode", choices=("none", "fixed_q", "em_q"),
                        default="none")
    parser.add_argument("--em-updates", type=int, default=0)
    parser.add_argument("--em-interval", type=int, default=10)
    parser.add_argument("--gradient-mode", choices=("flat", "blockwise", "decoupled"),
                        default="blockwise")
    parser.add_argument("--proxy-seeds", nargs="+", type=int,
                        default=[2711])
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "construct", "evaluate"),
                        default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cat_cfg = M.Config(seed=271006, cat_steps=args.cat_steps,
                       text_mode="nearest", text_sources=5)
    gm_cfg = GM.Config(steps=args.gm_steps,
                       gradient_matching_mode=args.gradient_mode,
                       proxy_seeds=tuple(args.proxy_seeds),
                       candidate_cap=args.candidate_cap,
                       new_trust_radius=args.new_trust_radius,
                       lambda_cat_anchor=args.lambda_cat_anchor,
                       lambda_route_anchor=args.lambda_route_anchor,
                       topology_mode=args.topology_mode,
                       em_updates=args.em_updates,
                       em_interval=args.em_interval,
                       straight_through_route=args.straight_through_route)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = protocol(args, cat_cfg, gm_cfg)
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != manifest:
            raise ValueError("existing output root has a different protocol")
    else:
        save(protocol_path, manifest)
    data = {}
    for name in args.datasets:
        value = BASE.load_data(name, args.minilm, args.device, args.cache_root)
        value["real_adj"] = P.normalized_adjacency(
            value["edge_index"], len(value["x"]), args.device)
        value["x_device"] = value["x"].float().to(args.device)
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
        data[name] = value
    phases = ("construct", "evaluate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "construct":
                    construct_cell(data[name], order, args, cat_cfg, gm_cfg)
                else:
                    evaluate_cell(data[name], order, args)
    if args.phase == "all":
        all_rows = []
        for name in args.datasets:
            for order in args.orders:
                all_rows.extend(read(args.output_root / name / order /
                                     "results.json")["rows"])
        save(args.output_root / "results.json", {
            "protocol": manifest,
            "rows": all_rows,
        })
    print("MULTIMODAL_GRADIENT_MATCHING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
