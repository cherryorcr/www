"""Compare the compact Unified-Q method with CaT and current GM controls.

All arms use the same current-delta-only stream, budget, prompt-only Student,
and report-dev evaluation.  Unified-Q derives X, text payload, and topology
from one class-masked differentiable assignment matrix.  The boundary arm is
allowed to read only current-to-historical edge IDs and the parent support-ID
sketch; it never reads historical raw features or text.
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
import unified_q_multimodal as UQ


ARMS = ("cat_x", "cat_gm_decoupled", "unified_q_edge_free", "unified_q",
        "unified_q_cat_anchor", "unified_q_boundary")


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


def _memory_hash(memory: dict) -> str:
    digest = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "labels", "mass", "depth", "stage", "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _edge_pairs(data: dict, current_ids: torch.Tensor,
                history_ids: torch.Tensor) -> torch.Tensor:
    """Return (current-local, historical-global) boundary edge IDs."""
    if history_ids is None or not len(history_ids):
        return torch.empty((2, 0), dtype=torch.long)
    edges = data["edge_index"].long().cpu()
    current_ids = current_ids.long().cpu()
    history_ids = history_ids.long().cpu()
    current_map = torch.full((len(data["x"]),), -1, dtype=torch.long)
    current_map[current_ids] = torch.arange(len(current_ids))
    history_mask = torch.zeros(len(data["x"]), dtype=torch.bool)
    history_mask[history_ids] = True
    left_current = current_map[edges[0]] >= 0
    right_current = current_map[edges[1]] >= 0
    left_history = history_mask[edges[0]]
    right_history = history_mask[edges[1]]
    selected = (left_current & right_history) | (right_current & left_history)
    if not bool(selected.any()):
        return torch.empty((2, 0), dtype=torch.long)
    chosen = edges[:, selected]
    current_local = torch.where(current_map[chosen[0]] >= 0,
                                current_map[chosen[0]], current_map[chosen[1]])
    historical_global = torch.where(current_map[chosen[0]] >= 0,
                                    chosen[1], chosen[0])
    pairs = torch.stack((current_local, historical_global))
    # The source graph can already contain both directions.  Duplicates are
    # harmless for the sketch but removing exact duplicates makes audits clear.
    return torch.unique(pairs, dim=1)


def delta_view(data: dict, ids: torch.Tensor, stage: int,
               history_ids: torch.Tensor) -> dict:
    ids = ids.long().cpu()
    if len(ids.unique()) != len(ids) or not bool(data["pool_mask"][ids].all()):
        raise ValueError("construction source is not unique pool-only delta")
    prompt_emb = data.get("prompt_emb")
    return {
        "ids": ids,
        "stage": int(stage),
        "x": data["x"][ids].clone(),
        "y": data["y"][ids].clone(),
        "text_emb": data["text_emb"][ids].clone(),
        "texts": [data["texts"][i] for i in ids.tolist()],
        "prompt_emb": prompt_emb,
        "edge_index": BASE.induced_edges(data["edge_index"], ids, len(data["x"])),
        "boundary_edge_index": _edge_pairs(data, ids, history_ids),
    }


def _clone_block(block: dict, labels: torch.Tensor) -> dict:
    result = {}
    for key, value in block.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    result["labels"] = labels.detach().cpu().long().clone()
    return result


def _append_edge_free(parent: dict | None, block: dict) -> dict:
    if parent is None:
        return block
    old_n = len(parent["x"])
    edges = block.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    if edges.numel():
        raise AssertionError("edge-free control unexpectedly contains topology")
    return {
        "x": torch.cat((parent["x"], block["x"])),
        "text_emb": torch.cat((parent["text_emb"], block["text_emb"])),
        "labels": torch.cat((parent["labels"], block["labels"])),
        "mass": torch.cat((parent["mass"], block["mass"])),
        "depth": torch.cat((parent["depth"], block["depth"])),
        "stage": torch.cat((parent["stage"], block["stage"])),
        "texts": list(parent["texts"]) + list(block["texts"]),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }


def _check_memory(memory: dict, parent: dict | None, budget: int,
                  preserve_parent: bool = True) -> None:
    if len(memory["x"]) != int(budget) or len(memory["texts"]) != int(budget):
        raise AssertionError("memory budget or text count mismatch")
    for key in ("x", "text_emb", "labels", "mass", "depth", "stage"):
        if key not in memory or len(memory[key]) != int(budget):
            raise AssertionError(f"invalid memory field {key}")
        if not bool(torch.isfinite(memory[key].float()).all()):
            raise AssertionError(f"nonfinite memory field {key}")
    edges = memory.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    if edges.shape[0] != 2 or (edges.numel() and
                               (int(edges.min()) < 0 or
                                int(edges.max()) >= int(budget))):
        raise AssertionError("invalid memory edge endpoint")
    if parent is None or not preserve_parent:
        return
    old_n = len(parent["x"])
    for key in ("text_emb", "labels", "mass", "depth", "stage"):
        if not torch.equal(memory[key][:old_n], parent[key]):
            raise AssertionError(f"parent changed outside X: {key}")
    if memory["texts"][:old_n] != parent["texts"]:
        raise AssertionError("parent text changed")
    old_edges = edges[:, (edges < old_n).all(dim=0)]
    parent_edges = parent.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    if not torch.equal(old_edges, parent_edges):
        raise AssertionError("parent topology changed")


def _protocol(args, cat_cfg: M.Config, gm_cfg: GM.Config,
              uq_cfg: UQ.Config) -> dict:
    return {
        "name": "unified-q-multimodal-continual-tag-20261006",
        "status": "compact unified-Q feasibility comparison",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": list(args.arms),
        "report_seeds": args.report_seeds,
        "construction_access": "current raw delta, current boundary edge IDs, parent synthetic memory and support-ID sketch",
        "historical_raw_reads": 0,
        "budget": "same cumulative 5% legal-pool budget",
        "student": P.model_config(),
        "topology": "Q-derived soft topology, hard materialized for saved memory",
        "configs": {"cat": asdict(cat_cfg), "gm": asdict(gm_cfg), "unified_q": asdict(uq_cfg)},
        "arms_definition": {
            "cat_x": "CaT X with current-delta nearest text attachment, edge-free",
            "cat_gm_decoupled": "existing CaT warm-started decoupled multimodal GM, edge-free",
            "unified_q_edge_free": "one Q derives X/text, but saved Student topology is edge-free",
            "unified_q": "one class-masked Q derives X/text/topology without boundary sketch",
            "unified_q_cat_anchor": "Unified-Q with CaT X as a bounded anchor term",
            "unified_q_boundary": "Unified-Q with current-to-historical boundary sketch",
        },
        "selection": "construction never reads report/test labels or F1",
        "evaluation": "same prompt-only Student, Macro-F1 and Accuracy",
    }


def construct_cell(data: dict, order: str, args, cat_cfg: M.Config,
                   gm_cfg: GM.Config, uq_cfg: UQ.Config) -> None:
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    construction_path = root / "construction.json"
    if construction_path.exists():
        return
    ids, budgets = BASE.set_stream(data, order)
    parents = {arm: None for arm in args.arms}
    records = []
    history_ids = torch.empty(0, dtype=torch.long)
    for stage in range(3):
        delta = delta_view(data, ids[stage], stage, history_ids)
        old_rows = len(next(iter(parents.values()))["x"]) \
            if next(iter(parents.values())) is not None else 0
        new_rows = int(budgets[stage] - old_rows)
        labels = M.row_labels(M.ratio_allocation(delta["y"], new_rows))
        started = time.perf_counter()
        base_block, cat_audit = M.cat_block(
            delta, new_rows, data["encoder"], replace(cat_cfg, text_sources=5), args.device)
        x0 = base_block["x"].detach().cpu()
        for arm in args.arms:
            parent = parents[arm]
            arm_started = time.perf_counter()
            if arm == "cat_x":
                block = _clone_block(base_block, labels)
                block["edge_index"] = torch.empty((2, 0), dtype=torch.long)
                audit = {"constructor": "cat_x_plus_cctp_text", "cat_audit": cat_audit}
                memory = _append_edge_free(parent, block)
            elif arm == "cat_gm_decoupled":
                cfg = replace(gm_cfg, update_x=True, update_route=True,
                              allow_old_update=False, gradient_matching_mode="decoupled",
                              topology_mode="none")
                block, audit, _ = GM.optimize_stage(
                    delta, parent, x0, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device)
                audit["cat_audit"] = cat_audit
                memory = GM.append_memory(parent, block)
            else:
                use_boundary = arm == "unified_q_boundary"
                anchor = x0 if arm == "unified_q_cat_anchor" else None
                block, audit = UQ.optimize_stage(
                    delta, parent, labels, data["encoder"], data["prompt_emb"],
                    uq_cfg, args.device, cat_anchor_x=anchor,
                    use_boundary=use_boundary,
                    use_topology=arm != "unified_q_edge_free")
                audit["cat_audit"] = cat_audit if anchor is not None else None
                memory = UQ.append_memory(parent, block)
            _check_memory(memory, parent, int(budgets[stage]))
            memory_path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, memory_path)
            record = {
                "arm": arm,
                "stage": int(stage),
                "rows": int(budgets[stage]),
                "memory_file": memory_path.name,
                "memory_sha256": _memory_hash(memory),
                "file_sha256": file_hash(memory_path),
                "delta_ids": ids[stage].tolist(),
                "history_ids": history_ids.tolist(),
                "boundary_edges": int(delta["boundary_edge_index"].shape[1]),
                "parent_sha256": _memory_hash(parent) if parent else None,
                "construction_seconds": float(time.perf_counter() - arm_started),
                "cat_seconds_shared": float(time.perf_counter() - started),
                "audit": audit,
            }
            records.append(record)
            parents[arm] = memory
            print(json.dumps({"event": "constructed", "dataset": data["dataset"],
                              "order": order, "stage": stage, "arm": arm,
                              "seconds": record["construction_seconds"]}), flush=True)
        history_ids = torch.cat((history_ids, ids[stage].long().cpu()))
    save(construction_path, {"budgets": budgets, "records": records})


def evaluate_cell(data: dict, order: str, args) -> None:
    BASE.set_stream(data, order)
    root = args.output_root / data["dataset"] / order
    construction = read(root / "construction.json")
    rows = []
    for item in construction["records"]:
        memory = torch.load(root / item["memory_file"], map_location="cpu",
                            weights_only=False)
        steps = int(args.student_steps or U.protocol_v3.student_steps(len(memory["x"])))
        scores = []
        for seed in args.report_seeds:
            model = P.train_prompt_student(
                memory, data["prompt_emb"], int(seed), args.device, steps,
                lr=0.001, batch=24)
            scores.append({"seed": int(seed), "steps": steps,
                           **BASE.score_model(model, data, data["stage_classes"],
                                              int(item["stage"]))})
            del model
        aggregate = BASE.aggregate_stage_scores(scores)
        rows.append({"dataset": data["dataset"], "order": order,
                     "arm": item["arm"], "stage": int(item["stage"]),
                     "memory_rows": int(item["rows"]), "aggregate": aggregate,
                     "per_student": scores,
                     "construction_seconds": item["construction_seconds"]})
        save(root / "progress.json", {"rows": rows})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                          "order": order, "stage": item["stage"],
                          "arm": item["arm"], "f1": aggregate["mean_f1"],
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
    parser.add_argument("--gm-steps", type=int, default=30)
    parser.add_argument("--uq-steps", type=int, default=30)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "construct", "evaluate"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cat_cfg = M.Config(seed=271006, cat_steps=args.cat_steps,
                       text_sources=5, text_mode="nearest")
    gm_cfg = GM.Config(steps=args.gm_steps, gradient_matching_mode="decoupled",
                       topology_mode="none", proxy_seeds=(2711,))
    uq_cfg = UQ.Config(steps=args.uq_steps, proxy_seeds=(2711,),
                       materialize_top_k=1, straight_through_text=True,
                       hard_topology_forward=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = _protocol(args, cat_cfg, gm_cfg, uq_cfg)
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != manifest:
            raise ValueError("existing output root has a different protocol")
    else:
        save(protocol_path, manifest)
    data = {}
    for name in args.datasets:
        value = BASE.load_data(name, args.minilm, args.device, args.cache_root)
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
        data[name] = value
    phases = ("construct", "evaluate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "construct":
                    construct_cell(data[name], order, args, cat_cfg, gm_cfg, uq_cfg)
                else:
                    evaluate_cell(data[name], order, args)
    if args.phase == "all":
        all_rows = []
        for name in args.datasets:
            for order in args.orders:
                all_rows.extend(read(args.output_root / name / order / "results.json")["rows"])
        save(args.output_root / "results.json", {"protocol": manifest, "rows": all_rows})
    print("UNIFIED_Q_MULTIMODAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

