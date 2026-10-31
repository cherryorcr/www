"""Re-evaluate common TAG condensation arms with one frozen prompt Student.

This runner is intentionally separate from every historical Student runner.
Each construction arm receives the same current-delta-only stream and memory
budget.  The downstream model has no classifier head: at evaluation it uses
only encoded class prompts.  The old Student results remain a separate
historical table and are never mixed with this protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import external_continual_mechanisms as M
import graphtext_prompt_student as PS
import protocol_v3
import run_public_continual_baselines_large_20261001 as PUB
import run_semantic_recursive_unified_baselines_20260926 as BASE
import run_student_compatible_append as SCA
import run_unified_v3_continual as U
import cat_multimodal_minimal as MM
import x_conditioned_payload_routing as XCR
import x_conditioned_payload_partition as XPP


ARMS = (
    "cat_nearest",
    "formal_prompt_margin",
    "formal_prompt_margin_x_refine",
    "formal_prompt_margin_text_topology",
    "gecc_hard_tag",
    "random_real",
    "k_center",
    "kmeans_medoid",
    "fixed_x_xsp",
    "xsp_x_cattext",
    "xsp_st",
    "full_xsp",
    "x_conditioned_routing",
    "x_conditioned_routing_x_coverage",
    "x_conditioned_routing_coherent",
    "x_conditioned_routing_coherent_adaptive",
    "x_conditioned_routing_coherent_balanced",
    "x_conditioned_routing_coherent_anchor",
)
REAL_ARMS = {
    "random_real": "random_real",
    "k_center": "k_center",
    "kmeans_medoid": "kmeans_medoid",
}
TOPOLOGY_ARMS = {"formal_prompt_margin_text_topology"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                         encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "labels", "mass", "depth", "stage",
                "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    digest.update(json.dumps(memory["texts"], ensure_ascii=True).encode())
    return digest.hexdigest()


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def with_labels(block: dict, labels: torch.Tensor) -> dict:
    out = dict(block)
    out["labels"] = labels.detach().cpu().long()
    if "edge_index" not in out:
        out["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    return out


def append_memory(parent: dict | None, block: dict) -> dict:
    if parent is None:
        return block
    base = M.append_edge_free(parent, block)
    base["labels"] = torch.cat((parent["labels"], block["labels"])).long()
    return base


def append_topology_memory(parent: dict | None, block: dict) -> dict:
    """Append a block while preserving old edges and adding bounded cross edges."""
    if parent is None:
        return block
    out = {key: torch.cat((parent[key], block[key]))
           for key in ("x", "text_emb", "mass", "depth", "stage")}
    out["texts"] = list(parent["texts"]) + list(block["texts"])
    out["labels"] = torch.cat((parent["labels"], block["labels"])).long()
    out["edge_index"] = SCA.append_topology(parent, block, cross_degree=1)
    return out


def check_memory(memory: dict, parent: dict | None, budget: int,
                 topology: bool = False) -> None:
    if len(memory["x"]) != int(budget):
        raise AssertionError("memory row budget mismatch")
    if not topology and memory["edge_index"].numel():
        raise AssertionError("prompt-small protocol must be edge-free")
    if topology and memory["edge_index"].numel():
        edge = memory["edge_index"].long()
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise AssertionError("invalid synthetic topology shape")
        if int(edge.min()) < 0 or int(edge.max()) >= len(memory["x"]):
            raise AssertionError("synthetic topology index out of bounds")
    for key in ("x", "text_emb", "labels", "mass", "depth", "stage"):
        if len(memory[key]) != int(budget):
            raise AssertionError(f"memory field length mismatch: {key}")
        if not bool(torch.isfinite(memory[key]).all()):
            raise AssertionError(f"nonfinite memory field: {key}")
    if len(memory["texts"]) != int(budget):
        raise AssertionError("memory/text length mismatch")
    if parent is not None:
        old = len(parent["x"])
        for key in ("x", "text_emb", "labels", "mass", "depth", "stage"):
            if not torch.equal(memory[key][:old], parent[key]):
                raise AssertionError(f"sealed parent prefix changed: {key}")
        if memory["texts"][:old] != parent["texts"]:
            raise AssertionError("sealed parent text prefix changed")
        if topology and not torch.equal(
                memory["edge_index"][:,
                    (memory["edge_index"] < old).all(dim=0)],
                parent["edge_index"]):
            raise AssertionError("sealed old-old topology changed")


def _xsp_block(delta: dict, cat_block: dict, labels: torch.Tensor,
               data: dict, arm: str, args) -> tuple[dict, dict]:
    source_features = M.propagated_features(delta, (0.0, 1.0), args.device)
    variant = "st" if arm == "xsp_st" else "current"
    cfg = XPP.PartitionConfig(
        steps=0 if arm == "fixed_x_xsp" else int(args.partition_steps),
        hidden=256,
        lr_x=5e-4,
        text_chars=320,
        max_sources=6,
        straight_through_sinkhorn=(variant == "st"),
        lambda_coherence=0.0,
        lambda_hard=0.0,
        hard_refresh_steps=0,
        seed=int(args.construction_seed) + 152,
    )
    block, audit = XPP.build_x_conditioned_slot_partition(
        delta=delta,
        source_features=source_features,
        x0=cat_block["x"],
        labels=labels,
        encoder=data["encoder"],
        cfg=cfg,
        mass=cat_block["mass"],
        device=args.device,
    )
    if arm == "xsp_x_cattext":
        block["texts"] = list(cat_block["texts"])
        block["text_emb"] = cat_block["text_emb"].clone()
    audit.update({
        "prompt_student_arm": arm,
        "x_source": "CaT anchor" if arm == "fixed_x_xsp" else "XSP refinement",
        "text_source": ("CaT nearest top-5" if arm == "xsp_x_cattext"
                         else "XSP slot-conditioned payload"),
        "historical_raw_reads": 0,
        "current_delta_only": True,
        "old_memory_updated": False,
        "topology_updated": False,
    })
    return with_labels(block, labels), audit


def _x_conditioned_routing_block(delta: dict, cat_block: dict,
                                 cat_audit: dict, labels: torch.Tensor,
                                 data: dict, args,
                                 candidate_mode: str = "standard",
                                 prompt_student_arm: str = "x_conditioned_routing",
                                 balanced_materialization: bool = False,
                                 adaptive_slots: bool = False
                                 ) -> tuple[dict, dict]:
    """Route each CaT synthetic row to a bounded real text slot bank."""
    source_features = M.propagated_features(delta, (0.0, 1.0), args.device)
    cfg = MM.MMConfig(
        seed=int(args.construction_seed) + 76,
        text_sources=5,
        text_chars=320,
        support_pool_size=32,
        chunk_chars=120,
        payload_max_chunks=8,
        payload_max_sources=5,
        payload_graph_weight=0.35,
        payload_text_weight=0.45,
        payload_prompt_weight=0.20,
        payload_redundancy=0.20,
        payload_require_lead=True,
        payload_contrastive_weight=0.35,
        payload_source_repeat_penalty=0.10,
        payload_assignment_text_weight=0.50,
        payload_assignment_variance_weight=0.25,
        lambda_prompt_margin=0.15,
        prompt_margin=0.05,
    )
    block, audit = XCR.attach_x_conditioned_multi_payload(
        delta=delta,
        source_features=source_features,
        x0=cat_block["x"],
        text0=cat_block["text_emb"],
        base_texts=cat_block["texts"],
        base_source_ids=cat_audit["text_source_ids"],
        labels=labels,
        encoder=data["encoder"],
        cfg=cfg,
        mass=cat_block["mass"],
        device=args.device,
        max_slots=int(args.routing_max_slots),
        routing_steps=int(args.routing_steps),
        routing_temperature=float(args.routing_temperature),
        load_weight=float(args.routing_load_weight),
        anchor_weight=float(args.routing_anchor_weight),
        candidate_mode=candidate_mode,
        balanced_materialization=balanced_materialization,
        cctp_anchor_swap=(prompt_student_arm ==
                          "x_conditioned_routing_coherent_anchor"),
            adaptive_slots=bool(adaptive_slots),
    )
    audit.update({
        "prompt_student_arm": prompt_student_arm,
        "base_constructor": "CaT class-wise MMD X + CCTP top-5",
        "base_cat_audit": cat_audit,
        "historical_raw_reads": 0,
        "current_delta_only": True,
        "old_memory_updated": False,
        "topology_updated": False,
    })
    return with_labels(block, labels), audit


def _formal_prompt_margin_x_refine(
        delta: dict, formal_block: dict, labels: torch.Tensor,
        data: dict, args) -> tuple[dict, dict]:
    """Refine only CaT X while keeping the formal prompt-margin payload.

    This is deliberately not a new text selector: the payload is copied from
    the formal prompt-margin construction, while X is updated by the bounded
    X-conditioned partition objective.  The separation makes the effect of
    bounded X refinement identifiable and preserves the same hard payload
    materialization used by the strongest baseline arm.
    """
    source_features = M.propagated_features(delta, (0.0, 1.0), args.device)
    cfg = XPP.PartitionConfig(
        steps=int(args.x_refine_steps),
        hidden=256,
        lr_x=5e-4,
        text_chars=320,
        max_sources=6,
        trust_radius=float(args.x_refine_trust_radius),
        lambda_anchor=float(args.x_refine_anchor),
        straight_through_sinkhorn=False,
        lambda_coherence=0.0,
        lambda_hard=0.0,
        hard_refresh_steps=0,
        seed=int(args.construction_seed) + 911,
    )
    refined, refine_audit = XPP.build_x_conditioned_slot_partition(
        delta=delta,
        source_features=source_features,
        x0=formal_block["x"],
        labels=labels,
        encoder=data["encoder"],
        cfg=cfg,
        mass=formal_block["mass"],
        device=args.device,
    )
    out = dict(formal_block)
    out["x"] = refined["x"]
    out["labels"] = labels.detach().cpu().long()
    out["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    audit = {
        "constructor": "CaT X + formal prompt-margin payload + bounded X refinement",
        "payload_source": "formal_prompt_margin block (unchanged)",
        "x_source": "bounded X-conditioned refinement from formal X",
        "refinement": refine_audit,
        "historical_raw_reads": 0,
        "current_delta_only": True,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "trust_radius": float(args.x_refine_trust_radius),
        "x_refine_steps": int(args.x_refine_steps),
    }
    return out, audit


def build_stage_blocks(data: dict, ids: torch.Tensor, rows: int, stage: int,
                       args) -> tuple[dict[str, dict], dict[str, dict]]:
    delta = PUB.delta_view(data, ids, stage)
    labels = M.row_labels(M.ratio_allocation(delta["y"], rows))
    blocks, audits = {}, {}
    requested = set(args.arms)

    need_cat = bool(requested & {
        "cat_nearest", "formal_prompt_margin",
        "formal_prompt_margin_x_refine", "fixed_x_xsp",
        "formal_prompt_margin_text_topology", "xsp_x_cattext",
        "xsp_st", "full_xsp", "x_conditioned_routing",
        "x_conditioned_routing_x_coverage",
        "x_conditioned_routing_coherent",
        "x_conditioned_routing_coherent_adaptive",
        "x_conditioned_routing_coherent_balanced",
        "x_conditioned_routing_coherent_anchor"})
    cat_block = None
    if need_cat:
        cat_cfg = M.Config(
            seed=int(args.construction_seed),
            cat_steps=int(args.cat_steps),
            text_sources=5,
            text_mode="nearest",
        )
        cat_block, cat_audit = M.cat_block(
            delta, rows, data["encoder"], cat_cfg, args.device)
        cat_block = with_labels(cat_block, labels)
        if "cat_nearest" in requested:
            blocks["cat_nearest"] = cat_block
            audits["cat_nearest"] = {
                "constructor": "CaT class-wise MMD + nearest top-5 text",
                "cat_audit": cat_audit,
                "historical_raw_reads": 0,
                "current_delta_only": True,
                "old_memory_updated": False,
                "topology_updated": False,
            }
        if "formal_prompt_margin" in requested:
            block, audit = PUB.build_external(
                data, ids, rows, "cat_mmd_tag_class_assigned_prompt_margin",
                stage, args.device,
                M.Config(seed=int(args.construction_seed),
                         cat_steps=int(args.cat_steps),
                         text_sources=5, text_mode="nearest"), args)
            blocks["formal_prompt_margin"] = with_labels(block, labels)
            audits["formal_prompt_margin"] = audit
        if "x_conditioned_routing" in requested:
            block, audit = _x_conditioned_routing_block(
                delta, cat_block, cat_audit, labels, data, args)
            blocks["x_conditioned_routing"] = block
            audits["x_conditioned_routing"] = audit
        if "x_conditioned_routing_x_coverage" in requested:
            block, audit = _x_conditioned_routing_block(
                delta, cat_block, cat_audit, labels, data, args,
                candidate_mode="x_response_coverage",
                prompt_student_arm="x_conditioned_routing_x_coverage")
            blocks["x_conditioned_routing_x_coverage"] = block
            audits["x_conditioned_routing_x_coverage"] = audit
        if "x_conditioned_routing_coherent" in requested:
            block, audit = _x_conditioned_routing_block(
                delta, cat_block, cat_audit, labels, data, args,
                candidate_mode="coherent_raw_medoid",
                prompt_student_arm="x_conditioned_routing_coherent")
            blocks["x_conditioned_routing_coherent"] = block
            audits["x_conditioned_routing_coherent"] = audit
        if "x_conditioned_routing_coherent_adaptive" in requested:
            block, audit = _x_conditioned_routing_block(
                delta, cat_block, cat_audit, labels, data, args,
                candidate_mode="coherent_raw_medoid",
                prompt_student_arm="x_conditioned_routing_coherent_adaptive",
                adaptive_slots=True)
            blocks["x_conditioned_routing_coherent_adaptive"] = block
            audits["x_conditioned_routing_coherent_adaptive"] = audit
        if "x_conditioned_routing_coherent_balanced" in requested:
            block, audit = _x_conditioned_routing_block(
                delta, cat_block, cat_audit, labels, data, args,
                candidate_mode="coherent_raw_medoid",
                prompt_student_arm="x_conditioned_routing_coherent_balanced",
                balanced_materialization=True)
            blocks["x_conditioned_routing_coherent_balanced"] = block
            audits["x_conditioned_routing_coherent_balanced"] = audit
        if "x_conditioned_routing_coherent_anchor" in requested:
            block, audit = _x_conditioned_routing_block(
                delta, cat_block, cat_audit, labels, data, args,
                candidate_mode="coherent_raw_medoid_anchor",
                prompt_student_arm="x_conditioned_routing_coherent_anchor")
            blocks["x_conditioned_routing_coherent_anchor"] = block
            audits["x_conditioned_routing_coherent_anchor"] = audit
        if "formal_prompt_margin_text_topology" in requested:
            block, audit = PUB.build_external(
                data, ids, rows, "cat_mmd_tag_class_assigned_prompt_margin",
                stage, args.device,
                M.Config(seed=int(args.construction_seed),
                         cat_steps=int(args.cat_steps),
                         text_sources=5, text_mode="nearest"), args)
            block = with_labels(block, labels)
            block["edge_index"] = SCA.text_knn_edges(
                block["text_emb"], degree=1.0)
            audit = {
                **audit,
                "constructor": "formal prompt-margin payload + text-kNN synthetic topology",
                "topology": "within-block text-kNN; one nearest old cross edge at append",
                "historical_raw_reads": 0,
                "current_delta_only": True,
                "old_memory_updated": False,
                "topology_updated": True,
                "budget_updated": False,
            }
            blocks["formal_prompt_margin_text_topology"] = block
            audits["formal_prompt_margin_text_topology"] = audit
        if "formal_prompt_margin_x_refine" in requested:
            formal_block, formal_audit = PUB.build_external(
                data, ids, rows, "cat_mmd_tag_class_assigned_prompt_margin",
                stage, args.device,
                M.Config(seed=int(args.construction_seed),
                         cat_steps=int(args.cat_steps),
                         text_sources=5, text_mode="nearest"), args)
            formal_block = with_labels(formal_block, labels)
            block, audit = _formal_prompt_margin_x_refine(
                delta, formal_block, labels, data, args)
            audit["formal_payload_audit"] = formal_audit
            blocks["formal_prompt_margin_x_refine"] = block
            audits["formal_prompt_margin_x_refine"] = audit
        for arm in ("fixed_x_xsp", "xsp_x_cattext", "xsp_st", "full_xsp"):
            if arm in requested:
                blocks[arm], audits[arm] = _xsp_block(
                    delta, cat_block, labels, data, arm, args)

    if "gecc_hard_tag" in requested:
        cfg = M.Config(seed=int(args.construction_seed),
                       cat_steps=int(args.cat_steps), text_sources=5,
                       text_mode="nearest")
        block, audit = M.gecc_block(
            delta, rows, data["encoder"], cfg, args.device)
        blocks["gecc_hard_tag"] = with_labels(block, labels)
        audits["gecc_hard_tag"] = audit

    for arm, method in REAL_ARMS.items():
        if arm not in requested:
            continue
        block, audit = U.build_real_selection_block(
            data, ids, rows, method, stage, args.device)
        selected = audit.get("selected_source_ids")
        if selected is None:
            raise AssertionError("real baseline audit lacks selected source ids")
        selected_labels = data["y"][torch.tensor(selected)].long()
        block = with_labels(block, selected_labels)
        blocks[arm] = block
        audits[arm] = {
            **audit,
            "constructor": "label-free real-node coreset",
            "historical_raw_reads": 0,
            "current_delta_only": True,
            "old_memory_updated": False,
            "topology_updated": False,
        }
    return blocks, audits


def protocol_manifest(args, data: dict, order: str, budgets: list[int]) -> dict:
    return {
        "name": "prompt-zero-shot-small-20261006",
        "dataset": data["dataset"],
        "order": order,
        "stage_protocol": "three-stage class-incremental",
        "arms": list(args.arms),
        "memory_rate": 0.05,
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_update": False,
        "topology": ("formal payload arms edge-free; topology arm uses within-block "
                     "text-kNN plus one nearest-old cross edge"
                     if TOPOLOGY_ARMS.intersection(args.arms)
                     else "edge-free memory; Student supplies self-loops"),
        "budgets": [int(v) for v in budgets],
        "student": PS.model_config(),
        "student_seeds": [int(v) for v in args.report_seeds],
        "student_batch": int(args.student_batch),
        "student_steps": "max(50,min(2048,16*memory_rows))",
        "evaluation": "prompt-only zero-shot logits on report-dev nodes",
        "construction_seed": int(args.construction_seed),
        "cat_steps": int(args.cat_steps),
        "partition_steps": int(args.partition_steps),
    }


def prepare_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    construction_path = root / "construction.json"
    if construction_path.exists():
        return
    ids, budgets = BASE.set_stream(data, order)
    parents = {arm: None for arm in args.arms}
    records = []
    for stage, (delta_ids, budget) in enumerate(zip(ids, budgets)):
        old_rows = 0 if parents[args.arms[0]] is None else len(
            parents[args.arms[0]]["x"])
        new_rows = int(budget) - old_rows
        if new_rows <= 0 or new_rows > len(delta_ids):
            raise AssertionError(f"invalid rows for {data['dataset']}/{order}/s{stage}")
        started = time.perf_counter()
        blocks, audits = build_stage_blocks(
            data, delta_ids, new_rows, stage, args)
        for arm in args.arms:
            if arm in TOPOLOGY_ARMS:
                memory = append_topology_memory(parents[arm], blocks[arm])
            else:
                memory = append_memory(parents[arm], blocks[arm])
            check_memory(memory, parents[arm], int(budget),
                         topology=arm in TOPOLOGY_ARMS)
            path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, path)
            sync(args.device)
            record = {
                "dataset": data["dataset"],
                "order": order,
                "arm": arm,
                "stage": int(stage),
                "rows": int(budget),
                "delta_ids": [int(v) for v in delta_ids.tolist()],
                "memory_file": path.name,
                "memory_sha256": memory_hash(memory),
                "file_sha256": sha256_file(path),
                "parent_sha256": (memory_hash(parents[arm])
                                   if parents[arm] is not None else None),
                "construction_seconds": float(time.perf_counter() - started),
                "audit": audits[arm],
            }
            records.append(record)
            parents[arm] = memory
            print(json.dumps({"event": "constructed", "dataset": data["dataset"],
                              "order": order, "arm": arm, "stage": stage}),
                  flush=True)
    save(construction_path, {
        "protocol": protocol_manifest(args, data, order, budgets),
        "stage_classes": data["stage_classes"],
        "budgets": [int(v) for v in budgets],
        "records": records,
    })


def verify_cell(data: dict, order: str, args) -> dict:
    root = args.output_root / data["dataset"] / order
    construction = read(root / "construction.json")
    previous = {arm: None for arm in args.arms}
    ids, budgets = BASE.set_stream(data, order)
    for record in construction["records"]:
        arm, stage = record["arm"], int(record["stage"])
        if record["delta_ids"] != [int(v) for v in ids[stage].tolist()]:
            raise AssertionError("source delta changed")
        path = root / record["memory_file"]
        if sha256_file(path) != record["file_sha256"]:
            raise AssertionError("memory file changed")
        memory = torch.load(path, map_location="cpu", weights_only=False)
        if memory_hash(memory) != record["memory_sha256"]:
            raise AssertionError("memory content changed")
        check_memory(memory, previous[arm], int(budgets[stage]),
                     topology=arm in TOPOLOGY_ARMS)
        previous[arm] = memory
    return construction


def aggregate(scores: list[dict], stage: int) -> dict:
    def mean(key: str):
        values = [float(row[key]) for row in scores if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "mean_f1": mean("cumulative_f1"),
        "std_f1": float(np.std([r["cumulative_f1"] for r in scores], ddof=1))
        if len(scores) > 1 else 0.0,
        "mean_accuracy": mean("accuracy"),
        "std_accuracy": float(np.std([r["accuracy"] for r in scores], ddof=1))
        if len(scores) > 1 else 0.0,
        "old_f1": mean("old_f1"),
        "new_f1": mean("new_f1"),
        "stage_avg_f1": float(np.mean([
            np.mean([item["macro_f1"] for item in r["per_stage"]])
            for r in scores])),
        "stage_avg_accuracy": float(np.mean([
            np.mean([item["accuracy"] for item in r["per_stage"]])
            for r in scores])),
        "per_stage_f1": [float(np.mean([r["per_stage"][s]["macro_f1"]
                                         for r in scores]))
                          for s in range(stage + 1)],
        "per_stage_accuracy": [float(np.mean([r["per_stage"][s]["accuracy"]
                                               for r in scores]))
                                for s in range(stage + 1)],
    }


def evaluate_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(data, order, args)
    result_rows = []
    for record in construction["records"]:
        arm, stage = record["arm"], int(record["stage"])
        memory = torch.load(root / record["memory_file"],
                            map_location="cpu", weights_only=False)
        steps = int(protocol_v3.student_steps(
            len(memory["x"]), override=args.student_steps))
        scores = []
        for seed in args.report_seeds:
            score_path = root / f"{arm}_s{stage}_seed{seed}.json"
            checkpoint_path = root / f"{arm}_s{stage}_seed{seed}.pt"
            if score_path.exists() and checkpoint_path.exists():
                score = read(score_path)
            else:
                started = time.perf_counter()
                model = PS.train_prompt_student(
                    memory, data["prompt_emb"], int(seed), args.device, steps,
                    lr=args.student_lr, batch=args.student_batch)
                score = {
                    "seed": int(seed),
                    "steps": steps,
                    **BASE.score_model(model, data, data["stage_classes"], stage),
                    "student_seconds": float(time.perf_counter() - started),
                    "memory_sha256": record["memory_sha256"],
                }
                torch.save({"state": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()}},
                           checkpoint_path)
                score["checkpoint_sha256"] = sha256_file(checkpoint_path)
                save(score_path, score)
                del model
            if score.get("memory_sha256") != record["memory_sha256"]:
                raise AssertionError("score references a different memory")
            scores.append(score)
        result_rows.append({
            **record,
            "steps": steps,
            "aggregate": aggregate(scores, stage),
            "per_student": scores,
        })
        save(root / "progress.json", {"rows": result_rows})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                          "order": order, "arm": arm, "stage": stage,
                          "f1": result_rows[-1]["aggregate"]["mean_f1"],
                          "accuracy": result_rows[-1]["aggregate"]["mean_accuracy"]}),
              flush=True)
    save(root / "results.json", {
        "protocol": construction["protocol"],
        "stage_classes": construction["stage_classes"],
        "budgets": construction["budgets"],
        "rows": result_rows,
    })


def audit_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(data, order, args)
    maximum = 0.0
    fit_count = 0
    for record in construction["records"]:
        arm, stage = record["arm"], int(record["stage"])
        memory = torch.load(root / record["memory_file"],
                            map_location="cpu", weights_only=False)
        for seed in args.report_seeds:
            score = read(root / f"{arm}_s{stage}_seed{seed}.json")
            checkpoint_path = root / f"{arm}_s{stage}_seed{seed}.pt"
            if score["checkpoint_sha256"] != sha256_file(checkpoint_path):
                raise AssertionError("checkpoint changed")
            model = PS.GraphTextPromptStudent(int(memory["x"].shape[1])).to(
                args.device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                    weights_only=False)
            model.load_state_dict(checkpoint["state"])
            model.eval()
            current = BASE.score_model(model, data, data["stage_classes"], stage)
            for key in ("cumulative_f1", "accuracy", "old_f1", "new_f1"):
                if current.get(key) is not None and score.get(key) is not None:
                    maximum = max(maximum,
                                  abs(float(current[key]) - float(score[key])))
            fit_count += 1
            del model
    save(root / "audit.json", {
        "passed": True,
        "fit_count": fit_count,
        "maximum_metric_error_pp": maximum,
        "student": "GraphTextPromptStudent",
        "scope": "memory hash, sealed prefix, checkpoint reload, prompt-only logits",
    })


def write_summary(output_root: Path, datasets: list[str], orders: list[str],
                  arms: list[str]) -> None:
    rows = []
    for dataset in datasets:
        for order in orders:
            result = read(output_root / dataset / order / "results.json")
            for row in result["rows"]:
                if int(row["stage"]) == len(result["budgets"]) - 1:
                    agg = row["aggregate"]
                    rows.append({"dataset": dataset, "order": order,
                                 "arm": row["arm"], **agg})
    table = []
    for arm in arms:
        subset = [row for row in rows if row["arm"] == arm]
        if not subset:
            continue
        table.append({
            "arm": arm,
            "final_f1": float(np.mean([r["mean_f1"] for r in subset])),
            "final_accuracy": float(np.mean([r["mean_accuracy"] for r in subset])),
            "old_f1": float(np.mean([r["old_f1"] for r in subset
                                      if r["old_f1"] is not None])),
            "new_f1": float(np.mean([r["new_f1"] for r in subset])),
            "stage_avg_f1": float(np.mean([r["stage_avg_f1"] for r in subset])),
            "stage_avg_accuracy": float(np.mean(
                [r["stage_avg_accuracy"] for r in subset])),
            "cells": len(subset),
        })
    save(output_root / "summary.json", {"rows": rows, "overall": table,
                                          "student": PS.model_config()})
    lines = [
        "# Prompt zero-shot Student comparison", "",
        "All arms use the same current-delta-only stream, memory budgets, "
        "edge-free Student topology, and GraphTextPromptStudent.", "",
        "| Arm | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy | Cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['arm']} | {row['final_f1']:.2f} | "
            f"{row['final_accuracy']:.2f} | {row['old_f1']:.2f} | "
            f"{row['new_f1']:.2f} | {row['stage_avg_f1']:.2f} | "
            f"{row['stage_avg_accuracy']:.2f} | {row['cells']} |")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+",
                        default=["order19611", "order19612", "order19613"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    parser.add_argument("--construction-seed", type=int, default=270927)
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--partition-steps", type=int, default=30)
    parser.add_argument("--x-refine-steps", type=int, default=20)
    parser.add_argument("--x-refine-trust-radius", type=float, default=0.05)
    parser.add_argument("--x-refine-anchor", type=float, default=0.50)
    parser.add_argument("--routing-max-slots", type=int, default=3)
    parser.add_argument("--routing-steps", type=int, default=30)
    parser.add_argument("--routing-temperature", type=float, default=0.15)
    parser.add_argument("--routing-load-weight", type=float, default=0.03)
    parser.add_argument("--routing-anchor-weight", type=float, default=0.10)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--student-batch", type=int, default=24)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate", "audit"),
                        default="all")
    args = parser.parse_args()
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    if args.output_root.exists() and (args.output_root / "protocol.json").exists():
        old = read(args.output_root / "protocol.json")
        if old.get("arms") != args.arms or old.get("student") != PS.model_config():
            raise ValueError("existing output root has a different protocol")
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = {
        "name": "prompt-zero-shot-small-20261006",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": args.arms,
        "report_seeds": args.report_seeds,
        "construction_seed": int(args.construction_seed),
        "student": PS.model_config(),
        "student_batch": int(args.student_batch),
        "student_lr": float(args.student_lr),
        "student_steps": "max(50,min(2048,16*memory_rows))",
        "x_refine_steps": int(args.x_refine_steps),
        "x_refine_trust_radius": float(args.x_refine_trust_radius),
        "x_refine_anchor": float(args.x_refine_anchor),
        "routing_max_slots": int(args.routing_max_slots),
        "routing_steps": int(args.routing_steps),
        "routing_temperature": float(args.routing_temperature),
        "routing_load_weight": float(args.routing_load_weight),
        "routing_anchor_weight": float(args.routing_anchor_weight),
    }
    if not (args.output_root / "protocol.json").exists():
        save(args.output_root / "protocol.json", manifest)
    data = {}
    for name in args.datasets:
        value = BASE.load_data(name, args.minilm, args.device, args.cache_root)
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
        value["real_adj"] = BASE.G.norm_adj(value["edge_index"],
                                             len(value["x"]), args.device)
        value["x_device"] = value["x"].float().to(args.device)
        data[name] = value
    phases = ("prepare", "evaluate", "audit") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args)
                elif phase == "evaluate":
                    evaluate_cell(data[name], order, args)
                else:
                    audit_cell(data[name], order, args)
    if args.phase == "all":
        write_summary(args.output_root, args.datasets, args.orders, args.arms)
    print("PROMPT_ZERO_SHOT_SMALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
