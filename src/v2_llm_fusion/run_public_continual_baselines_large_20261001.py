"""Run public continual/coreset baselines under the current large TAG protocol.

This runner is intentionally separate from the frozen X+P scripts.  It keeps
the current edge-aware InfoNCE Student and base+4 budget, while allowing each
baseline to materialize its own legitimate memory topology:

* CaT-MMD-TAG and GECC-hard-TAG keep edge-free memories;
* real-node baselines use deterministic text-kNN intra-block edges and one
  text-nearest cross edge per new row;
* append-edge and X+P load sealed memories produced by their own runners.

CaT/GECC are mechanism adaptations, not official end-to-end reproductions.
The constructor receives only the current delta and the parent synthetic
memory.  Report/test labels are never used for construction or selection.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import cat_multimodal_minimal as MM
import external_continual_mechanisms as M
import x_conditioned_payload_routing as XCR
import run_student_compatible_append as SCA
import run_unified_v3_continual as U
import x_conditioned_payload_partition as XPP


CAT_ARMS = (
    "cat_mmd_tag",
    "cat_mmd_tag_tagsam",
    "cat_mmd_tag_graph_text",
    "cat_mmd_tag_graph_text_aligned",
    "cat_mmd_tag_text_center",
    "cat_mmd_tag_cctp_top3",
    "cat_mmd_tag_cctp_top5",
)
CAT_TEXT_MODES = {
    "cat_mmd_tag": "nearest",
    "cat_mmd_tag_tagsam": "tagsam_style",
    "cat_mmd_tag_graph_text": "graph_conditioned",
    "cat_mmd_tag_graph_text_aligned": "graph_text_aligned",
    "cat_mmd_tag_text_center": "text_center",
    # CCTP top-k means k graph-compatible source texts per synthetic row.
    # Mapping it to ``text_center`` silently made every row in a class share
    # one payload, creating false negatives for the Student InfoNCE loss.
    "cat_mmd_tag_cctp_top3": "nearest",
    "cat_mmd_tag_cctp_top5": "nearest",
}
CAT_TEXT_SOURCES = {
    "cat_mmd_tag_cctp_top3": 3,
    "cat_mmd_tag_cctp_top5": 5,
}
CLASS_SHARED_ARMS = (
    "cat_mmd_tag_class_shared_tagsam",
    "cat_mmd_tag_class_assigned",
    "cat_mmd_tag_class_assigned_two_anchor",
    "cat_mmd_tag_class_assigned_two_payload",
    "cat_mmd_tag_class_assigned_prompt_margin",
)
X_PARTITION_ARMS = (
    "cat_mmd_tag_x_slot_partition",
    "cat_mmd_tag_x_slot_partition_st",
    "cat_mmd_tag_x_slot_partition_anchor",
    "cat_mmd_tag_x_slot_partition_full",
)
ROUTING_ARMS = (
    "cat_mmd_tag_x_conditioned_routing",
    "cat_mmd_tag_x_conditioned_routing_coherent",
    "cat_mmd_tag_x_conditioned_routing_coherent_balanced",
    "cat_mmd_tag_x_conditioned_routing_coherent_anchor",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor",
)
EDGE_FREE_ARMS = (CAT_ARMS + CLASS_SHARED_ARMS + X_PARTITION_ARMS
                  + ROUTING_ARMS + ("gecc_hard_tag",))
BASELINE_ARMS = (
    "append_edge",
    "x_plus_p",
    "random_real",
    "k_center",
    "medoid",
    *CAT_ARMS,
    "gecc_hard_tag",
    "cat_mmd_tag_cctp_top3",
    "cat_mmd_tag_cctp_top5",
    *X_PARTITION_ARMS,
    *ROUTING_ARMS,
    *CLASS_SHARED_ARMS,
)
REAL_METHODS = {
    "random_real": "random_real",
    "k_center": "k_center",
    "medoid": "kmeans_medoid",
}


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    return str(tuple(value.shape)).encode() + str(value.dtype).encode() + value.numpy().tobytes()


def memory_sha256(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        digest.update(key.encode())
        digest.update(tensor_digest(memory[key]))
    for text in memory["texts"]:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    return value


def clone_memory(memory: dict) -> dict:
    return {key: clone_value(value) for key, value in memory.items()}


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total_nodes: int) -> torch.Tensor:
    ids = ids.long().cpu()
    remap = torch.full((int(total_nodes),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edges = edge_index.long().cpu()
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    valid = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
    return torch.stack((remap[edges[0, valid]], remap[edges[1, valid]]))


def set_pool_text_embeddings(data: dict, source: Path) -> dict:
    cache = source / "text_emb_pool_only.pt"
    if not cache.exists():
        raise FileNotFoundError(cache)
    text_emb = load(cache).float()
    if tuple(text_emb.shape) != (len(data["x"]), 384):
        raise ValueError(f"unexpected text cache shape: {tuple(text_emb.shape)}")
    excluded = ~data["pool_mask"]
    if bool(excluded.any()) and float(text_emb[excluded].abs().max()) != 0.0:
        raise AssertionError("pool text cache contains held-out rows")
    data["text_emb"] = text_emb
    return data


def stage_groups(manifest: dict) -> list[list[int]]:
    groups = manifest.get("stage_classes", manifest.get("groups"))
    if not groups:
        raise ValueError("source manifest has no stage class groups")
    return [[int(v) for v in group] for group in groups]


def permute_stage_groups(groups: list[list[int]], data: dict,
                         seed: int) -> list[list[int]]:
    """Create a reproducible class order with the registered group sizes.

    The canonical protocol fixes the number of classes in the base stage and
    each incremental stage.  For an independent-order check we preserve
    those sizes and only permute the class identities, so memory-rate and
    stage-count settings remain comparable.
    """
    pool_mask = data.get("pool_mask")
    labels = data["y"].detach().cpu().long()
    if pool_mask is not None:
        pool_mask = pool_mask.detach().cpu().bool()
        labels = labels[pool_mask]
    classes = sorted(int(v) for v in torch.unique(labels).tolist())
    expected = sum(len(group) for group in groups)
    if len(classes) != expected:
        raise AssertionError(
            f"order classes={len(classes)} but canonical groups={expected}")
    rng = np.random.default_rng(int(seed))
    permuted = [int(v) for v in rng.permutation(classes).tolist()]
    out, cursor = [], 0
    for group in groups:
        size = len(group)
        out.append(permuted[cursor:cursor + size])
        cursor += size
    return out


def configure_eval(data: dict, groups: list[list[int]], device: str) -> None:
    cumulative, seen = [], []
    for group in groups:
        seen.extend(group)
        cumulative.append(list(seen))
    data["stage_classes"] = groups
    U.G.STAGE_CLASSES = cumulative
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = U.G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)


@torch.no_grad()
def eval_stagewise_exact(model, data: dict, groups: list[list[int]],
                         stage: int) -> dict:
    """Evaluate one continual checkpoint without duplicating seen nodes.

    ``run_gate1_continual.eval_stagewise`` expects its global
    ``STAGE_CLASSES`` value to contain *incremental* groups.  The large
    baseline runner stores cumulative groups for compatibility with the old
    protocol, so calling that scorer here would append the same old nodes to
    several stage slices.  This scorer keeps the two notions separate:

    * cumulative metrics use every report-dev node from each seen class once;
    * per-stage metrics use only the newly arrived class group for that stage;
    * old/new F1 are means of per-class F1 over the corresponding class sets.
    """
    if stage < 0 or stage >= len(groups):
        raise ValueError(f"invalid stage {stage} for {len(groups)} groups")

    logits = model.logits(
        data["x_device"], data["real_adj"],
        data["prompt_emb"].to(data["x_device"].device))
    pred = logits.argmax(1).detach().cpu()
    y = data["y"].detach().cpu().long()
    report_mask = data["report_dev_mask"].detach().cpu().bool()

    def metric(mask: torch.Tensor, classes: list[int]) -> tuple[float, float, dict[int, float], int]:
        yy = y[mask].numpy()
        pp = pred[mask].numpy()
        labels = [int(v) for v in classes]
        if len(yy) == 0:
            return 0.0, 0.0, {int(v): 0.0 for v in labels}, 0
        per_class = f1_score(yy, pp, average=None, labels=labels,
                             zero_division=0)
        per_class_map = {int(cls): float(value)
                         for cls, value in zip(labels, per_class)}
        macro = float(f1_score(yy, pp, average="macro", labels=labels,
                               zero_division=0))
        accuracy = float(accuracy_score(yy, pp))
        return macro, accuracy, per_class_map, int(len(yy))

    seen_groups = groups[:stage + 1]
    per_stage = []
    for group in seen_groups:
        mask = report_mask & torch.isin(
            y, torch.tensor(group, dtype=torch.long))
        macro, accuracy, _, count = metric(mask, group)
        per_stage.append({
            "macro_f1": 100.0 * macro,
            "accuracy": 100.0 * accuracy,
            "classes": [int(v) for v in group],
            "n": count,
        })

    seen_classes = [int(c) for group in seen_groups for c in group]
    seen_mask = report_mask & torch.isin(
        y, torch.tensor(seen_classes, dtype=torch.long))
    cumulative_f1, cumulative_accuracy, per_class, count = metric(
        seen_mask, seen_classes)

    old_classes = [int(c) for group in groups[:stage] for c in group]
    new_classes = [int(c) for c in groups[stage]]
    old_f1 = (float(100.0 * np.mean([per_class.get(c, 0.0)
                                     for c in old_classes]))
              if old_classes else None)
    new_f1 = float(100.0 * np.mean([per_class.get(c, 0.0)
                                    for c in new_classes]))

    return {
        "per_stage": per_stage,
        "cumulative_macro_f1": 100.0 * cumulative_f1,
        "cumulative_accuracy": 100.0 * cumulative_accuracy,
        "old_mean_f1": old_f1,
        "current_f1": new_f1,
        "stage_avg_f1": float(np.mean([v["macro_f1"] for v in per_stage])),
        "stage_avg_accuracy": float(np.mean([v["accuracy"] for v in per_stage])),
        "seen_classes": seen_classes,
        "report_nodes": count,
    }


def stage_ids(data: dict, groups: list[list[int]]) -> list[torch.Tensor]:
    ids = []
    for group in groups:
        classes = torch.tensor(group, dtype=torch.long)
        current = torch.where(data["pool_mask"] & torch.isin(data["y"], classes))[0]
        ids.append(torch.sort(current).values)
    return ids


def delta_view(data: dict, ids: torch.Tensor, stage: int) -> dict:
    ids = ids.long().cpu()
    if len(ids.unique()) != len(ids) or not bool(data["pool_mask"][ids].all()):
        raise AssertionError("delta is not a unique pool-only source")
    prompt_emb = data.get("prompt_emb")
    if prompt_emb is None and data.get("prompts") is not None:
        prompt_emb = data["encoder"].encode(data["prompts"])
    return {
        "ids": ids,
        "stage": int(stage),
        "x": data["x"][ids].float(),
        "y": data["y"][ids].long(),
        "text_emb": data["text_emb"][ids].float(),
        "texts": [data["texts"][int(i)] for i in ids],
        "prompt_emb": prompt_emb,
        "edge_index": induced_edges(data["edge_index"], ids, len(data["x"])),
    }


def add_real_topology(block: dict) -> dict:
    block = clone_memory(block)
    block["edge_index"] = SCA.text_knn_edges(block["text_emb"], degree=1.0)
    return block


def append_real(parent: dict | None, block: dict) -> dict:
    if parent is None:
        return block
    return {
        "x": torch.cat((parent["x"], block["x"])),
        "text_emb": torch.cat((parent["text_emb"], block["text_emb"])),
        "texts": list(parent["texts"]) + list(block["texts"]),
        "mass": torch.cat((parent["mass"], block["mass"])),
        "depth": torch.cat((parent["depth"], block["depth"])),
        "stage": torch.cat((parent["stage"], block["stage"])),
        "edge_index": SCA.append_topology(parent, block, cross_degree=1),
    }


def build_real(data: dict, ids: torch.Tensor, rows: int, method: str,
               stage: int, device: str) -> tuple[dict, dict]:
    block, audit = U.build_real_selection_block(
        data, ids, rows, method, stage, device)
    selected = audit.get("selected_rows", 0)
    block = add_real_topology(block)
    audit.update({
        "topology": "text-kNN within block plus one text-nearest cross edge",
        "candidate_delta_ids": [int(v) for v in ids.tolist()],
        "selected_rows": int(selected),
        "historical_raw_reads": 0,
    })
    return block, audit


def build_external(data: dict, ids: torch.Tensor, rows: int, arm: str,
                   stage: int, device: str, cfg: M.Config,
                   args: argparse.Namespace) -> tuple[dict, dict]:
    delta = delta_view(data, ids, stage)
    if arm in ROUTING_ARMS:
        source_features = M.propagated_features(delta, (0.0, 1.0), device)
        labels = M.row_labels(M.ratio_allocation(delta["y"], rows))
        cat_cfg = replace(cfg, text_mode="nearest", text_sources=5)
        cctp_block, cctp_audit = M.cat_block(
            delta, rows, data["encoder"], cat_cfg, device)
        mm_cfg = MM.MMConfig(
            text_sources=5, text_chars=320, support_pool_size=32,
            chunk_chars=120, payload_max_chunks=8, payload_max_sources=5,
            payload_graph_weight=0.35, payload_text_weight=0.45,
            payload_prompt_weight=0.20, payload_redundancy=0.20,
            payload_require_lead=True, payload_contrastive_weight=0.35,
            payload_source_repeat_penalty=0.10,
            seed=int(cfg.seed) + 76)
        block, audit = XCR.attach_x_conditioned_multi_payload(
            delta=delta, source_features=source_features,
            x0=cctp_block["x"], text0=cctp_block["text_emb"],
            base_texts=cctp_block["texts"],
            base_source_ids=cctp_audit["text_source_ids"], labels=labels,
            encoder=data["encoder"], cfg=mm_cfg,
            mass=cctp_block["mass"], device=device,
            max_slots=int(args.routing_max_slots),
            routing_steps=int(args.routing_steps),
            routing_temperature=float(args.routing_temperature),
            load_weight=float(args.routing_load_weight),
            anchor_weight=float(args.routing_anchor_weight),
            candidate_mode=(
                "coherent_raw_medoid"
                if arm in ("cat_mmd_tag_x_conditioned_routing_coherent",
                           "cat_mmd_tag_x_conditioned_routing_coherent_balanced",
                           "cat_mmd_tag_x_conditioned_routing_coherent_adaptive")
                else "coherent_raw_medoid_anchor"
                if arm in ("cat_mmd_tag_x_conditioned_routing_coherent_anchor",
                           "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor")
                else "standard"),
            balanced_materialization=(
                arm == "cat_mmd_tag_x_conditioned_routing_coherent_balanced"),
            cctp_anchor_swap=(
                arm in ("cat_mmd_tag_x_conditioned_routing_coherent_anchor",
                        "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor")),
            adaptive_slots=(
                arm in ("cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
                        "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor")))
        audit.update({
            "large_runner_arm": arm,
            "base_constructor": "CaT class-wise MMD X + CCTP top-5",
            "base_cat_audit": cctp_audit,
            "historical_raw_reads": 0, "current_delta_only": True,
            "old_memory_updated": False, "topology_updated": False,
            "construction_seed": int(cfg.seed),
            "payload_seed": int(cfg.seed) + 76,
            "large_graph_migration": (
                "x-conditioned-coherent-adaptive-slot-routing-v1"
                if arm in ("cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
                           "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor")
                else "x-conditioned-coherent-multi-payload-routing-v1"
                if arm in ("cat_mmd_tag_x_conditioned_routing_coherent",
                           "cat_mmd_tag_x_conditioned_routing_coherent_balanced")
                else "x-conditioned-coherent-cctp-anchor-routing-v1"
                if arm == "cat_mmd_tag_x_conditioned_routing_coherent_anchor"
                else "x-conditioned-multi-payload-routing-v1"),
        })
        return block, audit
    if arm in X_PARTITION_ARMS:
        # CaT supplies the X slots.  The new text module only sees the current
        # delta and uses balanced graph/text transport to build one payload
        # per slot; it does not broadcast a class-level prototype.
        source_features = M.propagated_features(delta, (0.0, 1.0), device)
        labels = M.row_labels(M.ratio_allocation(delta["y"], rows))
        cat_cfg = replace(cfg, text_mode="nearest", text_sources=5)
        cctp_block, cctp_audit = M.cat_block(
            delta, rows, data["encoder"], cat_cfg, device)
        variant = {
            "cat_mmd_tag_x_slot_partition": "current",
            "cat_mmd_tag_x_slot_partition_st": "st",
            "cat_mmd_tag_x_slot_partition_anchor": "anchor",
            "cat_mmd_tag_x_slot_partition_full": "full",
        }[arm]
        partition_cfg = XPP.PartitionConfig(
            steps=30,
            hidden=256,
            lr_x=5e-4,
            text_chars=320,
            max_sources=6,
            straight_through_sinkhorn=variant in ("st", "anchor", "full"),
            lambda_coherence=0.25 if variant in ("anchor", "full") else 0.0,
            lambda_hard=0.25 if variant == "full" else 0.0,
            hard_refresh_steps=10 if variant == "full" else 0,
            seed=int(cfg.seed) + 152,
        )
        block, audit = XPP.build_x_conditioned_slot_partition(
            delta=delta,
            source_features=source_features,
            x0=cctp_block["x"],
            labels=labels,
            encoder=data["encoder"],
            cfg=partition_cfg,
            mass=cctp_block["mass"],
            device=device,
        )
        audit.update({
            "large_runner_arm": arm,
            "partition_variant": variant,
            "base_constructor": "CaT class-wise MMD X + X-conditioned balanced partition",
            "base_cat_audit": cctp_audit,
            "historical_raw_reads": 0,
            "current_delta_only": True,
            "old_memory_updated": False,
            "topology_updated": False,
            "budget_updated": False,
            "x_updated": True,
            "construction_seed": int(cfg.seed),
            "partition_seed": int(cfg.seed) + 152,
            "large_graph_migration": "x-conditioned-slot-partition-v1",
        })
        return block, audit
    if arm in CLASS_SHARED_ARMS:
        # Keep the exact large-graph CCTP/CaT X construction as the anchor,
        # then replace only the materialized text payload with the class-shared
        # real-chunk strategy already audited on Cora/Citeseer.  The source
        # delta is the only raw text pool consulted at this stage.
        source_features = M.propagated_features(delta, (0.0, 1.0), device)
        labels = M.row_labels(M.ratio_allocation(delta["y"], rows))
        # Formal assignment needs a row-specific CCTP anchor.  Keep
        # ``text_center`` in the bounded candidate bank below, but do not use
        # it as the default payload: broadcasting one class-center payload
        # here creates the same false-negative risk as the old CCTP adapter.
        base_text_mode = ("text_center"
                          if arm == "cat_mmd_tag_class_shared_tagsam"
                          else "nearest")
        cat_cfg = replace(cfg, text_mode=base_text_mode, text_sources=5)
        cctp_block, cctp_audit = M.cat_block(
            delta, rows, data["encoder"], cat_cfg, device)
        mm_cfg = MM.MMConfig(
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
            # Preserve the previous stream when cfg.seed is the registered
            # default: 270927 + 76 == 271003.  Making this derived from the
            # explicit construction seed lets independent runs change both
            # the CaT X initialization and the payload materialization while
            # keeping the protocol auditable.
            seed=int(cfg.seed) + 76,
        )
        if arm == "cat_mmd_tag_class_assigned":
            block, audit = MM.attach_class_assigned_chunk_payload(
                delta,
                source_features,
                cctp_block["x"],
                cctp_block["text_emb"],
                cctp_block["texts"],
                cctp_audit["text_source_ids"],
                labels,
                data["encoder"],
                mm_cfg,
                cctp_block["mass"],
                device,
            )
        elif arm == "cat_mmd_tag_class_assigned_two_anchor":
            block, audit = MM.attach_class_assigned_chunk_payload(
                delta,
                source_features,
                cctp_block["x"],
                cctp_block["text_emb"],
                cctp_block["texts"],
                cctp_audit["text_source_ids"],
                labels,
                data["encoder"],
                mm_cfg,
                cctp_block["mass"],
                device,
                candidate_anchor_count=2,
            )
        elif arm == "cat_mmd_tag_class_assigned_two_payload":
            block, audit = MM.attach_class_assigned_chunk_payload(
                delta,
                source_features,
                cctp_block["x"],
                cctp_block["text_emb"],
                cctp_block["texts"],
                cctp_audit["text_source_ids"],
                labels,
                data["encoder"],
                mm_cfg,
                cctp_block["mass"],
                device,
                candidate_anchor_count=2,
                max_payloads=2,
            )
        elif arm == "cat_mmd_tag_class_assigned_prompt_margin":
            block, audit = MM.attach_class_assigned_chunk_payload(
                delta,
                source_features,
                cctp_block["x"],
                cctp_block["text_emb"],
                cctp_block["texts"],
                cctp_audit["text_source_ids"],
                labels,
                data["encoder"],
                mm_cfg,
                cctp_block["mass"],
                device,
                candidate_anchor_count=1,
                max_payloads=1,
                use_prompt_margin=True,
            )
        else:
            block, audit = MM.attach_class_shared_chunk_payload(
                delta,
                source_features,
                cctp_block["x"],
                cctp_block["text_emb"],
                cctp_block["texts"],
                cctp_audit["text_source_ids"],
                labels,
                data["encoder"],
                mm_cfg,
                cctp_block["mass"],
                device,
            )
        audit.update({
            "large_runner_arm": arm,
            "base_constructor": "CaT class-wise MMD X + CCTP top-5",
            "base_cat_audit": cctp_audit,
            "historical_raw_reads": 0,
            "current_delta_only": True,
            "old_memory_updated": False,
            "topology_updated": False,
            "x_updated": False,
            "text_sources": 5,
            "text_chars": 320,
            "construction_seed": int(cfg.seed),
            "payload_seed": int(cfg.seed) + 76,
            "large_graph_migration": "small-formal-class-shared-payload-v1",
        })
        return block, audit
    if arm in CAT_ARMS:
        builder = M.cat_block
        arm_cfg = replace(cfg, text_mode=CAT_TEXT_MODES[arm],
                          text_sources=CAT_TEXT_SOURCES.get(arm, cfg.text_sources))
    else:
        builder = M.gecc_block
        arm_cfg = cfg
    block, audit = builder(delta, rows, data["encoder"], arm_cfg, device)
    audit["historical_raw_reads"] = 0
    audit["source_ids"] = [int(v) for v in ids.tolist()]
    audit["topology"] = "edge-free self-loop-only Student graph"
    return block, audit


def check_memory(memory: dict, rows: int, parent: dict | None,
                 allow_edges: bool = True) -> None:
    if len(memory["x"]) != int(rows) or len(memory["texts"]) != int(rows):
        raise AssertionError("memory rows/text count mismatch")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if len(memory[key]) != rows or not bool(torch.isfinite(memory[key]).all()):
            raise AssertionError(f"invalid memory field {key}")
    edge = memory["edge_index"].long()
    if edge.shape[0] != 2 or (edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= rows)):
        raise AssertionError("invalid topology endpoint")
    if not allow_edges and edge.numel():
        raise AssertionError("edge-free mechanism contains non-empty edges")
    if parent is not None:
        old = len(parent["x"])
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            if not torch.equal(memory[key][:old], parent[key]):
                raise AssertionError(f"parent prefix changed: {key}")
        if memory["texts"][:old] != parent["texts"]:
            raise AssertionError("parent text prefix changed")
        parent_edge = parent["edge_index"].long()
        prefix_edge = edge[:, (edge < old).all(0)]

        # Sealed memories and append helpers may emit the same old-old edge
        # set in a different column order.  Compare the edge set, not its
        # serialization order; endpoint identities and multiplicity remain
        # exact after canonicalization.
        def canonical_edges(value: torch.Tensor, n: int) -> torch.Tensor:
            if value.numel() == 0:
                return value.reshape(2, 0)
            keys = value[0].long() * int(n) + value[1].long()
            order = torch.argsort(keys, stable=True)
            return value[:, order]

        if not torch.equal(canonical_edges(prefix_edge, old),
                           canonical_edges(parent_edge, old)):
            raise AssertionError("parent old-old edges changed")


def student_score(memory: dict, stage: int, data: dict, seeds: list[int],
                  device: str, batch: int, steps_override: int | None = None) -> dict:
    syn = {
        "x": memory["x"],
        "text_emb": memory["text_emb"],
        "texts": memory["texts"],
        "edge_index": memory["edge_index"],
        "weights": memory.get("mass", torch.ones(len(memory["x"]))),
    }
    steps = U.protocol_v3.student_steps(len(memory["x"]), override=steps_override)
    scores = []
    started = time.perf_counter()
    for seed in seeds:
        model = U.G.train_student(memory, int(seed), device, steps,
                                   lr=0.001, batch=batch)
        scores.append(eval_stagewise_exact(model, data, data["stage_classes"],
                                           stage))
        del model

    def sample_std(values: list[float]) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        "student_steps": int(steps),
        "student_batch": int(batch),
        "student_seeds": [int(v) for v in seeds],
        "student_seconds": float(time.perf_counter() - started),
        "per_student": scores,
        "final_f1": float(np.mean([v["cumulative_macro_f1"] for v in scores])),
        "final_f1_std": sample_std([v["cumulative_macro_f1"] for v in scores]),
        "accuracy": float(np.mean([v.get("accuracy", v.get("cumulative_accuracy", 0.0)) for v in scores])),
        "accuracy_std": sample_std([v.get("accuracy", v.get("cumulative_accuracy", 0.0)) for v in scores]),
        "old_f1": (float(np.mean([v["old_mean_f1"] for v in scores])) if stage > 0 else None),
        "new_f1": float(np.mean([v["current_f1"] for v in scores])),
        "stage_f1": [float(np.mean([v["per_stage"][s]["macro_f1"] for v in scores])) for s in range(stage + 1)],
        "stage_accuracy": [float(np.mean([v["per_stage"][s]["accuracy"] for v in scores])) for s in range(stage + 1)],
        "stage_avg_f1": float(np.mean([v["stage_avg_f1"] for v in scores])),
        "stage_avg_accuracy": float(np.mean([v["stage_avg_accuracy"] for v in scores])),
    }


def source_memory(root: Path, dataset: str, stage: int, arm: str) -> tuple[dict, Path]:
    if arm == "append_edge":
        path = root / f"{dataset}_base4" / f"append_fixed_s{stage}_memory.pt"
        return load(path), path
    if arm == "x_plus_p":
        path = root / f"{dataset}_base4" / f"xpa_s{stage}_memory.pt"
        return load(path), path
    raise ValueError(arm)


def source_root_manifest(root: Path, dataset: str,
                         fallback_root: Path | None = None) -> dict:
    """Read the registered protocol, tolerating legacy result layouts.

    The sealed append evaluator stores its metrics in ``results.json`` and
    its protocol in a sibling ``protocol.json``.  The X+P runner stores both
    the groups and budgets in its protocol, so it is a valid same-protocol
    fallback when the append artifact predates this baseline runner.
    """
    candidates = [root / f"{dataset}_base4" / "protocol.json",
                  root / f"{dataset}_base4" / "results.json"]
    if fallback_root is not None:
        candidates.extend([
            fallback_root / f"{dataset}_base4" / "protocol.json",
            fallback_root / f"{dataset}_base4" / "results.json",
        ])
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol")
        candidate = protocol if isinstance(protocol, dict) else payload
        if candidate.get("groups") or candidate.get("stage_classes"):
            return candidate
    raise ValueError(f"no protocol with stage groups found for {dataset}")


def run_dataset(dataset: str, args: argparse.Namespace, cfg: M.Config) -> dict:
    source = args.source_root / f"{dataset}_base4"
    manifest = source_root_manifest(args.source_root, dataset, args.xpa_root)
    groups = stage_groups(manifest)
    budgets = [int(v) for v in manifest.get("budgets", [])]
    if len(budgets) != len(groups):
        raise AssertionError("budget/group length mismatch")
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], args.minilm,
                          args.device, n_stages=5, split_mode="base_incremental",
                          base_fraction=0.4,
                          report_dev_mod=int(manifest.get("report_dev_mod", 3)))
    if args.order_seed is not None:
        groups = permute_stage_groups(groups, data, int(args.order_seed))
    if args.text_cache_root is not None:
        cache_source = args.text_cache_root / dataset
        if not (cache_source / "text_emb_pool_only.pt").exists():
            cache_source = args.text_cache_root / f"{dataset}_base4"
    else:
        cache_source = source
    data = set_pool_text_embeddings(data, cache_source)
    configure_eval(data, groups, args.device)
    ids = stage_ids(data, groups)
    expected_counts = manifest.get("stage_train_counts")
    if (expected_counts is not None and args.order_seed is None
            and [len(v) for v in ids] != [int(v) for v in expected_counts]):
        raise AssertionError("current loader and frozen source stage counts differ")

    out = args.output_root / f"{dataset}_base4"
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    parent_by_arm: dict[str, dict] = {}
    records_by_arm: dict[str, list[dict]] = {arm: [] for arm in args.arms}
    for stage, (delta_ids, rows_budget) in enumerate(zip(ids, budgets)):
        for arm in args.arms:
            started = time.perf_counter()
            construction_audit = {"arm": arm, "stage": stage,
                                  "historical_raw_reads": 0,
                                  "report_test_labels_used_for_construction": False}
            if arm in ("append_edge", "x_plus_p"):
                memory, source_path = source_memory(
                    args.source_root if arm == "append_edge" else args.xpa_root,
                    dataset, stage, arm)
                construction_audit.update({"status": "sealed_source",
                                           "source_memory": str(source_path),
                                           "constructed": False})
            else:
                old = parent_by_arm.get(arm)
                new_rows = int(rows_budget) - (len(old["x"]) if old is not None else 0)
                if new_rows <= 0 or new_rows > len(delta_ids):
                    raise AssertionError(f"invalid new rows for {arm}: {new_rows}")
                if arm in REAL_METHODS:
                    block, audit = build_real(data, delta_ids, new_rows,
                                              REAL_METHODS[arm], stage, args.device)
                    memory = append_real(old, block)
                    construction_audit.update(audit)
                else:
                    block, audit = build_external(
                        data, delta_ids, new_rows, arm, stage, args.device, cfg,
                        args)
                    memory = M.append_edge_free(old, block)
                    construction_audit.update(audit)
                construction_audit["constructed"] = True
            allow_edges = arm not in EDGE_FREE_ARMS
            check_memory(memory, rows_budget, parent_by_arm.get(arm), allow_edges)
            memory_path = out / f"{arm}_s{stage}_memory.pt"
            # Summary-only runs keep the live memory in RAM for the next stage
            # but write only an auditable digest, avoiding large .pt artifacts.
            if args.no_memory_artifacts or arm in ("append_edge", "x_plus_p"):
                memory_path = Path(str(memory_path) + ".reference.json")
                reference = {"sha256": memory_sha256(memory)}
                if arm in ("append_edge", "x_plus_p"):
                    reference["source"] = construction_audit["source_memory"]
                save_json(memory_path, reference)
            else:
                torch.save(memory, memory_path)
            construction_seconds = time.perf_counter() - started
            print(json.dumps({"event": "student_start", "dataset": dataset,
                              "stage": stage, "arm": arm,
                              "rows": int(rows_budget),
                              "construction_seconds": construction_seconds}),
                  flush=True)
            score = student_score(memory, stage, data, args.student_seeds,
                                  args.device, args.student_batch,
                                  args.student_steps)
            record = {
                "dataset": dataset, "stage": stage, "arm": arm,
                "memory_rows": int(rows_budget),
                "memory_sha256": memory_sha256(memory),
                "memory_file": str(memory_path.name),
                "construction_seconds": float(construction_seconds),
                "construction_audit": construction_audit,
                "score": score,
            }
            records_by_arm[arm].append(record)
            rows.append(record)
            parent_by_arm[arm] = memory
            print(json.dumps({"event": "baseline_stage_complete", "dataset": dataset,
                              "stage": stage, "arm": arm,
                              "f1": score["final_f1"], "accuracy": score["accuracy"]}),
                  flush=True)
    for arm in args.arms:
        traj = records_by_arm[arm]
        final = traj[-1]["score"]
        records_by_arm[arm] = {
            "final_f1": final["final_f1"],
            "final_f1_std": final["final_f1_std"],
            "final_accuracy": final["accuracy"],
            "final_accuracy_std": final["accuracy_std"],
            "accuracy": final["accuracy"],
            "accuracy_std": final["accuracy_std"],
            "old_f1": final["old_f1"],
            "new_f1": final["new_f1"],
            # Stage-average is the mean over the independent newly-arrived
            # class slices at the final checkpoint, not a mean of cumulative
            # checkpoint scores (which would mix two different quantities).
            "stage_avg_f1": final["stage_avg_f1"],
            "stage_avg_accuracy": final["stage_avg_accuracy"],
            "trajectory_avg_f1": float(np.mean([v["score"]["final_f1"] for v in traj])),
            "trajectory_avg_accuracy": float(np.mean([v["score"]["accuracy"] for v in traj])),
            "worst_stage_f1": float(min(v["score"]["final_f1"] for v in traj)),
            "construction_seconds": float(sum(v["construction_seconds"] for v in traj)),
            "student_seconds": float(sum(v["score"]["student_seconds"] for v in traj) / len(args.student_seeds)),
        }
    result = {
        "protocol": {
            "name": "public-continuous-baselines-large-20261001",
            "dataset": dataset, "stage_protocol": "base+4",
            "groups": groups, "budgets": budgets, "arms": args.arms,
            "memory_rate": 0.05,
            "student": "same original edge-aware 2-layer GCN + text projection + bidirectional InfoNCE",
            "student_seeds": args.student_seeds, "student_batch": args.student_batch,
            "student_steps": "max(50,min(2048,16*memory_rows))",
            "historical_raw_reads": 0,
            "report_test_labels_used_for_construction": False,
            "cat_gecc_status": "mechanism-adapted, not official TAG reproductions",
            "construction_seed": int(args.construction_seed),
            "order_seed": (int(args.order_seed)
                           if args.order_seed is not None else None),
            "payload_seed_offset": 76,
            "no_memory_artifacts": bool(args.no_memory_artifacts),
        },
        "dataset": dataset, "rows": rows, "overall": records_by_arm,
    }
    save_json(out / "results.json", result)
    return result


def summarize(args: argparse.Namespace, results: list[dict]) -> dict:
    methods = args.arms
    overall = {arm: [] for arm in methods}
    for result in results:
        for arm in methods:
            overall[arm].append(result["overall"][arm])
    table = []
    for arm in methods:
        table.append({"arm": arm, **{
            key: float(np.mean([(row.get(key, row.get("accuracy"))
                                 if key == "final_accuracy" else row[key])
                                for row in overall[arm]
                                if (row.get(key, row.get("accuracy"))
                                    if key == "final_accuracy" else row[key]) is not None]))
            for key in ("final_f1", "final_accuracy", "stage_avg_f1",
                        "stage_avg_accuracy", "old_f1", "new_f1",
                        "worst_stage_f1", "construction_seconds", "student_seconds")
        }})
    reference = {row["arm"]: row for row in table}
    pairs = []
    has_append = "append_edge" in methods
    has_xp = "x_plus_p" in methods
    for arm in methods:
        if arm in ("append_edge", "x_plus_p"):
            continue
        values = []
        for result in results:
            value = {"dataset": result["dataset"]}
            if has_append:
                value.update(
                    f1_vs_append=(result["overall"][arm]["final_f1"]
                                  - result["overall"]["append_edge"]["final_f1"]),
                    accuracy_vs_append=(result["overall"][arm]["final_accuracy"]
                                        - result["overall"]["append_edge"]["final_accuracy"]))
            if has_xp:
                value["f1_vs_xp"] = (
                    result["overall"][arm]["final_f1"]
                    - result["overall"]["x_plus_p"]["final_f1"])
            values.append(value)
        pair = {"arm": arm, "cells": values}
        if has_append:
            deltas = [v["f1_vs_append"] for v in values]
            pair.update(
                mean_f1_vs_append=float(np.mean(deltas)),
                positive_vs_append=int(sum(v > 0 for v in deltas)))
        if has_xp:
            deltas = [v["f1_vs_xp"] for v in values]
            pair.update(
                mean_f1_vs_xp=float(np.mean(deltas)),
                positive_vs_xp=int(sum(v > 0 for v in deltas)))
        pairs.append(pair)
    summary = {"protocol": results[0]["protocol"], "datasets": [r["dataset"] for r in results],
               "overall": table, "paired": pairs,
               "limits": "Three large datasets, one frozen base+4 order per dataset, one construction seed; methods are mechanism-adapted unless marked sealed source."}
    save_json(args.output_root / "summary.json", summary)
    lines = ["# Large-graph public baseline comparison", "", "All scores are percentages; adapted methods are not official end-to-end TAG reproductions.", "",
             "| Method | Final F1 | Final Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Construction s | Student s/seed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("|{arm}|{final_f1:.2f}|{final_accuracy:.2f}|{stage_avg_f1:.2f}|{old_f1:.2f}|{new_f1:.2f}|{worst_stage_f1:.2f}|{construction_seconds:.1f}|{student_seconds:.1f}|".format(**row))
    pair_headers = ["Adapted method"]
    if has_append:
        pair_headers += ["Mean F1 vs append", "Positive cells vs append"]
    if has_xp:
        pair_headers += ["Mean F1 vs X+P", "Positive cells vs X+P"]
    lines += ["", "| " + " | ".join(pair_headers) + " |",
              "|" + "|".join("---" if i == 0 else "---:" for i in range(len(pair_headers))) + "|"]
    for row in pairs:
        values = [row["arm"]]
        if has_append:
            values += [f"{row['mean_f1_vs_append']:+.2f}",
                       f"{row['positive_vs_append']}/{len(row['cells'])}"]
        if has_xp:
            values += [f"{row['mean_f1_vs_xp']:+.2f}",
                       f"{row['positive_vs_xp']}/{len(row['cells'])}"]
        lines.append("|" + "|".join(values) + "|")
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["arxiv", "bookchild", "sportsfit"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--xpa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument(
        "--text-cache-root", type=Path, default=None,
        help="directory containing <dataset>/text_emb_pool_only.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="+", choices=BASELINE_ARMS,
                        default=list(BASELINE_ARMS))
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--student-steps", type=int, default=None,
                        help="smoke-only override; omit for the registered protocol")
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--gecc-restarts", type=int, default=20)
    parser.add_argument("--routing-max-slots", type=int, default=3)
    parser.add_argument("--routing-steps", type=int, default=30)
    parser.add_argument("--routing-temperature", type=float, default=0.15)
    parser.add_argument("--routing-load-weight", type=float, default=0.03)
    parser.add_argument("--routing-anchor-weight", type=float, default=0.10)
    parser.add_argument(
        "--construction-seed", type=int, default=270927,
        help="seed for CaT construction; class-shared payload uses seed+76",
    )
    parser.add_argument(
        "--order-seed", type=int, default=None,
        help="optional independent class-order seed; preserves base+4 group sizes",
    )
    parser.add_argument("--no-memory-artifacts", action="store_true",
                        help="write only per-stage memory digests instead of .pt artifacts")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = M.Config(seed=int(args.construction_seed),
                   cat_steps=args.cat_steps,
                   gecc_restarts=args.gecc_restarts)
    results = []
    for dataset in args.datasets:
        results.append(run_dataset(dataset, args, cfg))
    summary = summarize(args, results)
    print(json.dumps({"event": "complete", "output": str(args.output_root),
                      "datasets": args.datasets, "overall": summary["overall"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
