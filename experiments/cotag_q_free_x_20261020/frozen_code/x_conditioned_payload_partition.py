"""X-conditioned graph/text partitioning for continual TAG condensation.

This module keeps CaT's feature condensation as the warm start, then treats
each CaT synthetic row as a semantic slot.  Current-delta raw nodes are
softly transported to those slots with a balanced entropic transport plan.
Each slot is materialized from its own assigned raw-text cluster, so a class
does not collapse to one broadcast payload.

The text encoder remains frozen.  Gradients update the continuous X slots;
the final strings are re-encoded after deterministic cluster materialization.
No historical raw graph, report labels, or Student response is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

import external_continual_mechanisms as M


@dataclass(frozen=True)
class PartitionConfig:
    """Global construction settings; no dataset-specific payload count."""

    steps: int = 30
    hidden: int = 256
    lr_x: float = 5e-4
    graph_weight: float = 0.5
    text_weight: float = 0.5
    transport_temperature: float = 0.07
    sinkhorn_iters: int = 8
    lambda_transport: float = 0.50
    lambda_text: float = 0.20
    lambda_align: float = 0.25
    lambda_anchor: float = 0.50
    trust_radius: float = 0.05
    text_ridge: float = 1e-2
    text_chars: int = 320
    max_sources: int = 6
    medoid_pool: int = 64
    seed: int = 271079
    straight_through_sinkhorn: bool = False
    lambda_coherence: float = 0.0
    lambda_hard: float = 0.0
    hard_refresh_steps: int = 0


def _bound(dim: int, hidden: int) -> float:
    return math.sqrt(6.0 / float(dim + hidden))


def _class_mmd(raw: torch.Tensor, synth: torch.Tensor,
               raw_labels: torch.Tensor, synth_labels: torch.Tensor,
               weights: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    for weight in weights:
        losses.append(M.class_mmd(
            M.random_features(raw, weight),
            M.random_features(synth, weight),
            raw_labels, synth_labels))
    return torch.stack(losses).mean()


def _predict_text(x_value: torch.Tensor,
                  graph_text_map: torch.Tensor) -> torch.Tensor:
    x_norm = F.normalize(x_value.float(), dim=1)
    ones = torch.ones((len(x_norm), 1), device=x_value.device,
                      dtype=x_norm.dtype)
    return F.normalize(torch.cat((x_norm, ones), dim=1) @ graph_text_map, dim=1)


def _sinkhorn_plan(cost: torch.Tensor, temperature: float,
                   iterations: int) -> torch.Tensor:
    """Return a balanced plan with row/column mass fixed by cluster slots."""
    if cost.ndim != 2 or not cost.numel():
        raise ValueError("Sinkhorn cost must be a non-empty matrix")
    n_rows, n_slots = cost.shape
    tau = max(float(temperature), 1e-4)
    log_kernel = (-cost.float() / tau).clamp(min=-60.0, max=60.0)
    log_row = torch.full((n_rows,), -math.log(float(n_rows)),
                         device=cost.device, dtype=cost.dtype)
    log_col = torch.full((n_slots,), -math.log(float(n_slots)),
                         device=cost.device, dtype=cost.dtype)
    log_u = torch.zeros_like(log_row)
    log_v = torch.zeros_like(log_col)
    for _ in range(max(1, int(iterations))):
        log_u = log_row - torch.logsumexp(
            log_kernel + log_v.view(1, -1), dim=1)
        log_v = log_col - torch.logsumexp(
            log_kernel + log_u.view(-1, 1), dim=0)
    return torch.exp(log_u.view(-1, 1) + log_kernel + log_v.view(1, -1))


def _slot_cost(raw_graph: torch.Tensor, raw_text: torch.Tensor,
               x_slots: torch.Tensor, predicted_text: torch.Tensor,
               cfg: PartitionConfig) -> torch.Tensor:
    graph_sim = F.normalize(raw_graph.float(), dim=1) @ F.normalize(
        x_slots.float(), dim=1).T
    text_sim = F.normalize(raw_text.float(), dim=1) @ F.normalize(
        predicted_text.float(), dim=1).T
    return (float(cfg.graph_weight) * (1.0 - graph_sim)
            + float(cfg.text_weight) * (1.0 - text_sim))


def _plan_for_classes(raw_graph: torch.Tensor, raw_text: torch.Tensor,
                      raw_labels: torch.Tensor, x_slots: torch.Tensor,
                      slot_labels: torch.Tensor,
                      predicted_text: torch.Tensor,
                      cfg: PartitionConfig,
                      differentiable: bool = False) -> tuple[dict[int, dict],
                                                            torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
    """Build class-restricted plans and slot-level soft text prototypes."""
    soft_rows: list[torch.Tensor | None] = [None] * len(x_slots)
    anchor_rows: list[torch.Tensor | None] = [None] * len(x_slots)
    confidence_rows: list[torch.Tensor | None] = [None] * len(x_slots)
    records: dict[int, dict] = {}
    for cls in torch.unique(slot_labels, sorted=True).tolist():
        raw_idx = torch.where(raw_labels == int(cls))[0]
        slot_idx = torch.where(slot_labels == int(cls))[0]
        if not len(raw_idx) or not len(slot_idx):
            raise ValueError(f"class {cls} has no raw nodes or X slots")
        local_cost = _slot_cost(
            raw_graph.index_select(0, raw_idx),
            raw_text.index_select(0, raw_idx),
            x_slots.index_select(0, slot_idx),
            predicted_text.index_select(0, slot_idx), cfg)
        # The plan is re-solved as X moves.  The default path keeps the
        # original detached plan.  The straight-through path uses the same
        # forward values but exposes a differentiable Sinkhorn plan during
        # backward, allowing text losses to update X through the assignment.
        plan_forward = _sinkhorn_plan(
            local_cost.detach(), cfg.transport_temperature,
            cfg.sinkhorn_iters)
        if differentiable:
            plan_grad = _sinkhorn_plan(
                local_cost, cfg.transport_temperature, cfg.sinkhorn_iters)
            plan = plan_forward + (plan_grad - plan_grad.detach())
        else:
            plan = plan_forward
        local_text = raw_text.index_select(0, raw_idx)
        col_mass = plan.sum(dim=0).clamp_min(1e-12)
        soft_text = F.normalize((plan.T @ local_text) / col_mass.view(-1, 1),
                                dim=1)
        class_anchor = F.normalize(local_text.mean(dim=0, keepdim=True),
                                   dim=1).expand(len(slot_idx), -1)
        normalized_plan = plan / col_mass.view(1, -1).clamp_min(1e-12)
        entropy = -(normalized_plan.clamp_min(1e-12)
                    * normalized_plan.clamp_min(1e-12).log()).sum(dim=0)
        confidence = 1.0 - entropy / math.log(max(2, len(raw_idx)))
        confidence = confidence.clamp(0.0, 1.0)
        for local_slot, global_slot in enumerate(slot_idx.tolist()):
            soft_rows[int(global_slot)] = soft_text[local_slot]
            anchor_rows[int(global_slot)] = class_anchor[local_slot]
            confidence_rows[int(global_slot)] = confidence[local_slot]
        records[int(cls)] = {
            "raw_indices": raw_idx,
            "slot_indices": slot_idx,
            "plan": plan,
            "cost": local_cost,
            "soft_text": soft_text,
            "class_anchor": class_anchor,
            "confidence": confidence,
        }
    if any(value is None for value in soft_rows):
        raise AssertionError("every X slot must receive a text prototype")
    if any(value is None for value in anchor_rows):
        raise AssertionError("every X slot must receive a class anchor")
    if any(value is None for value in confidence_rows):
        raise AssertionError("every X slot must receive a confidence")
    return (
        records,
        torch.stack([value for value in soft_rows if value is not None]),
        torch.stack([value for value in anchor_rows if value is not None]),
        torch.stack([value for value in confidence_rows if value is not None]),
    )


def _materialize_slot(delta: dict, local_indices: torch.Tensor,
                      weights: torch.Tensor, raw_text: torch.Tensor,
                      used_texts: set[str], cfg: PartitionConfig) -> tuple[str, list[int], dict]:
    """Materialize one payload directly from one transport-defined cluster."""
    local_indices = local_indices.detach().cpu().long()
    weights = weights.detach().cpu().float()
    weights = weights / weights.sum().clamp_min(1e-12)
    source_texts = [str(delta["texts"][int(i)]) for i in local_indices.tolist()]
    normalized = [" ".join(value.split()) for value in source_texts]
    vectors = F.normalize(raw_text.detach().cpu().float(), dim=1)
    pool_size = min(len(local_indices), max(1, int(cfg.medoid_pool)))
    order = torch.argsort(weights, descending=True)[:pool_size]
    candidate_vectors = vectors.index_select(0, order)
    similarity = candidate_vectors @ vectors.index_select(0, local_indices).T
    medoid_cost = ((1.0 - similarity) * weights.view(1, -1)).sum(dim=1)
    medoid_order = torch.argsort(medoid_cost)
    anchor = int(order[int(medoid_order[0])])
    for candidate in order[medoid_order].tolist():
        key = normalized[int(candidate)]
        if key and key not in used_texts:
            anchor = int(candidate)
            break

    selected = [anchor]
    selected_keys = {normalized[anchor]}
    selected_chars = len(normalized[anchor])
    remaining = [int(value) for value in torch.argsort(weights, descending=True).tolist()]
    while len(selected) < max(1, int(cfg.max_sources)):
        best = None
        best_score = float("-inf")
        for candidate in remaining:
            if candidate in selected or not normalized[candidate]:
                continue
            if normalized[candidate] in selected_keys:
                continue
            separator = 1 if selected else 0
            if selected_chars + separator + len(normalized[candidate]) > int(cfg.text_chars):
                continue
            novelty = 1.0 - max(float(vectors[candidate] @ vectors[s])
                                for s in selected)
            score = float(weights[candidate]) * (0.5 + max(0.0, novelty))
            if score > best_score:
                best_score = score
                best = candidate
        if best is None:
            break
        selected.append(best)
        selected_keys.add(normalized[best])
        selected_chars += 1 + len(normalized[best])

    # Preserve source order inside the cluster only after the set of selected
    # sources is fixed.  This makes the final string deterministic and
    # auditable while keeping the medoid/coverage objective above.
    selected = sorted(selected)
    pieces = [normalized[index] for index in selected if normalized[index]]
    if not pieces:
        pieces = ["unknown"]
    per_piece = max(1, (int(cfg.text_chars) - max(0, len(pieces) - 1))
                    // len(pieces))
    text = " ".join(piece[:per_piece] for piece in pieces)[:int(cfg.text_chars)]
    # Track source-level anchors, rather than the final concatenated string,
    # so later slots do not silently reuse the same medoid when alternatives
    # exist.  This is a capacity constraint, not a dataset-specific rule.
    used_texts.update(normalized[index] for index in selected
                      if normalized[index])
    source_ids = [int(delta["ids"][int(local_indices[index])])
                  for index in selected]
    audit = {
        "selected_local_indices": [int(local_indices[index]) for index in selected],
        "selected_source_ids": source_ids,
        "cluster_mass": float(weights.max()),
        "effective_source_count": float(1.0 / (weights.square().sum().clamp_min(1e-12))),
        "text_length": len(text),
        "anchor_local_index": int(local_indices[anchor]),
    }
    return text, source_ids, audit


def _materialize_plan_texts(delta: dict, plans: dict[int, dict],
                            raw_text: torch.Tensor, slot_count: int,
                            cfg: PartitionConfig) -> list[str]:
    """Materialize temporary hard payloads for the soft/hard consistency loss."""
    texts = [""] * int(slot_count)
    used_by_class: dict[int, set[str]] = {}
    raw_text_cpu = raw_text.detach().cpu()
    for cls, record in plans.items():
        raw_idx = record["raw_indices"].detach().cpu()
        plan = record["plan"].detach().cpu()
        slot_idx = record["slot_indices"].detach().cpu()
        hard_slot = torch.argmax(plan, dim=1)
        used = used_by_class.setdefault(int(cls), set())
        for local_slot, global_slot in enumerate(slot_idx.tolist()):
            cluster_mask = hard_slot == int(local_slot)
            if bool(cluster_mask.any()):
                cluster_raw_idx = raw_idx[cluster_mask]
                cluster_weights = plan[cluster_mask, local_slot]
            else:
                count = min(max(1, int(cfg.max_sources)), len(raw_idx))
                top = torch.argsort(plan[:, local_slot], descending=True)[:count]
                cluster_raw_idx = raw_idx[top]
                cluster_weights = plan[top, local_slot]
            text, _, _ = _materialize_slot(
                delta, cluster_raw_idx, cluster_weights, raw_text_cpu,
                used, cfg)
            texts[int(global_slot)] = text
    return texts


def build_x_conditioned_slot_partition(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        labels: torch.Tensor, encoder, cfg: PartitionConfig,
        mass: torch.Tensor | None, device: str) -> tuple[dict, dict]:
    """Refine CaT X and materialize one cluster-conditioned text per X slot."""
    required = ("x", "y", "text_emb", "texts", "ids")
    missing = [key for key in required if key not in delta]
    if missing:
        raise ValueError(f"delta lacks required fields: {missing}")
    if len(x0) != len(labels):
        raise ValueError("X and slot labels must have the same number of rows")
    source_features = source_features.detach().to(device).float()
    x0 = x0.detach().to(device).float()
    labels = labels.detach().long().to(device)
    raw_labels = delta["y"].detach().long().to(device)
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_graph = source_features
    if len(raw_graph) != len(raw_text) or len(raw_graph) != len(raw_labels):
        raise ValueError("raw graph, text, and labels must have matching rows")

    graph_text_map, fit_cosine = M._fit_graph_text_map(
        raw_graph, raw_text, float(cfg.text_ridge))
    x_param = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([x_param], lr=float(cfg.lr_x))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    graph_bank = []
    text_bank = []
    graph_bound = _bound(x0.shape[1], cfg.hidden)
    text_bound = _bound(raw_text.shape[1], cfg.hidden)
    for _ in range(4):
        graph_bank.append(torch.empty((x0.shape[1], cfg.hidden), device=device)
                          .uniform_(-graph_bound, graph_bound, generator=generator))
        text_bank.append(torch.empty((raw_text.shape[1], cfg.hidden), device=device)
                         .uniform_(-text_bound, text_bound, generator=generator))

    def evaluate(value: torch.Tensor, need_grad: bool) -> tuple[dict[int, dict], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted = _predict_text(value, graph_text_map)
        plans, soft, anchors, confidence = _plan_for_classes(
            raw_graph, raw_text, raw_labels, value, labels, predicted, cfg,
            differentiable=need_grad and cfg.straight_through_sinkhorn)
        transport = torch.stack([
            (record["plan"] * record["cost"]).sum()
            for record in plans.values()]).mean()
        coherence = ((1.0 - confidence)
                     * (1.0 - F.cosine_similarity(soft, anchors, dim=1))).mean()
        return plans, soft, anchors, confidence, transport, coherence

    with torch.no_grad():
        initial_plans, initial_soft, initial_anchors, initial_confidence, initial_transport, initial_coherence = evaluate(x0, False)
        scale_x = _class_mmd(raw_graph, x0, raw_labels, labels,
                             graph_bank).clamp_min(1e-8)
        scale_text = _class_mmd(raw_text, initial_soft, raw_labels, labels,
                                text_bank).clamp_min(1e-8)
        scale_align = (1.0 - F.cosine_similarity(
            _predict_text(x0, graph_text_map), initial_soft, dim=1)).mean().clamp_min(1e-5)
        scale_transport = initial_transport.clamp_min(1e-5)

    trace = []
    hard_target = None
    # ``steps=0`` is a real ablation: keep the CaT anchor exactly fixed and
    # only materialize the slot-conditioned payload from that anchor.  Using
    # ``max(1, steps)`` silently turned the fixed-X control into a one-step
    # X update, which invalidated the attribution experiment.
    for step in range(max(0, int(cfg.steps))):
        plans, soft_text, anchors, confidence, transport, coherence = evaluate(x_param, True)
        loss_x = _class_mmd(raw_graph, x_param, raw_labels, labels, graph_bank)
        loss_text = _class_mmd(raw_text, soft_text, raw_labels, labels, text_bank)
        predicted_text = _predict_text(x_param, graph_text_map)
        align = (1.0 - F.cosine_similarity(
            predicted_text, soft_text, dim=1)).mean()
        anchor = (x_param - x0).square().mean()
        if (cfg.lambda_hard > 0.0 and cfg.hard_refresh_steps > 0
                and (hard_target is None or step % int(cfg.hard_refresh_steps) == 0)):
            with torch.no_grad():
                hard_texts = _materialize_plan_texts(
                    delta, plans, raw_text, len(x_param), cfg)
                hard_target = F.normalize(
                    encoder.encode(hard_texts).detach().to(device).float(), dim=1)
        if hard_target is None:
            hard_loss = x_param.sum() * 0.0
        else:
            hard_loss = (1.0 - F.cosine_similarity(
                soft_text, hard_target, dim=1)).mean()
        loss = (loss_x / scale_x
                + float(cfg.lambda_text) * loss_text / scale_text
                + float(cfg.lambda_transport) * transport / scale_transport
                + float(cfg.lambda_align) * align / scale_align
                + float(cfg.lambda_anchor) * anchor
                + float(cfg.lambda_coherence) * coherence
                + float(cfg.lambda_hard) * hard_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite X-conditioned partition loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta_x = x_param - x0
            row_norm = delta_x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            allowed = float(cfg.trust_radius) * x0.norm(
                dim=1, keepdim=True).clamp_min(1e-6)
            x_param.copy_(x0 + delta_x * (allowed / row_norm).clamp(max=1.0))
        if step == 0 or (step + 1) % 10 == 0 or step == cfg.steps - 1:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "x_mmd": float(loss_x.detach().cpu()),
                "text_mmd": float(loss_text.detach().cpu()),
                "transport": float(transport.detach().cpu()),
                "align": float(align.detach().cpu()),
                "anchor": float(anchor.detach().cpu()),
                "coherence": float(coherence.detach().cpu()),
                "hard": float(hard_loss.detach().cpu()),
            })

    with torch.no_grad():
        final_plans, final_soft, final_anchors, final_confidence, final_transport, final_coherence = evaluate(x_param, False)
    final_x = x_param.detach().cpu()
    texts: list[str] = []
    text_source_ids: list[list[int]] = []
    slot_audit: list[dict] = [None] * len(labels)
    used_by_class: dict[int, set[str]] = {}
    raw_text_cpu = raw_text.detach().cpu()
    for cls in torch.unique(labels, sorted=True).tolist():
        record = final_plans[int(cls)]
        raw_idx = record["raw_indices"].detach().cpu()
        plan = record["plan"].detach().cpu()
        slot_idx = record["slot_indices"].detach().cpu()
        hard_slot = torch.argmax(plan, dim=1)
        used = used_by_class.setdefault(int(cls), set())
        for local_slot, global_slot in enumerate(slot_idx.tolist()):
            cluster_mask = hard_slot == int(local_slot)
            if bool(cluster_mask.any()):
                cluster_raw_idx = raw_idx[cluster_mask]
                cluster_weights = plan[cluster_mask, local_slot]
            else:
                # Entropic plans can be tied when two X slots are identical.
                # Keep materialization legal and slot-conditioned by using a
                # small top-mass fallback rather than the whole class.
                count = min(max(1, int(cfg.max_sources)), len(raw_idx))
                top = torch.argsort(plan[:, local_slot], descending=True)[:count]
                cluster_raw_idx = raw_idx[top]
                cluster_weights = plan[top, local_slot]
            text, ids, material = _materialize_slot(
                delta, cluster_raw_idx, cluster_weights, raw_text_cpu,
                used, cfg)
            while len(texts) <= int(global_slot):
                texts.append("")
                text_source_ids.append([])
            texts[int(global_slot)] = text
            text_source_ids[int(global_slot)] = ids
            local_cost = record["cost"][:, local_slot].detach().cpu()
            # Sinkhorn columns carry mass 1 / number_of_slots.  Normalize
            # before reporting slot-level entropy, source weights, and
            # effective source count; otherwise those diagnostics scale with
            # the number of slots and are not comparable across classes.
            weights = plan[:, local_slot]
            normalized_weights = weights / weights.sum().clamp_min(1e-12)
            entropy = -(normalized_weights.clamp_min(1e-12)
                        * normalized_weights.clamp_min(1e-12).log()).sum()
            slot_audit[int(global_slot)] = {
                "class": int(cls),
                "slot": int(global_slot),
                "raw_candidate_count": int(len(cluster_raw_idx)),
                "top_source_ids": [int(delta["ids"][int(cluster_raw_idx[i])])
                                    for i in torch.argsort(cluster_weights, descending=True)[:8].tolist()],
                "top_source_weights": [float(normalized_weights[i])
                                        for i in torch.argsort(normalized_weights, descending=True)[:8].tolist()],
                "mean_assignment_cost": float((normalized_weights * local_cost).sum()),
                "assignment_entropy": float(entropy),
                "assignment_effective_sources": float(
                    1.0 / normalized_weights.square().sum().clamp_min(1e-12)),
                "hard_cluster_size": int(len(cluster_raw_idx)),
                "materialization": material,
            }
    hard_text = F.normalize(encoder.encode(texts).cpu().float(), dim=1)
    unique_texts = len(set(texts))
    block = {
        "x": final_x,
        "text_emb": hard_text,
        "texts": texts,
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x))),
        "depth": torch.zeros(len(final_x), dtype=torch.long),
        "stage": torch.full((len(final_x),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    relative = (final_x - x0.detach().cpu()).norm(dim=1) / x0.detach().cpu().norm(
        dim=1).clamp_min(1e-6)
    audit = {
        "constructor": "cat_x_conditioned_slot_partition",
        "objective": "CaT MMD + balanced graph/text transport + slot text alignment",
        "payload_generator": "slot_conditioned_cluster_materialization",
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "current_delta_only": True,
        "class_shared": False,
        "class_prototype_count": "one materialized payload per CaT X slot",
        "slot_count_by_class": {
            str(int(cls)): int((labels == int(cls)).sum())
            for cls in torch.unique(labels, sorted=True).tolist()},
        "unique_text_count": int(unique_texts),
        "unique_text_ratio": float(unique_texts / max(1, len(texts))),
        "graph_text_fit_cosine": float(fit_cosine),
        "x_updated": True,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "assignment": "class-restricted balanced entropic Sinkhorn; one plan column per X slot",
        "assignment_uses_x": True,
        "assignment_uses_text": True,
        "transport_gradient": (
            "straight-through Sinkhorn exposes assignment gradients to X"
            if cfg.straight_through_sinkhorn else
            "plan is recomputed from detached cost; transport/alignment losses backpropagate through X"),
        "coherence_anchor": {
            "enabled": bool(cfg.lambda_coherence > 0),
            "lambda": float(cfg.lambda_coherence),
            "confidence": "1 - normalized assignment entropy",
        },
        "hard_consistency": {
            "enabled": bool(cfg.lambda_hard > 0 and cfg.hard_refresh_steps > 0),
            "lambda": float(cfg.lambda_hard),
            "refresh_steps": int(cfg.hard_refresh_steps),
        },
        "sinkhorn": {
            "temperature": float(cfg.transport_temperature),
            "iterations": int(cfg.sinkhorn_iters),
            "row_mass": "1/n_class",
            "column_mass": "1/m_class",
        },
        "text_source_ids": text_source_ids,
        "slot_audit": slot_audit,
        "config": asdict(cfg),
        "trace": trace,
        "final_transport": float(final_transport.cpu()),
        "x_update_norm": float((final_x - x0.detach().cpu()).norm()),
        "x_update_relative_max": float(relative.max()),
        "x_update_relative_mean": float(relative.mean()),
        "soft_payload_reencoded": True,
    }
    return block, audit

