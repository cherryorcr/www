"""Teacher-guided P/X revision with a projected support message kernel.

This module is the next isolated experiment after the first Teacher/P/X
implementation.  It keeps the raw access boundary, text, and memory budget
unchanged, but makes the assignment P participate in graph propagation:

    support graph --Q(P)--> synthetic slot message kernel

    The Teacher is trained on the legal support graph.  The Candidate Student is
    unrolled on the projected support kernel, and the final kernel is materialized
    as a binary edge index with the base memory's per-row degree budget.  The
    optional straight-through path makes the Candidate's forward pass use that
    same hard topology while retaining a soft backward path for P.  ``mode``
    controls whether X, P, or both are learnable.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import time

import torch
import torch.nn.functional as F

import teacher_guided_payload_revision as T


def _binary_adjacency(edge_index: torch.Tensor, n: int, device: str,
                      dtype: torch.dtype) -> torch.Tensor:
    edge_index = edge_index.long().to(device)
    matrix = torch.zeros((n, n), device=device, dtype=dtype)
    if edge_index.numel():
        values = torch.ones(edge_index.shape[1], device=device, dtype=dtype)
        matrix.index_put_((edge_index[0], edge_index[1]), values, accumulate=True)
    return matrix


def _normalize_dense(weight: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0]
    weight = weight.clamp_min(0)
    weight = weight + torch.eye(n, device=weight.device, dtype=weight.dtype)
    degree = weight.sum(dim=1).clamp_min(1e-9)
    inv = degree.rsqrt()
    return inv[:, None] * weight * inv[None, :]


def _support_adjacency(support: dict, device: str,
                       dtype: torch.dtype) -> torch.Tensor:
    n = len(support["x"])
    return _normalize_dense(_binary_adjacency(
        support["edge_index"], n, device, dtype))


def _groups_with_provenance(class_id: torch.Tensor, old_rows: int) -> dict[int, torch.Tensor]:
    """Balance old rows by stored synthetic class provenance when available."""
    class_id = class_id.long().cpu()
    result = {}
    old_values = class_id[:int(old_rows)]
    known_old = sorted(int(v) for v in old_values.unique().tolist() if int(v) >= 0)
    if known_old:
        for value in known_old:
            rows = torch.where(old_values == value)[0]
            if len(rows):
                result[value] = rows
    else:
        old = torch.arange(min(int(old_rows), len(class_id)))
        if len(old):
            result[-1] = old
    for value in sorted(int(v) for v in class_id[int(old_rows):].unique().tolist()
                       if int(v) >= 0):
        rows = torch.where(class_id[int(old_rows):] == value)[0] + int(old_rows)
        if len(rows):
            if value in result:
                result[value] = torch.cat((result[value], rows))
            else:
                result[value] = rows
    return result


def _candidate_groups_with_provenance(class_id: torch.Tensor,
                                      slot_class: torch.Tensor,
                                      old_rows: int) -> dict[int, torch.Tensor]:
    result = {}
    old_values = class_id[:int(old_rows)].long().cpu()
    known_old = sorted(int(v) for v in old_values.unique().tolist() if int(v) >= 0)
    if known_old:
        for value in known_old:
            rows = torch.where(old_values == value)[0]
            if len(rows):
                result[value] = rows
    else:
        result[-1] = torch.arange(int(old_rows))
    for value in sorted(int(v) for v in slot_class.unique().tolist() if int(v) >= 0):
        rows = torch.where(slot_class == value)[0] + int(old_rows)
        if value in result:
            result[value] = torch.cat((result[value], rows))
        else:
            result[value] = rows
    return result


def _rows_with_class_mix(groups: dict[int, torch.Tensor], batch: int,
                         seed: int, device: str, old_rows: int,
                         old_class_mix: float) -> torch.Tensor:
    """Keep legacy old-row mass while adding a bounded class-aware fraction."""
    old_rows = int(old_rows)
    mix = float(old_class_mix)
    if mix <= 0:
        old_values = [value for value in groups.values()
                      if len(value) and bool((value < old_rows).any())]
        new_groups = [value for value in groups.values()
                      if len(value) and bool((value >= old_rows).any())]
        old = torch.unique(torch.cat([value[value < old_rows].cpu()
                                      for value in old_values])) if old_values else torch.empty(0, dtype=torch.long)
        legacy = {-1: old}
        legacy.update({index: value[value >= old_rows]
                       for index, value in enumerate(new_groups)})
        return T._balanced_rows(legacy, batch, seed, device)

    old_groups = [value[value < old_rows].cpu() for value in groups.values()
                  if len(value) and bool((value < old_rows).any())]
    old_groups = [value for value in old_groups if len(value)]
    new_groups = [value[value >= old_rows].cpu() for value in groups.values()
                  if len(value) and bool((value >= old_rows).any())]
    old = torch.unique(torch.cat(old_groups)) if old_groups else torch.empty(0, dtype=torch.long)
    if not len(old):
        return T._balanced_rows(groups, batch, seed, device)

    per_group = max(1, int(batch) // (1 + len(new_groups)))
    old_balanced = min(per_group, int(round(per_group * mix)))
    old_uniform = per_group - old_balanced
    rows = []
    if old_uniform:
        rows.append(T._sample(old, old_uniform, int(seed) * 7919 + 17, device))
    if old_balanced:
        count = len(old_groups)
        base = old_balanced // count
        remainder = old_balanced % count
        for index, values in enumerate(old_groups):
            take = base + (1 if index < remainder else 0)
            if take:
                rows.append(T._sample(values, take, int(seed) * 7919 + 101 + index, device))
    for index, values in enumerate(new_groups):
        rows.append(T._sample(values, per_group,
                              int(seed) * 7919 + 1001 + index, device))
    result = torch.cat(rows) if rows else T._balanced_rows(groups, batch, seed, device)
    return result[:int(batch)]


def _assignment_matrix(assignment: dict, old_rows: int, support_rows: int,
                       candidate_rows: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-to-slot Q and the dense slot-to-raw P matrix.

    Q has one row per legal support node and one column per candidate slot.
    Old support rows map identically.  New raw rows are normalized over the
    slots that selected them; uncovered raw rows fall back to their nearest
    synthetic slot so every support row has a valid destination.
    """
    indices = assignment["indices"].to(device)
    probs = torch.softmax(assignment["logits"], dim=1)
    old_rows = int(old_rows)
    raw_rows = int(support_rows - old_rows)
    new_rows = int(candidate_rows - old_rows)
    if raw_rows <= 0 or new_rows <= 0:
        raise ValueError("support/candidate blocks must contain new rows")
    slot_to_raw = probs.new_zeros((new_rows, raw_rows))
    slot_to_raw = slot_to_raw.scatter_add(1, indices, probs)
    raw_mass = slot_to_raw.sum(dim=0)
    normalized = slot_to_raw.T / raw_mass.clamp_min(1e-8).unsqueeze(1)
    missing = raw_mass <= 1e-8
    if bool(missing.any()):
        nearest = assignment["distance"].to(device).argmin(dim=0)
        fallback = F.one_hot(nearest, num_classes=new_rows).to(probs.dtype)
        normalized = torch.where(missing.unsqueeze(1), fallback, normalized)
    old_identity = torch.eye(old_rows, device=device, dtype=probs.dtype)
    old_to_new = probs.new_zeros((old_rows, new_rows))
    raw_to_old = probs.new_zeros((raw_rows, old_rows))
    q = torch.cat((
        torch.cat((old_identity, old_to_new), dim=1),
        torch.cat((raw_to_old, normalized), dim=1),
    ), dim=0)
    return q, slot_to_raw


def _projected_kernel(support_binary: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Q^T A_support Q is the message operator induced by the legal support.
    projected = q.T @ support_binary @ q
    return projected.clamp_min(0)


def _straight_through_adjacency(mixed: torch.Tensor,
                                base_edge: torch.Tensor,
                                swap_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the final materialized topology in the forward pass.

    The hard matrix is produced by the exact same degree-preserving materializer
    used at output time.  The straight-through expression has the hard normalized
    adjacency as its forward value and the normalized soft kernel as its backward
    path.  This removes the previous soft-training/binary-evaluation mismatch
    without pretending that top-k edge selection is differentiable.
    """
    hard_edge = _materialize_edges(mixed, base_edge, swap_fraction=swap_fraction)
    hard = _binary_adjacency(hard_edge, mixed.shape[0], mixed.device, mixed.dtype)
    hard_adj = _normalize_dense(hard)
    soft_adj = _normalize_dense(mixed)
    straight = hard_adj.detach() - soft_adj.detach() + soft_adj
    return straight, hard


def _new_plasticity_loss(candidate_z: torch.Tensor,
                         candidate_text_z: torch.Tensor,
                         slot_class: torch.Tensor,
                         margin: float) -> torch.Tensor:
    """Encourage new-class graph/text alignment and inter-class separation."""
    classes = sorted(int(v) for v in slot_class.unique().tolist())
    if not classes:
        return candidate_z.new_zeros(())
    graph_centers, text_centers = [], []
    for cls in classes:
        mask = slot_class == int(cls)
        graph_centers.append(candidate_z[mask].mean(0))
        text_centers.append(candidate_text_z[mask].mean(0))
    graph_centers = F.normalize(torch.stack(graph_centers), dim=1)
    text_centers = F.normalize(torch.stack(text_centers), dim=1)
    align = (1.0 - (graph_centers * text_centers).sum(1)).mean()
    if len(classes) < 2:
        return align
    sim = graph_centers @ graph_centers.T
    upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    distances = 1.0 - sim[upper]
    separation = F.relu(float(margin) - distances).mean()
    return align + separation


def _materialize_edges(weight: torch.Tensor, base_edge: torch.Tensor,
                       swap_fraction: float = 1.0) -> torch.Tensor:
    """Materialize weights while optionally preserving most base neighbors."""
    if not 0.0 <= float(swap_fraction) <= 1.0:
        raise ValueError("swap_fraction must be in [0, 1]")
    weight = weight.detach().cpu()
    base_edge = base_edge.detach().cpu().long()
    n = weight.shape[0]
    degree = torch.bincount(base_edge[0], minlength=n)
    rows, cols = [], []
    for row in range(n):
        k = min(int(degree[row]), max(0, n - 1))
        if k <= 0:
            continue
        base_neighbors = set(int(value) for value in
                             base_edge[1, base_edge[0] == row].tolist()
                             if int(value) != row)
        base_neighbors = list(base_neighbors)[:k]
        swap_count = min(len(base_neighbors), int(math.floor(k * float(swap_fraction))))
        if swap_fraction < 1.0 and k >= 4 and swap_count == 0:
            swap_count = 1
        scores = weight[row].clone()
        scores[row] = -float("inf")
        if swap_fraction >= 1.0 or not base_neighbors:
            chosen = torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()
        elif swap_count == 0:
            chosen = base_neighbors
        else:
            base_scores = [(float(scores[col]), col) for col in base_neighbors]
            base_scores.sort()
            keep = [col for _, col in base_scores[swap_count:]]
            blocked = set(keep)
            candidate_scores = [(float(scores[col]), col) for col in range(n)
                                if col != row and col not in blocked]
            candidate_scores.sort(reverse=True)
            additions = [col for _, col in candidate_scores[:swap_count]]
            chosen = keep + additions
            if len(chosen) < k:
                fallback = [col for _, col in candidate_scores[swap_count:]
                            if col not in chosen]
                chosen.extend(fallback[:k - len(chosen)])
            chosen = chosen[:k]
        rows.extend([row] * len(chosen))
        cols.extend(int(value) for value in chosen)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def _train_teacher(support: dict, class_id: torch.Tensor, old_rows: int,
                   seed: int, device: str, cfg: T.Config) -> dict:
    T.validate_memory(support)
    model = T._student(seed, support["x"].shape[1],
                        support["text_emb"].shape[1], device)
    params0 = T._clone_params(T._params(model), detach=True)
    x = support["x"].to(device)
    text = support["text_emb"].to(device)
    binary = _binary_adjacency(support["edge_index"], len(x), device, x.dtype)
    adjacency = _normalize_dense(binary)
    groups = _groups_with_provenance(class_id, old_rows)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.teacher_lr))
    for step in range(int(cfg.teacher_steps)):
        rows = _rows_with_class_mix(
            groups, cfg.batch, int(seed) * 1009 + step, device,
            old_rows, cfg.old_class_mix)
        loss = T.contrastive_loss(T._params(model), x, text, adjacency, rows)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    params_t = T._clone_params(T._params(model), detach=True)
    with torch.no_grad():
        z = T.encode_graph(params_t, x, adjacency).detach()
    return {
        "seed": int(seed), "params0": params0, "params_t": params_t,
        "z": z, "final_loss": float(loss.detach().cpu()),
        "teacher_steps": int(cfg.teacher_steps),
    }


def _target_new(values: torch.Tensor, slot_to_raw: torch.Tensor,
                old_rows: int) -> torch.Tensor:
    raw_values = values[int(old_rows):]
    return slot_to_raw @ raw_values


def _objective(base: dict, support: dict, assignment: dict,
               relative: torch.Tensor, old_rows: int, references: list[dict],
               device: str, cfg: T.Config, create_graph: bool,
               message_mix: float, swap_fraction: float) -> tuple[torch.Tensor, dict, torch.Tensor]:
    candidate_x = base["x"].to(device) + relative
    candidate_text = base["text_emb"].to(device)
    support_binary = _binary_adjacency(
        support["edge_index"], len(support["x"]), device, candidate_x.dtype)
    q, slot_to_raw = _assignment_matrix(
        assignment, old_rows, len(support["x"]), len(candidate_x), device)
    projected = _projected_kernel(support_binary, q)
    base_binary = _binary_adjacency(
        base["edge_index"], len(candidate_x), device, candidate_x.dtype)
    mixed = ((1.0 - float(message_mix)) * base_binary
             + float(message_mix) * projected)
    if cfg.ste_forward and float(message_mix) != 0.0:
        candidate_adj, _ = _straight_through_adjacency(
            mixed, base["edge_index"], swap_fraction)
    else:
        candidate_adj = _normalize_dense(mixed)
    candidate_groups = _candidate_groups_with_provenance(
        support["class_id"], assignment["slot_class"], old_rows)
    probs = torch.softmax(assignment["logits"], dim=1)
    total = candidate_x.new_zeros(())
    records = []
    for ref in references:
        params_c = T._unroll(ref["params0"], candidate_x, candidate_text,
                             candidate_adj, candidate_groups, ref["seed"],
                             cfg, create_graph, row_selector=_rows_with_class_mix,
                             old_rows=old_rows, old_class_mix=cfg.old_class_mix)
        candidate_z = T.encode_graph(params_c, candidate_x, candidate_adj)
        candidate_text_z = T.encode_text(params_c, candidate_text)
        target_z = F.normalize(q.T @ ref["z"].to(device), dim=1)
        target_new_z = target_z[int(old_rows):]
        new_feature = T._class_pooled_feature(
            candidate_z[int(old_rows):], target_new_z,
            assignment["slot_class"])
        old_feature = (1.0 - F.cosine_similarity(
            candidate_z[:int(old_rows)], target_z[:int(old_rows)], dim=1,
            eps=1e-8)).mean()
        target_new_x = _target_new(support["x"].to(device), slot_to_raw,
                                   old_rows)
        payload = (1.0 - F.cosine_similarity(
            candidate_x[int(old_rows):], target_new_x, dim=1,
            eps=1e-8)).mean()
        trajectory = T._trajectory_cosine(
            params_c, ref["params0"], ref["params_t"], device)
        plasticity = _new_plasticity_loss(
            candidate_z[int(old_rows):], candidate_text_z[int(old_rows):],
            assignment["slot_class"].to(device), cfg.plasticity_margin)
        class_loss, coverage = T._assignment_penalties(
            probs, assignment["indices"].to(device),
            assignment["raw_class"].to(device), assignment["slot_class"])
        total = (total + cfg.feature_weight * new_feature
                 + cfg.old_feature_weight * old_feature
                 + cfg.payload_weight * payload
                 + cfg.trajectory_weight * trajectory
                 + cfg.class_weight * class_loss
                 + cfg.assignment_weight * coverage
                 + cfg.plasticity_weight * plasticity)
        records.append({
            "seed": int(ref["seed"]),
            "new_feature": float(new_feature.detach()),
            "old_feature": float(old_feature.detach()),
            "payload": float(payload.detach()),
            "trajectory": float(trajectory.detach()),
            "plasticity": float(plasticity.detach()),
            "class_loss": float(class_loss.detach()),
            "coverage": float(coverage.detach()),
            "teacher_loss": float(ref["final_loss"]),
        })
    total = total / max(1, len(references))
    scales = base["x"].to(device).norm(dim=1).clamp_min(1e-6)
    anchor = (relative[:int(old_rows)] /
              scales[:int(old_rows)].unsqueeze(1)).square().mean()
    total = total + cfg.anchor_weight * anchor
    record = {
        "seeds": records,
        "anchor": float(anchor.detach()),
        "assignment_entropy": float((-(probs * probs.clamp_min(1e-8).log()).sum(1)).mean().detach()),
        "projected_edge_mass": float(projected.detach().sum()),
        "message_mix": float(message_mix),
        "ste_forward": bool(cfg.ste_forward),
        "plasticity_weight": float(cfg.plasticity_weight),
    }
    return total, record, mixed


def optimize(base: dict, support: dict, class_id: torch.Tensor, old_rows: int,
             stage: int, device: str = "cpu", cfg: T.Config = T.Config(),
             learn_x: bool = True, learn_p: bool = True,
             message_mix: float = 1.0, swap_fraction: float = 1.0) -> tuple[dict, dict, dict]:
    T.validate_memory(base)
    T.validate_memory(support)
    if not 0.0 <= float(message_mix) <= 1.0:
        raise ValueError("message_mix must be in [0, 1]")
    if not 0.0 <= float(swap_fraction) <= 1.0:
        raise ValueError("swap_fraction must be in [0, 1]")
    if len(class_id) != len(support["x"]):
        raise ValueError("class_id must align with support rows")
    base_hash, support_hash = T.memory_hash(base), T.memory_hash(support)
    assignment = T.build_sparse_assignment(base, support, old_rows, class_id,
                                           cfg, device)
    initial_logits = assignment["logits"].detach().clone()

    # A static topology arm is useful for attribution: it materializes the
    # fixed initial assignment without changing X or learning P.  It does not
    # train a Teacher because no continuous revision is being performed.
    if not learn_x and not learn_p:
        result = T.clone_memory(base)
        with torch.no_grad():
            q, _ = _assignment_matrix(
                assignment, old_rows, len(support["x"]), len(base["x"]), device)
            support_binary = _binary_adjacency(
                support["edge_index"], len(support["x"]), device,
                base["x"].dtype)
            projected = _projected_kernel(support_binary, q)
            base_binary = _binary_adjacency(
                base["edge_index"], len(base["x"]), device, base["x"].dtype)
            mixed = ((1.0 - float(message_mix)) * base_binary
                     + float(message_mix) * projected)
            if float(message_mix) == 0.0:
                result["edge_index"] = base["edge_index"].clone()
            else:
                result["edge_index"] = _materialize_edges(
                    mixed, base["edge_index"], swap_fraction=swap_fraction)
        assignment_out = {
            "indices": assignment["indices"].tolist(),
            "weights": torch.softmax(initial_logits, dim=1).tolist(),
            "slot_class": assignment["slot_class"].tolist(),
            "raw_class": assignment["raw_class"].tolist(),
            "top_r": int(assignment["indices"].shape[1]),
            "construction_seconds": 0.0,
        }
        audit = {
            "version": "teacher-guided-message-revision-v2",
            "stage": int(stage),
            "mode": {"learn_x": False, "learn_p": False},
            "message_mix": float(message_mix),
            "swap_fraction": float(swap_fraction),
            "config": cfg.dict(),
            "old_rows": int(old_rows),
            "new_rows": int(len(base["x"]) - old_rows),
            "input_base_sha256": base_hash,
            "input_support_sha256": support_hash,
            "output_sha256": T.memory_hash(result),
            "selected_outer_step": 0,
            "validation_objective": None,
            "baseline_validation_objective": None,
            "gate_fallback": False,
            "teacher_seeds": [],
            "validation_teacher_seeds": [],
            "curve": [],
            "p_mean_abs_delta": 0.0,
            "p_kl_to_initial": 0.0,
            "edge_count_before": int(base["edge_index"].shape[1]),
            "edge_count_after": int(result["edge_index"].shape[1]),
            "text_changed": False,
            "budget_changed": False,
            "topology_changed": not torch.equal(
                result["edge_index"], base["edge_index"]),
            "historical_raw_reads": 0,
            "report_labels_used": False,
            "old_class_provenance": bool((support["class_id"][:old_rows] >= 0).any()),
        }
        return result, audit, assignment_out

    train_refs = [_train_teacher(support, class_id, old_rows, seed, device, cfg)
                  for seed in cfg.teacher_seeds]
    val_refs = [_train_teacher(support, class_id, old_rows, seed, device, cfg)
                for seed in cfg.validation_teacher_seeds]
    relative = torch.nn.Parameter(torch.zeros_like(base["x"].to(device)),
                                  requires_grad=bool(learn_x))
    p_logits = torch.nn.Parameter(assignment["logits"].detach().clone(),
                                  requires_grad=bool(learn_p))
    assignment["logits"] = p_logits
    params = [value for value in (relative, p_logits) if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(cfg.outer_lr))
    base_rows = base["x"].to(device).detach()
    curve, best, best_val, best_step = [], None, math.inf, 0
    started = time.perf_counter()

    # Evaluate the unrevised candidate before any outer update.  The previous
    # implementation always selected an updated checkpoint because ``best``
    # started at +inf.  This made a proxy-worse revision unavoidable.  The
    # baseline gate turns the identity update into a real no-op candidate.
    zero_relative = torch.zeros_like(base["x"].to(device))
    baseline_loss, baseline_record, _ = _objective(
        base, support, assignment, zero_relative, old_rows, val_refs,
        device, cfg, False, message_mix, swap_fraction)
    baseline_val = float(baseline_loss.detach())
    best_val = baseline_val if cfg.baseline_gate else math.inf

    for step in range(1, int(cfg.outer_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_record, _ = _objective(
            base, support, assignment, relative, old_rows, train_refs,
            device, cfg, True, message_mix, swap_fraction)
        if not torch.isfinite(train_loss):
            raise FloatingPointError("non-finite message-guided objective")
        train_loss.backward()
        if learn_x and relative.grad is None:
            raise FloatingPointError("missing X gradient")
        if learn_p and p_logits.grad is None:
            raise FloatingPointError("missing P gradient")
        optimizer.step()
        T._project(relative, base_rows, old_rows, cfg.old_radius, cfg.new_radius)
        val_loss, val_record, _ = _objective(
            base, support, assignment, relative, old_rows, val_refs,
            device, cfg, False, message_mix, swap_fraction)
        score = float(val_loss.detach())
        curve.append({
            "step": int(step), "train_objective": float(train_loss.detach()),
            "validation_objective": score, "train": train_record,
            "validation": val_record,
            "x_grad_norm": float(relative.grad.norm().detach()) if relative.grad is not None else 0.0,
            "p_grad_norm": float(p_logits.grad.norm().detach()) if p_logits.grad is not None else 0.0,
        })
        if score < best_val - float(cfg.gate_margin):
            best_val, best_step = score, step
            best = (relative.detach().clone(), p_logits.detach().clone())
    gate_fallback = best is None
    if gate_fallback:
        best_relative = torch.zeros_like(base["x"])
        best_logits = initial_logits.clone()
        best_val = baseline_val
    else:
        best_relative, best_logits = best
    result = T.clone_memory(base)
    result["x"] = base["x"].clone() + best_relative.cpu()
    assignment["logits"] = best_logits
    with torch.no_grad():
        q, _ = _assignment_matrix(assignment, old_rows, len(support["x"]),
                                  len(base["x"]), device)
        support_binary = _binary_adjacency(
            support["edge_index"], len(support["x"]), device,
            result["x"].dtype)
        projected = _projected_kernel(support_binary, q)
        base_binary = _binary_adjacency(
            base["edge_index"], len(base["x"]), device, result["x"].dtype)
        mixed = ((1.0 - float(message_mix)) * base_binary
                 + float(message_mix) * projected)
        if not gate_fallback and float(message_mix) != 0.0:
            materialized = _materialize_edges(
                mixed, base["edge_index"], swap_fraction=swap_fraction)
            result["edge_index"] = materialized
        probs = torch.softmax(best_logits, dim=1).cpu()
        relative_ratio = ((result["x"] - base["x"]).norm(dim=1) /
                          base["x"].norm(dim=1).clamp_min(1e-6))
    if T.memory_hash(base) != base_hash or T.memory_hash(support) != support_hash:
        raise AssertionError("optimizer mutated input memory")
    for key in ("text_emb", "texts", "mass", "depth", "stage"):
        left, right = base[key], result[key]
        equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
        if not equal:
            raise AssertionError(f"immutable field changed: {key}")
    initial_probs = torch.softmax(initial_logits, dim=1).cpu()
    assignment_out = {
        "indices": assignment["indices"].tolist(),
        "weights": probs.tolist(),
        "slot_class": assignment["slot_class"].tolist(),
        "raw_class": assignment["raw_class"].tolist(),
        "top_r": int(assignment["indices"].shape[1]),
        "construction_seconds": float(time.perf_counter() - started),
    }
    audit = {
        "version": ("teacher-guided-message-revision-v3"
                     if cfg.ste_forward or cfg.plasticity_weight else
                     "teacher-guided-message-revision-v2"),
        "stage": int(stage), "mode": {"learn_x": bool(learn_x), "learn_p": bool(learn_p)},
        "message_mix": float(message_mix), "swap_fraction": float(swap_fraction),
        "ste_forward": bool(cfg.ste_forward),
        "plasticity_weight": float(cfg.plasticity_weight),
        "plasticity_margin": float(cfg.plasticity_margin),
        "config": cfg.dict(), "old_rows": int(old_rows),
        "new_rows": int(len(base["x"]) - old_rows),
        "input_base_sha256": base_hash, "input_support_sha256": support_hash,
        "output_sha256": T.memory_hash(result), "selected_outer_step": int(best_step),
        "validation_objective": float(best_val),
        "baseline_validation_objective": float(baseline_val),
        "baseline_validation": baseline_record,
        "gate_fallback": bool(gate_fallback),
        "gate_margin": float(cfg.gate_margin),
        "max_relative_displacement": float(relative_ratio.max()),
        "mean_relative_displacement": float(relative_ratio.mean()),
        "old_max_relative_displacement": float(relative_ratio[:old_rows].max()) if old_rows else 0.0,
        "new_max_relative_displacement": float(relative_ratio[old_rows:].max()) if len(relative_ratio) > old_rows else 0.0,
        "teacher_seeds": list(cfg.teacher_seeds),
        "validation_teacher_seeds": list(cfg.validation_teacher_seeds),
        "teacher_final_losses": {str(ref["seed"]): ref["final_loss"]
                                 for ref in train_refs + val_refs},
        "curve": curve,
        "p_mean_abs_delta": float((probs - initial_probs).abs().mean()),
        "p_kl_to_initial": float((probs.clamp_min(1e-8) *
                                   (probs.clamp_min(1e-8).log() -
                                    initial_probs.clamp_min(1e-8).log())).sum(1).mean()),
        "edge_count_before": int(base["edge_index"].shape[1]),
        "edge_count_after": int(result["edge_index"].shape[1]),
        "text_changed": False, "budget_changed": False,
        "topology_changed": not torch.equal(
            result["edge_index"], base["edge_index"]),
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "old_class_provenance": bool((support["class_id"][:old_rows] >= 0).any()),
    }
    return result, audit, assignment_out
