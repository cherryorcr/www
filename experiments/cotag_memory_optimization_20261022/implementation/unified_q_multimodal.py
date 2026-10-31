"""Unified Q-based multimodal continual TAG condensation.

The constructor uses one differentiable assignment matrix Q to derive the
new synthetic features, text payloads, and topology.  Q is class masked and
capacity balanced with Sinkhorn.  A legal boundary sketch can connect the
current raw delta to already materialized parent slots without rereading
historical raw features or text.

This module is intentionally independent from the older CaT-warm-start GM
implementation.  It is a compact research branch for testing whether the
three separately optimized components can be replaced by one object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import graphtext_prompt_student as P
import unified_q_pair_distribution as J
from unified_q_payload_projection import q_text_coverage, q_text_medoid
from unified_q_blocks import ClassBlocks


@dataclass(frozen=True)
class Config:
    proxy_hidden: int = 128
    proxy_out: int = 128
    proxy_seeds: tuple[int, ...] = (2711,)
    steps: int = 30
    lr_q: float = 0.15
    lr_x: float = 5e-3
    temperature: float = 0.35
    sinkhorn_steps: int = 30
    sinkhorn_tolerance: float = 0.0
    sinkhorn_max_steps: int = 512
    feature_weight: float = 1.0
    message_weight: float = 0.75
    text_weight: float = 0.35
    lambda_gradient: float = 1.0
    lambda_propagation: float = 0.50
    lambda_q_anchor: float = 0.10
    lambda_x_anchor: float = 0.10
    lambda_cat_anchor: float = 0.25
    lambda_distribution: float = 0.0
    distribution_hidden: int = 512
    distribution_bank_size: int = 4
    distribution_chunk_size: int = 4096
    lambda_joint_distribution: float = 0.0
    joint_distribution_mode: str = "paired"
    joint_distribution_hidden: int = 64
    joint_distribution_banks: int = 2
    joint_distribution_chunk: int = 2048
    trust_radius: float = 0.05
    materialize_top_k: int = 1
    straight_through_text: bool = True
    hard_topology_forward: bool = True
    support_cap: int = 0
    text_chars: int = 320
    text_materialization: str = "q_max"
    feature_pooling: str = "raw"
    topology_edge_factor: float = 1.0
    boundary_edge_factor: float = 1.0
    init_mode: str = "diverse_message_text"
    loss_weighting: str = "group_class"
    pair_chunk_size: int = 256
    text_assignment: str = "learned"
    topology_assignment: str = "learned"
    optimization: str = "joint"
    train_q: bool = True
    train_residual: bool = True
    pair_weight: float = 0.5
    propagation_metric: str = "dimension_mean"
    topology_refine_swaps: int = 0
    old_update_radius: float = 0.0
    lambda_old_anchor: float = 0.1
    random_q: bool = False
    # Edge-free proxy graphs contain only self loops.  Keep the historical
    # dense path as the default for exact reproduction, but allow resource
    # audits to use the equivalent sparse operator.
    sparse_edgefree_forward: bool = False
    assignment_storage: str = "dense"
    seed: int = 271006


def _norm(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1)


def _unique_undirected_index(edge_index: torch.Tensor, n: int,
                             device: str | torch.device) -> torch.Tensor:
    """Return one binary directed pair per undirected edge.

    The TAG loaders commonly store both directions already.  Canonicalizing
    here keeps the constructor's propagation identical to the Student's
    de-duplicated normalized adjacency.
    """
    if edge_index is None or not edge_index.numel():
        return torch.empty((2, 0), dtype=torch.long, device=device)
    edges = edge_index.long().to(device)
    if int(edges.min()) < 0 or int(edges.max()) >= int(n):
        raise ValueError("edge endpoint outside graph")
    edges = edges[:, edges[0] != edges[1]]
    if not edges.numel():
        return torch.empty((2, 0), dtype=torch.long, device=device)
    directed = torch.cat((edges, edges.flip(0)), dim=1)
    keys = torch.unique(directed[0] * int(n) + directed[1], sorted=True)
    return torch.stack((keys // int(n), keys % int(n)))


def _sparse_adjacency(edge_index: torch.Tensor, n: int,
                      device: str | torch.device) -> torch.Tensor:
    index = _unique_undirected_index(edge_index, n, device)
    values = torch.ones(index.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(index, values, (int(n), int(n)),
                                   device=device).coalesce()


def _adjacency_mm(adjacency: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return (torch.sparse.mm(adjacency, value)
            if adjacency.is_sparse else adjacency @ value)


def _dense_adjacency(edge_index: torch.Tensor, n: int,
                     device: str | torch.device,
                     weights: torch.Tensor | None = None) -> torch.Tensor:
    """Build a symmetric unnormalized adjacency with no implicit loops."""
    result = torch.zeros((int(n), int(n)), dtype=torch.float32, device=device)
    index = _unique_undirected_index(edge_index, n, device)
    if not index.numel():
        return result
    if weights is not None:
        raise ValueError("binary edge helper does not accept edge weights")
    values = torch.ones(index.shape[1], dtype=result.dtype, device=device)
    result.index_put_((index[0], index[1]), values, accumulate=False)
    return result


def _normalize_dense(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[0]
    value = adjacency.clone()
    value = value + torch.eye(n, dtype=value.dtype, device=value.device)
    degree = value.sum(1).clamp_min(1.0)
    scale = degree.rsqrt()
    return scale[:, None] * value * scale[None, :]


def _sinkhorn_chunk(value: torch.Tensor, col_log_mass: torch.Tensor,
                    steps: int) -> torch.Tensor:
    for _ in range(int(steps)):
        value = value - torch.logsumexp(value, dim=1, keepdim=True)
        value = value + col_log_mass[None, :] - torch.logsumexp(value, dim=0, keepdim=True)
    return value


_SINKHORN_LARGE_BLOCK = 1_000_000


def _log_sinkhorn(scores: torch.Tensor, column_mass: torch.Tensor,
                  steps: int, tolerance: float = 0.0, max_steps: int = 512,
                  audit: list | None = None) -> torch.Tensor:
    """Return a differentiable row-mass-one, column-capacity assignment."""
    rows, cols = scores.shape
    value = scores.float()
    col_log_mass = column_mass.to(value.device, value.dtype).clamp_min(1e-8).log()
    used = max(1, int(steps))
    if tolerance > 0:
        if max_steps < used:
            raise ValueError("Sinkhorn maximum is smaller than minimum steps")
        with torch.no_grad():
            probe = _sinkhorn_chunk(value.detach(), col_log_mass, used)
            while True:
                candidate = probe.exp()
                row_error = float((candidate.sum(1) - 1).abs().max())
                col_error = float(((candidate.sum(0) - column_mass) / column_mass).abs().max())
                if max(row_error, col_error) <= tolerance:
                    break
                if used >= max_steps:
                    raise RuntimeError(f"Sinkhorn failed tolerance {tolerance}: row={row_error}, col={col_error}")
                count = min(8, max_steps - used)
                probe = _sinkhorn_chunk(probe, col_log_mass, count)
                used += count
    large_block = value.numel() >= _SINKHORN_LARGE_BLOCK
    if torch.is_grad_enabled() and value.requires_grad and (used > 64 or large_block):
        # Large class blocks need recomputation even with only 30 iterations.
        # This changes activation storage, not the transport or its gradients.
        chunk = 8 if large_block else 16
        for start in range(0, used, chunk):
            count = min(chunk, used - start)
            value = checkpoint(_sinkhorn_chunk, value, col_log_mass, count, use_reentrant=False)
    else:
        value = _sinkhorn_chunk(value, col_log_mass, used)
    result = value.exp()
    if audit is not None:
        with torch.no_grad():
            audit.append({"rows": rows, "cols": cols, "iterations": used,
                "row_error": float((result.sum(1) - 1).abs().max()),
                "column_relative_error": float(((result.sum(0) - column_mass) / column_mass).abs().max())})
    return result


def _masked_sinkhorn(logits: torch.Tensor, raw_labels: torch.Tensor,
                     slot_labels: torch.Tensor, temperature: float,
                     steps: int, tolerance: float = 0.0,
                     max_steps: int = 512, audit: list | None = None) -> torch.Tensor:
    """Apply Sinkhorn independently inside each class block."""
    if isinstance(logits, ClassBlocks):
        blocks = []
        for value in logits.blocks:
            capacity = value.new_full((value.shape[1],), value.shape[0] / value.shape[1])
            blocks.append(_log_sinkhorn(value / max(float(temperature), 1e-6),
                                       capacity, steps, tolerance, max_steps, audit))
        return ClassBlocks(logits.shape, logits.rows, logits.slots, blocks)
    device = logits.device
    q = torch.zeros_like(logits)
    raw_labels = raw_labels.to(device).long()
    slot_labels = slot_labels.to(device).long()
    for cls in torch.unique(raw_labels, sorted=True).tolist():
        raw_ids = torch.where(raw_labels == int(cls))[0]
        slot_ids = torch.where(slot_labels == int(cls))[0]
        if not len(slot_ids):
            raise ValueError(f"no slot for current class {cls}")
        local_logits = logits.index_select(0, raw_ids).index_select(1, slot_ids)
        capacity = torch.full(
            (len(slot_ids),), float(len(raw_ids)) / float(len(slot_ids)),
            dtype=local_logits.dtype, device=device)
        local_q = _log_sinkhorn(local_logits / max(float(temperature), 1e-6),
                                capacity, steps, tolerance, max_steps, audit)
        q[raw_ids[:, None], slot_ids[None, :]] = local_q
    return q


def _append_text(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= int(limit):
        return value
    prefix = value[:int(limit)]
    pos = prefix.rfind(" ")
    return prefix[:pos].strip() if pos >= 8 else prefix.strip()


def _boundary_matrix(delta: dict, parent: dict | None,
                     device: str | torch.device) -> torch.Tensor:
    """Map current-local to historical raw edges onto old synthetic slots."""
    n = len(delta["x"])
    old_n = 0 if parent is None else len(parent["x"])
    matrix = torch.zeros((n, old_n), dtype=torch.float32, device=device)
    if parent is None or old_n == 0:
        return matrix
    boundary = delta.get("boundary_edge_index",
                         torch.empty((2, 0), dtype=torch.long))
    if boundary is None or not boundary.numel():
        return matrix
    support = parent.get("support_ids", [])
    raw_to_slot: dict[int, int] = {}
    for slot, values in enumerate(support):
        for raw_id in values:
            raw_to_slot.setdefault(int(raw_id), int(slot))
    for local_id, raw_old_id in boundary.long().cpu().T.tolist():
        slot = raw_to_slot.get(int(raw_old_id))
        if slot is not None and 0 <= int(local_id) < n:
            matrix[int(local_id), slot] += 1.0
    # A high-degree incoming node must not dominate the message operator only
    # because it exposes many historical edge IDs.  Preserve the boundary
    # direction while giving each current node at most unit total boundary
    # mass.  The same normalized sketch is used by the real and synthetic
    # proxy graphs.
    matrix = matrix / matrix.sum(1, keepdim=True).clamp_min(1.0)
    return matrix


def _slot_prototypes(delta: dict, slot_labels: torch.Tensor, cfg: Config,
                     device: str | torch.device, raw_message: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                list[int]]:
    raw_x = _norm(delta["x"].float().to(device))
    raw_t = _norm(delta["text_emb"].float().to(device))
    labels = delta["y"].long().to(device)
    if cfg.init_mode not in ("legacy_spread", "diverse_message_text"):
        raise ValueError(f"unknown initialization: {cfg.init_mode}")
    selected_rows = torch.empty(len(slot_labels), dtype=torch.long, device=device)
    for cls in torch.unique(slot_labels, sorted=True).tolist():
        ids = torch.where(labels == int(cls))[0]
        slots = torch.where(slot_labels.to(device) == int(cls))[0]
        if not len(ids):
            raise ValueError(f"no current raw support for class {cls}")
        slot_count = len(slots)
        if slot_count > len(ids):
            raise ValueError("more slots than raw rows in a class")
        if cfg.init_mode == "legacy_spread":
            chosen = torch.arange(slot_count, device=device) * len(ids) // slot_count
        else:
            # Cache nearest selected distance; O(n_c m_c d), not all pairs.
            signature = _norm(torch.cat((
                math.sqrt(cfg.feature_weight) * raw_x[ids],
                math.sqrt(cfg.message_weight) * raw_message[ids],
                math.sqrt(cfg.text_weight) * raw_t[ids],
            ), dim=1))
            center = _norm(signature.mean(0, keepdim=True)).squeeze(0)
            next_id = (signature @ center).argmax()
            selected = torch.zeros(len(ids), dtype=torch.bool, device=device)
            nearest = torch.full((len(ids),), torch.inf, device=device)
            chosen_list = []
            for _ in range(slot_count):
                chosen_list.append(next_id)
                selected[next_id] = True
                nearest = torch.minimum(nearest, 1.0 - signature @ signature[next_id])
                next_id = nearest.masked_fill(selected, -torch.inf).argmax()
            chosen = torch.stack(chosen_list)
        selected_rows[slots] = ids[chosen]
    slot_message = (raw_x[selected_rows] if cfg.init_mode == "legacy_spread"
                    else raw_message[selected_rows])
    selected_ids = delta["ids"][selected_rows.cpu()].long().tolist()
    return raw_x[selected_rows], raw_t[selected_rows], slot_message, selected_ids


def _neighbor_message(delta: dict, device: str | torch.device) -> torch.Tensor:
    """Return normalized mean neighbor messages without an n-by-n tensor."""
    x = _norm(delta["x"].float().to(device))
    adjacency = delta.get("_raw_adjacency")
    if adjacency is None:
        adjacency = _sparse_adjacency(
            delta.get("edge_index", torch.empty((2, 0), dtype=torch.long)),
            len(x), device)
    message = torch.sparse.mm(adjacency, x)
    return message


def _initial_logits(delta: dict, parent: dict | None,
                    slot_labels: torch.Tensor, cfg: Config,
                    device: str | torch.device,
                    boundary: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    raw_x = _norm(delta["x"].float().to(device))
    raw_t = _norm(delta["text_emb"].float().to(device))
    raw_labels = delta["y"].long().to(device)
    if boundary is None:
        boundary = _boundary_matrix(delta, parent, device)
    old_x = (parent["x"].float().to(device) if parent is not None
             else torch.empty((0, raw_x.shape[1]), device=device))
    raw_message = _neighbor_message(delta, device)
    if old_x.numel():
        raw_message = raw_message + boundary @ _norm(old_x)
    raw_message = _norm(raw_message + 1e-6 * raw_x)
    slot_x, slot_t, slot_message, selected_ids = _slot_prototypes(
        delta, slot_labels, cfg, device, raw_message)
    blocked = cfg.assignment_storage == "class_blocks"
    logits = (None if blocked else torch.full((len(raw_x), len(slot_labels)), -1e4,
                                              dtype=torch.float32, device=device))
    block_rows, block_slots, block_values = [], [], []
    costs = []
    for cls in torch.unique(raw_labels, sorted=True).tolist():
        raw_ids = torch.where(raw_labels == int(cls))[0]
        slot_ids = torch.where(slot_labels.to(device) == int(cls))[0]
        feature_score = raw_x.index_select(0, raw_ids) @ slot_x.index_select(0, slot_ids).T
        message_score = raw_message.index_select(0, raw_ids) @ slot_message.index_select(0, slot_ids).T
        text_score = raw_t.index_select(0, raw_ids) @ slot_t.index_select(0, slot_ids).T
        score = (float(cfg.feature_weight) * feature_score
                 + float(cfg.message_weight) * message_score
                 + float(cfg.text_weight) * text_score)
        if blocked:
            block_rows.append(raw_ids)
            block_slots.append(slot_ids)
            block_values.append(score)
        else:
            logits[raw_ids[:, None], slot_ids[None, :]] = score
        costs.append({"class": int(cls), "raw_rows": int(len(raw_ids)),
                      "slots": int(len(slot_ids)),
                      "feature_mean": float(feature_score.mean().detach().cpu()),
                      "message_mean": float(message_score.mean().detach().cpu()),
                      "text_mean": float(text_score.mean().detach().cpu())})
    if blocked:
        logits = ClassBlocks((len(raw_x), len(slot_labels)), block_rows, block_slots, block_values)
    q0 = _masked_sinkhorn(logits, raw_labels, slot_labels,
                          cfg.temperature, cfg.sinkhorn_steps,
                          cfg.sinkhorn_tolerance, cfg.sinkhorn_max_steps)
    return logits, q0, {"class_costs": costs,
                        "boundary_nonzero": int((boundary > 0).sum().item()),
                        "prototype_mode": str(cfg.init_mode),
                        "prototype_source_ids": selected_ids}


def _soft_topology(q: torch.Tensor, delta: dict, parent: dict | None,
                   boundary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full soft synthetic adjacency and its new-node components."""
    device = q.device
    new_n = q.shape[1]
    old_n = 0 if parent is None else len(parent["x"])
    raw_adj = delta.get("_raw_adjacency")
    if raw_adj is None:
        raw_adj = _sparse_adjacency(
            delta.get("edge_index", torch.empty((2, 0), dtype=torch.long)),
            len(delta["x"]), device)
    mass = q.sum(0).clamp_min(1e-6)
    new_new = q.T @ torch.sparse.mm(raw_adj, q)
    new_new = new_new / torch.sqrt(mass[:, None] * mass[None, :])
    new_new.fill_diagonal_(0.0)
    cross = q.T @ boundary
    cross = cross / torch.sqrt(mass[:, None])
    full = torch.zeros((old_n + new_n, old_n + new_n), dtype=torch.float32,
                       device=device)
    if parent is not None:
        old_adj = _dense_adjacency(parent.get("edge_index",
                                    torch.empty((2, 0), dtype=torch.long)),
                                   old_n, device)
        full[:old_n, :old_n] = old_adj
    full[old_n:, old_n:] = new_new
    if old_n:
        full[old_n:, :old_n] = cross
        full[:old_n, old_n:] = cross.T
    return _normalize_dense(full), new_new, cross


def _reference_graph(delta: dict, parent: dict | None,
                     boundary: torch.Tensor, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the legal real-side graph: parent slots plus current raw delta."""
    current_x = delta["x"].float().to(device)
    old_n = 0 if parent is None else len(parent["x"])
    n = old_n + len(current_x)
    edge_parts = []
    if parent is not None:
        parent_edges = parent.get("edge_index",
                                  torch.empty((2, 0), dtype=torch.long)).long()
        if parent_edges.numel():
            edge_parts.append(parent_edges.cpu())
    raw_edges = delta.get("edge_index",
                          torch.empty((2, 0), dtype=torch.long)).long().cpu()
    if raw_edges.numel():
        edge_parts.append(raw_edges + old_n)
    if old_n and boundary.numel():
        cross = torch.nonzero(boundary.detach().cpu() > 0, as_tuple=False)
        if len(cross):
            cross_edges = torch.stack((cross[:, 0] + old_n, cross[:, 1]))
            edge_parts.extend((cross_edges, cross_edges.flip(0)))
    edges = (torch.cat(edge_parts, dim=1)
             if edge_parts else torch.empty((2, 0), dtype=torch.long))
    adjacency = P.normalized_adjacency(edges, n, str(device))
    return adjacency, torch.cat(
        (parent["x"].float().to(device) if parent is not None
         else torch.empty((0, current_x.shape[1]), device=device), current_x), 0)


def _encode_graph(model: P.GraphTextPromptStudent, x: torch.Tensor,
                  adjacency: torch.Tensor) -> torch.Tensor:
    h = F.relu(_adjacency_mm(adjacency, model.g1.lin(x)))
    return F.normalize(_adjacency_mm(adjacency, model.g2.lin(h)), dim=1)


def _group_class_weights(labels: torch.Tensor, group_ids: torch.Tensor,
                         mode: str, device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
    labels = labels.long().to(device)
    group_ids = group_ids.long().to(device)
    if mode == "node_mean":
        return torch.full((len(labels),), 1.0 / max(1, len(labels)),
                          dtype=dtype, device=device)
    if mode != "group_class":
        raise ValueError(f"unknown loss weighting: {mode}")
    pairs = torch.stack((group_ids, labels), dim=1)
    unique, inverse, counts = torch.unique(
        pairs, dim=0, return_inverse=True, return_counts=True)
    present_groups = torch.unique(group_ids, sorted=True)
    group_mass = 1.0 / max(1, len(present_groups))
    weights = torch.zeros(len(labels), dtype=dtype, device=device)
    for pair_id in range(len(unique)):
        group = int(unique[pair_id, 0])
        class_count = max(1, int((unique[:, 0] == group).sum().item()))
        weights[inverse == pair_id] = group_mass / class_count / max(
            1, int(counts[pair_id]))
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def _proxy_weights(ref_labels, ref_groups, syn_labels, syn_groups, mode, device, dtype):
    if mode != "student_mass":
        return (_group_class_weights(ref_labels, ref_groups, mode, device, dtype),
                _group_class_weights(syn_labels, syn_groups, mode, device, dtype))
    ref = torch.stack((ref_groups.to(device), ref_labels.to(device)), dim=1)
    syn = torch.stack((syn_groups.to(device), syn_labels.to(device)), dim=1)
    if not len(ref) or not len(syn):
        raise ValueError("student-mass weighting requires nonempty reference and memory")
    ref_pairs = {tuple(int(v) for v in row.tolist())
                 for row in torch.unique(ref, dim=0)}
    syn_pairs = {tuple(int(v) for v in row.tolist())
                 for row in torch.unique(syn, dim=0)}
    missing_reference = syn_pairs - ref_pairs
    if missing_reference:
        raise ValueError(
            "student-mass weighting requires every synthetic stage/class "
            f"group in reference; missing={sorted(missing_reference)}")
    rw = torch.zeros(len(ref), device=device, dtype=dtype)
    sw = torch.full((len(syn),), 1.0 / len(syn), device=device, dtype=dtype)
    # Match the memory sampler's group masses without overweighting
    # uncompressed delta rows.  Reference-only groups intentionally receive
    # zero mass because no synthetic slot represents them.
    for pair in torch.unique(syn, dim=0):
        ri, si = (ref == pair).all(1), (syn == pair).all(1)
        rw[ri] = sw[si].sum() / ri.sum()
    if not torch.isfinite(rw).all() or not torch.isfinite(sw).all():
        raise FloatingPointError("nonfinite student-mass proxy weights")
    if not torch.isclose(rw.sum(), torch.ones((), device=device, dtype=dtype)):
        raise AssertionError("reference student-mass weights do not sum to one")
    return rw, sw


def _loss(model: P.GraphTextPromptStudent, memory: dict,
          adjacency: torch.Tensor, prompt_emb: torch.Tensor,
          group_ids: torch.Tensor | None = None,
          row_ids: torch.Tensor | None = None,
          loss_weighting: str = "group_class", chunk_size: int = 0,
          pair_weight: float = 0.5,
          sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    graph = _encode_graph(model, memory["x"].float(), adjacency)
    text = model.encode_text(memory["text_emb"].float())
    if row_ids is not None:
        row_ids = row_ids.long().to(graph.device)
        graph = graph.index_select(0, row_ids)
        text = text.index_select(0, row_ids)
    scale = model.logit_scale.exp().clamp(max=100.0)
    labels = memory["labels"].long().to(graph.device)
    if row_ids is not None:
        labels = labels.index_select(0, row_ids)
    if group_ids is None:
        group_ids = torch.zeros(len(memory["labels"]), dtype=torch.long)
    group_ids = group_ids.long().to(graph.device)
    if row_ids is not None:
        group_ids = group_ids.index_select(0, row_ids)
    if sample_weights is None:
        weights = _group_class_weights(
            labels, group_ids, loss_weighting, graph.device, graph.dtype)
    else:
        if len(sample_weights) != len(memory["labels"]):
            raise ValueError("sample weights must cover the unsampled memory")
        weights = sample_weights.to(graph.device)
        if row_ids is not None:
            weights = weights.index_select(0, row_ids)
        weights = P._normalized_sample_weights(weights, len(labels), graph.device, graph.dtype)
    pair = P.multi_positive_pair_loss(
        graph, text, labels, scale,
        sample_weights=(None if loss_weighting == "node_mean" and sample_weights is None else weights),
        chunk_size=chunk_size)
    logits = scale * (graph @ model.encode_text(prompt_emb.float().to(graph.device)).T)
    prompt_values = F.cross_entropy(logits, labels, reduction="none")
    prompt = (weights * prompt_values).sum()
    margin = P.prompt_margin_loss(logits, labels, 0.20,
                                 sample_weights=weights)
    return float(pair_weight) * pair + prompt + 0.1 * margin


def _gradient_distance(real: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(real.view(1, -1), synthetic.view(1, -1))
    norm_gap = (torch.log(real.norm().clamp_min(1e-8))
                - torch.log(synthetic.norm().clamp_min(1e-8))).abs()
    return 1.0 - cosine.squeeze(0) + 0.25 * norm_gap


def _flat_grads(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in grads])


@torch.no_grad()
def _distribution_target(message: torch.Tensor, labels: torch.Tensor,
                         cfg: Config, seed: int) -> dict:
    """Cache CaT-style nonlinear class moments, using only current raw delta."""
    if min(cfg.distribution_hidden, cfg.distribution_bank_size,
           cfg.distribution_chunk_size) < 1:
        raise ValueError("distribution bank dimensions and chunk size must be positive")
    classes, inverse, counts = torch.unique(labels.to(message.device).long(),
        sorted=True, return_inverse=True, return_counts=True)
    if len(labels) != len(message) or not len(classes):
        raise ValueError("distribution target needs nonempty aligned labels")
    generator = torch.Generator(device=message.device).manual_seed(int(seed))
    bound = math.sqrt(6.0 / (message.shape[1] + cfg.distribution_hidden))
    weights = torch.empty((cfg.distribution_bank_size, message.shape[1],
        cfg.distribution_hidden), device=message.device, dtype=message.dtype)
    weights.uniform_(-bound, bound, generator=generator)
    means = []
    for weight in weights:
        total = message.new_zeros((len(classes), cfg.distribution_hidden))
        for start in range(0, len(message), cfg.distribution_chunk_size):
            stop = start + cfg.distribution_chunk_size
            features = F.normalize(F.relu(message[start:stop] @ weight), dim=1)
            total.index_add_(0, inverse[start:stop], features)
        means.append(total / counts[:, None])
    return {"weights": weights, "means": torch.stack(means), "classes": classes}


def _distribution_loss(message: torch.Tensor, labels: torch.Tensor,
                       target: dict) -> torch.Tensor:
    """CaT class-sum MMD, averaged over a fixed random Encoder bank."""
    classes, inverse, counts = torch.unique(labels.to(message.device).long(),
        sorted=True, return_inverse=True, return_counts=True)
    if len(labels) != len(message) or not torch.equal(classes, target["classes"]):
        raise ValueError("real/synthetic distribution classes must match")
    losses = []
    for weight, real_mean in zip(target["weights"], target["means"]):
        features = F.normalize(F.relu(message @ weight), dim=1)
        total = features.new_zeros((len(classes), features.shape[1]))
        total = total.index_add(0, inverse, features)
        losses.append((total / counts[:, None] - real_mean).square().sum())
    return torch.stack(losses).mean()


def _materialize_edges(q: torch.Tensor, delta: dict, parent: dict | None,
                       boundary: torch.Tensor, cfg: Config,
                       components: tuple[torch.Tensor, torch.Tensor] | None = None,
                       x_new: torch.Tensor | None = None,
                       target_message: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, dict]:
    if components is None:
        _, new_new, cross = _soft_topology(q, delta, parent, boundary)
    else:
        new_new, cross = components
    new_n = q.shape[1]
    old_n = 0 if parent is None else len(parent["x"])
    raw_edges = delta.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    raw_count = int(_unique_undirected_index(
        raw_edges, len(delta["x"]), q.device).shape[1] // 2)
    target_new = min(int(new_n * (new_n - 1) // 2), int(round(
        float(cfg.topology_edge_factor) * raw_count * max(1, new_n) /
        max(1, len(delta["x"])))))
    edges = []
    actual_new = 0
    if target_new > 0:
        tri = torch.triu_indices(new_n, new_n, offset=1, device=q.device)
        values = new_new[tri[0], tri[1]]
        chosen = torch.topk(values, k=min(target_new, len(values))).indices
        chosen = chosen[values[chosen] > 0]
        selected = tri[:, chosen] + old_n
        edges.extend((selected, selected.flip(0)))
        actual_new = len(chosen)
    # The raw boundary graph is compressed just like the intra-delta graph:
    # preserve the average number of boundary incidences per new slot rather
    # than copying every raw boundary edge into the much smaller memory.
    boundary_rows = int((boundary.sum(1) > 0).sum().item())
    target_cross = min(old_n * new_n, int(round(
        float(cfg.boundary_edge_factor) * boundary_rows * new_n /
        max(1, len(delta["x"])))) )
    if target_cross > 0 and old_n:
        values = cross.reshape(-1)
        chosen = torch.topk(values, k=min(target_cross, len(values))).indices
        chosen = chosen[values[chosen] > 0]
        selected = torch.stack((chosen % old_n, old_n + chosen // old_n))
        edges.extend((selected, selected.flip(0)))
    edge_index = (torch.cat(edges, dim=1).long().cpu().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    refinement = {}
    if cfg.topology_refine_swaps:
        if target_cross or boundary.numel() and bool(boundary.any()):
            raise ValueError("degree-preserving diagnostic requires no boundary edges")
        if x_new is None or target_message is None:
            raise ValueError("propagation refinement needs current features and target")
        from unified_q_sparsification import refine_degree_preserving
        local_edges, refinement = refine_degree_preserving(
            edge_index - old_n, new_new, x_new, target_message,
            int(cfg.topology_refine_swaps))
        edge_index = local_edges + old_n
    return edge_index, {
        "raw_intra_directed_edges": int(raw_edges.shape[1]),
        "raw_boundary_rows_with_boundary": boundary_rows,
        "new_new_edges": int(actual_new),
        "new_new_edge_budget": int(target_new),
        "new_old_edges": int(edge_index.shape[1] // 2 - actual_new),
        "soft_new_new_mass": float(new_new.sum().detach().cpu()),
        "soft_cross_mass": float(cross.sum().detach().cpu()),
        **refinement,
    }


def _hard_adjacency(edges: torch.Tensor, parent: dict | None,
                    n: int, device: str | torch.device,
                    sparse_edgefree: bool = False) -> torch.Tensor:
    if parent is not None:
        edges = torch.cat((parent.get("edge_index", torch.empty((2, 0), dtype=torch.long)).cpu(),
                           edges.cpu()), dim=1)
    adjacency = P.normalized_adjacency(edges, n, str(device))
    if sparse_edgefree and not edges.numel():
        return adjacency
    return adjacency.to_dense()


def _text_rows(q, raw_text, raw_labels, labels, cfg):
    if cfg.text_materialization == "q_max":
        return q.argmax(dim=0)
    if isinstance(q, ClassBlocks):
        raise ValueError("class-block storage currently supports q_max text only")
    if cfg.text_materialization == "q_coverage":
        return q_text_coverage(q, raw_text, raw_labels, labels.to(q.device))
    return q_text_medoid(q, raw_text, raw_labels, labels.to(q.device))


def _materialize_text(delta: dict, q: torch.Tensor, labels: torch.Tensor,
                      encoder, cfg: Config, candidate_texts: list[str] | None = None,
                      selected_rows: torch.Tensor | None = None):
    raw_texts = delta["texts"]
    raw_ids_global = delta["ids"].long().tolist()
    raw_labels = delta["y"].long()
    texts, source_ids, audits = [], [], []
    for slot, cls in enumerate(labels.tolist()):
        candidates = torch.where(raw_labels == int(cls))[0]
        scores = q[candidates, slot]
        # Match argmax's first-index tie rule, especially when a class has
        # one slot and Sinkhorn assigns exactly equal mass to many sources.
        order = candidates[torch.argsort(scores, descending=True, stable=True)]
        take = order[:max(1, min(int(cfg.materialize_top_k), len(order)))]
        if selected_rows is not None:
            take = selected_rows[slot:slot + 1].long().cpu()
            if len(take) != 1 or int(take[0]) < 0 or int(take[0]) >= len(raw_labels) or int(raw_labels[take[0]]) != int(cls):
                raise ValueError("hard text projection escaped the current class")
        source_texts = candidate_texts if candidate_texts is not None else raw_texts
        parts = [_append_text(source_texts[int(i)], cfg.text_chars) for i in take.tolist()]
        text = _append_text(" ".join(parts), cfg.text_chars)
        texts.append(text)
        source_ids.append([int(raw_ids_global[int(i)]) for i in take.tolist()])
        audits.append({"slot": int(slot), "class": int(cls),
                       "chosen_source_ids": source_ids[-1],
                       "weights": [float(q[int(i), slot].detach().cpu())
                                   for i in take.tolist()],
                       "text_length": len(text)})
    encoded = _norm(torch.as_tensor(encoder.encode(texts)).float().cpu())
    return texts, source_ids, encoded, audits


def _support_ids(delta: dict, q: torch.Tensor, labels: torch.Tensor,
                 cfg: Config) -> list[list[int]]:
    hard = q.argmax(dim=1).detach().cpu()
    groups: list[list[int]] = [[] for _ in range(len(labels))]
    ids = delta["ids"].long().tolist()
    for raw_row, slot in enumerate(hard.tolist()):
        groups[int(slot)].append(int(ids[raw_row]))
    if int(cfg.support_cap) > 0:
        groups = [values[:int(cfg.support_cap)] for values in groups]
    return groups


def _append(parent: dict | None, block: dict) -> dict:
    if parent is None:
        return block
    # Unified-Q materializes stage edges in the complete parent+new coordinate
    # system so that cross-stage edges can be retained.  Do not shift them a
    # second time here; only the old parent edge list is already in place.
    block_edges = block["edge_index"]
    parent_edges = parent.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    return {
        "x": torch.cat((block.get("revised_parent_x", parent["x"]).cpu(), block["x"].cpu())),
        "text_emb": torch.cat((parent["text_emb"].cpu(), block["text_emb"].cpu())),
        "labels": torch.cat((parent["labels"].cpu(), block["labels"].cpu())),
        "mass": torch.cat((parent["mass"].cpu(), block["mass"].cpu())),
        "depth": torch.cat((parent["depth"].cpu(), block["depth"].cpu())),
        "stage": torch.cat((parent["stage"].cpu(), block["stage"].cpu())),
        "texts": list(parent["texts"]) + list(block["texts"]),
        "edge_index": torch.cat((parent_edges.cpu(), block_edges.cpu()), dim=1),
        "support_ids": list(parent.get("support_ids", [])) + list(block["support_ids"]),
        "text_source_ids": list(parent.get("text_source_ids", [])) + list(block["text_source_ids"]),
    }


def optimize_stage(delta: dict, parent: dict | None, labels: torch.Tensor,
                   encoder, prompt_emb: torch.Tensor, cfg: Config,
                   device: str, cat_anchor_x: torch.Tensor | None = None,
                   use_boundary: bool = True,
                   use_topology: bool = True,
                   diagnostic_callback=None, optimization_callback=None) -> tuple[dict, dict]:
    """Jointly optimize Q and an X residual, then derive text/topology from Q."""
    device = str(device)
    if cfg.steps < 0 or cfg.sinkhorn_steps < 1:
        raise ValueError("optimization steps must be nonnegative and Sinkhorn steps positive")
    if not math.isfinite(cfg.lambda_distribution) or cfg.lambda_distribution < 0:
        raise ValueError("distribution weight must be finite and nonnegative")
    if cfg.lambda_distribution and cat_anchor_x is not None:
        raise ValueError("in-loop distribution matching does not use a pre-trained CaT anchor")
    if not math.isfinite(cfg.lambda_joint_distribution) or cfg.lambda_joint_distribution < 0:
        raise ValueError("joint distribution weight must be finite and nonnegative")
    if cfg.lambda_joint_distribution and (use_topology or cat_anchor_x is not None):
        raise ValueError("joint distribution control requires edge-free, no pretrained anchor")
    if cfg.joint_distribution_mode not in ("paired", "marginal", "shuffled"):
        raise ValueError("unknown joint distribution mode")
    if cfg.materialize_top_k != 1:
        raise ValueError("hard-consistent optimization currently requires top-1 text")
    if cfg.text_materialization not in ("q_max", "q_text_medoid", "q_coverage"):
        raise ValueError("unknown hard text projection")
    if cfg.feature_pooling not in ("raw", "message"):
        raise ValueError("unknown feature pooling space")
    if cfg.feature_pooling == "message" and (use_topology or cat_anchor_x is not None):
        raise ValueError("message pooling is an edge-free, no-pretrained-anchor control")
    if cfg.text_assignment not in ("learned", "frozen", "detached") or cfg.topology_assignment not in ("learned", "frozen"):
        raise ValueError("unknown assignment ablation")
    if cfg.optimization not in ("joint", "alternating"):
        raise ValueError("unknown optimization schedule")
    if cfg.propagation_metric not in ("dimension_mean", "cosine"):
        raise ValueError("unknown propagation metric")
    if cfg.assignment_storage not in ("dense", "class_blocks"):
        raise ValueError("unknown assignment storage")
    blocked = cfg.assignment_storage == "class_blocks"
    if blocked and (use_topology or use_boundary or cfg.random_q or
                    cfg.text_materialization != "q_max" or cfg.lambda_joint_distribution):
        raise ValueError("class-block storage is restricted to the primary edge-free objective")
    old_n = 0 if parent is None else len(parent["x"])
    labels = labels.long().cpu()
    raw_labels = delta["y"].long().to(device)
    candidate_texts = [_append_text(text, cfg.text_chars) for text in delta["texts"]]
    candidate_encoded = _norm(torch.as_tensor(encoder.encode(candidate_texts)).float().to(device))
    delta = dict(delta)
    # Use exactly the text object that can later be materialized.  This avoids
    # optimizing a stale cache embedding and evaluating a re-encoded string.
    delta["text_emb"] = candidate_encoded.detach().cpu()
    delta["_raw_adjacency"] = _sparse_adjacency(
        delta.get("edge_index", torch.empty((2, 0), dtype=torch.long)),
        len(delta["x"]), device)
    raw_x = delta["x"].float().to(device)
    raw_text = delta["text_emb"].float().to(device)
    boundary = (_boundary_matrix(delta, parent, device) if use_boundary
                else torch.zeros((len(delta["x"]), 0 if parent is None else len(parent["x"])), device=device))
    logits0, q0, init_audit = _initial_logits(delta, parent, labels, cfg, device, boundary)
    if cfg.random_q:
        # Literal random-Q control: retain only class legality and balanced
        # Sinkhorn capacity; discard the structured feature/message/text costs.
        random_logits = torch.full_like(logits0, -1e4)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg.seed) + 1009 * int(delta["stage"]) + 424242)
        for cls in torch.unique(raw_labels, sorted=True).tolist():
            raw_ids = torch.where(raw_labels == int(cls))[0]
            slot_ids = torch.where(labels.to(device) == int(cls))[0]
            random_logits[raw_ids[:, None], slot_ids[None, :]] = torch.randn(
                (len(raw_ids), len(slot_ids)), generator=generator, device=device)
        q0 = _masked_sinkhorn(random_logits, raw_labels, labels.to(device),
                              cfg.temperature, cfg.sinkhorn_steps,
                              cfg.sinkhorn_tolerance, cfg.sinkhorn_max_steps)
        logits0 = random_logits
        init_audit = dict(init_audit, random_q=True,
                          random_q_seed=int(cfg.seed) + 1009 * int(delta["stage"]) + 424242,
                          structured_costs_ignored=True)
    q_logits = (logits0.map(lambda b: torch.nn.Parameter(b.clone())) if blocked
                else torch.nn.Parameter(logits0.clone()))
    q_params = q_logits.parameters() if blocked else [q_logits]
    pooling_features = raw_x
    if cfg.feature_pooling == "message":
        ref_adj, ref_x = _reference_graph(delta, parent, boundary, device)
        target_message = _adjacency_mm(ref_adj, ref_x).detach()
        pooling_features = target_message[-len(raw_labels):]
    pooled0 = (q0.T @ pooling_features /
               q0.sum(0).clamp_min(1e-6)[:, None])
    residual = torch.nn.Parameter(torch.zeros_like(pooled0))
    old_base = (parent["x"].float().to(device) if parent is not None
                else torch.empty((0, raw_x.shape[1]), device=device))
    old_residual = torch.nn.Parameter(torch.zeros_like(old_base))
    old_residual.requires_grad_(bool(old_n and cfg.old_update_radius != 0))
    params = q_params + [residual]
    param_groups = [
        {"params": q_params, "lr": float(cfg.lr_q)},
        {"params": [residual], "lr": float(cfg.lr_x)},
    ]
    if old_n and cfg.old_update_radius != 0:
        params.append(old_residual)
        param_groups.append({"params": [old_residual], "lr": float(cfg.lr_x)})
    optimizer = torch.optim.Adam(param_groups)
    sinkhorn_audit = []

    def assignment():
        return _masked_sinkhorn(q_logits, raw_labels, labels.to(device),
            cfg.temperature, cfg.sinkhorn_steps, cfg.sinkhorn_tolerance,
            cfg.sinkhorn_max_steps, sinkhorn_audit)

    if cfg.feature_pooling == "raw":
        ref_adj, ref_x = _reference_graph(delta, parent, boundary, device)
    ref_text = (torch.cat((parent["text_emb"].float().to(device),
                           delta["text_emb"].float().to(device)))
                if parent is not None else delta["text_emb"].float().to(device))
    ref_labels = (torch.cat((parent["labels"].long().to(device),
                             raw_labels)) if parent is not None else raw_labels)
    ref_groups = torch.cat((torch.zeros(old_n), torch.ones(len(raw_labels)))).long().to(device)
    syn_groups = torch.cat((torch.zeros(old_n), torch.ones(len(labels)))).long().to(device)
    syn_labels = (torch.cat((parent["labels"].long().to(device), labels.to(device)))
                  if parent is not None else labels.to(device))
    ref_weights, syn_weights = _proxy_weights(
        ref_labels, ref_groups, syn_labels, syn_groups, cfg.loss_weighting,
        raw_x.device, raw_x.dtype)
    proxy_weighting = "node_mean" if cfg.loss_weighting == "student_mass" else cfg.loss_weighting
    if cfg.feature_pooling == "raw":
        target_message = _adjacency_mm(ref_adj, ref_x).detach()
    distribution_targets = []
    if cfg.lambda_distribution:
        # Match the same one-hop representation on both sides.  With an
        # edge-free synthetic graph this reduces to CaT's propagated-raw/X loss.
        bank_seed = int(cfg.seed) + 1009 * int(delta["stage"]) + 999983
        distribution_targets = [_distribution_target(
            target_message[-len(raw_labels):], raw_labels, cfg, bank_seed + offset)
            for offset in (0, 1)]
    joint_target = None
    if cfg.lambda_joint_distribution:
        joint_target = J.target(target_message[-len(raw_labels):], raw_text, raw_labels,
            hidden=cfg.joint_distribution_hidden, banks=cfg.joint_distribution_banks,
            chunk=cfg.joint_distribution_chunk, seed=int(cfg.seed) + 1009 * int(delta["stage"]) + 3141593)
    prompt_emb = prompt_emb.float().to(device)
    real_grads = []
    models = []
    for seed in (cfg.proxy_seeds if cfg.lambda_gradient else ()):
        torch.manual_seed(int(cfg.seed) + int(seed))
        model = P.GraphTextPromptStudent(int(ref_x.shape[1]), hidden=int(cfg.proxy_hidden),
                                         out=int(cfg.proxy_out), text_dim=int(ref_text.shape[1])).to(device)
        real_memory = {"x": ref_x, "text_emb": ref_text, "labels": ref_labels}
        real_loss = _loss(model, real_memory, ref_adj, prompt_emb,
                          group_ids=ref_groups, loss_weighting=proxy_weighting,
                          chunk_size=cfg.pair_chunk_size, pair_weight=cfg.pair_weight,
                          sample_weights=(ref_weights if cfg.loss_weighting == "student_mass" else None))
        real_grads.append(tuple(g.detach() for g in torch.autograd.grad(
            real_loss, tuple(model.parameters()), create_graph=False)))
        models.append(model)

    trace = []
    frozen_text = raw_text[_text_rows(q0, raw_text, raw_labels, labels, cfg)].detach()
    joint_initial = (float(J.loss(pooled0.detach(), frozen_text, labels.to(device),
                                 joint_target, cfg.joint_distribution_mode)) if joint_target is not None else None)
    initial_edges, initial_topology_audit = (None, None)
    initial_hard = None
    if use_topology:
        with torch.no_grad():
            initial_target = _norm(q0.T @ target_message[-len(raw_labels):] / q0.sum(0)[:, None])
            initial_edges, initial_topology_audit = _materialize_edges(
                q0, delta, parent, boundary, cfg, x_new=pooled0, target_message=initial_target)
        initial_hard = (_hard_adjacency(
                        initial_edges, parent, old_n + len(labels), device,
                        sparse_edgefree=cfg.sparse_edgefree_forward)
                    if initial_edges is not None else None)
    distribution_initial = []
    if distribution_targets:
        with torch.no_grad():
            initial_x = torch.cat((old_base, pooled0))
            initial_adj = (initial_hard if initial_hard is not None else
                _hard_adjacency(torch.empty((2, 0), dtype=torch.long), parent,
                                len(initial_x), device,
                                sparse_edgefree=cfg.sparse_edgefree_forward))
            initial_message = _adjacency_mm(initial_adj, initial_x)[-len(labels):]
            distribution_initial = [float(_distribution_loss(
                initial_message, labels, target)) for target in distribution_targets]
    q_update_count = residual_update_count = 0
    old_update_count = 0
    for step in range(int(cfg.steps)):
        q = assignment()
        mass = q.sum(0).clamp_min(1e-6)
        x_new = q.T @ pooling_features / mass[:, None] + residual
        soft_text_new = _norm(q.T @ raw_text / mass[:, None])
        if cfg.text_assignment == "frozen":
            text_new = frozen_text
        elif cfg.text_assignment == "detached":
            # Keep the current hard choice but remove only the text-to-Q path.
            text_new = raw_text.index_select(0, _text_rows(q, raw_text, raw_labels, labels, cfg)).detach()
        elif cfg.straight_through_text:
            hard_ids = _text_rows(q, raw_text, raw_labels, labels, cfg)
            hard_text_new = raw_text.index_select(0, hard_ids)
            text_new = hard_text_new + (soft_text_new - soft_text_new.detach())
        else:
            text_new = soft_text_new
        if parent is None:
            full_x, full_text, full_labels = x_new, text_new, labels.to(device)
        else:
            full_x = torch.cat((old_base + old_residual, x_new))
            full_text = torch.cat((parent["text_emb"].float().to(device), text_new))
            full_labels = torch.cat((parent["labels"].long().to(device), labels.to(device)))
        if use_topology and cfg.topology_assignment == "frozen":
            syn_adj = initial_hard
        elif use_topology:
            soft_adj, new_new, cross = _soft_topology(q, delta, parent, boundary)
            if cfg.hard_topology_forward:
                hard_edges, _ = _materialize_edges(q.detach(), delta, parent, boundary, cfg,
                    (new_new.detach(), cross.detach()), x_new=x_new.detach(),
                    target_message=_norm(q.detach().T @ target_message[-len(raw_labels):] / mass.detach()[:, None]))
                hard_adj = _hard_adjacency(
                    hard_edges, parent, len(full_x), device,
                    sparse_edgefree=cfg.sparse_edgefree_forward)
                syn_adj = hard_adj + (soft_adj - soft_adj.detach())
            else:
                syn_adj = soft_adj
        else:
            syn_adj = _hard_adjacency(
                torch.empty((2, 0), dtype=torch.long), parent, len(full_x), device,
                sparse_edgefree=cfg.sparse_edgefree_forward)
        synthetic = {"x": full_x, "text_emb": full_text, "labels": full_labels}
        terms = []
        for model, real_grad in zip(models, real_grads):
            syn_loss = _loss(model, synthetic, syn_adj, prompt_emb,
                             group_ids=syn_groups, loss_weighting=proxy_weighting,
                             chunk_size=cfg.pair_chunk_size, pair_weight=cfg.pair_weight)
            syn_grad = torch.autograd.grad(syn_loss, tuple(model.parameters()),
                                           create_graph=True, retain_graph=True)
            terms.append(_gradient_distance(_flat_grads(real_grad), _flat_grads(syn_grad)))
        gm = torch.stack(terms).mean() if terms else x_new.sum() * 0.0
        target_new = _norm((q.T @ target_message[-len(raw_labels):]) / mass[:, None])
        message_new = _adjacency_mm(syn_adj, full_x)[-len(labels):]
        synthetic_message = _norm(message_new)
        distribution = (_distribution_loss(message_new, labels, distribution_targets[0])
                        if distribution_targets else gm * 0.0)
        joint_distribution = (J.loss(message_new, text_new, labels.to(device), joint_target,
                                    cfg.joint_distribution_mode) if joint_target is not None else gm * 0.0)
        propagation = ((target_new - synthetic_message).square().mean()
                       if cfg.propagation_metric == "dimension_mean" else
                       (1 - (target_new * synthetic_message).sum(1)).mean())
        q_anchor = q.anchor(q0) if blocked else (q - q0.detach()).square().mean()
        x_anchor = (residual / pooled0.norm(dim=1, keepdim=True).clamp_min(1e-3)).square().mean()
        old_anchor = ((old_residual / old_base.norm(dim=1, keepdim=True).clamp_min(1e-3)).square().sum(1).mean()
                      if old_n and cfg.old_update_radius != 0 else gm * 0.0)
        cat_anchor = gm * 0.0
        if cat_anchor_x is not None:
            cat_anchor = ((x_new - cat_anchor_x.to(device).float()) /
                          cat_anchor_x.to(device).float().norm(dim=1, keepdim=True).clamp_min(1e-3)).square().mean()
        loss = (float(cfg.lambda_gradient) * gm
                + float(cfg.lambda_distribution) * distribution
                + float(cfg.lambda_joint_distribution) * joint_distribution
                + float(cfg.lambda_propagation) * propagation
                + float(cfg.lambda_q_anchor) * q_anchor
                + float(cfg.lambda_x_anchor) * x_anchor
                + float(cfg.lambda_cat_anchor) * cat_anchor
                + float(cfg.lambda_old_anchor) * old_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite unified-Q loss")
        if optimization_callback is not None:
            # Observe the live graph before backward without changing RNG or .grad.
            py_state, np_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                optimization_callback(dict(step=step, config=cfg, q=q, q0=q0,
                    q_logits=q_logits, residual=residual, x_new=x_new,
                    text_new=text_new, soft_text=soft_text_new, raw_text=raw_text,
                    raw_labels=raw_labels, labels=labels.to(device),
                    gm=gm, distribution=distribution, propagation=propagation,
                    joint_distribution=joint_distribution,
                    loss=loss, delta=delta))
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
        optimizer.zero_grad(set_to_none=True)
        joint_q_norm = None
        if joint_target is not None and (step == 0 or step + 1 == int(cfg.steps)):
            grad_joint = torch.autograd.grad(joint_distribution, q_logits, retain_graph=True)[0]
            joint_q_norm = float(grad_joint.detach().norm())
            del grad_joint
        loss.backward()
        q_grad_norm = (float(torch.stack([p.grad.norm() for p in q_params if p.grad is not None]).norm())
                       if any(p.grad is not None for p in q_params) else 0.0)
        residual_grad_norm = float(residual.grad.norm().detach()) if residual.grad is not None else 0.0
        update_q = cfg.train_q and (cfg.optimization == "joint" or step % 2 == 0)
        update_r = cfg.train_residual and (cfg.optimization == "joint" or step % 2 == 1)
        if not update_q:
            for parameter in q_params:
                parameter.grad = None
        if not update_r:
            residual.grad = None
        q_update_count += int(update_q)
        residual_update_count += int(update_r)
        old_update_count += int(old_n > 0 and cfg.old_update_radius != 0)
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            q_now = assignment()
            pooled = q_now.T @ pooling_features / q_now.sum(0).clamp_min(1e-6)[:, None]
            limit = float(cfg.trust_radius) * pooled.norm(dim=1, keepdim=True).clamp_min(1e-3)
            residual.copy_((residual * (limit / residual.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)))
            if old_n and cfg.old_update_radius > 0:
                old_limit = cfg.old_update_radius * old_base.norm(dim=1, keepdim=True).clamp_min(1e-3)
                old_residual.mul_((old_limit / old_residual.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1))
        if step == 0 or step + 1 == int(cfg.steps) or (step + 1) % 10 == 0:
            trace.append({"step": int(step + 1), "loss": float(loss.detach().cpu()),
                          "gradient_match": float(gm.detach().cpu()),
                          "distribution_match": float(distribution.detach().cpu()),
                          "joint_distribution_match": float(joint_distribution.detach().cpu()),
                          "joint_distribution_q_gradient_norm": joint_q_norm,
                          "propagation": float(propagation.detach().cpu()),
                          "propagation_metric": cfg.propagation_metric,
                          "old_anchor": float(old_anchor.detach().cpu()),
                          "q_anchor": float(q_anchor.detach().cpu()),
                          "q_gradient_norm": q_grad_norm,
                          "residual_gradient_norm": residual_grad_norm,
                          "q_update": update_q, "residual_update": update_r,
                          "cat_anchor": float(cat_anchor.detach().cpu())})

    with torch.no_grad():
        q = assignment()
        mass = q.sum(0).clamp_min(1e-6)
        x_new = q.T @ pooling_features / mass[:, None] + residual
        if use_topology and cfg.topology_assignment == "frozen":
            edge_index, topology_audit = initial_edges, initial_topology_audit
        elif use_topology:
            edge_index, topology_audit = _materialize_edges(q, delta, parent,
                boundary, cfg, x_new=x_new,
                target_message=_norm(q.T @ target_message[-len(raw_labels):] / mass[:, None]))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            topology_audit = {"mode": "edge_free"}
    text_q = q0 if cfg.text_assignment == "frozen" else q
    final_text_rows = _text_rows(text_q, raw_text, raw_labels, labels, cfg)
    texts, source_ids, hard_text, text_audit = _materialize_text(
        delta, text_q.detach().cpu(), labels, encoder, cfg, candidate_texts,
        selected_rows=(None if cfg.text_materialization == "q_max" else final_text_rows.cpu()))
    expected_text = raw_text[final_text_rows].detach().cpu()
    text_error = float((expected_text - hard_text).abs().max())
    if text_error > 1e-4:
        raise AssertionError(f"materialized text differs from hard forward: {text_error}")
    distribution_final = []
    if distribution_targets:
        with torch.no_grad():
            final_x = torch.cat((old_base + old_residual, x_new))
            final_adj = _hard_adjacency(
                edge_index, parent, len(final_x), device,
                sparse_edgefree=cfg.sparse_edgefree_forward)
            final_message = _adjacency_mm(final_adj, final_x)[-len(labels):]
            distribution_final = [float(_distribution_loss(
                final_message, labels, target)) for target in distribution_targets]
    block = {
        "x": x_new.detach().cpu().float(),
        "text_emb": hard_text.detach().cpu().float(),
        "labels": labels.clone(),
        "mass": mass.detach().cpu().float(),
        "depth": torch.zeros(len(labels), dtype=torch.long),
        "stage": torch.full((len(labels),), int(delta["stage"]), dtype=torch.long),
        "texts": texts,
        "edge_index": edge_index.cpu().long(),
        "support_ids": _support_ids(delta, q.detach().cpu(), labels, cfg),
        "text_source_ids": source_ids,
    }
    if old_n and cfg.old_update_radius != 0:
        block["revised_parent_x"] = (old_base + old_residual).detach().cpu()
    capacity = torch.zeros_like(mass)
    for cls in torch.unique(raw_labels):
        slots = labels.to(device) == cls
        capacity[slots] = (raw_labels == cls).sum() / slots.sum()
    soft_text_final = _norm(q.T @ raw_text / mass[:, None])
    hard_text_final = raw_text.index_select(0, final_text_rows)
    coverage_values = []
    for cls in torch.unique(labels.to(device), sorted=True):
        raw_cls = torch.where(raw_labels == cls)[0]
        slot_cls = torch.where(labels.to(device) == cls)[0]
        if len(raw_cls) and len(slot_cls):
            coverage_values.append(float((raw_text.index_select(0, raw_cls) @
                hard_text_final.index_select(0, slot_cls).T).max(1).values.mean().cpu()))
    text_projection = {
        "rule": str(cfg.text_materialization),
        "selected_rows": [int(v) for v in final_text_rows.detach().cpu().tolist()],
        "unique_source_count": int(torch.unique(final_text_rows).numel()),
        "slot_count": int(len(final_text_rows)),
        "hard_soft_cosine": float((hard_text_final * soft_text_final).sum(1).mean().cpu()),
        "class_coverage_cosine": (float(sum(coverage_values) / len(coverage_values))
                                  if coverage_values else None),
    }
    audit = {
        "constructor": "unified_q_multimodal",
        "objective": "Q-derived X/text/topology with multimodal gradient and propagation matching",
        "historical_raw_reads": 0,
        "uses_boundary_sketch": bool(use_boundary and parent is not None),
        "uses_q_topology": bool(use_topology),
        "straight_through_text": bool(cfg.straight_through_text),
        "sparse_edgefree_forward": bool(cfg.sparse_edgefree_forward),
        "text_materialization_rule": cfg.text_materialization,
        "text_projection": text_projection,
        "feature_pooling": cfg.feature_pooling,
        "hard_topology_forward": bool(cfg.hard_topology_forward and use_topology),
        "q_initialization": "class-masked structured feature/message/text cost",
        "q_initialization_mode": str(cfg.init_mode),
        "loss_weighting": str(cfg.loss_weighting),
        "proxy_real_rows": int(len(ref_labels)),
        "proxy_synthetic_rows": int(len(syn_labels)),
        "real_new_loss_mass": float(ref_weights[old_n:].sum()),
        "synthetic_new_loss_mass": float(syn_weights[old_n:].sum()),
        "text_assignment": cfg.text_assignment,
        "topology_assignment": cfg.topology_assignment,
        "text_hard_forward_max_error": text_error,
        "unique_text_count": len(set(texts)),
        "unique_text_ratio": len(set(texts)) / max(1, len(texts)),
        "optimization_schedule": str(cfg.optimization),
        "q_update_count": int(q_update_count),
        "residual_update_count": int(residual_update_count),
        "q_cost": init_audit,
        "q_row_sum_max_error": float((q.sum(1) - 1.0).abs().max().cpu()),
        "q_column_mass_min": float(q.sum(0).min().cpu()),
        "q_column_mass_max": float(q.sum(0).max().cpu()),
        "q_column_relative_error": float(((mass - capacity) / capacity).abs().max()),
        "sinkhorn_max_iterations": max(v["iterations"] for v in sinkhorn_audit),
        "sinkhorn_max_row_error": max(v["row_error"] for v in sinkhorn_audit),
        "sinkhorn_max_column_relative_error": max(v["column_relative_error"] for v in sinkhorn_audit),
        "old_update_count": old_update_count,
        "old_x_relative_change_max": (float((old_residual.detach().norm(dim=1) /
            old_base.norm(dim=1).clamp_min(1e-3)).max()) if old_n else 0.0),
        "q_entropy": (float(q.entropy().cpu()) if blocked else
                      float((-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1).mean()).cpu())),
        "assignment_storage": cfg.assignment_storage,
        "assignment_parameter_entries": sum(p.numel() for p in q_params),
        "boundary_edges": int(boundary.sum().item()),
        "topology": topology_audit,
        "text_materialization": text_audit,
        "support_ids": block["support_ids"],
        "config": asdict(cfg),
        "cat_anchor_used": cat_anchor_x is not None,
        "distribution_matching": {
            "enabled": bool(distribution_targets),
            "target_scope": "current-delta rows of legal reference one-hop messages",
            "synthetic_scope": "current slots after materialized synthetic one-hop propagation",
            "warm_start": False,
            "reduction": "sum over classes/features, mean over random Encoder bank",
            "initial_train": distribution_initial[0] if distribution_initial else None,
            "final_train": distribution_final[0] if distribution_final else None,
            "initial_holdout_bank": distribution_initial[1] if distribution_initial else None,
            "final_holdout_bank": distribution_final[1] if distribution_final else None,
            "holdout_used_for_optimization": False,
        },
        "x_residual_norm": float(residual.detach().norm().cpu()),
        "joint_pair_distribution": {
            "enabled": joint_target is not None,
            "mode": cfg.joint_distribution_mode,
            "kernel": "product of fixed normalized ReLU random-feature kernels",
            "reference_scope": "current-delta legal one-hop message and exactly materializable text candidates",
            "shuffling_scope": "new target statistic only; candidate pool, initialization and source IDs unchanged",
            "initial_loss": joint_initial,
            "final_loss": (float(J.loss(x_new, hard_text.to(device), labels.to(device), joint_target,
                                        cfg.joint_distribution_mode)) if joint_target is not None else None),
            "reference_pair_shuffle_mmd": joint_target["pair_shuffle_mmd"] if joint_target is not None else None,
            "finite_feature_kernel_not_universal": True,
        },
        "x_residual_ratio_max": float((residual.detach().norm(dim=1) /
            (x_new.detach() - residual.detach()).norm(dim=1).clamp_min(1e-3)).max()),
        "implementation": "unified-q-consistent-balanced-v2",
        "reference_adjacency_layout": str(ref_adj.layout),
        "topology_operator": "binary simple undirected + one self loop; Student normalization",
        "trace": trace,
    }
    if diagnostic_callback is not None:
        # Read-only diagnostics consume Q before it is released; never retain
        # dense assignments in the persistent condensation memory.  A callback
        # may construct temporary tensors/models, so preserve every process RNG
        # stream around it; otherwise a diagnostic can silently change the
        # following stage or Student replay while leaving the method unchanged.
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with torch.no_grad():
                diagnostic_callback(dict(q=q.detach(), q0=q0.detach(), delta=delta,
                    boundary=boundary, target_message=target_message[-len(raw_labels):],
                    block=block, parent=parent, config=cfg))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
    return block, audit


def append_memory(parent: dict | None, block: dict) -> dict:
    return _append(parent, block)

