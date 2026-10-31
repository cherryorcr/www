"""CaT initialization with one-step multimodal gradient matching.

This is a small, independent construction module.  The default branch keeps
the topology edge-free/self-loop-only for controlled comparison.  An optional
fixed topology branch coarsens the current-delta graph through a balanced
raw-node-to-slot assignment, so the same propagation operator can be used by
the real and synthetic sides of gradient matching.

The constructor may read only the current raw delta and the already
materialized parent memory.  Text strings are represented during optimization
by differentiable routing over current-delta text embeddings.  The final
memory stores real, re-encoded strings selected from that candidate bank.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F

import graphtext_prompt_student as P


@dataclass(frozen=True)
class Config:
    proxy_hidden: int = 128
    proxy_out: int = 128
    proxy_seeds: tuple[int, ...] = (2711,)
    gradient_matching_mode: str = "blockwise"
    steps: int = 40
    lr_x: float = 5e-3
    lr_route: float = 5e-2
    temperature: float = 0.20
    lambda_pair: float = 0.5
    lambda_prompt: float = 1.0
    lambda_margin: float = 0.1
    prompt_margin: float = 0.20
    lambda_cat_anchor: float = 0.5
    lambda_old_anchor: float = 1.0
    lambda_route_anchor: float = 1.0
    lambda_entropy: float = 1e-3
    new_trust_radius: float = 0.05
    old_trust_radius: float = 0.02
    allow_old_update: bool = False
    update_x: bool = True
    update_route: bool = True
    candidate_cap: int = 0
    materialize_top_k: int = 1
    straight_through_route: bool = False
    text_chars: int = 320
    topology_mode: str = "none"
    topology_temperature: float = 0.20
    topology_sinkhorn_steps: int = 40
    topology_feature_weight: float = 1.0
    topology_text_weight: float = 0.25
    topology_refine_steps: int = 100
    topology_refine_lr: float = 1e-3
    topology_refine_hidden: int = 512
    topology_refine_anchor: float = 0.25
    em_updates: int = 0
    em_interval: int = 10
    em_message_weight: float = 1.0
    seed: int = 271006


def _norm(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1)


def _log_sinkhorn(scores: torch.Tensor, column_mass: torch.Tensor,
                  steps: int) -> torch.Tensor:
    """Return a row-stochastic, capacity-balanced assignment in log space."""
    if scores.ndim != 2 or not scores.numel():
        raise ValueError("Sinkhorn scores must be a non-empty matrix")
    rows, cols = scores.shape
    row_log_mass = torch.zeros(rows, dtype=scores.dtype, device=scores.device)
    col_log_mass = column_mass.to(scores.device, scores.dtype).clamp_min(1e-8).log()
    value = scores.float()
    for _ in range(max(1, int(steps))):
        value = value + row_log_mass[:, None] - torch.logsumexp(
            value, dim=1, keepdim=True)
        value = value + col_log_mass[None, :] - torch.logsumexp(
            value, dim=0, keepdim=True)
    return value.exp()


def _undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    """Canonicalize an edge list for average-degree accounting."""
    edges = edge_index.long().cpu()
    if not edges.numel():
        return torch.empty((2, 0), dtype=torch.long)
    edges = edges[:, edges[0] != edges[1]]
    low = torch.minimum(edges[0], edges[1])
    high = torch.maximum(edges[0], edges[1])
    keys = torch.unique(low * (int(high.max()) + 1) + high, sorted=True)
    base = int(high.max()) + 1
    return torch.stack((keys // base, keys % base))


def _aggregate_soft_edges(q: torch.Tensor, edge_index: torch.Tensor,
                          chunk_size: int = 4096) -> torch.Tensor:
    """Compute Q^T A Q without materializing the raw dense adjacency."""
    slots = q.shape[1]
    result = torch.zeros((slots, slots), dtype=q.dtype, device=q.device)
    edges = edge_index.long().to(q.device)
    if not edges.numel():
        return result
    for start in range(0, edges.shape[1], max(1, int(chunk_size))):
        part = edges[:, start:start + max(1, int(chunk_size))]
        left = q.index_select(0, part[0])
        right = q.index_select(0, part[1])
        result = result + torch.bmm(
            left.unsqueeze(2), right.unsqueeze(1)).sum(dim=0)
    result.fill_diagonal_(0.0)
    return 0.5 * (result + result.T)


def _class_mmd(real: torch.Tensor, synth: torch.Tensor,
               real_labels: torch.Tensor, synth_labels: torch.Tensor) -> torch.Tensor:
    values = []
    for cls in torch.unique(synth_labels, sorted=True):
        real_rows = real[real_labels == cls]
        synth_rows = synth[synth_labels == cls]
        if not len(real_rows) or not len(synth_rows):
            raise ValueError("topology-aware CaT class block is empty")
        values.append((real_rows.mean(0) - synth_rows.mean(0)).square().sum())
    return torch.stack(values).sum()


def _random_features(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _norm(F.relu(value @ weight))


def refine_topology_consistent_x(delta: dict, x0: torch.Tensor,
                                 labels: torch.Tensor, edge_index: torch.Tensor,
                                 q: torch.Tensor, cfg: Config,
                                 device: str) -> tuple[torch.Tensor, dict]:
    """Refine X so its synthetic propagation matches the raw propagated target.

    Original CaT optimizes X directly against already-propagated raw features.
    Once synthetic edges are materialized, that would double-propagate X in the
    Student.  This refinement instead optimizes A_s X_s against A_r X_r.
    """
    device = str(device)
    raw_x = delta["x"].float().to(device)
    raw_labels = delta["y"].long().to(device)
    raw_adj = P.normalized_adjacency(delta.get("edge_index",
                                                torch.empty((2, 0), dtype=torch.long)),
                                     len(raw_x), device)
    target = torch.sparse.mm(raw_adj, raw_x)
    syn_adj = P.normalized_adjacency(edge_index, len(x0), device)
    mass = q.sum(dim=0).clamp_min(1e-6).to(device)
    assigned = (q.T @ raw_x) / mass[:, None]
    # Preserve CaT's already useful solution; Q-weighted means are retained
    # only as a diagnostic initialization reference.
    synth = torch.nn.Parameter(x0.detach().to(device).clone())
    bound = math.sqrt(6.0 / (raw_x.shape[1] + int(cfg.topology_refine_hidden)))
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 33017)
    banks = [torch.empty((raw_x.shape[1], int(cfg.topology_refine_hidden)),
                         device=device).uniform_(-bound, bound, generator=gen)
             for _ in range(4)]
    optimizer = torch.optim.Adam([synth], lr=float(cfg.topology_refine_lr))
    trace = []
    for step in range(max(0, int(cfg.topology_refine_steps))):
        losses = []
        propagated_synth = torch.sparse.mm(syn_adj, synth)
        for weight in banks:
            losses.append(_class_mmd(
                _random_features(target, weight),
                _random_features(propagated_synth, weight),
                raw_labels, labels.to(device)))
        message_loss = torch.stack(losses).mean()
        scale = x0.to(device).norm(dim=1, keepdim=True).clamp_min(1e-3)
        anchor = (((synth - x0.to(device)) / scale) ** 2).mean()
        loss = message_loss + float(cfg.topology_refine_anchor) * anchor
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite topology-consistent CaT loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([synth], max_norm=5.0)
        optimizer.step()
        if step == 0 or step + 1 == int(cfg.topology_refine_steps) or \
                (step + 1) % 25 == 0:
            trace.append({"step": int(step + 1),
                          "random_space_loss": float(message_loss.detach().cpu()),
                          "anchor": float(anchor.detach().cpu())})
    return synth.detach(), {
        "steps": int(cfg.topology_refine_steps),
        "lr": float(cfg.topology_refine_lr),
        "init": "CaT X0; Q-weighted raw mean is diagnostic only",
        "anchor_weight": float(cfg.topology_refine_anchor),
        "distance_from_q_init": float((synth.detach() - assigned).norm().cpu()),
        "update_norm_from_cat_x0": float((synth.detach() - x0.to(device)).norm().cpu()),
        "trace": trace,
    }


def build_fixed_topology(delta: dict, x0: torch.Tensor,
                         labels: torch.Tensor, raw_text: torch.Tensor,
                         initial_text: torch.Tensor, cfg: Config,
                         device: str,
                         previous_edge_index: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Build a fixed synthetic topology from a balanced slot assignment.

    The assignment is class masked and uses only the current raw delta.  The
    returned edge list is materialized before gradient matching and is reused
    unchanged by construction and evaluation.  This is deliberately not a
    differentiable edge optimizer; EM-like reassignment is a later stage.
    """
    device = str(device)
    raw_x = delta["x"].float().to(device)
    raw_text = _norm(raw_text.float().to(device))
    raw_edges = delta.get("edge_index",
                          torch.empty((2, 0), dtype=torch.long))
    raw_adj = P.normalized_adjacency(raw_edges, len(raw_x), device)
    raw_message = torch.sparse.mm(raw_adj, raw_x)
    raw_message = _norm(raw_message)
    slot_x = _norm(x0.float().to(device))
    slot_text = _norm(initial_text.float().to(device))
    slot_message = None
    if previous_edge_index is not None and previous_edge_index.numel():
        previous_adj = P.normalized_adjacency(
            previous_edge_index, len(x0), device)
        slot_message = _norm(torch.sparse.mm(previous_adj, x0.float().to(device)))
    raw_labels = delta["y"].long().cpu()
    slot_labels = labels.long().cpu()
    q = torch.zeros((len(raw_x), len(x0)), dtype=torch.float32, device=device)
    audit_classes = []
    for cls in torch.unique(raw_labels, sorted=True).tolist():
        raw_ids = torch.where(raw_labels == int(cls))[0]
        slot_ids = torch.where(slot_labels == int(cls))[0]
        if not len(slot_ids):
            raise ValueError(f"no synthetic slot for current class {cls}")
        graph_score = raw_message.index_select(0, raw_ids.to(device)) @ \
            slot_x.index_select(0, slot_ids.to(device)).T
        text_score = raw_text.index_select(0, raw_ids.to(device)) @ \
            slot_text.index_select(0, slot_ids.to(device)).T
        scores = (float(cfg.topology_feature_weight) * graph_score
                  + float(cfg.topology_text_weight) * text_score)
        if slot_message is not None:
            message_score = raw_message.index_select(0, raw_ids.to(device)) @ \
                slot_message.index_select(0, slot_ids.to(device)).T
            scores = scores + float(cfg.em_message_weight) * message_score
        capacity = torch.full(
            (len(slot_ids),), float(len(raw_ids)) / float(len(slot_ids)),
            dtype=scores.dtype, device=device)
        local_q = _log_sinkhorn(
            scores / max(float(cfg.topology_temperature), 1e-6),
            capacity, cfg.topology_sinkhorn_steps)
        q[raw_ids.to(device)[:, None], slot_ids.to(device)[None, :]] = local_q
        audit_classes.append({
            "class": int(cls),
            "raw_rows": int(len(raw_ids)),
            "slots": int(len(slot_ids)),
            "column_mass_target": float(capacity[0].detach().cpu()),
        })

    soft_edges = _aggregate_soft_edges(q, raw_edges)
    pairs = _undirected_pairs(raw_edges)
    raw_edge_count = int(pairs.shape[1])
    raw_average_degree = (2.0 * raw_edge_count / len(raw_x)
                          if len(raw_x) else 0.0)
    # Preserve the average degree under compression instead of forcing a
    # synthetic edge when the current-delta induced graph contains almost no
    # observable edges.
    target_edges = min(raw_edge_count, int(round(
        raw_average_degree * len(x0) / 2.0))) if raw_edge_count else 0
    max_degree = max(0, len(x0) - 1)
    chosen = torch.zeros_like(soft_edges, dtype=torch.bool)
    if target_edges:
        values = soft_edges.clone()
        values.fill_diagonal_(-torch.inf)
        tri = torch.triu_indices(len(x0), len(x0), offset=1,
                                 device=soft_edges.device)
        tri_values = values[tri[0], tri[1]]
        take = min(int(target_edges), int(len(tri_values)))
        selected = torch.topk(tri_values, k=take).indices
        chosen[tri[0][selected], tri[1][selected]] = True
        chosen[tri[1][selected], tri[0][selected]] = True
    edge_index = torch.nonzero(chosen, as_tuple=False).T.contiguous().cpu().long()
    row_mass = q.sum(dim=1)
    col_mass = q.sum(dim=0)
    audit = {
        "topology_mode": "fixed_balanced_slot_coarsening",
        "assignment": ("class-masked Sinkhorn over propagated X, text cosine"
                       + (" and current synthetic message cosine"
                          if slot_message is not None else "")),
        "q_row_sum_max_error": float((row_mass - 1.0).abs().max().detach().cpu()),
        "q_column_mass_min": float(col_mass.min().detach().cpu()),
        "q_column_mass_max": float(col_mass.max().detach().cpu()),
        "q_mean_entropy": float((-(q.clamp_min(1e-12) *
                                    q.clamp_min(1e-12).log()).sum(1).mean())
                                 .detach().cpu()),
        "raw_undirected_edges": raw_edge_count,
        "raw_average_degree": float(raw_average_degree),
        "target_undirected_edges": int(target_edges),
        "target_average_degree": float(2.0 * target_edges / len(x0))
        if len(x0) else 0.0,
        "soft_edge_weight_sum": float(soft_edges.sum().detach().cpu()),
        "materialized_undirected_edges": int(edge_index.shape[1] // 2),
        "materialized_directed_edges": int(edge_index.shape[1]),
        "class_audit": audit_classes,
    }
    return edge_index, audit, q


def _flat_grads(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def _gradient_distance(real: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
    real_n = real.norm().clamp_min(1e-8)
    syn_n = synthetic.norm().clamp_min(1e-8)
    cosine = F.cosine_similarity(real.view(1, -1), synthetic.view(1, -1))
    norm_gap = (torch.log(real_n) - torch.log(syn_n)).abs()
    return 1.0 - cosine.squeeze(0) + 0.25 * norm_gap


def _gradient_distance_blockwise(model: P.GraphTextPromptStudent,
                                real: tuple[torch.Tensor, ...],
                                synthetic: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Match graph and text parameter blocks separately.

    A single flattened vector can let the much larger text projection block
    dominate the graph encoder direction.  Blockwise matching keeps both
    signals present without adding a new model component.
    """
    names = [name for name, _ in model.named_parameters()]
    graph_ids = [i for i, name in enumerate(names)
                 if name.startswith("g1.") or name.startswith("g2.")]
    text_ids = [i for i, name in enumerate(names)
                if name.startswith("text_proj") or name == "logit_scale"]
    values = []
    for ids in (graph_ids, text_ids):
        if not ids:
            continue
        real_block = torch.cat([real[i].reshape(-1) for i in ids])
        syn_block = torch.cat([synthetic[i].reshape(-1) for i in ids])
        values.append(_gradient_distance(real_block, syn_block))
    if not values:
        raise RuntimeError("proxy Student has no recognized gradient blocks")
    return torch.stack(values).mean()


def _gradient_distance_ids(model: P.GraphTextPromptStudent,
                           real: tuple[torch.Tensor, ...],
                           synthetic: tuple[torch.Tensor, ...],
                           prefixes: tuple[str, ...]) -> torch.Tensor:
    names = [name for name, _ in model.named_parameters()]
    ids = [i for i, name in enumerate(names)
           if any(name.startswith(prefix) for prefix in prefixes)]
    if not ids:
        raise RuntimeError(f"proxy Student has no parameters for {prefixes}")
    real_block = torch.cat([real[i].reshape(-1) for i in ids])
    syn_block = torch.cat([synthetic[i].reshape(-1) for i in ids])
    return _gradient_distance(real_block, syn_block)


def _multimodal_loss(model: P.GraphTextPromptStudent, memory: dict,
                     prompt_emb: torch.Tensor) -> torch.Tensor:
    x = memory["x"].float()
    text = memory["text_emb"].float()
    labels = memory["labels"].long()
    adj = P.normalized_adjacency(memory["edge_index"], len(x), str(x.device))
    graph = model.encode_graph(x, adj)
    text_value = model.encode_text(text)
    scale = model.logit_scale.exp().clamp(max=100.0)
    pair = P.multi_positive_pair_loss(graph, text_value, labels, scale)
    logits = scale * (graph @ model.encode_text(prompt_emb).T)
    prompt = F.cross_entropy(logits, labels)
    margin = P.prompt_margin_loss(logits, labels, 0.20)
    return (0.5 * pair + prompt + 0.1 * margin)


def _make_reference(delta: dict, parent: dict | None) -> dict:
    delta_labels = delta["y"].long().cpu()
    delta_edges = delta.get("edge_index",
                            torch.empty((2, 0), dtype=torch.long)).long().cpu()
    if parent is None:
        return {
            "x": delta["x"].float().cpu(),
            "text_emb": delta["text_emb"].float().cpu(),
            "labels": delta_labels,
            "edge_index": delta_edges,
        }
    old_n = len(parent["x"])
    parent_edges = parent.get("edge_index",
                              torch.empty((2, 0), dtype=torch.long)).long().cpu()
    shifted = delta_edges + old_n
    return {
        "x": torch.cat((parent["x"].float().cpu(), delta["x"].float().cpu())),
        "text_emb": torch.cat((parent["text_emb"].float().cpu(),
                                delta["text_emb"].float().cpu())),
        "labels": torch.cat((parent["labels"].long().cpu(), delta_labels)),
        "edge_index": torch.cat((parent_edges, shifted), dim=1),
    }


def _candidate_indices(delta: dict, labels: torch.Tensor,
                       cap: int) -> list[torch.Tensor]:
    raw_labels = delta["y"].long().cpu()
    candidates: list[torch.Tensor] = []
    for cls in labels.long().cpu().tolist():
        values = torch.where(raw_labels == int(cls))[0].long()
        if not len(values):
            raise ValueError(f"no current-delta text candidates for class {cls}")
        if int(cap) > 0 and len(values) > int(cap):
            # This is only a computational cap.  It is deterministic and does
            # not consult the final evaluation metric.
            values = values[:int(cap)]
        candidates.append(values)
    return candidates


def _soft_text(route: list[torch.Tensor], candidates: list[torch.Tensor],
               raw_text: torch.Tensor, temperature: float,
               straight_through: bool = False):
    rows, audits = [], []
    for logits, ids in zip(route, candidates):
        probabilities = F.softmax(logits / max(float(temperature), 1e-6), dim=0)
        values = raw_text.index_select(0, ids.to(raw_text.device))
        if straight_through:
            hard = F.one_hot(probabilities.argmax(), num_classes=len(probabilities))
            weights = hard.to(probabilities.dtype) + probabilities - probabilities.detach()
            rows.append(weights @ values)
        else:
            rows.append(probabilities @ values)
        with torch.no_grad():
            order = torch.argsort(probabilities, descending=True)
            entropy = -(probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()).sum()
            audits.append({
                "candidate_local_indices": ids.cpu()[order.cpu()].tolist(),
                "candidate_probabilities": probabilities[order].cpu().tolist(),
                "entropy": float(entropy.cpu()),
                "top1_probability": float(probabilities[order[0]].cpu()),
            })
    return _norm(torch.stack(rows)), audits


def _append_text(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= int(limit):
        return value
    # Prefer a complete sentence, then a word boundary, before the hard cap.
    prefix = value[:int(limit)]
    for mark in (".", "!", "?", ";", "。", "！", "？", "；"):
        pos = prefix.rfind(mark)
        if pos >= max(8, int(limit) // 2):
            return prefix[:pos + 1].strip()
    pos = prefix.rfind(" ")
    return prefix[:pos].strip() if pos >= 8 else prefix.strip()


def _materialize(delta: dict, candidates: list[torch.Tensor],
                 route: list[torch.Tensor], encoder, cfg: Config):
    texts, source_ids, audits = [], [], []
    for row, (logits, ids) in enumerate(zip(route, candidates)):
        probabilities = F.softmax(logits / max(float(cfg.temperature), 1e-6), dim=0)
        order = torch.argsort(probabilities, descending=True)
        take = order[:max(1, min(int(cfg.materialize_top_k), len(order)))]
        chosen = ids[take.cpu()].cpu().tolist()
        parts = [_append_text(delta["texts"][int(i)], cfg.text_chars)
                 for i in chosen]
        text = _append_text(" ".join(parts), cfg.text_chars)
        texts.append(text)
        source_ids.append([int(delta["ids"][int(i)]) for i in chosen])
        audits.append({
            "row": int(row),
            "chosen_local_indices": [int(i) for i in chosen],
            "chosen_source_ids": source_ids[-1],
            "probabilities": probabilities[take].detach().cpu().tolist(),
            "text_length": len(text),
        })
    encoded = encoder.encode(texts)
    encoded = torch.as_tensor(encoded).float().cpu()
    return texts, source_ids, _norm(encoded), audits


def _new_block(x: torch.Tensor, text_emb: torch.Tensor, texts: list[str],
               source_ids: list[list[int]], labels: torch.Tensor,
               stage: int, edge_index: torch.Tensor | None = None) -> dict:
    return {
        "x": x.detach().cpu().float(),
        "text_emb": text_emb.detach().cpu().float(),
        "texts": list(texts),
        "labels": labels.detach().cpu().long(),
        "mass": torch.ones(len(x)),
        "depth": torch.zeros(len(x), dtype=torch.long),
        "stage": torch.full((len(x),), int(stage), dtype=torch.long),
        "edge_index": (edge_index.detach().cpu().long().contiguous()
                        if edge_index is not None else
                        torch.empty((2, 0), dtype=torch.long)),
        "text_source_ids": source_ids,
    }


def append_memory(parent: dict | None, block: dict,
                  updated_old_x: torch.Tensor | None = None) -> dict:
    if parent is None:
        return block
    old_n = len(parent["x"])
    old_x = (updated_old_x.detach().cpu().float()
             if updated_old_x is not None else parent["x"].cpu().float())
    block_edges = block.get("edge_index",
                            torch.empty((2, 0), dtype=torch.long)).cpu().long()
    if block_edges.numel():
        block_edges = block_edges + old_n
    parent_edges = parent.get("edge_index",
                              torch.empty((2, 0), dtype=torch.long)).cpu().long()
    result = {
        "x": torch.cat((old_x, block["x"])),
        "text_emb": torch.cat((parent["text_emb"].cpu(), block["text_emb"])),
        "labels": torch.cat((parent["labels"].cpu(), block["labels"])),
        "mass": torch.cat((parent["mass"].cpu(), block["mass"])),
        "depth": torch.cat((parent["depth"].cpu(), block["depth"])),
        "stage": torch.cat((parent["stage"].cpu(), block["stage"])),
        "texts": list(parent["texts"]) + list(block["texts"]),
        "edge_index": torch.cat((parent_edges, block_edges), dim=1),
    }
    return result


def optimize_stage(delta: dict, parent: dict | None, x0: torch.Tensor,
                   labels: torch.Tensor, encoder, prompt_emb: torch.Tensor,
                   cfg: Config, device: str) -> tuple[dict, dict, torch.Tensor | None]:
    """Optimize new X and text routing against one or more proxy students."""
    device = str(device)
    x0 = x0.detach().to(device).float()
    labels = labels.detach().long().cpu()
    # The optimized candidate must be the same object that is later stored in
    # memory.  The old implementation optimized the raw cached embedding but
    # materialized a truncated string and re-encoded it, creating a hidden
    # soft-to-hard objective mismatch.
    candidate_texts = [
        _append_text(text, cfg.text_chars) for text in delta["texts"]
    ]
    candidate_encoded = torch.as_tensor(
        encoder.encode(candidate_texts)).float().to(device)
    raw_text = _norm(candidate_encoded)
    prompt_emb = prompt_emb.detach().to(device).float()
    candidates = _candidate_indices(delta, labels, cfg.candidate_cap)
    source_x = _norm(delta["x"].detach().to(device).float())
    route_values = []
    for row, ids in enumerate(candidates):
        # A graph-conditioned initialization gives the routing branch a
        # meaningful starting point without selecting from final-dev metrics.
        scores = source_x.index_select(0, ids.to(device)) @ _norm(x0[row:row + 1]).squeeze(0)
        route_values.append(scores.detach() / max(float(cfg.temperature), 1e-6))
    route_initial = [value.clone() for value in route_values]
    route = [torch.nn.Parameter(value.clone(), requires_grad=bool(cfg.update_route))
             if cfg.update_route else value.clone()
             for value in route_values]
    new_x = torch.nn.Parameter(x0.clone(), requires_grad=bool(cfg.update_x))
    old_x0 = None if parent is None else parent["x"].detach().to(device).float()
    old_x = (torch.nn.Parameter(old_x0.clone())
             if old_x0 is not None and cfg.allow_old_update else old_x0)
    optim_params = ([new_x] if cfg.update_x else []) + \
        ([value for value in route if isinstance(value, torch.nn.Parameter)]
         if cfg.update_route else [])
    if isinstance(old_x, torch.nn.Parameter):
        optim_params.append(old_x)
    param_groups = []
    if cfg.update_x:
        param_groups.append({"params": [new_x], "lr": float(cfg.lr_x)})
    if cfg.update_route:
        param_groups.append({"params": route, "lr": float(cfg.lr_route)})
    if isinstance(old_x, torch.nn.Parameter):
        param_groups.append({"params": [old_x], "lr": float(cfg.lr_x)})
    if not param_groups:
        raise ValueError("at least one of update_x/update_route must be true")
    optimizer = torch.optim.Adam(param_groups)

    with torch.no_grad():
        initial_soft_text, _ = _soft_text(
            route, candidates, raw_text, cfg.temperature,
            straight_through=False)
    if cfg.topology_mode == "none":
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        topology_audit = {
            "topology_mode": "none",
            "materialized_directed_edges": 0,
        }
        topology_q = None
    elif cfg.topology_mode == "fixed_q":
        new_edge_index, topology_audit, topology_q = build_fixed_topology(
            delta, x0, labels, raw_text, initial_soft_text, cfg, device)
        if new_edge_index.numel():
            refined_x, refine_audit = refine_topology_consistent_x(
                delta, x0, labels, new_edge_index, topology_q, cfg, device)
            new_x.data.copy_(refined_x)
            topology_audit["x_refinement"] = refine_audit
        else:
            topology_audit["x_refinement"] = {
                "steps": 0,
                "skipped": "no materialized raw topology under fixed budget",
            }
    elif cfg.topology_mode == "em_q":
        new_edge_index, topology_audit, topology_q = build_fixed_topology(
            delta, x0, labels, raw_text, initial_soft_text, cfg, device)
        if new_edge_index.numel():
            refined_x, refine_audit = refine_topology_consistent_x(
                delta, x0, labels, new_edge_index, topology_q, cfg, device)
            new_x.data.copy_(refined_x)
            topology_audit["x_refinement"] = refine_audit
        else:
            topology_audit["x_refinement"] = {
                "steps": 0,
                "skipped": "no materialized raw topology under fixed budget",
            }
    else:
        raise ValueError(f"unknown topology_mode={cfg.topology_mode}")

    reference = _make_reference(delta, parent)
    reference = {key: value.to(device) if isinstance(value, torch.Tensor)
                 else value for key, value in reference.items()}
    proxy_models = []
    real_grads = []
    for seed in cfg.proxy_seeds:
        torch.manual_seed(int(cfg.seed) + int(seed))
        model = P.GraphTextPromptStudent(
            int(x0.shape[1]), hidden=int(cfg.proxy_hidden),
            out=int(cfg.proxy_out), text_dim=int(raw_text.shape[1])).to(device)
        model.eval()
        params = tuple(model.parameters())
        real_loss = _multimodal_loss(model, reference, prompt_emb)
        real = torch.autograd.grad(real_loss, params, create_graph=False,
                                   retain_graph=False)
        proxy_models.append(model)
        real_grads.append(tuple(g.detach() for g in real))

    trace = []
    em_trace = []
    em_done = 0
    for step in range(max(1, int(cfg.steps))):
        soft_text, route_audit = _soft_text(
            route, candidates, raw_text, cfg.temperature,
            straight_through=cfg.straight_through_route)
        if old_x is None:
            full_x = new_x
            full_text = soft_text
            full_labels = labels.to(device)
        else:
            full_x = torch.cat((old_x, new_x), dim=0)
            old_text = parent["text_emb"].detach().to(device).float()
            full_text = torch.cat((old_text, soft_text), dim=0)
            full_labels = torch.cat((parent["labels"].detach().to(device),
                                     labels.to(device)))
        synthetic = {
            "x": full_x,
            "text_emb": full_text,
            "labels": full_labels,
            "edge_index": (torch.cat((
                parent.get("edge_index", torch.empty((2, 0), dtype=torch.long))
                .to(device),
                new_edge_index.to(device) + len(parent["x"])
                if parent is not None and new_edge_index.numel()
                else new_edge_index.to(device)), dim=1)
                if parent is not None else new_edge_index.to(device)),
        }
        gm_terms = []
        for model, real_grad in zip(proxy_models, real_grads):
            params = tuple(model.parameters())
            syn_loss = _multimodal_loss(model, synthetic, prompt_emb)
            syn = torch.autograd.grad(syn_loss, params, create_graph=True,
                                      retain_graph=True)
            if cfg.gradient_matching_mode == "blockwise":
                gm_terms.append(_gradient_distance_blockwise(
                    model, real_grad, syn))
            elif cfg.gradient_matching_mode == "decoupled":
                # Match the graph block through X with the text payload held
                # constant, and match the text block through routing with X
                # held constant.  Both targets still come from the same
                # multimodal proxy loss.
                syn_graph_memory = dict(synthetic)
                syn_graph_memory["text_emb"] = full_text.detach()
                graph_loss = _multimodal_loss(model, syn_graph_memory,
                                               prompt_emb)
                graph_grads = torch.autograd.grad(
                    graph_loss, params, create_graph=True, retain_graph=True)
                syn_text_memory = dict(synthetic)
                syn_text_memory["x"] = full_x.detach()
                text_loss = _multimodal_loss(model, syn_text_memory,
                                             prompt_emb)
                text_grads = torch.autograd.grad(
                    text_loss, params, create_graph=True, retain_graph=True)
                graph_term = _gradient_distance_ids(
                    model, real_grad, graph_grads, ("g1.", "g2."))
                text_term = _gradient_distance_ids(
                    model, real_grad, text_grads, ("text_proj", "logit_scale"))
                gm_terms.append(0.5 * (graph_term + text_term))
            elif cfg.gradient_matching_mode == "flat":
                gm_terms.append(_gradient_distance(
                    _flat_grads(real_grad), _flat_grads(syn)))
            else:
                raise ValueError(
                    f"unknown gradient_matching_mode={cfg.gradient_matching_mode}")
        gm = torch.stack(gm_terms).mean()
        new_scale = x0.norm(dim=1, keepdim=True).clamp_min(1e-3)
        new_anchor = (((new_x - x0) / new_scale) ** 2).mean()
        old_anchor = (new_x.sum() * 0.0)
        if isinstance(old_x, torch.nn.Parameter):
            old_scale = old_x0.norm(dim=1, keepdim=True).clamp_min(1e-3)
            old_anchor = (((old_x - old_x0) / old_scale) ** 2).mean()
        entropy_terms = []
        for logits in route:
            probabilities = F.softmax(logits / max(float(cfg.temperature), 1e-6), dim=0)
            entropy_terms.append(-(probabilities.clamp_min(1e-12)
                                   * probabilities.clamp_min(1e-12).log()).sum())
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else gm * 0.0
        route_anchor_terms = []
        for logits, initial in zip(route, route_initial):
            scale = initial.detach().norm().clamp_min(1.0)
            route_anchor_terms.append(((logits - initial.detach()) / scale).pow(2).mean())
        route_anchor = (torch.stack(route_anchor_terms).mean()
                        if route_anchor_terms else gm * 0.0)
        loss = (gm + float(cfg.lambda_cat_anchor) * new_anchor
                + float(cfg.lambda_old_anchor) * old_anchor
                + float(cfg.lambda_entropy) * entropy
                + float(cfg.lambda_route_anchor) * route_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite multimodal gradient matching loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if optim_params:
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            if cfg.update_x:
                delta_new = new_x - x0
                limit_new = float(cfg.new_trust_radius) * x0.norm(dim=1, keepdim=True).clamp_min(1e-3)
                scale_new = (limit_new / delta_new.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                new_x.copy_(x0 + delta_new * scale_new)
            if isinstance(old_x, torch.nn.Parameter):
                delta_old = old_x - old_x0
                limit_old = float(cfg.old_trust_radius) * old_x0.norm(dim=1, keepdim=True).clamp_min(1e-3)
                scale_old = (limit_old / delta_old.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                old_x.copy_(old_x0 + delta_old * scale_old)
            if (cfg.topology_mode == "em_q" and
                    em_done < int(cfg.em_updates) and
                    (step + 1) % max(1, int(cfg.em_interval)) == 0):
                em_text, _ = _soft_text(
                    route, candidates, raw_text, cfg.temperature,
                    straight_through=False)
                new_edge_index, em_audit, topology_q = build_fixed_topology(
                    delta, new_x.detach(), labels, raw_text, em_text, cfg,
                    device, previous_edge_index=new_edge_index)
                topology_audit = em_audit
                topology_audit["em_update_index"] = int(em_done + 1)
                em_trace.append({
                    "update": int(em_done + 1),
                    "edge_count": int(new_edge_index.shape[1] // 2),
                    "q_entropy": em_audit["q_mean_entropy"],
                })
                em_done += 1
        if step == 0 or step + 1 == int(cfg.steps) or (step + 1) % 10 == 0:
            trace.append({
                "step": int(step + 1),
                "loss": float(loss.detach().cpu()),
                "gradient_match": float(gm.detach().cpu()),
                "new_anchor": float(new_anchor.detach().cpu()),
                "old_anchor": float(old_anchor.detach().cpu()),
                "route_entropy": float(entropy.detach().cpu()),
                "route_anchor": float(route_anchor.detach().cpu()),
            })

    texts, source_ids, hard_text, materialization = _materialize(
        delta, candidates, route, encoder, cfg)
    with torch.no_grad():
        soft_text, final_route_audit = _soft_text(
            route, candidates, raw_text, cfg.temperature)
    block = _new_block(new_x, hard_text, texts, source_ids, labels,
                       int(delta["stage"]), new_edge_index)
    old_final = old_x.detach().cpu() if isinstance(old_x, torch.nn.Parameter) else None
    hard_soft_cos = F.cosine_similarity(soft_text.cpu(), hard_text, dim=1)
    audit = {
        "constructor": "cat_one_step_multimodal_gradient_matching",
        "objective": "normalized one-step proxy gradient matching + CaT/old anchors",
        "gradient_matching_mode": cfg.gradient_matching_mode,
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "topology_updated": bool(cfg.topology_mode == "em_q" and em_done),
        "topology": ("EM-like Q^T A Q coarsening"
                     if cfg.topology_mode == "em_q" else
                     "fixed Q^T A Q coarsening"
                     if cfg.topology_mode == "fixed_q" else
                     "edge-free/self-loop-only proxy"),
        "topology_audit": topology_audit,
        "em_trace": em_trace,
        "caT_initialization": True,
        "x_updated_after_cat": bool(cfg.update_x),
        "text_routing_updated": bool(cfg.update_route),
        "old_memory_updated": bool(old_final is not None),
        "allow_old_update": bool(cfg.allow_old_update),
        "proxy_seeds": [int(v) for v in cfg.proxy_seeds],
        "candidate_counts": [int(len(v)) for v in candidates],
        "candidate_embedding_source": "final_materialized_text_reencoded",
        "candidate_text_lengths": [len(value) for value in candidate_texts],
        "straight_through_route": bool(cfg.straight_through_route),
        "route_audit": final_route_audit,
        "materialization": materialization,
        "text_source_ids": source_ids,
        "soft_hard_cosine_mean": float(hard_soft_cos.mean()),
        "soft_hard_cosine_min": float(hard_soft_cos.min()),
        "new_x_update_norm": float((new_x.detach().cpu() - x0.cpu()).norm()),
        "old_x_update_norm": (float((old_final - old_x0.cpu()).norm())
                              if old_final is not None else 0.0),
        "route_anchor": float(cfg.lambda_route_anchor),
        "config": asdict(cfg),
        "trace": trace,
    }
    return block, audit, old_final
