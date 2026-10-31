"""Student-guided, graph-conditioned discrete text assignment.

This module is deliberately P-only.  The incoming synthetic ``X`` block and
the Student architecture are fixed; only the assignment logits from current
delta texts to the new synthetic rows are optimized.  The continuous
assignment is used during a short differentiable Student unroll and is then
projected back to auditable top-k raw text sources before MiniLM encoding.

The module does not read historical raw nodes.  ``delta`` is the only source
of candidate texts and ``parent`` is an already materialized synthetic memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PConfig:
    """Construction and unroll controls for the P-only experiment."""

    text_sources: int = 3
    text_chars: int = 320
    attention_temperature: float = 0.15
    outer_steps: int = 0
    outer_lr: float = 0.08
    unroll_steps: int = 0
    student_lr: float = 1e-3
    student_batch: int = 16
    student_seed: int = 90401
    lambda_unroll: float = 1.0
    lambda_prompt: float = 0.5
    lambda_entropy: float = 0.005
    lambda_coverage: float = 0.02
    lambda_semantic: float = 0.05
    lambda_anchor: float = 0.50
    anchor_blend: float = 0.95
    anchor_strength: float = 8.0
    anchor_diversity: float = 0.15
    max_anchor_shift: float = 0.20
    straight_through_topk: bool = True
    graph_weight: float = 0.50
    text_center_weight: float = 0.30
    prompt_weight: float = 0.20
    max_grad_norm: float = 5.0
    projection_candidate_pool: int = 8
    projection_unroll_steps: int = -1
    projection_min_gain: float = 0.0
    projection_lambda_prompt: float = 0.50
    projection_min_cosine: float = 0.98
    projection_max_accepts: int = 1
    projection_require_prompt_gain: bool = True


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-8)


def _attention_scores(delta: dict, source_features: torch.Tensor,
                      x_row: torch.Tensor, cls: int,
                      candidates: torch.Tensor, cfg: PConfig) -> torch.Tensor:
    """Build a deterministic graph-conditioned attention prior.

    Graph compatibility comes from the current delta propagation feature and
    the fixed synthetic X row.  Text compatibility comes from the candidate
    text center and the class prompt.  All three terms are normalized before
    mixing, so the selector has no trainable projection hidden in this prior.
    """
    graph = _normalize(source_features[candidates])
    query = _normalize(x_row.view(1, -1)).squeeze(0)
    graph_score = graph @ query

    text_emb = delta.get("text_emb")
    if text_emb is None:
        raise ValueError("P-only assignment requires delta text_emb")
    text = _normalize(text_emb[candidates])
    text_center = _normalize(text.mean(0, keepdim=True)).squeeze(0)
    text_score = text @ text_center

    prompt_score = torch.zeros_like(text_score)
    prompt_emb = delta.get("prompt_emb")
    if prompt_emb is not None and 0 <= int(cls) < len(prompt_emb):
        prompt = _normalize(prompt_emb[int(cls)].view(1, -1)).squeeze(0)
        prompt_score = text @ prompt

    return (float(cfg.graph_weight) * graph_score
            + float(cfg.text_center_weight) * text_score
            + float(cfg.prompt_weight) * prompt_score)


def _cctp_anchor_scores(delta: dict, cls: int,
                        candidates: torch.Tensor, cfg: PConfig) -> tuple[
                            torch.Tensor, list[int]]:
    """Approximate the frozen CCTP text-center + MMR top-k assignment."""
    text = _normalize(delta["text_emb"][candidates])
    center = _normalize(text.mean(0, keepdim=True)).squeeze(0)
    score = text @ center
    k = min(int(cfg.text_sources), len(candidates))
    available = torch.ones(len(candidates), dtype=torch.bool, device=score.device)
    selected = []
    for _ in range(k):
        adjusted = score.clone()
        if selected:
            chosen = text[torch.tensor(selected, device=text.device)]
            adjusted = adjusted - float(cfg.anchor_diversity) * (
                text @ chosen.T).max(dim=1).values
        adjusted[~available] = -torch.inf
        best = int(torch.argmax(adjusted).item())
        selected.append(best)
        available[best] = False
    # Keep the anchor auditable but allow the outer optimizer to move mass.
    logits = score.clone()
    if selected:
        logits[torch.tensor(selected, device=logits.device)] += float(cfg.anchor_strength)
    remaining = [i for i in range(len(candidates)) if i not in selected]
    remaining.sort(key=lambda i: float(score[i]), reverse=True)
    return logits, selected + remaining


def _soft_embeddings(delta: dict, candidate_sets: list[torch.Tensor],
                     logits: list[torch.Tensor], temperature: float) -> tuple[
                         list[torch.Tensor], list[torch.Tensor]]:
    """Return row-wise P and the corresponding differentiable text vectors."""
    out_p, out_t = [], []
    for candidates, row_logits in zip(candidate_sets, logits):
        p = torch.softmax(row_logits / float(temperature), dim=0)
        text = _normalize(delta["text_emb"][candidates])
        soft = _normalize((p.unsqueeze(1) * text).sum(0, keepdim=True)).squeeze(0)
        out_p.append(p)
        out_t.append(soft)
    return out_p, out_t


def _embeddings_from_probabilities(delta: dict,
                                   candidate_sets: list[torch.Tensor],
                                   probabilities: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for candidates, p in zip(candidate_sets, probabilities):
        text = _normalize(delta["text_emb"][candidates])
        out.append(_normalize((p.unsqueeze(1) * text).sum(0, keepdim=True)).squeeze(0))
    return out


def _bound_to_anchor(probabilities: list[torch.Tensor],
                     initial: list[torch.Tensor],
                     max_shift: float) -> tuple[list[torch.Tensor], float]:
    """Apply a row-wise trust region after the differentiable outer loop."""
    shifts = [0.5 * (p - p0).abs().sum()
              for p, p0 in zip(probabilities, initial)]
    observed = float(torch.stack(shifts).mean().detach().cpu()) if shifts else 0.0
    if not shifts or observed <= float(max_shift):
        return probabilities, observed
    scale = float(max_shift) / max(observed, 1e-8)
    bounded = []
    for p, p0 in zip(probabilities, initial):
        value = p0 + scale * (p - p0)
        bounded.append(value / value.sum().clamp_min(1e-8))
    return bounded, observed


def _functional_student_forward(params: dict[str, torch.Tensor], x: torch.Tensor,
                                text_emb: torch.Tensor,
                                adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass matching the original two-layer GCN Student."""
    h = F.linear(x, params["g1.lin.weight"], params["g1.lin.bias"])
    h = torch.sparse.mm(adj, F.relu(h)) if adj.is_sparse else adj @ F.relu(h)
    h = F.linear(h, params["g2.lin.weight"], params["g2.lin.bias"])
    h = torch.sparse.mm(adj, h) if adj.is_sparse else adj @ h
    graph = _normalize(h)
    text = _normalize(F.linear(text_emb, params["text_proj.weight"],
                               params["text_proj.bias"]))
    return graph, text


def _infonce_loss(params: dict[str, torch.Tensor], x: torch.Tensor,
                  text_emb: torch.Tensor, adj: torch.Tensor,
                  rows: torch.Tensor) -> torch.Tensor:
    graph, text = _functional_student_forward(params, x, text_emb, adj)
    rows = rows.long()
    scale = params["logit_scale"].exp().clamp(max=100)
    logits = scale * (graph[rows] @ text[rows].T)
    target = torch.arange(len(rows), device=rows.device)
    return (F.cross_entropy(logits, target)
            + F.cross_entropy(logits.T, target)) * 0.5


def _prompt_loss(params: dict[str, torch.Tensor], x: torch.Tensor,
                 text_emb: torch.Tensor, adj: torch.Tensor,
                 new_rows: torch.Tensor, labels: torch.Tensor,
                 prompt_emb: torch.Tensor | None) -> torch.Tensor:
    """New-class prompt CE used only as an outer selector signal."""
    if prompt_emb is None or len(new_rows) == 0:
        return x.sum() * 0.0
    graph, _ = _functional_student_forward(params, x, text_emb, adj)
    classes = sorted(set(int(v) for v in labels.detach().cpu().tolist()))
    prompt_rows = prompt_emb[classes].to(x.device).float()
    prompt = _normalize(F.linear(prompt_rows, params["text_proj.weight"],
                                 params["text_proj.bias"]))
    scale = params["logit_scale"].exp().clamp(max=100)
    logits = scale * (graph[new_rows] @ prompt.T)
    lookup = {cls: i for i, cls in enumerate(classes)}
    target = torch.tensor([lookup[int(v)] for v in labels.detach().cpu().tolist()],
                          dtype=torch.long, device=x.device)
    return F.cross_entropy(logits, target)


def _entropy(probabilities: list[torch.Tensor]) -> torch.Tensor:
    if not probabilities:
        return torch.tensor(0.0)
    return torch.stack([-(p * p.clamp_min(1e-8).log()).sum()
                        for p in probabilities]).mean()


def _coverage_loss(probabilities: list[torch.Tensor], labels: torch.Tensor,
                   candidate_sets: list[torch.Tensor]) -> torch.Tensor:
    """Encourage different synthetic rows to cover different source texts."""
    if not probabilities:
        return torch.tensor(0.0)
    terms = []
    labels_cpu = labels.detach().cpu()
    for cls in sorted(set(int(v) for v in labels_cpu.tolist())):
        rows = [i for i, value in enumerate(labels_cpu.tolist()) if int(value) == cls]
        if len(rows) < 2:
            continue
        candidates = candidate_sets[rows[0]]
        if any(not torch.equal(candidate_sets[i], candidates) for i in rows[1:]):
            continue
        marginal = torch.stack([probabilities[i] for i in rows]).mean(0)
        terms.append((marginal * marginal.clamp_min(1e-8).log()).sum())
    if not terms:
        return probabilities[0].sum() * 0.0
    # Negative entropy: minimizing this maximizes the class-level coverage.
    return torch.stack(terms).mean()


def _clone_params(model: torch.nn.Module, device: str) -> dict[str, torch.Tensor]:
    return {name: value.detach().to(device).clone().requires_grad_(True)
            for name, value in model.named_parameters()}


def _make_batches(total_rows: int, steps: int, batch: int,
                  seed: int, device: str) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    out = []
    for _ in range(max(0, int(steps))):
        perm = torch.randperm(total_rows, generator=generator)
        out.append(perm[:min(int(batch), total_rows)].to(device))
    return out


def _run_unroll(base_params: dict[str, torch.Tensor], x: torch.Tensor,
                text_emb: torch.Tensor, adj: torch.Tensor,
                batches: list[torch.Tensor], lr: float) -> tuple[
                    dict[str, torch.Tensor], torch.Tensor]:
    params = {name: value.detach().clone().requires_grad_(True)
              for name, value in base_params.items()}
    last = None
    for rows in batches:
        last = _infonce_loss(params, x, text_emb, adj, rows)
        grads = torch.autograd.grad(last, tuple(params.values()),
                                    create_graph=True, allow_unused=False)
        params = {name: value - float(lr) * grad
                  for (name, value), grad in zip(params.items(), grads)}
    if last is None:
        last = _infonce_loss(params, x, text_emb, adj,
                             torch.arange(len(x), device=x.device))
    return params, last


def _run_adam_response(base_params: dict[str, torch.Tensor], x: torch.Tensor,
                       text_emb: torch.Tensor, adj: torch.Tensor,
                       batches: list[torch.Tensor], lr: float) -> tuple[
                           dict[str, torch.Tensor], torch.Tensor]:
    """Run a first-order Adam response trajectory matching final Student training."""
    params = {name: value.detach().clone().requires_grad_(True)
              for name, value in base_params.items()}
    moments = {name: torch.zeros_like(value) for name, value in params.items()}
    squares = {name: torch.zeros_like(value) for name, value in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    last = None
    for step, rows in enumerate(batches, start=1):
        last = _infonce_loss(params, x, text_emb, adj, rows)
        grads = torch.autograd.grad(last, tuple(params.values()),
                                    create_graph=False, allow_unused=False)
        updated = {}
        for (name, value), grad in zip(params.items(), grads):
            moments[name] = beta1 * moments[name] + (1.0 - beta1) * grad
            squares[name] = beta2 * squares[name] + (1.0 - beta2) * grad.square()
            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step
            direction = (moments[name] / bias1) / (
                squares[name].div(bias2).sqrt().add(eps))
            updated[name] = (value - float(lr) * direction).detach().requires_grad_(True)
        params = updated
    return params, last


def _build_inner_memory(parent: dict | None, x: torch.Tensor,
                        text_emb: torch.Tensor) -> dict:
    if parent is None:
        return {"x": x.float().cpu(), "text_emb": text_emb.float().cpu(),
                "mass": torch.ones(len(x)),
                "depth": torch.zeros(len(x), dtype=torch.long),
                "stage": torch.zeros(len(x), dtype=torch.long),
                "texts": [""] * len(x)}
    return {
        "x": torch.cat((parent["x"].float(), x.float().cpu())),
        "text_emb": torch.cat((parent["text_emb"].float(), text_emb.float().cpu())),
        "mass": torch.cat((parent["mass"].float(), torch.ones(len(x)))),
        "depth": torch.cat((parent["depth"].long(), torch.zeros(len(x), dtype=torch.long))),
        "stage": torch.cat((parent["stage"].long(), torch.zeros(len(x), dtype=torch.long))),
        "texts": list(parent["texts"]) + [""] * len(x),
    }


def _materialize_texts(delta: dict, candidate_sets: list[torch.Tensor],
                      probabilities: list[torch.Tensor], cfg: PConfig,
                      anchor_orders: list[list[int]] | None = None) -> tuple[
                          list[str], list[list[int]]]:
    texts, source_ids = [], []
    for candidates, p in zip(candidate_sets, probabilities):
        k = min(int(cfg.text_sources), len(candidates))
        chosen = torch.topk(p, k=k, largest=True, sorted=True).indices
        if anchor_orders is not None:
            rank = {int(value): index for index, value
                    in enumerate(anchor_orders[len(texts)])}
            ordered = sorted(chosen.detach().cpu().tolist(),
                             key=lambda value: rank.get(int(value), len(rank)))
            chosen = torch.tensor(ordered, dtype=torch.long, device=p.device)
        chosen_ids = candidates[chosen]
        source_ids.append(delta["ids"][chosen_ids.detach().cpu()].long().tolist())
        separator_budget = max(0, k - 1)
        per = max(1, (int(cfg.text_chars) - separator_budget) // max(1, k))
        text = " ".join(str(delta["texts"][int(i)])[:per]
                         for i in chosen_ids.detach().cpu().tolist())
        texts.append(text[:int(cfg.text_chars)])
    return texts, source_ids


def _compose_text(delta: dict, candidates: torch.Tensor,
                  local_indices: list[int], cfg: PConfig) -> tuple[str, list[int]]:
    """Compose one auditable text in the same order used by the CCTP anchor."""
    chosen = [int(candidates[index]) for index in local_indices]
    k = len(chosen)
    separator_budget = max(0, k - 1)
    per = max(1, (int(cfg.text_chars) - separator_budget) // max(1, k))
    text = " ".join(str(delta["texts"][index])[:per] for index in chosen)
    return text[:int(cfg.text_chars)], [int(delta["ids"][index]) for index in chosen]


def _projection_variants(delta: dict, source_features: torch.Tensor,
                         x_row: torch.Tensor, cls: int,
                         candidates: torch.Tensor,
                         anchor_order: list[int], cfg: PConfig) -> list[tuple[str, list[int], list[int]]]:
    """Enumerate a small one-slot replacement neighborhood around the anchor.

    The neighborhood is deliberately discrete and small.  It never changes
    the number or order of anchor slots except for one replacement, which
    keeps the frozen text encoder's concatenation behavior controlled.
    """
    k = min(int(cfg.text_sources), len(candidates))
    anchor = list(anchor_order[:k])
    attention = _attention_scores(
        delta, source_features, x_row, int(cls), candidates, cfg)
    ranked = torch.argsort(attention, descending=True).detach().cpu().tolist()
    pool = list(anchor)
    for value in ranked:
        if value not in pool:
            pool.append(int(value))
        if len(pool) >= max(k, int(cfg.projection_candidate_pool)):
            break
    variants = []
    for slot in range(k):
        for replacement in pool:
            if replacement in anchor:
                continue
            local = list(anchor)
            local[slot] = int(replacement)
            text, source_ids = _compose_text(delta, candidates, local, cfg)
            variants.append((text, source_ids, local))
    return variants


def _projection_score(params: dict[str, torch.Tensor], x: torch.Tensor,
                      text_emb: torch.Tensor, adj: torch.Tensor,
                      rows: torch.Tensor, new_rows: torch.Tensor,
                      labels: torch.Tensor,
                      prompt_emb: torch.Tensor | None,
                      lambda_prompt: float) -> torch.Tensor:
    """Score a materialized text candidate in the actual Student space."""
    loss = _infonce_loss(params, x, text_emb, adj, rows)
    if prompt_emb is not None and len(new_rows) > 0:
        loss = loss + float(lambda_prompt) * _prompt_loss(
            params, x, text_emb, adj, new_rows, labels, prompt_emb)
    return loss


def _projection_components(params: dict[str, torch.Tensor], x: torch.Tensor,
                           text_emb: torch.Tensor, adj: torch.Tensor,
                           rows: torch.Tensor, new_rows: torch.Tensor,
                           labels: torch.Tensor,
                           prompt_emb: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    infonce = _infonce_loss(params, x, text_emb, adj, rows)
    if prompt_emb is None or len(new_rows) == 0:
        prompt = infonce.detach() * 0.0
    else:
        prompt = _prompt_loss(params, x, text_emb, adj, new_rows, labels, prompt_emb)
    return infonce, prompt


def build_projection_aware_block(delta: dict, source_features: torch.Tensor,
                                 x: torch.Tensor, labels: torch.Tensor,
                                 parent: dict | None, encoder, cfg: PConfig,
                                 student_api, device: str) -> tuple[dict, dict]:
    """Construct text by discrete projection-aware Student response search.

    ``P`` is not optimized through a soft surrogate here.  Instead, the
    CCTP anchor is materialized, each row gets a bounded one-slot swap
    neighborhood, every candidate is re-encoded by the frozen MiniLM, and a
    short Student response proxy accepts only a lower-loss candidate.  The
    method remains append-only and reads only current-delta raw text.
    """
    if len(x) != len(labels):
        raise ValueError("projection-aware X/label row mismatch")
    source_features = source_features.detach().to(device).float()
    x = x.detach().to(device).float()
    labels = labels.detach().long().to(device)
    delta_device = dict(delta)
    delta_device["y"] = delta["y"].detach().to(device).long()
    delta_device["text_emb"] = delta["text_emb"].detach().to(device).float()
    if delta.get("prompt_emb") is not None:
        delta_device["prompt_emb"] = delta["prompt_emb"].detach().to(device).float()

    candidate_sets, anchor_orders = [], []
    anchor_texts, anchor_source_ids = [], []
    variants_per_row = []
    for row, cls in enumerate(labels.tolist()):
        candidates = torch.where(delta_device["y"] == int(cls))[0].long()
        if len(candidates) == 0:
            raise ValueError(f"no current-delta candidates for class {cls}")
        _, anchor_order = _cctp_anchor_scores(delta_device, int(cls), candidates, cfg)
        text, source_ids = _compose_text(
            delta, candidates, anchor_order[:int(cfg.text_sources)], cfg)
        candidate_sets.append(candidates)
        anchor_orders.append(anchor_order)
        anchor_texts.append(text)
        anchor_source_ids.append(source_ids)
        variants_per_row.append(_projection_variants(
            delta_device, source_features, x[row], int(cls), candidates,
            anchor_order, cfg))

    anchor_embedding = encoder.encode(anchor_texts).to(device).float()
    old_rows = 0 if parent is None else len(parent["x"])
    inner_memory = _build_inner_memory(parent, x, anchor_embedding)
    inner_trainable = student_api.memory_to_trainable(inner_memory)
    fixed_edges = inner_trainable["edge_index"].detach().cpu()
    total_rows = len(inner_memory["x"])
    adj = student_api.norm_adj(fixed_edges, total_rows, device)
    full_x = torch.cat((parent["x"].to(device).float(), x), dim=0) if parent else x
    current_text = torch.cat((parent["text_emb"].to(device).float(), anchor_embedding), dim=0) if parent else anchor_embedding
    new_rows = torch.arange(old_rows, total_rows, device=device)
    all_rows = torch.arange(total_rows, device=device)

    torch.manual_seed(int(cfg.student_seed))
    student = student_api.CLIPStudent(int(inner_memory["x"].shape[1])).to(device)
    base_params = _clone_params(student, device)
    del student
    projection_steps = int(cfg.projection_unroll_steps)
    if projection_steps < 0:
        projection_steps = max(50, min(2048, 16 * total_rows))
    # Match the formal Student protocol: Adam, batch 16, and the same
    # seed-derived minibatch sequence as final scratch training.
    batches = _make_batches(total_rows, projection_steps, cfg.student_batch,
                            cfg.student_seed * 7 + 1, device)
    response_params, _ = _run_adam_response(
        base_params, full_x, current_text, adj, batches, cfg.student_lr)
    response_params = {name: value.detach() for name, value in response_params.items()}
    prompt_emb = delta_device.get("prompt_emb")
    current_infonce, current_prompt = _projection_components(
        response_params, full_x, current_text, adj, all_rows, new_rows,
        labels, prompt_emb)
    current_score = current_infonce + float(cfg.projection_lambda_prompt) * current_prompt
    accepted = []
    candidate_counts = []
    gains = []
    cosine_shifts = []
    accepted_total = 0
    for row, variants in enumerate(variants_per_row):
        candidate_counts.append(len(variants))
        if not variants:
            accepted.append(False)
            gains.append(0.0)
            cosine_shifts.append(0.0)
            continue
        candidate_embeddings = encoder.encode([value[0] for value in variants]).to(device).float()
        best_score = current_score
        best_infonce, best_prompt = current_infonce, current_prompt
        best_index = None
        best_cosine = 1.0
        for index in range(len(variants)):
            cosine = float(F.cosine_similarity(
                current_text[old_rows + row].view(1, -1),
                candidate_embeddings[index].view(1, -1), dim=1).item())
            if cosine < float(cfg.projection_min_cosine):
                continue
            proposal = current_text.clone()
            proposal[old_rows + row] = candidate_embeddings[index]
            proposal_infonce, proposal_prompt = _projection_components(
                response_params, full_x, proposal, adj, all_rows, new_rows,
                labels, prompt_emb)
            if (cfg.projection_require_prompt_gain
                    and prompt_emb is not None
                    and float(proposal_prompt.detach().cpu())
                    >= float(current_prompt.detach().cpu()) - 1e-8):
                continue
            score = proposal_infonce + float(cfg.projection_lambda_prompt) * proposal_prompt
            if float(score.detach().cpu()) < float(best_score.detach().cpu()):
                best_score = score
                best_infonce, best_prompt = proposal_infonce, proposal_prompt
                best_index = index
                best_cosine = cosine
        gain = float((current_score - best_score).detach().cpu())
        if (best_index is not None
                and gain > float(cfg.projection_min_gain)
                and accepted_total < int(cfg.projection_max_accepts)):
            current_text[old_rows + row] = candidate_embeddings[best_index]
            anchor_texts[row] = variants[best_index][0]
            anchor_source_ids[row] = variants[best_index][1]
            current_score = best_score
            current_infonce, current_prompt = best_infonce, best_prompt
            accepted_total += 1
            accepted.append(True)
            gains.append(gain)
            cosine_shifts.append(best_cosine)
        else:
            accepted.append(False)
            gains.append(0.0)
            cosine_shifts.append(1.0)

    block = {
        "x": x.detach().cpu(),
        "text_emb": current_text[old_rows:].detach().cpu(),
        "texts": list(anchor_texts),
        "mass": torch.ones(len(x)),
        "depth": torch.zeros(len(x), dtype=torch.long),
        "stage": torch.full((len(x),), int(delta["stage"]), dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    audit = {
        "selector": "projection_aware_one_slot_student_response",
        "current_delta_only": True,
        "fixed_x": True,
        "fixed_inner_topology": True,
        "inner_topology_edges": int(fixed_edges.shape[1]),
        "parent_rows": int(old_rows),
        "new_rows": int(len(x)),
        "row_classes": labels.detach().cpu().tolist(),
        "text_source_ids": anchor_source_ids,
        "top_k": int(cfg.text_sources),
        "text_chars": int(cfg.text_chars),
        "projection_candidate_pool": int(cfg.projection_candidate_pool),
        "projection_unroll_steps": int(projection_steps),
        "projection_min_gain": float(cfg.projection_min_gain),
        "projection_lambda_prompt": float(cfg.projection_lambda_prompt),
        "projection_min_cosine": float(cfg.projection_min_cosine),
        "projection_max_accepts": int(cfg.projection_max_accepts),
        "projection_require_prompt_gain": bool(cfg.projection_require_prompt_gain),
        "candidate_counts": candidate_counts,
        "accepted_rows": int(sum(accepted)),
        "accepted_mask": accepted,
        "proxy_gains": gains,
        "accepted_cosines": cosine_shifts,
        "proxy_total_gain": float(sum(gains)),
        "assignment_parameterization": "CCTP anchor plus bounded one-slot discrete projection",
        "projection": "every candidate concatenated and re-encoded by frozen MiniLM",
    }
    return block, audit


def build_p_only_block(delta: dict, source_features: torch.Tensor,
                       x: torch.Tensor, labels: torch.Tensor,
                       parent: dict | None, encoder, cfg: PConfig,
                       student_api, device: str) -> tuple[dict, dict]:
    """Optimize current-delta text assignment while keeping X fixed."""
    if len(x) != len(labels):
        raise ValueError("P-only X/label row mismatch")
    if cfg.text_sources < 1:
        raise ValueError("text_sources must be positive")

    source_features = source_features.detach().to(device).float()
    x = x.detach().to(device).float()
    labels = labels.detach().long().to(device)
    text_emb = delta["text_emb"].detach().to(device).float()
    delta_device = dict(delta)
    delta_device["y"] = delta["y"].detach().to(device).long()
    delta_device["text_emb"] = text_emb
    if delta.get("prompt_emb") is not None:
        delta_device["prompt_emb"] = delta["prompt_emb"].detach().to(device).float()
    candidate_sets = []
    initial_scores = []
    initial_probabilities = []
    anchor_orders = []
    for row, cls in enumerate(labels.tolist()):
        candidates = torch.where(delta_device["y"] == int(cls))[0].long()
        if len(candidates) == 0:
            raise ValueError(f"no current-delta candidates for class {cls}")
        attention = _attention_scores(delta_device, source_features, x[row],
                                      int(cls), candidates, cfg)
        anchor, anchor_order = _cctp_anchor_scores(
            delta_device, int(cls), candidates, cfg)
        scores = ((1.0 - float(cfg.anchor_blend)) * attention
                  + float(cfg.anchor_blend) * anchor)
        candidate_sets.append(candidates)
        anchor_orders.append(anchor_order)
        initial_scores.append(scores.detach())
        initial_probabilities.append(torch.softmax(
            scores / float(cfg.attention_temperature), dim=0).detach())

    # Each row has its own candidate set.  This is the explicit assignment P;
    # no historical raw source can enter the parameter list.
    p_logits = torch.nn.ParameterList([
        torch.nn.Parameter(score.clone()) for score in initial_scores
    ]).to(device)
    initial_p = [p.clone() for p in initial_probabilities]

    # The inner Student graph is fixed once from the initial soft P.  This
    # prevents the P selector from silently changing the topology operator.
    _, initial_soft = _soft_embeddings(delta_device, candidate_sets,
                                        list(p_logits), cfg.attention_temperature)
    inner_memory = _build_inner_memory(parent, x, torch.stack(initial_soft))
    inner_trainable = student_api.memory_to_trainable(inner_memory)
    fixed_edges = inner_trainable["edge_index"].detach().cpu()
    total_rows = len(inner_memory["x"])
    adj = student_api.norm_adj(fixed_edges, total_rows, device)

    torch.manual_seed(int(cfg.student_seed))
    student = student_api.CLIPStudent(int(inner_memory["x"].shape[1])).to(device)
    base_params = _clone_params(student, device)
    del student
    batches = _make_batches(total_rows, cfg.unroll_steps, cfg.student_batch,
                            cfg.student_seed + 17, device)
    outer_optimizer = torch.optim.Adam(list(p_logits), lr=float(cfg.outer_lr))
    trace = []

    old_rows = 0 if parent is None else len(parent["x"])
    new_rows = torch.arange(old_rows, total_rows, device=device)
    current_labels = labels
    for outer_step in range(max(0, int(cfg.outer_steps))):
        probabilities, _ = _soft_embeddings(
            delta_device, candidate_sets, list(p_logits), cfg.attention_temperature)
        if cfg.straight_through_topk:
            forward_probabilities = []
            for probability in probabilities:
                k = min(int(cfg.text_sources), len(probability))
                hard = torch.zeros_like(probability)
                hard[torch.topk(probability, k=k).indices] = 1.0 / float(k)
                forward_probabilities.append(
                    hard + probability - probability.detach())
            soft_rows = _embeddings_from_probabilities(
                delta_device, candidate_sets, forward_probabilities)
        else:
            soft_rows = _embeddings_from_probabilities(
                delta_device, candidate_sets, probabilities)
        soft_new = torch.stack(soft_rows)
        if parent is None:
            full_x = x
            full_text = soft_new
        else:
            full_x = torch.cat((parent["x"].to(device).float(), x), dim=0)
            full_text = torch.cat((parent["text_emb"].to(device).float(), soft_new), dim=0)

        final_params, inner_loss = _run_unroll(
            base_params, full_x, full_text, adj, batches, cfg.student_lr)
        prompt_loss = _prompt_loss(
            final_params, full_x, full_text, adj, new_rows, current_labels,
            delta_device.get("prompt_emb"))
        full_rows = torch.arange(total_rows, device=device)
        final_infonce = _infonce_loss(final_params, full_x, full_text,
                                      adj, full_rows)
        entropy = _entropy(probabilities)
        coverage = _coverage_loss(probabilities, labels, candidate_sets)
        semantic = torch.stack([
            -(p * score.detach()).sum()
            for p, score in zip(probabilities, initial_scores)
        ]).mean()
        anchor = torch.stack([
            (p - p0).square().sum()
            for p, p0 in zip(probabilities, initial_p)
        ]).mean()
        loss = (float(cfg.lambda_unroll) * final_infonce
                + float(cfg.lambda_prompt) * prompt_loss
                + float(cfg.lambda_anchor) * anchor
                + float(cfg.lambda_entropy) * entropy
                + float(cfg.lambda_coverage) * coverage
                + float(cfg.lambda_semantic) * semantic)
        outer_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p_logits), float(cfg.max_grad_norm))
        outer_optimizer.step()
        trace.append({"step": outer_step + 1,
                      "loss": float(loss.detach().cpu()),
                      "student_infonce": float(final_infonce.detach().cpu()),
                      "prompt_loss": float(prompt_loss.detach().cpu()),
                      "anchor": float(anchor.detach().cpu()),
                      "entropy": float(entropy.detach().cpu()),
                      "coverage": float(coverage.detach().cpu()),
                      "inner_last_loss": float(inner_loss.detach().cpu())})

    probabilities, _ = _soft_embeddings(delta_device, candidate_sets,
                                         list(p_logits), cfg.attention_temperature)
    probabilities, anchor_shift = _bound_to_anchor(
        [p.detach() for p in probabilities], initial_p,
        cfg.max_anchor_shift)
    texts, source_ids = _materialize_texts(delta, candidate_sets,
                                           probabilities, cfg, anchor_orders)
    text_embedding = encoder.encode(texts).cpu()
    block = {
        "x": x.detach().cpu(),
        "text_emb": text_embedding,
        "texts": texts,
        "mass": torch.ones(len(x)),
        "depth": torch.zeros(len(x), dtype=torch.long),
        "stage": torch.full((len(x),), int(delta["stage"]), dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    final_p = [p.detach().cpu() for p in probabilities]
    audit = {
        "selector": "student_guided_graph_conditioned_p_only",
        "current_delta_only": True,
        "fixed_x": True,
        "fixed_inner_topology": True,
        "inner_topology_edges": int(fixed_edges.shape[1]),
        "parent_rows": int(old_rows),
        "new_rows": int(len(x)),
        "row_classes": labels.detach().cpu().tolist(),
        "text_source_ids": source_ids,
        "candidate_counts": [int(len(v)) for v in candidate_sets],
        "top_k": int(cfg.text_sources),
        "text_chars": int(cfg.text_chars),
        "attention_temperature": float(cfg.attention_temperature),
        "anchor_blend": float(cfg.anchor_blend),
        "anchor_strength": float(cfg.anchor_strength),
        "anchor_diversity": float(cfg.anchor_diversity),
        "max_anchor_shift": float(cfg.max_anchor_shift),
        "observed_anchor_shift": float(anchor_shift),
        "straight_through_topk": bool(cfg.straight_through_topk),
        "outer_steps": int(cfg.outer_steps),
        "unroll_steps": int(cfg.unroll_steps),
        "outer_lr": float(cfg.outer_lr),
        "student_lr": float(cfg.student_lr),
        "student_seed": int(cfg.student_seed),
        "initial_probability_entropy": float(_entropy(initial_p).item()),
        "final_probability_entropy": float(_entropy(final_p).item()),
        "initial_topk": [torch.topk(p, min(int(cfg.text_sources), len(p))).indices.tolist()
                          for p in initial_p],
        "final_topk": [torch.topk(p, min(int(cfg.text_sources), len(p))).indices.tolist()
                        for p in final_p],
        "anchor_orders": [order[:int(cfg.text_sources)] for order in anchor_orders],
        "trace": trace,
        "assignment_parameterization": "row-wise masked softmax over same-class current-delta text embeddings",
        "projection": "hard top-k real current-delta sources, fixed character budget, frozen encoder",
    }
    return block, audit

