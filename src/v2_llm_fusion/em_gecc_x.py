"""EM-style GECC initialization with free synthetic-X optimization.

This branch is an experimental construction method, not an official GECC
reproduction.  It alternates class-constrained responsibility updates (E) and
Student gradient matching on free synthetic features (M).  The historical
synthetic prefix is read-only and the resulting memory is edge-free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import external_continual_mechanisms as C
import graphtext_prompt_student as P
import multimodal_gradient_matching as GM


@dataclass(frozen=True)
class Config:
    em_rounds: int = 4
    m_steps: int = 5
    lr_x: float = 5e-3
    q_temperature: float = 0.20
    q_sinkhorn_steps: int = 40
    feature_score_weight: float = 0.45
    text_score_weight: float = 0.25
    student_score_weight: float = 0.30
    geometry_weight: float = 0.25
    proxy_hidden: int = 128
    proxy_out: int = 128
    proxy_seed: int = 2711
    seed: int = 271006
    text_chars: int = 320


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1)


def _device_memory(memory: dict | None, device: str) -> dict | None:
    if memory is None:
        return None
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in memory.items()}


def _sinkhorn(scores: torch.Tensor, row_labels: torch.Tensor,
              slot_labels: torch.Tensor, temperature: float,
              steps: int) -> torch.Tensor:
    """Class-masked balanced transport with row mass one."""
    q = torch.zeros_like(scores)
    row_labels = row_labels.to(scores.device).long()
    slot_labels = slot_labels.to(scores.device).long()
    for cls in torch.unique(row_labels, sorted=True).tolist():
        rows = torch.where(row_labels == int(cls))[0]
        cols = torch.where(slot_labels == int(cls))[0]
        if not len(cols) or len(rows) < len(cols):
            raise ValueError(f"infeasible class-{cls} assignment")
        local = scores[rows][:, cols] / max(float(temperature), 1e-6)
        capacity = torch.full((len(cols),), len(rows) / len(cols),
                              dtype=local.dtype, device=local.device)
        q[rows[:, None], cols[None, :]] = GM._log_sinkhorn(
            local, capacity, int(steps))
    return q


def _hard_text_rows(q: torch.Tensor, raw_labels: torch.Tensor,
                    slot_labels: torch.Tensor) -> torch.Tensor:
    """Choose one distinct, same-class raw text source per synthetic slot."""
    selected = torch.full((len(slot_labels),), -1, dtype=torch.long,
                          device=q.device)
    for cls in torch.unique(slot_labels, sorted=True).tolist():
        rows = torch.where(raw_labels == int(cls))[0]
        cols = torch.where(slot_labels == int(cls))[0]
        local = q[rows][:, cols].detach().cpu().numpy()
        row_ids, col_ids = linear_sum_assignment(-local)
        selected[cols[torch.as_tensor(col_ids, device=q.device)]] = rows[
            torch.as_tensor(row_ids, device=q.device)]
    if bool((selected < 0).any()):
        raise AssertionError("every synthetic slot must receive one text")
    return selected


def _geometry_target(q: torch.Tensor, propagated: torch.Tensor) -> torch.Tensor:
    mass = q.sum(0).clamp_min(1e-8)
    return (q.T @ propagated) / mass[:, None]


def _e_step(delta: dict, x: torch.Tensor, slot_labels: torch.Tensor,
            raw_text: torch.Tensor, init_or_current_text: torch.Tensor,
            propagated: torch.Tensor, model: P.GraphTextPromptStudent,
            raw_adj: torch.Tensor, synthetic_adj: torch.Tensor,
            cfg: Config) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Update responsibilities from graph, text and fixed-Student agreement."""
    with torch.no_grad():
        raw_graph = model.encode_graph(delta["x"].float().to(x.device), raw_adj)
        synthetic_graph = model.encode_graph(x.float(), synthetic_adj)
        feature_score = _normalize(propagated) @ _normalize(x).T
        text_score = _normalize(raw_text) @ _normalize(init_or_current_text).T
        student_score = raw_graph @ synthetic_graph.T
        total_weight = (float(cfg.feature_score_weight)
                        + float(cfg.text_score_weight)
                        + float(cfg.student_score_weight))
        scores = (float(cfg.feature_score_weight) * feature_score
                  + float(cfg.text_score_weight) * text_score
                  + float(cfg.student_score_weight) * student_score)
        scores = scores / max(total_weight, 1e-8)
        q = _sinkhorn(scores, delta["y"].to(scores.device).long(), slot_labels.long(),
                      cfg.q_temperature, cfg.q_sinkhorn_steps)
        selected = _hard_text_rows(q, delta["y"].to(scores.device).long(),
                                   slot_labels.long())
        mass = q.sum(0).clamp_min(1e-8)
        soft_text = _normalize(q.T @ raw_text / mass[:, None])
        probabilities = q / mass[None, :]
        entropy = -(probabilities.clamp_min(1e-12)
                    * probabilities.clamp_min(1e-12).log()).sum(0)
        class_counts = torch.stack([
            (delta["y"].to(scores.device) == int(cls)).sum()
            for cls in slot_labels.tolist()]).float()
        normalized_entropy = entropy / class_counts.log().clamp_min(1e-8)
        return q.detach(), selected, {
            "q_entropy_mean": float(entropy.mean().cpu()),
            "q_entropy_normalized": float(normalized_entropy.mean().cpu()),
            "q_row_sum_max_error": float((q.sum(1) - 1.0).abs().max().cpu()),
            "q_slot_mass_cv": float((q.sum(0).std(unbiased=False) /
                                      q.sum(0).mean().clamp_min(1e-8)).cpu()),
            "soft_text": soft_text,
            "score_feature_mean": float(feature_score.mean().cpu()),
            "score_text_mean": float(text_score.mean().cpu()),
            "score_student_mean": float(student_score.mean().cpu()),
        }


def _gradient_stats(real: tuple[torch.Tensor, ...],
                    synthetic: tuple[torch.Tensor, ...]) -> tuple[float, float]:
    left = torch.cat([value.reshape(-1) for value in real])
    right = torch.cat([value.detach().reshape(-1) for value in synthetic])
    return (float(F.cosine_similarity(left, right, dim=0).detach().cpu()),
            float((left.norm() / right.norm().clamp_min(1e-8)).detach().cpu()))


def optimize_stage(delta: dict, parent: dict | None, x_init: torch.Tensor,
                   init_text_emb: torch.Tensor, labels: torch.Tensor,
                   encoder, prompt_emb: torch.Tensor, cfg: Config,
                   device: str, *, use_gradient_matching: bool = True,
                   use_geometry: bool = True) -> tuple[dict, dict]:
    """Alternately update Q/text support and unconstrained current-stage X.

    ``x_init`` must be the current-delta GECC hard-clustering output.  It is
    only an initialization; the M-step optimizes ``new_x`` directly.  No old
    synthetic feature is an optimizer parameter.
    """
    if not use_gradient_matching and not use_geometry:
        raise ValueError("at least one M-step objective must be active")
    device = str(device)
    labels = labels.detach().long().to(device)
    raw_labels = delta["y"].long().to(device)
    raw_x = delta["x"].float().to(device)
    x0 = x_init.detach().float().to(device)
    if len(labels) != len(x0) or len(labels) < 1:
        raise ValueError("GECC initialization and slot labels do not align")
    if parent is not None and parent.get("edge_index", torch.empty((2, 0))).numel():
        raise ValueError("EM-GECC-X expects the frozen edge-free prefix")

    candidate_texts = [GM._append_text(text, cfg.text_chars)
                       for text in delta["texts"]]
    raw_text = _normalize(torch.as_tensor(encoder.encode(candidate_texts),
                                          dtype=torch.float32, device=device))
    prompt_emb = prompt_emb.detach().float().to(device)
    propagated = C.propagated_features(delta, (0.0, 0.3, 0.9), device)
    raw_adj = P.normalized_adjacency(delta.get(
        "edge_index", torch.empty((2, 0), dtype=torch.long)), len(raw_x), device)
    synthetic_adj = P.normalized_adjacency(
        torch.empty((2, 0), dtype=torch.long), len(x0), device)

    torch.manual_seed(int(cfg.seed) + int(cfg.proxy_seed))
    model = P.GraphTextPromptStudent(
        int(x0.shape[1]), hidden=int(cfg.proxy_hidden),
        out=int(cfg.proxy_out), text_dim=int(raw_text.shape[1])).to(device).eval()
    params = tuple(model.parameters())
    reference = GM._make_reference(delta, parent)
    reference = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in reference.items()}
    with torch.enable_grad():
        real_loss = GM._multimodal_loss(model, reference, prompt_emb)
        real_grads = tuple(value.detach() for value in torch.autograd.grad(
            real_loss, params, create_graph=False, retain_graph=False))

    new_x = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([new_x], lr=float(cfg.lr_x))
    old_n = 0 if parent is None else len(parent["x"])
    old_device = _device_memory(parent, device)
    trace, e_trace = [], []
    selected = torch.zeros(len(labels), dtype=torch.long, device=device)
    q = torch.empty((len(raw_x), len(labels)), device=device)
    current_text = _normalize(init_text_emb.detach().float().to(device))
    latest_soft_text = current_text
    total_steps = max(1, int(cfg.em_rounds)) * max(1, int(cfg.m_steps))

    for em_round in range(max(1, int(cfg.em_rounds))):
        q, selected, e_audit = _e_step(
            delta, new_x.detach(), labels, raw_text, current_text, propagated,
            model, raw_adj, synthetic_adj, cfg)
        current_text = raw_text[selected]
        latest_soft_text = e_audit.pop("soft_text")
        target_x = _geometry_target(q, propagated).detach()
        e_trace.append({"round": em_round + 1,
                        "selected_local_rows": selected.detach().cpu().tolist(),
                        **e_audit})
        syn_text = current_text
        syn_labels = labels

        for m_step in range(max(1, int(cfg.m_steps))):
            if old_device is None:
                full_x, full_text, full_labels = new_x, syn_text, syn_labels
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                full_x = torch.cat((old_device["x"].float(), new_x), dim=0)
                full_text = torch.cat((old_device["text_emb"].float(), syn_text), dim=0)
                full_labels = torch.cat((old_device["labels"].long(), syn_labels), dim=0)
                old_edges = old_device.get("edge_index",
                    torch.empty((2, 0), dtype=torch.long, device=device)).long()
                edge_index = old_edges
            synthetic = {"x": full_x, "text_emb": full_text,
                         "labels": full_labels, "edge_index": edge_index}
            if use_gradient_matching:
                syn_loss = GM._multimodal_loss(model, synthetic, prompt_emb)
                syn_grads = torch.autograd.grad(
                    syn_loss, params, create_graph=True, retain_graph=True)
                gm_loss = GM._gradient_distance_blockwise(
                    model, real_grads, syn_grads)
                cos, norm_ratio = _gradient_stats(real_grads, syn_grads)
            else:
                syn_grads = None
                gm_loss = new_x.sum() * 0.0
                cos, norm_ratio = None, None
            if use_geometry:
                geometry_loss = (((new_x - target_x) /
                                  target_x.norm(dim=1, keepdim=True).clamp_min(1e-3))
                                 .square().mean())
            else:
                geometry_loss = new_x.sum() * 0.0
            objective = (gm_loss + float(cfg.geometry_weight) * geometry_loss)
            if not torch.isfinite(objective):
                raise FloatingPointError("non-finite EM-GECC-X M-step objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            grad_norm = float(new_x.grad.norm().detach().cpu()) if new_x.grad is not None else 0.0
            torch.nn.utils.clip_grad_norm_([new_x], max_norm=5.0)
            optimizer.step()
            step = em_round * max(1, int(cfg.m_steps)) + m_step + 1
            trace.append({"step": step, "em_round": em_round + 1,
                          "gm": float(gm_loss.detach().cpu()),
                          "geometry": float(geometry_loss.detach().cpu()),
                          "objective": float(objective.detach().cpu()),
                          "gradient_cosine": cos,
                          "gradient_norm_ratio": norm_ratio,
                          "x_gradient_norm": grad_norm})

    # One final E-step ensures saved payloads correspond to the final X.
    q, selected, final_e = _e_step(
        delta, new_x.detach(), labels, raw_text, current_text, propagated,
        model, raw_adj, synthetic_adj, cfg)
    soft_text = final_e.pop("soft_text")
    selected_cpu = selected.detach().cpu()
    source_local = selected_cpu.tolist()
    texts = [candidate_texts[i] for i in source_local]
    source_ids = [int(delta["ids"][i]) for i in source_local]
    hard_text = raw_text[selected].detach()
    soft_hard = F.cosine_similarity(soft_text, hard_text, dim=1)
    pairwise = hard_text @ hard_text.T
    off_diag = ~torch.eye(len(hard_text), dtype=torch.bool, device=device)
    text_cos = float(pairwise[off_diag].mean().cpu()) if len(hard_text) > 1 else 1.0
    x_norm = new_x.detach()
    class_centers = torch.stack([x_norm[labels == cls].mean(0)
                                 for cls in torch.unique(labels, sorted=True)])
    center_cos = class_centers @ class_centers.T
    center_off = ~torch.eye(len(class_centers), dtype=torch.bool, device=device)
    block = {
        "x": new_x.detach().cpu().float(),
        "text_emb": hard_text.detach().cpu().float(),
        "texts": texts,
        "labels": labels.detach().cpu(),
        "mass": torch.ones(len(labels)),
        "depth": torch.zeros(len(labels), dtype=torch.long),
        "stage": torch.full((len(labels),), int(delta["stage"]), dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "text_source_ids": [[value] for value in source_ids],
    }
    if parent is not None:
        result = {key: torch.cat((parent[key].detach().cpu(), block[key]), dim=0)
                  for key in ("x", "text_emb", "labels", "mass", "depth", "stage")}
        result["texts"] = list(parent["texts"]) + list(block["texts"])
        result["edge_index"] = parent.get("edge_index",
            torch.empty((2, 0), dtype=torch.long)).detach().cpu().long().clone()
        result["text_source_ids"] = (list(parent.get("text_source_ids", []))
                                     + list(block["text_source_ids"]))
    else:
        result = block
    audit = {
        "method": "EM-GECC-X",
        "status": "mechanism-adapted experimental branch; not official GECC",
        "objective_variant": ("full" if use_gradient_matching and use_geometry
                              else "no-GM" if use_geometry else "no-geometry"),
        "initialization": "GECC hard-cluster centers on current-delta propagated features",
        "e_step": "class-masked capacity-balanced Q from propagated-feature, text, and fixed-Student embedding compatibility",
        "m_step": "free new-X update by one-step Student gradient matching plus optional Q-conditioned geometry constraint",
        "historical_raw_reads": 0,
        "old_prefix_frozen": True,
        "old_x_update_norm": 0.0,
        "new_x_update_norm": float((new_x.detach() - x0).norm().cpu()),
        "total_m_steps": total_steps,
        "q_row_sum_max_error": final_e["q_row_sum_max_error"],
        "q_slot_mass_cv": final_e["q_slot_mass_cv"],
        "q_entropy_mean": final_e["q_entropy_mean"],
        "q_entropy_normalized": final_e["q_entropy_normalized"],
        "slot_text_coverage": len(set(source_ids)) / max(1, len(source_ids)),
        "selected_text_ids": source_ids,
        "selected_text_pairwise_cosine_mean": text_cos,
        "soft_hard_text_cosine_mean": float(soft_hard.mean().cpu()),
        "soft_hard_text_cosine_min": float(soft_hard.min().cpu()),
        "class_center_cosine_mean": (float(center_cos[center_off].mean().cpu())
                                      if bool(center_off.any()) else None),
        "final_feature_score_mean": final_e["score_feature_mean"],
        "final_text_score_mean": final_e["score_text_mean"],
        "final_student_score_mean": final_e["score_student_mean"],
        "trace": trace,
        "e_trace": e_trace,
        "config": asdict(cfg),
    }
    return result, audit
