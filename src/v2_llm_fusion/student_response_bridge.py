"""Minimal Student-response bridge for CaT-style continual construction.

The bridge deliberately keeps the experiment narrow.  A sealed synthetic
parent and the current CaT block are concatenated with edge-free topology;
only the new block's ``X`` and/or continuous text embedding can move.  A short
differentiable Adam unroll of the canonical GraphCLIP Student supplies the
outer response signal.  No historical raw graph, P/A assignment, topology
revision, budget change, or selector is introduced here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

import cat_multimodal_minimal as MM
import direct_student_outer as DSO


@dataclass(frozen=True)
class ResponseConfig:
    inner_steps: int = 8
    inner_lr: float = 1e-3
    inner_batch: int = 16
    outer_steps: int = 4
    outer_lr: float = 2e-2
    response_seeds: tuple[int, ...] = (9300, 9301)
    old_weight: float = 0.5
    new_weight: float = 1.0
    prompt_weight: float = 1.0
    support_weight: float = 1.0
    support_prompt_weight: float = 1.0
    new_margin_weight: float = 0.0
    new_margin: float = 0.20
    lambda_anchor_x: float = 0.05
    lambda_anchor_text: float = 0.05
    x_radius: float = 0.20
    text_radius: float = 0.25
    mmd_weight: float = 0.0
    mmd_hidden: int = 128
    mmd_probes: int = 4
    seed: int = 271119
    max_target_rows: int = 64
    max_context_rows: int = 256

    def __post_init__(self) -> None:
        if self.inner_steps < 1 or self.outer_steps < 1:
            raise ValueError("inner_steps and outer_steps must be positive")
        if self.inner_lr <= 0 or self.outer_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.inner_batch < 2:
            raise ValueError("inner_batch must be at least two")
        if not self.response_seeds:
            raise ValueError("response_seeds must not be empty")
        if set(self.response_seeds) != set(int(v) for v in self.response_seeds):
            raise ValueError("response seeds must be integer-like")
        for name in ("old_weight", "new_weight", "prompt_weight",
                     "support_weight", "support_prompt_weight",
                     "new_margin_weight", "new_margin",
                     "lambda_anchor_x", "lambda_anchor_text", "x_radius",
                     "text_radius", "mmd_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if (self.old_weight == 0 and self.new_weight == 0
                and self.new_margin_weight == 0):
            raise ValueError("at least one response weight must be positive")
        if self.mmd_hidden < 1 or self.mmd_probes < 1:
            raise ValueError("invalid MMD probe configuration")
        if self.max_target_rows < 1 or self.max_context_rows < 2:
            raise ValueError("invalid response row limits")


def _empty_edges(device: str) -> torch.Tensor:
    # The canonical ``norm_adj`` helper concatenates CPU self-loop indices
    # before moving the result to the requested device.  Keep this empty edge
    # tensor CPU-side so the bridge follows that existing contract.
    del device
    return torch.empty((2, 0), dtype=torch.long)


def _row_subset(old_rows: int, total: int, max_rows: int,
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Select deterministic old/new target rows without using labels."""
    new_start = int(old_rows)
    old = torch.arange(min(new_start, int(max_rows)), device=device)
    remaining = max(1, int(max_rows) - len(old))
    new = torch.arange(new_start,
                       min(int(total), new_start + remaining),
                       device=device)
    return old, new


def _context_rows(old_rows: int, total: int, max_rows: int,
                  device: str) -> torch.Tensor:
    if total <= max_rows:
        return torch.arange(total, device=device)
    # Keep both sides of the continual boundary in the negative context.
    old_budget = min(int(old_rows), max_rows // 2)
    new_budget = max_rows - old_budget
    old = torch.arange(old_budget, device=device)
    new = torch.arange(int(old_rows),
                       min(total, int(old_rows) + new_budget), device=device)
    context = torch.cat((old, new))
    if len(context) < 2:
        return torch.arange(min(total, max_rows), device=device)
    return context


def _balanced_support_rows(labels: torch.Tensor, max_rows: int,
                           device: str) -> torch.Tensor:
    """Choose a deterministic, class-balanced legal current-delta subset."""
    labels = labels.detach().cpu().long()
    chosen: list[int] = []
    classes = torch.unique(labels, sorted=True).tolist()
    for offset in range(max(1, int(max_rows))):
        for cls in classes:
            matches = torch.where(labels == int(cls))[0]
            if offset < len(matches) and len(chosen) < int(max_rows):
                chosen.append(int(matches[offset]))
    if not chosen:
        raise ValueError("support subset is empty")
    return torch.tensor(chosen, dtype=torch.long, device=device)


def _response_for_seed(x: torch.Tensor, text: torch.Tensor,
                       adjacency: torch.Tensor, old_rows: int,
                       seed: int, cfg: ResponseConfig,
                       create_graph: bool,
                       prompt_emb: torch.Tensor | None = None,
                       prompt_ids: torch.Tensor | None = None,
                       new_labels: torch.Tensor | None = None,
                       support_x: torch.Tensor | None = None,
                       support_text: torch.Tensor | None = None,
                       support_labels: torch.Tensor | None = None,
                       support_adjacency: torch.Tensor | None = None) -> dict:
    dso_cfg = DSO.Config(
        inner_steps=cfg.inner_steps,
        inner_lr=cfg.inner_lr,
        inner_batch=cfg.inner_batch,
        outer_steps=1,
        outer_lr=cfg.outer_lr,
        train_seeds=(int(seed),),
        validation_seeds=(int(seed) + 1000003,),
    )
    unrolled = DSO.unroll_student(
        x, text, adjacency, int(seed), dso_cfg, create_graph=create_graph)
    params = unrolled["params"]
    graph = DSO._functional_graph(params, x, adjacency)
    encoded = DSO._functional_text(params, text)
    context = _context_rows(old_rows, len(x), cfg.max_context_rows, x.device)
    old_targets, new_targets = _row_subset(
        old_rows, len(x), cfg.max_target_rows, x.device)
    zero = graph.sum() * 0.0
    old_loss = zero
    if len(old_targets) >= 1 and len(context) >= 2:
        old_loss = DSO._nce(graph, encoded, old_targets, params["logit_scale"],
                             context_rows=context)
    new_loss = zero
    if len(new_targets) >= 1 and len(context) >= 2:
        new_loss = DSO._nce(graph, encoded, new_targets, params["logit_scale"],
                             context_rows=context)
    prompt_loss = zero
    if prompt_emb is not None and new_labels is not None and len(new_targets):
        prompt_emb = prompt_emb.to(x.device).float()
        prompt_ids = (torch.arange(len(prompt_emb), device=x.device,
                                    dtype=torch.long)
                      if prompt_ids is None else prompt_ids.to(x.device).long())
        encoded_prompt = DSO._functional_text(params, prompt_emb)
        local = (new_targets - int(old_rows)).long()
        target_classes = new_labels.to(x.device).long().index_select(0, local)
        positions = []
        for class_id in target_classes.detach().cpu().tolist():
            matches = torch.where(prompt_ids == int(class_id))[0]
            if len(matches) != 1:
                raise ValueError(f"cannot resolve prompt for class {class_id}")
            positions.append(int(matches[0]))
        target = torch.tensor(positions, dtype=torch.long, device=x.device)
        logits = (params["logit_scale"].exp().clamp(max=100)
                  * (graph[new_targets] @ encoded_prompt.T))
        prompt_loss = F.cross_entropy(logits, target)
    new_margin_loss = zero
    if prompt_emb is not None and new_labels is not None and len(new_targets):
        # The margin is evaluated on the same post-unroll graph/text prompt
        # logits used by the downstream zero-shot report.  It gives X a
        # class-boundary signal instead of only asking it to match support
        # rows or synthetic row identities.
        prompt_emb = prompt_emb.to(x.device).float()
        prompt_ids = (torch.arange(len(prompt_emb), device=x.device,
                                    dtype=torch.long)
                      if prompt_ids is None else prompt_ids.to(x.device).long())
        encoded_prompt = DSO._functional_text(params, prompt_emb)
        local = (new_targets - int(old_rows)).long()
        target_classes = new_labels.to(x.device).long().index_select(0, local)
        positions = []
        for class_id in target_classes.detach().cpu().tolist():
            matches = torch.where(prompt_ids == int(class_id))[0]
            if len(matches) != 1:
                raise ValueError(f"cannot resolve prompt for class {class_id}")
            positions.append(int(matches[0]))
        target = torch.tensor(positions, dtype=torch.long, device=x.device)
        logits = (params["logit_scale"].exp().clamp(max=100)
                  * (graph[new_targets] @ encoded_prompt.T))
        positive = logits.gather(1, target[:, None]).squeeze(1)
        negative_logits = logits.clone()
        negative_logits.scatter_(1, target[:, None], float("-inf"))
        hardest_negative = negative_logits.max(dim=1).values
        new_margin_loss = F.relu(
            float(cfg.new_margin) - positive + hardest_negative).mean()
    support_loss = zero
    support_prompt_loss = zero
    if (support_x is not None and support_text is not None
            and support_labels is not None):
        support_x = support_x.to(x.device).float()
        support_text = support_text.to(x.device).float()
        support_labels = support_labels.to(x.device).long()
        support_adjacency = (support_adjacency if support_adjacency is not None
                             else DSO.fixed_norm_adj(
                                 torch.empty((2, 0), dtype=torch.long),
                                 len(support_x), str(x.device)))
        support_graph = DSO._functional_graph(
            params, support_x, support_adjacency)
        support_encoded = DSO._functional_text(params, support_text)
        support_rows = _balanced_support_rows(
            support_labels, cfg.max_target_rows, x.device)
        support_context = _context_rows(
            0, len(support_x), cfg.max_context_rows, x.device)
        support_loss = DSO._nce(
            support_graph, support_encoded, support_rows,
            params["logit_scale"], context_rows=support_context)
        if prompt_emb is not None and len(support_rows):
            prompt_emb = prompt_emb.to(x.device).float()
            prompt_ids = (torch.arange(len(prompt_emb), device=x.device,
                                       dtype=torch.long)
                          if prompt_ids is None else prompt_ids.to(x.device).long())
            encoded_prompt = DSO._functional_text(params, prompt_emb)
            target_classes = support_labels.index_select(0, support_rows)
            positions = []
            for class_id in target_classes.detach().cpu().tolist():
                matches = torch.where(prompt_ids == int(class_id))[0]
                if len(matches) != 1:
                    raise ValueError(f"cannot resolve prompt for class {class_id}")
                positions.append(int(matches[0]))
            target = torch.tensor(positions, dtype=torch.long, device=x.device)
            support_logits = (params["logit_scale"].exp().clamp(max=100)
                              * (support_graph[support_rows]
                                 @ encoded_prompt.T))
            support_prompt_loss = F.cross_entropy(support_logits, target)
    response = (float(cfg.old_weight) * old_loss
                + float(cfg.new_weight) * new_loss
                + float(cfg.prompt_weight) * prompt_loss
                + float(cfg.support_weight) * support_loss
                + float(cfg.support_prompt_weight) * support_prompt_loss
                + float(cfg.new_margin_weight) * new_margin_loss)
    return {
        "response": response,
        "old": old_loss,
        "new": new_loss,
        "prompt": prompt_loss,
        "new_margin": new_margin_loss,
        "support": support_loss,
        "support_prompt": support_prompt_loss,
        "inner_final": unrolled["inner_losses"][-1],
    }


def _probe_weights(dim: int, cfg: ResponseConfig, device: str) -> list[torch.Tensor]:
    bound = math.sqrt(6.0 / float(dim + cfg.mmd_hidden))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 17)
    return [torch.empty((dim, cfg.mmd_hidden), device=device).uniform_(
        -bound, bound, generator=generator)
        for _ in range(int(cfg.mmd_probes))]


def _mmd(raw: torch.Tensor, synth: torch.Tensor, raw_labels: torch.Tensor,
         synth_labels: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    values = [MM._class_mmd(raw, synth, raw_labels, synth_labels, weight)
              for weight in weights]
    return torch.stack(values).mean()


def _project_x_delta(delta: torch.Tensor, x0: torch.Tensor,
                     radius: float) -> None:
    if radius <= 0:
        delta.zero_()
        return
    base_norm = x0.norm(dim=1).clamp_min(1.0)
    max_norm = float(radius) * base_norm
    norm = delta.norm(dim=1).clamp_min(1e-12)
    factor = (max_norm / norm).clamp(max=1.0)
    delta.mul_(factor.unsqueeze(1))


def _project_text_delta(delta: torch.Tensor, radius: float) -> None:
    if radius <= 0:
        delta.zero_()
        return
    norm = delta.norm(dim=1).clamp_min(1e-12)
    factor = (float(radius) / norm).clamp(max=1.0)
    delta.mul_(factor.unsqueeze(1))


def _candidate(parent: dict | None, x_new: torch.Tensor,
               text_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if parent is None:
        return x_new, text_new
    old_x = parent["x"].detach().to(x_new.device).float()
    old_t = parent["text_emb"].detach().to(text_new.device).float()
    return torch.cat((old_x, x_new)), torch.cat((old_t, text_new))


def optimize_response_block(
    delta: dict,
    source_features: torch.Tensor,
    x0: torch.Tensor,
    text0: torch.Tensor,
    texts: list[str],
    source_ids: list[list[int]],
    labels: torch.Tensor,
    parent: dict | None,
    cfg: ResponseConfig,
    mode: str,
    mass: torch.Tensor | None,
    device: str,
    prompt_emb: torch.Tensor | None = None,
    prompt_ids: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    """Optimize only the current block using a short Student response.

    ``response_t_only`` updates text payloads; ``response_xt`` updates both
    payloads; ``mmd_response_xt`` adds the existing current-delta CaT MMD as a
    stabilizing term.  The parent prefix is detached and never edited.
    """
    valid_modes = ("response_t_only", "response_x_only", "response_xt",
                   "mmd_response_xt", "support_t_only", "support_x_only",
                   "support_xt", "mmd_support_x_only", "mmd_support_xt")
    valid_modes = valid_modes + ("new_margin_x_only",)
    if mode not in valid_modes:
        raise ValueError(f"unknown response bridge mode: {mode}")
    x0 = x0.detach().to(device).float()
    text0 = F.normalize(text0.detach().to(device).float(), dim=1)
    source_features = source_features.detach().to(device).float()
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_labels = delta["y"].detach().long().to(device)
    labels = labels.detach().long().to(device)
    if prompt_emb is not None:
        prompt_emb = prompt_emb.detach().to(device).float()
    if prompt_ids is not None:
        prompt_ids = prompt_ids.detach().long().to(device)
    if parent is not None and parent["edge_index"].numel():
        raise ValueError("response bridge requires edge-free parent topology")

    x_delta = torch.nn.Parameter(torch.zeros_like(x0))
    text_delta = torch.nn.Parameter(torch.zeros_like(text0))
    text_editable = not mode.endswith("_x_only")
    x_editable = mode.endswith("_xt") or mode.endswith("_x_only")
    parameters = [text_delta] if text_editable else []
    if x_editable:
        parameters.insert(0, x_delta)
    optimizer = torch.optim.Adam(parameters, lr=float(cfg.outer_lr))
    adjacency = DSO.fixed_norm_adj(_empty_edges(device),
                                   (0 if parent is None else len(parent["x"]))
                                   + len(x0), device)
    support_adjacency = DSO.fixed_norm_adj(
        _empty_edges(device), len(source_features), device)
    support_mode = mode.startswith("support_") or mode.startswith("mmd_support_")
    support_x = source_features if support_mode else None
    support_t = raw_text if support_mode else None
    support_y = raw_labels if support_mode else None
    x_weights = _probe_weights(source_features.shape[1], cfg, device)
    text_weights = _probe_weights(raw_text.shape[1], cfg,
                                  device) if cfg.mmd_weight > 0 else []
    trace = []

    def values() -> tuple[torch.Tensor, torch.Tensor]:
        x_new = x0 + x_delta
        text_new = F.normalize(text0 + text_delta, dim=1)
        return _candidate(parent, x_new, text_new)

    for outer_step in range(max(1, int(cfg.outer_steps))):
        candidate_x, candidate_text = values()
        seed_parts = [_response_for_seed(
            candidate_x, candidate_text, adjacency,
            0 if parent is None else len(parent["x"]), seed, cfg,
            create_graph=True, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids, new_labels=labels,
            support_x=support_x, support_text=support_t,
            support_labels=support_y, support_adjacency=support_adjacency)
            for seed in cfg.response_seeds]
        old_loss = torch.stack([part["old"] for part in seed_parts]).mean()
        new_loss = torch.stack([part["new"] for part in seed_parts]).mean()
        prompt_loss = torch.stack([part["prompt"] for part in seed_parts]).mean()
        support_loss = torch.stack([part["support"] for part in seed_parts]).mean()
        support_prompt_loss = torch.stack([
            part["support_prompt"] for part in seed_parts]).mean()
        new_margin_loss = torch.stack([
            part["new_margin"] for part in seed_parts]).mean()
        response_loss = torch.stack([part["response"] for part in seed_parts]).mean()
        text_value = candidate_text[-len(x0):]
        anchor_x = x_delta.square().mean()
        anchor_text = (1.0 - F.cosine_similarity(text_value, text0, dim=1)).mean()
        mmd_x = candidate_x[-len(x0):].sum() * 0.0
        mmd_text = candidate_text[-len(x0):].sum() * 0.0
        if cfg.mmd_weight > 0:
            mmd_x = _mmd(source_features, candidate_x[-len(x0):],
                         raw_labels, labels, x_weights)
            mmd_text = _mmd(raw_text, text_value, raw_labels, labels,
                             text_weights)
        loss = (response_loss
                + float(cfg.lambda_anchor_x) * anchor_x
                + float(cfg.lambda_anchor_text) * anchor_text
                + float(cfg.mmd_weight) * (mmd_x + mmd_text))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Student response bridge loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _project_x_delta(x_delta, x0, cfg.x_radius)
            if text_editable:
                _project_text_delta(text_delta, cfg.text_radius)
        trace.append({
            "outer_step": outer_step + 1,
            "loss": float(loss.detach().cpu()),
            "response": float(response_loss.detach().cpu()),
            "old_response": float(old_loss.detach().cpu()),
            "new_response": float(new_loss.detach().cpu()),
            "prompt_response": float(prompt_loss.detach().cpu()),
            "support_response": float(support_loss.detach().cpu()),
            "support_prompt_response": float(
                support_prompt_loss.detach().cpu()),
            "new_margin_response": float(new_margin_loss.detach().cpu()),
            "x_anchor": float(anchor_x.detach().cpu()),
            "text_anchor": float(anchor_text.detach().cpu()),
            "x_mmd": float(mmd_x.detach().cpu()),
            "text_mmd": float(mmd_text.detach().cpu()),
            "inner_final": float(torch.stack([
                part["inner_final"].detach() for part in seed_parts]).mean().cpu()),
        })

    with torch.no_grad():
        final_x, final_text = values()
    final_x_new = final_x[-len(x0):].detach().cpu()
    final_text_new = F.normalize(final_text[-len(x0):].detach(), dim=1).cpu()
    final_response = []
    # ``create_graph=False`` removes second-order edges, but the inner
    # unroll still needs first-order autograd.  Do not wrap this pass in
    # ``torch.no_grad()``.
    for seed in cfg.response_seeds:
        part = _response_for_seed(
            final_x.detach(), final_text.detach(), adjacency,
            0 if parent is None else len(parent["x"]), seed, cfg,
            create_graph=False, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids, new_labels=labels,
            support_x=support_x, support_text=support_t,
            support_labels=support_y, support_adjacency=support_adjacency)
        final_response.append({
            "seed": int(seed),
            "old": float(part["old"].detach().cpu()),
            "new": float(part["new"].detach().cpu()),
            "prompt": float(part["prompt"].detach().cpu()),
            "new_margin": float(part["new_margin"].detach().cpu()),
            "support": float(part["support"].detach().cpu()),
            "support_prompt": float(part["support_prompt"].detach().cpu()),
            "response": float(part["response"].detach().cpu()),
        })

    block = {
        "x": final_x_new,
        "text_emb": final_text_new,
        "texts": list(texts),
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x_new))),
        "depth": torch.zeros(len(final_x_new), dtype=torch.long),
        "stage": torch.full((len(final_x_new),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    audit = {
        "constructor": f"cat_student_response_{mode}",
        "student_response_enabled": True,
        "response_objective": "short differentiable canonical GCN + bidirectional InfoNCE unroll",
        "prompt_objective": "new-block graph response against frozen public class prompts",
        "new_margin_objective": (
            "hinge margin between the new-class prompt and hardest competing prompt"),
        "support_response_objective": (
            "current-delta propagated graph/text response when using support_* mode"),
        "response_seeds": [int(v) for v in cfg.response_seeds],
        "inner_steps": int(cfg.inner_steps),
        "outer_steps": int(cfg.outer_steps),
        "old_rows": 0 if parent is None else len(parent["x"]),
        "editable_rows": len(x0),
        "historical_raw_reads": 0,
        "old_prefix_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "mode": mode,
        "editable_components": ([("X" if x_editable else "")
                                  , ("T" if text_editable else "")]),
        "config": asdict(cfg),
        "row_classes": labels.cpu().tolist(),
        "text_source_ids": source_ids,
        "trace": trace,
        "final_response": final_response,
        "x_update_norm": float((final_x_new - x0.cpu()).norm()),
        "text_update_norm": float((final_text_new - text0.cpu()).norm()),
        "max_x_row_relative_delta": float((final_x_new - x0.cpu()).norm(dim=1).div(
            x0.cpu().norm(dim=1).clamp_min(1.0)).max()),
        "final_text_norm_mean": float(final_text_new.norm(dim=1).mean()),
    }
    return block, audit

