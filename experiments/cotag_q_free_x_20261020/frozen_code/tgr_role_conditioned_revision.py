"""Role-conditioned bounded revision of synthetic node features.

This module is the conservative next step after the aligned TGR-v3 arm.  It
keeps the initial sparse assignment ``P`` and the base topology fixed, but it
does allow historical synthetic rows to move.  Rows receive construction-time
roles from semantic boundary proximity, teacher gradient sensitivity, and
degree.  Core rows get a small trust region, boundary rows get a larger one,
and new rows use the normal new-row radius.

The objective preserves old class centers and class-to-class relations while
still optimizing the new block through a differentiable Student unroll.  No
final F1, report labels, or historical raw graph is used for role selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as F

import teacher_guided_message_revision as M
import teacher_guided_payload_revision as T


@dataclass(frozen=True)
class Config:
    core_fraction: float = 0.50
    boundary_fraction: float = 0.15
    core_radius: float = 0.02
    regular_radius: float = 0.05
    boundary_radius: float = 0.08
    new_radius: float = 0.15
    boundary_weight: float = 0.55
    gradient_weight: float = 0.35
    degree_weight: float = 0.10
    old_center_weight: float = 0.25
    old_relation_weight: float = 0.20
    old_text_weight: float = 0.10
    role_anchor_weight: float = 0.05
    old_probe_weight: float = 1.00
    new_probe_weight: float = 0.50
    raw_new_probe_weight: float = 1.00
    raw_class_weight: float = 1.00
    raw_prompt_weight: float = 1.00
    old_prompt_weight: float = 2.00
    old_prompt_tolerance: float = 0.0
    gradient_surgery: bool = False
    old_probe_tolerance: float = 0.0
    response_alignment_gate: bool = False
    response_alignment_threshold: float = 0.0
    gate_margin: float = 0.0
    role_seed: int = 9440

    def __post_init__(self) -> None:
        if not 0.0 < float(self.core_fraction) <= 1.0:
            raise ValueError("core_fraction must be in (0, 1]")
        if not 0.0 <= float(self.boundary_fraction) <= 1.0:
            raise ValueError("boundary_fraction must be in [0, 1]")
        if float(self.core_fraction) + float(self.boundary_fraction) > 1.0 + 1e-8:
            raise ValueError("core and boundary fractions exceed one")
        for name in ("core_radius", "regular_radius", "boundary_radius",
                     "new_radius", "boundary_weight", "gradient_weight",
                     "degree_weight", "old_center_weight",
                     "old_relation_weight", "old_text_weight",
                     "role_anchor_weight", "old_probe_weight",
                     "new_probe_weight", "raw_new_probe_weight",
                     "raw_class_weight", "raw_prompt_weight",
                     "old_prompt_weight", "old_prompt_tolerance",
                     "old_probe_tolerance", "gate_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not isinstance(self.response_alignment_gate, bool):
            raise ValueError("response_alignment_gate must be boolean")
        if not isinstance(self.gradient_surgery, bool):
            raise ValueError("gradient_surgery must be boolean")
        if not math.isfinite(float(self.response_alignment_threshold)):
            raise ValueError("invalid response_alignment_threshold")

    def dict(self) -> dict:
        return asdict(self)


def _normalize_score(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if value.numel() == 0:
        return value
    lo, hi = value.min(), value.max()
    return (value - lo) / (hi - lo).clamp_min(1e-8)


def _old_groups(class_id: torch.Tensor, old_rows: int) -> dict[int, torch.Tensor]:
    values = class_id[:int(old_rows)].long().cpu()
    known = sorted(int(v) for v in values.unique().tolist() if int(v) >= 0)
    if not known:
        return {-1: torch.arange(int(old_rows), dtype=torch.long)}
    return {value: torch.where(values == value)[0] for value in known
            if bool((values == value).any())}


def _class_centers(z: torch.Tensor, class_id: torch.Tensor,
                   old_rows: int) -> tuple[list[int], torch.Tensor]:
    groups = _old_groups(class_id, old_rows)
    classes, centers = [], []
    for value, rows in groups.items():
        if len(rows):
            classes.append(int(value))
            centers.append(z[rows.to(z.device)].mean(0))
    if not centers:
        return [], z.new_empty((0, z.shape[1]))
    return classes, F.normalize(torch.stack(centers), dim=1)


def _old_relation_loss(candidate_z: torch.Tensor, target_z: torch.Tensor,
                       class_id: torch.Tensor, old_rows: int) -> torch.Tensor:
    classes_c, centers_c = _class_centers(candidate_z[:int(old_rows)], class_id,
                                           old_rows)
    classes_t, centers_t = _class_centers(target_z[:int(old_rows)], class_id,
                                           old_rows)
    if not classes_c or classes_c != classes_t or len(classes_c) < 2:
        return candidate_z.new_zeros(())
    relation_c = centers_c @ centers_c.T
    relation_t = centers_t @ centers_t.T
    mask = ~torch.eye(len(classes_c), dtype=torch.bool, device=candidate_z.device)
    return F.mse_loss(relation_c[mask], relation_t[mask])


def _old_center_loss(candidate_z: torch.Tensor, target_z: torch.Tensor,
                     class_id: torch.Tensor, old_rows: int) -> torch.Tensor:
    groups = _old_groups(class_id, old_rows)
    values = []
    for rows in groups.values():
        if not len(rows):
            continue
        index = rows.to(candidate_z.device)
        values.append(1.0 - F.cosine_similarity(
            candidate_z[index].mean(0, keepdim=True),
            target_z[index].mean(0, keepdim=True), dim=1, eps=1e-8).mean())
    return torch.stack(values).mean() if values else candidate_z.new_zeros(())


def _old_text_consistency(candidate_z: torch.Tensor,
                          candidate_text_z: torch.Tensor,
                          class_id: torch.Tensor,
                          old_rows: int) -> torch.Tensor:
    groups = _old_groups(class_id, old_rows)
    values = []
    for rows in groups.values():
        if not len(rows):
            continue
        index = rows.to(candidate_z.device)
        graph_center = candidate_z[index].mean(0, keepdim=True)
        text_center = candidate_text_z[index].mean(0, keepdim=True)
        values.append(1.0 - F.cosine_similarity(
            graph_center, text_center, dim=1, eps=1e-8).mean())
    return torch.stack(values).mean() if values else candidate_z.new_zeros(())


def _teacher_gradient_score(memory: dict, class_id: torch.Tensor,
                           old_rows: int, reference: dict, device: str,
                           teacher_cfg: T.Config) -> torch.Tensor:
    """Estimate row sensitivity using one fixed Teacher loss batch."""
    x = memory["x"].to(device).float().detach().requires_grad_(True)
    text = memory["text_emb"].to(device).float()
    adjacency = T.G.norm_adj(memory["edge_index"], len(x), device)
    groups = M._groups_with_provenance(class_id, old_rows)
    rows = M._rows_with_class_mix(
        groups, min(int(teacher_cfg.batch), len(x)),
        int(reference["seed"]) * 1013 + 7, device, old_rows,
        float(teacher_cfg.old_class_mix))
    loss = T.contrastive_loss(reference["params_t"], x, text, adjacency, rows)
    grad, = torch.autograd.grad(loss, x, allow_unused=False)
    return grad[:int(old_rows)].norm(dim=1).detach().cpu()


def select_roles(base: dict, support: dict, class_id: torch.Tensor,
                 old_rows: int, reference: dict, device: str,
                 teacher_cfg: T.Config, cfg: Config,
                 gradient_memory: dict | None = None,
                 gradient_class_id: torch.Tensor | None = None) -> dict:
    """Select deterministic core/regular/boundary roles from construction data."""
    old_rows = int(old_rows)
    if old_rows <= 0:
        return {
            "role": torch.empty(0, dtype=torch.long),
            "score": torch.empty(0),
            "boundary_score": torch.empty(0),
            "gradient_score": torch.empty(0),
            "degree_score": torch.empty(0),
            "radius": torch.empty(0),
        }
    old_x = F.normalize(base["x"][:old_rows].float(), dim=1)
    old_t = F.normalize(base["text_emb"][:old_rows].float(), dim=1)
    new_x = F.normalize(support["x"][old_rows:].float(), dim=1)
    new_t = F.normalize(support["text_emb"][old_rows:].float(), dim=1)
    if len(new_x):
        old_joint = F.normalize(torch.cat((old_x, old_t), dim=1), dim=1)
        new_joint = F.normalize(torch.cat((new_x, new_t), dim=1), dim=1)
        boundary = (old_joint @ new_joint.T).max(dim=1).values
        boundary = _normalize_score(boundary)
    else:
        boundary = torch.zeros(old_rows)
    gradient_memory = support if gradient_memory is None else gradient_memory
    gradient_class_id = class_id if gradient_class_id is None else gradient_class_id
    gradient = _normalize_score(_teacher_gradient_score(
        gradient_memory, gradient_class_id, old_rows, reference, device,
        teacher_cfg))
    degree = torch.bincount(base["edge_index"][0].long(), minlength=len(base["x"]))[:old_rows]
    degree = _normalize_score(degree.float())
    score = (float(cfg.boundary_weight) * boundary +
             float(cfg.gradient_weight) * gradient +
             float(cfg.degree_weight) * degree)

    order = torch.argsort(score, descending=True, stable=True)
    boundary_count = min(old_rows, max(1, int(round(old_rows * cfg.boundary_fraction)))
                         if cfg.boundary_fraction > 0 else 0)
    core_count = min(old_rows - boundary_count,
                     max(0, int(round(old_rows * cfg.core_fraction))))
    role = torch.ones(old_rows, dtype=torch.long)
    if core_count:
        role[order[-core_count:]] = 0
    if boundary_count:
        role[order[:boundary_count]] = 2
    radius_values = torch.tensor(
        [cfg.core_radius, cfg.regular_radius, cfg.boundary_radius],
        dtype=torch.float32)
    return {
        "role": role,
        "score": score,
        "boundary_score": boundary,
        "gradient_score": gradient,
        "degree_score": degree,
        "radius": radius_values[role],
        "order_descending": order,
        "counts": {"core": int((role == 0).sum()),
                   "regular": int((role == 1).sum()),
                   "boundary": int((role == 2).sum())},
    }


def _project_role(relative: torch.Tensor, base_rows: torch.Tensor,
                  old_rows: int, roles: dict, new_radius: float) -> None:
    with torch.no_grad():
        scale = base_rows.norm(dim=1, keepdim=True).clamp_min(1e-6)
        limits = torch.full((len(base_rows), 1), float(new_radius),
                            device=relative.device, dtype=relative.dtype)
        if int(old_rows):
            limits[:int(old_rows), 0] = roles["radius"].to(relative.device)
        limits = limits * scale
        norms = relative.norm(dim=1, keepdim=True).clamp_min(1e-12)
        relative.mul_((limits / norms).clamp(max=1.0))


def _role_anchor(relative: torch.Tensor, base_rows: torch.Tensor,
                 old_rows: int, roles: dict) -> torch.Tensor:
    if int(old_rows) == 0:
        return relative.new_zeros(())
    scale = base_rows[:int(old_rows)].norm(dim=1, keepdim=True).clamp_min(1e-6)
    radius = roles["radius"].to(relative.device).unsqueeze(1).clamp_min(1e-6)
    normalized = relative[:int(old_rows)] / scale / radius
    return normalized.square().mean()


def _probe_rows(total_rows: int, old_rows: int) -> dict[str, torch.Tensor]:
    """Use synthetic-only old/new rows as a fixed construction-time probe."""
    return {
        "old": torch.arange(int(old_rows), dtype=torch.long),
        "new": torch.arange(int(old_rows), int(total_rows), dtype=torch.long),
    }


def _probe_baselines(base: dict, references: list[dict], old_rows: int,
                     candidate_groups: dict[int, torch.Tensor], device: str,
                     teacher_cfg: T.Config,
                     probe_rows: dict[str, torch.Tensor]) -> list[dict]:
    """Compute fixed Student losses on the unmodified synthetic memory."""
    base_x = base["x"].to(device)
    base_text = base["text_emb"].to(device)
    base_adj = T.G.norm_adj(base["edge_index"], len(base_x), device)
    result = []
    for ref in references:
        params = T._unroll(
            ref["params0"], base_x, base_text, base_adj, candidate_groups,
            ref["seed"], teacher_cfg, False,
            row_selector=M._rows_with_class_mix, old_rows=old_rows,
            old_class_mix=float(teacher_cfg.old_class_mix))
        values = {}
        for name, rows in probe_rows.items():
            if len(rows):
                values[name] = float(T.contrastive_loss(
                    params, base_x, base_text, base_adj,
                    rows.to(device)).detach())
            else:
                values[name] = 0.0
        result.append(values)
    return result


def _raw_delta_baselines(base: dict, support: dict, references: list[dict],
                         old_rows: int, candidate_groups: dict[int, torch.Tensor],
                         device: str, teacher_cfg: T.Config) -> list[float]:
    """Evaluate the base-trained Student on the currently arriving raw delta."""
    base_x = base["x"].to(device)
    base_text = base["text_emb"].to(device)
    base_adj = T.G.norm_adj(base["edge_index"], len(base_x), device)
    raw_x = support["x"][int(old_rows):].to(device).float()
    raw_text = support["text_emb"][int(old_rows):].to(device).float()
    raw_edge = support["edge_index"].long()
    keep = (raw_edge[0] >= int(old_rows)) & (raw_edge[1] >= int(old_rows))
    raw_edge = raw_edge[:, keep] - int(old_rows)
    raw_adj = T.G.norm_adj(raw_edge, len(raw_x), device)
    values = []
    for ref in references:
        params = T._unroll(
            ref["params0"], base_x, base_text, base_adj, candidate_groups,
            ref["seed"], teacher_cfg, False,
            row_selector=M._rows_with_class_mix, old_rows=old_rows,
            old_class_mix=float(teacher_cfg.old_class_mix))
        rows = torch.arange(len(raw_x), device=device, dtype=torch.long)
        values.append(float(T.contrastive_loss(
            params, raw_x, raw_text, raw_adj, rows).detach()))
    return values


def _raw_class_loss(params: dict, raw_x: torch.Tensor, raw_text: torch.Tensor,
                    raw_adj: torch.Tensor, raw_class: torch.Tensor) -> torch.Tensor:
    """Class-prototype loss on current raw delta using construction labels."""
    classes = sorted(int(v) for v in raw_class.unique().tolist() if int(v) >= 0)
    if not classes:
        return raw_x.new_zeros(())
    graph_z = T.encode_graph(params, raw_x, raw_adj)
    text_z = T.encode_text(params, raw_text)
    centers = F.normalize(torch.stack([
        text_z[raw_class.to(raw_x.device) == int(value)].mean(0)
        for value in classes]), dim=1)
    target = torch.tensor([classes.index(int(v)) for v in raw_class.tolist()],
                          dtype=torch.long, device=raw_x.device)
    logits = params["logit_scale"].exp().clamp(max=100) * graph_z @ centers.T
    return F.cross_entropy(logits, target)


def _raw_class_baselines(base: dict, support: dict, references: list[dict],
                         old_rows: int, candidate_groups: dict[int, torch.Tensor],
                         device: str, teacher_cfg: T.Config) -> list[float]:
    base_x = base["x"].to(device)
    base_text = base["text_emb"].to(device)
    base_adj = T.G.norm_adj(base["edge_index"], len(base_x), device)
    raw_x = support["x"][int(old_rows):].to(device).float()
    raw_text = support["text_emb"][int(old_rows):].to(device).float()
    raw_edge = support["edge_index"].long()
    keep = (raw_edge[0] >= int(old_rows)) & (raw_edge[1] >= int(old_rows))
    raw_edge = raw_edge[:, keep] - int(old_rows)
    raw_adj = T.G.norm_adj(raw_edge, len(raw_x), device)
    raw_class = support["class_id"][int(old_rows):].long()
    values = []
    for ref in references:
        params = T._unroll(
            ref["params0"], base_x, base_text, base_adj, candidate_groups,
            ref["seed"], teacher_cfg, False,
            row_selector=M._rows_with_class_mix, old_rows=old_rows,
            old_class_mix=float(teacher_cfg.old_class_mix))
        values.append(float(_raw_class_loss(
            params, raw_x, raw_text, raw_adj, raw_class).detach()))
    return values


def _raw_prompt_loss(params: dict, raw_x: torch.Tensor,
                     raw_adj: torch.Tensor, raw_class: torch.Tensor,
                     prompt_emb: torch.Tensor) -> torch.Tensor:
    classes = sorted(int(v) for v in raw_class.unique().tolist() if int(v) >= 0)
    if not classes:
        return raw_x.new_zeros(())
    graph_z = T.encode_graph(params, raw_x, raw_adj)
    prompts = prompt_emb.to(raw_x.device).float()[classes]
    prompt_z = T.encode_text(params, prompts)
    target = torch.tensor([classes.index(int(v)) for v in raw_class.tolist()],
                          dtype=torch.long, device=raw_x.device)
    logits = params["logit_scale"].exp().clamp(max=100) * graph_z @ prompt_z.T
    return F.cross_entropy(logits, target)


def _raw_prompt_baselines(base: dict, support: dict, references: list[dict],
                          old_rows: int, candidate_groups: dict[int, torch.Tensor],
                          device: str, teacher_cfg: T.Config,
                          prompt_emb: torch.Tensor) -> list[float]:
    base_x = base["x"].to(device)
    base_text = base["text_emb"].to(device)
    base_adj = T.G.norm_adj(base["edge_index"], len(base_x), device)
    raw_x = support["x"][int(old_rows):].to(device).float()
    raw_edge = support["edge_index"].long()
    keep = (raw_edge[0] >= int(old_rows)) & (raw_edge[1] >= int(old_rows))
    raw_edge = raw_edge[:, keep] - int(old_rows)
    raw_adj = T.G.norm_adj(raw_edge, len(raw_x), device)
    raw_class = support["class_id"][int(old_rows):].long()
    values = []
    for ref in references:
        params = T._unroll(
            ref["params0"], base_x, base_text, base_adj, candidate_groups,
            ref["seed"], teacher_cfg, False,
            row_selector=M._rows_with_class_mix, old_rows=old_rows,
            old_class_mix=float(teacher_cfg.old_class_mix))
        values.append(float(_raw_prompt_loss(
            params, raw_x, raw_adj, raw_class, prompt_emb).detach()))
    return values


def _prompt_class_loss(params: dict, x: torch.Tensor, adj: torch.Tensor,
                       class_id: torch.Tensor, prompt_emb: torch.Tensor,
                       rows: torch.Tensor | None = None) -> torch.Tensor:
    classes = sorted(int(v) for v in class_id.unique().tolist() if int(v) >= 0)
    if not classes:
        return x.new_zeros(())
    graph_z = T.encode_graph(params, x, adj)
    if rows is not None:
        graph_z = graph_z[rows.to(x.device)]
    prompt_z = T.encode_text(params, prompt_emb.to(x.device).float()[classes])
    target = torch.tensor([classes.index(int(v)) for v in class_id.tolist()],
                          dtype=torch.long, device=x.device)
    logits = params["logit_scale"].exp().clamp(max=100) * graph_z @ prompt_z.T
    return F.cross_entropy(logits, target)


def _old_prompt_baselines(base: dict, references: list[dict], old_rows: int,
                          candidate_groups: dict[int, torch.Tensor],
                          class_id: torch.Tensor, device: str,
                          teacher_cfg: T.Config,
                          prompt_emb: torch.Tensor) -> list[float]:
    base_x = base["x"].to(device)
    base_text = base["text_emb"].to(device)
    base_adj = T.G.norm_adj(base["edge_index"], len(base_x), device)
    old_class = class_id[:int(old_rows)].long().cpu()
    values = []
    for ref in references:
        params = T._unroll(
            ref["params0"], base_x, base_text, base_adj, candidate_groups,
            ref["seed"], teacher_cfg, False,
            row_selector=M._rows_with_class_mix, old_rows=old_rows,
            old_class_mix=float(teacher_cfg.old_class_mix))
        values.append(float(_prompt_class_loss(
            params, base_x, base_adj, old_class, prompt_emb,
            torch.arange(int(old_rows), device=device)).detach()))
    return values


def _response_alignment(base: dict, reference: dict, old_rows: int,
                        candidate_groups: dict[int, torch.Tensor], device: str,
                        teacher_cfg: T.Config,
                        probe_rows: dict[str, torch.Tensor]) -> dict:
    """Measure whether old-row gradients for old/new probes agree."""
    x = base["x"].to(device).float().detach().requires_grad_(True)
    text = base["text_emb"].to(device).float()
    adj = T.G.norm_adj(base["edge_index"], len(x), device)
    params = T._unroll(reference["params0"], x, text, adj, candidate_groups,
                       reference["seed"], teacher_cfg, True,
                       row_selector=M._rows_with_class_mix,
                       old_rows=old_rows,
                       old_class_mix=float(teacher_cfg.old_class_mix))
    old_loss = T.contrastive_loss(params, x, text, adj,
                                  probe_rows["old"].to(device))
    new_loss = T.contrastive_loss(params, x, text, adj,
                                  probe_rows["new"].to(device))
    old_grad, = torch.autograd.grad(old_loss, x, retain_graph=True)
    new_grad, = torch.autograd.grad(new_loss, x)
    old_grad, new_grad = old_grad[:int(old_rows)], new_grad[:int(old_rows)]
    old_norm = old_grad.norm(dim=1)
    new_norm = new_grad.norm(dim=1)
    cosine = (old_grad * new_grad).sum(1) / (
        old_norm * new_norm).clamp_min(1e-8)
    return {"cosine": cosine.detach().cpu(),
            "old_norm": old_norm.detach().cpu(),
            "new_norm": new_norm.detach().cpu()}


def _protection_gradient(base: dict, reference: dict, old_rows: int,
                         candidate_groups: dict[int, torch.Tensor],
                         class_id: torch.Tensor, device: str,
                         teacher_cfg: T.Config,
                         prompt_emb: torch.Tensor | None) -> torch.Tensor:
    """Gradient of the old-row retention objective at the base memory."""
    x = base["x"].to(device).float().detach().requires_grad_(True)
    text = base["text_emb"].to(device).float()
    adj = T.G.norm_adj(base["edge_index"], len(x), device)
    params = T._unroll(reference["params0"], x, text, adj, candidate_groups,
                       reference["seed"], teacher_cfg, True,
                       row_selector=M._rows_with_class_mix,
                       old_rows=old_rows,
                       old_class_mix=float(teacher_cfg.old_class_mix))
    old_rows_tensor = torch.arange(int(old_rows), device=device)
    if prompt_emb is not None:
        loss = _prompt_class_loss(
            params, x, adj, class_id[:int(old_rows)].long(), prompt_emb,
            old_rows_tensor)
    else:
        loss = T.contrastive_loss(params, x, text, adj, old_rows_tensor)
    grad, = torch.autograd.grad(loss, x, allow_unused=False)
    return grad[:int(old_rows)].detach()


def _objective(base: dict, support: dict, assignment: dict, relative: torch.Tensor,
               old_rows: int, references: list[dict], base_references: list[dict],
               class_id: torch.Tensor,
               roles: dict, device: str, teacher_cfg: T.Config,
               role_cfg: Config, create_graph: bool,
               probe_rows: dict[str, torch.Tensor],
               probe_baselines: list[dict], raw_new_baselines: list[float],
               raw_class_baselines: list[float],
               raw_prompt_baselines: list[float] | None,
               prompt_emb: torch.Tensor | None,
               old_prompt_baselines: list[float] | None) -> tuple[torch.Tensor, dict]:
    candidate_x = base["x"].to(device) + relative
    candidate_text = base["text_emb"].to(device)
    candidate_adj = T.G.norm_adj(base["edge_index"], len(candidate_x), device)
    indices = assignment["indices"].to(device)
    probs = torch.softmax(assignment["logits"], dim=1)
    slot_class = assignment["slot_class"].to(device)
    slot_to_raw = probs.new_zeros((len(slot_class), len(support["x"]) - int(old_rows)))
    slot_to_raw = slot_to_raw.scatter_add(1, indices, probs)
    raw_mass = slot_to_raw.sum(0).clamp_min(1e-8)
    slot_to_raw = slot_to_raw / raw_mass.unsqueeze(0)
    target_new_x = slot_to_raw @ support["x"].to(device)[int(old_rows):]
    raw_x = support["x"][int(old_rows):].to(device).float()
    raw_text = support["text_emb"][int(old_rows):].to(device).float()
    raw_edge = support["edge_index"].long()
    keep = (raw_edge[0] >= int(old_rows)) & (raw_edge[1] >= int(old_rows))
    raw_edge = raw_edge[:, keep] - int(old_rows)
    raw_adj = T.G.norm_adj(raw_edge, len(raw_x), device)
    raw_rows = torch.arange(len(raw_x), device=device, dtype=torch.long)
    raw_class = support["class_id"][int(old_rows):].long()
    candidate_groups = M._candidate_groups_with_provenance(
        class_id, assignment["slot_class"], old_rows)
    total = candidate_x.new_zeros(())
    records = []
    for index, (ref, base_ref, probe_baseline) in enumerate(
            zip(references, base_references, probe_baselines)):
        params_c = T._unroll(
            ref["params0"], candidate_x, candidate_text, candidate_adj,
            candidate_groups, ref["seed"], teacher_cfg, create_graph,
            row_selector=M._rows_with_class_mix, old_rows=old_rows,
            old_class_mix=float(teacher_cfg.old_class_mix))
        candidate_z = T.encode_graph(params_c, candidate_x, candidate_adj)
        candidate_text_z = T.encode_text(params_c, candidate_text)
        q, _ = M._assignment_matrix(
            assignment, old_rows, len(support["x"]), len(candidate_x), device)
        support_target_z = F.normalize(q.T @ ref["z"].to(device), dim=1)
        base_target_old = F.normalize(base_ref["z"][:int(old_rows)].to(device), dim=1)
        target_z = torch.cat((base_target_old, support_target_z[int(old_rows):]), dim=0)
        new_feature = T._class_pooled_feature(
            candidate_z[int(old_rows):], target_z[int(old_rows):], slot_class)
        old_feature = (1.0 - F.cosine_similarity(
            candidate_z[:int(old_rows)], target_z[:int(old_rows)], dim=1,
            eps=1e-8)).mean()
        payload = (1.0 - F.cosine_similarity(
            candidate_x[int(old_rows):], target_new_x, dim=1,
            eps=1e-8)).mean()
        trajectory = T._trajectory_cosine(
            params_c, ref["params0"], ref["params_t"], device)
        old_center = _old_center_loss(
            candidate_z, target_z, class_id, old_rows)
        old_relation = _old_relation_loss(
            candidate_z, target_z, class_id, old_rows)
        old_text = _old_text_consistency(
            candidate_z, candidate_text_z, class_id, old_rows)
        role_anchor = _role_anchor(
            relative, base["x"].to(device), old_rows, roles)
        old_probe = (T.contrastive_loss(
            params_c, candidate_x, candidate_text, candidate_adj,
            probe_rows["old"].to(device))
                     if len(probe_rows["old"]) else candidate_x.new_zeros(()))
        new_probe = (T.contrastive_loss(
            params_c, candidate_x, candidate_text, candidate_adj,
            probe_rows["new"].to(device))
                     if len(probe_rows["new"]) else candidate_x.new_zeros(()))
        old_probe_guard = F.relu(
            old_probe - float(probe_baseline["old"]) -
            float(role_cfg.old_probe_tolerance))
        new_probe_delta = new_probe - float(probe_baseline["new"])
        raw_new_probe = T.contrastive_loss(
            params_c, raw_x, raw_text, raw_adj, raw_rows)
        raw_new_probe_delta = raw_new_probe - float(raw_new_baselines[index])
        raw_class_probe = _raw_class_loss(
            params_c, raw_x, raw_text, raw_adj, raw_class)
        raw_class_delta = raw_class_probe - float(raw_class_baselines[index])
        if prompt_emb is not None and raw_prompt_baselines is not None:
            raw_prompt_probe = _raw_prompt_loss(
                params_c, raw_x, raw_adj, raw_class, prompt_emb)
            raw_prompt_delta = raw_prompt_probe - float(raw_prompt_baselines[index])
        else:
            raw_prompt_probe = candidate_x.new_zeros(())
            raw_prompt_delta = candidate_x.new_zeros(())
        if prompt_emb is not None and old_prompt_baselines is not None:
            old_prompt_probe = _prompt_class_loss(
                params_c, candidate_x, candidate_adj,
                class_id[:int(old_rows)].long(), prompt_emb,
                torch.arange(int(old_rows), device=device))
            old_prompt_guard = F.relu(
                old_prompt_probe - float(old_prompt_baselines[index]) -
                float(role_cfg.old_prompt_tolerance))
        else:
            old_prompt_probe = candidate_x.new_zeros(())
            old_prompt_guard = candidate_x.new_zeros(())
        total = (total + teacher_cfg.feature_weight * new_feature +
                 teacher_cfg.old_feature_weight * old_feature +
                 teacher_cfg.payload_weight * payload +
                 teacher_cfg.trajectory_weight * trajectory +
                 role_cfg.old_center_weight * old_center +
                 role_cfg.old_relation_weight * old_relation +
                 role_cfg.old_text_weight * old_text +
                 role_cfg.role_anchor_weight * role_anchor +
                 role_cfg.old_probe_weight * old_probe_guard +
                 role_cfg.new_probe_weight * new_probe_delta +
                 role_cfg.raw_new_probe_weight * raw_new_probe_delta +
                 role_cfg.raw_class_weight * raw_class_delta +
                 role_cfg.raw_prompt_weight * raw_prompt_delta +
                 role_cfg.old_prompt_weight * old_prompt_guard)
        records.append({"seed": int(ref["seed"]),
                        "new_feature": float(new_feature.detach()),
                        "old_feature": float(old_feature.detach()),
                        "payload": float(payload.detach()),
                        "trajectory": float(trajectory.detach()),
                        "old_center": float(old_center.detach()),
                        "old_relation": float(old_relation.detach()),
                        "old_text": float(old_text.detach()),
                        "role_anchor": float(role_anchor.detach()),
                        "old_probe": float(old_probe.detach()),
                        "new_probe": float(new_probe.detach()),
                        "old_probe_guard": float(old_probe_guard.detach()),
                        "new_probe_delta": float(new_probe_delta.detach()),
                        "raw_new_probe": float(raw_new_probe.detach()),
                        "raw_new_probe_delta": float(raw_new_probe_delta.detach()),
                        "raw_class_probe": float(raw_class_probe.detach()),
                        "raw_class_delta": float(raw_class_delta.detach()),
                        "raw_prompt_probe": float(raw_prompt_probe.detach()),
                        "raw_prompt_delta": float(raw_prompt_delta.detach()),
                        "old_prompt_probe": float(old_prompt_probe.detach()),
                        "old_prompt_guard": float(old_prompt_guard.detach()),
                        "teacher_loss": float(ref["final_loss"])})
    total = total / max(1, len(references))
    return total, {"seeds": records}


def optimize(base: dict, support: dict, class_id: torch.Tensor, old_rows: int,
             stage: int, device: str = "cpu", teacher_cfg: T.Config = T.Config(),
             role_cfg: Config = Config(),
             prompt_emb: torch.Tensor | None = None) -> tuple[dict, dict, dict]:
    """Run bounded X revision with fixed initial P and fixed topology."""
    T.validate_memory(base)
    T.validate_memory(support)
    if len(class_id) != len(support["x"]):
        raise ValueError("class_id must align with support rows")
    old_rows = int(old_rows)
    base_hash, support_hash = T.memory_hash(base), T.memory_hash(support)
    assignment = T.build_sparse_assignment(base, support, old_rows, class_id,
                                           teacher_cfg, device)
    initial_logits = assignment["logits"].detach().clone()
    train_refs = [M._train_teacher(support, class_id, old_rows, seed, device,
                                   teacher_cfg)
                  for seed in teacher_cfg.teacher_seeds]
    val_refs = [M._train_teacher(support, class_id, old_rows, seed, device,
                                 teacher_cfg)
                for seed in teacher_cfg.validation_teacher_seeds]
    base_class_id = torch.cat((class_id[:old_rows].long().cpu(),
                               assignment["slot_class"].long().cpu()))
    base_train_refs = [M._train_teacher(base, base_class_id, old_rows, seed,
                                        device, teacher_cfg)
                       for seed in teacher_cfg.teacher_seeds]
    base_val_refs = [M._train_teacher(base, base_class_id, old_rows, seed,
                                      device, teacher_cfg)
                     for seed in teacher_cfg.validation_teacher_seeds]
    roles = select_roles(base, support, class_id, old_rows,
                         base_train_refs[0], device, teacher_cfg, role_cfg,
                         gradient_memory=base, gradient_class_id=base_class_id)
    candidate_groups = M._candidate_groups_with_provenance(
        class_id, assignment["slot_class"], old_rows)
    probe_rows = _probe_rows(len(base["x"]), old_rows)
    train_probe_baselines = _probe_baselines(
        base, train_refs, old_rows, candidate_groups, device, teacher_cfg,
        probe_rows)
    val_probe_baselines = _probe_baselines(
        base, val_refs, old_rows, candidate_groups, device, teacher_cfg,
        probe_rows)
    train_raw_baselines = _raw_delta_baselines(
        base, support, train_refs, old_rows, candidate_groups, device,
        teacher_cfg)
    val_raw_baselines = _raw_delta_baselines(
        base, support, val_refs, old_rows, candidate_groups, device,
        teacher_cfg)
    train_raw_class_baselines = _raw_class_baselines(
        base, support, train_refs, old_rows, candidate_groups, device,
        teacher_cfg)
    val_raw_class_baselines = _raw_class_baselines(
        base, support, val_refs, old_rows, candidate_groups, device,
        teacher_cfg)
    train_raw_prompt_baselines = (
        _raw_prompt_baselines(base, support, train_refs, old_rows,
                              candidate_groups, device, teacher_cfg, prompt_emb)
        if prompt_emb is not None else None)
    val_raw_prompt_baselines = (
        _raw_prompt_baselines(base, support, val_refs, old_rows,
                              candidate_groups, device, teacher_cfg, prompt_emb)
        if prompt_emb is not None else None)
    train_old_prompt_baselines = (
        _old_prompt_baselines(base, train_refs, old_rows, candidate_groups,
                              class_id, device, teacher_cfg, prompt_emb)
        if prompt_emb is not None else None)
    val_old_prompt_baselines = (
        _old_prompt_baselines(base, val_refs, old_rows, candidate_groups,
                              class_id, device, teacher_cfg, prompt_emb)
        if prompt_emb is not None else None)
    alignment = _response_alignment(
        base, base_train_refs[0], old_rows, candidate_groups, device,
        teacher_cfg, probe_rows)
    protection_gradient = _protection_gradient(
        base, base_train_refs[0], old_rows, candidate_groups, class_id,
        device, teacher_cfg, prompt_emb)
    if role_cfg.response_alignment_gate and old_rows:
        eligible = alignment["cosine"] >= float(role_cfg.response_alignment_threshold)
        roles["radius"] = roles["radius"].clone()
        roles["radius"][~eligible] = 0.0
    roles["response_alignment"] = alignment
    relative = torch.nn.Parameter(torch.zeros_like(base["x"].to(device)))
    optimizer = torch.optim.Adam([relative], lr=float(teacher_cfg.outer_lr))
    base_rows = base["x"].to(device).detach()
    started = time.perf_counter()
    baseline_loss, baseline_record = _objective(
        base, support, assignment, torch.zeros_like(relative), old_rows, val_refs,
        base_val_refs,
        class_id, roles, device, teacher_cfg, role_cfg, False,
        probe_rows, val_probe_baselines, val_raw_baselines,
        val_raw_class_baselines, val_raw_prompt_baselines, prompt_emb,
        val_old_prompt_baselines)
    baseline_val = float(baseline_loss.detach())
    best_val, best_step, best_relative = baseline_val, 0, None
    curve = []
    for step in range(1, int(teacher_cfg.outer_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_record = _objective(
            base, support, assignment, relative, old_rows, train_refs,
            base_train_refs,
            class_id, roles, device, teacher_cfg, role_cfg, True,
            probe_rows, train_probe_baselines, train_raw_baselines,
            train_raw_class_baselines, train_raw_prompt_baselines, prompt_emb,
            train_old_prompt_baselines)
        if not torch.isfinite(train_loss):
            raise FloatingPointError("non-finite role-conditioned objective")
        train_loss.backward()
        if relative.grad is None or not torch.isfinite(relative.grad).all():
            raise FloatingPointError("missing/nonfinite X gradient")
        if role_cfg.gradient_surgery and old_rows:
            current = relative.grad[:old_rows]
            protect = protection_gradient.to(current.device)
            dot = (current * protect).sum(dim=1)
            denom = protect.square().sum(dim=1).clamp_min(1e-12)
            conflict = torch.minimum(dot, torch.zeros_like(dot))
            relative.grad[:old_rows] = current - (
                conflict / denom).unsqueeze(1) * protect
        grad_norm = float(relative.grad.norm().detach())
        optimizer.step()
        _project_role(relative, base_rows, old_rows, roles, role_cfg.new_radius)
        val_loss, val_record = _objective(
            base, support, assignment, relative, old_rows, val_refs,
            base_val_refs,
            class_id, roles, device, teacher_cfg, role_cfg, False,
            probe_rows, val_probe_baselines, val_raw_baselines,
            val_raw_class_baselines, val_raw_prompt_baselines, prompt_emb,
            val_old_prompt_baselines)
        score = float(val_loss.detach())
        curve.append({"step": int(step),
                      "train_objective": float(train_loss.detach()),
                      "validation_objective": score,
                      "gradient_norm": grad_norm,
                      "train": train_record, "validation": val_record})
        if score < best_val - float(role_cfg.gate_margin):
            best_val, best_step = score, step
            best_relative = relative.detach().clone()
    gate_fallback = best_relative is None
    if gate_fallback:
        best_relative = torch.zeros_like(relative)
    result = T.clone_memory(base)
    result["x"] = (base["x"].to(device) + best_relative).detach().cpu()
    if T.memory_hash(base) != base_hash or T.memory_hash(support) != support_hash:
        raise AssertionError("optimizer mutated input memory")
    for key in T.IMMUTABLE_FIELDS:
        left, right = base[key], result[key]
        equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
        if not equal:
            raise AssertionError(f"immutable field changed: {key}")
    with torch.no_grad():
        relative_ratio = ((result["x"] - base["x"]).norm(dim=1) /
                          base["x"].norm(dim=1).clamp_min(1e-6))
    audit = {
        "version": "tgr-role-conditioned-bounded-revision-v1",
        "stage": int(stage), "teacher_config": teacher_cfg.dict(),
        "role_config": role_cfg.dict(), "old_rows": old_rows,
        "new_rows": int(len(base["x"]) - old_rows),
        "input_base_sha256": base_hash, "input_support_sha256": support_hash,
        "output_sha256": T.memory_hash(result),
        "selected_outer_step": int(best_step),
        "validation_objective": float(best_val),
        "baseline_validation_objective": float(baseline_val),
        "gate_fallback": bool(gate_fallback),
        "roles": {key: (value.tolist() if isinstance(value, torch.Tensor) else value)
                  for key, value in roles.items()
                  if key not in ("order_descending", "response_alignment")},
        "curve": curve,
        "max_relative_displacement": float(relative_ratio.max()),
        "mean_relative_displacement": float(relative_ratio.mean()),
        "old_max_relative_displacement": float(relative_ratio[:old_rows].max()) if old_rows else 0.0,
        "new_max_relative_displacement": float(relative_ratio[old_rows:].max()) if len(relative_ratio) > old_rows else 0.0,
        "p_updated": False, "topology_updated": False, "text_changed": False,
        "old_target": "synthetic-only teacher on base topology",
        "new_target": "support teacher projected through fixed initial P",
        "student_probe": "synthetic old/new rows; old retention hinge plus new loss delta",
        "raw_delta_probe": "current raw delta graph/text InfoNCE after synthetic Student unroll",
        "raw_class_probe": "current raw delta construction-label text-prototype loss",
        "raw_prompt_probe": "current raw delta loss against official class prompts",
        "old_prompt_probe": "old synthetic rows loss against official class prompts",
        "response_alignment": {
            key: value.tolist() for key, value in roles["response_alignment"].items()
        },
        "gradient_surgery": bool(role_cfg.gradient_surgery),
        "budget_changed": False, "historical_raw_reads": 0,
        "report_labels_used": False,
    }
    assignment_out = {
        "indices": assignment["indices"].tolist(),
        "weights": torch.softmax(initial_logits, dim=1).tolist(),
        "slot_class": assignment["slot_class"].tolist(),
        "raw_class": assignment["raw_class"].tolist(),
        "top_r": int(assignment["indices"].shape[1]),
        "construction_seconds": float(time.perf_counter() - started),
    }
    return result, audit, assignment_out
