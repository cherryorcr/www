"""Teacher-guided payload-consistent revision for Semantic-Recursive memory.

This is the first complete Teacher/P/X implementation.  It deliberately keeps
text, budget, and topology fixed so that the new construction signal is
isolated.  The legal support is previous synthetic memory plus the current raw
delta only.  A short-trained canonical Student acts as the Teacher; a sparse
soft assignment P maps current raw nodes to new synthetic slots; X and P are
optimized through a short differentiable Student unroll.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

HERE = __import__("pathlib").Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen_v1_2026_09_09"
if str(FROZEN) not in sys.path:
    sys.path.insert(0, str(FROZEN))
import run_gate1_continual as G  # noqa: E402


IMMUTABLE_FIELDS = ("text_emb", "texts", "mass", "depth", "stage",
                    "edge_index")


@dataclass(frozen=True)
class Config:
    teacher_steps: int = 4
    inner_steps: int = 2
    outer_steps: int = 4
    teacher_lr: float = 1e-3
    inner_lr: float = 1e-3
    outer_lr: float = 5e-3
    batch: int = 16
    top_r: int = 8
    temperature: float = 0.20
    feature_weight: float = 1.0
    trajectory_weight: float = 0.50
    payload_weight: float = 0.25
    old_feature_weight: float = 0.50
    anchor_weight: float = 0.10
    assignment_weight: float = 0.10
    class_weight: float = 0.50
    # Optional second-order-safe plasticity term used by the aligned v3 arm.
    # Defaults preserve the v1/v2 objective exactly.
    plasticity_weight: float = 0.0
    plasticity_margin: float = 0.20
    ste_forward: bool = False
    old_radius: float = 0.05
    new_radius: float = 0.15
    teacher_seeds: tuple[int, ...] = (9300, 9301)
    validation_teacher_seeds: tuple[int, ...] = (9310,)
    old_class_mix: float = 0.0
    baseline_gate: bool = True
    gate_margin: float = 0.0

    def __post_init__(self) -> None:
        if min(self.teacher_steps, self.inner_steps, self.outer_steps) < 1:
            raise ValueError("optimization steps must be positive")
        if self.batch < 2 or self.top_r < 1:
            raise ValueError("batch/top_r are invalid")
        if self.temperature <= 0 or self.teacher_lr <= 0 or self.inner_lr <= 0 or self.outer_lr <= 0:
            raise ValueError("learning rates and temperature must be positive")
        for name in ("feature_weight", "trajectory_weight", "payload_weight",
                     "old_feature_weight", "anchor_weight", "assignment_weight",
                     "class_weight", "plasticity_weight", "plasticity_margin",
                     "old_radius", "new_radius"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(float(self.old_class_mix)) or not 0 <= float(self.old_class_mix) <= 1:
            raise ValueError("old_class_mix must be in [0, 1]")
        if not isinstance(self.baseline_gate, bool):
            raise ValueError("baseline_gate must be boolean")
        if not isinstance(self.ste_forward, bool):
            raise ValueError("ste_forward must be boolean")
        if not math.isfinite(float(self.gate_margin)) or float(self.gate_margin) < 0:
            raise ValueError("gate_margin must be non-negative")
        if set(self.teacher_seeds) & set(self.validation_teacher_seeds):
            raise ValueError("teacher and validation seeds must be disjoint")

    def dict(self) -> dict:
        return asdict(self)


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def clone_memory(memory: dict) -> dict:
    return {key: clone_value(value) for key, value in memory.items()}


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def memory_hash(memory: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(memory):
        h.update(key.encode())
        value = memory[key]
        if isinstance(value, torch.Tensor):
            h.update(_tensor_digest(value).encode())
        elif isinstance(value, list):
            h.update(json.dumps(value, ensure_ascii=True).encode())
    return h.hexdigest()


def validate_memory(memory: dict) -> None:
    required = {"x", *IMMUTABLE_FIELDS}
    missing = required.difference(memory)
    if missing:
        raise ValueError(f"memory is missing fields: {sorted(missing)}")
    n = len(memory["x"])
    if memory["x"].ndim != 2 or memory["text_emb"].ndim != 2:
        raise ValueError("x/text_emb must be rank-2")
    if len(memory["text_emb"]) != n or len(memory["texts"]) != n:
        raise ValueError("memory row fields are not aligned")
    if not torch.isfinite(memory["x"]).all() or not torch.isfinite(memory["text_emb"]).all():
        raise ValueError("memory contains non-finite values")
    edge = memory["edge_index"]
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("invalid edge_index")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= n):
        raise ValueError("edge endpoint outside memory")


def _student(seed: int, in_dim: int, text_dim: int, device: str):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return G.CLIPStudent(in_dim, text_dim=text_dim).to(device)


def _params(model) -> OrderedDict:
    return OrderedDict((name, value) for name, value in model.named_parameters())


def _clone_params(params: OrderedDict, detach: bool = False) -> OrderedDict:
    if detach:
        return OrderedDict((name, value.detach().clone().requires_grad_(False))
                           for name, value in params.items())
    return OrderedDict((name, value.clone().requires_grad_(True))
                       for name, value in params.items())


def _linear(params: OrderedDict, prefix: str, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, params[f"{prefix}.weight"], params[f"{prefix}.bias"])


def encode_graph(params: OrderedDict, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    h = F.relu(adjacency @ _linear(params, "g1.lin", x))
    h = adjacency @ _linear(params, "g2.lin", h)
    return F.normalize(h, dim=1)


def encode_text(params: OrderedDict, text: torch.Tensor) -> torch.Tensor:
    return F.normalize(_linear(params, "text_proj", text), dim=1)


def contrastive_loss(params: OrderedDict, x: torch.Tensor, text: torch.Tensor,
                     adjacency: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    graph = encode_graph(params, x, adjacency)[rows]
    text_z = encode_text(params, text[rows])
    logits = params["logit_scale"].exp().clamp(max=100) * graph @ text_z.T
    target = torch.arange(len(rows), device=x.device)
    return (F.cross_entropy(logits, target) +
            F.cross_entropy(logits.T, target)) / 2


def _sample(indices: torch.Tensor, count: int, seed: int,
            device: str) -> torch.Tensor:
    indices = indices.long().cpu()
    if len(indices) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if len(indices) >= count:
        selected = indices[torch.randperm(len(indices), generator=gen)[:count]]
    else:
        selected = indices[torch.randint(len(indices), (count,), generator=gen)]
    return selected.to(device)


def _balanced_rows(groups: dict[int, torch.Tensor], batch: int,
                   seed: int, device: str) -> torch.Tensor:
    available = [(int(key), value) for key, value in sorted(groups.items())
                 if len(value)]
    if not available:
        raise ValueError("no rows available for Student batch")
    per = max(1, int(batch) // len(available))
    rows = [_sample(value, per, int(seed) * 7919 + index * 101,
                    device) for index, (_, value) in enumerate(available)]
    rows = torch.cat(rows)
    if len(rows) > batch:
        rows = rows[:batch]
    return rows


def _groups(class_id: torch.Tensor, old_rows: int) -> dict[int, torch.Tensor]:
    class_id = class_id.long().cpu()
    result = {}
    old = torch.arange(min(int(old_rows), len(class_id)))
    if len(old):
        result[-1] = old
    for value in sorted(int(v) for v in class_id[int(old_rows):].unique().tolist()
                       if int(v) >= 0):
        result[value] = torch.where(class_id == value)[0]
    return result


def train_teacher(support: dict, class_id: torch.Tensor, old_rows: int,
                  seed: int, device: str, cfg: Config) -> dict:
    validate_memory(support)
    model = _student(seed, support["x"].shape[1], support["text_emb"].shape[1], device)
    params0 = _clone_params(_params(model), detach=True)
    x = support["x"].to(device)
    text = support["text_emb"].to(device)
    adj = G.norm_adj(support["edge_index"], len(x), device)
    groups = _groups(class_id, old_rows)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.teacher_lr))
    for step in range(int(cfg.teacher_steps)):
        rows = _balanced_rows(groups, cfg.batch, int(seed) * 1009 + step, device)
        loss = contrastive_loss(_params(model), x, text, adj, rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    params_t = _clone_params(_params(model), detach=True)
    with torch.no_grad():
        z = encode_graph(params_t, x, adj).detach()
    return {
        "seed": int(seed), "params0": params0, "params_t": params_t,
        "z": z, "final_loss": float(loss.detach().cpu()),
        "teacher_steps": int(cfg.teacher_steps),
    }


def build_sparse_assignment(base: dict, support: dict, old_rows: int,
                            class_id: torch.Tensor, cfg: Config,
                            device: str) -> dict:
    new_support = support["x"][int(old_rows):].float()
    new_text = support["text_emb"][int(old_rows):].float()
    new_class = class_id[int(old_rows):].long().cpu()
    new_x = base["x"][int(old_rows):].float()
    new_t = base["text_emb"][int(old_rows):].float()
    if len(new_support) == 0 or len(new_x) == 0:
        raise ValueError("new support and new synthetic block must be non-empty")
    raw_joint = torch.cat((F.normalize(new_support, dim=1), new_text), dim=1)
    syn_joint = torch.cat((F.normalize(new_x, dim=1), new_t), dim=1)
    distance = torch.cdist(syn_joint, raw_joint)
    r = min(int(cfg.top_r), len(new_support))
    indices, logits, slot_class = [], [], []
    for row in range(len(new_x)):
        nearest = int(distance[row].argmin())
        cls = int(new_class[nearest])
        slot_class.append(cls)
        same = torch.where(new_class == cls)[0]
        chosen = []
        if len(same):
            k = min(max(1, r // 2), len(same))
            chosen.extend(same[torch.topk(distance[row, same], k, largest=False).indices].tolist())
        if len(chosen) < r:
            order = torch.argsort(distance[row]).tolist()
            chosen.extend(value for value in order if value not in chosen)
        chosen = chosen[:r]
        indices.append(chosen)
        logits.append((-distance[row, torch.tensor(chosen)] /
                       float(cfg.temperature)).tolist())
    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "logits": torch.tensor(logits, dtype=torch.float32, device=device,
                                requires_grad=True),
        "slot_class": torch.tensor(slot_class, dtype=torch.long),
        "raw_class": new_class,
        "distance": distance,
    }


def _project(relative: torch.Tensor, base_rows: torch.Tensor,
             selected_old: int, old_radius: float, new_radius: float) -> None:
    with torch.no_grad():
        scales = base_rows.norm(dim=1, keepdim=True).clamp_min(1e-6)
        limits = torch.full_like(scales, float(new_radius))
        limits[:int(selected_old)] = float(old_radius)
        limits = limits * scales
        norms = relative.norm(dim=1, keepdim=True).clamp_min(1e-12)
        relative.mul_((limits / norms).clamp(max=1.0))


def _candidate_groups(slot_class: torch.Tensor, old_rows: int) -> dict[int, torch.Tensor]:
    groups = {-1: torch.arange(int(old_rows))}
    for value in sorted(int(v) for v in slot_class.unique().tolist() if int(v) >= 0):
        groups[value] = torch.where(slot_class == value)[0] + int(old_rows)
    return groups


def _unroll(params0: OrderedDict, x: torch.Tensor, text: torch.Tensor,
            adjacency: torch.Tensor, groups: dict[int, torch.Tensor],
            seed: int, cfg: Config, create_graph: bool,
            row_selector=None, old_rows: int | None = None,
            old_class_mix: float = 0.0) -> OrderedDict:
    params = _clone_params(params0, detach=False)
    for step in range(int(cfg.inner_steps)):
        if row_selector is None:
            rows = _balanced_rows(groups, cfg.batch, int(seed) * 1009 + step, x.device)
        else:
            rows = row_selector(groups, cfg.batch, int(seed) * 1009 + step,
                                x.device, int(old_rows or 0), float(old_class_mix))
        loss = contrastive_loss(params, x, text, adjacency, rows)
        grads = torch.autograd.grad(loss, tuple(params.values()),
                                    create_graph=create_graph,
                                    allow_unused=False)
        params = OrderedDict((name, value - float(cfg.inner_lr) * grad)
                             for (name, value), grad in zip(params.items(), grads))
    return params


def _weighted_target(values: torch.Tensor, indices: torch.Tensor,
                     probs: torch.Tensor) -> torch.Tensor:
    selected = values[indices.to(values.device)]
    return (probs.unsqueeze(-1) * selected).sum(1)


def _assignment_penalties(probs: torch.Tensor, indices: torch.Tensor,
                          raw_class: torch.Tensor,
                          slot_class: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = raw_class[indices]
    matches = classes == slot_class.to(classes.device).unsqueeze(1)
    class_mass = (probs * matches.float()).sum(1).clamp_min(1e-8)
    class_loss = -class_mass.log().mean()
    column_mass = probs.sum(0)
    coverage = ((column_mass - column_mass.mean()) /
                column_mass.mean().clamp_min(1e-8)).square().mean()
    return class_loss, coverage


def _trajectory_cosine(params_c: OrderedDict, params0: OrderedDict,
                       params_t: OrderedDict, device: str) -> torch.Tensor:
    candidate_delta = torch.cat([
        (params_c[name] - params0[name].to(device)).reshape(-1)
        for name in params_c])
    teacher_delta = torch.cat([
        (params_t[name].to(device) - params0[name].to(device)).reshape(-1)
        for name in params_c])
    return 1.0 - F.cosine_similarity(
        candidate_delta, teacher_delta, dim=0, eps=1e-8)


def _class_pooled_feature(candidate_z: torch.Tensor,
                          target_z: torch.Tensor,
                          slot_class: torch.Tensor) -> torch.Tensor:
    values = []
    for cls in sorted(int(v) for v in slot_class.unique().tolist()):
        mask = slot_class == int(cls)
        if bool(mask.any()):
            values.append(1.0 - F.cosine_similarity(
                candidate_z[mask].mean(0, keepdim=True),
                target_z[mask].mean(0, keepdim=True), dim=1, eps=1e-8).mean())
    if not values:
        return candidate_z.new_zeros(())
    return torch.stack(values).mean()


def _objective(base: dict, support: dict, assignment: dict, relative: torch.Tensor,
               old_rows: int, references: list[dict], class_id: torch.Tensor,
               device: str, cfg: Config, create_graph: bool) -> tuple[torch.Tensor, dict]:
    candidate_x = base["x"].to(device).clone()
    candidate_x = candidate_x + relative
    candidate_text = base["text_emb"].to(device)
    candidate_adj = G.norm_adj(base["edge_index"], len(candidate_x), device)
    indices = assignment["indices"].to(device)
    probs = torch.softmax(assignment["logits"], dim=1)
    raw_x = support["x"].to(device)[int(old_rows):]
    raw_class = assignment["raw_class"].to(device)
    slot_class = assignment["slot_class"].to(device)
    candidate_groups = _candidate_groups(assignment["slot_class"], old_rows)
    total = candidate_x.new_zeros(())
    records = []
    for ref in references:
        params_c = _unroll(ref["params0"], candidate_x, candidate_text,
                           candidate_adj, candidate_groups, ref["seed"], cfg,
                           create_graph)
        candidate_z = encode_graph(params_c, candidate_x, candidate_adj)
        teacher_z = ref["z"].to(device)
        target_new_z = _weighted_target(teacher_z[int(old_rows):], indices,
                                        probs)
        target_new_x = _weighted_target(raw_x, indices, probs)
        new_feature = _class_pooled_feature(
            candidate_z[int(old_rows):], target_new_z, slot_class)
        old_feature = (1.0 - F.cosine_similarity(
            candidate_z[:int(old_rows)], teacher_z[:int(old_rows)], dim=1,
            eps=1e-8)).mean()
        payload = (1.0 - F.cosine_similarity(
            candidate_x[int(old_rows):], target_new_x, dim=1, eps=1e-8)).mean()
        trajectory = _trajectory_cosine(
            params_c, ref["params0"], ref["params_t"], device)
        class_loss, coverage = _assignment_penalties(probs, indices, raw_class,
                                                     slot_class)
        total = (total + cfg.feature_weight * new_feature +
                 cfg.old_feature_weight * old_feature +
                 cfg.payload_weight * payload +
                 cfg.trajectory_weight * trajectory +
                 cfg.class_weight * class_loss +
                 cfg.assignment_weight * coverage)
        records.append({"seed": int(ref["seed"]),
                        "new_feature": float(new_feature.detach()),
                        "old_feature": float(old_feature.detach()),
                        "payload": float(payload.detach()),
                        "trajectory": float(trajectory.detach()),
                        "class_loss": float(class_loss.detach()),
                        "coverage": float(coverage.detach()),
                        "teacher_loss": float(ref["final_loss"])})
    total = total / max(1, len(references))
    scales = base["x"].to(device).norm(dim=1).clamp_min(1e-6)
    anchor = (relative[:int(old_rows)] /
              scales[:int(old_rows)].unsqueeze(1)).square().mean()
    total = total + cfg.anchor_weight * anchor
    return total, {"seeds": records, "anchor": float(anchor.detach()),
                   "assignment_entropy": float((-(probs * probs.clamp_min(1e-8).log()).sum(1)).mean().detach())}


def optimize(base: dict, support: dict, class_id: torch.Tensor, old_rows: int,
             stage: int, device: str = "cpu", cfg: Config = Config()) -> tuple[dict, dict, dict]:
    validate_memory(base)
    validate_memory(support)
    if len(class_id) != len(support["x"]):
        raise ValueError("class_id must align with support rows")
    base_hash, support_hash = memory_hash(base), memory_hash(support)
    assignment = build_sparse_assignment(base, support, old_rows, class_id, cfg, device)
    assignment_indices = assignment["indices"].clone()
    assignment_raw_class = assignment["raw_class"].clone()
    assignment_slot_class = assignment["slot_class"].clone()
    train_refs = [train_teacher(support, class_id, old_rows, seed, device, cfg)
                  for seed in cfg.teacher_seeds]
    val_refs = [train_teacher(support, class_id, old_rows, seed, device, cfg)
                for seed in cfg.validation_teacher_seeds]
    relative = torch.nn.Parameter(torch.zeros_like(base["x"].to(device)))
    p_logits = torch.nn.Parameter(assignment["logits"])
    assignment["logits"] = p_logits
    optimizer = torch.optim.Adam([relative, p_logits], lr=float(cfg.outer_lr))
    base_rows = base["x"].to(device).detach()
    curve, best, best_val, best_step = [], None, math.inf, 0
    started = time.perf_counter()
    for step in range(1, int(cfg.outer_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_record = _objective(
            base, support, assignment, relative, old_rows, train_refs,
            class_id, device, cfg, True)
        if not torch.isfinite(train_loss):
            raise FloatingPointError("non-finite teacher-guided objective")
        train_loss.backward()
        if relative.grad is None or p_logits.grad is None:
            raise FloatingPointError("missing X/P gradient")
        optimizer.step()
        _project(relative, base_rows, old_rows, cfg.old_radius, cfg.new_radius)
        val_loss, val_record = _objective(
            base, support, assignment, relative, old_rows, val_refs,
            class_id, device, cfg, False)
        score = float(val_loss.detach())
        row = {"step": int(step), "train_objective": float(train_loss.detach()),
               "validation_objective": score, "train": train_record,
               "validation": val_record,
               "x_grad_norm": float(relative.grad.norm().detach()),
               "p_grad_norm": float(p_logits.grad.norm().detach())}
        curve.append(row)
        if score < best_val:
            best_val, best_step = score, step
            best = (relative.detach().clone(), p_logits.detach().clone())
    if best is None:
        raise AssertionError("no validation checkpoint produced")
    best_relative, best_logits = best
    result = clone_memory(base)
    result["x"] = base["x"].clone()
    result["x"] = result["x"] + best_relative.cpu()
    if memory_hash(base) != base_hash or memory_hash(support) != support_hash:
        raise AssertionError("optimizer mutated input memory")
    for key in IMMUTABLE_FIELDS:
        left, right = base[key], result[key]
        equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
        if not equal:
            raise AssertionError(f"immutable field changed: {key}")
    with torch.no_grad():
        rel = (result["x"] - base["x"]).norm(dim=1)
        scale = base["x"].norm(dim=1).clamp_min(1e-6)
        rel_ratio = rel / scale
        probs = torch.softmax(best_logits, dim=1).cpu()
    audit = {
        "version": "teacher-guided-payload-revision-v1",
        "stage": int(stage), "config": cfg.dict(), "old_rows": int(old_rows),
        "new_rows": int(len(base["x"]) - old_rows),
        "input_base_sha256": base_hash, "input_support_sha256": support_hash,
        "output_sha256": memory_hash(result), "selected_outer_step": int(best_step),
        "validation_objective": float(best_val),
        "max_relative_displacement": float(rel_ratio.max()),
        "mean_relative_displacement": float(rel_ratio.mean()),
        "old_max_relative_displacement": float(rel_ratio[:old_rows].max()) if old_rows else 0.0,
        "new_max_relative_displacement": float(rel_ratio[old_rows:].max()) if len(rel_ratio) > old_rows else 0.0,
        "teacher_seeds": list(cfg.teacher_seeds),
        "validation_teacher_seeds": list(cfg.validation_teacher_seeds),
        "teacher_final_losses": {str(ref["seed"]): ref["final_loss"]
                                 for ref in train_refs + val_refs},
        "curve": curve, "text_changed": False, "budget_changed": False,
        "topology_changed": False, "historical_raw_reads": 0,
        "report_labels_used": False,
    }
    assignment_out = {
        "indices": assignment_indices.tolist(),
        "weights": probs.tolist(),
        "slot_class": assignment_slot_class.tolist(),
        "raw_class": assignment_raw_class.tolist(),
        "top_r": int(assignment_indices.shape[1]),
        "construction_seconds": float(time.perf_counter() - started),
    }
    return result, audit, assignment_out
