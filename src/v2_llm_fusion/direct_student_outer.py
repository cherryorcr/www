"""Direct Student outer optimization for continual synthetic memory.

This module is the explicit Student-compatible construction operator requested
for the next TGC experiment. It is deliberately separate from the earlier
gc_sr_gradient implementation. GC-SR matches a Student's initial gradient;
this module unrolls the canonical Student for K Adam steps on a candidate
memory and differentiates a held-out, label-free old/new support loss back to
the candidate memory.

The legal information boundary is represented by two inputs:

base
    The candidate synthetic memory to be edited.
support
    Previous synthetic rows followed by the current raw delta. It is the only
    object used as the construction target. The caller must prepare it; this
    optimizer never loads historical raw data.

Registered modes are x, x-class, x-provenance, x-slot-rep, x-geometry, p, a,
x+p, and x+p+a:

* X uses bounded differentiable feature updates.
* P uses a balanced soft assignment from current-delta rows to new synthetic
  rows and transports X/text embeddings.
* A uses a fixed candidate cross-edge pool and a straight-through top-k
  topology during the Student unroll.

The first experimental promotion should use x only. P and A are included here
so that their constraints, materialization, and audits share exactly the same
outer objective rather than being implemented as unrelated selectors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time
from typing import Iterable

import torch
import torch.nn.functional as F

import run_tgc_revision_gate as R


G = R.G


MODES = ("x", "x-class", "x-provenance", "x-slot-rep", "x-geometry",
         "x-class-rep", "p", "a", "x+p", "x+p+a", "x-class+p",
         "x-class+p+a")


def _mode_parts(mode: str) -> tuple[str, ...]:
    return tuple(str(mode).split("+"))


def _x_mode(mode: str) -> str | None:
    return next((part for part in _mode_parts(mode)
                 if part == "x" or part.startswith("x-")), None)


def _has_x_component(mode: str) -> bool:
    return _x_mode(mode) is not None


def _x_variable_name(mode: str) -> str:
    x_mode = _x_mode(mode)
    if x_mode == "x-class":
        return "x_class_relative"
    if x_mode == "x-provenance":
        return "x_provenance_relative"
    if x_mode == "x-slot-rep":
        return "x_slot_rep_relative"
    if x_mode == "x-geometry":
        return "x_geometry_relative"
    if x_mode == "x-class-rep":
        return "x_class_rep_relative"
    return "x_relative"


def _has_component(mode: str, component: str) -> bool:
    return (mode == component
            or (component == "x" and _has_x_component(mode))
            or component in mode.split("+"))


@dataclass(frozen=True)
class Config:
    """Frozen knobs for a direct Student outer run."""

    inner_steps: int = 4
    inner_lr: float = 0.001
    inner_batch: int = 16
    outer_steps: int = 4
    outer_lr: float = 0.02
    support_batch: int = 16
    old_weight: float = 0.25
    new_weight: float = 1.0
    # Optional construction-time emphasis for new support response.  The
    # default preserves the historical objective exactly; experiments may
    # increase it without changing X/P/A parameterization or access rules.
    new_loss_scale: float = 1.0
    prompt_weight: float = 0.0
    prompt_margin_weight: float = 0.0
    prompt_margin: float = 0.20
    x_old_radius: float = 0.05
    x_new_radius: float = 0.20
    x_anchor_weight: float = 0.05
    x_new_prior_weight: float = 0.0
    old_barrier_weight: float = 0.0
    old_barrier_margin: float = 0.0
    operator_aligned: bool = False
    operator_aux_weight: float = 0.0
    freeze_text_inner: bool = False
    class_center_weight: float = 0.0
    p_mix: float = 0.25
    p_temperature: float = 0.20
    p_sinkhorn_iters: int = 24
    p_balance_weight: float = 0.01
    p_class_constrained: bool = False
    p_old_context_ratio: float = 0.0
    p_old_context_cap: float = 0.0
    p_response_gate: bool = False
    p_gate_margin: float = 0.0
    p_gate_old_tolerance: float = 0.0
    p_gate_per_class: bool = False
    a_pool: int = 4
    a_temperature: float = 0.20
    a_existing_logit: float = 3.0
    a_max_revisions_per_row: int | None = None
    a_revision_margin: float = 0.0
    a_message_weight: float = 0.0
    a_marginal: bool = False
    a_marginal_inner_steps: int = 2
    a_marginal_new_margin: float = 0.0
    a_marginal_old_tolerance: float = 0.0
    a_entropy_weight: float = 0.0
    geometry_anchor_weight: float = 0.50
    geometry_margin_weight: float = 1.00
    geometry_smoothness_weight: float = 0.25
    geometry_margin: float = 0.20
    candidate_prompt_margin_weight: float = 0.0
    candidate_prompt_margin: float = 0.20
    old_graph_anchor_weight: float = 0.0
    old_graph_anchor_fraction: float = 1.0
    class_representation_weight: float = 1.0
    slot_representation_weight: float = 0.25
    slot_mass_weighted: bool = False
    train_seeds: tuple[int, ...] = (9300,)
    validation_seeds: tuple[int, ...] = (9310,)
    plan_seed: int = 9320

    def __post_init__(self) -> None:
        if self.inner_steps < 1 or self.outer_steps < 1:
            raise ValueError("inner_steps and outer_steps must be positive")
        if self.inner_lr <= 0 or self.outer_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.inner_batch < 2 or self.support_batch < 2:
            raise ValueError("batches must contain at least two rows")
        if not 0 <= self.p_mix <= 1:
            raise ValueError("p_mix must be in [0, 1]")
        if self.p_temperature <= 0 or self.a_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.p_sinkhorn_iters < 1 or self.a_pool < 1:
            raise ValueError("invalid assignment/topology iteration count")
        if self.a_marginal_inner_steps < 1:
            raise ValueError("a_marginal_inner_steps must be positive")
        if (self.a_max_revisions_per_row is not None
                and self.a_max_revisions_per_row < 0):
            raise ValueError("a_max_revisions_per_row must be nonnegative")
        for name in ("old_weight", "new_weight", "new_loss_scale",
                     "prompt_weight",
                     "prompt_margin_weight", "prompt_margin",
                     "x_old_radius",
                      "x_new_radius", "x_anchor_weight", "class_center_weight",
                     "old_barrier_weight", "old_barrier_margin",
                     "operator_aux_weight",
                     "x_new_prior_weight", "p_balance_weight",
                      "p_old_context_ratio", "p_old_context_cap",
                       "p_gate_margin", "p_gate_old_tolerance",
                     "a_entropy_weight", "geometry_anchor_weight",
                     "a_existing_logit",
                     "a_revision_margin",
                       "a_message_weight", "a_marginal_new_margin",
                      "a_marginal_old_tolerance",
                     "geometry_margin_weight", "geometry_smoothness_weight",
                     "geometry_margin", "candidate_prompt_margin_weight",
                     "candidate_prompt_margin", "old_graph_anchor_weight",
                     "old_graph_anchor_fraction",
                     "class_representation_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if (not math.isfinite(float(self.slot_representation_weight))
                or float(self.slot_representation_weight) < 0):
            raise ValueError("invalid slot_representation_weight")
        if self.new_weight == 0 and self.old_weight == 0:
            raise ValueError("at least one support loss must be weighted")
        if set(self.train_seeds) & set(self.validation_seeds):
            raise ValueError("train and validation seeds must be disjoint")

    def dict(self) -> dict:
        return asdict(self)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _aggregate_slot_losses(losses: list[torch.Tensor], groups: list[int],
                           base: dict, selected: torch.Tensor,
                           group_source: torch.Tensor,
                           mass_weighted: bool) -> torch.Tensor:
    """Aggregate slot-routed losses without letting tiny slots dominate.

    The default remains the historical equal-slot mean.  The optional
    mass-weighted path uses the frozen synthetic mass represented by each
    editable slot, so the outer update follows the same payload measure as
    the provenance assignment.
    """
    if not losses:
        raise ValueError("cannot aggregate an empty slot loss list")
    stacked = torch.stack(losses)
    if not mass_weighted:
        return stacked.mean()
    mass = base.get("mass")
    if mass is None:
        raise ValueError("mass-weighted slot aggregation requires base.mass")
    selected = selected.long()
    source = group_source.long().to(selected.device)
    weights = []
    mass = mass.to(selected.device).float()
    for group_id in groups:
        rows = selected[source[selected] == int(group_id)]
        if len(rows) != 1:
            raise ValueError("mass-weighted slot aggregation expects one row per slot")
        weights.append(mass[rows[0]].clamp_min(0.0))
    weights = torch.stack(weights)
    if float(weights.sum()) <= 0:
        return stacked.mean()
    return (stacked * weights).sum() / weights.sum()


def clone_memory(memory: dict) -> dict:
    return {key: _clone(value) for key, value in memory.items()}


def _digest_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(memory):
        digest.update(key.encode())
        value = memory[key]
        if isinstance(value, torch.Tensor):
            digest.update(_digest_tensor(value).encode())
        elif isinstance(value, (list, tuple)):
            digest.update(json.dumps(list(value), ensure_ascii=True).encode())
        elif isinstance(value, (str, int, float, bool)) or value is None:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def validate_memory(memory: dict, name: str = "memory") -> None:
    required = {"x", "text_emb", "texts", "edge_index"}
    missing = sorted(required.difference(memory))
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    x, text, edge = memory["x"], memory["text_emb"], memory["edge_index"]
    if x.ndim != 2 or text.ndim != 2 or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError(f"{name} has invalid tensor ranks")
    n = len(x)
    if len(text) != n or len(memory["texts"]) != n:
        raise ValueError(f"{name} row fields have inconsistent lengths")
    if n < 2 or not torch.isfinite(x).all() or not torch.isfinite(text).all():
        raise ValueError(f"{name} contains invalid payload")
    if edge.numel() and (edge.min() < 0 or edge.max() >= n):
        raise ValueError(f"{name} contains an invalid edge endpoint")


def _student(seed: int, in_dim: int, text_dim: int, device: str):
    """Create the same two-layer GCN/InfoNCE Student used downstream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = G.CLIPStudent(in_dim, text_dim=text_dim).to(device)
    return model


def _matmul_adj(adj: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if adj.layout == torch.sparse_coo:
        return torch.sparse.mm(adj, values)
    return adj @ values


def _functional_graph(params: dict, x: torch.Tensor,
                      adjacency: torch.Tensor) -> torch.Tensor:
    h = _matmul_adj(adjacency, F.linear(
        x, params["g1.lin.weight"], params["g1.lin.bias"]))
    h = F.relu(h)
    h = _matmul_adj(adjacency, F.linear(
        h, params["g2.lin.weight"], params["g2.lin.bias"]))
    return F.normalize(h, dim=1)


def _functional_text(params: dict, text: torch.Tensor) -> torch.Tensor:
    return F.normalize(F.linear(text, params["text_proj.weight"],
                                params["text_proj.bias"]), dim=1)


def _nce(graph: torch.Tensor, text: torch.Tensor, rows: torch.Tensor,
         logit_scale: torch.Tensor,
         context_rows: torch.Tensor | None = None) -> torch.Tensor:
    """Compute symmetric InfoNCE for targets against a negative context.

    The default is the original square in-batch loss.  A provenance group can
    contain only one raw row, so it uses that row as the target and a fixed
    context containing the target plus other rows from the same sealed split.
    This preserves a valid contrastive signal without inventing extra raw
    samples for the synthetic slot.
    """
    if len(rows) < 1:
        raise ValueError("InfoNCE requires at least one target row")
    if context_rows is None:
        context_rows = rows
    if len(context_rows) < 2:
        raise ValueError("InfoNCE context requires at least two rows")
    context_rows = context_rows.long()
    positions = []
    context_list = context_rows.detach().cpu().tolist()
    for row in rows.detach().cpu().tolist():
        try:
            positions.append(context_list.index(int(row)))
        except ValueError as exc:
            raise ValueError("every InfoNCE target must be in its context") from exc
    labels = torch.tensor(positions, dtype=torch.long, device=graph.device)
    scale = logit_scale.exp().clamp(max=100)
    graph_logits = scale * graph[rows] @ text[context_rows].T
    text_logits = scale * text[rows] @ graph[context_rows].T
    return (F.cross_entropy(graph_logits, labels)
            + F.cross_entropy(text_logits, labels)) / 2


def _adam_step(params: dict, grads: tuple[torch.Tensor, ...],
               first: dict, second: dict, step: int,
               lr: float, frozen_names: set[str] | None = None
               ) -> tuple[dict, dict, dict]:
    """Differentiable Adam update matching the canonical Student optimizer."""
    new, next_first, next_second = {}, {}, {}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    frozen_names = frozen_names or set()
    for (name, param), grad in zip(params.items(), grads):
        if name in frozen_names:
            new[name] = param
            next_first[name], next_second[name] = first[name], second[name]
            continue
        m = beta1 * first[name] + (1 - beta1) * grad
        v = beta2 * second[name] + (1 - beta2) * grad.square()
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        # Put epsilon inside the square root.  The outer pass differentiates
        # through Adam; sqrt(v) has an infinite derivative at v=0 for sparse
        # or ReLU-dead parameters, while sqrt(v + eps) is finite and is the
        # usual numerically stable meta-learning form.
        new[name] = param - lr * m_hat / (v_hat + eps).sqrt()
        next_first[name], next_second[name] = m, v
    return new, next_first, next_second


def unroll_student(x: torch.Tensor, text: torch.Tensor, adjacency: torch.Tensor,
                   seed: int, cfg: Config, create_graph: bool) -> dict:
    """Run K differentiable Student updates on candidate memory."""
    model = _student(seed, x.shape[1], text.shape[1], str(x.device))
    params = dict(model.named_parameters())
    first = {key: torch.zeros_like(value) for key, value in params.items()}
    second = {key: torch.zeros_like(value) for key, value in params.items()}
    generator = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 1)
    losses = []
    frozen_names = (set(params) & {
        "text_proj.weight", "text_proj.bias", "logit_scale"
    } if cfg.freeze_text_inner else set())
    for step in range(1, int(cfg.inner_steps) + 1):
        rows = torch.randperm(len(x), generator=generator)[:min(
            int(cfg.inner_batch), len(x))].to(x.device)
        graph = _functional_graph(params, x, adjacency)
        encoded = _functional_text(params, text)
        loss = _nce(graph, encoded, rows, params["logit_scale"])
        grads = torch.autograd.grad(
            loss, tuple(params.values()), create_graph=create_graph,
            retain_graph=create_graph, allow_unused=False)
        params, first, second = _adam_step(
            params, grads, first, second, step, float(cfg.inner_lr),
            frozen_names=frozen_names)
        losses.append(loss)
        if not create_graph:
            params = {key: value.detach().requires_grad_(True)
                      for key, value in params.items()}
            first = {key: value.detach() for key, value in first.items()}
            second = {key: value.detach() for key, value in second.items()}
    return {"params": params, "inner_losses": losses}


def _group_batches(indices: torch.Tensor, batch: int, seed: int) -> list[torch.Tensor]:
    if len(indices) < 2:
        return []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    size = min(int(batch), len(shuffled))
    return [shuffled[start:start + size] for start in range(0, len(shuffled), size)
            if len(shuffled[start:start + size]) >= 2]


def _context_batches(targets: torch.Tensor, available: torch.Tensor,
                     batch: int, seed: int) -> list[dict]:
    """Build target/context batches without changing the sealed split."""
    if len(targets) < 1 or len(available) < 2:
        return []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    targets = targets[torch.randperm(len(targets), generator=generator)]
    available = available.long()
    output = []
    for start in range(0, len(targets), max(1, int(batch))):
        target = targets[start:start + max(1, int(batch))]
        context = target.tolist()
        for row in available.tolist():
            if row not in context:
                context.append(int(row))
            if len(context) >= max(2, int(batch)):
                break
        if len(context) < 2:
            continue
        output.append({"targets": target.tolist(),
                       "context": context[:max(2, int(batch))]})
    return output


def make_support_plan(support: dict, old_rows: int, batch: int,
                      seed: int,
                      support_class_id: torch.Tensor | None = None,
                      stratify_new: bool = False,
                      support_slot_id: torch.Tensor | None = None,
                      stratify_slot: bool = False) -> dict:
    """Create fixed, label-free old/new train/validation support batches."""
    n = len(support["x"])
    if old_rows < 2 or old_rows > n - 2:
        raise ValueError("support must contain at least two old and two new rows")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    plan = {"seed": int(seed), "old_rows": int(old_rows), "splits": {}}
    # Sample one permutation per group and partition it once.  Re-sampling
    # independently for train and validation can make the two support losses
    # overlap, which leaks the outer-training rows into model selection.
    groups = {"old": torch.arange(old_rows)}
    if stratify_new and stratify_slot:
        raise ValueError("new support can be stratified by class or slot, not both")
    if stratify_new:
        if support_class_id is None or len(support_class_id) != n:
            raise ValueError("stratified support plan requires aligned class ids")
        class_ids = sorted(set(int(value) for value in
                               support_class_id[old_rows:].tolist()))
        if not class_ids or any(value < 0 for value in class_ids):
            raise ValueError("stratified support plan requires new class ids")
        groups["new"] = None
    elif stratify_slot:
        if support_slot_id is None or len(support_slot_id) != n:
            raise ValueError("stratified slot plan requires aligned slot ids")
        slot_id = support_slot_id.detach().cpu().long()
        if bool((slot_id[:int(old_rows)] >= 0).any()):
            raise ValueError("old support rows must have slot id -1")
        if bool((slot_id[int(old_rows):] < 0).any()):
            raise ValueError("new support rows need nonnegative slot ids")
        groups["new"] = None
    else:
        groups["new"] = torch.arange(old_rows, n)
    for name, values in groups.items():
        if name == "new" and (stratify_new or stratify_slot):
            continue
        if len(values) < 4:
            raise ValueError(
                f"support group {name} needs at least four rows for disjoint "
                "train/validation InfoNCE support")
        shuffled = values[torch.randperm(len(values), generator=generator)]
        cut = max(2, min(len(values) - 2, int(round(len(values) * 0.75))))
        split_values = {"train": shuffled[:cut],
                        "validation": shuffled[cut:]}
        for split_name, selected in split_values.items():
            batches = _group_batches(selected, batch, int(seed) + len(name))
            if not batches:
                raise ValueError(f"support group {name} cannot form InfoNCE batch")
            plan["splits"].setdefault(split_name, {})[name] = [
                rows.tolist() for rows in batches]
    if stratify_new:
        # Keep each class separated inside the shared plan.  _class_plan can
        # then select one class without reconstructing or leaking the split.
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 97)
        for class_id in class_ids:
            values = torch.where(support_class_id == int(class_id))[0]
            values = values[values >= int(old_rows)]
            if len(values) < 4:
                raise ValueError(
                    f"new class {class_id} needs at least four support rows")
            shuffled = values[torch.randperm(len(values), generator=generator)]
            cut = max(2, min(len(values) - 2, int(round(len(values) * 0.75))))
            # Keep the split membership first.  A class may contribute one
            # held-out row, which is still a valid target when the other
            # held-out new-class rows provide its negative context.
            plan["splits"].setdefault("_stratified_new_rows", {})
            for split_name, selected in (("train", shuffled[:cut]),
                                         ("validation", shuffled[cut:])):
                if len(selected) == 0:
                    raise ValueError(
                        f"new class {class_id} has no {split_name} support row")
                plan["splits"].setdefault("_stratified_new_rows", {}).setdefault(
                    split_name, {})[int(class_id)] = selected.tolist()
        # Convert each sealed class partition into target/context batches.  The
        # targets stay class-specific, while contexts stay inside the same
        # train or validation partition and may contain other new classes.
        raw_rows = plan["splits"].pop("_stratified_new_rows")
        for split_name, class_rows in raw_rows.items():
            available = torch.tensor(
                [row for rows in class_rows.values() for row in rows],
                dtype=torch.long)
            for class_id, selected in sorted(class_rows.items()):
                targets = torch.tensor(selected, dtype=torch.long)
                batches = _context_batches(
                    targets, available, batch,
                    int(seed) + 97 + int(class_id))
                if not batches:
                    raise ValueError(
                        f"new class {class_id} cannot form {split_name} batch")
                plan["splits"].setdefault(split_name, {}).setdefault(
                    "new", []).extend(batches)
    if stratify_slot:
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 197)
        slot_ids = sorted(set(int(value) for value in
                              support_slot_id[old_rows:].tolist()))
        selected_by_split = {"train": {}, "validation": {}}
        for slot_id in slot_ids:
            values = torch.where(support_slot_id == int(slot_id))[0]
            values = values[values >= int(old_rows)]
            if len(values) < 1:
                raise ValueError(f"slot {slot_id} has no support rows")
            shuffled = values[torch.randperm(len(values), generator=generator)]
            if len(values) == 1:
                cut = 1
            else:
                cut = max(1, min(len(values) - 1,
                                 int(round(len(values) * 0.75))))
            for split_name, selected in (("train", shuffled[:cut]),
                                         ("validation", shuffled[cut:])):
                if len(selected) == 0:
                    continue
                selected_by_split[split_name][slot_id] = selected
        for split_name, slot_values in selected_by_split.items():
            if not slot_values:
                continue
            available = torch.cat(list(slot_values.values()))
            for slot_id, selected in sorted(slot_values.items()):
                batches = _context_batches(
                    selected, available, batch,
                    int(seed) + 197 + int(slot_id))
                if not batches:
                    raise ValueError(
                        f"slot {slot_id} cannot form target/context batch")
                plan["splits"].setdefault(split_name, {}).setdefault(
                    "new", []).extend(batches)
    return plan


def _plan_tensors(plan: dict, split: str, device: str) -> dict:
    output = {}
    for name, values in plan["splits"][split].items():
        output[name] = []
        for rows in values:
            if isinstance(rows, dict):
                output[name].append({
                    "targets": torch.tensor(rows["targets"], dtype=torch.long,
                                             device=device),
                    "context": torch.tensor(rows["context"], dtype=torch.long,
                                             device=device),
                })
            else:
                output[name].append(torch.tensor(
                    rows, dtype=torch.long, device=device))
    return output


def _class_plan(plan: dict, support_class_id: torch.Tensor,
                class_id: int, batch: int) -> dict:
    """Filter an existing sealed train/validation plan to one new class.

    The split is created before this function is called.  Filtering therefore
    cannot move a row from validation into train, while a class-aware outer
    step only receives the rows that can legitimately supervise its slots.
    """
    class_id = int(class_id)
    support_class_id = support_class_id.detach().cpu().long()
    result = {"seed": plan["seed"], "old_rows": plan["old_rows"],
              "splits": {}}
    for split_name, groups in plan["splits"].items():
        new_batches = groups.get("new", [])
        if not new_batches:
            raise ValueError(
                f"class {class_id} has no {split_name} support batch")

        # A class-stratified plan stores a target/context pair.  The target
        # rows are class-specific, but the context deliberately contains
        # other new classes as negatives.  Filter only the targets; rebuilding
        # the batch from class rows would discard those negatives and would
        # incorrectly reject a valid one-row validation target.
        context_pool = []
        for raw_batch in new_batches:
            _, raw_context = _batch_targets_context(raw_batch)
            raw_context = (raw_context if raw_context is not None
                           else _batch_targets_context(raw_batch)[0])
            context_pool.extend(torch.as_tensor(
                raw_context, dtype=torch.long).tolist())

        filtered_batches = []
        target_count = 0
        for raw_batch in new_batches:
            targets, context = _batch_targets_context(raw_batch)
            targets = torch.as_tensor(targets, dtype=torch.long)
            if len(targets) == 0:
                continue
            if bool((targets < 0).any()) or bool(
                    (targets >= len(support_class_id)).any()):
                raise ValueError("support plan contains an invalid target row")
            keep = support_class_id[targets] == class_id
            targets = targets[keep]
            if len(targets) == 0:
                continue

            if context is None:
                context = targets
            context = torch.as_tensor(context, dtype=torch.long)
            context_values = context.tolist()
            context_set = set(context_values)
            for target in targets.tolist():
                if target not in context_set:
                    context_values.append(int(target))
                    context_set.add(int(target))
            # InfoNCE needs at least one negative.  If a caller supplied a
            # minimal context, extend it using rows from the same sealed
            # split; this changes no train/validation membership.
            for candidate in context_pool:
                if len(context_values) >= 2:
                    break
                if candidate not in context_set:
                    context_values.append(int(candidate))
                    context_set.add(int(candidate))
            if len(context_values) < 2:
                raise ValueError(
                    f"class {class_id} has fewer than two {split_name} "
                    "context rows")
            filtered_batches.append({
                "targets": targets.tolist(),
                "context": context_values,
            })
            target_count += len(targets)

        if target_count == 0:
            raise ValueError(
                f"class {class_id} has no {split_name} target rows")
        result["splits"][split_name] = {
            "old": groups.get("old", []),
            "new": filtered_batches,
        }
    return result


def _slot_plan(plan: dict, support_slot_id: torch.Tensor,
               slot_id: int, batch: int) -> dict:
    """Filter a sealed support split to one synthetic-slot provenance group.

    The train/validation partition is made before this function is called.
    Filtering an existing partition keeps the outer objective from moving a
    raw row between train and validation merely because it belongs to a
    particular slot.  Rebatching only changes the order within the already
    sealed split.
    """
    support_slot_id = support_slot_id.detach().cpu().long()
    slot_id = int(slot_id)
    if len(support_slot_id) <= int(plan["old_rows"]):
        raise ValueError("slot ids must cover current support rows")
    if bool((support_slot_id[:int(plan["old_rows"])] >= 0).any()):
        raise ValueError("old support rows must have slot id -1")
    result = {"seed": plan["seed"], "old_rows": plan["old_rows"],
              "splits": {}}
    for split_name, groups in plan["splits"].items():
        old_batches = groups.get("old", [])
        new_batches = groups.get("new", [])
        if not new_batches:
            raise ValueError(f"support plan has no new rows in {split_name}")
        all_values = torch.cat([
            torch.as_tensor(_batch_targets_context(rows)[0], dtype=torch.long)
            if isinstance(rows, dict)
            else torch.tensor(rows, dtype=torch.long)
            for rows in new_batches
        ])
        values = all_values[support_slot_id[all_values] == slot_id]
        if len(values) == 0:
            # A one-row slot may be assigned entirely to the other sealed
            # split.  The global validation path handles this case; a slot
            # objective should simply have no batches here.
            result["splits"][split_name] = {"old": old_batches, "new": []}
            continue
        slot_batches = _context_batches(
            values, all_values, batch, int(plan["seed"]) + 193 + slot_id)
        if not slot_batches:
            raise ValueError(
                f"slot {slot_id} cannot form target/context support batch")
        result["splits"][split_name] = {
            "old": old_batches,
            "new": slot_batches,
        }
    return result


def _batch_targets_context(batch) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        return batch["targets"], batch["context"]
    return batch, None


def support_losses(params: dict, support: dict, support_adj: torch.Tensor,
                   plan: dict, split: str, cfg: Config,
                   support_class_id: torch.Tensor | None = None,
                   prompt_emb: torch.Tensor | None = None,
                   prompt_ids: torch.Tensor | None = None,
                   class_filter: int | None = None) -> dict:
    sx = support["x"].to(params["g1.lin.weight"].device)
    st = support["text_emb"].to(sx.device)
    graph = _functional_graph(params, sx, support_adj)
    text = _functional_text(params, st)
    group_batches = _plan_tensors(plan, split, str(sx.device))
    result = {}
    for name, batches in group_batches.items():
        if not batches:
            continue
        values = []
        for batch in batches:
            targets, context = _batch_targets_context(batch)
            values.append(_nce(graph, text, targets,
                               params["logit_scale"], context))
        result[name] = torch.stack(values).mean()
    total = sx.new_zeros(())
    if "old" in result:
        total = total + float(cfg.old_weight) * result["old"]
    if "new" in result:
        total = total + (float(cfg.new_weight)
                         * float(cfg.new_loss_scale) * result["new"])
    if float(cfg.prompt_weight) > 0 or float(cfg.prompt_margin_weight) > 0:
        if support_class_id is None or prompt_emb is None:
            raise ValueError(
                "prompt objectives require support_class_id and prompt_emb")
        if prompt_ids is None:
            prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)
        prompt_ids = prompt_ids.to(sx.device)
        class_id = support_class_id.to(sx.device).long()
        prompt_batches = []
        for batch in group_batches.get("new", []):
            rows, _ = _batch_targets_context(batch)
            prompt_batches.append(rows)
        values = []
        margin_values = []
        encoded_prompt = _functional_text(
            params, prompt_emb.to(sx.device))
        for rows in prompt_batches:
            valid = class_id[rows] >= 0
            rows = rows[valid]
            if len(rows) < 1:
                continue
            targets = (class_id[rows, None] == prompt_ids[None, :]).long()
            if not bool(targets.any(dim=1).all()):
                raise ValueError("support class id is absent from prompt ids")
            target = targets.argmax(dim=1)
            graph = _functional_graph(params, sx, support_adj)
            logits = params["logit_scale"].exp().clamp(max=100) * (
                graph[rows] @ encoded_prompt.T)
            if float(cfg.prompt_weight) > 0:
                values.append(F.cross_entropy(logits, target))
            if float(cfg.prompt_margin_weight) > 0:
                true_logit = logits.gather(1, target[:, None]).squeeze(1)
                negative = logits.masked_fill(targets.bool(), float("-inf"))
                hardest_negative = negative.max(dim=1).values
                margin_values.append(F.relu(
                    float(cfg.prompt_margin) - true_logit + hardest_negative
                ).mean())
        if float(cfg.prompt_weight) > 0:
            if not values:
                raise ValueError("prompt loss has no valid new support rows")
            result["prompt"] = torch.stack(values).mean()
            total = total + float(cfg.prompt_weight) * result["prompt"]
        if float(cfg.prompt_margin_weight) > 0:
            if not margin_values:
                raise ValueError("prompt margin has no valid new support rows")
            result["prompt_margin"] = torch.stack(margin_values).mean()
            total = total + float(cfg.prompt_margin_weight) * result["prompt_margin"]
    if float(cfg.class_center_weight) > 0:
        if class_filter is None or support_class_id is None or prompt_emb is None:
            raise ValueError(
                "class center objective requires class_filter, class ids and prompts")
        class_id = support_class_id.to(sx.device).long()
        class_rows = torch.cat([
            _batch_targets_context(batch)[0]
            for batch in group_batches.get("new", [])
        ])
        class_rows = class_rows[class_id[class_rows] == int(class_filter)]
        if len(class_rows) < 1:
            raise ValueError("class center objective has no support rows")
        if prompt_ids is None:
            prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)
        prompt_ids = prompt_ids.to(sx.device)
        target_positions = torch.where(prompt_ids == int(class_filter))[0]
        if len(target_positions) != 1:
            raise ValueError("class center objective cannot resolve class prompt")
        encoded_prompt = _functional_text(params, prompt_emb.to(sx.device))
        center = F.normalize(graph[class_rows].mean(dim=0, keepdim=True), dim=1)[0]
        target = encoded_prompt[target_positions[0]]
        result["class_center"] = 1.0 - (center * target).sum()
        total = total + float(cfg.class_center_weight) * result["class_center"]
    result["total"] = total
    return result


def _response_pair(params: dict, support: dict, support_adj: torch.Tensor,
                   plan: dict, cfg: Config,
                   support_class_id: torch.Tensor | None = None,
                   prompt_emb: torch.Tensor | None = None,
                   prompt_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return new and old Student response losses on a sealed support plan."""
    values = support_losses(params, support, support_adj, plan, "validation", cfg,
                            support_class_id=support_class_id,
                            prompt_emb=prompt_emb, prompt_ids=prompt_ids)
    return values.get("new", values["total"]), values.get("old", values["total"])


def _paired_p_response_gate(
        base: dict, support: dict, old_rows: int, variables: dict,
        mode: str, selected: torch.Tensor, cfg: Config, device: str,
        seed: int, plan: dict, support_adj: torch.Tensor,
        support_class_id: torch.Tensor | None = None,
        prompt_emb: torch.Tensor | None = None,
        prompt_ids: torch.Tensor | None = None,
        base_slot_class_id: torch.Tensor | None = None,
        base_slot_group_id: torch.Tensor | None = None) -> tuple[dict, dict]:
    """Compare mixed-P and pure-new-P under one identical Student probe.

    The probe is deliberately outside the differentiable outer objective.  It
    uses the same Student initialization, minibatch trajectory, support split,
    and inner-step count for both candidates.  The gate therefore measures the
    incremental response of old-context transport instead of comparing the
    unrelated magnitudes of new and old losses inside one candidate.
    """
    pure, pure_info = _candidate(
        base, support, old_rows, variables, mode, selected, cfg, device,
        base_slot_class_id=base_slot_class_id,
        base_slot_group_id=base_slot_group_id,
        support_class_id=support_class_id,
        response_gate={"keep_old_context": 0.0})

    pure_student = unroll_student(
        pure["x"], pure["text_emb"], pure_info["adjacency"],
        int(seed), cfg, False)
    pure_new, pure_old = _response_pair(
        pure_student["params"], support, support_adj, plan, cfg,
        support_class_id=support_class_id, prompt_emb=prompt_emb,
        prompt_ids=prompt_ids)

    def evaluate(mask: torch.Tensor,
                 response_plan: dict = plan) -> tuple[float, float]:
        mixed, mixed_info = _candidate(
            base, support, old_rows, variables, mode, selected, cfg, device,
            base_slot_class_id=base_slot_class_id,
            base_slot_group_id=base_slot_group_id,
            support_class_id=support_class_id,
            response_gate={"keep_old_context_rows": mask.detach().cpu().tolist()})
        mixed_student = unroll_student(
            mixed["x"], mixed["text_emb"], mixed_info["adjacency"],
            int(seed), cfg, False)
        mixed_new, mixed_old = _response_pair(
            mixed_student["params"], support, support_adj, response_plan, cfg,
            support_class_id=support_class_id, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids)
        return float(mixed_new.detach()), float(mixed_old.detach())

    new_count = len(base["x"]) - int(old_rows)
    use_per_class = bool(
        cfg.p_gate_per_class and base_slot_class_id is not None
        and len(base_slot_class_id) == len(base["x"]))
    row_keep = torch.zeros(new_count, dtype=torch.bool)
    class_records = []
    if use_per_class:
        class_ids = base_slot_class_id.detach().cpu().long()[old_rows:]
        for class_id in sorted(set(int(value) for value in class_ids.tolist())):
            mask = class_ids == int(class_id)
            if support_class_id is None:
                raise ValueError("per-class P gate requires support class ids")
            class_plan = _class_plan(
                plan, support_class_id, int(class_id), cfg.support_batch)
            pure_class_new, pure_class_old = _response_pair(
                pure_student["params"], support, support_adj, class_plan, cfg,
                support_class_id=support_class_id, prompt_emb=prompt_emb,
                prompt_ids=prompt_ids)
            mixed_new, mixed_old = evaluate(mask, class_plan)
            delta_new = mixed_new - float(pure_class_new.detach())
            delta_old = mixed_old - float(pure_class_old.detach())
            keep = (delta_new < -float(cfg.p_gate_margin)
                    and delta_old <= float(cfg.p_gate_old_tolerance))
            if keep:
                row_keep[mask] = True
            class_records.append({
                "class_id": int(class_id),
                "row_count": int(mask.sum()),
                "keep_old_context": float(keep),
                "new_delta": float(delta_new),
                "old_delta": float(delta_old),
                "mixed_new": float(mixed_new),
                "mixed_old": float(mixed_old),
                "pure_new": float(pure_class_new.detach()),
                "pure_old": float(pure_class_old.detach()),
            })
        gate = {"keep_old_context_rows": row_keep.tolist()}
        audit = {
            "p_response_gate": float(row_keep.float().mean()),
            "p_response_gate_rows": row_keep.tolist(),
            "p_response_gate_per_class": True,
            "p_response_gate_classes": class_records,
            "p_response_new_delta": float(sum(
                row["new_delta"] for row in class_records)
                / max(1, len(class_records))),
            "p_response_old_delta": float(sum(
                row["old_delta"] for row in class_records)
                / max(1, len(class_records))),
            "p_response_pure_new": float(pure_new.detach()),
            "p_response_pure_old": float(pure_old.detach()),
            "p_gate_margin": float(cfg.p_gate_margin),
            "p_gate_old_tolerance": float(cfg.p_gate_old_tolerance),
            "p_response_gate_rule": (
                "per-class: delta_new < -p_gate_margin and "
                "delta_old <= p_gate_old_tolerance"),
        }
        return gate, audit

    mixed_new, mixed_old = evaluate(torch.ones(new_count, dtype=torch.bool))
    delta_new = mixed_new - float(pure_new.detach())
    delta_old = mixed_old - float(pure_old.detach())
    keep = (delta_new < -float(cfg.p_gate_margin)
            and delta_old <= float(cfg.p_gate_old_tolerance))
    gate = {"keep_old_context": float(keep)}
    audit = {
        "p_response_gate": float(keep),
        "p_response_gate_per_class": False,
        "p_response_new_delta": float(delta_new),
        "p_response_old_delta": float(delta_old),
        "p_response_mixed_new": float(mixed_new),
        "p_response_mixed_old": float(mixed_old),
        "p_response_pure_new": float(pure_new.detach()),
        "p_response_pure_old": float(pure_old.detach()),
        "p_gate_margin": float(cfg.p_gate_margin),
        "p_gate_old_tolerance": float(cfg.p_gate_old_tolerance),
        "p_response_gate_rule": (
            "delta_new < -p_gate_margin and "
            "delta_old <= p_gate_old_tolerance"),
    }
    return gate, audit


def _slot_text_targets(params: dict, candidate: dict, support: dict,
                       old_rows: int, support_slot_id: torch.Tensor,
                       base_slot_group_id: torch.Tensor) -> torch.Tensor:
    """Build candidate-row text targets from sealed slot provenance.

    Old candidate rows already correspond to the old prefix of ``support``.
    New candidate rows receive the mean current-delta text embedding assigned
    to their provenance slot.  This keeps the outer loss on the candidate
    propagation operator while using only the current legal support.
    """
    n = len(candidate["x"])
    if len(base_slot_group_id) != n or len(support_slot_id) != len(support["x"]):
        raise ValueError("slot target metadata is not aligned")
    if old_rows < 1 or old_rows > n:
        raise ValueError("invalid old_rows for slot text targets")
    group_ids = base_slot_group_id[old_rows:].long()
    expected = torch.arange(n - old_rows, dtype=torch.long)
    if not torch.equal(torch.sort(group_ids.cpu()).values, expected):
        raise ValueError("base slot groups must be a contiguous candidate suffix")
    targets = support["text_emb"][:old_rows].float().clone()
    new_targets = []
    for group_id in range(n - old_rows):
        rows = torch.where(support_slot_id == int(group_id))[0]
        rows = rows[rows >= int(old_rows)]
        if len(rows) == 0:
            raise ValueError(f"slot {group_id} has no current support rows")
        new_targets.append(support["text_emb"][rows].float().mean(dim=0))
    targets = torch.cat((targets, torch.stack(new_targets)), dim=0)
    return _functional_text(params, targets.to(params["g1.lin.weight"].device))


def _class_text_targets(params: dict, candidate: dict, support: dict,
                        old_rows: int, support_class_id: torch.Tensor,
                        base_slot_class_id: torch.Tensor) -> torch.Tensor:
    """Build candidate-row text targets from current-delta class payloads.

    The candidate graph has one or more synthetic rows per incoming class.
    Each new row receives the mean frozen text embedding of the legal current
    raw delta for its class.  This keeps the target label-free with respect to
    report/test nodes and prevents an operator-aligned X+P run from silently
    falling back to the separate raw-support adjacency.
    """
    n = len(candidate["x"])
    if len(base_slot_class_id) != n or len(support_class_id) != len(support["x"]):
        raise ValueError("class target metadata is not aligned")
    if old_rows < 1 or old_rows > n:
        raise ValueError("invalid old_rows for class text targets")
    class_ids = base_slot_class_id[old_rows:].long()
    if bool((class_ids < 0).any()):
        raise ValueError("new candidate rows need nonnegative class ids")
    support_ids = support_class_id.long()
    if bool((support_ids[old_rows:] < 0).any()):
        raise ValueError("new support rows need nonnegative class ids")
    targets = support["text_emb"][:old_rows].float().clone()
    new_targets = []
    for class_id in class_ids.tolist():
        rows = torch.where(support_ids == int(class_id))[0]
        rows = rows[rows >= int(old_rows)]
        if len(rows) == 0:
            raise ValueError(
                f"class {class_id} has no current support rows")
        new_targets.append(support["text_emb"][rows].float().mean(dim=0))
    targets = torch.cat((targets, torch.stack(new_targets)), dim=0)
    return _functional_text(params, targets.to(params["g1.lin.weight"].device))


def operator_aligned_losses(params: dict, candidate: dict,
                            candidate_adj: torch.Tensor, support: dict,
                            old_rows: int, cfg: Config,
                            support_slot_id: torch.Tensor | None = None,
                            base_slot_group_id: torch.Tensor | None = None,
                            support_class_id: torch.Tensor | None = None,
                            base_slot_class_id: torch.Tensor | None = None,
                            prompt_emb: torch.Tensor | None = None,
                            prompt_ids: torch.Tensor | None = None,
                            group_filter: int | None = None) -> dict:
    """Evaluate slot targets with the same propagation operator as Student.

    The earlier support objective trained on the candidate graph but measured
    the outer response on a separate support graph.  This operator-aligned
    variant uses candidate graph rows as graph targets and current-delta slot
    or class text prototypes as text targets, so the graph operator is
    identical on both sides of the outer loss.  Exactly one target mapping is
    required: slot provenance for ``x-provenance`` or class provenance for
    ``x-class+p``.
    """
    device = params["g1.lin.weight"].device
    graph = _functional_graph(params, candidate["x"].to(device), candidate_adj)
    if ((support_slot_id is None) != (base_slot_group_id is None)):
        raise ValueError("slot operator alignment needs both slot mappings")
    if support_slot_id is None:
        if support_class_id is None or base_slot_class_id is None:
            raise ValueError("class operator alignment needs both class mappings")
    elif support_class_id is not None and base_slot_class_id is None:
        raise ValueError("class operator alignment needs both class mappings")
    if support_slot_id is not None:
        text = _slot_text_targets(
            params, candidate, support, old_rows,
            support_slot_id.detach().cpu(),
            base_slot_group_id.detach().cpu())
    elif support_class_id is not None:
        text = _class_text_targets(
            params, candidate, support, old_rows,
            support_class_id.detach().cpu(),
            base_slot_class_id.detach().cpu())
    else:
        raise ValueError("operator alignment needs slot or class mappings")
    n = len(candidate["x"])
    context = torch.arange(n, dtype=torch.long, device=device)
    old_targets = torch.arange(old_rows, dtype=torch.long, device=device)
    new_targets = torch.arange(old_rows, n, dtype=torch.long, device=device)
    if group_filter is not None:
        new_rows = torch.arange(old_rows, n, dtype=torch.long, device=device)
        if base_slot_group_id is not None:
            filter_ids = base_slot_group_id.detach().cpu().long()[old_rows:]
        else:
            filter_ids = base_slot_class_id.detach().cpu().long()[old_rows:]
        mask = filter_ids == int(group_filter)
        new_targets = new_rows[mask.to(device)]
        if len(new_targets) == 0:
            raise ValueError("operator-aligned group filter selected no rows")
    if len(old_targets) < 2 or len(new_targets) < 1:
        raise ValueError("operator-aligned loss needs old and new targets")
    scale = params["logit_scale"].exp().clamp(max=100)

    def loss_for(rows: torch.Tensor) -> torch.Tensor:
        labels = rows.clone()
        graph_logits = scale * graph[rows] @ text[context].T
        text_logits = scale * text[rows] @ graph[context].T
        return (F.cross_entropy(graph_logits, labels)
                + F.cross_entropy(text_logits, labels)) / 2

    result = {"old": loss_for(old_targets),
              "new": loss_for(new_targets)}
    total = float(cfg.old_weight) * result["old"]
    total = total + float(cfg.new_weight) * result["new"]
    if float(cfg.prompt_weight) > 0 or float(cfg.prompt_margin_weight) > 0:
        if base_slot_class_id is None or prompt_emb is None:
            raise ValueError(
                "operator-aligned prompt objective requires slot classes and prompts")
        if prompt_ids is None:
            prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)
        new_rows = torch.arange(old_rows, n, dtype=torch.long, device=device)
        class_ids = base_slot_class_id.detach().cpu().long()[old_rows:]
        if bool((class_ids < 0).any()):
            raise ValueError("new candidate slots need construction-time class ids")
        encoded_prompt = _functional_text(params, prompt_emb.to(device))
        prompt_ids = prompt_ids.to(device).long()
        target_positions = []
        for class_id in class_ids.tolist():
            positions = torch.where(prompt_ids == int(class_id))[0]
            if len(positions) != 1:
                raise ValueError(f"cannot resolve prompt for class {class_id}")
            target_positions.append(int(positions[0]))
        target = torch.tensor(target_positions, dtype=torch.long, device=device)
        prompt_logits = scale * graph[new_rows] @ encoded_prompt.T
        if group_filter is not None:
            if base_slot_group_id is not None:
                filter_ids = base_slot_group_id.detach().cpu().long()[old_rows:]
            else:
                filter_ids = class_ids
            mask = filter_ids == int(group_filter)
            prompt_logits = prompt_logits[mask.to(device)]
            target = target[mask.to(device)]
        if float(cfg.prompt_weight) > 0:
            result["prompt"] = F.cross_entropy(prompt_logits, target)
            total = total + float(cfg.prompt_weight) * result["prompt"]
        if float(cfg.prompt_margin_weight) > 0:
            true_logit = prompt_logits.gather(1, target[:, None]).squeeze(1)
            negative = prompt_logits.clone()
            negative.scatter_(1, target[:, None], float("-inf"))
            result["prompt_margin"] = F.relu(
                float(cfg.prompt_margin) - true_logit + negative.max(dim=1).values
            ).mean()
            total = total + float(cfg.prompt_margin_weight) * result["prompt_margin"]
    result["total"] = total
    return result


def operator_aligned_class_losses(params: dict, candidate: dict,
                                  candidate_adj: torch.Tensor, support: dict,
                                  old_rows: int, cfg: Config,
                                  support_class_id: torch.Tensor,
                                  base_slot_class_id: torch.Tensor,
                                  prompt_emb: torch.Tensor | None = None,
                                  prompt_ids: torch.Tensor | None = None,
                                  group_filter: int | None = None
                                  ) -> dict:
    """Operator-aligned objective with class-level positives.

    A class may own several synthetic slots.  Treating those slots as unique
    InfoNCE instances while giving them the same class-mean text target makes
    them negatives of one another.  This variant therefore keeps an old-row
    instance loss, but uses a many-to-one class loss for new rows and a
    class-mean graph prototype for the reverse direction.
    """
    device = params["g1.lin.weight"].device
    graph = _functional_graph(params, candidate["x"].to(device), candidate_adj)
    support_class_id = support_class_id.detach().cpu().long()
    base_slot_class_id = base_slot_class_id.detach().cpu().long()
    new_slot_classes = base_slot_class_id[old_rows:]
    if bool((new_slot_classes < 0).any()):
        raise ValueError("new candidate rows need nonnegative class ids")
    class_ids = sorted(set(int(value) for value in new_slot_classes.tolist()))
    if not class_ids:
        raise ValueError("class operator alignment needs new classes")
    class_targets = []
    for class_id in class_ids:
        rows = torch.where(support_class_id == int(class_id))[0]
        rows = rows[rows >= int(old_rows)]
        if len(rows) == 0:
            raise ValueError(f"class {class_id} has no current support rows")
        class_targets.append(support["text_emb"][rows].float().mean(dim=0))
    class_text = _functional_text(params, torch.stack(class_targets).to(device))

    old_rows_device = torch.arange(old_rows, dtype=torch.long, device=device)
    new_rows_device = torch.arange(len(candidate["x"]), dtype=torch.long,
                                   device=device)[old_rows:]
    old_text = _functional_text(
        params, candidate["text_emb"][:old_rows].float().to(device))
    scale = params["logit_scale"].exp().clamp(max=100)
    old_labels = torch.arange(old_rows, dtype=torch.long, device=device)
    old_graph_logits = scale * graph[old_rows_device] @ old_text.T
    old_text_logits = scale * old_text @ graph[old_rows_device].T
    old_loss = (F.cross_entropy(old_graph_logits, old_labels)
                + F.cross_entropy(old_text_logits, old_labels)) / 2

    class_position = {class_id: index
                      for index, class_id in enumerate(class_ids)}
    new_labels = torch.tensor(
        [class_position[int(value)] for value in new_slot_classes.tolist()],
        dtype=torch.long, device=device)
    new_graph_logits = scale * graph[new_rows_device] @ class_text.T
    new_graph_loss = F.cross_entropy(new_graph_logits, new_labels)
    class_graph = []
    for class_id in class_ids:
        rows = new_rows_device[new_labels == class_position[class_id]]
        class_graph.append(graph[rows].mean(dim=0))
    class_graph = F.normalize(torch.stack(class_graph), dim=1)
    new_text_logits = scale * class_text @ class_graph.T
    class_labels = torch.arange(len(class_ids), dtype=torch.long, device=device)
    new_text_loss = F.cross_entropy(new_text_logits, class_labels)
    new_loss = (new_graph_loss + new_text_loss) / 2
    result = {
        "old": old_loss,
        "new": new_loss,
        "operator_aligned_classwise": True,
        "operator_aligned_class_count": len(class_ids),
    }
    total = float(cfg.old_weight) * old_loss + float(cfg.new_weight) * new_loss
    if float(cfg.prompt_weight) > 0 or float(cfg.prompt_margin_weight) > 0:
        if prompt_emb is None:
            raise ValueError("class operator prompt objective needs prompts")
        if prompt_ids is None:
            prompt_ids = torch.arange(len(prompt_emb), dtype=torch.long)
        encoded_prompt = _functional_text(params, prompt_emb.to(device))
        prompt_ids = prompt_ids.to(device).long()
        positions = []
        for class_id in new_slot_classes.tolist():
            match = torch.where(prompt_ids == int(class_id))[0]
            if len(match) != 1:
                raise ValueError(f"cannot resolve prompt for class {class_id}")
            positions.append(int(match[0]))
        prompt_target = torch.tensor(positions, dtype=torch.long, device=device)
        prompt_logits = scale * graph[new_rows_device] @ encoded_prompt.T
        if float(cfg.prompt_weight) > 0:
            result["prompt"] = F.cross_entropy(prompt_logits, prompt_target)
            total = total + float(cfg.prompt_weight) * result["prompt"]
        if float(cfg.prompt_margin_weight) > 0:
            true_logit = prompt_logits.gather(
                1, prompt_target[:, None]).squeeze(1)
            negative = prompt_logits.clone()
            negative.scatter_(1, prompt_target[:, None], float("-inf"))
            result["prompt_margin"] = F.relu(
                float(cfg.prompt_margin) - true_logit +
                negative.max(dim=1).values).mean()
            total = total + float(cfg.prompt_margin_weight) * result[
                "prompt_margin"]
    result["total"] = total
    return result


def geometry_x_losses(params: dict, candidate: dict,
                      candidate_adj: torch.Tensor, selected: torch.Tensor,
                      cfg: Config,
                      base_slot_class_id: torch.Tensor,
                      prompt_emb: torch.Tensor,
                      prompt_ids: torch.Tensor | None = None) -> dict:
    """Measure the fixed geometry-X objective on candidate synthetic rows.

    This is intentionally a candidate-side objective.  The graph encoder sees
    the same candidate adjacency used by the Student inner loop, and the
    class margin is evaluated against the public class prompts.  It is not a
    selector: the loss is added to the outer objective and gradients flow only
    to the editable X rows.  Text, topology, P, and memory size stay fixed.
    """
    device = params["g1.lin.weight"].device
    selected = selected.to(device).long()
    if len(selected) < 1:
        raise ValueError("geometry-X requires editable candidate rows")
    if bool((selected < 0).any()) or bool((selected >= len(candidate["x"])).any()):
        raise ValueError("geometry-X selected rows are outside candidate")
    class_ids = base_slot_class_id.to(device).long()[selected]
    if bool((class_ids < 0).any()):
        raise ValueError("geometry-X selected rows need class ids")
    prompt_ids = (torch.arange(len(prompt_emb), dtype=torch.long, device=device)
                  if prompt_ids is None else prompt_ids.to(device).long())
    encoded_prompt = _functional_text(params, prompt_emb.to(device))
    graph = _functional_graph(params, candidate["x"].to(device), candidate_adj)
    positions = []
    for class_id in class_ids.detach().cpu().tolist():
        matches = torch.where(prompt_ids == int(class_id))[0]
        if len(matches) != 1:
            raise ValueError(f"cannot resolve prompt for candidate class {class_id}")
        positions.append(int(matches[0]))
    target = torch.tensor(positions, dtype=torch.long, device=device)
    scale = params["logit_scale"].exp().clamp(max=100)
    logits = scale * graph[selected] @ encoded_prompt.T
    true_logit = logits.gather(1, target[:, None]).squeeze(1)
    negative = logits.clone()
    negative.scatter_(1, target[:, None], float("-inf"))
    hardest_negative = negative.max(dim=1).values
    margin = F.relu(float(cfg.geometry_margin) - true_logit
                    + hardest_negative).mean()
    return {"geometry_margin": margin,
            "geometry_selected_rows": int(len(selected)),
            "geometry_class_count": int(len(torch.unique(class_ids)))}


def candidate_class_margin_loss(params: dict, candidate: dict,
                                candidate_adj: torch.Tensor,
                                selected: torch.Tensor,
                                base_slot_class_id: torch.Tensor,
                                prompt_emb: torch.Tensor,
                                prompt_ids: torch.Tensor | None,
                                margin: float) -> dict:
    """Class-conditional margin on the same synthetic graph used by Student.

    This is the differentiable objective for the minimal ``X-class+P+A``
    pilot.  Unlike the earlier support prompt loss, both the graph encoder and
    the target synthetic rows use the candidate propagation operator.  The
    current-delta class ids are construction-time metadata, not report/test
    labels.
    """
    device = params["g1.lin.weight"].device
    selected = selected.to(device).long()
    class_ids = base_slot_class_id.to(device).long()[selected]
    if len(selected) < 1 or bool((class_ids < 0).any()):
        raise ValueError("candidate class margin needs labelled new slots")
    prompt_ids = (torch.arange(len(prompt_emb), dtype=torch.long,
                               device=device)
                  if prompt_ids is None else prompt_ids.to(device).long())
    encoded_prompt = _functional_text(params, prompt_emb.to(device))
    graph = _functional_graph(params, candidate["x"].to(device), candidate_adj)
    positions = []
    for class_id in class_ids.detach().cpu().tolist():
        matches = torch.where(prompt_ids == int(class_id))[0]
        if len(matches) != 1:
            raise ValueError(f"cannot resolve prompt for class {class_id}")
        positions.append(int(matches[0]))
    target = torch.tensor(positions, dtype=torch.long, device=device)
    scale = params["logit_scale"].exp().clamp(max=100)
    logits = scale * graph[selected] @ encoded_prompt.T
    true_logit = logits.gather(1, target[:, None]).squeeze(1)
    negative = logits.clone()
    negative.scatter_(1, target[:, None], float("-inf"))
    hardest_negative = negative.max(dim=1).values
    value = F.relu(float(margin) - true_logit + hardest_negative).mean()
    return {"candidate_prompt_margin": value,
            "candidate_prompt_selected_rows": int(len(selected)),
            "candidate_prompt_class_count": int(len(torch.unique(class_ids)))}


def old_graph_anchor_loss(params: dict, candidate: dict,
                          candidate_adj: torch.Tensor, base: dict,
                          old_rows: int,
                          fraction: float = 1.0) -> torch.Tensor:
    """Keep old synthetic representations stable under candidate propagation.

    The base graph is a sealed synthetic reference.  Its representation is
    evaluated with detached Student parameters, while the candidate branch
    remains differentiable.  This prevents new X/P/A updates from freely
    moving old rows through normalized message passing without introducing
    historical raw data or report labels into construction.
    """
    if old_rows < 1:
        raise ValueError("old graph anchor requires old synthetic rows")
    device = params["g1.lin.weight"].device
    fraction = min(1.0, max(0.0, float(fraction)))
    candidate_graph = _functional_graph(
        params, candidate["x"].to(device), candidate_adj)[:int(old_rows)]
    base_adj = fixed_norm_adj(base["edge_index"], len(base["x"]), str(device))
    detached_params = {key: value.detach() for key, value in params.items()}
    with torch.no_grad():
        base_graph = _functional_graph(
            detached_params, base["x"].to(device), base_adj)[:int(old_rows)]
    similarity = (F.normalize(candidate_graph, dim=1)
                  * F.normalize(base_graph, dim=1)).sum(dim=1)
    if fraction == 0.0:
        return candidate_graph.sum() * 0.0
    count = max(1, int(round(float(old_rows) * fraction)))
    # Anchor the most changed old rows.  This keeps the constraint focused on
    # propagation damage instead of freezing every historical representation.
    displacement = (1.0 - similarity.detach()).clamp_min(0.0)
    selected = torch.topk(displacement, k=min(count, len(displacement))).indices
    return (1.0 - similarity[selected]).mean()


def class_representation_loss(params: dict, candidate: dict,
                              candidate_adj: torch.Tensor,
                              support: dict, support_adj: torch.Tensor,
                              selected: torch.Tensor, split: str,
                              plan: dict, support_class_id: torch.Tensor,
                              base_slot_class_id: torch.Tensor) -> dict:
    """Align mass-weighted synthetic class responses to current-delta responses.

    The raw support response is a detached target. Only the candidate-side
    graph representation contributes gradients to the editable synthetic X.
    Train and validation targets use disjoint rows from the sealed plan.
    """
    device = params["g1.lin.weight"].device
    selected = selected.to(device).long()
    slot_class = base_slot_class_id.to(device).long()[selected]
    raw_class = support_class_id.to(device).long()
    if bool((slot_class < 0).any()):
        raise ValueError("class representation requires class ids for selected slots")
    if len(raw_class) != len(support["x"]):
        raise ValueError("support class ids must align with support rows")
    target_batches = plan["splits"][split].get("new", [])
    target_rows = sorted({
        int(row)
        for raw_batch in target_batches
        for row in (
            _batch_targets_context(raw_batch)[0].tolist()
            if hasattr(_batch_targets_context(raw_batch)[0], "tolist")
            else list(_batch_targets_context(raw_batch)[0])
        )
    })
    if not target_rows:
        raise ValueError(f"class representation has no {split} raw rows")
    target_rows = torch.tensor(target_rows, dtype=torch.long, device=device)

    candidate_graph = _functional_graph(
        params, candidate["x"].to(device), candidate_adj)
    detached_params = {key: value.detach() for key, value in params.items()}
    with torch.no_grad():
        support_graph = _functional_graph(
            detached_params, support["x"].to(device), support_adj)

    losses = []
    for class_id in torch.unique(slot_class, sorted=True):
        slot_rows = selected[slot_class == class_id]
        raw_rows = target_rows[raw_class[target_rows] == class_id]
        if not len(raw_rows):
            # A sealed split may contain no row for a small incoming class.
            continue
        mass = candidate.get("mass")
        if mass is None:
            weights = torch.ones(len(slot_rows), device=device)
        else:
            weights = mass.to(device).float()[slot_rows].clamp_min(0)
            if float(weights.sum()) <= 0:
                weights = torch.ones_like(weights)
        synthetic_center = (
            candidate_graph[slot_rows] * weights[:, None]
        ).sum(0) / weights.sum().clamp_min(1e-8)
        raw_center = support_graph[raw_rows].mean(0)
        synthetic_center = F.normalize(synthetic_center, dim=0)
        raw_center = F.normalize(raw_center, dim=0)
        losses.append(1.0 - (synthetic_center * raw_center.detach()).sum())
    if not losses:
        raise ValueError(f"class representation has no usable {split} classes")
    return {"class_representation": torch.stack(losses).mean(),
            "class_representation_classes": int(len(losses)),
            "class_representation_skipped_classes": [
                int(class_id) for class_id in torch.unique(slot_class, sorted=True)
                if not bool((raw_class[target_rows] == class_id).any())],
            "class_representation_raw_rows": int(len(target_rows))}


def slot_representation_loss(params: dict, candidate: dict,
                             candidate_adj: torch.Tensor, support: dict,
                             support_adj: torch.Tensor, old_rows: int,
                             selected: torch.Tensor, split: str,
                             plan: dict, support_slot_id: torch.Tensor,
                             base_slot_group_id: torch.Tensor,
                             group_filter: int | None) -> dict:
    """Align one synthetic slot with its own current-delta response.

    ``support_slot_id`` is a sealed, capacity-conserving provenance map.  The
    raw response is detached so the outer variable receives gradients only
    through the candidate synthetic row.  Unlike the class-level objective,
    this keeps distinct modes inside one class separate.
    """
    device = params["g1.lin.weight"].device
    selected = selected.to(device).long()
    groups = base_slot_group_id.to(device).long()[selected]
    support_slot_id = support_slot_id.to(device).long()
    if len(support_slot_id) != len(support["x"]):
        raise ValueError("support slot ids must align to support rows")
    candidate_groups = sorted(set(int(value) for value in groups.detach().cpu().tolist()))
    if group_filter is not None:
        candidate_groups = [int(group_filter)]
    candidate_graph = _functional_graph(
        params, candidate["x"].to(device), candidate_adj)
    detached_params = {key: value.detach() for key, value in params.items()}
    with torch.no_grad():
        support_graph = _functional_graph(
            detached_params, support["x"].to(device), support_adj)
    values = []
    skipped_groups = []
    raw_count = 0
    for current_group in candidate_groups:
        positions = torch.where(groups == int(current_group))[0]
        if len(positions) != 1:
            raise ValueError("slot representation expects one candidate row per slot")
        candidate_row = selected[positions[0]]
        target_rows = []
        for batch in plan["splits"][split].get("new", []):
            rows, _ = _batch_targets_context(batch)
            row_values = rows.tolist() if hasattr(rows, "tolist") else rows
            target_rows.extend(int(row) for row in row_values)
        target_rows = sorted(set(target_rows))
        target_rows = [row for row in target_rows
                       if int(support_slot_id[row]) == int(current_group)
                       and row >= int(old_rows)]
        if not target_rows:
            # A small slot can be assigned entirely to the sealed train
            # split.  The global validation objective must remain defined for
            # the other slots; do not turn an absent validation target into a
            # construction failure.
            skipped_groups.append(int(current_group))
            continue
        target_rows = torch.tensor(target_rows, dtype=torch.long, device=device)
        with torch.no_grad():
            target = F.normalize(support_graph[target_rows].mean(0), dim=0)
        values.append(1.0 - (F.normalize(candidate_graph[candidate_row], dim=0)
                             * target).sum())
        raw_count += int(len(target_rows))
    if not values:
        # Keep a differentiable zero for callers that request a split with no
        # held-out rows at all.  The audit records the absence explicitly.
        value = candidate_graph[selected].sum() * 0.0
    else:
        value = torch.stack(values).mean()
    return {"slot_representation": value,
            "slot_representation_group": (int(group_filter)
                                           if group_filter is not None else None),
            "slot_representation_groups": int(len(values)),
            "slot_representation_raw_rows": int(raw_count),
            "slot_representation_skipped_groups": skipped_groups}


def _undirected_pairs(edge_index: torch.Tensor, n: int) -> list[tuple[int, int]]:
    pairs = set()
    for left, right in edge_index.detach().cpu().T.tolist():
        left, right = int(left), int(right)
        if left == right:
            continue
        pairs.add((min(left, right), max(left, right)))
    return sorted(pairs)


def _weighted_norm_adj(src: torch.Tensor, dst: torch.Tensor,
                       values: torch.Tensor, n: int) -> torch.Tensor:
    self_ids = torch.arange(n, device=values.device)
    all_src = torch.cat((src, self_ids))
    all_dst = torch.cat((dst, self_ids))
    all_values = torch.cat((values, torch.ones(n, device=values.device,
                                               dtype=values.dtype)))
    degree = torch.zeros(n, device=values.device, dtype=values.dtype)
    degree.scatter_add_(0, all_src, all_values)
    inv = degree.clamp_min(1e-9).pow(-0.5)
    normalized = all_values * inv[all_src] * inv[all_dst]
    return torch.sparse_coo_tensor(
        torch.stack((all_src, all_dst)), normalized, (n, n)).coalesce()


def fixed_norm_adj(edge_index: torch.Tensor, n: int, device: str) -> torch.Tensor:
    return G.norm_adj(edge_index.long(), n, device)


def _init_p_logits(base: dict, support: dict, old_rows: int,
                   cfg: Config, device: str) -> torch.Tensor:
    new_n = len(base["x"]) - old_rows
    source_x = support["x"][old_rows:].to(device)
    source_t = support["text_emb"][old_rows:].to(device)
    if new_n < 1 or len(source_x) < 2:
        raise ValueError("P requires new candidate rows and current-delta rows")
    target_x = base["x"][old_rows:].to(device)
    target_t = base["text_emb"][old_rows:].to(device)
    sim = F.normalize(target_x.float(), dim=1) @ F.normalize(source_x.float(), dim=1).T
    sim = sim + F.normalize(target_t.float(), dim=1) @ F.normalize(source_t.float(), dim=1).T
    return sim / float(cfg.p_temperature)


def sinkhorn_assignment(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """Balanced nonnegative P with row mass 1 and uniform column mass."""
    rows, cols = logits.shape
    values = (logits - logits.max()).exp().clamp_min(1e-12)
    target_col = float(rows) / float(cols)
    for _ in range(int(iterations)):
        values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        values = values * (target_col / values.sum(dim=0, keepdim=True).clamp_min(1e-12))
    values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return values


def class_constrained_assignment(logits: torch.Tensor,
                                 target_class_id: torch.Tensor,
                                 source_class_id: torch.Tensor,
                                 iterations: int) -> torch.Tensor:
    """Run balanced P independently inside each construction-time class.

    ``logits`` has one row per synthetic new slot and one column per current
    delta row.  The class ids are construction metadata, not report labels.
    Restricting each block prevents a slot for one incoming class from
    transporting payload from another class while retaining differentiable
    Sinkhorn balancing within the class.
    """
    target_class_id = target_class_id.to(logits.device).long()
    source_class_id = source_class_id.to(logits.device).long()
    if len(target_class_id) != logits.shape[0]:
        raise ValueError("target class ids must align with P rows")
    if len(source_class_id) != logits.shape[1]:
        raise ValueError("source class ids must align with P columns")
    if bool((target_class_id < 0).any()) or bool((source_class_id < 0).any()):
        raise ValueError("class-constrained P requires nonnegative new class ids")
    blocks = []
    for class_id in torch.unique(target_class_id, sorted=True):
        target_rows = torch.where(target_class_id == class_id)[0]
        source_rows = torch.where(source_class_id == class_id)[0]
        if not len(source_rows):
            raise ValueError(f"class {int(class_id)} has no current-delta source rows")
        block = sinkhorn_assignment(
            logits.index_select(0, target_rows).index_select(1, source_rows),
            iterations)
        dense = logits.new_zeros((len(target_rows), logits.shape[1]))
        dense[:, source_rows] = block
        blocks.append((target_rows, dense))
    result = logits.new_zeros(logits.shape)
    for target_rows, dense in blocks:
        result[target_rows] = dense
    return result


def _build_a_pool(base: dict, old_rows: int, cfg: Config,
                  device: str) -> tuple[list[list[int]], list[int]]:
    n = len(base["x"])
    new_ids = list(range(old_rows, n))
    old_ids = list(range(old_rows))
    if not old_ids or not new_ids:
        raise ValueError("A requires old and new candidate rows")
    old_x, new_x = base["x"].to(device), base["x"].to(device)
    old_t, new_t = base["text_emb"].to(device), base["text_emb"].to(device)
    score = F.normalize(new_x[old_rows:].float(), dim=1) @ F.normalize(old_x[:old_rows].float(), dim=1).T
    score = score + F.normalize(new_t[old_rows:].float(), dim=1) @ F.normalize(old_t[:old_rows].float(), dim=1).T
    existing = set(_undirected_pairs(base["edge_index"], n))
    pools, degrees = [], []
    for i, new_id in enumerate(new_ids):
        k = min(max(1, int(cfg.a_pool)), len(old_ids))
        candidate = torch.topk(score[i], k, sorted=True).indices.tolist()
        candidate = sorted(set(int(v) for v in candidate))
        for old_id in old_ids:
            if (min(old_id, new_id), max(old_id, new_id)) in existing:
                if old_id not in candidate:
                    candidate.append(old_id)
        candidate = sorted(candidate)
        current_degree = sum((min(old_id, new_id), max(old_id, new_id)) in existing
                             for old_id in candidate)
        pools.append(candidate)
        # Preserve the original cross-block degree exactly.  A new row that
        # had no cross-block edge must remain edge-free when materialized.
        degrees.append(int(current_degree))
    return pools, degrees


def _init_a_logits(base: dict, old_rows: int, cfg: Config,
                   device: str) -> tuple[torch.Tensor, list[list[int]], list[int]]:
    pools, degrees = _build_a_pool(base, old_rows, cfg, device)
    existing = set(_undirected_pairs(base["edge_index"], len(base["x"])))
    logits = []
    for new_id, candidates in enumerate(pools, start=old_rows):
        logits.append(torch.tensor([
            float(cfg.a_existing_logit)
            if (min(old_id, new_id), max(old_id, new_id)) in existing else 0.0
            for old_id in candidates], device=device))
    max_width = max(len(row) for row in logits)
    padded = torch.full((len(logits), max_width), -20.0, device=device)
    for i, row in enumerate(logits):
        padded[i, :len(row)] = row
    if float(cfg.a_message_weight) > 0:
        padded = _message_aware_a_logits(base, old_rows, pools, padded, cfg, device)
    return padded, pools, degrees


def _a_weights(logits: torch.Tensor, pools: list[list[int]],
               degrees: list[int], temperature: float,
               straight_through: bool,
               existing_masks: list[list[bool]] | None = None,
               max_revisions_per_row: int | None = None,
               revision_margin: float = 0.0
               ) -> tuple[torch.Tensor, list[list[int]]]:
    weights, chosen = [], []
    for i, pool in enumerate(pools):
        width = len(pool)
        degree = min(int(degrees[i]), width)
        if degree <= 0:
            # Preserve the C0 zero cross-degree during the differentiable
            # inner loop as well as during hard materialization.
            soft = logits[i, :width] * 0.0
            hard = soft
            weights.append(hard)
            chosen.append([])
            continue
        values = logits[i, :width]
        if (max_revisions_per_row is not None
                and existing_masks is not None):
            mask = torch.tensor(existing_masks[i], dtype=torch.bool,
                                device=values.device)
            existing = torch.where(mask)[0]
            if len(existing) != degree:
                raise ValueError(
                    "A revision mask must contain exactly the original degree")
            revision = min(int(max_revisions_per_row), degree)
            preserve_count = degree - revision
            preserve = (torch.topk(values[existing], preserve_count,
                                    sorted=True).indices
                        if preserve_count else torch.empty(
                            0, dtype=torch.long, device=values.device))
            preserve = existing[preserve]
            remaining_mask = torch.ones(width, dtype=torch.bool,
                                        device=values.device)
            remaining_mask[preserve] = False
            remaining = torch.where(remaining_mask)[0]
            if revision > len(remaining):
                raise ValueError("A revision cap exceeds candidate pool width")
            if revision:
                # For the common one-cross-edge case, make revision an
                # explicit confidence gate: a new edge must beat the best
                # original edge by ``revision_margin``.  The sigmoid keeps
                # the relaxed operator differentiable, while hard forward
                # uses the same thresholded decision.
                original = torch.where(mask)[0]
                new_candidates = torch.where(~mask)[0]
                if (revision == 1 and degree == 1
                        and revision_margin > 0 and len(new_candidates)):
                    best_original = original[torch.argmax(values[original])]
                    best_new = new_candidates[torch.argmax(values[new_candidates])]
                    gap = values[best_new] - values[best_original]
                    gate = torch.sigmoid(
                        (gap - float(revision_margin)) / float(temperature))
                    soft = torch.zeros_like(values)
                    soft[best_original] = 1.0 - gate
                    new_soft = F.softmax(
                        values[new_candidates] / float(temperature), dim=0)
                    soft[new_candidates] = gate * new_soft
                    top = (torch.stack((best_new,))
                           if bool((gap > float(revision_margin)).detach())
                           else torch.stack((best_original,)))
                else:
                    soft = torch.zeros_like(values)
                    soft[preserve] = 1.0
                    soft[remaining] = float(revision) * F.softmax(
                        values[remaining] / float(temperature), dim=0)
                    revised = torch.topk(values[remaining], revision,
                                         sorted=True).indices
                    top = torch.cat((preserve, remaining[revised]))
            else:
                soft = torch.zeros_like(values)
                soft[preserve] = 1.0
                top = preserve
        else:
            # The relaxed operator must carry the same total cross-edge mass
            # as the eventual binary graph.
            soft = float(degree) * F.softmax(
                values / float(temperature), dim=0)
            top = torch.topk(values, degree, sorted=True).indices
        # The hard forward graph contains ``degree`` binary edges.  Each
        # selected edge therefore carries unit mass.
        hard = torch.zeros_like(soft).scatter(0, top, 1.0)
        value = hard.detach() - soft.detach() + soft if straight_through else soft
        weights.append(value)
        chosen.append(top.detach().cpu().tolist())
    return torch.cat(weights), chosen


def _a_revision_counts(chosen: list[list[int]],
                       existing_masks: list[list[bool]]) -> list[int]:
    """Count selected candidate positions that were not original edges."""
    return [
        sum(not existing_masks[row][position]
            for position in selected)
        for row, selected in enumerate(chosen)
    ]


def _candidate_adjacency(base: dict, old_rows: int, a_logits: torch.Tensor,
                         cfg: Config, device: str,
                         straight_through: bool) -> tuple[torch.Tensor, dict]:
    pools, degrees = _build_a_pool(base, old_rows, cfg, device)
    existing = set(_undirected_pairs(base["edge_index"], len(base["x"])))
    existing_masks = [[
        (min(old_id, new_id), max(old_id, new_id)) in existing
        for old_id in pool]
        for new_id, pool in enumerate(pools, start=old_rows)]
    weights, chosen = _a_weights(a_logits, pools, degrees,
                                 cfg.a_temperature, straight_through,
                                 existing_masks=existing_masks,
                                 max_revisions_per_row=cfg.a_max_revisions_per_row,
                                 revision_margin=cfg.a_revision_margin)
    revisions = _a_revision_counts(chosen, existing_masks)
    pairs = _undirected_pairs(base["edge_index"], len(base["x"]))
    fixed = [(u, v) for u, v in pairs if not (u < old_rows <= v)]
    src, dst, vals = [], [], []
    for u, v in fixed:
        src.extend((u, v)); dst.extend((v, u)); vals.extend((1.0, 1.0))
    cursor = 0
    for new_id, pool in enumerate(pools, start=old_rows):
        row_weights = weights[cursor:cursor + len(pool)]
        cursor += len(pool)
        for old_id, value in zip(pool, row_weights):
            src.extend((old_id, new_id)); dst.extend((new_id, old_id))
            vals.extend((value, value))
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    if vals:
        values = torch.stack([v if isinstance(v, torch.Tensor)
                              else torch.tensor(v, device=device)
                              for v in vals])
    else:
        values = torch.empty(0, device=device)
    adj = _weighted_norm_adj(src_t, dst_t, values, len(base["x"]))
    audit = {"candidate_pools": pools, "degrees": degrees,
             "chosen_positions": chosen,
             "hard_cross_edges": int(sum(len(v) for v in chosen)),
             "hard_revisions_per_row": revisions,
             "hard_revised_cross_edges": int(sum(revisions)),
             "max_hard_revisions_per_row": int(max(revisions, default=0)),
             "max_revisions_per_row": cfg.a_max_revisions_per_row}
    return adj, audit


def _message_aware_a_logits(base: dict, old_rows: int, pools: list[list[int]],
                            logits: torch.Tensor, cfg: Config,
                            device: str) -> torch.Tensor:
    """Add a frozen local message compatibility prior to A candidates.

    The prior uses the current synthetic graph message before the edge swap
    and the old-node payload/text similarity. It is a bounded candidate
    ranking aid; the Student outer loss still supplies the differentiable
    update and all degree/revision constraints remain unchanged.
    """
    n = len(base["x"])
    adj = fixed_norm_adj(base["edge_index"], n, device)
    x = base["x"].to(device).float()
    msg = _matmul_adj(adj, x)
    new_x = F.normalize(msg[old_rows:], dim=1)
    old_x = F.normalize(msg[:old_rows], dim=1)
    values = logits.clone()
    for i, pool in enumerate(pools):
        new_id = old_rows + i
        # Prefer an old endpoint whose message direction is compatible with
        # the new node while penalizing candidates that already dominate its
        # original cross-block message.
        candidate_old = old_x[torch.tensor(pool, device=device)]
        score = (new_x[i:i + 1] @ candidate_old.T)[0]
        values[i, :len(pool)] = values[i, :len(pool)] + float(cfg.a_message_weight) * score
    return values


def _materialize_a(base: dict, old_rows: int, a_logits: torch.Tensor,
                   cfg: Config) -> tuple[torch.Tensor, dict]:
    device = str(a_logits.device)
    pools, degrees = _build_a_pool(base, old_rows, cfg, device)
    existing = set(_undirected_pairs(base["edge_index"], len(base["x"])))
    existing_masks = [[
        (min(old_id, new_id), max(old_id, new_id)) in existing
        for old_id in pool]
        for new_id, pool in enumerate(pools, start=old_rows)]
    _, chosen = _a_weights(a_logits, pools, degrees,
                           cfg.a_temperature, straight_through=True,
                           existing_masks=existing_masks,
                           max_revisions_per_row=cfg.a_max_revisions_per_row,
                           revision_margin=cfg.a_revision_margin)
    revisions = _a_revision_counts(chosen, existing_masks)
    pairs = _undirected_pairs(base["edge_index"], len(base["x"]))
    fixed = {(u, v) for u, v in pairs if not (u < old_rows <= v)}
    for new_id, pool in enumerate(pools, start=old_rows):
        for position in chosen[new_id - old_rows]:
            old_id = pool[position]
            fixed.add((min(old_id, new_id), max(old_id, new_id)))
    directed = []
    for u, v in sorted(fixed):
        directed.extend(((u, v), (v, u)))
    edge = (torch.tensor(directed, dtype=torch.long).T
            if directed else torch.empty((2, 0), dtype=torch.long))
    return edge, {"hard_cross_edges": int(sum(len(v) for v in chosen)),
                  "hard_revisions_per_row": revisions,
                  "hard_revised_cross_edges": int(sum(revisions)),
                  "max_hard_revisions_per_row": int(max(revisions, default=0)),
                  "max_revisions_per_row": cfg.a_max_revisions_per_row,
                  "cross_pairs": [[pool[pos] for pos in chosen[i]]
                                  for i, pool in enumerate(pools)]}


def _candidate(base: dict, support: dict, old_rows: int, variables: dict,
               mode: str, selected: torch.Tensor, cfg: Config,
               device: str, materialize: bool = False,
               base_slot_class_id: torch.Tensor | None = None,
               base_slot_group_id: torch.Tensor | None = None,
               support_class_id: torch.Tensor | None = None,
               gradient_class: int | None = None,
               gradient_group: int | None = None,
               response_gate: dict | None = None) -> tuple[dict, dict]:
    base_x = base["x"].to(device)
    base_text = base["text_emb"].to(device)
    candidate_x = base_x
    candidate_text = base_text
    audit = {}
    if "p_logits" in variables:
        if cfg.p_class_constrained:
            if base_slot_class_id is None or support_class_id is None:
                raise ValueError(
                    "class-constrained P requires candidate and support class ids")
            target_classes = base_slot_class_id.to(device).long()[old_rows:]
            source_classes = support_class_id.to(device).long()[old_rows:]
            assignment = class_constrained_assignment(
                variables["p_logits"], target_classes, source_classes,
                cfg.p_sinkhorn_iters)
        else:
            assignment = sinkhorn_assignment(
                variables["p_logits"], cfg.p_sinkhorn_iters)
        audit["p_class_constrained"] = bool(cfg.p_class_constrained)
        if cfg.p_class_constrained:
            target_classes = base_slot_class_id.to(device).long()[old_rows:]
            source_classes = support_class_id.to(device).long()[old_rows:]
            cross_class = target_classes[:, None] != source_classes[None, :]
            cross_mass = assignment.masked_fill(~cross_class, 0.0).sum(dim=1)
            audit["p_cross_class_mass_max"] = float(cross_mass.max().detach())
            audit["p_cross_class_mass_total"] = float(cross_mass.sum().detach())
        source_x = support["x"][old_rows:].to(device)
        source_text = support["text_emb"][old_rows:].to(device)
        if float(cfg.p_old_context_ratio) > 0:
            old_x = base["x"][:old_rows].to(device)
            old_text = base["text_emb"][:old_rows].to(device)
            ratio = min(float(cfg.p_old_context_ratio),
                        float(cfg.p_old_context_cap))
            old_logits = F.normalize(base["x"][old_rows:].float().to(device), dim=1) @ F.normalize(old_x.float(), dim=1).T
            old_logits = old_logits + F.normalize(base["text_emb"][old_rows:].float().to(device), dim=1) @ F.normalize(old_text.float(), dim=1).T
            old_assignment = sinkhorn_assignment(old_logits / float(cfg.p_temperature), cfg.p_sinkhorn_iters)
            transported_x = ((1.0 - ratio) * (assignment @ source_x)
                             + ratio * (old_assignment @ old_x))
            transported_text = ((1.0 - ratio) * (assignment @ source_text)
                                + ratio * (old_assignment @ old_text))
            audit_ratio = ratio
            audit["p_old_context_assignment"] = old_assignment
        else:
            transported_x = assignment @ source_x
            transported_text = assignment @ source_text
            audit_ratio = 0.0
        if response_gate is not None and audit_ratio > 0:
            new_transport_x = assignment @ source_x
            new_transport_text = assignment @ source_text
            row_values = response_gate.get("keep_old_context_rows")
            if row_values is not None:
                row_gate = torch.as_tensor(
                    row_values, dtype=transported_x.dtype, device=device)
                if len(row_gate) != len(transported_x):
                    raise ValueError(
                        "per-row P gate must align with candidate new rows")
                row_gate = row_gate.clamp(0.0, 1.0).unsqueeze(1)
                old_transport_x = old_assignment @ old_x
                old_transport_text = old_assignment @ old_text
                transported_x = new_transport_x + row_gate * float(ratio) * (
                    old_transport_x - new_transport_x)
                transported_text = new_transport_text + row_gate * float(ratio) * (
                    old_transport_text - new_transport_text)
                audit["p_response_gate_rows"] = row_gate[:, 0].detach().cpu().tolist()
                audit["p_response_gate"] = float(row_gate.mean())
                audit["p_old_context_row_ratio"] = (
                    row_gate[:, 0] * float(ratio)).detach().cpu().tolist()
                audit_ratio = float(row_gate.mean()) * float(ratio)
            else:
                gate_value = float(response_gate.get("keep_old_context", 1.0))
                if gate_value < 0.5:
                    transported_x = new_transport_x
                    transported_text = new_transport_text
                    audit_ratio = 0.0
                audit["p_response_gate"] = gate_value
        elif cfg.p_response_gate and audit_ratio > 0:
            # Gate is applied after the bounded mixture using a deterministic
            # local compatibility test.  It never reads report/test metrics.
            target = F.normalize(base["x"][old_rows:].float().to(device), dim=1)
            mixed_score = (target * F.normalize(transported_x, dim=1)).sum(1)
            new_only = assignment @ source_x
            new_score = (target * F.normalize(new_only, dim=1)).sum(1)
            old_score = (target * F.normalize(old_assignment @ old_x, dim=1)).sum(1)
            gain = mixed_score - new_score
            gate = (gain >= float(cfg.p_gate_margin)).float().mean()
            if float(gate) < 0.5:
                transported_x = assignment @ source_x
                transported_text = assignment @ source_text
                audit_ratio = 0.0
            audit["p_response_gate"] = float(gate)
        mix = float(cfg.p_mix)
        candidate_x = base_x.clone()
        candidate_text = base_text.clone()
        candidate_x[old_rows:] = ((1 - mix) * candidate_x[old_rows:]
                                  + mix * transported_x)
        candidate_text[old_rows:] = ((1 - mix) * candidate_text[old_rows:]
                                     + mix * transported_text)
        audit["p_assignment"] = assignment
        audit["p_old_context_ratio"] = audit_ratio
    variable_name = _x_variable_name(mode)
    if variable_name in variables:
        scale = base_x[selected].norm(dim=1, keepdim=True).clamp_min(1e-6)
        delta = scale * variables[variable_name]
        if variable_name == "x_class_relative" and gradient_class is not None:
            if base_slot_class_id is None:
                raise ValueError("class-aware candidate requires slot class ids")
            selected_classes = base_slot_class_id.to(device)[selected].long()
            delta = delta * (selected_classes == int(gradient_class)).unsqueeze(1)
        if (variable_name in ("x_provenance_relative", "x_slot_rep_relative")
                and gradient_group is not None):
            if base_slot_group_id is None:
                raise ValueError(
                    "provenance-aware candidate requires slot group ids")
            selected_groups = base_slot_group_id.to(device)[selected].long()
            delta = delta * (selected_groups == int(gradient_group)).unsqueeze(1)
        full = torch.zeros_like(candidate_x)
        full = full.index_add(0, selected, delta)
        candidate_x = candidate_x + full
        audit["x_delta"] = delta
    candidate_adj = fixed_norm_adj(base["edge_index"], len(base_x), device)
    if "a_logits" in variables:
        candidate_adj, a_audit = _candidate_adjacency(
            base, old_rows, variables["a_logits"], cfg, device,
            straight_through=not materialize)
        audit.update(a_audit)
    candidate = clone_memory(base)
    candidate["x"] = candidate_x
    candidate["text_emb"] = candidate_text
    if materialize:
        candidate["x"] = candidate_x.detach().cpu()
        candidate["text_emb"] = candidate_text.detach().cpu()
        if "a_logits" in variables:
            edge, edge_audit = _materialize_a(base, old_rows,
                                              variables["a_logits"], cfg)
            candidate["edge_index"] = edge.cpu()
            audit.update(edge_audit)
    return candidate, {**audit, "adjacency": candidate_adj}


def _pair_edge_index(pairs: set[tuple[int, int]], device: str) -> torch.Tensor:
    """Materialize a sorted undirected pair set as a symmetric edge index."""
    directed = []
    for left, right in sorted(pairs):
        directed.extend(((left, right), (right, left)))
    if not directed:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(directed, dtype=torch.long, device=device).T.contiguous()


def _swap_cross_pair(pairs: set[tuple[int, int]], remove_old: int,
                     new_id: int, add_old: int) -> set[tuple[int, int]]:
    """Return one bounded cross-edge swap without changing edge count."""
    removed = (min(int(remove_old), int(new_id)),
               max(int(remove_old), int(new_id)))
    added = (min(int(add_old), int(new_id)),
             max(int(add_old), int(new_id)))
    if removed not in pairs:
        raise ValueError("cannot remove a non-existing cross edge")
    if added in pairs:
        raise ValueError("bounded swap would duplicate an existing edge")
    result = set(pairs)
    result.remove(removed)
    result.add(added)
    return result


def marginal_a_revision(
        candidate: dict, support: dict, old_rows: int, cfg: Config,
        plan: dict, support_adj: torch.Tensor, device: str,
        support_class_id: torch.Tensor | None = None,
        prompt_emb: torch.Tensor | None = None,
        prompt_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Apply a bounded, label-free Student-response edge-swap pass.

    Each new synthetic row is allowed at most one replacement of one original
    cross-block edge.  Candidates come from the small semantic pool already
    used by A.  A swap is accepted only when a short, paired Student probe
    improves the new response and does not exceed the old-response tolerance.
    The probe is used only for construction; report/test labels never enter it.
    """
    n = len(candidate["x"])
    pairs = set(_undirected_pairs(candidate["edge_index"], n))
    audit = {
        "a_marginal": True,
        "a_marginal_inner_steps": int(cfg.a_marginal_inner_steps),
        "a_marginal_validation_seed": (
            int(cfg.validation_seeds[0]) if cfg.validation_seeds else None),
        "a_marginal_candidate_count": 0,
        "a_marginal_accepted_count": 0,
        "a_marginal_accepted_rows": [],
        "a_marginal_new_margin": float(cfg.a_marginal_new_margin),
        "a_marginal_old_tolerance": float(cfg.a_marginal_old_tolerance),
    }
    original_cross = [
        int(sum((min(old_id, new_id), max(old_id, new_id)) in pairs
                for old_id in range(int(old_rows))))
        for new_id in range(int(old_rows), n)
    ]
    audit["a_marginal_original_edge_count"] = int(len(pairs))
    audit["a_marginal_original_cross_degrees"] = original_cross
    if (cfg.a_max_revisions_per_row is None
            or int(cfg.a_max_revisions_per_row) <= 0):
        audit["a_marginal_final_new_response"] = None
        audit["a_marginal_final_old_response"] = None
        audit["a_marginal_edge_count"] = int(len(pairs))
        audit["a_marginal_final_cross_degrees"] = list(original_cross)
        audit["a_marginal_edge_count_conserved"] = True
        return candidate["edge_index"].detach().cpu(), audit

    seed = int(cfg.validation_seeds[0])
    probe_cfg = replace(
        cfg, inner_steps=min(int(cfg.inner_steps), int(cfg.a_marginal_inner_steps)),
        p_response_gate=False)

    def response(current_pairs: set[tuple[int, int]]) -> tuple[float, float]:
        # ``G.norm_adj`` appends CPU self-loops before moving the result to
        # ``device``; keep temporary edge indices on CPU to match that API.
        edge_index = _pair_edge_index(current_pairs, "cpu")
        adjacency = fixed_norm_adj(edge_index, n, device)
        unrolled = unroll_student(
            candidate["x"].to(device), candidate["text_emb"].to(device),
            adjacency, seed, probe_cfg, False)
        new_loss, old_loss = _response_pair(
            unrolled["params"], support, support_adj, plan, probe_cfg,
            support_class_id=support_class_id, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids)
        return float(new_loss.detach()), float(old_loss.detach())

    baseline_new, baseline_old = response(pairs)
    pools, degrees = _build_a_pool(candidate, old_rows, cfg, device)
    for offset, pool in enumerate(pools):
        new_id = int(old_rows + offset)
        neighbors = sorted(
            old_id for old_id in range(int(old_rows))
            if (min(old_id, new_id), max(old_id, new_id)) in pairs)
        if not neighbors or int(degrees[offset]) <= 0:
            continue
        alternatives = [int(old_id) for old_id in pool if int(old_id) not in neighbors]
        best = None
        for remove_old in neighbors:
            for add_old in alternatives:
                audit["a_marginal_candidate_count"] += 1
                trial_pairs = _swap_cross_pair(
                    pairs, remove_old, new_id, add_old)
                trial_new, trial_old = response(trial_pairs)
                delta_new = trial_new - baseline_new
                delta_old = trial_old - baseline_old
                if (delta_new < -float(cfg.a_marginal_new_margin)
                        and delta_old <= float(cfg.a_marginal_old_tolerance)
                        and (best is None or delta_new < best["delta_new"])):
                    best = {
                        "pairs": trial_pairs, "remove_old": int(remove_old),
                        "add_old": int(add_old), "new": trial_new,
                        "old": trial_old, "delta_new": delta_new,
                        "delta_old": delta_old,
                    }
        if best is not None:
            pairs = best["pairs"]
            baseline_new, baseline_old = best["new"], best["old"]
            audit["a_marginal_accepted_count"] += 1
            audit["a_marginal_accepted_rows"].append({
                "new_row": new_id,
                "removed_old_row": best["remove_old"],
                "added_old_row": best["add_old"],
                "new_delta": best["delta_new"],
                "old_delta": best["delta_old"],
            })
    audit["a_marginal_final_new_response"] = float(baseline_new)
    audit["a_marginal_final_old_response"] = float(baseline_old)
    audit["a_marginal_edge_count"] = int(len(pairs))
    audit["a_marginal_final_cross_degrees"] = [
        int(sum((min(old_id, new_id), max(old_id, new_id)) in pairs
                for old_id in range(int(old_rows))))
        for new_id in range(int(old_rows), n)
    ]
    audit["a_marginal_edge_count_conserved"] = (
        int(len(pairs)) == int(audit["a_marginal_original_edge_count"]))
    return _pair_edge_index(pairs, "cpu"), audit


def _project_x(relative: torch.Tensor, base_rows: torch.Tensor,
               selected: torch.Tensor, old_rows: int, cfg: Config) -> None:
    with torch.no_grad():
        scales = base_rows.norm(dim=1, keepdim=True).clamp_min(1e-6)
        limits = torch.where(
            (selected < int(old_rows)).unsqueeze(1),
            torch.full_like(scales, float(cfg.x_old_radius)),
            torch.full_like(scales, float(cfg.x_new_radius)))
        norms = relative.norm(dim=1, keepdim=True).clamp_min(1e-12)
        relative.mul_((limits / norms).clamp(max=1.0))


def _make_variables(base: dict, support: dict, old_rows: int,
                    selected: list[int], mode: str, cfg: Config,
                    device: str,
                    base_slot_class_id: torch.Tensor | None = None,
                    base_slot_group_id: torch.Tensor | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    variables = {}
    if _has_component(mode, "x"):
        if not selected:
            raise ValueError("X mode requires editable rows")
        variable_name = _x_variable_name(mode)
        variables[variable_name] = torch.nn.Parameter(torch.zeros(
            (len(selected), base["x"].shape[1]), device=device))
        if _x_mode(mode) == "x-class":
            if base_slot_class_id is None or len(base_slot_class_id) != len(base["x"]):
                raise ValueError("x-class requires base slot class ids")
            selected_classes = base_slot_class_id.to(device)[torch.tensor(
                selected, dtype=torch.long, device=device)].long()
            if bool((selected_classes < 0).any()):
                raise ValueError("x-class editable rows need class ids")
            if any(int(row) < int(old_rows) for row in selected):
                raise ValueError("x-class only edits new synthetic rows")
        if mode in ("x-provenance", "x-slot-rep"):
            if (base_slot_group_id is None
                    or len(base_slot_group_id) != len(base["x"])):
                raise ValueError(
                    "provenance slot modes require base slot group ids")
            selected_groups = base_slot_group_id.to(device)[torch.tensor(
                selected, dtype=torch.long, device=device)].long()
            if bool((selected_groups < 0).any()):
                raise ValueError(
                    "provenance slot modes need slot group ids")
            if any(int(row) < int(old_rows) for row in selected):
                raise ValueError(
                    "provenance slot modes only edit new synthetic rows")
        if mode == "x-geometry":
            if base_slot_class_id is None or len(base_slot_class_id) != len(base["x"]):
                raise ValueError("x-geometry requires base slot class ids")
            selected_classes = base_slot_class_id.to(device)[torch.tensor(
                selected, dtype=torch.long, device=device)].long()
            if bool((selected_classes < 0).any()):
                raise ValueError("x-geometry editable rows need class ids")
            if any(int(row) < int(old_rows) for row in selected):
                raise ValueError("x-geometry only edits new synthetic rows")
        if _x_mode(mode) == "x-class-rep":
            if base_slot_class_id is None or len(base_slot_class_id) != len(base["x"]):
                raise ValueError("x-class-rep requires base slot class ids")
            selected_classes = base_slot_class_id.to(device)[torch.tensor(
                selected, dtype=torch.long, device=device)].long()
            if bool((selected_classes < 0).any()):
                raise ValueError("x-class-rep editable rows need class ids")
            if any(int(row) < int(old_rows) for row in selected):
                raise ValueError("x-class-rep only edits new synthetic rows")
    if _has_component(mode, "p"):
        variables["p_logits"] = torch.nn.Parameter(
            _init_p_logits(base, support, old_rows, cfg, device))
    if _has_component(mode, "a") and not cfg.a_marginal:
        variables["a_logits"] = torch.nn.Parameter(
            _init_a_logits(base, old_rows, cfg, device)[0])
    return variables


def _regularizers(variables: dict, base: dict, selected: torch.Tensor,
                  old_rows: int, cfg: Config, mode: str = "",
                  base_slot_class_id: torch.Tensor | None = None,
                  base_slot_group_id: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, dict]:
    any_value = next(iter(variables.values()))
    total = any_value.new_zeros(())
    record = {}
    variable_name = next((name for name in (
        "x_relative", "x_class_relative", "x_provenance_relative",
        "x_slot_rep_relative", "x_geometry_relative",
        "x_class_rep_relative")
        if name in variables), None)
    if variable_name is not None:
        rel = variables[variable_name]
        old = selected < int(old_rows)
        new = ~old
        if mode in ("x-geometry", "x-class-rep", "x-slot-rep"):
            slot_ids = (base_slot_group_id if mode == "x-slot-rep"
                        else base_slot_class_id)
            if slot_ids is None:
                raise ValueError("class/slot-aware X regularization requires ids")
            if bool(old.any()):
                raise ValueError("class-aware X cannot edit old rows")
            anchor = rel.square().mean()
        else:
            anchor = (rel[old].square().mean()
                      if bool(old.any()) else rel.new_zeros(()))
        prior = rel[new].square().mean() if bool(new.any()) else rel.new_zeros(())
        if mode in ("x-geometry", "x-class-rep", "x-slot-rep"):
            total = total + float(cfg.geometry_anchor_weight) * anchor
        else:
            total = total + float(cfg.x_anchor_weight) * anchor
        total = total + float(cfg.x_new_prior_weight) * prior
        record.update(x_anchor=float(anchor.detach()),
                      x_new_prior=float(prior.detach()))
        if _x_mode(mode) in ("x-geometry", "x-class-rep"):
            class_ids = slot_ids.to(rel.device).long()[selected]
            if bool((class_ids < 0).any()):
                raise ValueError("geometry-X selected rows need class ids")
            base_rows = base["x"].to(rel.device)[selected]
            scales = base_rows.norm(dim=1, keepdim=True).clamp_min(1e-6)
            current_rows = base_rows + scales * rel
            current_norm = F.normalize(current_rows, dim=1)
            base_norm = F.normalize(base_rows, dim=1)
            smooth_values = []
            for class_id in sorted(set(int(value) for value in
                                       class_ids.detach().cpu().tolist())):
                positions = torch.where(class_ids == int(class_id))[0]
                if len(positions) < 2:
                    continue
                current_sim = current_norm[positions] @ current_norm[positions].T
                base_sim = base_norm[positions] @ base_norm[positions].T
                mask = ~torch.eye(len(positions), dtype=torch.bool,
                                   device=rel.device)
                smooth_values.append((current_sim[mask] - base_sim[mask]).square().mean())
            smoothness = (torch.stack(smooth_values).mean()
                          if smooth_values else rel.new_zeros(()))
            total = total + float(cfg.geometry_smoothness_weight) * smoothness
            if _x_mode(mode) == "x-geometry":
                record["geometry_anchor"] = float(anchor.detach())
                record["geometry_smoothness"] = float(smoothness.detach())
            elif _x_mode(mode) == "x-class-rep":
                record["class_representation_anchor"] = float(anchor.detach())
                record["class_representation_smoothness"] = float(
                    smoothness.detach())
            else:
                record["slot_representation_anchor"] = float(anchor.detach())
                record["slot_representation_smoothness"] = float(
                    smoothness.detach())
    if "p_logits" in variables:
        assignment = sinkhorn_assignment(variables["p_logits"], cfg.p_sinkhorn_iters)
        target = assignment.new_full((assignment.shape[1],),
                                     float(assignment.shape[0]) / assignment.shape[1])
        balance = (assignment.sum(0) - target).square().mean()
        total = total + float(cfg.p_balance_weight) * balance
        entropy = -(assignment.clamp_min(1e-12)
                    * assignment.clamp_min(1e-12).log()).sum(1).mean()
        record.update(p_balance=float(balance.detach()),
                      p_entropy=float(entropy.detach()))
    return total, record


def _objective(base: dict, support: dict, old_rows: int,
               selected: torch.Tensor, variables: dict, mode: str,
               cfg: Config, plan: dict, split: str, seeds: Iterable[int],
               device: str, create_graph: bool,
               support_class_id: torch.Tensor | None = None,
               prompt_emb: torch.Tensor | None = None,
               prompt_ids: torch.Tensor | None = None,
               base_slot_class_id: torch.Tensor | None = None,
               base_slot_group_id: torch.Tensor | None = None,
               support_slot_id: torch.Tensor | None = None,
               gradient_class: int | None = None,
               gradient_group: int | None = None,
               support_class_filter: int | None = None,
               support_group_filter: int | None = None,
               old_reference: float | None = None) -> tuple[torch.Tensor, dict]:
    class_filter = (gradient_class if support_class_filter is None
                    else support_class_filter)
    if mode == "x-class" and class_filter is not None:
        if support_class_id is None:
            raise ValueError("class-aware objective requires support class ids")
        plan = _class_plan(plan, support_class_id, class_filter,
                           cfg.support_batch)
    if mode == "x-provenance" and support_group_filter is not None:
        if support_slot_id is None:
            raise ValueError(
                "provenance-aware objective requires support slot ids")
        plan = _slot_plan(plan, support_slot_id, support_group_filter,
                          cfg.support_batch)
    if mode == "x-slot-rep" and support_group_filter is not None:
        if support_slot_id is None:
            raise ValueError("slot-representation objective requires slot ids")
        plan = _slot_plan(plan, support_slot_id, support_group_filter,
                          cfg.support_batch)
    support_adj = fixed_norm_adj(support["edge_index"], len(support["x"]), device)
    losses, old_values, rows = [], [], []
    geometry_records = []
    for seed in seeds:
        response_gate = None
        response_gate_audit = {}
        if (cfg.p_response_gate and float(cfg.p_old_context_ratio) > 0
                and "p_logits" in variables):
            response_gate, response_gate_audit = _paired_p_response_gate(
                base, support, old_rows, variables, mode, selected, cfg, device,
                int(seed), plan, support_adj,
                support_class_id=support_class_id, prompt_emb=prompt_emb,
                prompt_ids=prompt_ids, base_slot_class_id=base_slot_class_id,
                base_slot_group_id=base_slot_group_id)
        candidate, candidate_info = _candidate(
            base, support, old_rows, variables, mode, selected, cfg, device,
            base_slot_class_id=base_slot_class_id,
            base_slot_group_id=base_slot_group_id,
            support_class_id=support_class_id,
            gradient_class=gradient_class,
            gradient_group=gradient_group,
            response_gate=response_gate)
        unrolled = unroll_student(
            candidate["x"], candidate["text_emb"], candidate_info["adjacency"],
            int(seed), cfg, create_graph)
        if cfg.operator_aligned and mode in (
                "x-provenance", "x-class+p", "x-class+p+a"):
            if mode == "x-provenance":
                if support_slot_id is None or base_slot_group_id is None:
                    raise ValueError(
                        "operator-aligned provenance requires slot metadata")
                support_values = operator_aligned_losses(
                    unrolled["params"], candidate, candidate_info["adjacency"],
                    support, old_rows, cfg,
                    support_slot_id=support_slot_id,
                    base_slot_group_id=base_slot_group_id,
                    base_slot_class_id=base_slot_class_id,
                    prompt_emb=prompt_emb, prompt_ids=prompt_ids,
                    group_filter=support_group_filter)
            else:
                if support_class_id is None or base_slot_class_id is None:
                    raise ValueError(
                        "operator-aligned class mode requires class metadata")
                support_values = operator_aligned_class_losses(
                    unrolled["params"], candidate, candidate_info["adjacency"],
                    support, old_rows, cfg,
                    support_class_id=support_class_id,
                    base_slot_class_id=base_slot_class_id,
                    prompt_emb=prompt_emb, prompt_ids=prompt_ids,
                    group_filter=support_class_filter)
        else:
            support_values = support_losses(
                unrolled["params"], support, support_adj, plan, split, cfg,
                support_class_id=support_class_id, prompt_emb=prompt_emb,
                prompt_ids=prompt_ids, class_filter=class_filter)
        operator_aux_values = {}
        if (float(cfg.operator_aux_weight) > 0
                and mode in ("x-class+p", "x-class+p+a")):
            if support_class_id is None or base_slot_class_id is None:
                raise ValueError(
                    "operator auxiliary loss requires class metadata")
            operator_aux_values = operator_aligned_class_losses(
                unrolled["params"], candidate, candidate_info["adjacency"],
                support, old_rows, cfg,
                support_class_id=support_class_id,
                base_slot_class_id=base_slot_class_id,
                prompt_emb=prompt_emb, prompt_ids=prompt_ids)
            support_values["operator_aux"] = operator_aux_values["total"]
            support_values["total"] = (
                support_values["total"]
                + float(cfg.operator_aux_weight)
                * operator_aux_values["total"])
        geometry_values = {}
        if mode == "x-geometry":
            if base_slot_class_id is None:
                raise ValueError("geometry-X requires base slot classes")
            if float(cfg.geometry_margin_weight) > 0 and prompt_emb is None:
                raise ValueError(
                    "geometry margin requires prompt embeddings")
            geometry_values = geometry_x_losses(
                unrolled["params"], candidate, candidate_info["adjacency"],
                selected, cfg, base_slot_class_id, prompt_emb, prompt_ids)
            support_values["total"] = (
                support_values["total"]
                + float(cfg.geometry_margin_weight)
                * geometry_values["geometry_margin"])
            geometry_records.append(geometry_values)
        if mode in ("x-class-rep", "x-class+p+a"):
            if support_class_id is None or base_slot_class_id is None:
                raise ValueError(
                    "class representation requires support and base slot class ids")
            representation_values = class_representation_loss(
                unrolled["params"], candidate, candidate_info["adjacency"],
                support, support_adj, selected, split, plan,
                support_class_id, base_slot_class_id)
            support_values["total"] = (
                support_values["total"]
                + float(cfg.class_representation_weight)
                * representation_values["class_representation"])
        elif mode == "x-slot-rep":
            if support_slot_id is None or base_slot_group_id is None:
                raise ValueError(
                    "x-slot-rep requires support and candidate slot metadata")
            representation_values = slot_representation_loss(
                unrolled["params"], candidate, candidate_info["adjacency"],
                support, support_adj, old_rows, selected, split, plan,
                support_slot_id, base_slot_group_id,
                group_filter=(gradient_group if gradient_group is not None
                              else support_group_filter))
            support_values["total"] = (
                support_values["total"]
                + float(cfg.slot_representation_weight)
                * representation_values["slot_representation"])
        else:
            representation_values = {}
        candidate_prompt_values = {}
        if (mode == "x-class+p+a"
                and float(cfg.candidate_prompt_margin_weight) > 0):
            if base_slot_class_id is None or prompt_emb is None:
                raise ValueError(
                    "x-class+p+a candidate margin requires slot classes and prompts")
            new_selected = selected[selected >= int(old_rows)]
            candidate_prompt_values = candidate_class_margin_loss(
                unrolled["params"], candidate, candidate_info["adjacency"],
                new_selected, base_slot_class_id, prompt_emb, prompt_ids,
                float(cfg.candidate_prompt_margin))
            support_values["total"] = (
                support_values["total"]
                + float(cfg.candidate_prompt_margin_weight)
                * candidate_prompt_values["candidate_prompt_margin"])
        old_graph_anchor = torch.tensor(float("nan"), device=device)
        if float(cfg.old_graph_anchor_weight) > 0:
            old_graph_anchor = old_graph_anchor_loss(
                unrolled["params"], candidate, candidate_info["adjacency"],
                base, old_rows, cfg.old_graph_anchor_fraction)
            support_values["total"] = (
                support_values["total"]
                + float(cfg.old_graph_anchor_weight) * old_graph_anchor)
        losses.append(support_values["total"])
        if "old" in support_values:
            old_values.append(support_values["old"])
        rows.append({"seed": int(seed),
                     "old_loss": float(support_values.get("old", torch.tensor(float("nan"))).detach()),
                     "new_loss": float(support_values.get("new", torch.tensor(float("nan"))).detach()),
                     "prompt_loss": float(support_values.get(
                         "prompt", torch.tensor(float("nan"))).detach()),
                     "prompt_margin_loss": float(support_values.get(
                         "prompt_margin", torch.tensor(float("nan"))).detach()),
                     "operator_aux_loss": float(support_values.get(
                         "operator_aux", torch.tensor(float("nan"))).detach()),
                     "class_center_loss": float(support_values.get(
                         "class_center", torch.tensor(float("nan"))).detach()),
                     "geometry_margin_loss": float(geometry_values.get(
                         "geometry_margin", torch.tensor(float("nan"))).detach()),
                     "geometry_selected_rows": geometry_values.get(
                         "geometry_selected_rows"),
                     "geometry_class_count": geometry_values.get(
                         "geometry_class_count"),
                     "class_representation_loss": float(
                         representation_values.get(
                             "class_representation",
                             torch.tensor(float("nan"))).detach()),
                     "class_representation_classes": representation_values.get(
                         "class_representation_classes"),
                     "class_representation_skipped_classes": representation_values.get(
                         "class_representation_skipped_classes", []),
                     "slot_representation_loss": float(
                         representation_values.get(
                             "slot_representation",
                             torch.tensor(float("nan"))).detach()),
                     "slot_representation_raw_rows": representation_values.get(
                         "slot_representation_raw_rows"),
                     "slot_representation_skipped_groups": representation_values.get(
                         "slot_representation_skipped_groups", []),
                     "candidate_prompt_margin_loss": float(
                         candidate_prompt_values.get(
                             "candidate_prompt_margin",
                             torch.tensor(float("nan"))).detach()),
                     "candidate_prompt_selected_rows": candidate_prompt_values.get(
                         "candidate_prompt_selected_rows"),
                     "candidate_prompt_class_count": candidate_prompt_values.get(
                         "candidate_prompt_class_count"),
                     "old_graph_anchor_loss": float(old_graph_anchor.detach()),
                     "total_loss": float(support_values["total"].detach()),
                     "inner_loss": float(torch.stack(unrolled["inner_losses"])[-1].detach()),
                     **response_gate_audit,
                     "candidate_audit": {key: value for key, value in candidate_info.items()
                                         if key != "adjacency"}})
    support_total = torch.stack(losses).mean()
    total = support_total
    response_record = {}
    if float(cfg.old_barrier_weight) > 0:
        if old_reference is None:
            raise ValueError("old barrier requires a C0 old-response reference")
        if not old_values:
            raise ValueError("old barrier requires an old support loss")
        old_response = torch.stack(old_values).mean()
        excess = F.relu(
            old_response - (float(old_reference) + float(cfg.old_barrier_margin)))
        old_barrier = excess.square()
        total = total + float(cfg.old_barrier_weight) * old_barrier
        response_record = {
            "old_response": float(old_response.detach()),
            "old_reference": float(old_reference),
            "old_excess": float(excess.detach()),
            "old_barrier": float(old_barrier.detach()),
        }
    if mode == "x-geometry" and geometry_records:
        geometry_margin = torch.stack([
            value["geometry_margin"] for value in geometry_records]).mean()
        geometry_selected_rows = geometry_records[0]["geometry_selected_rows"]
        geometry_class_count = geometry_records[0]["geometry_class_count"]
    else:
        geometry_margin = torch.tensor(float("nan"), device=device)
        geometry_selected_rows = None
        geometry_class_count = None
    regularizer, reg_record = _regularizers(
        variables, base, selected, old_rows, cfg, mode=mode,
        base_slot_class_id=base_slot_class_id,
        base_slot_group_id=base_slot_group_id)
    return total + regularizer, {"seeds": rows,
                                 "support_loss": float(support_total.detach()),
                                 "geometry_margin_loss": float(
                                     geometry_margin.detach()),
                                 "geometry_selected_rows": geometry_selected_rows,
                                 "geometry_class_count": geometry_class_count,
                                 **response_record,
                                 "regularizers": reg_record,
                                 "regularizer": float(regularizer.detach())}


def _baseline(base: dict, support: dict, old_rows: int, cfg: Config,
              plan: dict, seeds: Iterable[int], device: str,
              support_class_id: torch.Tensor | None = None,
              prompt_emb: torch.Tensor | None = None,
              prompt_ids: torch.Tensor | None = None) -> dict:
    support_adj = fixed_norm_adj(support["edge_index"], len(support["x"]), device)
    result = {}
    for split in ("train", "validation"):
        values = []
        for seed in seeds:
            adj = fixed_norm_adj(base["edge_index"], len(base["x"]), device)
            unrolled = unroll_student(base["x"].to(device),
                                      base["text_emb"].to(device), adj,
                                      int(seed), cfg, False)
            losses = support_losses(unrolled["params"], support, support_adj,
                                    plan, split, cfg,
                                    support_class_id=support_class_id,
                                    prompt_emb=prompt_emb,
                                    prompt_ids=prompt_ids)
            values.append({key: float(value.detach()) for key, value in losses.items()})
        result[split] = values
    return result


def _relative_displacement(base: dict, result: dict) -> list[float]:
    values = (result["x"] - base["x"]).norm(dim=1)
    denom = base["x"].norm(dim=1).clamp_min(1e-6)
    return (values / denom).tolist()


def _immutable_check(base: dict, result: dict, mode: str) -> None:
    for key in ("texts", "mass", "depth", "stage"):
        if key in base and key in result:
            left, right = base[key], result[key]
            equal = (torch.equal(left, right) if isinstance(left, torch.Tensor)
                     else left == right)
            if not equal:
                raise AssertionError(f"direct outer changed immutable field {key}")
    if not _has_component(mode, "p") and not torch.equal(base["text_emb"], result["text_emb"]):
        raise AssertionError("text embedding changed outside P mode")
    if not _has_component(mode, "a") and not torch.equal(base["edge_index"], result["edge_index"]):
        raise AssertionError("topology changed outside A mode")
    if (not _has_component(mode, "x") and not _has_component(mode, "p")
            and not torch.equal(base["x"], result["x"])):
        raise AssertionError("X changed outside X mode")
    if len(result["x"]) != len(base["x"]):
        raise AssertionError("direct outer changed memory budget")
    if not torch.isfinite(result["x"]).all() or not torch.isfinite(result["text_emb"]).all():
        raise AssertionError("direct outer produced non-finite payload")


def _response_gate_rows(record: dict) -> list[dict]:
    """Collect gate rows from a possibly class/slot-nested validation record."""
    if not isinstance(record, dict):
        return []
    if "p_response_gate" in record:
        return [record]
    rows = []
    seeds = record.get("seeds")
    if isinstance(seeds, list):
        rows.extend(row for row in seeds
                    if isinstance(row, dict) and "p_response_gate" in row)
    for key, value in record.items():
        if key in {"seeds", "classwise", "slotwise", "global"}:
            continue
        if isinstance(value, dict):
            rows.extend(_response_gate_rows(value))
    for key in ("classwise", "slotwise", "global"):
        value = record.get(key)
        if isinstance(value, dict):
            rows.extend(_response_gate_rows(value))
    return rows


def optimize(base: dict, support: dict, old_rows: int,
             editable_rows: list[int] | torch.Tensor, mode: str = "x",
             device: str = "cpu", cfg: Config = Config(),
             support_class_id: torch.Tensor | None = None,
             prompt_emb: torch.Tensor | None = None,
             prompt_ids: torch.Tensor | None = None,
             base_slot_class_id: torch.Tensor | None = None,
             support_slot_id: torch.Tensor | None = None,
             base_slot_group_id: torch.Tensor | None = None) -> tuple[dict, dict]:
    """Run a direct Student outer optimization and materialize one candidate."""
    validate_memory(base, "base")
    validate_memory(support, "support")
    if old_rows < 2 or old_rows > len(support["x"]) - 2:
        raise ValueError("old_rows must leave at least two support new rows")
    if old_rows > len(base["x"]):
        raise ValueError("old_rows exceeds candidate memory")
    selected = sorted(set(int(value) for value in torch.as_tensor(
        editable_rows, dtype=torch.long).tolist()))
    if selected and (min(selected) < 0 or max(selected) >= len(base["x"])):
        raise ValueError("editable row outside candidate memory")
    if _has_x_component(mode) and not selected:
        raise ValueError("X mode requires at least one editable row")
    base_hash, support_hash = memory_hash(base), memory_hash(support)
    if support_class_id is not None and len(support_class_id) != len(support["x"]):
        raise ValueError("support_class_id must align with support rows")
    if base_slot_class_id is not None and len(base_slot_class_id) != len(base["x"]):
        raise ValueError("base_slot_class_id must align with base rows")
    if support_slot_id is not None and len(support_slot_id) != len(support["x"]):
        raise ValueError("support_slot_id must align with support rows")
    if base_slot_group_id is not None and len(base_slot_group_id) != len(base["x"]):
        raise ValueError("base_slot_group_id must align with base rows")
    if _has_component(mode, "p") and cfg.p_class_constrained:
        if support_class_id is None or base_slot_class_id is None:
            raise ValueError(
                "class-constrained P requires support and base slot class ids")
        base_new_classes = base_slot_class_id[int(old_rows):].long()
        support_new_classes = support_class_id[int(old_rows):].long()
        if bool((base_new_classes < 0).any()) or bool(
                (support_new_classes < 0).any()):
            raise ValueError(
                "class-constrained P requires nonnegative new class ids")
    if mode == "x-class":
        if support_class_id is None or base_slot_class_id is None:
            raise ValueError("x-class requires support and base slot class ids")
        if any(int(row) < int(old_rows) for row in selected):
            raise ValueError("x-class cannot edit old synthetic rows")
    if mode == "x-class+p+a":
        if base_slot_class_id is None:
            raise ValueError("x-class+p+a requires base slot class ids")
        if any(int(row) < int(old_rows) for row in selected):
            raise ValueError("x-class+p+a cannot edit old synthetic rows")
        selected_classes = base_slot_class_id[torch.tensor(
            selected, dtype=torch.long)]
        if bool((selected_classes < 0).any()):
            raise ValueError("x-class+p+a selected rows need class ids")
        if float(cfg.candidate_prompt_margin_weight) > 0 and prompt_emb is None:
            raise ValueError(
                "x-class+p+a candidate margin requires prompt embeddings")
    if mode == "x-geometry":
        if base_slot_class_id is None:
            raise ValueError("x-geometry requires base slot class ids")
        if len(base_slot_class_id) != len(base["x"]):
            raise ValueError("base_slot_class_id must align with base rows")
        if any(int(row) < int(old_rows) for row in selected):
            raise ValueError("x-geometry cannot edit old synthetic rows")
        if bool((base_slot_class_id[torch.tensor(selected, dtype=torch.long)] < 0).any()):
            raise ValueError("x-geometry selected rows need class ids")
    if mode == "x-class-rep":
        if support_class_id is None or base_slot_class_id is None:
            raise ValueError("x-class-rep requires support and base slot class ids")
        if any(int(row) < int(old_rows) for row in selected):
            raise ValueError("x-class-rep cannot edit old synthetic rows")
        if bool((base_slot_class_id[torch.tensor(
                selected, dtype=torch.long)] < 0).any()):
            raise ValueError("x-class-rep selected rows need class ids")
    if mode in ("x-provenance", "x-slot-rep"):
        if support_slot_id is None or base_slot_group_id is None:
            raise ValueError(
                "provenance slot modes require support slot ids and base slot groups")
        support_slot_id = support_slot_id.detach().cpu().long()
        base_slot_group_id = base_slot_group_id.detach().cpu().long()
        if bool((support_slot_id[:int(old_rows)] >= 0).any()):
            raise ValueError("old support rows must have slot id -1")
        if bool((support_slot_id[int(old_rows):] < 0).any()):
            raise ValueError("new support rows need nonnegative slot ids")
        if bool((base_slot_group_id[:int(old_rows)] >= 0).any()):
            raise ValueError("old base rows must have slot group id -1")
        if any(int(row) < int(old_rows) for row in selected):
            raise ValueError("provenance slot modes cannot edit old synthetic rows")
        selected_groups = base_slot_group_id[torch.tensor(
            selected, dtype=torch.long)]
        if bool((selected_groups < 0).any()):
            raise ValueError("selected provenance rows need nonnegative groups")
        support_groups = set(int(value) for value in support_slot_id[old_rows:].tolist())
        base_groups = set(int(value) for value in selected_groups.tolist())
        if not base_groups.issubset(support_groups):
            raise ValueError("every editable slot group needs support rows")
    if float(cfg.prompt_weight) > 0 and (support_class_id is None or prompt_emb is None):
        raise ValueError("prompt_weight requires support_class_id and prompt_emb")
    plan = make_support_plan(
        support, old_rows, cfg.support_batch, cfg.plan_seed,
        support_class_id=support_class_id,
        stratify_new=(
            _x_mode(mode) in ("x-class", "x-class-rep")
            or bool(cfg.p_gate_per_class)
        ),
        support_slot_id=support_slot_id,
        stratify_slot=(mode in ("x-provenance", "x-slot-rep")))
    variables = _make_variables(base, support, old_rows, selected, mode, cfg,
                                device, base_slot_class_id=base_slot_class_id,
                                base_slot_group_id=base_slot_group_id)
    parameters = list(variables.values())
    optimizer = torch.optim.Adam(parameters, lr=float(cfg.outer_lr))
    selected_device = torch.tensor(selected, dtype=torch.long, device=device)
    base_rows = (base["x"].to(device)[selected_device].detach()
                 if selected else None)
    baseline = _baseline(base, support, old_rows, cfg, plan,
                         tuple(cfg.validation_seeds), device,
                         support_class_id=support_class_id,
                         prompt_emb=prompt_emb, prompt_ids=prompt_ids)
    train_baseline = None
    if float(cfg.old_barrier_weight) > 0:
        train_baseline = _baseline(
            base, support, old_rows, cfg, plan, tuple(cfg.train_seeds), device,
            support_class_id=support_class_id, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids)

    def old_reference(records: dict, split: str) -> float | None:
        if float(cfg.old_barrier_weight) <= 0:
            return None
        values = [row["old"] for row in records[split] if "old" in row]
        if not values:
            raise ValueError("C0 baseline has no old support response")
        return float(sum(values) / len(values))

    train_old_reference = (old_reference(train_baseline, "train")
                           if train_baseline is not None else None)
    validation_old_reference = old_reference(baseline, "validation")
    best_value, best_step, best_state = math.inf, None, None
    curve = []
    started = time.perf_counter()
    for step in range(1, int(cfg.outer_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        if mode in ("x-class", "x-provenance", "x-slot-rep"):
            group_source = (base_slot_class_id if mode == "x-class"
                            else base_slot_group_id)
            groups = sorted(set(int(value) for value in group_source.to(device)[
                selected_device].detach().cpu().tolist()))
            group_losses, group_records = [], {}
            for group_id in groups:
                kwargs = {
                    "support_class_id": support_class_id,
                    "prompt_emb": prompt_emb,
                    "prompt_ids": prompt_ids,
                    "base_slot_class_id": base_slot_class_id,
                    "base_slot_group_id": base_slot_group_id,
                    "support_slot_id": support_slot_id,
                }
                if mode == "x-class":
                    kwargs.update(gradient_class=group_id,
                                 support_class_filter=group_id)
                else:
                    kwargs.update(gradient_group=group_id,
                                 support_group_filter=group_id)
                group_loss, group_record = _objective(
                    base, support, old_rows, selected_device, variables, mode,
                    cfg, plan, "train", cfg.train_seeds, device, True,
                    old_reference=train_old_reference,
                    **kwargs)
                group_losses.append(group_loss)
                group_records[str(group_id)] = group_record
            loss = _aggregate_slot_losses(
                group_losses, groups, base, selected_device, group_source,
                bool(cfg.slot_mass_weighted and mode in
                     ("x-provenance", "x-slot-rep")))
            record_key = "classwise" if mode == "x-class" else "slotwise"
            record = {record_key: group_records,
                      "support_loss": float(loss.detach())}
        else:
            loss, record = _objective(
                base, support, old_rows, selected_device, variables, mode,
                cfg, plan, "train", cfg.train_seeds, device, True,
                    old_reference=train_old_reference,
                    support_class_id=support_class_id, prompt_emb=prompt_emb,
                    prompt_ids=prompt_ids, base_slot_class_id=base_slot_class_id,
                    base_slot_group_id=base_slot_group_id,
                    support_slot_id=support_slot_id)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite direct Student objective")
        loss.backward()
        gradient_norms = {}
        for name, parameter in variables.items():
            if (parameter.grad is None
                    and name == "a_logits"
                    and cfg.a_max_revisions_per_row == 0):
                # cap=0 is an explicit frozen-topology control.  Its
                # straight-through operator intentionally removes the A
                # dependence, so a zero gradient is the expected result.
                gradient_norms[name] = 0.0
                continue
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"missing/non-finite outer gradient for {name}")
            gradient_norms[name] = float(parameter.grad.norm().detach())
        optimizer.step()
        if base_rows is not None:
            variable_name = next(name for name in (
                "x_relative", "x_class_relative", "x_provenance_relative",
                    "x_slot_rep_relative", "x_geometry_relative",
                    "x_class_rep_relative")
                if name in variables)
            _project_x(variables[variable_name], base_rows,
                       selected_device, old_rows, cfg)
        if mode == "x-class":
            group_source = (base_slot_class_id if mode == "x-class"
                            else base_slot_group_id)
            group_losses, group_records = [], {}
            for group_id in groups:
                kwargs = {
                    "support_class_id": support_class_id,
                    "prompt_emb": prompt_emb,
                    "prompt_ids": prompt_ids,
                    "base_slot_class_id": base_slot_class_id,
                    "base_slot_group_id": base_slot_group_id,
                    "support_slot_id": support_slot_id,
                }
                if mode == "x-class":
                    kwargs.update(support_class_filter=group_id)
                else:
                    kwargs.update(support_group_filter=group_id)
                group_loss, group_record = _objective(
                    base, support, old_rows, selected_device, variables, mode,
                    cfg, plan, "validation", cfg.validation_seeds, device, False,
                    old_reference=validation_old_reference,
                    **kwargs)
                group_losses.append(group_loss)
                group_records[str(group_id)] = group_record
            val_loss = torch.stack(group_losses).mean()
            val_record = {"classwise": group_records,
                          "support_loss": float(val_loss.detach())}
        elif mode in ("x-provenance", "x-slot-rep"):
            # Tiny provenance slots may have no held-out row.  Select the
            # outer checkpoint using one global sealed validation objective,
            # while training gradients remain slot-routed above.
            val_loss, val_record = _objective(
                base, support, old_rows, selected_device, variables, mode,
                cfg, plan, "validation", cfg.validation_seeds, device, False,
                old_reference=validation_old_reference,
                support_class_id=support_class_id, prompt_emb=prompt_emb,
                prompt_ids=prompt_ids, base_slot_class_id=base_slot_class_id,
                base_slot_group_id=base_slot_group_id,
                support_slot_id=support_slot_id)
            val_record = {"global": val_record,
                          "support_loss": float(val_loss.detach())}
        else:
            val_loss, val_record = _objective(
                base, support, old_rows, selected_device, variables, mode,
                    cfg, plan, "validation", cfg.validation_seeds, device, False,
                    old_reference=validation_old_reference,
                    support_class_id=support_class_id, prompt_emb=prompt_emb,
                    prompt_ids=prompt_ids, base_slot_class_id=base_slot_class_id,
                    base_slot_group_id=base_slot_group_id,
                    support_slot_id=support_slot_id)
        score = float(val_loss.detach())
        record = {"step": step, "train_objective": float(loss.detach()),
                  "gradient_norms": gradient_norms, "train": record,
                  "validation_objective": score, "validation": val_record}
        curve.append(record)
        if score < best_value:
            best_value = score
            best_step = step
            best_state = {key: value.detach().clone()
                          for key, value in variables.items()}
    if best_state is None:
        raise AssertionError("direct Student outer produced no validation checkpoint")
    for key, value in best_state.items():
        variables[key].data.copy_(value)

    material_response_gate = None
    response_gate_summary = {}
    if (cfg.p_response_gate and float(cfg.p_old_context_ratio) > 0
            and "p_logits" in variables):
        best_validation = curve[int(best_step) - 1]["validation"]
        gate_rows = _response_gate_rows(best_validation)
        if not gate_rows:
            raise AssertionError(
                "response gate was enabled but no validation gate decision was recorded")
        if cfg.p_gate_per_class and all(
                "p_response_gate_rows" in row for row in gate_rows):
            row_lists = [list(row["p_response_gate_rows"])
                         for row in gate_rows]
            row_count = len(row_lists[0])
            if any(len(values) != row_count for values in row_lists):
                raise AssertionError(
                    "per-class P gate decisions have inconsistent row counts")
            row_keep = [
                float(sum(float(values[index]) for values in row_lists)
                      / len(row_lists)) >= 0.5
                for index in range(row_count)]
            material_response_gate = {
                "keep_old_context_rows": row_keep}
            response_gate_summary = {
                "source": "best_validation_checkpoint",
                "per_class": True,
                "seed_count": int(len(gate_rows)),
                "row_count": int(row_count),
                "keep_count": int(sum(row_keep)),
                "keep_rate": float(sum(row_keep) / max(1, row_count)),
                "keep_old_context_rows": row_keep,
                "new_delta_mean": float(sum(
                    float(row["p_response_new_delta"]) for row in gate_rows)
                    / len(gate_rows)),
                "old_delta_mean": float(sum(
                    float(row["p_response_old_delta"]) for row in gate_rows)
                    / len(gate_rows)),
            }
        else:
            keep_values = [float(row["p_response_gate"]) for row in gate_rows]
            keep = sum(keep_values) >= 0.5 * len(keep_values)
            material_response_gate = {"keep_old_context": float(keep)}
            response_gate_summary = {
                "source": "best_validation_checkpoint",
                "per_class": False,
                "seed_count": int(len(gate_rows)),
                "keep_count": int(sum(keep_values)),
                "keep_rate": float(sum(keep_values) / len(keep_values)),
                "keep_old_context": float(keep),
                "new_delta_mean": float(sum(
                    float(row["p_response_new_delta"]) for row in gate_rows)
                    / len(gate_rows)),
                "old_delta_mean": float(sum(
                    float(row["p_response_old_delta"]) for row in gate_rows)
                    / len(gate_rows)),
            }
    result, material_audit = _candidate(
        base, support, old_rows, variables, mode, selected_device, cfg, device,
        materialize=True, base_slot_class_id=base_slot_class_id,
        base_slot_group_id=base_slot_group_id,
        support_class_id=support_class_id,
        response_gate=material_response_gate)
    if cfg.a_marginal and _has_component(mode, "a"):
        support_adj = fixed_norm_adj(
            support["edge_index"], len(support["x"]), device)
        revised_edge_index, marginal_audit = marginal_a_revision(
            result, support, old_rows, cfg, plan, support_adj, device,
            support_class_id=support_class_id, prompt_emb=prompt_emb,
            prompt_ids=prompt_ids)
        result["edge_index"] = revised_edge_index
        material_audit.update(marginal_audit)
    _immutable_check(base, result, mode)
    if memory_hash(base) != base_hash or memory_hash(support) != support_hash:
        raise AssertionError("direct outer mutated input memory")
    displacement = _relative_displacement(base, result)
    audit = {
        "version": "direct-student-outer-v2",
        "mode": mode,
        "config": cfg.dict(),
        "old_rows": int(old_rows),
        "editable_rows": selected,
        "input_base_sha256": base_hash,
        "input_support_sha256": support_hash,
        "output_sha256": memory_hash(result),
        "selected_outer_step": int(best_step),
        "selected_validation_objective": float(best_value),
        "baseline_validation": baseline["validation"],
        "baseline_train": (train_baseline["train"]
                            if train_baseline is not None else None),
        "old_barrier": {
            "weight": float(cfg.old_barrier_weight),
            "margin": float(cfg.old_barrier_margin),
            "train_reference": train_old_reference,
            "validation_reference": validation_old_reference,
        },
        "curve": curve,
        "support_plan": plan,
        "relative_displacement": displacement,
        "max_relative_displacement": float(max(displacement)),
        "mean_relative_displacement": float(sum(displacement) / len(displacement)),
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "inner_student_steps": int(cfg.inner_steps),
        "outer_gradient_variables": list(variables),
        "class_aware": bool(_x_mode(mode) == "x-class"
                             or mode in ("x-class+p", "x-class+p+a")),
        "operator_aligned_aware": bool(
            cfg.operator_aligned and mode in (
                "x-provenance", "x-class+p", "x-class+p+a")),
        "operator_aligned_classwise": bool(
            cfg.operator_aligned and mode in ("x-class+p", "x-class+p+a")),
        "p_class_constrained": bool(
            cfg.p_class_constrained and _has_component(mode, "p")),
        "candidate_prompt_margin_aware": bool(
            mode == "x-class+p+a"
            and float(cfg.candidate_prompt_margin_weight) > 0),
        "class_representation_aware": bool(
            mode in ("x-class-rep", "x-class+p+a")
            and float(cfg.class_representation_weight) > 0),
        "provenance_aware": bool(mode in ("x-provenance", "x-slot-rep")),
        "slot_representation_aware": bool(mode == "x-slot-rep"),
        "base_slot_class_id": (base_slot_class_id.detach().cpu().tolist()
                                if base_slot_class_id is not None else None),
        "base_slot_group_id": (base_slot_group_id.detach().cpu().tolist()
                                if base_slot_group_id is not None else None),
        "support_slot_id": (support_slot_id.detach().cpu().tolist()
                            if support_slot_id is not None else None),
        "construction_seconds": float(time.perf_counter() - started),
        "materialization": {key: value for key, value in material_audit.items()
                            if key != "adjacency" and key != "p_assignment"},
        "p_response_gate_summary": response_gate_summary,
    }
    if "p_assignment" in material_audit:
        assignment = material_audit["p_assignment"].detach()
        audit["p_assignment_entropy"] = float(
            -(assignment.clamp_min(1e-12)
              * assignment.clamp_min(1e-12).log()).sum(1).mean())
        audit["p_column_sums"] = assignment.sum(0).tolist()
    return result, audit


def stop_gradient_control(base: dict, support: dict, old_rows: int,
                          editable_rows: list[int], mode: str, device: str,
                          cfg: Config) -> tuple[dict, dict]:
    """Negative control for X: block the Student-to-memory path."""
    if mode != "x":
        raise NotImplementedError("stop-gradient control is registered for X")
    result = clone_memory(base)
    audit = {"version": "direct-student-outer-stop-gradient-v1",
             "mode": mode, "editable_rows": list(editable_rows),
             "max_relative_displacement": 0.0,
             "nonzero_outer_gradient": False,
             "historical_raw_reads": 0, "report_labels_used": False}
    return result, audit
