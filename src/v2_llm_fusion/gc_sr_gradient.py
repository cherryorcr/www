"""Gradient-Consistent Semantic-Recursive (GC-SR) X revision.

This module is intentionally a small, isolated extension of the sealed
Semantic-Recursive memory.  It does not change text, memory size, or the
binary topology.  A construction-time objective compares the gradient that a
fresh canonical Student would receive from a legal mixed support
``previous synthetic memory + current raw delta`` with the gradient received
from the candidate synthetic memory.  Only selected ``x`` rows are editable.

The implementation is label-free with respect to report/test data.  It is a
construction operator, not the downstream Student itself.  The default
configuration uses a global graph-output summary as the optional output
matching term; callers can set ``output_weight=0`` for the pure gradient
diagnostic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "frozen_v1_2026_09_09"
if str(FROZEN) not in sys.path:
    sys.path.insert(0, str(FROZEN))
import run_gate1_continual as G  # noqa: E402


IMMUTABLE_FIELDS = ("text_emb", "texts", "mass", "depth", "stage",
                    "edge_index")


@dataclass(frozen=True)
class Config:
    """Small registered configuration for the first GC-SR diagnostic."""

    outer_steps: int = 8
    outer_lr: float = 0.01
    gradient_weight: float = 1.0
    output_weight: float = 0.0
    new_gradient_weight: float = 0.0
    old_gradient_weight: float = 0.0
    old_anchor_weight: float = 0.05
    new_prior_weight: float = 0.0
    old_radius: float = 0.05
    new_radius: float = 0.20
    batch: int = 16
    eval_every: int = 2
    train_seeds: tuple[int, ...] = (9100, 9101)
    validation_seeds: tuple[int, ...] = (9110, 9111)

    def __post_init__(self) -> None:
        if self.outer_steps < 1 or self.outer_lr <= 0:
            raise ValueError("invalid outer optimization configuration")
        if self.eval_every < 1 or self.batch < 0:
            raise ValueError("invalid evaluation configuration")
        for name in ("gradient_weight", "output_weight", "new_gradient_weight",
                     "old_gradient_weight", "old_anchor_weight", "new_prior_weight",
                     "old_radius", "new_radius"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not (self.gradient_weight or self.output_weight or
                self.new_gradient_weight or self.old_gradient_weight):
            raise ValueError("at least one learnable matching term is required")
        if set(self.train_seeds) & set(self.validation_seeds):
            raise ValueError("train and validation seeds must be disjoint")

    def dict(self) -> dict:
        return asdict(self)


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return value


def clone_memory(memory: dict) -> dict:
    return {key: clone_value(value) for key, value in memory.items()}


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(memory):
        value = memory[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(_tensor_digest(value).encode())
        elif isinstance(value, list):
            digest.update(json.dumps(value, ensure_ascii=True).encode())
    return digest.hexdigest()


def validate_memory(memory: dict) -> None:
    required = {"x", *IMMUTABLE_FIELDS}
    missing = sorted(required.difference(memory))
    if missing:
        raise ValueError(f"memory is missing fields: {missing}")
    if memory["x"].ndim != 2 or memory["text_emb"].ndim != 2:
        raise ValueError("x and text_emb must be rank-2 tensors")
    n = len(memory["x"])
    if len(memory["text_emb"]) != n or len(memory["texts"]) != n:
        raise ValueError("memory row fields have inconsistent lengths")
    if not torch.isfinite(memory["x"]).all() or not torch.isfinite(memory["text_emb"]).all():
        raise ValueError("memory contains non-finite payload")
    edge = memory["edge_index"]
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge.numel() and (edge.min() < 0 or edge.max() >= n):
        raise ValueError("edge_index contains an invalid row")


def _student(seed: int, in_dim: int, text_dim: int, device: str):
    # Keep model initialization independent across reference/candidate calls.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = G.CLIPStudent(in_dim, text_dim=text_dim).to(device)
    return model


def _rows(n: int, seed: int, batch: int, device: str) -> torch.Tensor:
    if n <= 0:
        raise ValueError("cannot sample from an empty memory")
    if batch <= 0 or batch >= n:
        return torch.arange(n, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + 17)
    return torch.randperm(n, generator=generator)[:int(batch)].to(device)


def _loss(model, x: torch.Tensor, text: torch.Tensor, adjacency: torch.Tensor,
          rows: torch.Tensor) -> torch.Tensor:
    graph = model.encode_graph(x, adjacency)[rows]
    encoded_text = model.encode_text(text[rows])
    logits = model.logit_scale.exp().clamp(max=100) * graph @ encoded_text.T
    target = torch.arange(len(rows), device=x.device)
    return (F.cross_entropy(logits, target)
            + F.cross_entropy(logits.T, target)) / 2


def _flat(grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in grads])


def _stats(memory: dict, seed: int, device: str, batch: int,
           create_graph: bool, indices: torch.Tensor | None = None) -> dict:
    """Return one fresh Student's loss gradient and graph-output summary."""
    validate_memory(memory)
    x = memory["x"].to(device)
    text = memory["text_emb"].to(device)
    adjacency = G.norm_adj(memory["edge_index"], len(x), device)
    model = _student(seed, x.shape[1], text.shape[1], device)
    if indices is None:
        rows = _rows(len(x), seed, batch, device)
    else:
        candidates = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if len(candidates) == 0:
            raise ValueError("group gradient requested for an empty row set")
        if batch > 0 and len(candidates) > batch:
            generator = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + 29)
            candidates = candidates[torch.randperm(len(candidates), generator=generator)[:int(batch)]]
        rows = candidates.to(device)
    loss = _loss(model, x, text, adjacency, rows)
    grads = torch.autograd.grad(loss, tuple(model.parameters()),
                                create_graph=create_graph,
                                allow_unused=False)
    graph = model.encode_graph(x, adjacency)
    return {
        "gradient": _flat(grads),
        "graph_mean": graph.mean(0),
        "loss": loss,
        "rows": rows.detach().cpu(),
    }


def _relative_anchor(relative: torch.Tensor, base_rows: torch.Tensor,
                     selected: torch.Tensor, old_rows: int) -> torch.Tensor:
    old_mask = selected < int(old_rows)
    if not bool(old_mask.any()):
        return relative.new_zeros(())
    scales = base_rows[old_mask].norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (relative[old_mask] / scales).square().mean()


def _prior_penalty(relative: torch.Tensor, selected: torch.Tensor,
                   old_rows: int) -> torch.Tensor:
    new_mask = selected >= int(old_rows)
    if not bool(new_mask.any()):
        return relative.new_zeros(())
    return relative[new_mask].square().mean()


def project_trust_region(relative: torch.Tensor, base_rows: torch.Tensor,
                         selected: torch.Tensor, old_rows: int,
                         old_radius: float, new_radius: float) -> None:
    """Project each editable row into its old/new relative-radius ball."""
    with torch.no_grad():
        scales = base_rows.norm(dim=1, keepdim=True).clamp_min(1e-6)
        limits = torch.where(
            (selected < int(old_rows)).unsqueeze(1),
            torch.full_like(scales, float(old_radius)),
            torch.full_like(scales, float(new_radius)),
        ) * scales
        norms = relative.norm(dim=1, keepdim=True).clamp_min(1e-12)
        relative.mul_((limits / norms).clamp(max=1.0))


def _candidate(memory: dict, selected: torch.Tensor,
               relative: torch.Tensor) -> dict:
    out = clone_memory(memory)
    out["x"] = memory["x"].clone()
    out["x"][selected.cpu()] = memory["x"][selected.cpu()] + relative.detach().cpu()
    return out


def _objective(base: dict, support: dict, selected: torch.Tensor,
               relative: torch.Tensor, old_rows: int, references: dict,
               seeds: tuple[int, ...], device: str, cfg: Config,
               create_graph: bool) -> tuple[torch.Tensor, dict]:
    candidate_x = base["x"].to(device).clone()
    candidate_x[selected] = candidate_x[selected] + relative
    candidate = clone_memory(base)
    candidate["x"] = candidate_x
    total = candidate_x.new_zeros(())
    records = []
    for seed in seeds:
        current = _stats(candidate, int(seed), device, cfg.batch, create_graph)
        reference = references[int(seed)]
        ref_grad = reference["gradient"].to(device)
        cosine = F.cosine_similarity(current["gradient"], ref_grad, dim=0,
                                     eps=1e-8)
        gradient_loss = 1.0 - cosine
        output_loss = current["graph_mean"].sub(
            reference["graph_mean"].to(device)).square().mean()
        total = total + cfg.gradient_weight * gradient_loss
        total = total + cfg.output_weight * output_loss
        records.append({
            "seed": int(seed),
            "gradient_cosine": float(cosine.detach()),
            "gradient_loss": float(gradient_loss.detach()),
            "output_loss": float(output_loss.detach()),
            "candidate_loss": float(current["loss"].detach()),
            "reference_loss": float(reference["loss"]),
        })
        group_records = {}
        if cfg.new_gradient_weight or cfg.old_gradient_weight:
            group_specs = (
                ("old", cfg.old_gradient_weight,
                 torch.arange(0, int(old_rows), device=device)),
                ("new", cfg.new_gradient_weight,
                 torch.arange(int(old_rows), len(candidate_x), device=device)),
            )
            for name, weight, group_rows in group_specs:
                if not weight or len(group_rows) == 0:
                    continue
                group_current = _stats(candidate, int(seed), device, cfg.batch,
                                       create_graph, group_rows)
                group_reference = reference[name]
                group_cosine = F.cosine_similarity(
                    group_current["gradient"],
                    group_reference["gradient"].to(device), dim=0, eps=1e-8)
                total = total + float(weight) * (1.0 - group_cosine)
                group_records[name] = {
                    "gradient_cosine": float(group_cosine.detach()),
                    "gradient_loss": float((1.0 - group_cosine).detach()),
                    "candidate_loss": float(group_current["loss"].detach()),
                    "reference_loss": float(group_reference["loss"]),
                }
            records[-1]["groups"] = group_records
    total = total / max(1, len(seeds))
    anchor = _relative_anchor(relative, base["x"].to(device)[selected],
                               selected, old_rows)
    prior = _prior_penalty(relative, selected, old_rows)
    total = total + cfg.old_anchor_weight * anchor
    total = total + cfg.new_prior_weight * prior
    return total, {"seeds": records, "anchor_loss": float(anchor.detach()),
                   "new_prior_loss": float(prior.detach())}


def _references(memory: dict, seeds: tuple[int, ...], device: str,
                batch: int, split_old_rows: int | None = None) -> dict:
    result = {}
    for seed in seeds:
        stats = _stats(memory, int(seed), device, batch, False)
        row = {
            "gradient": stats["gradient"].detach().cpu(),
            "graph_mean": stats["graph_mean"].detach().cpu(),
            "loss": float(stats["loss"].detach().cpu()),
            "rows": stats["rows"].tolist(),
        }
        if split_old_rows is not None:
            for name, indices in (
                ("old", torch.arange(0, int(split_old_rows))),
                ("new", torch.arange(int(split_old_rows), len(memory["x"]))),
            ):
                if len(indices) == 0:
                    continue
                group = _stats(memory, int(seed), device, batch, False, indices)
                row[name] = {
                    "gradient": group["gradient"].detach().cpu(),
                    "graph_mean": group["graph_mean"].detach().cpu(),
                    "loss": float(group["loss"].detach().cpu()),
                    "rows": group["rows"].tolist(),
                }
        result[int(seed)] = row
    return result


def optimize(base: dict, support: dict, editable_rows: list[int] | torch.Tensor,
             old_rows: int, stage: int, device: str = "cpu",
             cfg: Config = Config()) -> tuple[dict, dict]:
    """Optimize selected synthetic ``x`` rows against legal support gradients.

    ``base`` is the sealed Semantic-Recursive/C0 candidate. ``support`` is
    the legal mixed teacher support and may contain a different number of
    rows. The function never mutates either input and leaves all fields except
    ``x`` bitwise unchanged in the returned memory.
    """
    validate_memory(base)
    validate_memory(support)
    selected = torch.as_tensor(editable_rows, dtype=torch.long)
    if selected.numel() == 0:
        raise ValueError("at least one editable row is required")
    selected = torch.unique(selected, sorted=True)
    if int(selected.min()) < 0 or int(selected.max()) >= len(base["x"]):
        raise ValueError("editable row is outside base memory")
    if old_rows < 0 or old_rows > len(base["x"]):
        raise ValueError("invalid old_rows")
    base_hash = memory_hash(base)
    support_hash = memory_hash(support)
    started = time.perf_counter()
    train_refs = _references(base_support := support, cfg.train_seeds,
                             device, cfg.batch, old_rows)
    val_refs = _references(base_support, cfg.validation_seeds,
                           device, cfg.batch, old_rows)
    selected_device = selected.to(device)
    base_rows = base["x"].to(device)[selected_device].detach()
    relative = torch.nn.Parameter(torch.zeros_like(base_rows))
    optimizer = torch.optim.Adam([relative], lr=float(cfg.outer_lr))
    best = None
    best_step = 0
    best_val = math.inf
    curve = []
    for step in range(1, int(cfg.outer_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, record = _objective(base, support, selected_device, relative,
                                  old_rows, train_refs, cfg.train_seeds,
                                  device, cfg, True)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite GC-SR objective")
        loss.backward()
        if relative.grad is None or not torch.isfinite(relative.grad).all():
            raise FloatingPointError("missing/non-finite X hypergradient")
        grad_norm = float(relative.grad.norm().detach())
        optimizer.step()
        project_trust_region(relative, base_rows, selected_device, old_rows,
                             cfg.old_radius, cfg.new_radius)
        row = {"step": step, "train_objective": float(loss.detach()),
               "gradient_norm": grad_norm, "train": record}
        if step == 1 or step % int(cfg.eval_every) == 0 or step == cfg.outer_steps:
            # ``create_graph=False`` releases the second-order path, but the
            # first-order Student gradient is still required for validation.
            # Do not wrap this call in ``torch.no_grad()``.
            val_loss, val_record = _objective(
                base, support, selected_device, relative, old_rows,
                val_refs, cfg.validation_seeds, device, cfg, False)
            score = float(val_loss.detach())
            row.update(validation_objective=score, validation=val_record)
            if score < best_val:
                best_val = score
                best_step = step
                best = relative.detach().clone()
        curve.append(row)
    if best is None:
        raise AssertionError("GC-SR did not produce a validation checkpoint")
    result = clone_memory(base)
    result["x"] = base["x"].clone()
    result["x"][selected] = base["x"][selected] + best.cpu()
    if memory_hash(base) != base_hash or memory_hash(support) != support_hash:
        raise AssertionError("GC-SR mutated an input memory")
    for key in IMMUTABLE_FIELDS:
        left, right = base[key], result[key]
        equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
        if not equal:
            raise AssertionError(f"GC-SR changed immutable field: {key}")
    with torch.no_grad():
        displacement = (result["x"][selected] - base["x"][selected]).norm(dim=1)
        scale = base["x"][selected].norm(dim=1).clamp_min(1e-6)
        relative_displacement = displacement / scale
    audit = {
        "version": "gc-sr-gradient-v1",
        "stage": int(stage),
        "config": cfg.dict(),
        "editable_rows": selected.tolist(),
        "old_rows": int(old_rows),
        "new_rows": int(len(base["x"]) - old_rows),
        "input_base_sha256": base_hash,
        "input_support_sha256": support_hash,
        "output_sha256": memory_hash(result),
        "selected_outer_step": int(best_step),
        "validation_objective": float(best_val),
        "max_relative_displacement": float(relative_displacement.max()),
        "mean_relative_displacement": float(relative_displacement.mean()),
        "relative_displacement": relative_displacement.tolist(),
        "nonzero_hypergradient_steps": int(sum(
            row["gradient_norm"] > 0 for row in curve)),
        "construction_seconds": float(time.perf_counter() - started),
        "curve": curve,
        "text_changed": False,
        "budget_changed": False,
        "topology_changed": False,
        "historical_raw_reads": 0,
        "report_labels_used": False,
    }
    return result, audit


def diagnostic(base: dict, support: dict, device: str = "cpu",
               cfg: Config = Config()) -> dict:
    """Measure the initial gradient/output mismatch without editing memory."""
    refs = _references(support, cfg.validation_seeds, device, cfg.batch,
                       min(len(base["x"]), len(support["x"])))
    rows = []
    for seed in cfg.validation_seeds:
        current = _stats(base, int(seed), device, cfg.batch, False)
        ref = refs[int(seed)]
        cosine = F.cosine_similarity(current["gradient"],
                                     ref["gradient"].to(device), dim=0)
        output = current["graph_mean"].sub(
            ref["graph_mean"].to(device)).square().mean()
        rows.append({"seed": int(seed), "gradient_cosine": float(cosine.detach()),
                     "output_loss": float(output.detach()),
                     "base_loss": float(current["loss"].detach()),
                     "support_loss": float(ref["loss"])})
    return {"version": "gc-sr-gradient-diagnostic-v1", "rows": rows,
            "mean_gradient_cosine": float(sum(r["gradient_cosine"] for r in rows) / len(rows)),
            "mean_output_loss": float(sum(r["output_loss"] for r in rows) / len(rows)),
            "base_sha256": memory_hash(base), "support_sha256": memory_hash(support),
            "report_labels_used": False}


def row_sensitivity(memory: dict, seed: int = 9120, device: str = "cpu",
                    batch: int = 0) -> torch.Tensor:
    """Compute label-free Student loss sensitivity for every synthetic row."""
    validate_memory(memory)
    x = memory["x"].to(device).detach().clone().requires_grad_(True)
    text = memory["text_emb"].to(device)
    adjacency = G.norm_adj(memory["edge_index"], len(x), device)
    model = _student(seed, x.shape[1], text.shape[1], device)
    rows = _rows(len(x), seed, batch, device)
    loss = _loss(model, x, text, adjacency, rows)
    grad = torch.autograd.grad(loss, x, create_graph=False)[0]
    return grad.norm(dim=1).detach().cpu()


def select_editable_rows(memory: dict, old_rows: int, max_old_rows: int,
                         max_new_rows: int, device: str = "cpu",
                         seed: int = 9120, batch: int = 0) -> tuple[list[int], dict]:
    """Select high-sensitivity old/new X rows without report labels."""
    scores = row_sensitivity(memory, seed=seed, device=device, batch=batch)
    old_scores = scores[:int(old_rows)]
    new_scores = scores[int(old_rows):]
    old_count = min(int(max_old_rows), len(old_scores))
    new_count = min(int(max_new_rows), len(new_scores))
    old_idx = (torch.topk(old_scores, old_count, largest=True, sorted=True).indices.tolist()
               if old_count else [])
    new_idx = ((torch.topk(new_scores, new_count, largest=True, sorted=True).indices
                + int(old_rows)).tolist() if new_count else [])
    selected = sorted(int(v) for v in old_idx + new_idx)
    return selected, {
        "mode": "student_sensitivity",
        "seed": int(seed),
        "batch": int(batch),
        "old_rows": int(old_rows),
        "max_old_rows": int(max_old_rows),
        "max_new_rows": int(max_new_rows),
        "selected_rows": selected,
        "old_scores": old_scores.tolist(),
        "new_scores": new_scores.tolist(),
    }
