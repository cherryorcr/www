"""Minimal Teacher-Guided Continual Revision (TGC-R).

This module implements the smallest non-append-only extension of C0:

* the teacher may read only the previous synthetic memory and the current raw
  delta;
* only a bounded, non-protected subset of old synthetic ``x`` rows may move;
* old text, old topology, mass, depth and birth stage are immutable;
* the incoming C0 block is handled by the caller and is never changed here;
* a revise-or-freeze gate may accept zero revisions.

The teacher is a construction-time instrument.  It is discarded before the
canonical downstream student is trained.  No labels, report-dev nodes, test
nodes or historical raw nodes are used by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
import torch.nn.functional as F

import run_student_compatible_append as SCA


@dataclass(frozen=True)
class RevisionConfig:
    """Frozen knobs for the first minimal TGC-R experiment."""

    protected_fraction: float = 0.80
    revision_fraction_of_old: float = 0.10
    revision_radius: float = 0.10
    teacher_steps: int = 128
    teacher_batch: int = 64
    teacher_lr: float = 1e-3
    teacher_seed: int = 9000
    preserve_tolerance: float = 0.002
    min_global_loss_gain: float = 0.0
    min_gate_score: float = 0.0
    max_old_response_drift: float = 0.02
    mean_old_response_drift: float = 0.002
    shuffle_seed: int = 9002

    def __post_init__(self):
        for value in (self.protected_fraction, self.revision_fraction_of_old):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("fractions must be finite and in [0,1]")
        for value in (self.revision_radius, self.preserve_tolerance,
                      self.max_old_response_drift, self.mean_old_response_drift):
            if not math.isfinite(value) or value < 0:
                raise ValueError("revision bounds must be finite and nonnegative")
        if self.teacher_steps < 0 or self.teacher_batch < 2 or self.teacher_lr <= 0:
            raise ValueError("invalid teacher training configuration")

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


def tensor_hash(value: torch.Tensor) -> str:
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
            digest.update(tensor_hash(value).encode())
        elif isinstance(value, list):
            digest.update(json.dumps(value, ensure_ascii=True).encode())
    return digest.hexdigest()


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total_nodes: int) -> torch.Tensor:
    """Return the current-delta induced graph with local row indices."""
    ids = ids.long().cpu()
    if len(ids) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    remap = torch.full((int(total_nodes),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edges = edge_index.long().cpu()
    local = remap[edges]
    valid = (local >= 0).all(0)
    return local[:, valid].contiguous()


def raw_delta_block(data: dict, ids: torch.Tensor, stage: int) -> dict:
    """Materialize only the currently arriving raw delta for the teacher."""
    ids = torch.sort(ids.long().cpu()).values
    if (not len(ids) or len(ids.unique()) != len(ids)
            or ids.min() < 0 or ids.max() >= len(data["x"])
            or not data["pool_mask"][ids].all()
            or not torch.isin(data["y"][ids],
                              torch.tensor(data["stage_classes"][stage])).all()):
        raise ValueError("teacher sources must be the current pool delta")
    return {
        "x": data["x"][ids].float().clone(),
        "text_emb": data["text_emb"][ids].float().clone(),
        "texts": [data["texts"][int(i)] for i in ids],
        "mass": torch.ones(len(ids), dtype=torch.float32),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
        "edge_index": induced_edges(data["edge_index"], ids,
                                     len(data["x"])),
        # This is an audit-only field.  It is never copied to deployed memory.
        "current_delta_ids": ids,
    }


def _concat_support(old: dict, delta: dict) -> dict:
    """Combine old synthetic rows and current raw rows for teacher fitting."""
    if len(old["x"]) == 0:
        raise ValueError("TGC-R requires non-empty old memory")
    block = {
        "x": delta["x"],
        "text_emb": delta["text_emb"],
        "texts": delta["texts"],
        "mass": delta["mass"],
        "depth": delta["depth"],
        "stage": delta["stage"],
        "edge_index": delta["edge_index"],
    }
    edges = SCA.append_topology(old, block, cross_degree=1, device="cpu")
    return {
        "x": torch.cat((old["x"].float(), block["x"].float())),
        "text_emb": torch.cat((old["text_emb"].float(),
                                block["text_emb"].float())),
        "texts": list(old["texts"]) + list(block["texts"]),
        "mass": torch.cat((old["mass"].float(), block["mass"].float())),
        "depth": torch.cat((old["depth"].long(), block["depth"].long())),
        "stage": torch.cat((old["stage"].long(), block["stage"].long())),
        "edge_index": edges,
        "old_rows": int(len(old["x"])),
        "current_delta_ids": delta.get("current_delta_ids"),
    }


def _info_nce(model, x: torch.Tensor, text: torch.Tensor,
              adjacency: torch.Tensor, rows: torch.Tensor | None = None):
    graph = model.encode_graph(x, adjacency)
    if rows is not None:
        graph = graph[rows]
        text = text[rows]
    encoded_text = model.encode_text(text)
    logits = model.logit_scale.exp().clamp(max=100) * graph @ encoded_text.T
    labels = torch.arange(len(graph), device=x.device)
    return (F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)) / 2


def _teacher_fit(support: dict, cfg: RevisionConfig, device: str):
    """Fit a small legal teacher on the old synthetic/current-delta support."""
    torch.manual_seed(int(cfg.teacher_seed))
    model = SCA.G.CLIPStudent(int(support["x"].shape[1]),
                             text_dim=support["text_emb"].shape[1]).to(device)
    x = support["x"].to(device)
    text = support["text_emb"].to(device)
    adjacency = SCA.G.norm_adj(support["edge_index"], len(x), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.teacher_lr))
    generator = torch.Generator().manual_seed(int(cfg.teacher_seed) * 7 + 1)
    batch = min(max(1, int(cfg.teacher_batch)), len(x))
    losses = []
    model.train()
    for _ in range(max(0, int(cfg.teacher_steps))):
        rows = torch.randperm(len(x), generator=generator, device="cpu")[:batch]
        rows = rows.to(device)
        loss = _info_nce(model, x, text, adjacency, rows)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite TGC-R teacher loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, adjacency, losses


def _teacher_observation(model, support: dict, adjacency: torch.Tensor,
                         device: str) -> dict:
    """Collect gradients and response statistics before any revision."""
    x = support["x"].to(device).detach().clone().requires_grad_(True)
    text = support["text_emb"].to(device)
    loss = _info_nce(model, x, text, adjacency)
    grad = torch.autograd.grad(loss, x, create_graph=False)[0].detach()
    with torch.no_grad():
        graph = model.encode_graph(x.detach(), adjacency)
        encoded_text = model.encode_text(text)
        alignment = (graph * encoded_text).sum(1).detach()
    old_rows = int(support["old_rows"])
    return {
        "loss": float(loss.detach().cpu()),
        "gradient": grad.cpu(),
        "gradient_norm": grad.norm(dim=1).cpu(),
        "alignment": alignment.cpu(),
        "old_alignment_mean": float(alignment[:old_rows].mean().cpu()),
        "old_rows": old_rows,
    }


def _joint_points(memory: dict) -> torch.Tensor:
    x = F.normalize(memory["x"].float(), dim=1)
    t = F.normalize(memory["text_emb"].float(), dim=1)
    return F.normalize(torch.cat((x, t), dim=1), dim=1)


def protected_rows(memory: dict, fraction: float) -> torch.Tensor:
    """Select a deterministic coverage core that the teacher cannot edit."""
    n = len(memory["x"])
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    count = min(n, int(math.ceil(float(fraction) * n)))
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    points = _joint_points(memory)
    mass = memory["mass"].float()
    first = int(torch.argmax(mass).item())
    selected = [first]
    min_distance = 1.0 - points @ points[first]
    for _ in range(1, count):
        candidates = [i for i in range(n) if i not in selected]
        if not candidates:
            break
        candidate = max(candidates,
                        key=lambda i: (float(min_distance[i]), -int(i)))
        selected.append(int(candidate))
        min_distance = torch.minimum(min_distance,
                                     1.0 - points @ points[candidate])
    return torch.tensor(sorted(selected), dtype=torch.long)


def _delta_affinity(old: dict, delta: dict) -> torch.Tensor:
    old_x = F.normalize(old["x"].float(), dim=1)
    old_t = F.normalize(old["text_emb"].float(), dim=1)
    delta_x = F.normalize(delta["x"].float(), dim=1)
    delta_t = F.normalize(delta["text_emb"].float(), dim=1)
    return 0.5 * (old_x @ delta_x.T).max(1).values + 0.5 * (
        old_t @ delta_t.T).max(1).values


def _evaluate(model, x: torch.Tensor, text: torch.Tensor,
              adjacency: torch.Tensor, old_rows: int) -> tuple[float, float]:
    with torch.no_grad():
        loss = _info_nce(model, x, text, adjacency)
        graph = model.encode_graph(x, adjacency)
        encoded_text = model.encode_text(text)
        alignment = (graph * encoded_text).sum(1)
    return float(loss.cpu()), float(alignment[:old_rows].mean().cpu())


def revise_old_memory(old: dict, delta: dict,
                      cfg: RevisionConfig = RevisionConfig(),
                      device: str = "cpu", mode: str = "gate",
                      certificate: dict | None = None) -> tuple[dict, dict]:
    """Apply gated, bounded revisions to old synthetic ``x`` only.

    The returned memory has exactly the same rows and topology as ``old``.
    The function deliberately returns a no-op when the gate has no eligible
    rows or no proposal passes its preservation checks.
    """
    if mode not in {"gate", "no_preserve", "shuffled"}:
        raise ValueError("unknown TGC-R arm")
    out = clone_memory(old)
    if not all(torch.isfinite(m[k]).all() for m in (old, delta)
               for k in ("x", "text_emb")):
        raise ValueError("nonfinite teacher inputs")
    old_n = len(old["x"])
    if old_n == 0 or len(delta["x"]) == 0:
        return out, {
            "version": "tgc-r-v1",
            "status": "no_op_empty_support",
            "updated_rows": [],
            "protected_rows": list(range(old_n)),
            "eligible_rows": [],
            "teacher_access": ["previous_synthetic_memory", "current_raw_delta"],
            "forbidden_access": ["historical_raw_graph", "historical_raw_text",
                                  "report_dev", "test"],
        }

    core = protected_rows(old, cfg.protected_fraction)
    protected = set(int(v) for v in core.tolist())
    quota = min(len([i for i in range(old_n) if i not in protected]),
                max(0, int(math.floor(float(cfg.revision_fraction_of_old)
                                    * old_n)))
                )
    base_audit = {
        "version": "tgc-r-phase-a-v2",
        "mode": mode,
        "config": cfg.dict(),
        "old_rows": old_n,
        "delta_rows": int(len(delta["x"])),
        "protected_rows": core.tolist(),
        "eligible_rows": [i for i in range(old_n) if i not in protected],
        "revision_quota": int(quota),
        "teacher_access": ["previous_synthetic_memory", "current_raw_delta"],
        "forbidden_access": ["historical_raw_graph", "historical_raw_text",
                              "report_dev", "test"],
        "old_memory_sha256": memory_hash(old),
        "current_delta_ids_sha256": tensor_hash(delta["current_delta_ids"]),
    }
    if quota <= 0:
        base_audit.update(status="no_op_zero_revision_quota", updated_rows=[])
        return out, base_audit

    support = _concat_support(old, delta)
    teacher, adjacency, losses = _teacher_fit(support, cfg, device)
    observation = _teacher_observation(teacher, support, adjacency, device)
    gradient = observation["gradient"]
    gradient_norm = observation["gradient_norm"]
    affinity = _delta_affinity(old, delta)
    residual = 1.0 - observation["alignment"]
    utility = gradient_norm[:old_n] * (1.0 + residual[:old_n].clamp_min(0.0)) \
        * (1.0 + affinity.clamp_min(0.0))
    eligible = torch.tensor(base_audit["eligible_rows"], dtype=torch.long)
    permutation = eligible.clone()
    if mode == "shuffled":
        gen = torch.Generator().manual_seed(cfg.shuffle_seed)
        permutation = eligible[torch.randperm(len(eligible), generator=gen)]
        # Shuffle both row-ranking evidence and update directions. The true
        # loss/preservation checks stay unchanged in this negative control.
        utility[eligible] = utility[permutation].clone()
        gradient[eligible] = gradient[permutation].clone()
    ranked = sorted((i for i in range(old_n) if i not in protected),
                    key=lambda i: (-float(utility[i]), int(i)))
    ranked = ranked[:quota]

    x = support["x"].to(device).detach().clone()
    text = support["text_emb"].to(device)
    baseline_loss, baseline_old_alignment = _evaluate(
        teacher, x, text, adjacency, old_n)
    reference_loss, reference_alignment = baseline_loss, baseline_old_alignment
    with torch.no_grad():
        reference_graph = teacher.encode_graph(x, adjacency)[:old_n].clone()
    if certificate is not None:
        certificate.update(state={k: v.detach().cpu().clone()
                                  for k, v in teacher.state_dict().items()},
                           old_graph_response=reference_graph.cpu(),
                           delta_ids=delta["current_delta_ids"].clone(),
                           reference_loss=reference_loss,
                           reference_alignment=reference_alignment,
                           eligible=eligible, permutation=permutation)
    updated = []
    traces = []
    radius = max(0.0, float(cfg.revision_radius))
    for row in ranked:
        current = x[row].detach().clone()
        direction = -gradient[row].to(device)
        direction_norm = float(direction.norm().detach().cpu())
        if direction_norm <= 1e-12 or radius <= 0.0:
            traces.append({"row": int(row), "accepted": False,
                           "reason": "zero_gradient_or_radius"})
            continue
        max_disp = radius * max(float(current.norm().detach().cpu()), 1e-6)
        proposal = current + direction / direction.norm().clamp_min(1e-12) * max_disp
        candidate_x = x.detach().clone()
        candidate_x[row] = proposal
        candidate_loss, candidate_old_alignment = _evaluate(
            teacher, candidate_x, text, adjacency, old_n)
        global_gain = baseline_loss - candidate_loss
        old_gain = candidate_old_alignment - baseline_old_alignment
        gate_score = global_gain + old_gain
        with torch.no_grad():
            candidate_graph = teacher.encode_graph(candidate_x, adjacency)[:old_n]
            drift = (1 - (candidate_graph * reference_graph).sum(1)).clamp_min(0)
        max_drift, mean_drift = float(drift.max()), float(drift.mean())
        preserve_ok = bool(
            candidate_old_alignment >= reference_alignment - cfg.preserve_tolerance
            and max_drift <= cfg.max_old_response_drift
            and mean_drift <= cfg.mean_old_response_drift)
        accepted = bool(
            global_gain > float(cfg.min_global_loss_gain)
            and global_gain > float(cfg.min_gate_score)
            and (mode == "no_preserve" or preserve_ok)
        )
        trace = {
            "row": int(row),
            "utility": float(utility[row]),
            "gradient_norm": float(gradient_norm[row]),
            "delta_affinity": float(affinity[row]),
            "baseline_loss": float(baseline_loss),
            "candidate_loss": float(candidate_loss),
            "global_loss_gain": float(global_gain),
            "baseline_old_alignment": float(baseline_old_alignment),
            "candidate_old_alignment": float(candidate_old_alignment),
            "old_alignment_gain": float(old_gain),
            "gate_score": float(gate_score),
            "acceptance_score": float(global_gain),
            "preserve_ok": preserve_ok,
            "max_old_response_drift": max_drift,
            "mean_old_response_drift": mean_drift,
            "relative_displacement": float(
                (proposal - current).norm().detach().cpu()
                / max(float(current.norm().detach().cpu()), 1e-6)),
            "accepted": accepted,
        }
        traces.append(trace)
        if accepted:
            x[row] = proposal
            baseline_loss = candidate_loss
            baseline_old_alignment = candidate_old_alignment
            out["x"][row] = proposal.detach().cpu()
            updated.append(int(row))

    base_audit.update({
        "status": "complete",
        "updated_rows": updated,
        "teacher_steps": int(cfg.teacher_steps),
        "teacher_loss_first": losses[0] if losses else None,
        "teacher_loss_last": losses[-1] if losses else None,
        "teacher_support_rows": int(len(support["x"])),
        "teacher_support_sha256": memory_hash(support),
        "baseline_loss": float(observation["loss"]),
        "final_loss": float(baseline_loss),
        "baseline_old_alignment": float(observation["old_alignment_mean"]),
        "final_old_alignment": float(baseline_old_alignment),
        "traces": traces,
        "teacher_model_discarded": True,
        "shuffled_from_rows": permutation.tolist(),
        "functional_certificate_scope": "fixed teacher on legal support, not true-old risk bound",
    })
    return out, base_audit


def assert_revision_invariants(old: dict, revised: dict, audit: dict,
                               cfg: RevisionConfig) -> None:
    """Raise if a revision violated the minimal TGC-R contract."""
    n = len(old["x"])
    if len(revised["x"]) != n:
        raise AssertionError("TGC-R changed old memory row count")
    for key in ("text_emb", "mass", "depth", "stage"):
        if not torch.equal(old[key], revised[key]):
            raise AssertionError(f"TGC-R changed immutable field: {key}")
    if old["texts"] != revised["texts"]:
        raise AssertionError("TGC-R changed old text")
    if not torch.equal(old["edge_index"], revised["edge_index"]):
        raise AssertionError("TGC-R changed old topology")
    protected = set(int(v) for v in audit.get("protected_rows", []))
    changed = torch.where((old["x"] != revised["x"]).any(1))[0].tolist()
    if set(changed) != set(audit.get("updated_rows", [])):
        raise AssertionError("changed rows disagree with revision record")
    for row in protected:
        if not torch.equal(old["x"][row], revised["x"][row]):
            raise AssertionError(f"TGC-R changed protected row {row}")
    max_relative = 0.0
    for row in range(n):
        denom = float(old["x"][row].float().norm().clamp_min(1e-6))
        relative = float((revised["x"][row].float() - old["x"][row].float()).norm()) / denom
        max_relative = max(max_relative, relative)
        if relative > float(cfg.revision_radius) + 1e-5:
            raise AssertionError(f"TGC-R radius exceeded at row {row}")
    if len(audit.get("updated_rows", [])) > int(audit.get("revision_quota", n)):
        raise AssertionError("TGC-R revision budget exceeded")
    audit["max_relative_displacement"] = float(max_relative)
