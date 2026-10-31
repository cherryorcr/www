"""C0-M: bounded core-buffer maintenance for continual synthetic memory.

The C0 construction of the incoming block is unchanged.  C0-M only changes
how a small replaceable historical buffer is maintained:

* a deterministic farthest-point core is protected forever;
* only redundant buffer rows are eligible for a bounded joint X/text update;
* the update is paired with a novel row from the current C0 delta;
* the old topology and total row budget remain unchanged;
* new rows are appended exactly as in C0.

This is deliberately response-free.  A student is used downstream for the
same evaluation as C0, but no short-training F1 selector is used to decide a
maintenance update.  The previous low-mass ``bounded_refresh`` is retained
as a negative control in the runner and is not this algorithm.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

import run_student_compatible_append as SCA


@dataclass(frozen=True)
class MaintenanceConfig:
    buffer_fraction: float = 0.20
    refresh_fraction_of_old: float = 0.10
    blend_alpha: float = 0.50
    anchor_radius: float = 0.25
    core_rule: str = "joint_farthest_point"
    target_rule: str = "highest_redundancy"
    candidate_rule: str = "highest_novelty_to_core"

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


def joint_points(memory: dict) -> torch.Tensor:
    x = F.normalize(memory["x"].float(), dim=1)
    t = F.normalize(memory["text_emb"].float(), dim=1)
    return F.normalize(torch.cat((x, t), dim=1), dim=1)


def _argmax_stable(values: torch.Tensor, candidates: torch.Tensor) -> int:
    best = float(values[candidates[0]])
    best_id = int(candidates[0])
    for item in candidates[1:].tolist():
        value = float(values[item])
        if value > best or (value == best and int(item) < best_id):
            best, best_id = value, int(item)
    return best_id


def select_core_buffer(memory: dict, cfg: MaintenanceConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a coverage-preserving core and the remaining replaceable buffer."""
    n = len(memory["x"])
    if n == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    core_count = max(1, min(n, int(math.ceil((1.0 - cfg.buffer_fraction) * n))))
    points = joint_points(memory)
    mass = memory["mass"].float()
    first = _argmax_stable(mass, torch.arange(n))
    selected = [first]
    min_distance = 1.0 - points @ points[first]
    for _ in range(1, core_count):
        candidates = torch.tensor([i for i in range(n) if i not in selected],
                                  dtype=torch.long)
        if len(candidates) == 0:
            break
        nxt = _argmax_stable(min_distance, candidates)
        selected.append(nxt)
        min_distance = torch.minimum(min_distance, 1.0 - points @ points[nxt])
    core = torch.tensor(sorted(selected), dtype=torch.long)
    mask = torch.ones(n, dtype=torch.bool)
    mask[core] = False
    buffer = torch.where(mask)[0]
    return core, buffer


def initialize_metadata(memory: dict, cfg: MaintenanceConfig) -> dict:
    out = clone_memory(memory)
    core, buffer = select_core_buffer(out, cfg)
    out["core_rows"] = core
    out["buffer_rows"] = buffer
    out["buffer_anchor_x"] = out["x"][buffer].clone()
    out["buffer_anchor_text"] = out["text_emb"][buffer].clone()
    out["maintenance_version"] = "c0m-core-buffer-v1"
    return out


def _safe_norm(value: torch.Tensor) -> torch.Tensor:
    return value.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _redundancy_scores(points: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    if len(rows) == 0:
        return torch.empty(0)
    sim = points[rows] @ points.T
    # A row is replaceable when it has a close duplicate elsewhere.
    local = rows.unsqueeze(1)
    sim.scatter_(1, local, -2.0)
    return sim.max(1).values


def _novelty_scores(old: dict, block: dict) -> torch.Tensor:
    new_points = joint_points(block)
    core = old["core_rows"]
    if len(core) == 0:
        return torch.ones(len(new_points))
    old_points = joint_points(old)
    similarity = new_points @ old_points[core].T
    return 1.0 - similarity.max(dim=1).values


def _bounded_blend(old_value: torch.Tensor, candidate: torch.Tensor,
                   anchor: torch.Tensor, radius: float,
                   blend_alpha: float) -> tuple[torch.Tensor, float]:
    alpha = max(0.0, min(1.0, float(blend_alpha)))
    proposal = (1.0 - alpha) * old_value + alpha * candidate
    displacement = proposal - anchor
    limit = float(radius) * float(anchor.norm().clamp_min(1e-12))
    length = float(displacement.norm())
    if length > limit:
        proposal = anchor + displacement * (limit / max(length, 1e-12))
    relative = float((proposal - anchor).norm() /
                     anchor.norm().clamp_min(1e-12))
    return proposal, relative


def maintain(old: dict, block: dict, cfg: MaintenanceConfig) -> tuple[dict, dict]:
    """Refresh a bounded, redundant buffer and return audit metadata."""
    out = clone_memory(old)
    buffer = old["buffer_rows"].long()
    if len(buffer) == 0 or len(block["x"]) == 0:
        return out, {"eligible_rows": buffer.tolist(), "updated_rows": [],
                      "candidate_rows": [], "count": 0,
                      "max_anchor_relative": 0.0,
                      "reason": "empty_buffer_or_delta"}

    count = min(len(buffer), max(1, int(math.ceil(
        cfg.refresh_fraction_of_old * len(old["x"])))), len(block["x"]))
    points = joint_points(old)
    redundancy = _redundancy_scores(points, buffer)
    target_order = sorted(range(len(buffer)),
                          key=lambda i: (-float(redundancy[i]), int(buffer[i])))
    targets = buffer[torch.tensor(target_order[:count], dtype=torch.long)]
    novelty = _novelty_scores(old, block)
    candidate_order = sorted(range(len(block["x"])),
                             key=lambda i: (-float(novelty[i]), int(i)))
    candidates = torch.tensor(candidate_order[:count], dtype=torch.long)

    anchor_x = old["buffer_anchor_x"]
    anchor_t = old["buffer_anchor_text"]
    target_position = {int(row): index for index, row in enumerate(buffer.tolist())}
    updates = []
    max_relative = 0.0
    for target, candidate in zip(targets.tolist(), candidates.tolist()):
        position = target_position[int(target)]
        x_value, x_relative = _bounded_blend(
            old["x"][target].float(), block["x"][candidate].float(),
            anchor_x[position].float(), cfg.anchor_radius, cfg.blend_alpha)
        t_value, t_relative = _bounded_blend(
            old["text_emb"][target].float(), block["text_emb"][candidate].float(),
            anchor_t[position].float(), cfg.anchor_radius, cfg.blend_alpha)
        out["x"][target] = x_value
        # The student normalizes text at use time.  Keeping the blended value
        # unnormalized makes alpha=0 a true content no-op and preserves the
        # anchor-radius audit in the stored representation.
        out["text_emb"][target] = t_value
        out["texts"][target] = block["texts"][candidate]
        out["mass"][target] = ((1.0 - cfg.blend_alpha) * old["mass"][target]
                                + cfg.blend_alpha * block["mass"][candidate])
        out["depth"][target] = old["depth"][target] + 1
        out["stage"][target] = block["stage"][candidate]
        max_relative = max(max_relative, x_relative, t_relative)
        updates.append({"old_row": int(target), "new_row": int(candidate),
                        "redundancy": float(redundancy[position]),
                        "novelty": float(novelty[candidate]),
                        "x_anchor_relative": x_relative,
                        "text_anchor_relative": t_relative})

    return out, {
        "eligible_rows": buffer.tolist(),
        "updated_rows": [item["old_row"] for item in updates],
        "candidate_rows": [item["new_row"] for item in updates],
        "count": len(updates),
        "updates": updates,
        "max_anchor_relative": max_relative,
        "rule": cfg.dict(),
    }


def append(old: dict | None, block: dict, keep: int,
           cfg: MaintenanceConfig = MaintenanceConfig(), device: str | None = None) -> tuple[dict, dict]:
    if old is None:
        out = {key: clone_value(value) for key, value in block.items()}
        out["edge_index"] = SCA.memory_edges(block, device=device)
        out = initialize_metadata(out, cfg)
        return out, {"old_rows": 0, "new_rows": len(block["x"]),
                     "updated_rows": [], "core_rows": out["core_rows"].tolist(),
                     "buffer_rows": out["buffer_rows"].tolist(),
                     "old_old_edges_unchanged": True,
                     "config": cfg.dict()}
    if int(keep) != len(old["x"]):
        raise ValueError("C0-M is append-growing and cannot prune old rows")
    working, maintenance = maintain(old, block, cfg)
    n = len(working["x"])
    out = {
        key: (working[key] + list(block[key]) if key == "texts"
              else torch.cat((working[key], block[key])))
        for key in ("x", "text_emb", "texts", "mass", "depth", "stage")
    }
    out["edge_index"] = SCA.append_topology(
        working, block, cross_degree=1, device=device)
    out["core_rows"] = old["core_rows"].clone()
    out["buffer_rows"] = torch.cat((old["buffer_rows"].clone(),
                                     torch.arange(n, n + len(block["x"]),
                                                 dtype=torch.long)))
    out["buffer_anchor_x"] = torch.cat((old["buffer_anchor_x"].clone(),
                                        block["x"].clone()))
    out["buffer_anchor_text"] = torch.cat((old["buffer_anchor_text"].clone(),
                                            block["text_emb"].clone()))
    out["maintenance_version"] = "c0m-core-buffer-v1"
    old_edges = old["edge_index"]
    new_old_edges = out["edge_index"]
    new_old_edges = new_old_edges[:, (new_old_edges < n).all(0)]
    if not torch.equal(old_edges, new_old_edges):
        raise AssertionError("C0-M changed old-old topology")
    if len(out["x"]) != n + len(block["x"]):
        raise AssertionError("C0-M row budget changed")
    max_relative = 0.0
    if len(old["buffer_rows"]):
        positions = {int(row): i for i, row in enumerate(old["buffer_rows"].tolist())}
        for row in old["buffer_rows"].tolist():
            i = positions[int(row)]
            relative = float((out["x"][row] - out["buffer_anchor_x"][i]).norm() /
                             out["buffer_anchor_x"][i].norm().clamp_min(1e-12))
            max_relative = max(max_relative, relative)
    return out, {
        "old_rows": n, "new_rows": len(block["x"]),
        "maintenance": maintenance,
        "updated_rows": maintenance["updated_rows"],
        "core_rows": out["core_rows"].tolist(),
        "buffer_rows": out["buffer_rows"].tolist(),
        "max_anchor_relative": max(max_relative,
                                    maintenance.get("max_anchor_relative", 0.0)),
        "old_old_edges_unchanged": True,
        "config": cfg.dict(),
    }


def assert_invariants(old: dict | None, block: dict, memory: dict,
                      audit: dict, cfg: MaintenanceConfig) -> None:
    n = 0 if old is None else len(old["x"])
    if len(memory["x"]) != n + len(block["x"]):
        raise AssertionError("C0-M budget mismatch")
    for key in ("text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key][n:], block[key]):
            raise AssertionError(f"C0-M appended {key} changed")
    if memory["texts"][n:] != block["texts"]:
        raise AssertionError("C0-M appended text changed")
    if old is None:
        return
    for key in ("core_rows", "buffer_rows", "buffer_anchor_x", "buffer_anchor_text"):
        if key not in memory:
            raise AssertionError(f"C0-M missing metadata {key}")
    if not torch.equal(memory["core_rows"], old["core_rows"]):
        raise AssertionError("C0-M core membership changed")
    old_edges = memory["edge_index"][:, (memory["edge_index"] < n).all(0)]
    if not torch.equal(old_edges, old["edge_index"]):
        raise AssertionError("C0-M old-old edge block changed")
    if float(audit["max_anchor_relative"]) > cfg.anchor_radius + 1e-6:
        raise AssertionError("C0-M anchor radius exceeded")
    protected = memory["core_rows"]
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key][protected], old[key][protected]):
            raise AssertionError(f"C0-M protected core changed: {key}")
    if memory["texts"]:
        old_core_texts = [old["texts"][int(i)] for i in protected]
        new_core_texts = [memory["texts"][int(i)] for i in protected]
        if old_core_texts != new_core_texts:
            raise AssertionError("C0-M protected core text changed")
