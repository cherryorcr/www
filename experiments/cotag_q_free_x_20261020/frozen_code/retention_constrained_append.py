"""Deterministic synthetic-only retention constraints for the v5 experiment."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

import run_small_comparison_audited as C

G, SCA = C.G, C.SCA
ARMS = ("c0", "core", "topology", "full")


@dataclass(frozen=True)
class RetentionConfig:
    buffer_fraction: float = 0.10
    refresh_alpha: float = 0.25
    anchor_radius: float = 0.05
    refresh_response_bound: float = 0.01
    operator_bound: float = 0.05
    response_bound: float = 0.02

    def dict(self):
        return asdict(self)


def support(memory):
    return C.training_adjacency(memory, "cpu").coalesce()


def sparse(indices, values, n):
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def block_diagonal(old, new):
    n = old.shape[0]
    return sparse(torch.cat((old.indices(), new.indices() + n), 1),
                  torch.cat((old.values(), new.values())), n + new.shape[0])


def mixture(base, proposal, strength):
    result = sparse(torch.cat((base.indices(), proposal.indices()), 1),
                    torch.cat(((1.0 - strength) * base.values(),
                               strength * proposal.values())), base.shape[0])
    keep = result.values() != 0
    return sparse(result.indices()[:, keep], result.values()[keep], base.shape[0])


def difference(a, b):
    return sparse(torch.cat((a.indices(), b.indices()), 1),
                  torch.cat((a.values(), -b.values())), a.shape[0])


def ratio(delta, reference):
    return float(torch.linalg.vector_norm(delta.double()) /
                 torch.linalg.vector_norm(reference.double()).clamp_min(1e-12))


def drift(old, memory):
    if old is None:
        return {"operator": 0.0, "old_block_operator": 0.0,
                "one_hop": 0.0, "two_hop": 0.0, "raw_x": 0.0}
    a, b = support(old).double(), support(memory).double()
    n, total = len(old["x"]), len(memory["x"])
    padded = sparse(a.indices(), a.values(), total)
    delta = difference(b, padded)
    old_rows = delta.indices()[0] < n
    old_block = old_rows & (delta.indices()[1] < n)
    old_x, x = old["x"].double(), memory["x"].double()
    h1, h2 = torch.sparse.mm(a, old_x), torch.sparse.mm(b, x)
    return {"operator": ratio(delta.values()[old_rows], a.values()),
            "old_block_operator": ratio(delta.values()[old_block], a.values()),
            "one_hop": ratio(h2[:n] - h1, h1),
            "two_hop": ratio(torch.sparse.mm(b, h2)[:n] - torch.sparse.mm(a, h1),
                              torch.sparse.mm(a, h1)),
            "raw_x": ratio(x[:n] - old_x, old_x)}


def new_buffer(block, offset, cfg):
    count = min(max(0, len(block["x"]) - 1),
                math.floor(cfg.buffer_fraction * len(block["x"])))
    local = torch.argsort(block["mass"], stable=True)[:count]
    return local + offset, block["x"][local].clone()


def refresh(old, block, cfg):
    updated = deepcopy(old)
    rows = old["buffer_rows"]
    meta = {"eligible_rows": rows.tolist(), "refreshed_rows": [], "scale": 0.0,
            "one_hop": 0.0, "two_hop": 0.0, "max_anchor_relative": 0.0}
    if len(rows) == 0:
        return updated, meta
    sim = 0.5 * (F.normalize(old["x"][rows].float(), dim=1) @
                 F.normalize(block["x"].float(), dim=1).T)
    sim += 0.5 * (F.normalize(old["text_emb"][rows].float(), dim=1) @
                  F.normalize(block["text_emb"].float(), dim=1).T)
    nearest = sim.argmax(1)
    anchor = old["buffer_anchor_x"]
    proposal = (1 - cfg.refresh_alpha) * old["x"][rows] + cfg.refresh_alpha * block["x"][nearest]
    residual = proposal - anchor
    radius = cfg.anchor_radius * anchor.norm(dim=1, keepdim=True)
    proposal = anchor + residual * (radius / residual.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    delta = torch.zeros_like(old["x"])
    delta[rows] = proposal - old["x"][rows]
    a = support(old).double()
    h = torch.sparse.mm(a, old["x"].double())
    dh = torch.sparse.mm(a, delta.double())
    r1 = ratio(dh, h)
    r2 = ratio(torch.sparse.mm(a, dh), torch.sparse.mm(a, h))
    scale = min(1.0, cfg.refresh_response_bound / max(r1, 1e-12),
                cfg.refresh_response_bound / max(r2, 1e-12))
    updated["x"] = old["x"] + (scale * (1 - 1e-6)) * delta
    changed = (updated["x"][rows] != old["x"][rows]).any(1)
    actual_delta = updated["x"].double() - old["x"].double()
    dh = torch.sparse.mm(a, actual_delta)
    meta.update(refreshed_rows=rows[changed].tolist(), scale=scale,
                nearest_new_rows=nearest.tolist(), one_hop=ratio(dh, h),
                two_hop=ratio(torch.sparse.mm(a, dh), torch.sparse.mm(a, h)),
                max_anchor_relative=float(((updated["x"][rows] - anchor).norm(dim=1) /
                                           anchor.norm(dim=1).clamp_min(1e-12)).max()))
    return updated, meta


def constrained_support(old, memory, block, raw_edges, cfg):
    old_s = support(old).double()
    base = block_diagonal(old_s, support(block).double())
    proposal = G.norm_adj(raw_edges, len(memory["x"]), "cpu").double()
    delta = difference(proposal, base)
    n = len(old["x"])
    x = memory["x"].double()
    old_x = old["x"].double()
    target1 = torch.sparse.mm(old_s, old_x)
    target2 = torch.sparse.mm(old_s, target1)
    bx, ex = torch.sparse.mm(base, x), torch.sparse.mm(delta, x)
    # Polynomial coefficients yield a monotone sufficient bound, including
    # buffer changes present at lambda=0. No student or dev labels are used.
    constant2 = torch.sparse.mm(base, bx)[:n] - target2
    linear2 = (torch.sparse.mm(base, ex) + torch.sparse.mm(delta, bx))[:n]
    quadratic2 = torch.sparse.mm(delta, ex)[:n]
    coefficients = {
        "operator": ratio(delta.values()[delta.indices()[0] < n], old_s.values()),
        "one_hop": [ratio(bx[:n] - target1, target1), ratio(ex[:n], target1)],
        "two_hop": [ratio(constant2, target2), ratio(linear2, target2),
                    ratio(quadratic2, target2)],
    }

    def feasible(value):
        return (value * coefficients["operator"] <= cfg.operator_bound
                and sum(c * value ** i for i, c in enumerate(coefficients["one_hop"])) <= cfg.response_bound
                and sum(c * value ** i for i, c in enumerate(coefficients["two_hop"])) <= cfg.response_bound)

    if not feasible(0.0):
        raise AssertionError("buffer update leaves no feasible topology")
    lo, hi = 0.0, 1.0
    if feasible(1.0):
        lo = 1.0
    else:
        for _ in range(48):
            mid = (lo + hi) / 2
            if feasible(mid):
                lo = mid
            else:
                hi = mid
    strength = lo * (1 - 1e-6)
    final = mixture(base, proposal, strength).float().coalesce()
    return final, {"lambda": strength, "bound_coefficients": coefficients}


def append(old, block, arm, cfg=RetentionConfig()):
    if arm not in ARMS:
        raise ValueError(arm)
    core = arm in ("core", "full")
    topology = arm in ("topology", "full")
    n = 0 if old is None else len(old["x"])
    meta = {"arm": arm, "old_rows": n, "new_rows": len(block["x"]),
            "refresh": None, "topology": None}
    if old is None:
        memory = deepcopy(block)
        memory["edge_index"] = SCA.memory_edges(block)
    else:
        working, refresh_meta = refresh(old, block, cfg) if core else (old, None)
        raw_old = dict(working)
        raw_old["edge_index"] = old.get("raw_edge_index", old["edge_index"])
        raw_edges = SCA.append_topology(raw_old, block, cross_degree=1, device="cpu")
        memory = {key: (list(working[key]) + list(block[key]) if key == "texts"
                        else torch.cat((working[key], block[key])))
                  for key in ("x", "text_emb", "texts", "mass", "depth", "stage")}
        memory["edge_index"] = raw_edges
        meta["refresh"] = refresh_meta
        if topology:
            final, meta["topology"] = constrained_support(old, memory, block, raw_edges, cfg)
            memory.update(edge_index=final.indices(), adj_values=final.values(),
                          adj_normalized=True, raw_edge_index=raw_edges)
    if core:
        rows, anchors = new_buffer(block, n, cfg)
        memory["buffer_rows"] = torch.cat((old["buffer_rows"], rows)) if old else rows
        memory["buffer_anchor_x"] = torch.cat((old["buffer_anchor_x"], anchors)) if old else anchors
    return memory, meta


def training_hash(memory):
    tensors = {key: memory[key] for key in ("x", "text_emb")}
    a = support(memory)
    tensors.update(support_indices=a.indices(), support_values=a.values())
    return C.tensor_hash(tensors)


def audit_append(old, block, memory, arm, cfg=RetentionConfig()):
    n = 0 if old is None else len(old["x"])
    if len(memory["x"]) != n + len(block["x"]):
        raise AssertionError("budget changed")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key][n:], block[key]):
            raise AssertionError(f"new block {key} changed")
    if memory["texts"][n:] != block["texts"]:
        raise AssertionError("new text changed")
    if not torch.isfinite(memory["x"]).all():
        raise AssertionError("nonfinite features")
    if old is not None:
        for key in ("text_emb", "mass", "depth", "stage"):
            if not torch.equal(memory[key][:n], old[key]):
                raise AssertionError(f"historical {key} changed")
        if memory["texts"][:n] != old["texts"]:
            raise AssertionError("historical text changed")
        protected = torch.ones(n, dtype=torch.bool)
        if arm in ("core", "full"):
            protected[old["buffer_rows"]] = False
            if not torch.equal(memory["buffer_rows"][:len(old["buffer_rows"])], old["buffer_rows"]):
                raise AssertionError("buffer membership changed")
            rows, anchors = memory["buffer_rows"], memory["buffer_anchor_x"]
            if not torch.equal(anchors[:len(old["buffer_rows"])], old["buffer_anchor_x"]):
                raise AssertionError("birth anchor changed")
            relative = (memory["x"][rows] - anchors).norm(dim=1) / anchors.norm(dim=1).clamp_min(1e-12)
            if len(relative) and float(relative.max()) > cfg.anchor_radius + 1e-6:
                raise AssertionError("cumulative anchor constraint failed")
            a = support(old).double()
            h, delta = old["x"].double(), memory["x"][:n].double() - old["x"].double()
            for _ in range(2):
                h, delta = torch.sparse.mm(a, h), torch.sparse.mm(a, delta)
                if ratio(delta, h) > cfg.refresh_response_bound + 1e-6:
                    raise AssertionError("refresh constraint failed")
        if not torch.equal(memory["x"][:n][protected], old["x"][protected]):
            raise AssertionError("protected core changed")
        raw = memory.get("raw_edge_index", memory["edge_index"])
        old_raw = old.get("raw_edge_index", old["edge_index"])
        if not torch.equal(raw[:, (raw < n).all(0)], old_raw):
            raise AssertionError("old raw edges changed")
    measured = drift(old, memory)
    if arm in ("topology", "full") and old is not None:
        for key, limit in (("operator", cfg.operator_bound), ("one_hop", cfg.response_bound),
                           ("two_hop", cfg.response_bound)):
            if measured[key] > limit + 1e-6:
                raise AssertionError(f"actual {key} constraint failed")
    return {"passed": True, "drift": measured, "training_sha256": training_hash(memory),
            "protected_content_rows": n - (len(old["buffer_rows"]) if old and arm in ("core", "full") else 0)}
