"""Scalable deterministic sparse construction for TGC-R on larger graphs."""
from __future__ import annotations

from dataclasses import asdict
import math
import time

import torch
import torch.nn.functional as F

import tgc_r7_numerics as N
from tgc_r7_stable import initial_assignment
from tgc_deterministic_spmm import from_sparse

J, R, T, M, K = N.J, N.R, N.T, N.M, N.K
S = J.S


def context(old, delta, base, q, teacher_cfg, device, edge_limit=65536, audit_dense=False):
    started = time.perf_counter()
    support = M.lifted_support(old, delta, base, q)
    x = support["x"].to(device)
    text = support["text_emb"].to(device)
    normalized_sparse = M.adjacency(support, "cpu").to(device)
    raw_sparse = J.raw_adjacency(support, "cpu").to(device)
    adjacency = from_sparse(normalized_sparse, edge_limit=edge_limit)
    raw_adjacency = from_sparse(raw_sparse, edge_limit=edge_limit)
    models, audits, parts = [], [], []
    for seed in teacher_cfg.teacher_seeds:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(seed * 7 + 1)
        sequence, losses = [], []
        old_n = len(old["x"])
        for _ in range(teacher_cfg.teacher_steps):
            ids = M.sample_rows(len(x), old_n, teacher_cfg.teacher_batch, generator,
                                "compact", old_n / len(base["x"]))
            sequence.append(ids)
            loss = T._info_nce(model, x, text, adjacency, ids.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        params = dict(model.named_parameters())
        with torch.no_grad():
            parts.extend((S.graph(params, x, adjacency),
                          F.normalize(F.linear(text, params["text_proj.weight"],
                                               params["text_proj.bias"]), dim=1)))
        audits.append({"seed": seed, "steps": teacher_cfg.teacher_steps,
                       "batch_sequence_hash": N.fingerprint({"ids": torch.stack(sequence)}),
                       "state_sha256": N.fingerprint(model.state_dict()),
                       "loss_first": losses[0], "loss_last": losses[-1]})
        models.append(model)
    h = torch.cat(parts, 1).detach() / math.sqrt(len(parts))
    operator_audit = None
    if audit_dense:
        probe_weight = torch.linspace(0.25, 1.25, x.numel(), device=device,
                                     dtype=x.dtype).reshape_as(x)
        operator_audit = {
            "normalized": dense_numeric_audit(normalized_sparse, x, probe_weight, edge_limit),
            "raw": dense_numeric_audit(raw_sparse, x, probe_weight, edge_limit),
        }
    return {"support": support, "h": h,
            "adjacency": raw_adjacency, "normalized": adjacency, "models": models,
            "audit": {"teachers": audits, "parent_hash": T.memory_hash(old),
                      "support_hash": T.memory_hash(support), "q_hash": N.fingerprint(q),
                      "h_hash": N.fingerprint(h), "operator_audit": operator_audit,
                      "current_delta_ids": delta["current_delta_ids"].tolist(),
                      "edge_limit": edge_limit, "seconds": time.perf_counter()-started}}


def _balanced(values, old_n, total_n):
    old_fraction = old_n / total_n
    return old_fraction * values[:old_n].mean() + (1-old_fraction) * values[old_n:].mean()


def objective(p, initial, h, adjacency, old_n, cfg):
    u = J.full_assignment(p, old_n)
    mass = u.sum(0).clamp_min(1e-9)
    z = u.T @ h / mass[:, None]
    projected = u.T @ (adjacency @ u)
    transition = projected / projected.sum(1, keepdim=True).clamp_min(1e-9)
    degree = adjacency.row_sums().clamp_min(1e-9)
    message = (adjacency @ h) / degree[:, None]
    # Keep the frozen dense R7 weighting exactly: old rows are balanced
    # against the number of memory rows (old rows + new synthetic rows),
    # not against the number of raw support nodes used to build them.
    memory_rows = u.shape[1]
    reconstruction = _balanced((h-u@z).square().sum(1), old_n, memory_rows)
    propagation = _balanced((message-u@(transition@z)).square().sum(1), old_n, memory_rows)
    anchor = (p * (p.clamp_min(1e-12).log()
                   - initial.clamp_min(1e-12).log())).sum(1).mean()
    balance = (p.sum(0).clamp_min(1e-9).log()
               - initial.sum(0).log()).square().mean()
    loss = reconstruction + cfg.message*propagation + cfg.anchor*anchor + cfg.mass*balance
    return loss, {"reconstruction": reconstruction, "message": propagation,
                  "anchor": anchor, "mass": balance}, projected


def learn(ctx, q, labels, old_n, cfg):
    h, adjacency = ctx["h"], ctx["adjacency"]
    initial, uniform, allowed, classes = initial_assignment(q, labels, cfg, h.dtype, h.device)
    logits = torch.nn.Parameter(initial.clamp_min(1e-12).log())
    optimizer = torch.optim.Adam([logits], lr=cfg.lr)
    curve = []
    for step in range(cfg.steps+1):
        p = ((1-cfg.floor) * logits.masked_fill(~allowed, -1e9).softmax(1)
             + cfg.floor*uniform)
        loss, terms, projected = objective(p, initial, h, adjacency, old_n, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite deterministic assignment objective")
        if step in (0, 1, 32, 64, cfg.steps):
            curve.append({"step": step, "loss": float(loss.detach()),
                          **{name: float(value.detach()) for name, value in terms.items()}})
        if step == cfg.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("nonfinite deterministic assignment gradient")
        optimizer.step()
    p = p.detach().cpu()
    normalized = p / p.sum(0)
    audit = {"curve": curve, "config": asdict(cfg), "row_classes": classes.cpu().tolist(),
             "mass": p.sum(0).tolist(),
             "min_effective_sources": float((1/normalized.square().sum(0)).min()),
             "assignment_sha256": R.C.tensor_hash({"p": p}),
             "teacher_surrogate_only": True, "report_labels_used": False}
    return p, projected.detach().cpu(), audit


def construct(old, delta, base, q, labels, encoder, teacher_cfg, device, cfg,
              edge_limit=65536, audit_dense=False):
    R.sync(device)
    started = time.perf_counter()
    source_hashes = {"old": T.memory_hash(old), "delta": T.memory_hash(delta),
                     "base": T.memory_hash(base)}
    ctx = context(old, delta, base, q, teacher_cfg, device, edge_limit, audit_dense)
    p, projected, assignment = learn(ctx, q, labels, len(old["x"]), cfg)
    candidates, material = K.materialize(base, delta, p, projected, encoder, cfg)
    R.sync(device)
    if source_hashes != {"old": T.memory_hash(old), "delta": T.memory_hash(delta),
                         "base": T.memory_hash(base)}:
        raise AssertionError("deterministic construction mutated an input")
    return candidates, p, {"context": ctx["audit"], "assignment": assignment,
                           "materialization": material, "source_hashes": source_hashes,
                           "seconds": time.perf_counter()-started,
                           "historical_raw_reads": 0, "report_labels_used": False,
                           "deterministic_sparse_reduction": True}


def apply_x_anchor(memory, c0, old_rows, lam):
    if not 0 <= lam <= 1:
        raise ValueError("anchor coefficient must lie in [0,1]")
    if len(memory["x"]) != len(c0["x"]) or not 0 <= old_rows < len(memory["x"]):
        raise ValueError("memory row mismatch")
    result = T.clone_memory(memory)
    result["x"][old_rows:] = ((1-lam)*c0["x"][old_rows:].float()
                              + lam*memory["x"][old_rows:].float())
    return result


def dense_numeric_audit(sparse, x, weight, edge_limit=65536):
    """Report output/gradient errors and deterministic repeat hashes."""
    segmented = from_sparse(sparse, edge_limit=edge_limit)
    dense = sparse.to_dense()
    x1 = x.detach().clone().requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    actual, expected = segmented @ x1, dense @ x2
    ga, = torch.autograd.grad((actual*weight).sum(), x1)
    ge, = torch.autograd.grad((expected*weight).sum(), x2)
    repeats = [N.fingerprint({"y": (segmented @ x.detach()).detach()}) for _ in range(3)]
    if len(set(repeats)) != 1:
        raise AssertionError("segmented reduction repeated forward is not deterministic")
    return {"max_forward_abs": float((actual-expected).abs().max()),
            "max_backward_abs": float((ga-ge).abs().max()),
            "row_sum_max_abs": float((segmented.row_sums()-dense.sum(1)).abs().max()),
            "repeat_hashes": repeats, "bitwise_repeat_equal": len(set(repeats)) == 1,
            "nnz": int(sparse._nnz()), "shape": list(sparse.shape), "edge_limit": edge_limit}
