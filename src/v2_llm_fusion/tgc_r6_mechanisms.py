"""Controlled R5 mechanism interventions; no report data enters construction."""
from dataclasses import replace
import math
import time

import torch
import torch.nn.functional as F
import tgc_shared_assignment as J

R, T, M, S = J.R, J.T, J.M, J.S


def blend_parent(c0, revised, bits):
    if len(bits) != 3 or any(b not in "01" for b in bits):
        raise ValueError("parent bits must encode X,text,topology")
    result = T.clone_memory(c0)
    for bit, keys in zip(bits, (("x", "mass"), ("text_emb", "texts"), ("edge_index",))):
        if bit == "1":
            for key in keys:
                result[key] = T.clone_value(revised[key])
    assert torch.equal(c0["stage"], revised["stage"])
    if bits == "000":
        assert T.memory_hash(result) == T.memory_hash(c0)
    if bits == "111":
        assert T.memory_hash(result) == T.memory_hash(revised)
    return result


def encode_models(models, memory, adjacency, device):
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    parts = []
    with torch.no_grad():
        for model in models:
            params = dict(model.named_parameters())
            parts.append(S.graph(params, x, adjacency))
            parts.append(F.normalize(F.linear(text, params["text_proj.weight"], params["text_proj.bias"]), dim=1))
    return torch.cat(parts, 1).detach() / math.sqrt(len(parts))


def context(old, delta, base, q, teacher_cfg, device, kind="trained"):
    R.sync(device)
    started = time.perf_counter()
    support = M.lifted_support(old, delta, base, q)
    models, audits = [], []
    adjacency = M.adjacency(support, device)
    if kind == "geometry":
        h = torch.cat((F.normalize(support["x"].to(device), dim=1),
                       F.normalize(support["text_emb"].to(device), dim=1)), 1) / math.sqrt(2)
    else:
        for seed in teacher_cfg.teacher_seeds:
            if kind == "trained":
                model, audit = M.fit_teacher(support, base, seed, "compact", teacher_cfg, device)
            elif kind == "random":
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    model = R.G.CLIPStudent(support["x"].shape[1], text_dim=support["text_emb"].shape[1]).to(device)
                model.eval()
                audit = {"seed": seed, "steps": 0, "state_sha256": R.C.tensor_hash(model.state_dict())}
            else:
                raise ValueError(kind)
            models.append(model)
            audits.append(audit)
        h = encode_models(models, support, adjacency, device)
    R.sync(device)
    return {"support": support, "h": h.detach(), "adjacency": J.raw_adjacency(support, device),
            "models": models, "audit": {"kind": kind, "teachers": audits,
            "support_hash": T.memory_hash(support), "parent_hash": T.memory_hash(old),
            "seconds": time.perf_counter() - started, "current_delta_ids": delta["current_delta_ids"].tolist()}}


def symmetric_dense(a):
    inv = a.sum(1).clamp_min(1e-9).rsqrt()
    return inv[:, None] * a * inv[None, :]


def symmetric_sparse(a):
    a = a.coalesce()
    inv = torch.sparse.sum(a, 1).to_dense().clamp_min(1e-9).rsqrt()
    i = a.indices()
    return torch.sparse_coo_tensor(i, a.values() * inv[i[0]] * inv[i[1]], a.shape).coalesce()


def objective(p, initial, h, adjacency, old_n, cfg, mode="rw", reconstruction_weight=1.):
    if mode == "rw" and reconstruction_weight == 1.:
        return J.objective(p, initial, h, adjacency, old_n, cfg)
    u = J.full_assignment(p, old_n)
    z = (u.T @ h) / u.sum(0).clamp_min(1e-9)[:, None]
    coarse = u.T @ torch.sparse.mm(adjacency, u)
    if mode == "symmetric":
        message = torch.sparse.mm(symmetric_sparse(adjacency), h)
        transition = symmetric_dense(coarse)
    elif mode == "rw":
        message = torch.sparse.mm(adjacency, h) / torch.sparse.sum(adjacency, 1).to_dense().clamp_min(1e-9)[:, None]
        transition = coarse / coarse.sum(1, keepdim=True).clamp_min(1e-9)
    else:
        raise ValueError(mode)
    fraction = old_n / u.shape[1]

    def balanced(v):
        return fraction * v[:old_n].mean() + (1 - fraction) * v[old_n:].mean()

    recon = balanced((h - u @ z).square().sum(1))
    msg = balanced((message - u @ (transition @ z)).square().sum(1))
    anchor = (p * (p.clamp_min(1e-12).log() - initial.clamp_min(1e-12).log())).sum(1).mean()
    mass = (p.sum(0).clamp_min(1e-9).log() - initial.sum(0).log()).square().mean()
    loss = reconstruction_weight * recon + cfg.message * msg + cfg.anchor * anchor + cfg.mass * mass
    return loss, {"reconstruction": recon, "message": msg, "anchor": anchor, "mass": mass}, coarse


def learn(q, labels, ctx, old_n, cfg, mode="rw", reconstruction_weight=1.):
    h, a = ctx["h"], ctx["adjacency"]
    if mode == "rw" and reconstruction_weight == 1. and cfg.steps > 0:
        return J.learn(q, labels, h, a, old_n, cfg)
    k, device = int(q.max()) + 1, h.device
    q, labels = q.to(device), labels.to(device)
    classes = J.row_classes(q, labels, k)
    allowed = labels[:, None] == classes[None, :]
    uniform = allowed.float() / allowed.sum(1, keepdim=True)
    initial = (1 - cfg.initial_uniform) * F.one_hot(q, k) + cfg.initial_uniform * uniform
    logits = torch.nn.Parameter(initial.clamp_min(1e-12).log())
    optimizer = torch.optim.Adam([logits], lr=cfg.lr)
    curve = []
    for step in range(cfg.steps + 1):
        p = (1 - cfg.floor) * logits.masked_fill(~allowed, -1e9).softmax(1) + cfg.floor * uniform
        loss, terms, coarse = objective(p, initial, h, a, old_n, cfg, mode, reconstruction_weight)
        assert torch.isfinite(loss)
        if step in (0, 1, 32, 64, cfg.steps):
            curve.append({"step": step, "loss": float(loss.detach()),
                          **{k: float(v.detach()) for k, v in terms.items()}})
        if step == cfg.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("nonfinite assignment gradient")
        optimizer.step()
    p = p.detach().cpu()
    normalized = p / p.sum(0)
    return p, coarse.detach().cpu(), {
        "curve": curve, "row_classes": classes.cpu().tolist(), "mode": mode,
        "reconstruction_weight": reconstruction_weight, "message_weight": cfg.message,
        "assignment_sha256": R.C.tensor_hash({"p": p}),
        "min_effective_sources": float((1 / normalized.square().sum(0)).min()),
        "teacher_surrogate_only": True, "report_labels_used": False}


def materialize(base, delta, p, coarse, encoder, cfg, p_topology=None, coarse_topology=None):
    candidates, audit = J.materialize(base, delta, p, coarse, encoder, cfg)
    memory = candidates["joint"]
    if p_topology is not None:
        memory["edge_index"] = J.edges_from_projection(base, len(base["x"]) - p.shape[1], coarse_topology)
        audit["content_and_topology_share_assignment"] = False
        audit["topology_assignment_sha256"] = R.C.tensor_hash({"p": p_topology})
    return memory, audit


def weighted_variant(memory, coarse):
    """Same selected nonself edges, projected weights, common GCN normalization."""
    result = T.clone_memory(memory)
    n, e = len(memory["x"]), memory["edge_index"]
    ids = torch.cat((e, torch.arange(n).repeat(2, 1)), 1)
    weights = coarse[ids[0], ids[1]].clamp_min(1e-9)
    degree = torch.zeros(n).scatter_add_(0, ids[0], weights)
    result.update(edge_index=ids, adj_values=weights * degree[ids[0]].rsqrt() * degree[ids[1]].rsqrt(),
                  adj_normalized=True)
    return result


def pooled_text_variant(memory, delta, p):
    result = T.clone_memory(memory)
    n = len(memory["x"]) - p.shape[1]
    result["text_emb"][n:] = F.normalize((p.T @ delta["text_emb"].float()) / p.sum(0)[:, None], dim=1)
    result["text_representation_scope"] = "embedding-only diagnostic; stored snippets do not generate these embeddings"
    return result


def gap(ctx, memory, p, device):
    if not ctx["models"]:
        return {"available": False}
    old_n = ctx["support"]["old_rows"]
    h = ctx["h"]
    u = J.full_assignment(p.to(device), old_n)
    z = u.T @ h / u.sum(0)[:, None]
    actual = encode_models(ctx["models"], memory, R.C.training_adjacency(memory, device), device)
    reference_message = torch.sparse.mm(symmetric_sparse(ctx["adjacency"]), h)
    actual_message = R.C.training_adjacency(memory, device) @ actual
    frac = old_n / len(memory["x"])

    def measure(v):
        return {"old": float(v[:old_n].mean()), "new": float(v[old_n:].mean()),
                "balanced": float(frac * v[:old_n].mean() + (1-frac) * v[old_n:].mean())}

    return {"available": True, "pooled_reconstruction": measure((h-u@z).square().sum(1)),
            "materialized_reconstruction": measure((h-u@actual).square().sum(1)),
            "materialized_symmetric_message": measure((reference_message-u@actual_message).square().sum(1)),
            "compact_teacher_vs_pooled_mse": float((actual-z).square().sum(1).mean()),
            "actual_representation_only": True}
