"""Teacher-space coarsening with one assignment for content and binary topology.

This is a compression surrogate, not an exact student-training hypergradient.
Only old synthetic memory and current raw data are accepted at this boundary.
"""
from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as F

import tgc_relational_response as M

R, T, S = M.R, M.T, M.S


@dataclass(frozen=True)
class Config:
    steps: int = 128
    lr: float = .05
    floor: float = .02
    initial_uniform: float = .1
    anchor: float = .05
    mass: float = .01
    message: float = 1.
    text_chars: int = 320
    text_sources: int = 5
    capacity_moves: int = 1


def raw_adjacency(support, device):
    n = len(support["x"])
    indices = torch.cat((support["edge_index"].cpu(), torch.arange(n).repeat(2, 1)), 1).to(device)
    values = torch.cat((support.get("edge_weight", torch.ones(support["edge_index"].shape[1])).cpu(),
                        torch.ones(n))).to(device)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def lift_soft(old, delta, compact, p):
    """Lift compact cross-links using the same soft source allocation."""
    old_n, k = len(old["x"]), p.shape[1]
    if p.shape != (len(delta["x"]), len(compact["x"]) - old_n):
        raise ValueError("assignment shape mismatch")
    torch.testing.assert_close(p.sum(1), torch.ones(len(p)), rtol=1e-5, atol=1e-6)
    edges = compact["edge_index"].cpu()
    oo = edges[:, (edges < old_n).all(0)]
    if not torch.equal(oo, old["edge_index"]):
        raise AssertionError("old edge block changed before lift")
    cross = edges[:, (edges[0] < old_n) & (edges[1] >= old_n)]
    matrix = torch.zeros(old_n, k)
    matrix[cross[0], cross[1] - old_n] = 1.
    weights = matrix @ (p / p.sum(0).clamp_min(1e-12)).T
    i, j = torch.where(weights > 0)
    forward = torch.stack((i, j + old_n))
    source_edges = delta["edge_index"].cpu() + old_n
    result = T._concat_support(old, delta)
    result["edge_index"] = torch.cat((oo, source_edges, forward, forward.flip(0)), 1)
    result["edge_weight"] = torch.cat((torch.ones(oo.shape[1] + source_edges.shape[1]),
                                       weights[i, j], weights[i, j]))
    return result


def teacher_views(support, compact, cfg, device):
    views, audits = [], []
    x, text, adj = support["x"].to(device), support["text_emb"].to(device), M.adjacency(support, device)
    for seed in cfg.teacher_seeds:
        model, audit = M.fit_teacher(support, compact, seed, "compact", cfg, device)
        with torch.no_grad():
            params = dict(model.named_parameters())
            views.append(S.graph(params, x, adj))
            views.append(F.normalize(F.linear(text, params["text_proj.weight"], params["text_proj.bias"]), dim=1))
        audits.append(audit)
        del model
    return torch.cat(views, 1).detach() / math.sqrt(len(views)), audits


def row_classes(assignment, labels, k):
    classes = []
    for j in range(k):
        values = labels[assignment == j].unique()
        if len(values) != 1:
            raise ValueError("empty or mixed-class initial cluster")
        classes.append(int(values[0]))
    return torch.tensor(classes, device=assignment.device)


def full_assignment(p, old_n):
    return torch.cat((F.pad(torch.eye(old_n, device=p.device, dtype=p.dtype), (0, p.shape[1])),
                      F.pad(p, (old_n, 0))), 0)


def objective(p, initial, h, adjacency, old_n, cfg):
    u = full_assignment(p, old_n)
    mass = u.sum(0).clamp_min(1e-9)
    z = u.T @ h / mass[:, None]
    coarse = u.T @ torch.sparse.mm(adjacency, u)
    transition = coarse / coarse.sum(1, keepdim=True).clamp_min(1e-9)
    degree = torch.sparse.sum(adjacency, 1).to_dense().clamp_min(1e-9)
    message = torch.sparse.mm(adjacency, h) / degree[:, None]
    reconstruction = (h - u @ z).square().sum(1)
    propagation = (message - u @ (transition @ z)).square().sum(1)
    old_fraction = old_n / u.shape[1]

    def balanced(value):
        return old_fraction * value[:old_n].mean() + (1 - old_fraction) * value[old_n:].mean()

    recon = balanced(reconstruction)
    msg = balanced(propagation)
    anchor = (p * (p.clamp_min(1e-12).log() - initial.clamp_min(1e-12).log())).sum(1).mean()
    balance = (p.sum(0).clamp_min(1e-9).log() - initial.sum(0).log()).square().mean()
    value = recon + cfg.message * msg + cfg.anchor * anchor + cfg.mass * balance
    return value, {"reconstruction": recon, "message": msg, "anchor": anchor, "mass": balance}, coarse


def learn(assignment, labels, h, adjacency, old_n, cfg=Config()):
    k, device = int(assignment.max()) + 1, h.device
    assignment, labels = assignment.to(device), labels.to(device)
    classes = row_classes(assignment, labels, k)
    allowed = labels[:, None] == classes[None, :]
    uniform = allowed.to(h.dtype) / allowed.sum(1, keepdim=True)
    initial = (1 - cfg.initial_uniform) * F.one_hot(assignment, k) + cfg.initial_uniform * uniform
    logits = torch.nn.Parameter(initial.clamp_min(1e-12).log())
    optimizer = torch.optim.Adam([logits], lr=cfg.lr)
    curve = []

    def distribution():
        soft = logits.masked_fill(~allowed, -1e9).softmax(1)
        return (1 - cfg.floor) * soft + cfg.floor * uniform

    for step in range(cfg.steps + 1):
        p = distribution()
        loss, terms, coarse = objective(p, initial, h, adjacency, old_n, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite assignment objective")
        if step in (0, 1, 32, 64, cfg.steps):
            curve.append({"step": step, "loss": float(loss.detach()),
                          **{key: float(value.detach()) for key, value in terms.items()}})
        if step == cfg.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("assignment gradient missing/nonfinite")
        optimizer.step()
    p = p.detach()
    assert torch.count_nonzero(p[~allowed]) == 0
    assert (p.sum(0) > 0).all()
    torch.testing.assert_close(p.sum(1), torch.ones(len(p), device=device), atol=2e-6, rtol=1e-5)
    normalized = p / p.sum(0)
    audit = {"curve": curve, "config": asdict(cfg), "row_classes": classes.cpu().tolist(),
             "mass": p.sum(0).cpu().tolist(), "min_effective_sources": float((1 / normalized.square().sum(0)).min()),
             "assignment_sha256": R.C.tensor_hash({"p": p.cpu()}),
             "teacher_surrogate_only": True, "report_labels_used": False}
    return p.cpu(), coarse.detach().cpu(), audit


def transfer_initial(q, labels, h, donor, receiver, device):
    """Merge closest donor clusters; split highest-error receiver cluster."""
    groups = {int(c): [torch.where(q == j)[0] for j in q[labels == c].unique().tolist()]
              for c in labels.unique().tolist()}
    source = groups[donor]
    centers = torch.stack([h[ids].mean(0) for ids in source])
    pairs = torch.triu_indices(len(source), len(source), 1)
    cost = (centers[pairs[0]] - centers[pairs[1]]).square().sum(1)
    a, b = pairs[:, int(cost.argmin())].tolist()
    source[a] = torch.cat((source[a], source[b])).sort().values
    del source[b]
    recipient = groups[receiver]
    possible = [i for i, ids in enumerate(recipient) if len(ids) > 1]
    if not possible:
        raise ValueError("no splittable receiver cluster")
    split = max(possible, key=lambda i: (float((h[recipient[i]] - h[recipient[i]].mean(0)).square().sum()), -i))
    ids = recipient[split]
    local = R.SCA.cluster_assign(h[ids].cpu(), torch.ones(len(ids)), 2, device=device).cpu()
    if len(local.unique()) != 2:
        raise ValueError("empty split cluster")
    recipient[split:split+1] = [ids[local == 0], ids[local == 1]]
    updated, at = torch.full_like(q, -1), 0
    for cls in sorted(groups):
        for ids in groups[cls]:
            updated[ids] = at
            at += 1
    assert (updated >= 0).all() and at == int(q.max()) + 1
    return updated


def capacity_search(q, labels, h, adjacency, old_n, cfg, device, baseline):
    """One class-row exchange, chosen by unregularized teacher distortion."""
    classes = row_classes(q, labels, int(q.max()) + 1)
    allocation = {int(c): int((classes == c).sum()) for c in classes.unique()}
    best, selected = baseline, allocation
    score = lambda a: a["curve"][-1]["reconstruction"] + cfg.message * a["curve"][-1]["message"]
    best_score = score(baseline[2])
    trials = [{"allocation": allocation, "score": best_score, "source": "unchanged"}]
    if cfg.capacity_moves != 1:
        raise ValueError("minimal capacity experiment registers one row exchange")
    for donor in sorted(allocation):
        for receiver in sorted(allocation):
            if donor == receiver or allocation[donor] <= 1:
                continue
            if allocation[receiver] >= int((labels == receiver).sum()):
                continue
            proposed = dict(allocation)
            proposed[donor] -= cfg.capacity_moves
            proposed[receiver] += cfg.capacity_moves
            init = transfer_initial(q, labels, h[old_n:].cpu(), donor, receiver, device)
            fit = learn(init, labels, h, adjacency, old_n, cfg)
            value = score(fit[2])
            trials.append({"allocation": proposed, "score": value, "source": "one_merge_one_split"})
            if value < best_score - 1e-8:
                best, selected, best_score = fit, proposed, value
    return best, {"trials": trials, "original_allocation": allocation, "selected_allocation": selected,
                   "total_rows": len(classes), "rows_transferred": sum(abs(selected[c] - allocation[c]) for c in allocation) // 2,
                   "score": "reconstruction + message, excludes allocation-dependent regularizers",
                   "initialization": "unchanged clusters retained; donor merge and receiver split only"}


def edges_from_projection(base, old_n, coarse):
    edges = base["edge_index"].cpu()
    old_edges = edges[:, (edges < old_n).all(0)]
    new_n = len(base["x"]) - old_n
    mass = coarse.sum(1).clamp_min(1e-9)
    score = coarse / torch.sqrt(mass[:, None] * mass[None, :])
    pairs = torch.triu_indices(new_n, new_n, 1) + old_n
    count = int(((edges[0] >= old_n) & (edges[1] >= old_n)).sum()) // 2
    rank = torch.argsort(score[pairs[0], pairs[1]], descending=True, stable=True)[:count]
    internal = pairs[:, rank]
    new_ids = torch.arange(old_n, len(base["x"]))
    nearest = score[:old_n, old_n:].argmax(0)
    cross = torch.stack((nearest, new_ids))
    result = torch.cat((old_edges, internal, internal.flip(0), cross, cross.flip(0)), 1)
    assert result.shape == edges.shape
    return result


def materialize(base, delta, p, coarse, encoder, cfg=Config()):
    old_n = len(base["x"]) - p.shape[1]
    content, topology = T.clone_memory(base), T.clone_memory(base)
    mass = p.sum(0)
    content["x"][old_n:] = (p.T @ delta["x"].float()) / mass[:, None]
    content["mass"][old_n:] = mass
    texts, provenance = [], []
    for row in range(p.shape[1]):
        members = torch.where(p[:, row] > 0)[0]
        chosen = members[torch.argsort(p[members, row], descending=True, stable=True)[:cfg.text_sources]]
        per = max(1, cfg.text_chars // len(chosen))
        text = " ".join(str(delta["texts"][int(i)])[:per] for i in chosen)[:cfg.text_chars]
        texts.append(text)
        provenance.append(chosen.tolist())
    content["texts"][old_n:] = texts
    content["text_emb"][old_n:] = encoder.encode(texts).cpu()
    topology["edge_index"] = edges_from_projection(base, old_n, coarse)
    joint = T.clone_memory(content)
    joint["edge_index"] = topology["edge_index"].clone()
    for candidate in (content, topology, joint):
        assert len(candidate["x"]) == len(base["x"])
        assert candidate["edge_index"].shape == base["edge_index"].shape
        assert torch.equal(candidate["x"][:old_n], base["x"][:old_n])
        assert candidate["texts"][:old_n] == base["texts"][:old_n]
    audit = {"selected_delta_local_text_ids": provenance,
             "text": R.SCA.text_audit(texts, getattr(encoder, "tokenizer", None), "shared_assignment", "g5_c320"),
             "new_x_relative_displacement_mean": float(((joint["x"][old_n:] - base["x"][old_n:]).norm(dim=1)
                                                         / base["x"][old_n:].norm(dim=1).clamp_min(1e-6)).mean()),
             "content_and_topology_share_assignment": True}
    return {"content": content, "topology": topology, "joint": joint}, audit


def construct(old, delta, base, q, labels, encoder, teacher_cfg, device, cfg=Config(), capacity=False):
    R.sync(device)
    started = time.perf_counter()
    source_hashes = {"old": T.memory_hash(old), "delta": T.memory_hash(delta), "base": T.memory_hash(base)}
    support = M.lifted_support(old, delta, base, q)
    h, teachers = teacher_views(support, base, teacher_cfg, device)
    adjacency = raw_adjacency(support, device)
    fit = learn(q, labels, h, adjacency, len(old["x"]), cfg)
    cap_audit = None
    if capacity:
        fit, cap_audit = capacity_search(q, labels, h, adjacency, len(old["x"]), cfg, device, fit)
    p, coarse, assignment_audit = fit
    candidates, material = materialize(base, delta, p, coarse, encoder, cfg)
    R.sync(device)
    assert source_hashes == {"old": T.memory_hash(old), "delta": T.memory_hash(delta), "base": T.memory_hash(base)}
    return candidates, p, {"teachers": teachers, "assignment": assignment_audit, "capacity": cap_audit,
                            "materialization": material, "source_hashes": source_hashes,
                            "seconds": time.perf_counter() - started, "historical_raw_reads": 0,
                            "report_labels_used": False, "current_delta_ids": delta["current_delta_ids"].tolist()}
