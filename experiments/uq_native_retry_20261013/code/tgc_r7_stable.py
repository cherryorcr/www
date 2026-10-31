"""Small-graph deterministic construction and actual-output response probes.

Dense adjacency is a construction diagnostic, not a scalable graph claim.
The canonical downstream student is not modified.
"""
from dataclasses import asdict
import math
import time

import torch
import torch.nn.functional as F
import tgc_r7_numerics as N

J, R, T, M, K = N.J, N.R, N.T, N.M, N.K
S = J.S


def dense_context(old, delta, base, q, teacher_cfg, device, trace=False, keep_probes=True):
    started = time.perf_counter()
    support = M.lifted_support(old, delta, base, q)
    raw = J.raw_adjacency(support, "cpu").to_dense().to(device)
    adj = M.adjacency(support, "cpu").to_dense().to(device)
    x, text = support["x"].to(device), support["text_emb"].to(device)
    models, audits, probes, parts = [], [], [], []
    for seed in teacher_cfg.teacher_seeds:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        gen = torch.Generator().manual_seed(seed * 7 + 1)
        sequence, states = [], []
        for step in range(teacher_cfg.teacher_steps + 1):
            if trace:
                states.append({"step": step, "state": N.fingerprint(model.state_dict())})
            if keep_probes and step in (0, 32, teacher_cfg.teacher_steps):
                probes.append({"seed": seed, "step": step,
                               "params": {k: v.detach().clone().requires_grad_() for k, v in model.named_parameters()}})
            if step == teacher_cfg.teacher_steps:
                break
            ids = M.sample_rows(len(x), len(old["x"]), teacher_cfg.teacher_batch, gen,
                                "compact", len(old["x"]) / len(base["x"]))
            sequence.append(ids)
            loss = T._info_nce(model, x, text, adj, ids.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        models.append(model)
        audits.append({"seed": seed, "steps": teacher_cfg.teacher_steps, "trace": states,
                       "state_sha256": N.fingerprint(model.state_dict()),
                       "batch_sequence_hash": N.fingerprint({"ids": torch.stack(sequence)})})
        with torch.no_grad():
            parts.extend((model.encode_graph(x, adj), model.encode_text(text)))
    R.sync(device)
    return {"support": support, "adjacency": raw, "normalized": adj,
            "h": torch.cat(parts, 1).detach() / math.sqrt(len(parts)), "models": models, "probes": probes,
            "audit": {"teachers": audits, "support_hash": T.memory_hash(support), "q_hash": N.fingerprint(q),
                      "adjacency_hash": N.fingerprint(adj), "parent_hash": T.memory_hash(old),
                      "current_delta_ids": delta["current_delta_ids"].tolist(), "seconds": time.perf_counter()-started}}


def initial_assignment(q, labels, cfg, dtype, device):
    k = int(q.max()) + 1
    q, labels = q.to(device), labels.to(device)
    classes = J.row_classes(q, labels, k)
    allowed = labels[:, None] == classes[None, :]
    uniform = allowed.to(dtype) / allowed.sum(1, keepdim=True)
    initial = (1 - cfg.initial_uniform) * F.one_hot(q, k) + cfg.initial_uniform * uniform
    return initial, uniform, allowed, classes


def dense_objective(p, initial, h, adjacency, old_n, cfg):
    u = J.full_assignment(p, old_n)
    mass = u.sum(0).clamp_min(1e-9)
    z = u.T @ h / mass[:, None]
    coarse = u.T @ (adjacency @ u)
    transition = coarse / coarse.sum(1, keepdim=True).clamp_min(1e-9)
    message = (adjacency @ h) / adjacency.sum(1).clamp_min(1e-9)[:, None]
    fraction = old_n / u.shape[1]

    def balanced(v):
        return fraction * v[:old_n].mean() + (1 - fraction) * v[old_n:].mean()

    recon = balanced((h - u @ z).square().sum(1))
    msg = balanced((message - u @ (transition @ z)).square().sum(1))
    anchor = (p * (p.clamp_min(1e-12).log() - initial.clamp_min(1e-12).log())).sum(1).mean()
    balance = (p.sum(0).clamp_min(1e-9).log() - initial.sum(0).log()).square().mean()
    return recon + cfg.message * msg + cfg.anchor * anchor + cfg.mass * balance, {
        "reconstruction": recon, "message": msg, "anchor": anchor, "mass": balance}, coarse


def learn(ctx, q, labels, old_n, cfg):
    h, a = ctx["h"], ctx["adjacency"]
    initial, uniform, allowed, classes = initial_assignment(q, labels, cfg, h.dtype, h.device)
    logits = torch.nn.Parameter(initial.clamp_min(1e-12).log())
    opt = torch.optim.Adam([logits], lr=cfg.lr)
    curve = []
    for step in range(cfg.steps + 1):
        p = (1 - cfg.floor) * logits.masked_fill(~allowed, -1e9).softmax(1) + cfg.floor * uniform
        loss, terms, coarse = dense_objective(p, initial, h, a, old_n, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("assignment loss is nonfinite")
        if step in (0, 1, 32, 64, cfg.steps):
            curve.append({"step": step, "loss": float(loss.detach()), "p_hash": N.fingerprint(p),
                          **{k: float(v.detach()) for k, v in terms.items()}})
        if step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("assignment gradient is nonfinite")
        opt.step()
    p0 = (1 - cfg.floor) * initial + cfg.floor * uniform
    return p.detach().cpu(), p0.detach().cpu(), {"curve": curve, "config": asdict(cfg),
        "assignment_sha256": R.C.tensor_hash({"p": p.detach().cpu()}), "row_classes": classes.cpu().tolist(),
        "teacher_surrogate_only": True, "report_labels_used": False}


def materialize(ctx, base, delta, p, encoder, cfg):
    u = J.full_assignment(p.to(ctx["h"].device), len(base["x"]) - p.shape[1])
    coarse = (u.T @ (ctx["adjacency"] @ u)).detach().cpu()
    memory, audit = K.materialize(base, delta, p, coarse, encoder, cfg)
    weights = p / p.sum(0)
    audit.update(min_effective_sources=float((1 / weights.square().sum(0)).min()),
                 max_source_weight=float(weights.max()))
    return memory, audit


def response_plan(q, old_n, rows, seed=8650, batches=4):
    gen = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(batches):
        compact = M.sample_rows(rows, old_n, 16, gen, "compact", old_n / rows)
        source = []
        for row in compact.tolist():
            if row < old_n:
                source.append(row)
            else:
                members = torch.where(q == row-old_n)[0]
                source.append(old_n + int(members[torch.randint(len(members), (), generator=gen)]))
        assert len(set(source)) == len(source)
        result.append((compact, torch.tensor(source)))
    return result


def response_targets(ctx, q, rows):
    support = ctx["support"]
    device = ctx["h"].device
    x, text = support["x"].to(device), support["text_emb"].to(device)
    plans = response_plan(q, support["old_rows"], rows)
    targets = []
    for probe in ctx["probes"]:
        params = probe["params"]
        for compact, source in plans:
            source = source.to(device)
            loss = S.contrastive(params, x, ctx["normalized"], text, source)
            grads = torch.autograd.grad(loss, tuple(params.values()))
            with torch.no_grad():
                logits = M.logits(params, x, ctx["normalized"], text, source, .07)
            targets.append({"params": params, "gradients": [v.detach() for v in grads],
                            "logits": logits.detach(), "source_loss": float(loss.detach()),
                            "compact": compact.to(device), "source": source,
                            "step": probe["step"], "seed": probe["seed"]})
    return targets


def response_metrics(ctx, memory, targets):
    device = ctx["h"].device
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = R.C.training_adjacency(memory, "cpu").to_dense().to(device)
    sx, st = (ctx["support"][k].to(device) for k in ("x", "text_emb"))
    records = []
    for target in targets:
        params, rows = target["params"], target["compact"]
        loss = S.contrastive(params, x, adj, text, rows)
        grads = torch.autograd.grad(loss, tuple(params.values()))
        cosine = torch.stack([F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
                              for a, b in zip(grads, target["gradients"]) if a.numel() > 1]).mean()
        # One fresh-Adam update at fixed weights; not a resumed teacher trajectory.
        updated = {k: v.detach() - .001 * g.detach() / (g.detach().abs() + 1e-8)
                   for (k, v), g in zip(params.items(), grads)}
        with torch.no_grad():
            after = S.contrastive(updated, sx, ctx["normalized"], st, target["source"])
            logits = M.logits(params, x, adj, text, rows, .07)
            relation = sum(F.kl_div(a.log_softmax(1), b.softmax(1), reduction="batchmean")
                           for a, b in ((logits, target["logits"]), (logits.T, target["logits"].T))) / 2
        records.append({"step": target["step"], "seed": target["seed"], "gradient_cosine": float(cosine),
                        "relation_kl": float(relation), "legal_one_step_gain": target["source_loss"]-float(after)})
    return {"records": records, **{key: sum(v[key] for v in records) / len(records)
            for key in ("gradient_cosine", "relation_kl", "legal_one_step_gain")},
            "scope": "matched legal-source/compact batches; teacher checkpoints0/32/128; no report labels; diagnostic only"}
