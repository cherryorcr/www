"""Anchored text attention with a genuine gradient through probe-student Adam."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

import run_p2a_graph_conditioned_text as A


class AnchoredAttention(torch.nn.Module):
    def __init__(self, dim=384, rank=16, seed=700):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.wq = torch.nn.Parameter(torch.zeros(rank, dim))
        self.wk = torch.nn.Parameter(torch.randn(rank, dim, generator=gen))

    def forward(self, keys, mask, query, anchor):
        base = torch.einsum("rkd,rd->rk", keys, query) / .2
        correction = torch.einsum("rkh,rh->rk", F.linear(keys, self.wk),
                                  F.linear(query, self.wq)) / math.sqrt(len(self.wq))
        scores = (base + 2 * torch.tanh(correction)).masked_fill(~mask, -torch.inf)
        weights = scores.softmax(1)
        pooled = (keys * weights[..., None]).sum(1)
        return F.normalize(.75 * anchor + .25 * pooled, dim=1), weights


def graph(params, x, adj):
    h = F.relu(adj @ F.linear(x, params["g1.lin.weight"], params["g1.lin.bias"]))
    return F.normalize(adj @ F.linear(h, params["g2.lin.weight"], params["g2.lin.bias"]), dim=1)


def contrastive(params, x, adj, text, rows):
    g = graph(params, x, adj)[rows]
    t = F.normalize(F.linear(text[rows], params["text_proj.weight"],
                            params["text_proj.bias"]), dim=1)
    logits = params["logit_scale"].exp().clamp(max=100) * g @ t.T
    target = torch.arange(len(rows), device=x.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def sqrt_zero_safe(value):
    # Adam's forward is unchanged at zero; choose a finite derivative there.
    return torch.where(value > 0, value.clamp_min(1e-30).sqrt(), torch.zeros_like(value))


def adam_update(params, gradients, first, second, step, lr=.001):
    new, m, v = {}, {}, {}
    for (name, param), grad in zip(params.items(), gradients):
        m[name] = torch.lerp(first[name], grad, .1)
        v[name] = second[name] * .999 + grad.square() * .001
        denom = sqrt_zero_safe(v[name]) / math.sqrt(1 - .999 ** step) + 1e-8
        new[name] = torch.addcdiv(param, m[name], denom, value=-lr / (1 - .9 ** step))
    return new, m, v


def unroll(x, adj, targets, seed, steps, differentiable=True):
    # Fork RNG so probe initialization cannot change another arm's state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = A.G.CLIPStudent(x.shape[1]).to(x.device)
    params = dict(model.named_parameters())
    first = {k: torch.zeros_like(v) for k, v in params.items()}
    second = {k: torch.zeros_like(v) for k, v in params.items()}
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    for step in range(1, steps + 1):
        rows = torch.randperm(len(x), generator=gen)[:min(16, len(x))].to(x.device)
        loss = contrastive(params, x, adj, targets, rows)
        grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=differentiable)
        params, first, second = adam_update(params, grads, first, second, step)
        if not differentiable:
            params = {k: v.detach().requires_grad_() for k, v in params.items()}
            first = {k: v.detach() for k, v in first.items()}
            second = {k: v.detach() for k, v in second.items()}
    return params


def source_split(count, seed, stage):
    if count < 10:
        raise ValueError("insufficient current source rows")
    gen = torch.Generator().manual_seed(seed * 1009 + stage)
    order = torch.randperm(count, generator=gen)
    n = max(2, count // 5)
    return order[n:], order[:n]


def batch(ids, gen, device):
    return ids[torch.randperm(len(ids), generator=gen)[:min(16, len(ids))]].to(device)


def semantic_targets(source_text, clusters, ids, anchor):
    mask = torch.zeros(len(source_text), dtype=torch.bool)
    mask[ids] = True
    targets, valid = [], []
    for i, members in enumerate(clusters):
        subset = members[mask[members]]
        valid.append(bool(len(subset)))
        targets.append(F.normalize(source_text[subset].mean(0), dim=0)
                       if len(subset) else anchor[i].cpu())
    return torch.stack(targets).to(anchor.device), torch.tensor(valid, device=anchor.device)


def prepare_evidence(source, joint, clusters, encoder, device, stage):
    started = time.perf_counter()
    texts, anchors, parts, audits = [], [], [], []
    for members in clusters:
        chosen = A.BT.select_members(joint, source["text_emb"], members, 5, False, .5)
        snippets = A.BT.bounded_parts(encoder.tokenizer,
            [source["texts"][int(i)] for i in chosen], 64)
        fused = " ".join(p for p in snippets if p)
        texts.append(fused)
        parts.append(snippets)
        audits.append(dict(selected_local_ids=chosen.tolist(), inputs=snippets,
            retained_tokens=A.BT.token_count(encoder.tokenizer, fused),
            part_tokens=sum(A.BT.token_count(encoder.tokenizer, p) for p in snippets),
            tokens_with_special=sum(len(encoder.tokenizer.encode(p, add_special_tokens=True))
                for p in [fused] + snippets)))
    # Preserve P2-A's per-row encoding batches for the shared concat control.
    anchors = torch.stack([encoder.encode([text])[0] for text in texts])
    concat_seconds = time.perf_counter() - started
    flat = encoder.encode([p for group in parts for p in group])
    keys = torch.zeros(len(clusters), 5, flat.shape[1])
    mask = torch.zeros(len(clusters), 5, dtype=torch.bool)
    offset = 0
    for i, group in enumerate(parts):
        keys[i, :len(group)] = F.normalize(flat[offset:offset + len(group)], dim=1)
        mask[i, :len(group)] = True
        offset += len(group)
    query_start = time.perf_counter()
    queries = A.T.make_queries(source["x"], source["text_emb"], clusters,
                              source["edge_index"], 9001 + 1009 * stage)
    A.B.sync(device)
    if any(max(r["retained_tokens"], r["part_tokens"]) > 64 for r in audits):
        raise AssertionError("token cap violated")
    audit = dict(rows=audits, concat_seconds=concat_seconds,
        evidence_seconds=time.perf_counter() - started,
        query_seconds=time.perf_counter() - query_start,
        encoder_seconds=encoder.seconds, encoder_sequences=encoder.sequences,
        query_stats=queries["stats"], budget_violations=0)
    tensors = dict(keys=keys.to(device), mask=mask.to(device),
                   query=queries["graph"].to(device), anchor=anchors.to(device))
    return texts, tensors, audit


def train_attention(mode, seed, stage, memory, old_n, source, clusters, evidence,
                    device, outer_steps=40, inner_steps=16, validation_steps=32):
    if mode not in {"semantic", "student"}:
        raise ValueError(mode)
    A.B.sync(device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model = AnchoredAttention(seed=seed).to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    train_ids, val_ids = source_split(len(source["x"]), seed, stage)
    x, old_t = memory["x"].to(device), memory["text_emb"][:old_n].detach().to(device)
    adj = A.C.training_adjacency(memory, device)
    rx, rt = source["x"].to(device), source["text_emb"].detach().to(device)
    radj = A.G.norm_adj(source["edge_index"], len(rx), device)
    sem_train, valid_train = semantic_targets(source["text_emb"], clusters, train_ids, evidence["anchor"])
    sem_val, valid_val = semantic_targets(source["text_emb"], clusters, val_ids, evidence["anchor"])
    if not valid_train.any() or not valid_val.any():
        raise ValueError("empty semantic objective")
    train_gen = torch.Generator().manual_seed(seed * 31 + stage)
    val_gen = torch.Generator().manual_seed(999 + seed * 31 + stage)
    val_batches = [batch(val_ids, val_gen, device) for _ in range(4)]
    old_rows = torch.arange(min(16, old_n), device=device)
    curve, best_state, best_step, best_value = [], initial_state, 0, float("inf")
    initial_value = None

    def objective(target, step, validation):
        anchor_loss = (1 - (target * evidence["anchor"]).sum(1)).mean()
        if mode == "semantic":
            goal, mask = (sem_val, valid_val) if validation else (sem_train, valid_train)
            real = (1 - (target[mask] * goal[mask]).sum(1)).mean()
            old = target.new_zeros(())
        else:
            all_t = torch.cat((old_t, target))
            params = unroll(x, adj, all_t, 310 if validation else (300 + (step - 1) % 3),
                validation_steps if validation else inner_steps, differentiable=not validation)
            batches = val_batches if validation else [batch(train_ids, train_gen, device)]
            real = torch.stack([contrastive(params, rx, radj, rt, rows) for rows in batches]).mean()
            old = contrastive(params, x, adj, all_t, old_rows) if old_n >= 2 else target.new_zeros(())
        return real + .25 * old + .1 * anchor_loss, dict(real=float(real.detach()),
            old=float(old.detach()), anchor=float(anchor_loss.detach()))

    for step in range(outer_steps + 1):
        if step:
            opt.zero_grad(set_to_none=True)
            target, _ = model(**evidence)
            loss, details = objective(target, step, False)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("missing/nonfinite attention gradients")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            curve.append(dict(step=step, train=float(loss.detach()), gradient_norm=float(norm), **details))
        if step % 5 == 0 or step == outer_steps:
            with torch.no_grad():
                target, _ = model(**evidence)
            value, details = objective(target.detach(), step, True)
            value = float(value.detach())
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite selection loss")
            if initial_value is None:
                initial_value = value
            curve.append(dict(step=step, validation=value, **details))
            if value < best_value:
                best_value, best_step = value, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        target, weights = model(**evidence)
    A.B.sync(device)
    audit = dict(mode=mode, seed=seed, seconds=time.perf_counter() - started,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20,
        train_local_ids=train_ids.tolist(), validation_local_ids=val_ids.tolist(),
        selected_step=best_step, initial_validation=initial_value, selected_validation=best_value,
        nonzero_gradient_steps=sum(r.get("gradient_norm", 0) > 0 for r in curve),
        initial_parameter_sha256=A.C.tensor_hash(initial_state),
        selected_parameter_sha256=A.C.tensor_hash(best_state),
        max_target_cosine_drift=float((1 - (target * evidence["anchor"]).sum(1)).max()),
        weights=weights.cpu().tolist(), parameter_count=sum(p.numel() for p in model.parameters()),
        curve=curve, historical_raw_reads=0, report_labels_used=False,
        unroll_optimizer="Adam numeric-compatible; zero sqrt derivative convention",
        train_probe_steps=outer_steps * inner_steps if mode == "student" else 0,
        validation_probe_steps=(sum("validation" in r for r in curve) * validation_steps
                                if mode == "student" else 0))
    return target.cpu(), best_state, audit
