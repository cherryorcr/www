"""Explicit tied/untied assignment experiment, not a new primary method.

The shared branch is tested against the frozen production constructor. The
untied branch changes only parameter sharing: Qx builds X/messages, Qt routes
real text. Both retain the class/capacity constraints and the same targets.
"""
from dataclasses import asdict

import torch

import unified_q_multimodal as U


def optimize(delta, parent, labels, encoder, prompt_emb, cfg, device, shared=False):
    if (cfg.old_update_radius or cfg.materialize_top_k != 1 or not cfg.train_q
            or not cfg.train_residual or cfg.optimization != "joint"
            or cfg.text_assignment != "learned" or not cfg.straight_through_text):
        raise ValueError("sharing experiment requires the frozen joint edge-free configuration")
    labels = labels.long().cpu()
    raw_labels = delta["y"].long().to(device)
    candidate_texts = [U._append_text(t, cfg.text_chars) for t in delta["texts"]]
    raw_text = U._norm(torch.as_tensor(encoder.encode(candidate_texts)).float().to(device))
    delta = dict(delta, text_emb=raw_text.detach().cpu())
    delta["_raw_adjacency"] = U._sparse_adjacency(delta["edge_index"], len(delta["x"]), device)
    raw_x = delta["x"].float().to(device)
    old_n = 0 if parent is None else len(parent["x"])
    boundary = torch.zeros((len(raw_x), old_n), device=device)
    logits0, q0, init_audit = U._initial_logits(delta, parent, labels, cfg, device, boundary)
    pooled0 = q0.T @ raw_x / q0.sum(0).clamp_min(1e-6)[:, None]
    zx = torch.nn.Parameter(logits0.clone())
    zt = zx if shared else torch.nn.Parameter(logits0.clone())
    residual = torch.nn.Parameter(torch.zeros_like(pooled0))
    parameters = [zx, residual] if shared else [zx, residual, zt]
    groups = [{"params": [zx], "lr": cfg.lr_q}, {"params": [residual], "lr": cfg.lr_x}]
    if not shared:
        groups.append({"params": [zt], "lr": cfg.lr_q})
    optimizer = torch.optim.Adam(groups)
    old_x = parent["x"].float().to(device) if parent else raw_x.new_empty((0, raw_x.shape[1]))
    old_text = parent["text_emb"].float().to(device) if parent else raw_text.new_empty((0, raw_text.shape[1]))
    old_labels = parent["labels"].long().to(device) if parent else raw_labels.new_empty(0)
    ref_adj, ref_x = U._reference_graph(delta, parent, boundary, device)
    ref_text = torch.cat((old_text, raw_text))
    ref_labels = torch.cat((old_labels, raw_labels))
    full_labels = torch.cat((old_labels, labels.to(device)))
    ref_groups = torch.cat((torch.zeros(old_n), torch.ones(len(raw_labels)))).long().to(device)
    syn_groups = torch.cat((torch.zeros(old_n), torch.ones(len(labels)))).long().to(device)
    target_message = U._adjacency_mm(ref_adj, ref_x).detach()[-len(raw_labels):]
    syn_adj = U._hard_adjacency(torch.empty((2, 0), dtype=torch.long), parent, old_n + len(labels), device)
    targets = []
    if cfg.lambda_distribution:
        bank_seed = cfg.seed + 1009 * int(delta["stage"]) + 999983
        targets = [U._distribution_target(target_message, raw_labels, cfg, bank_seed + off) for off in (0, 1)]
    prompt_emb = prompt_emb.float().to(device)
    models, real_grads = [], []
    for seed in (cfg.proxy_seeds if cfg.lambda_gradient else ()):
        torch.manual_seed(cfg.seed + seed)
        model = U.P.GraphTextPromptStudent(ref_x.shape[1], hidden=cfg.proxy_hidden,
            out=cfg.proxy_out, text_dim=ref_text.shape[1]).to(device)
        real_loss = U._loss(model, dict(x=ref_x, text_emb=ref_text, labels=ref_labels),
            ref_adj, prompt_emb, group_ids=ref_groups, loss_weighting=cfg.loss_weighting,
            chunk_size=cfg.pair_chunk_size, pair_weight=cfg.pair_weight)
        real_grads.append(tuple(g.detach() for g in torch.autograd.grad(real_loss, tuple(model.parameters()))))
        models.append(model)
    sinkhorn_x, sinkhorn_t = [], []

    def assignment(logits, audit):
        return U._masked_sinkhorn(logits, raw_labels, labels.to(device), cfg.temperature,
            cfg.sinkhorn_steps, cfg.sinkhorn_tolerance, cfg.sinkhorn_max_steps, audit)

    with torch.no_grad():
        initial_message = (syn_adj @ torch.cat((old_x, pooled0)))[-len(labels):]
        initial_dm = [float(U._distribution_loss(initial_message, labels, t)) for t in targets]
    trace = []
    for step in range(cfg.steps):
        qx = assignment(zx, sinkhorn_x)
        qt = qx if shared else assignment(zt, sinkhorn_t)
        mass = qx.sum(0).clamp_min(1e-6)
        x_new = qx.T @ raw_x / mass[:, None] + residual
        soft_text = U._norm(qt.T @ raw_text / qt.sum(0).clamp_min(1e-6)[:, None])
        hard_text = raw_text[qt.argmax(0)]
        text_new = hard_text + (soft_text - soft_text.detach())
        full_x = torch.cat((old_x, x_new))
        synthetic = dict(x=full_x, text_emb=torch.cat((old_text, text_new)), labels=full_labels)
        terms = []
        for model, real_grad in zip(models, real_grads):
            syn_loss = U._loss(model, synthetic, syn_adj, prompt_emb, group_ids=syn_groups,
                loss_weighting=cfg.loss_weighting, chunk_size=cfg.pair_chunk_size, pair_weight=cfg.pair_weight)
            syn_grad = torch.autograd.grad(syn_loss, tuple(model.parameters()), create_graph=True, retain_graph=True)
            terms.append(U._gradient_distance(U._flat_grads(real_grad), U._flat_grads(syn_grad)))
        gm = torch.stack(terms).mean() if terms else x_new.sum() * 0.
        target_new = U._norm(qx.T @ target_message / mass[:, None])
        message = (syn_adj @ full_x)[-len(labels):]
        distribution = U._distribution_loss(message, labels, targets[0]) if targets else gm * 0.
        normalized = U._norm(message)
        propagation = ((target_new - normalized).square().mean()
                       if cfg.propagation_metric == "dimension_mean" else
                       (1 - (target_new * normalized).sum(1)).mean())
        q_anchor_x = (qx - q0.detach()).square().mean()
        q_anchor_t = (qt - q0.detach()).square().mean()
        q_anchor = q_anchor_x if shared else .5 * (q_anchor_x + q_anchor_t)
        x_anchor = (residual / pooled0.norm(dim=1, keepdim=True).clamp_min(1e-3)).square().mean()
        loss = (cfg.lambda_gradient * gm + cfg.lambda_distribution * distribution
                + cfg.lambda_propagation * propagation + cfg.lambda_q_anchor * q_anchor
                + cfg.lambda_x_anchor * x_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite shared-Q control loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms = dict(qx_gradient_norm=float(zx.grad.norm()) if zx.grad is not None else 0.,
                     qt_gradient_norm=float(zt.grad.norm()) if zt.grad is not None else 0.,
                     residual_gradient_norm=float(residual.grad.norm()) if residual.grad is not None else 0.)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.)
        optimizer.step()
        with torch.no_grad():
            q_now = assignment(zx, sinkhorn_x)
            pooled = q_now.T @ raw_x / q_now.sum(0).clamp_min(1e-6)[:, None]
            limit = cfg.trust_radius * pooled.norm(dim=1, keepdim=True).clamp_min(1e-3)
            residual.mul_((limit / residual.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.))
        if step == 0 or step + 1 == cfg.steps or (step + 1) % 10 == 0:
            trace.append(dict(step=step + 1, loss=float(loss.detach()), gradient_match=float(gm.detach()),
                distribution_match=float(distribution.detach()), propagation=float(propagation.detach()),
                q_anchor=float(q_anchor.detach()), x_anchor=float(x_anchor.detach()), **norms))
    with torch.no_grad():
        qx = assignment(zx, sinkhorn_x)
        qt = qx if shared else assignment(zt, sinkhorn_t)
        mass = qx.sum(0).clamp_min(1e-6)
        pooled = qx.T @ raw_x / mass[:, None]
        x_final = pooled + residual
        final_message = (syn_adj @ torch.cat((old_x, x_final)))[-len(labels):]
        final_dm = [float(U._distribution_loss(final_message, labels, t)) for t in targets]
    texts, source_ids, hard_text, materialization = U._materialize_text(
        delta, qt.detach().cpu(), labels, encoder, cfg, candidate_texts)
    text_error = float((raw_text[qt.argmax(0)].detach().cpu() - hard_text).abs().max())
    if text_error > 1e-4:
        raise AssertionError("split-Q materialization mismatch")
    block = dict(x=x_final.detach().cpu().float(), text_emb=hard_text.cpu().float(), labels=labels,
        mass=mass.detach().cpu().float(), depth=torch.zeros(len(labels), dtype=torch.long),
        stage=torch.full((len(labels),), int(delta["stage"]), dtype=torch.long), texts=texts,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        support_ids=U._support_ids(delta, qx.detach().cpu(), labels, cfg), text_source_ids=source_ids)
    capacity = torch.zeros_like(mass)
    for cls in torch.unique(raw_labels):
        slots = labels.to(device) == cls
        capacity[slots] = (raw_labels == cls).sum() / slots.sum()
    assignments = {}
    mask = raw_labels[:, None] != labels.to(device)[None, :]
    for name, q, history in (("qx", qx, sinkhorn_x), ("qt", qt, sinkhorn_x if shared else sinkhorn_t)):
        assignments[name] = dict(row_error=float((q.sum(1) - 1).abs().max()),
            column_relative_error=float(((q.sum(0) - capacity) / capacity).abs().max()),
            cross_class_mass=float(q[mask].abs().sum()),
            max_iterations=max(v["iterations"] for v in history))
    audit = dict(constructor="shared_q_parameter_sharing_control", shared=shared, config=asdict(cfg),
        historical_raw_reads=0, uses_boundary_sketch=False, uses_q_topology=False,
        q_cost=init_audit, q_update_count=cfg.steps, qt_update_count=0 if shared else cfg.steps,
        residual_update_count=cfg.steps, old_update_count=0, old_x_relative_change_max=0.,
        text_hard_forward_max_error=text_error, unique_text_count=len(set(texts)),
        q_row_sum_max_error=assignments["qx"]["row_error"],
        q_column_relative_error=assignments["qx"]["column_relative_error"],
        sinkhorn_max_row_error=max(v["row_error"] for v in sinkhorn_x + sinkhorn_t),
        sinkhorn_max_column_relative_error=max(v["column_relative_error"] for v in sinkhorn_x + sinkhorn_t),
        sinkhorn_max_iterations=max(v["iterations"] for v in sinkhorn_x + sinkhorn_t),
        assignments=assignments, qx_qt_row_l1_mean=float((qx - qt).abs().sum(1).mean()),
        qx_qt_top1_disagreement=float((qx.argmax(0) != qt.argmax(0)).float().mean()),
        assignment_parameter_count=(1 if shared else 2) * logits0.numel(),
        qx_changed_norm=float((qx - q0).norm()), qt_changed_norm=float((qt - q0).norm()),
        x_residual_ratio_max=float((residual.detach().norm(dim=1) / pooled.norm(dim=1).clamp_min(1e-3)).max()),
        x_residual_norm=float(residual.detach().norm()), text_materialization=materialization,
        topology=dict(mode="edge_free"), cat_anchor_used=False,
        distribution_matching=dict(enabled=bool(targets), warm_start=False,
            initial_train=initial_dm[0] if targets else None, final_train=final_dm[0] if targets else None,
            initial_holdout_bank=initial_dm[1] if targets else None,
            final_holdout_bank=final_dm[1] if targets else None, holdout_used_for_optimization=False), trace=trace)
    return block, audit
