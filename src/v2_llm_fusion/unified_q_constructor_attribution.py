"""Constructor controls; the frozen Unified-Q implementation is not modified."""
from dataclasses import asdict, replace

import torch

import unified_q_multimodal as U


MAIN = "uq_dm_edgefree"
CORE = (MAIN, "no_node_text", "no_residual", "no_distribution", "free_x")
TEXT = ("no_text_init", "no_payload_gm", "stop_text_gradient")


def control_config(base, arm):
    changes = {
        MAIN: {}, "no_node_text": dict(text_weight=0., pair_weight=0.),
        "no_text_init": dict(text_weight=0.),
        "no_payload_gm": dict(pair_weight=0.),
        "stop_text_gradient": dict(text_assignment="detached"),
        "no_residual": dict(train_residual=False),
        "no_distribution": dict(lambda_distribution=0.), "free_x": {},
    }
    if arm not in changes:
        raise ValueError(f"unknown constructor control: {arm}")
    return replace(base, **changes[arm])


def project_free_x(x, pooled, radius):
    """Same conditional feasible set as pool(Q) + bounded residual."""
    offset = x - pooled
    limit = float(radius) * pooled.norm(dim=1, keepdim=True).clamp_min(1e-3)
    return pooled + offset * (limit / offset.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)


def optimize_control(delta, parent, labels, encoder, prompt_emb, cfg, device, arm):
    cfg = control_config(cfg, arm)
    if arm != "free_x":
        block, audit = U.optimize_stage(delta, parent, labels, encoder, prompt_emb,
            cfg, device, use_boundary=False, use_topology=False)
    else:
        block, audit = optimize_free_x(delta, parent, labels, encoder, prompt_emb, cfg, device)
    audit["constructor_control"] = arm
    audit["node_payload_used_in_initialization"] = bool(cfg.text_weight)
    audit["node_payload_used_in_proxy_loss"] = bool(cfg.lambda_gradient and cfg.pair_weight)
    audit["class_prompts_retained"] = True
    return block, audit


def optimize_free_x(delta, parent, labels, encoder, prompt_emb, cfg, device):
    """Untie X from differentiable pooling, without relaxing its feasible set.

    Q still routes text and defines the same propagation target/anchors. X is
    a separate Adam parameter initialized at pool(Q0), then projected around
    pool(Q) with the original radius. This is NOT unconstrained CaT-free-X.
    """
    if cfg.old_update_radius or cfg.materialize_top_k != 1:
        raise ValueError("free-X control requires frozen prefix and hard top1")
    if not cfg.train_q or not cfg.train_residual or cfg.optimization != "joint":
        raise ValueError("free-X control assumes joint Q/X updates")
    labels = labels.long().cpu()
    old_n = 0 if parent is None else len(parent["x"])
    raw_labels = delta["y"].long().to(device)
    candidate_texts = [U._append_text(t, cfg.text_chars) for t in delta["texts"]]
    raw_text = U._norm(torch.as_tensor(encoder.encode(candidate_texts)).float().to(device))
    delta = dict(delta, text_emb=raw_text.detach().cpu())
    delta["_raw_adjacency"] = U._sparse_adjacency(delta["edge_index"], len(delta["x"]), device)
    raw_x = delta["x"].float().to(device)
    boundary = torch.zeros((len(raw_x), old_n), device=device)
    logits0, q0, init_audit = U._initial_logits(delta, parent, labels, cfg, device, boundary)
    pooled0 = q0.T @ raw_x / q0.sum(0).clamp_min(1e-6)[:, None]
    q_logits = torch.nn.Parameter(logits0.clone())
    x_parameter = torch.nn.Parameter(pooled0.detach().clone())
    parameters = [q_logits, x_parameter]
    optimizer = torch.optim.Adam([
        {"params": [q_logits], "lr": cfg.lr_q},
        {"params": [x_parameter], "lr": cfg.lr_x}])
    old_x = parent["x"].float().to(device) if parent else raw_x.new_empty((0, raw_x.shape[1]))
    old_text = parent["text_emb"].float().to(device) if parent else raw_text.new_empty((0, raw_text.shape[1]))
    old_labels = parent["labels"].long().to(device) if parent else raw_labels.new_empty(0)
    ref_adj, ref_x = U._reference_graph(delta, parent, boundary, device)
    ref_text = torch.cat((old_text, raw_text))
    ref_labels = torch.cat((old_labels, raw_labels))
    full_labels = torch.cat((old_labels, labels.to(device)))
    ref_groups = torch.cat((torch.zeros(old_n), torch.ones(len(raw_labels)))).long().to(device)
    syn_groups = torch.cat((torch.zeros(old_n), torch.ones(len(labels)))).long().to(device)
    target_message = U._adjacency_mm(ref_adj, ref_x).detach()
    new_target = target_message[-len(raw_labels):]
    syn_adj = U._hard_adjacency(torch.empty((2, 0), dtype=torch.long), parent,
                               old_n + len(labels), device)
    targets = []
    if cfg.lambda_distribution:
        bank_seed = cfg.seed + 1009 * int(delta["stage"]) + 999983
        targets = [U._distribution_target(new_target, raw_labels, cfg, bank_seed + off)
                   for off in (0, 1)]
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
    sinkhorn_audit = []

    def assignment():
        return U._masked_sinkhorn(q_logits, raw_labels, labels.to(device), cfg.temperature,
            cfg.sinkhorn_steps, cfg.sinkhorn_tolerance, cfg.sinkhorn_max_steps, sinkhorn_audit)

    with torch.no_grad():
        initial_message = (syn_adj @ torch.cat((old_x, pooled0)))[-len(labels):]
        initial_dm = [float(U._distribution_loss(initial_message, labels, t)) for t in targets]
    trace = []
    for step in range(cfg.steps):
        q = assignment()
        mass = q.sum(0).clamp_min(1e-6)
        pooled = q.T @ raw_x / mass[:, None]
        soft_text = U._norm(q.T @ raw_text / mass[:, None])
        hard_text = raw_text[q.argmax(0)]
        text_new = hard_text + (soft_text - soft_text.detach())
        full_x = torch.cat((old_x, x_parameter))
        synthetic = dict(x=full_x, text_emb=torch.cat((old_text, text_new)), labels=full_labels)
        terms = []
        for model, real_grad in zip(models, real_grads):
            syn_loss = U._loss(model, synthetic, syn_adj, prompt_emb, group_ids=syn_groups,
                loss_weighting=cfg.loss_weighting, chunk_size=cfg.pair_chunk_size, pair_weight=cfg.pair_weight)
            syn_grad = torch.autograd.grad(syn_loss, tuple(model.parameters()), create_graph=True, retain_graph=True)
            terms.append(U._gradient_distance(U._flat_grads(real_grad), U._flat_grads(syn_grad)))
        gm = torch.stack(terms).mean() if terms else x_parameter.sum() * 0.
        target_new = U._norm(q.T @ new_target / mass[:, None])
        message = (syn_adj @ full_x)[-len(labels):]
        distribution = U._distribution_loss(message, labels, targets[0]) if targets else gm * 0.
        normalized = U._norm(message)
        propagation = ((target_new - normalized).square().mean()
                       if cfg.propagation_metric == "dimension_mean" else
                       (1 - (target_new * normalized).sum(1)).mean())
        q_anchor = (q - q0.detach()).square().mean()
        # Same function of (Q, X) as the coupled arm's normalized residual.
        x_anchor = ((x_parameter - pooled) / pooled0.norm(dim=1, keepdim=True).clamp_min(1e-3)).square().mean()
        loss = (cfg.lambda_gradient * gm + cfg.lambda_distribution * distribution
                + cfg.lambda_propagation * propagation + cfg.lambda_q_anchor * q_anchor
                + cfg.lambda_x_anchor * x_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite free-X control loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        q_norm = float(q_logits.grad.norm()) if q_logits.grad is not None else 0.
        x_norm = float(x_parameter.grad.norm()) if x_parameter.grad is not None else 0.
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.)
        optimizer.step()
        with torch.no_grad():
            q_now = assignment()
            pooled_now = q_now.T @ raw_x / q_now.sum(0).clamp_min(1e-6)[:, None]
            x_parameter.copy_(project_free_x(x_parameter, pooled_now, cfg.trust_radius))
        if step == 0 or step + 1 == cfg.steps or (step + 1) % 10 == 0:
            trace.append(dict(step=step + 1, loss=float(loss.detach()), gradient_match=float(gm.detach()),
                distribution_match=float(distribution.detach()), propagation=float(propagation.detach()),
                q_anchor=float(q_anchor.detach()), x_anchor=float(x_anchor.detach()),
                q_gradient_norm=q_norm, free_x_gradient_norm=x_norm))
    with torch.no_grad():
        q = assignment()
        mass = q.sum(0).clamp_min(1e-6)
        pooled = q.T @ raw_x / mass[:, None]
        x_final = x_parameter.detach()
        residual = x_final - pooled
        final_message = (syn_adj @ torch.cat((old_x, x_final)))[-len(labels):]
        final_dm = [float(U._distribution_loss(final_message, labels, t)) for t in targets]
    texts, source_ids, hard_text, materialization = U._materialize_text(
        delta, q.detach().cpu(), labels, encoder, cfg, candidate_texts)
    text_error = float((raw_text[q.argmax(0)].detach().cpu() - hard_text).abs().max())
    if text_error > 1e-4:
        raise AssertionError("free-X materialization mismatch")
    block = dict(x=x_final.cpu().float(), text_emb=hard_text.cpu().float(), labels=labels,
        mass=mass.detach().cpu().float(), depth=torch.zeros(len(labels), dtype=torch.long),
        stage=torch.full((len(labels),), int(delta["stage"]), dtype=torch.long), texts=texts,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        support_ids=U._support_ids(delta, q.detach().cpu(), labels, cfg), text_source_ids=source_ids)
    capacity = torch.zeros_like(mass)
    for cls in torch.unique(raw_labels):
        slots = labels.to(device) == cls
        capacity[slots] = (raw_labels == cls).sum() / slots.sum()
    audit = dict(constructor="projected_free_x_control", config=asdict(cfg),
        feature_parameterization="independent X and Q; same conditional trust region as coupled arm",
        historical_raw_reads=0, uses_boundary_sketch=False, uses_q_topology=False,
        q_cost=init_audit, q_update_count=cfg.steps, free_x_update_count=cfg.steps,
        residual_update_count=0, old_update_count=0, old_x_relative_change_max=0.,
        text_hard_forward_max_error=text_error, unique_text_count=len(set(texts)),
        q_row_sum_max_error=float((q.sum(1)-1).abs().max()),
        q_column_relative_error=float(((mass-capacity)/capacity).abs().max()),
        sinkhorn_max_row_error=max(v["row_error"] for v in sinkhorn_audit),
        sinkhorn_max_column_relative_error=max(v["column_relative_error"] for v in sinkhorn_audit),
        sinkhorn_max_iterations=max(v["iterations"] for v in sinkhorn_audit),
        x_residual_ratio_max=float((residual.norm(dim=1)/pooled.norm(dim=1).clamp_min(1e-3)).max()),
        x_residual_norm=float(residual.norm()), text_materialization=materialization,
        topology=dict(mode="edge_free"), cat_anchor_used=False,
        distribution_matching=dict(enabled=bool(targets), warm_start=False,
            initial_train=initial_dm[0] if targets else None, final_train=final_dm[0] if targets else None,
            initial_holdout_bank=initial_dm[1] if targets else None,
            final_holdout_bank=final_dm[1] if targets else None, holdout_used_for_optimization=False),
        trace=trace)
    return block, audit
