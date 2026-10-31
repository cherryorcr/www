"""One-factor assignment controls derived from the frozen 20261012 constructor."""
from dataclasses import asdict, replace

import torch

import unified_q_multimodal as U


def diagnostics(qx, qt, q0, raw_text, raw_labels, labels, texts):
    result = {}
    for name, q in (("feature", qx), ("text", qt)):
        entropy_sum = normalized_sum = 0.
        for cls in torch.unique(raw_labels):
            raw = raw_labels == cls
            slots = labels.to(q.device) == cls
            local = q[raw][:, slots]
            entropy = -(local.clamp_min(1e-12) * local.clamp_min(1e-12).log()).sum(1)
            entropy_sum += float(entropy.sum())
            count = int(slots.sum())
            if count > 1:
                divisor = float(torch.log(torch.tensor(count, dtype=torch.float64)))
                normalized_sum += float(entropy.sum()) / divisor
        winners = q.argmax(1)
        result[name] = dict(row_entropy=entropy_sum / len(q),
            normalized_row_entropy=normalized_sum / len(q),
            hard_slot_coverage=float(winners.unique().numel() / q.shape[1]),
            positive_mass_slot_coverage=float((q.sum(0) > 1e-6).float().mean()),
            relative_update=float((q - q0).norm() / q0.norm().clamp_min(1e-12)))
    soft = U._norm(qt.T @ raw_text / qt.sum(0).clamp_min(1e-6)[:, None])
    hard = raw_text[qt.argmax(0)]
    gaps = 1 - (soft * hard).sum(1)
    count = min(len(hard), 512)
    pair_sum = (hard[:count] @ hard[:count].T).sum() - count
    return dict(assignments=result, unique_source_ratio=float(qt.argmax(0).unique().numel() / qt.shape[1]),
        unique_text_ratio=len(set(texts)) / len(texts),
        within_memory_text_cosine=float(pair_sum / max(1, count * (count - 1))),
        text_cosine_scope="first up to 512 slots; includes across-class pairs",
        soft_hard_cosine_gap_mean=float(gaps.mean()), soft_hard_cosine_gap_max=float(gaps.max()),
        support_top1_agreement=float((qx.argmax(0) == qt.argmax(0)).float().mean()),
        raw_row_assignment_l1=float((qx - qt).abs().sum(1).mean()))


class LogitUpdates(torch.nn.Module):
    """Identical initial logits; optional equal-count update subspaces.

    Shared uses 2r directions per raw row. Independent uses r directions for
    each modality. This controls active count, but changes the per-branch
    subspace and is consequently a sensitivity experiment, not pure causality.
    """
    def __init__(self, initial, raw_labels, labels, shared, equal_budget, seed):
        super().__init__()
        self.initial = initial
        self.shared = shared
        self.equal_budget = equal_budget
        self.locations = []
        self.x = torch.nn.ParameterList()
        self.t = torch.nn.ParameterList()
        self.bases = []
        if not equal_budget:
            self.x.append(torch.nn.Parameter(initial.clone()))
            if not shared:
                self.t.append(torch.nn.Parameter(initial.clone()))
            return
        generator = torch.Generator(device=initial.device).manual_seed(seed + 88129)
        for cls in torch.unique(raw_labels, sorted=True):
            rows = torch.where(raw_labels == cls)[0]
            slots = torch.where(labels.to(initial.device) == cls)[0]
            rank = min(8, (len(slots) - 1) // 2)
            if rank == 0:
                continue
            basis = torch.randn((len(slots), 2 * rank), device=initial.device, generator=generator)
            basis -= basis.mean(0, keepdim=True)
            basis = torch.linalg.qr(basis, mode="reduced").Q
            self.locations.append((rows, slots))
            self.bases.append(basis)
            self.x.append(torch.nn.Parameter(initial.new_zeros((len(rows), 2 * rank if shared else rank))))
            if not shared:
                self.t.append(torch.nn.Parameter(initial.new_zeros((len(rows), rank))))
        if not self.x:
            raise ValueError("equal-budget control has no class with at least three slots")

    def logits(self, text=False):
        params = self.x if self.shared or not text else self.t
        if not self.equal_budget:
            return params[0]
        update = torch.zeros_like(self.initial)
        for parameter, (rows, slots), basis in zip(params, self.locations, self.bases):
            if self.shared:
                selected = basis
            else:
                rank = basis.shape[1] // 2
                selected = basis[:, rank:] if text else basis[:, :rank]
            update[rows[:, None], slots[None, :]] = parameter @ selected.T
        return self.initial + update


def optimize(delta, parent, labels, encoder, prompt_emb, cfg, device,
             shared=True, soft_stored=False, equal_budget=False,
             randomized_initial=False):
    if (cfg.old_update_radius or cfg.materialize_top_k != 1
            or cfg.optimization != "joint" or cfg.text_assignment != "learned"
            or cfg.feature_pooling != "raw" or cfg.loss_weighting == "student_mass"
            or cfg.lambda_joint_distribution):
        raise ValueError("control requires frozen joint edge-free configuration")
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
    if randomized_initial:
        # Permute rows within each class: preserve score scale/capacities while
        # removing the correspondence between the initial costs and raw nodes.
        generator = torch.Generator(device=device).manual_seed(cfg.seed + 1009 * int(delta["stage"]) + 710141)
        logits0 = logits0.clone()
        moved = 0
        for cls in torch.unique(raw_labels, sorted=True):
            rows = torch.where(raw_labels == cls)[0]
            permutation = torch.randperm(len(rows), generator=generator, device=device)
            logits0[rows] = logits0[rows[permutation]].clone()
            moved += int((permutation != torch.arange(len(rows), device=device)).sum())
        q0 = U._masked_sinkhorn(logits0, raw_labels, labels.to(device), cfg.temperature,
            cfg.sinkhorn_steps, cfg.sinkhorn_tolerance, cfg.sinkhorn_max_steps)
        init_audit = dict(init_audit, randomization="within-class row permutation of structured costs",
            moved_rows=moved, randomization_seed=cfg.seed + 1009 * int(delta["stage"]) + 710141)
    pooled0 = q0.T @ raw_x / q0.sum(0).clamp_min(1e-6)[:, None]
    updates = LogitUpdates(logits0, raw_labels, labels, shared, equal_budget,
                          cfg.seed + 1009 * int(delta["stage"]))
    updates.requires_grad_(cfg.train_q)
    residual = torch.nn.Parameter(torch.zeros_like(pooled0), requires_grad=cfg.train_residual)
    parameters = [p for p in updates.parameters() if p.requires_grad] + ([residual] if cfg.train_residual else [])
    groups = [{"params": list(updates.parameters()), "lr": cfg.lr_q}] if cfg.train_q else []
    if cfg.train_residual:
        groups.append({"params": [residual], "lr": cfg.lr_x})
    if not groups:
        raise ValueError("a fixed-Q control must retain a trainable residual")
    optimizer = torch.optim.Adam(groups)
    old_x = parent["x"].float().to(device) if parent else raw_x.new_empty((0, raw_x.shape[1]))
    old_text = parent["text_emb"].float().to(device) if parent else raw_text.new_empty((0, raw_text.shape[1]))
    old_labels = parent["labels"].long().to(device) if parent else raw_labels.new_empty(0)
    ref_adj, ref_x = U._reference_graph(delta, parent, boundary, device)
    ref_text, ref_labels = torch.cat((old_text, raw_text)), torch.cat((old_labels, raw_labels))
    full_labels = torch.cat((old_labels, labels.to(device)))
    ref_groups = torch.cat((torch.zeros(old_n), torch.ones(len(raw_labels)))).long().to(device)
    syn_groups = torch.cat((torch.zeros(old_n), torch.ones(len(labels)))).long().to(device)
    target_message = U._adjacency_mm(ref_adj, ref_x).detach()[-len(raw_labels):]
    syn_adj = U._hard_adjacency(torch.empty((2, 0), dtype=torch.long), parent, old_n + len(labels), device)
    bank_seed = cfg.seed + 1009 * int(delta["stage"]) + 999983
    targets = ([U._distribution_target(target_message, raw_labels, cfg, bank_seed + off) for off in (0, 1)]
               if cfg.lambda_distribution else [])
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
    sinkhorn_x, sinkhorn_t, trace = [], [], []

    def assignment(text=False):
        return U._masked_sinkhorn(updates.logits(text), raw_labels, labels.to(device), cfg.temperature,
            cfg.sinkhorn_steps, cfg.sinkhorn_tolerance, cfg.sinkhorn_max_steps,
            sinkhorn_t if text else sinkhorn_x)

    with torch.no_grad():
        initial_error = float((assignment() - q0).abs().max())
        initial_message = (syn_adj @ torch.cat((old_x, pooled0)))[-len(labels):]
        initial_dm = [float(U._distribution_loss(initial_message, labels, t)) for t in targets]
    for step in range(cfg.steps):
        qx = assignment()
        qt = qx if shared else assignment(True)
        mass = qx.sum(0).clamp_min(1e-6)
        x_new = qx.T @ raw_x / mass[:, None] + residual
        soft_text = U._norm(qt.T @ raw_text / qt.sum(0).clamp_min(1e-6)[:, None])
        hard_text = raw_text[qt.argmax(0)]
        text_new = soft_text if soft_stored else hard_text + (soft_text - soft_text.detach())
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
        propagation = ((target_new - normalized).square().mean() if cfg.propagation_metric == "dimension_mean"
                       else (1 - (target_new * normalized).sum(1)).mean())
        q_anchor_x, q_anchor_t = (qx - q0).square().mean(), (qt - q0).square().mean()
        q_anchor = q_anchor_x if shared else .5 * (q_anchor_x + q_anchor_t)
        x_anchor = (residual / pooled0.norm(dim=1, keepdim=True).clamp_min(1e-3)).square().mean()
        loss = (cfg.lambda_gradient * gm + cfg.lambda_distribution * distribution
                + cfg.lambda_propagation * propagation + cfg.lambda_q_anchor * q_anchor
                + cfg.lambda_x_anchor * x_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Q control loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        q_norm = sum(float(p.grad.square().sum()) for p in updates.parameters() if p.grad is not None) ** .5
        r_norm = float(residual.grad.norm()) if residual.grad is not None else 0.
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.)
        optimizer.step()
        with torch.no_grad():
            q_now = assignment()
            pooled = q_now.T @ raw_x / q_now.sum(0).clamp_min(1e-6)[:, None]
            limit = cfg.trust_radius * pooled.norm(dim=1, keepdim=True).clamp_min(1e-3)
            residual.mul_((limit / residual.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.))
        if step == 0 or step + 1 == cfg.steps or (step + 1) % 10 == 0:
            trace.append(dict(step=step + 1, loss=float(loss.detach()), gradient_match=float(gm.detach()),
                distribution_match=float(distribution.detach()), propagation=float(propagation.detach()),
                q_gradient_norm=q_norm, residual_gradient_norm=r_norm,
                pool_relative_change=float((pooled - pooled0).norm() / pooled0.norm().clamp_min(1e-12)),
                residual_relative_norm=float(residual.norm() / pooled.norm().clamp_min(1e-12))))
    with torch.no_grad():
        qx = assignment()
        qt = qx if shared else assignment(True)
        mass = qx.sum(0).clamp_min(1e-6)
        pooled = qx.T @ raw_x / mass[:, None]
        x_final = pooled + residual
        final_message = (syn_adj @ torch.cat((old_x, x_final)))[-len(labels):]
        final_dm = [float(U._distribution_loss(final_message, labels, t)) for t in targets]
        soft_text = U._norm(qt.T @ raw_text / qt.sum(0).clamp_min(1e-6)[:, None])
    texts, source_ids, hard_text, materialization = U._materialize_text(
        delta, qt.detach().cpu(), labels, encoder, cfg, candidate_texts)
    text_error = float((raw_text[qt.argmax(0)].detach().cpu() - hard_text).abs().max())
    if text_error > 1e-4:
        raise AssertionError("hard-forward source re-encoding mismatch")
    stored = soft_text.detach().cpu() if soft_stored else hard_text.cpu()
    block = dict(x=x_final.detach().cpu().float(), text_emb=stored.float(), labels=labels,
        mass=mass.detach().cpu().float(), depth=torch.zeros(len(labels), dtype=torch.long),
        stage=torch.full((len(labels),), int(delta["stage"]), dtype=torch.long), texts=texts,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        support_ids=U._support_ids(delta, qx.detach().cpu(), labels, cfg), text_source_ids=source_ids)
    capacity = torch.zeros_like(mass)
    for cls in torch.unique(raw_labels):
        slots = labels.to(device) == cls
        capacity[slots] = (raw_labels == cls).sum() / slots.sum()
    all_history = sinkhorn_x + sinkhorn_t
    cross_class = raw_labels[:, None] != labels.to(device)[None, :]
    assignments = {}
    for name, q in (("qx", qx), ("qt", qt)):
        assignments[name] = dict(row_error=float((q.sum(1) - 1).abs().max()),
            column_relative_error=float(((q.sum(0) - capacity) / capacity).abs().max()),
            cross_class_mass=float(q[cross_class].abs().sum()))
    audit = dict(constructor="assignment_ablation_20261014", shared=shared, config=asdict(cfg),
        randomized_initial=randomized_initial,
        equal_parameter_budget=equal_budget, initial_q_max_error=initial_error,
        parameterization="zero-initialized class-block orthogonal updates" if equal_budget else "full logits",
        parameter_budget_caveat="equal count changes per-branch update subspace" if equal_budget else None,
        assignment_parameter_count=sum(p.numel() for p in updates.parameters() if p.requires_grad),
        residual_parameter_count=residual.numel() if cfg.train_residual else 0,
        legal_assignment_entries=int(sum(int((raw_labels == c).sum()) * int((labels == c.cpu()).sum())
                                         for c in torch.unique(raw_labels))),
        historical_raw_reads=0, uses_boundary_sketch=False, uses_q_topology=False,
        q_update_count=cfg.steps if cfg.train_q else 0,
        qt_update_count=cfg.steps if cfg.train_q and not shared else 0,
        residual_update_count=cfg.steps if cfg.train_residual else 0, old_update_count=0,
        old_x_relative_change_max=0., text_hard_forward_max_error=None if soft_stored else text_error,
        text_storage="soft embedding; strings are provenance only" if soft_stored else "re-encoded real string",
        forward_and_storage_match=True,
        stored_vs_reference_string_max_error=float((stored - hard_text.cpu()).abs().max()),
        q_cost=init_audit, assignments=assignments,
        q_row_sum_max_error=max(v["row_error"] for v in assignments.values()),
        q_column_relative_error=max(v["column_relative_error"] for v in assignments.values()),
        sinkhorn_max_row_error=max(v["row_error"] for v in all_history),
        sinkhorn_max_column_relative_error=max(v["column_relative_error"] for v in all_history),
        sinkhorn_max_iterations=max(v["iterations"] for v in all_history),
        diagnostics=diagnostics(qx.detach(), qt.detach(), q0, raw_text, raw_labels, labels, texts),
        topology=dict(mode="edge_free"), cat_anchor_used=False, text_materialization=materialization,
        distribution_matching=dict(enabled=bool(targets), warm_start=False,
            initial_train=initial_dm[0] if targets else None, final_train=final_dm[0] if targets else None,
            initial_holdout_bank=initial_dm[1] if targets else None,
            final_holdout_bank=final_dm[1] if targets else None, holdout_used_for_optimization=False), trace=trace)
    return block, audit


def build(arm, delta, parent, labels, encoder, prompt_emb, cfg, device):
    options = {
        "control_shared": (True, False, False, True),
        "control_independent": (False, False, False, True),
        "control_shared_no_residual": (True, False, False, False),
        "control_independent_no_residual": (False, False, False, False),
        "control_shared_equal": (True, False, True, True),
        "control_independent_equal": (False, False, True, True),
        "control_soft_stored": (True, True, False, True),
    }
    if arm not in options:
        raise ValueError(arm)
    shared, soft, equal, residual = options[arm]
    return optimize(delta, parent, labels, encoder, prompt_emb,
        replace(cfg, train_residual=residual), device, shared=shared,
        soft_stored=soft, equal_budget=equal)
