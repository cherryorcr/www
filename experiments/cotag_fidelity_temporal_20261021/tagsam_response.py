"""Explicit response budgets; minibatch updates are not native full epochs."""
from __future__ import annotations

import time

import common as C
from common import R, torch
import torch.nn.functional as F
import run_unified_q_tagsam_20261008 as A

T = A.T


def inner_batches(rows, loops, batch, seed, mode):
    generator = torch.Generator().manual_seed(seed)
    if mode == 'updates':
        return [torch.randperm(rows, generator=generator)[:min(batch, rows)]
                for _ in range(loops)]
    if mode == 'epochs':
        if rows < batch:
            raise ValueError('Native full-epoch implementation has no complete batch')
        batches = []
        for _ in range(loops):
            order = torch.randperm(rows, generator=generator)
            batches.extend(order[start:start+batch]
                           for start in range(0, (rows//batch)*batch, batch))
        return batches
    raise ValueError(mode)


def optimize(initial, parent, raw, teacher, stage, device, loss_fn, steps,
             loops=10, mode='updates', reference_cap=256, batch=16,
             resample_reference=False, batch_local=False, trace_path=None):
    memory = R.L.append_edge_free(parent, dict(initial))
    keep = 0 if parent is None else len(parent['x'])
    adj = T.G.norm_adj(memory['edge_index'], len(memory['x']), device)
    real_adj = T.G.norm_adj(raw['edge_index'], len(raw['x']), device)
    rx, rt = raw['x'].to(device), raw['text_emb'].to(device)
    text = memory['text_emb'].to(device)
    initial_x = initial['x'].to(device)
    var = torch.nn.Parameter(initial_x.clone())
    prefix = memory['x'][:keep].to(device)
    rate_g = torch.nn.Parameter(torch.tensor(.005, device=device))
    rate_t = torch.nn.Parameter(torch.tensor(.005, device=device))
    outer = torch.optim.SGD([{'params': [var], 'lr': 100., 'momentum': .5},
                            {'params': [rate_g, rate_t], 'lr': 2e-6, 'momentum': .5}])
    reference = torch.randperm(len(rx), generator=torch.Generator().manual_seed(777+stage))[:reference_cap].to(device)
    with torch.no_grad():
        teacher_response = teacher.logits(rx, real_adj, rt[reference])[reference].detach()
    curve, start = [], time.perf_counter()
    for step in range(1, steps+1):
        torch.manual_seed(9001+stage*1009+step)
        student = T.G.CLIPStudent(var.shape[1]).to(device)
        params = dict(student.named_parameters())
        synthetic = torch.cat((prefix, var))
        batches = inner_batches(len(synthetic), loops, batch, 777+stage*1009+step, mode)
        for rows in batches:
            rows = rows.to(device)
            if batch_local:
                local_adj = torch.eye(len(rows), device=device)
                graph = T._func_graph(params, synthetic[rows], local_adj)
            else:
                graph = T._func_graph(params, synthetic, adj)[rows]
            text_features = T._func_text(params, text[rows])
            logits = params['logit_scale'].exp().clamp(max=100)*graph@text_features.T
            target = torch.arange(len(rows), device=device)
            inner_loss = (F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2
            gradients = torch.autograd.grad(inner_loss, tuple(params.values()), create_graph=True)
            params = {name: value-(rate_t if name.startswith('text_proj') else rate_g)*gradient
                      for (name, value), gradient in zip(params.items(), gradients)}
        if resample_reference:
            reference = torch.randperm(len(rx), generator=torch.Generator().manual_seed(
                700777+stage*1009+step))[:reference_cap].to(device)
            with torch.no_grad():
                teacher_response = teacher.logits(rx, real_adj, rt[reference])[reference].detach()
        graph = T._func_graph(params, rx, real_adj)[reference]
        text_features = T._func_text(params, rt[reference])
        response = params['logit_scale'].exp().clamp(max=100)*graph@text_features.T
        loss = loss_fn(response, teacher_response)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite response loss at {step}')
        outer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, [var, rate_g, rate_t])
        for param, gradient in zip((var, rate_g, rate_t), gradients):
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f'Nonfinite response gradient at {step}')
            param.grad = gradient
        torch.nn.utils.clip_grad_norm_([var], 5.)
        outer.step()
        curve.append(dict(step=step, loss=float(loss.detach()),
            graph_inner_lr=float(rate_g.detach()), text_inner_lr=float(rate_t.detach()),
            inner_updates=len(batches), x_shift_rms=float((var.detach()-initial_x).square().mean().sqrt())))
        if step == 1 or step % 25 == 0 or step == steps:
            if trace_path is not None:
                C.save(trace_path, curve)
            C.emit(event='tagsam_response', stage=stage, mode=mode, steps=steps, **curve[-1])
        del student, params, gradients, synthetic, loss, graph, text_features, response
    memory['x'] = torch.cat((prefix.cpu(), var.detach().cpu()))
    R.V.validate_memory(memory, parent, len(memory['x']))
    return memory, dict(optimizer_steps=steps, inner_mode=mode, inner_loops=loops,
        inner_batch=batch, inner_updates_per_outer=len(batches),
        inner_steps_total=steps*len(batches), reference_cap=reference_cap,
        reference_resampled=resample_reference, batch_local_graph=batch_local,
        final_graph_lr=float(rate_g.detach()), final_text_lr=float(rate_t.detach()),
        seconds=time.perf_counter()-start, curve=curve, source_rows=len(rx))
