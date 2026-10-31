"""Frozen-memory pairing controls; no constructor or Student changes."""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

import graphtext_prompt_student as P
import unified_q_multimodal as U


def within_stage_class_permutation(memory, seed):
    labels, stages = memory['labels'].cpu(), memory['stage'].cpu()
    groups = torch.stack((stages, labels), dim=1)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.arange(len(labels))
    for group in torch.unique(groups, dim=0, sorted=True):
        rows = torch.where((groups == group).all(1))[0]
        permutation[rows] = rows[torch.randperm(len(rows), generator=generator)]
    return permutation


def permute_payload(memory, permutation):
    n = len(memory['x'])
    permutation = permutation.long().cpu()
    if not torch.equal(permutation.sort().values, torch.arange(n)):
        raise ValueError('not a complete permutation')
    for key in ('labels', 'stage'):
        if not torch.equal(memory[key].cpu(), memory[key].cpu()[permutation]):
            raise ValueError('payload permutation crossed class or arrival stage')
    variant = dict(memory)
    variant['text_emb'] = memory['text_emb'][permutation].clone()
    for key in ('texts', 'text_source_ids'):
        variant[key] = [memory[key][i] for i in permutation.tolist()]
    for key in ('x', 'labels', 'mass', 'depth', 'stage', 'edge_index'):
        if not torch.equal(variant[key], memory[key]):
            raise AssertionError('fixed field changed: ' + key)
    if Counter(variant['texts']) != Counter(memory['texts']):
        raise AssertionError('text multiset changed')
    return variant


@torch.no_grad()
def text_geometry(memory):
    text = F.normalize(memory['text_emb'].double().cpu(), dim=1)
    groups = torch.stack((memory['stage'].cpu(), memory['labels'].cpu()), dim=1)
    records = []
    for group in torch.unique(groups, dim=0, sorted=True):
        ids = torch.where((groups == group).all(1))[0]
        strings = [memory['texts'][i] for i in ids.tolist()]
        counts = Counter(strings)
        n = len(ids)
        resultant = float(text[ids].mean(0).square().sum())
        records.append(dict(stage=int(group[0]), label=int(group[1]), rows=n,
            unique_texts=len(counts), unique_fraction=len(counts) / n,
            largest_duplicate_fraction=max(counts.values()) / n,
            normalized_text_variance=max(0., 1. - resultant),
            mean_pair_cosine=(n * resultant - 1.) / (n - 1) if n > 1 else None))
    return dict(rows=len(text), unique_texts=len(set(memory['texts'])),
        scope='in-memory text diversity only; not coverage of the raw distribution', classes=records)


def encoded_loss(graph, text, labels, prompt, scale, groups, weighting, rows=None, chunk=256):
    if rows is not None:
        graph, text, labels, groups = graph[rows], text[rows], labels[rows], groups[rows]
    weights = U._group_class_weights(labels, groups, weighting, graph.device, graph.dtype)
    pair = P.multi_positive_pair_loss(graph, text, labels, scale,
        sample_weights=None if weighting == 'node_mean' else weights, chunk_size=chunk)
    logits = scale * (graph @ prompt.T)
    return (.5 * pair + (weights * F.cross_entropy(logits, labels, reduction='none')).sum()
            + .1 * P.prompt_margin_loss(logits, labels, .2, sample_weights=weights))


def gradient_probe(memory, prompt, model_factory, device, seed, permutation, batches=32, batch=24):
    if batches < 4 or batches % 2:
        raise ValueError('use an even number of at least four diagnostic minibatches')
    torch.manual_seed(int(seed))
    x, text = memory['x'].float().to(device), memory['text_emb'].float().to(device)
    labels = memory['labels'].long().to(device)
    stage = memory['stage'].to(device)
    groups = (stage == stage.max()).long()
    model = model_factory(x.shape[1], hidden=384, out=384, text_dim=text.shape[1]).to(device)
    adj = P.normalized_adjacency(memory['edge_index'], len(x), device)
    graph = model.encode_graph(x, adj)
    encoded_text = model.encode_text(text)
    encoded_prompt = model.encode_text(prompt.float().to(device))
    scale = model.logit_scale.exp().clamp(max=100.)
    params = tuple(model.parameters())
    shuffled = encoded_text[permutation.to(device)]

    def measure(current_text, weighting, rows=None):
        loss = encoded_loss(graph, current_text, labels, encoded_prompt, scale,
                            groups, weighting, rows=rows)
        grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params, retain_graph=True)])
        return float(loss.detach()), grad.detach()

    def cosine(a, b):
        return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))

    def relative(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    full_loss, full = measure(encoded_text, 'group_class')
    perm_loss, perm = measure(shuffled, 'group_class')
    node_loss, node = measure(encoded_text, 'node_mean')
    half_a, half_b, perm_mean = torch.zeros_like(full), torch.zeros_like(full), torch.zeros_like(full)
    generator = torch.Generator().manual_seed(int(seed) * 7 + 1)
    losses, singleton_fraction = [], []
    for b in range(batches):
        rows = torch.randperm(len(x), generator=generator)[:min(batch, len(x))].to(device)
        loss, gradient = measure(encoded_text, 'node_mean', rows)
        ploss, pgradient = measure(shuffled, 'node_mean', rows)
        accumulator = half_a if b < batches // 2 else half_b
        accumulator.add_(gradient / (batches // 2))
        perm_mean.add_(pgradient / batches)
        losses.append(ploss - loss)
        counts = labels[rows, None].eq(labels[rows][None, :]).sum(1)
        singleton_fraction.append(float((counts == 1).float().mean()))
    mean = (half_a + half_b) / 2
    delta = torch.tensor(losses)
    return dict(scope='initialization-only parameter gradients, no model updates',
        model_seed=int(seed), minibatches=batches, batch=batch,
        full_group_loss=full_loss, full_node_loss=node_loss,
        full_permutation_loss_error=abs(full_loss - perm_loss),
        full_permutation_gradient_relative_error=relative(perm, full),
        full_group_to_node_gradient_cosine=cosine(full, node),
        full_group_to_minibatch_gradient_cosine=cosine(full, mean),
        full_node_to_minibatch_gradient_cosine=cosine(node, mean),
        minibatch_permutation_gradient_relative=relative(perm_mean, mean),
        minibatch_sampling_half_difference_relative=relative(half_a, half_b),
        minibatch_permutation_loss_delta_mean=float(delta.mean()),
        minibatch_permutation_loss_delta_std=float(delta.std()),
        minibatch_single_positive_fraction=sum(singleton_fraction) / batches)
