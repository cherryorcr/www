"""Deterministic delta-only graph-conditioned snippet selection for P2-A."""
from __future__ import annotations

import hashlib
import torch
import torch.nn.functional as F

import p2_text_operators as T

ARMS = ("center_concat", "mean_pool", "text_attention", "text_selection",
        "attribute_selection", "graph_selection", "graph_pool", "shuffled_graph")
BUDGET = 64
TEMPERATURE = 0.2
RELEVANCE = 0.8
SOURCE_CAP = 240
SNIPPET_TOKENS = 32
MAX_PARTS = 5


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((value.shape, value.dtype)).encode()
                          + value.numpy().tobytes()).hexdigest()


def snippet_bank(texts, encoder):
    """Use a common, explicitly costed prefix of every legal delta text."""
    tokenizer = encoder.tokenizer
    snippets, owners, offsets = [], [], []
    for owner, text in enumerate(texts):
        ids = tokenizer.encode(str(text), add_special_tokens=False)[:SOURCE_CAP]
        for offset in range(0, len(ids), SNIPPET_TOKENS):
            part = tokenizer.decode(ids[offset:offset + SNIPPET_TOKENS],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
            part = T.bounded_prefix(tokenizer, part, SNIPPET_TOKENS)
            if part.strip():
                snippets.append(part)
                owners.append(owner)
                offsets.append(offset)
    if not snippets:
        raise ValueError("delta has no usable text snippets")
    embeddings = encoder.encode(snippets).float().cpu()
    return dict(texts=snippets, owners=torch.tensor(owners), offsets=offsets,
                embeddings=embeddings,
                content_tokens=sum(T.token_count(tokenizer, s) for s in snippets),
                tokens_with_special=sum(len(tokenizer.encode(s, add_special_tokens=True))
                                        for s in snippets))


def make_queries(x, text_emb, clusters, local_edges, shuffle_seed):
    """Align attribute summaries by a paired-source kernel, not a raw X dot T.

    Only local, current-delta indices are accepted. Neighbor summaries use
    unique endpoints outside each cluster. No report/future/old raw inputs.
    """
    n = len(x)
    if local_edges.numel() and (local_edges.min() < 0 or local_edges.max() >= n):
        raise ValueError("nonlocal endpoint in construction graph")
    if len(text_emb) != n or not torch.isfinite(text_emb).all():
        raise ValueError("invalid paired source embeddings")
    joined = torch.cat(clusters)
    if not torch.equal(torch.sort(joined).values, torch.arange(n)):
        raise ValueError("source clusters must partition the legal delta")
    xn, tn = F.normalize(x.float(), dim=1), F.normalize(text_emb.float(), dim=1)
    attrs, texts, neighbors, stats = [], [], [], []
    for members in clusters:
        attribute = F.normalize(x[members].float().mean(0), dim=0)
        weights = torch.softmax((xn[members] @ attribute) / TEMPERATURE, dim=0)
        attrs.append(F.normalize((weights[:, None] * tn[members]).sum(0), dim=0))
        texts.append(F.normalize(tn[members].mean(0), dim=0))
        inside = torch.zeros(n, dtype=torch.bool)
        inside[members] = True
        src, dst = local_edges
        adjacent = torch.cat((dst[inside[src]], src[inside[dst]])).unique(sorted=True)
        adjacent = adjacent[~inside[adjacent]]
        neighbors.append(F.normalize(tn[adjacent].mean(0), dim=0)
                         if len(adjacent) else torch.zeros(tn.shape[1]))
        stats.append(dict(neighbor_count=len(adjacent), source_count=len(members),
                          neighbor_local_ids=adjacent.tolist(),
                          attribute_effective_support=float(1 / weights.square().sum())))
    attrs, texts, neighbors = map(torch.stack, (attrs, texts, neighbors))
    count = len(clusters)
    gen = torch.Generator().manual_seed(int(shuffle_seed))
    order = torch.randperm(count, generator=gen)
    permutation = torch.arange(count)
    if count > 1:
        permutation[order] = torch.roll(order, 1)
    graph = F.normalize(attrs + neighbors, dim=1)
    shuffled = F.normalize(attrs + neighbors[permutation], dim=1)
    return dict(attribute=attrs, text=texts, graph=graph, shuffled=shuffled,
                neighbors=neighbors, permutation=permutation, stats=stats)


def select_snippets(bank, members, query, tokenizer, budget=BUDGET):
    if not 1 <= budget <= 240:
        raise ValueError("invalid token cap")
    ids = torch.where(torch.isin(bank["owners"], members))[0].tolist()
    if not ids:
        return [], [], []
    keys = F.normalize(bank["embeddings"], dim=1)
    relevance = keys @ F.normalize(query, dim=0)
    selected, used = [], 0
    while ids and len(selected) < MAX_PARTS and used < budget:
        def score(i):
            redundancy = max(float(keys[i] @ keys[j]) for j in selected) if selected else 0.
            return (RELEVANCE * float(relevance[i]) - (1 - RELEVANCE) * redundancy, -i)
        chosen = max(ids, key=score)
        selected.append(chosen)
        ids.remove(chosen)
        used += T.token_count(tokenizer, bank["texts"][chosen])
    # Canonical source/offset order removes attention-order effects on MiniLM.
    selected.sort()
    parts = T.bounded_parts(tokenizer, [bank["texts"][i] for i in selected], budget)
    kept = [(i, p) for i, p in zip(selected, parts) if p.strip()]
    return [i for i, _ in kept], [p for _, p in kept], [float(relevance[i]) for i, _ in kept]


def construct(arm, joint, source_embeddings, members, texts, encoder,
              bank, queries, row, budget=BUDGET):
    if arm not in ARMS:
        raise ValueError(arm)
    tokenizer = encoder.tokenizer
    if arm in {"center_concat", "mean_pool", "text_attention"}:
        op = {"center_concat": "concat", "mean_pool": "mean_pool",
              "text_attention": "set_pool"}[arm]
        fused, target, audit = T.construct(joint, source_embeddings, members, texts,
                                          encoder, op, budget, MAX_PARTS)
        chosen = T.select_members(joint, source_embeddings, members, MAX_PARTS, False, .5)
        audit.update(selected_source_local_ids=chosen.tolist(), operator=arm)
        return fused, target, audit
    query_key = {"text_selection": "text", "attribute_selection": "attribute",
                 "graph_selection": "graph", "graph_pool": "graph",
                 "shuffled_graph": "shuffled"}[arm]
    query = queries[query_key][row]
    selected, parts, relevance = select_snippets(bank, members, query, tokenizer, budget)
    fused = " ".join(parts)
    weights = []
    if arm == "graph_pool" and parts:
        inputs = parts
        encoded = encoder.encode(parts)
        attention = torch.softmax((F.normalize(encoded, dim=1) @ query) / TEMPERATURE, dim=0)
        target = F.normalize((encoded * attention[:, None]).sum(0), dim=0)
        weights = attention.tolist()
    else:
        inputs = [fused]
        target = encoder.encode(inputs)[0]
    content = sum(T.token_count(tokenizer, p) for p in inputs)
    retained = T.token_count(tokenizer, fused)
    if max(content, retained) > budget:
        raise AssertionError("text token budget exceeded")
    audit = dict(operator=arm, content_token_cap=budget, retained_tokens=retained,
                 encoder_content_tokens=content, encoder_sequences=len(inputs),
                 encoder_tokens_with_special=sum(len(tokenizer.encode(p, add_special_tokens=True))
                                                 for p in inputs),
                 empty=not bool(fused), inputs=inputs, snippet_ids=selected,
                 selected_source_local_ids=[int(bank["owners"][i]) for i in selected],
                 query_sha256=tensor_sha(query), attention_weights=weights,
                 relevance=relevance,
                 input_sha256=hashlib.sha256("\0".join(inputs).encode()).hexdigest())
    return fused, target, audit
