"""Native TAGSAM neighborhood/PMI selector, with explicit interface repairs."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from uq_high_fidelity_baselines_20261012 import pinned_functions, verified_repo
import run_unified_q_matrix_20261008 as R

REPO = Path("/home/pc/research/TAGSAM_official")
COMMIT = "d812f06fd16db3cd4d497804a8e504d2c738faa8"
GPT_PATH = "/home/pc/huggingface-cache/models--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"


def native_language_model(device, repo=REPO):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    namespace = dict(torch=torch, np=np, GPT2Tokenizer=GPT2Tokenizer, GPT2LMHeadModel=GPT2LMHeadModel)
    pinned_functions(repo / "model.py", {"GPT2"}, namespace)
    model = namespace["GPT2"](device=device, location=GPT_PATH)
    selector = dict(math=math, np=np)
    pinned_functions(repo / "selection.py", {"get_probabilities", "get_npmi_matrix"}, selector)
    return model, selector


def pmi_matrix(model, sentences, native_word_probs, batch=8):
    """Native word-probability PMI; flush every ordered pair including the tail."""
    count = len(sentences)
    matrix = np.zeros((count, count))
    np.fill_diagonal(matrix, -1.)
    floor_count = 0

    def log_sum(probabilities):
        nonlocal floor_count
        floor_count += sum(p <= 0 for p in probabilities)
        return sum(math.log(max(p, np.finfo(np.float64).tiny)) for p in probabilities)

    single = native_word_probs(model, sentences)
    base = [log_sum(prob) for prob in single]
    pairs = [(i, j) for i in range(count) for j in range(count) if i != j]
    for start in range(0, len(pairs), batch):
        indices = pairs[start:start + batch]
        articles = [sentences[i] + " " + sentences[j] for i, j in indices]
        probabilities = native_word_probs(model, articles)
        for (i, j), probs in zip(indices, probabilities):
            conditional = log_sum(probs[len(sentences[i].split()):])
            matrix[i, j] = conditional - base[j]
    return matrix, dict(probability_floors=floor_count, ordered_pairs=len(pairs),
                        tail_batch_pairs=len(pairs) % batch)


def greedy_groups(matrix, ratio=.6):
    matrix = np.maximum(matrix, 0.)
    relevance = matrix.sum(1)
    penalty = np.zeros(len(matrix))
    selected = []
    for _ in range(math.ceil(len(matrix) * ratio)):
        score = relevance - penalty
        if selected:
            score[selected] = -np.inf
        index = int(np.argmax(score))
        penalty += matrix[:, index]
        selected.append(index)
    return sorted(selected)


def predecessor_csr(edges, rows):
    edges = edges.cpu().numpy()
    return sp.csr_matrix((np.ones(edges.shape[1], dtype=np.int8), (edges[1], edges[0])),
                         shape=(rows, rows))


def native_subset(center, predecessors, size=4):
    subset, frontier = {int(center)}, {int(center)}
    for _ in range(4):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(predecessors.indices[predecessors.indptr[node]:predecessors.indptr[node + 1]])
        frontier = next_frontier - subset
        subset.update(next_frontier)
        if len(subset) >= size:
            break
    return np.array(sorted(subset)[:size], dtype=np.int64)


def selector_block(delta, rows, encoder, seed, device):
    import spacy
    provenance = verified_repo(REPO, COMMIT, ("model.py", "selection.py", "distill.py", "dataset.py", "epoch.py"))
    model, native = native_language_model(device)
    nlp = spacy.load("en_core_web_sm")
    rng = np.random.RandomState(seed + 1009 * int(delta["stage"]))
    centers = rng.permutation(len(delta["x"]))[:rows]
    predecessors = predecessor_csr(delta["edge_index"], len(delta["x"]))
    sentence_cache, pmi_cache = {}, {}
    texts, features, supports, audit = [], [], [], Counter()

    def groups_for(node):
        if node not in sentence_cache:
            text = " ".join(str(delta["texts"][node]).split())[:320]
            sentences = ["".join(ch if ord(ch) < 128 else " " for ch in sent.text)
                         for sent in nlp(text).sents]
            groups = [" ".join(" ".join(sentences[i:i + 5]).split())
                      for i in range(0, len(sentences), 5)]
            sentence_cache[node] = [group for group in groups if group]
        return sentence_cache[node]

    for slot, center in enumerate(centers):
        subset = native_subset(center, predecessors)
        groups, ranges = [], []
        for node in subset:
            start = len(groups)
            groups.extend(groups_for(int(node)))
            ranges.append((start, len(groups)))
        if not groups:
            audit["empty_neighborhood_fallbacks"] += 1
            groups = ["No text available."]
            subset = np.array([center])
            ranges = [(0, 1)]
        key = tuple(groups)
        if key not in pmi_cache:
            matrix, details = pmi_matrix(model, groups, native["get_probabilities"])
            pmi_cache[key] = greedy_groups(matrix)
            audit.update(details)
        else:
            audit["duplicate_neighborhood_hits"] += 1
        selected = pmi_cache[key]
        text = " ".join(groups[index] for index in selected)[:320]
        weighted = torch.zeros_like(delta["x"][0])
        for node, (start, end) in zip(subset, ranges):
            if end == start:
                audit["empty_source_groups"] += 1
                continue
            ratio = sum(start <= index < end for index in selected) / (end - start)
            weighted += delta["x"][int(node)] * ratio
        texts.append(text)
        features.append(weighted)
        supports.append(delta["ids"][torch.from_numpy(subset)].tolist())
        if (slot + 1) % 100 == 0:
            print(f"TAGSAM_PMI stage={delta['stage']} slots={slot + 1}/{rows}", flush=True)
    del model, nlp
    torch.cuda.empty_cache()
    labels = delta["y"][torch.from_numpy(centers)].long().cpu()
    block = dict(x=torch.stack(features).float().cpu(), text_emb=encoder.encode(texts).cpu(),
        texts=texts, labels=labels, mass=torch.full((rows,), len(delta["x"]) / rows),
        depth=torch.zeros(rows, dtype=torch.long), stage=torch.full((rows,), delta["stage"], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long))
    return block, dict(native_source=provenance, selector="native GPT2-large word-probability PMI + greedy redundancy",
        center_ids=delta["ids"][torch.from_numpy(centers)].tolist(), source_ids=supports,
        summary_ratio=.6, max_sources=4, max_hops=4, sentence_group_size=5, diagnostics=dict(audit),
        label_rule="current-delta random center label; not stratified quota labels",
        adaptations=["current delta only; frozen prefix", "candidate and final text cap 320 characters",
            "common MiniLM replaces native BERT", "normalize whitespace before native word alignment",
            "flush native missing trailing PMI pair batch", "empty-source groups contribute zero feature weight",
            "zero rounded probabilities use float64 tiny before log"], official_reproduction=False)


def build_tagsam(delta, parent, rows, encoder, seed, device):
    import run_unified_q_tagsam_20261008 as A
    T = A.T
    initial, selector = selector_block(delta, rows, encoder, seed, device)
    raw = A.reference_memory(delta, parent)
    loss, provenance = T.official_loss(REPO)
    teacher = T.G.train_student(raw, 9000 + delta["stage"], device, 2048, lr=.001, batch=24)
    memory, response = T.optimize(initial, parent, raw, teacher, delta["stage"],
                                  "tagsam_selfloop", device, loss, 100)
    keep = 0 if parent is None else len(parent["x"])
    block = {key: value[keep:] for key, value in memory.items()
             if key in ("x", "text_emb", "texts", "mass", "depth", "stage")}
    block.update(labels=initial["labels"], edge_index=torch.empty((2, 0), dtype=torch.long))
    return block, dict(selector=selector, response=response, native_loss=provenance,
        outer_steps=100, inner_steps=10, reference_cap=256, teacher_steps=2048,
        adaptations=["existing common-architecture response adapter", "100x10 unroll vs native 5000x15",
                     "fixed common final Student optimizer, not distilled learning rates"],
        official_reproduction=False)
