"""Compact read-only diagnostics for real-text projection comparisons."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


@torch.no_grad()
def assignment_metrics(q, raw_text, raw_labels, labels, selected, texts):
    labels = labels.to(q.device)
    raw_labels = raw_labels.to(q.device)
    text = F.normalize(raw_text.to(q.device).float(), dim=1)
    mass = q.sum(0).clamp_min(1e-12)
    soft = F.normalize(q.T @ text / mass[:, None], dim=1)
    hard = text[selected]
    gaps = 1 - (soft * hard).sum(1)
    row_entropy_sum = normalized_entropy_sum = 0.0
    slot_entropy_sum = 0.0
    classes = []
    for cls in torch.unique(labels, sorted=True):
        raw = torch.where(raw_labels == cls)[0]
        slots = torch.where(labels == cls)[0]
        local = q.index_select(0, raw).index_select(1, slots)
        safe = local.clamp_min(1e-12)
        entropy = -(local * safe.log()).sum(1)
        row_entropy_sum += float(entropy.sum())
        if len(slots) > 1:
            normalized_entropy_sum += float(entropy.sum()) / math.log(len(slots))
        distribution = local / local.sum(0).clamp_min(1e-12)[None, :]
        slot_entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(0)
        slot_entropy_sum += float(slot_entropy.sum())
        cover_sum = 0.0
        for start in range(0, len(raw), 1024):
            emb = text[raw[start:start + 1024]]
            cover_sum += float((emb @ hard[slots].T).max(1).values.sum())
        sampled = hard[slots[:512]]
        pair_cosine = None
        if len(sampled) > 1:
            similarities = sampled @ sampled.T
            pair_cosine = float((similarities.sum() - similarities.diag().sum()) /
                                (len(sampled) * (len(sampled) - 1)))
        classes.append(dict(class_id=int(cls), raw_rows=len(raw), slots=len(slots),
            unique_sources=int(selected[slots].unique().numel()),
            class_coverage_cosine=cover_sum / len(raw),
            within_class_text_cosine=pair_cosine,
            pairwise_sample_slots=len(sampled),
            soft_hard_cosine_gap=float(gaps[slots].mean())))
    return dict(row_entropy=row_entropy_sum / len(q),
        normalized_row_entropy=normalized_entropy_sum / len(q),
        slot_source_entropy=slot_entropy_sum / q.shape[1],
        hard_slot_coverage=float(q.argmax(1).unique().numel() / q.shape[1]),
        positive_mass_slot_coverage=float((mass > 1e-6).float().mean()),
        unique_source_ratio=float(selected.unique().numel() / q.shape[1]),
        unique_text_ratio=len(set(texts)) / len(texts),
        soft_hard_cosine_gap_mean=float(gaps.mean()),
        soft_hard_cosine_gap_max=float(gaps.max()),
        class_coverage_cosine=sum(c["class_coverage_cosine"] for c in classes) / len(classes),
        per_class=classes,
        definitions="row entropy is over slot probabilities; source entropy is over column-normalized raw supports; class coverage is mean nearest same-class hard-text cosine")


class CoverageDiagnostics:
    def __init__(self):
        self.result = None
        self.seconds = 0.0

    def __call__(self, state):
        started = time.perf_counter()
        q, delta, block = state["q"], state["delta"], state["block"]
        positions = {int(value): i for i, value in enumerate(delta["ids"].tolist())}
        selected = torch.tensor([positions[sources[0]] for sources in block["text_source_ids"]],
                                dtype=torch.long, device=q.device)
        labels = block["labels"]
        self.result = assignment_metrics(q, delta["text_emb"], delta["y"], labels,
                                         selected, block["texts"])
        self.result["q_relative_update"] = float((q - state["q0"]).norm() /
                                                state["q0"].norm().clamp_min(1e-12))
        if q.is_cuda:
            torch.cuda.synchronize(q.device)
        self.seconds = time.perf_counter() - started

