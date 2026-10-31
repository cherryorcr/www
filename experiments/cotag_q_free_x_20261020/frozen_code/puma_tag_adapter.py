"""PUMA-inspired TAG adaptation for the frozen large-graph protocol.

This module keeps the mechanisms that distinguish PUMA from a plain CaT
adapter: one-time propagation, confidence-filtered pseudo labels, edge-free
memory and construction from a current delta.  It is deliberately named an
adaptation rather than an official PUMA reproduction because the downstream
Student is the shared zero-shot TAG Student used by this project.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier

import external_continual_mechanisms as M


PUMA_ARMS = ("puma_full", "puma_no_pseudo", "puma_shuffled_pseudo")


@dataclass(frozen=True)
class PUMAConfig:
    seed: int = 44
    pseudo_threshold: float = 0.80
    split_modulus: int = 5
    split_remainder: int = 0
    teacher_max_rows: int = 50000
    teacher_steps: int = 80
    text_sources: int = 5
    text_chars: int = 320


def _stable_hash(values: torch.Tensor) -> str:
    value = values.detach().cpu().long().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _ensure_teacher_support(labels: torch.Tensor, candidate: torch.Tensor,
                            teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one candidate row for any missing class into teacher support."""
    candidate = candidate.clone()
    teacher = teacher.clone()
    classes = torch.unique(labels, sorted=True)
    for cls in classes.tolist():
        if not bool((labels[teacher] == int(cls)).any()):
            available = torch.where(candidate & (labels == int(cls)))[0]
            if len(available):
                candidate[available[0]] = False
                teacher[available[0]] = True
    return candidate, teacher


def one_time_propagation(delta: dict, device: str) -> torch.Tensor:
    """Apply exactly one normalized hop on the current delta graph."""
    propagated = M.propagated_features(delta, (0.0, 1.0), device)
    return propagated.detach().cpu().float()


def _teacher_features(propagated: torch.Tensor,
                      text_emb: torch.Tensor) -> np.ndarray:
    graph = F.normalize(propagated.float(), dim=1)
    text = F.normalize(text_emb.float(), dim=1)
    return torch.cat((graph, text), dim=1).cpu().numpy().astype(np.float32)


def train_pseudo_teacher(propagated: torch.Tensor, text_emb: torch.Tensor,
                         labels: torch.Tensor, teacher_rows: torch.Tensor,
                         seed: int, max_rows: int, max_iter: int):
    """Train a bounded linear teacher on legal current-delta rows only."""
    support = teacher_rows.nonzero(as_tuple=False).flatten()
    if not len(support):
        raise ValueError("PUMA teacher has no legal support rows")
    if len(support) > int(max_rows):
        generator = torch.Generator().manual_seed(int(seed) + 901)
        selected = torch.randperm(len(support), generator=generator)[:int(max_rows)]
        support = support[selected]
    features = _teacher_features(propagated, text_emb)
    labels_np = labels.cpu().numpy().astype(np.int64)
    support_labels = labels_np[support.numpy()]
    unique_support = np.unique(support_labels)
    if len(unique_support) == 1:
        only = int(unique_support[0])
        return (torch.full((len(labels),), only, dtype=torch.long),
                torch.ones(len(labels), dtype=torch.float32),
                {"teacher": "single-class constant teacher",
                 "teacher_rows": int(len(support)),
                 "teacher_classes": [only],
                 "teacher_max_iter": 0,
                 "teacher_max_rows": int(max_rows)})
    teacher = SGDClassifier(
        loss="log_loss",
        max_iter=int(max_iter),
        tol=1e-3,
        alpha=1e-4,
        class_weight="balanced",
        random_state=int(seed),
        average=True,
    )
    teacher.fit(features[support.numpy()], labels_np[support.numpy()])
    probabilities = teacher.predict_proba(features)
    classes = torch.from_numpy(teacher.classes_.astype(np.int64))
    winner = probabilities.argmax(axis=1)
    predicted = classes[torch.from_numpy(winner).long()]
    confidence = torch.from_numpy(probabilities.max(axis=1).astype(np.float32))
    return predicted.long(), confidence, {
        "teacher": "SGDClassifier(log_loss) on one-hop graph plus frozen text",
        "teacher_rows": int(len(support)),
        "teacher_classes": [int(v) for v in teacher.classes_.tolist()],
        "teacher_max_iter": int(max_iter),
        "teacher_max_rows": int(max_rows),
    }


def _subset_delta(delta: dict, propagated: torch.Tensor, selected: torch.Tensor,
                  selected_labels: torch.Tensor) -> dict:
    selected = selected.long().cpu()
    out = {
        "ids": delta["ids"][selected].clone(),
        "stage": int(delta["stage"]),
        "x": propagated[selected].clone(),
        "y": selected_labels[selected].clone().long(),
        "text_emb": delta["text_emb"][selected].clone().float(),
        "texts": [delta["texts"][int(i)] for i in selected.tolist()],
        "prompt_emb": delta.get("prompt_emb"),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    return out


def build_puma_block(delta: dict, rows: int, encoder, cfg: PUMAConfig,
                     device: str, arm: str) -> tuple[dict, dict]:
    """Build one PUMA-inspired edge-free synthetic block."""
    if arm not in PUMA_ARMS:
        raise ValueError(f"unknown PUMA arm: {arm}")
    if int(rows) <= 0:
        raise ValueError("PUMA block requires positive rows")

    propagated = one_time_propagation(delta, device)
    original_ids = delta["ids"].long().cpu()
    candidate_mask = (original_ids.remainder(int(cfg.split_modulus))
                      == int(cfg.split_remainder))
    candidate_mask, teacher_mask = _ensure_teacher_support(
        delta["y"].long().cpu(), candidate_mask, ~candidate_mask)
    predicted, confidence, teacher_info = train_pseudo_teacher(
        propagated, delta["text_emb"].cpu(), delta["y"].long().cpu(),
        teacher_mask, int(cfg.seed) + 1009 * int(delta["stage"]),
        int(cfg.teacher_max_rows), int(cfg.teacher_steps))

    selected_labels = delta["y"].long().cpu().clone()
    confident = candidate_mask & (confidence >= float(cfg.pseudo_threshold))
    pseudo_rows = torch.where(confident)[0]
    if arm == "no_pseudo":
        selected = torch.where(teacher_mask)[0]
        pseudo_rows = torch.empty(0, dtype=torch.long)
    else:
        pseudo_labels = predicted[pseudo_rows].clone()
        if arm == "puma_shuffled_pseudo" and len(pseudo_labels) > 1:
            generator = torch.Generator().manual_seed(int(cfg.seed) + 37)
            pseudo_labels = pseudo_labels[torch.randperm(
                len(pseudo_labels), generator=generator)]
        selected_labels[pseudo_rows] = pseudo_labels
        selected = torch.cat((torch.where(teacher_mask)[0], pseudo_rows))
    if len(selected) == 0:
        raise ValueError("PUMA confidence filtering removed all current rows")

    local = _subset_delta(delta, propagated, selected, selected_labels)
    cat_cfg = M.Config(
        seed=int(cfg.seed) + 1009 * int(delta["stage"]),
        cat_steps=3000,
        cat_hidden=512,
        cat_lr=0.001,
        text_sources=int(cfg.text_sources),
        text_chars=int(cfg.text_chars),
        text_mode="nearest",
    )
    block, cat_audit = M.cat_block(local, int(rows), encoder, cat_cfg, device)
    candidate_count = int(candidate_mask.sum())
    audit = {
        "method": "PUMA-inspired-TAG-adapted",
        "arm": arm,
        "stage": int(delta["stage"]),
        "one_time_propagation": True,
        "propagation": "one normalized hop on current-delta induced graph",
        "pseudo_labeling": arm != "puma_no_pseudo",
        "confidence_threshold": float(cfg.pseudo_threshold),
        "candidate_rows": candidate_count,
        "teacher_rows": int(teacher_mask.sum()),
        "pseudo_rows": int(len(pseudo_rows)),
        "selected_rows": int(len(selected)),
        "candidate_ids_sha256": _stable_hash(original_ids[candidate_mask]),
        "teacher_ids_sha256": _stable_hash(original_ids[teacher_mask]),
        "selected_ids_sha256": _stable_hash(local["ids"]),
        "teacher": teacher_info,
        "cat_distribution_matching": cat_audit,
        "text_attachment": "nearest current-delta text under pseudo/teacher labels, top-5 re-encoded",
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology": "edge-free; Student supplies self-loops",
        "official_reproduction": False,
    }
    return block, audit
