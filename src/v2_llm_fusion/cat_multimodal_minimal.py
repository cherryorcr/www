"""Minimal multimodal extension of the current CaT-style constructor.

This module intentionally does not add a Teacher, Student unroll, topology
revision, or old-memory refresh.  It only tests whether the CaT class-wise
random-feature objective can optimize the current-stage continuous text
embeddings together with (or independently of) its X initialization.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re

import torch
import torch.nn.functional as F

import external_continual_mechanisms as M


@dataclass(frozen=True)
class MMConfig:
    steps: int = 300
    hidden: int = 256
    lr_text: float = 5e-3
    lr_x: float = 1e-3
    lambda_text: float = 1.0
    lambda_cross: float = 0.25
    lambda_anchor_text: float = 0.05
    lambda_anchor_x: float = 0.05
    assignment_temperature: float = 0.07
    text_sources: int = 3
    text_chars: int = 320
    text_ridge: float = 1e-2
    text_group_count: int = 4
    lambda_group_entropy: float = 0.02
    lambda_prompt_margin: float = 0.15
    prompt_margin: float = 0.05
    lambda_query_anchor: float = 0.5
    x_trust_radius: float = 0.05
    lambda_cctp_anchor: float = 0.5
    joint_base_logit: float = 1.5
    joint_selection_margin: float = 0.05
    lambda_joint_base: float = 0.5
    joint_x_trust_radius: float = 0.01
    support_pool_size: int = 32
    chunk_chars: int = 120
    payload_max_chunks: int = 8
    payload_max_sources: int = 6
    payload_graph_weight: float = 0.35
    payload_text_weight: float = 0.45
    payload_prompt_weight: float = 0.20
    payload_redundancy: float = 0.20
    payload_require_lead: bool = True
    payload_contrastive_weight: float = 0.35
    payload_source_repeat_penalty: float = 0.10
    # Class-level assignment weights.  These terms are evaluated over the
    # complete synthetic rows of a class, not only over its representative.
    payload_assignment_text_weight: float = 0.50
    payload_assignment_variance_weight: float = 0.25
    seed: int = 271003


def _bound(dim: int, hidden: int) -> float:
    return math.sqrt(6.0 / float(dim + hidden))


def _class_mmd(raw: torch.Tensor, synth: torch.Tensor,
               raw_labels: torch.Tensor, synth_labels: torch.Tensor,
               weight: torch.Tensor) -> torch.Tensor:
    real_features = M.random_features(raw, weight)
    synth_features = M.random_features(synth, weight)
    return M.class_mmd(real_features, synth_features,
                       raw_labels, synth_labels) / max(1, weight.shape[1])


def _soft_text_payload(source_features: torch.Tensor,
                       raw_text: torch.Tensor,
                       raw_labels: torch.Tensor,
                       x_value: torch.Tensor,
                       labels: torch.Tensor,
                       graph_text_map: torch.Tensor,
                       temperature: float) -> tuple[torch.Tensor, list[dict]]:
    """Build differentiable class-restricted text payloads from current X.

    The graph-to-text map is fitted on the current delta only.  Each synthetic
    row produces a text query from its current X, then softly transports mass
    over raw text candidates of the same class.  Keeping the transport soft
    during CaT optimization makes the text term differentiable with respect to
    X while preserving a legal, auditable source pool.
    """
    graph_query = F.normalize(x_value.float(), dim=1)
    ones = torch.ones((len(graph_query), 1), device=x_value.device,
                      dtype=graph_query.dtype)
    predicted_text = F.normalize(
        torch.cat((graph_query, ones), dim=1) @ graph_text_map, dim=1)
    raw_text = F.normalize(raw_text.float(), dim=1)
    soft_rows = []
    audits = []
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        candidates = torch.where(raw_labels == int(cls))[0]
        if len(candidates) == 0:
            raise ValueError(f"no text candidates for class {cls}")
        scores = (predicted_text[row:row + 1]
                  @ raw_text.index_select(0, candidates).T).squeeze(0)
        probabilities = F.softmax(scores / max(float(temperature), 1e-6),
                                   dim=0)
        soft_rows.append(probabilities @ raw_text.index_select(0, candidates))
        with torch.no_grad():
            entropy = -(probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()).sum()
            order = torch.argsort(probabilities, descending=True)
            audits.append({
                "class": int(cls),
                "candidate_ids": candidates[order].detach().cpu().tolist(),
                "candidate_probabilities": probabilities[order].detach().cpu().tolist(),
                "entropy": float(entropy.cpu()),
                "top1_probability": float(probabilities[order[0]].cpu()),
            })
    soft_text = F.normalize(torch.stack(soft_rows), dim=1)
    return soft_text, audits


def _materialize_soft_payload(delta: dict, labels: torch.Tensor,
                              assignments: list[dict], encoder,
                              text_sources: int,
                              text_chars: int) -> tuple[list[str], list[list[int]],
                                                        torch.Tensor, list[dict]]:
    """Turn final soft assignments into provenance strings and re-encoded text."""
    texts, source_ids, materialization = [], [], []
    for record in assignments:
        candidates = record["candidate_ids"]
        probabilities = torch.tensor(record["candidate_probabilities"])
        k = min(int(text_sources), len(candidates))
        chosen = [int(candidates[i]) for i in torch.argsort(
            probabilities, descending=True)[:k].tolist()]
        separator_budget = max(0, len(chosen) - 1)
        per = max(1, (int(text_chars) - separator_budget) // max(1, len(chosen)))
        text = " ".join(str(delta["texts"][i])[:per] for i in chosen)
        texts.append(text[:int(text_chars)])
        source_ids.append([int(delta["ids"][i]) for i in chosen])
        materialization.append({
            "chosen_local_indices": chosen,
            "chosen_source_ids": source_ids[-1],
            "text_length": len(texts[-1]),
        })
    hard_text = encoder.encode(texts).cpu()
    return texts, source_ids, F.normalize(hard_text, dim=1), materialization


def _concat_text_group(delta: dict, indices: list[int],
                       text_chars: int) -> str:
    """Materialize one legal text group using the same CCTP budget rule."""
    if "texts" not in delta:
        raise ValueError("text-group construction requires raw delta texts")
    separator_budget = max(0, len(indices) - 1)
    per = max(1, (int(text_chars) - separator_budget) // max(1, len(indices)))
    value = " ".join(str(delta["texts"][i])[:per] for i in indices)
    return value[:int(text_chars)]


def _group_key(indices: list[int]) -> tuple[int, ...]:
    """Use source order in the key because MiniLM sees concatenation order."""
    return tuple(int(i) for i in indices)


def _split_text_chunks(text: str, chunk_chars: int) -> list[dict]:
    """Split one source string into real, auditable text chunks.

    The offsets refer to the normalized string that is actually materialized.
    Sentence boundaries are preferred, but long sentences are cut into bounded
    windows so the candidate generator cannot create an unbounded payload.
    """
    limit = max(8, int(chunk_chars))
    normalized = " ".join(str(text).split())
    if not normalized:
        return [{"text": "", "start": 0, "end": 0}]
    pieces = [piece.strip() for piece in re.split(
        r"(?<=[.!?;])\s+|(?<=[。！？；])", normalized) if piece.strip()]
    if not pieces:
        pieces = [normalized]
    chunks: list[dict] = []
    cursor = 0
    for piece in pieces:
        start = normalized.find(piece, cursor)
        if start < 0:
            start = cursor
        offset = 0
        while offset < len(piece):
            end_offset = min(len(piece), offset + limit)
            if end_offset < len(piece):
                boundary = piece.rfind(" ", offset + 8, end_offset)
                if boundary > offset:
                    end_offset = boundary
            fragment = piece[offset:end_offset].strip()
            core = fragment.strip(" \t.,;:!?()[]{}")
            if fragment and core:
                fragment_start = start + offset
                chunks.append({
                    "text": fragment,
                    "start": int(fragment_start),
                    "end": int(fragment_start + len(fragment)),
                })
            if end_offset <= offset:
                end_offset = min(len(piece), offset + limit)
            offset = end_offset
            while offset < len(piece) and piece[offset].isspace():
                offset += 1
        cursor = start + len(piece)
    return chunks or [{"text": normalized[:limit], "start": 0,
                       "end": min(limit, len(normalized))}]


def _materialize_chunk_payload(chunks: list[dict], selected: list[int],
                               text_chars: int, max_sources: int) -> dict:
    """Materialize a variable-length chunk payload under a hard budget."""
    text_parts: list[str] = []
    selected_records: list[dict] = []
    source_ids: list[int] = []
    used_sources: set[int] = set()
    current_length = 0
    for index in selected:
        record = chunks[int(index)]
        source_local = int(record["source_local_index"])
        if source_local not in used_sources and len(used_sources) >= max(1, int(max_sources)):
            continue
        value = str(record["text"])
        separator = 1 if text_parts else 0
        remaining = int(text_chars) - current_length - separator
        if remaining <= 0:
            break
        if len(value) > remaining:
            value = value[:remaining].rstrip()
            boundary = value.rfind(" ")
            if boundary >= max(8, remaining // 2):
                value = value[:boundary].rstrip()
        if not value:
            continue
        text_parts.append(value)
        current_length += separator + len(value)
        used_sources.add(source_local)
        if source_local not in source_ids:
            source_ids.append(source_local)
        selected_records.append({
            "chunk_index": int(index),
            "source_local_index": source_local,
            "source_id": int(record["source_id"]),
            "start": int(record["start"]),
            "end": int(record["end"]),
            "text": value,
        })
        if current_length >= int(text_chars):
            break
    return {
        "text": " ".join(text_parts)[:int(text_chars)],
        "chunks": selected_records,
        "source_local_indices": source_ids,
        "source_ids": list(dict.fromkeys(
            int(record["source_id"]) for record in selected_records)),
    }


def _select_payload_chunks(pool: dict, mode: str, cfg: MMConfig) -> tuple[list[int], dict]:
    """Greedily select real chunks without enumerating source combinations.

    ``tagsam_chunk`` is a bounded TAGSAM-inspired proxy: it rewards marginal
    coverage of the local support texts and penalizes redundant chunks.  It is
    explicitly labeled a proxy rather than an official TAGSAM reproduction.
    """
    if mode not in ("text_mmr", "tagsam_chunk", "graph_role"):
        raise ValueError(f"unknown chunk payload mode: {mode}")
    embeddings = F.normalize(pool["chunk_embeddings"].float(), dim=1)
    graph_score = pool["graph_score"].float()
    prompt_score = pool["prompt_score"].float()
    role_score = pool["role_score"].float()
    contrastive_score = pool["contrastive_score"].float()
    support_text = F.normalize(pool["support_text_embeddings"].float(), dim=1)
    if mode == "text_mmr":
        base_score = (float(cfg.payload_text_weight) * pool["text_center_score"]
                      + float(cfg.payload_contrastive_weight) * contrastive_score
                      + float(cfg.payload_prompt_weight) * prompt_score)
    elif mode == "graph_role":
        base_score = (float(cfg.payload_graph_weight) * graph_score
                      + 0.45 * role_score
                      + float(cfg.payload_contrastive_weight) * contrastive_score
                      + float(cfg.payload_prompt_weight) * prompt_score)
    else:
        base_score = (float(cfg.payload_graph_weight) * graph_score
                      + float(cfg.payload_contrastive_weight) * contrastive_score
                      + float(cfg.payload_prompt_weight) * prompt_score)
    selected: list[int] = []
    available = torch.ones(len(embeddings), dtype=torch.bool)
    coverage = torch.zeros(len(support_text), dtype=torch.float32)
    trace: list[dict] = []
    for step in range(max(1, int(cfg.payload_max_chunks))):
        adjusted = base_score.clone()
        marginal = torch.zeros(len(embeddings), dtype=torch.float32)
        if mode == "tagsam_chunk":
            similarities = (embeddings @ support_text.T).clamp_min(0.0)
            marginal = (torch.maximum(coverage.unsqueeze(0), similarities).mean(1)
                        - coverage.mean())
            adjusted = adjusted + float(cfg.payload_text_weight) * marginal
        if selected:
            selected_embeddings = embeddings[torch.tensor(selected)]
            redundancy = (embeddings @ selected_embeddings.T).max(1).values
            adjusted = adjusted - float(cfg.payload_redundancy) * redundancy
            selected_sources = {
                int(pool["chunks"][value]["source_local_index"])
                for value in selected
            }
            for index, chunk in enumerate(pool["chunks"]):
                if int(chunk["source_local_index"]) in selected_sources:
                    adjusted[index] -= float(cfg.payload_source_repeat_penalty)
        if (not selected and bool(cfg.payload_require_lead)
                and bool(pool["lead_mask"].any())):
            adjusted[~pool["lead_mask"]] = -torch.inf
        adjusted[~available] = -torch.inf
        feasible = []
        for index in torch.argsort(adjusted, descending=True).tolist():
            trial = _materialize_chunk_payload(
                pool["chunks"], selected + [int(index)],
                int(cfg.text_chars), int(cfg.payload_max_sources))
            if trial["chunks"] and int(index) in [r["chunk_index"]
                                                   for r in trial["chunks"]]:
                feasible.append(int(index))
                break
        if not feasible:
            break
        best = feasible[0]
        selected.append(best)
        available[best] = False
        if mode == "tagsam_chunk":
            similarities = (embeddings[best] @ support_text.T).clamp_min(0.0)
            coverage = torch.maximum(coverage, similarities)
        trace.append({
            "step": int(step + 1),
            "chunk_index": int(best),
            "base_score": float(base_score[best]),
            "marginal_coverage": float(marginal[best]),
        })
        materialized = _materialize_chunk_payload(
            pool["chunks"], selected, int(cfg.text_chars),
            int(cfg.payload_max_sources))
        if len(materialized["chunks"]) < len(selected):
            selected.pop()
            break
        if len(materialized["text"]) >= int(cfg.text_chars):
            break
    if not selected:
        best = int(torch.argmax(base_score).item())
        selected = [best]
    return selected, {
        "selection_trace": trace,
        "mode": mode,
        "selected_chunk_indices": [int(v) for v in selected],
    }


def _build_catted_chunk_support_pool(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        labels: torch.Tensor, encoder, cfg: MMConfig,
        base_source_ids: list[list[int]] | None = None) -> list[dict]:
    """Build bounded CaT-conditioned local support pools and chunk banks."""
    source_features = source_features.detach().cpu().float()
    # CaT optimizes X in the original node-feature space.  The propagated
    # feature tensor remains available for diagnostics, but should not define
    # the support pool against an unpropagated synthetic X0.
    support_features = delta.get("x", source_features).detach().cpu().float()
    x0 = x0.detach().cpu().float()
    raw_labels = delta["y"].detach().cpu().long()
    raw_text = F.normalize(delta["text_emb"].detach().cpu().float(), dim=1)
    prompt_emb = delta.get("prompt_emb")
    if prompt_emb is not None:
        prompt_emb = F.normalize(prompt_emb.detach().cpu().float(), dim=1)
    degrees = torch.zeros(len(raw_labels), dtype=torch.float32)
    edge_index = delta.get("edge_index")
    if edge_index is not None and edge_index.numel():
        degrees.scatter_add_(0, edge_index[0].cpu().long(),
                             torch.ones(edge_index.shape[1]))
        degrees.scatter_add_(0, edge_index[1].cpu().long(),
                             torch.ones(edge_index.shape[1]))
    pools: list[dict] = []
    all_classes = sorted(torch.unique(raw_labels).tolist())
    class_centers = {}
    for value in all_classes:
        class_rows = raw_text[raw_labels == int(value)]
        class_centers[int(value)] = F.normalize(
            class_rows.mean(0, keepdim=True), dim=1).squeeze(0)
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        candidates = torch.where(raw_labels == int(cls))[0]
        if not len(candidates):
            raise ValueError(f"no support nodes for class {cls}")
        graph = F.normalize(support_features[candidates], dim=1)
        query = F.normalize(x0[row].view(1, -1), dim=1).squeeze(0)
        similarity = graph @ query
        graph_order = torch.argsort(similarity, descending=True, stable=True)
        text_center_all = F.normalize(
            raw_text[candidates].mean(0, keepdim=True), dim=1).squeeze(0)
        text_similarity = raw_text[candidates] @ text_center_all
        text_order = torch.argsort(text_similarity, descending=True, stable=True)
        row_anchor_ids = [] if base_source_ids is None else (
            [int(v) for v in base_source_ids[row]]
            if row < len(base_source_ids) else [])
        id_to_local = {int(delta["ids"][int(i)]): int(i)
                       for i in candidates.tolist()}
        anchor_local = [id_to_local[v] for v in row_anchor_ids
                        if v in id_to_local]
        ordered_support = list(anchor_local)
        ordered_support += [int(candidates[i]) for i in graph_order.tolist()]
        ordered_support += [int(candidates[i]) for i in text_order.tolist()]
        support_values = []
        seen_support = set()
        for value in ordered_support:
            if value not in seen_support:
                seen_support.add(value)
                support_values.append(value)
            if len(support_values) >= max(1, int(cfg.support_pool_size)):
                break
        support = torch.tensor(support_values, dtype=torch.long)
        support_order = torch.tensor(
            [int((candidates == value).nonzero(as_tuple=False)[0])
             for value in support.tolist()], dtype=torch.long)
        support_text = raw_text[support]
        # The support pool is graph-conditioned, but a text-MMR candidate
        # should be centered on the whole current class, not on the graph
        # biased subset.  Otherwise graph selection leaks into the text-only
        # attribution arm.
        class_center = text_center_all
        class_degrees = degrees[support]
        degree_min, degree_max = class_degrees.min(), class_degrees.max()
        role = ((class_degrees - degree_min) /
                (degree_max - degree_min).clamp_min(1e-6))
        chunks: list[dict] = []
        for source_local in support.tolist():
            source_text = str(delta["texts"][int(source_local)])
            source_chunks = _split_text_chunks(source_text, cfg.chunk_chars)
            for chunk_index, chunk in enumerate(source_chunks[:8]):
                chunks.append({
                    "source_local_index": int(source_local),
                    "source_id": int(delta["ids"][int(source_local)]),
                    "start": int(chunk["start"]),
                    "end": int(chunk["end"]),
                    "text": str(chunk["text"]),
                    "is_lead": bool(chunk_index == 0),
                    "source_graph_score": float(
                        F.normalize(support_features[int(source_local)].view(1, -1),
                                    dim=1).squeeze(0) @ query),
                })
        chunk_texts = [item["text"] for item in chunks]
        chunk_embeddings = F.normalize(
            encoder.encode(chunk_texts).detach().cpu().float(), dim=1)
        source_graph_scores = torch.tensor(
            [item["source_graph_score"] for item in chunks], dtype=torch.float32)
        source_indices = torch.tensor(
            [item["source_local_index"] for item in chunks], dtype=torch.long)
        lead_mask = torch.tensor([bool(item["is_lead"]) for item in chunks],
                                 dtype=torch.bool)
        text_center_score = chunk_embeddings @ class_center
        negative_centers = [class_centers[int(value)] for value in all_classes
                            if int(value) != int(cls)]
        if negative_centers:
            negative_score = (chunk_embeddings @
                              torch.stack(negative_centers).T).max(1).values
        else:
            negative_score = torch.zeros(len(chunks), dtype=torch.float32)
        contrastive_score = text_center_score - negative_score
        prompt_score = torch.zeros(len(chunks), dtype=torch.float32)
        if prompt_emb is not None and int(cls) < len(prompt_emb):
            prompt_score = chunk_embeddings @ prompt_emb[int(cls)]
        role_score = role[(support.unsqueeze(1) == source_indices).long().argmax(0)]
        pools.append({
            "row": int(row),
            "class": int(cls),
            "support_local_indices": support.tolist(),
            "support_ids": [int(delta["ids"][int(i)]) for i in support.tolist()],
            "support_graph_scores": similarity[support_order].tolist(),
            "support_text_scores": text_similarity[support_order].tolist(),
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "support_text_embeddings": support_text,
            "text_center_score": text_center_score,
            "contrastive_score": contrastive_score,
            "graph_score": source_graph_scores,
            "prompt_score": prompt_score,
            "role_score": role_score,
            "lead_mask": lead_mask,
        })
    return pools


def _build_chunk_payload_bank(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, labels: torch.Tensor,
        base_texts: list[str], base_source_ids: list[list[int]],
        encoder, cfg: MMConfig) -> dict:
    """Create real candidate payloads from CaT-conditioned local chunks."""
    pools = _build_catted_chunk_support_pool(
        delta, source_features, x0, labels, encoder, cfg, base_source_ids)
    id_to_local = {int(value): i for i, value in
                   enumerate(delta["ids"].detach().cpu().tolist())}
    rows: list[dict] = []
    modes = ("text_mmr", "tagsam_chunk", "graph_role", "anchor_mix")
    for row, pool in enumerate(pools):
        records: dict[str, dict] = {}
        base_ids = [int(v) for v in (base_source_ids[row]
                                     if row < len(base_source_ids) else [])]
        base_local = [id_to_local[v] for v in base_ids if v in id_to_local]
        records["cctp"] = {
            "mode": "cctp",
            "text": None,
            "source_ids": base_ids,
            "source_local_indices": base_local,
            "chunks": [],
            "selection_audit": {"mode": "cctp", "source_ids": base_ids},
        }
        # The actual CCTP string is not recoverable from an embedding, so the
        # caller passes it separately through ``base_texts`` below.
        for mode in ("text_mmr", "tagsam_chunk", "graph_role"):
            selected, selection_audit = _select_payload_chunks(pool, mode, cfg)
            materialized = _materialize_chunk_payload(
                pool["chunks"], selected, int(cfg.text_chars),
                int(cfg.payload_max_sources))
            records[mode] = {
                "mode": mode,
                "text": materialized["text"],
                "source_ids": materialized["source_ids"],
                "source_local_indices": materialized["source_local_indices"],
                "chunks": materialized["chunks"],
                "selection_audit": selection_audit,
            }
        selected_chunks = records["tagsam_chunk"]["chunks"]
        anchor_chars = max(64, int(cfg.text_chars * 0.45))
        anchor = str(base_texts[row])[:anchor_chars].rstrip()
        supplement_budget = max(0, int(cfg.text_chars) - len(anchor) - 1)
        supplement = records["tagsam_chunk"]["text"][:supplement_budget].rstrip()
        anchor_text = (anchor + " " + supplement).strip()[:int(cfg.text_chars)]
        anchor_source_ids = list(dict.fromkeys(
            base_ids + [int(item["source_id"]) for item in selected_chunks]))
        records["anchor_mix"] = {
            "mode": "anchor_mix",
            "text": anchor_text,
            "source_ids": anchor_source_ids,
            "source_local_indices": base_local,
            "chunks": selected_chunks,
            "selection_audit": {
                "mode": "anchor_mix",
                "anchor_chars": int(anchor_chars),
                "supplement_mode": "tagsam_chunk",
            },
        }
        rows.append({
            "row": int(row),
            "class": int(pool["class"]),
            "pool": pool,
            "records": records,
        })
    # Encode all generated payloads once.  CCTP is filled by the caller after
    # this function because its exact string is already materialized there.
    generated_texts = [rows[i]["records"][mode]["text"]
                       for i in range(len(rows)) for mode in modes]
    generated_embeddings = F.normalize(
        encoder.encode(generated_texts).detach().cpu().float(), dim=1)
    cursor = 0
    for row in rows:
        for mode in modes:
            row["records"][mode]["embedding"] = generated_embeddings[cursor]
            cursor += 1
    return {"rows": rows, "modes": list(modes),
            "config": {"support_pool_size": int(cfg.support_pool_size),
                       "chunk_chars": int(cfg.chunk_chars)}}


def attach_chunk_payload(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, base_texts: list[str],
        base_source_ids: list[list[int]], labels: torch.Tensor,
        encoder, cfg: MMConfig, mode: str,
        mass: torch.Tensor | None, device: str) -> tuple[dict, dict]:
    """Attach one fixed-X payload strategy for the attribution experiment."""
    two_prototype_mode = mode.startswith("class_shared_two_")
    shared_mode = mode.startswith("class_shared_")
    if two_prototype_mode:
        base_mode = mode.removeprefix("class_shared_two_")
    else:
        base_mode = mode.removeprefix("class_shared_") if shared_mode else mode
    if base_mode == "tagsam":
        base_mode = "tagsam_chunk"
    if base_mode not in ("cctp", "text_mmr", "tagsam_chunk", "graph_role",
                         "anchor_mix"):
        raise ValueError(f"unknown chunk payload mode: {mode}")
    bank = _build_chunk_payload_bank(
        delta, source_features, x0, text0, labels, base_texts,
        base_source_ids, encoder, cfg)
    selected_records = []
    representative_by_class: dict[int, int | list[int]] = {}
    assignment_by_row: dict[int, int] = {}
    if shared_mode:
        rows_by_class: dict[int, list[int]] = {}
        for index, row in enumerate(bank["rows"]):
            rows_by_class.setdefault(int(row["class"]), []).append(index)
        x_cpu = x0.detach().cpu().float()
        for cls, indices in rows_by_class.items():
            normalized = F.normalize(x_cpu[indices], dim=1)
            center = F.normalize(normalized.mean(0, keepdim=True), dim=1)
            first_local = int((normalized @ center.T).argmax())
            representatives = [int(indices[first_local])]
            if two_prototype_mode and len(indices) > 1:
                # Keep the class-center representative, then cover the
                # farthest direction in the same CaT feature space.
                distance = 1.0 - (normalized @ normalized[first_local])
                second_local = int(distance.argmax())
                if second_local != first_local:
                    representatives.append(int(indices[second_local]))
            representative_by_class[int(cls)] = (
                representatives if two_prototype_mode else representatives[0])
            prototype_x = F.normalize(x_cpu[representatives], dim=1)
            for local_index, row_index in enumerate(indices):
                assignment = int((normalized[local_index].view(1, -1) @
                                  prototype_x.T).argmax())
                assignment_by_row[int(row_index)] = int(representatives[assignment])
    for row_index, row in enumerate(bank["rows"]):
        if base_mode == "cctp":
            record = {
                "mode": "cctp",
                "text": str(base_texts[row["row"]]),
                "source_ids": [int(v) for v in base_source_ids[row["row"]]],
                "source_local_indices": row["records"]["cctp"][
                    "source_local_indices"],
                "chunks": [],
                "embedding": F.normalize(text0[row["row"]].detach().cpu().float(),
                                          dim=0),
                "selection_audit": {"mode": "cctp"},
            }
        else:
            source_row = row
            if shared_mode:
                representative = assignment_by_row.get(int(row_index))
                if representative is None:
                    value = representative_by_class[int(row["class"])]
                    representative = (value[0] if isinstance(value, list)
                                      else value)
                source_row = bank["rows"][representative]
            record = source_row["records"][base_mode]
        selected_records.append(record)
    block = {
        "x": x0.detach().cpu().float(),
        "text_emb": F.normalize(torch.stack(
            [record["embedding"] for record in selected_records]), dim=1),
        "texts": [record["text"] for record in selected_records],
        "mass": (mass.detach().cpu() if mass is not None else
                 torch.ones(len(selected_records))),
        "depth": torch.zeros(len(selected_records), dtype=torch.long),
        "stage": torch.full((len(selected_records),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    candidate_summary = []
    for row in bank["rows"]:
        pool = row["pool"]
        candidate_summary.append({
            "row": int(row["row"]),
            "class": int(row["class"]),
            "support_local_indices": [int(v) for v in pool["support_local_indices"]],
            "support_ids": [int(v) for v in pool["support_ids"]],
            "chunk_count": len(pool["chunks"]),
            "candidates": [{
                "mode": str(row["records"][candidate]["mode"]),
                "text_length": len(row["records"][candidate]["text"]),
                "source_ids": [int(v) for v in row["records"][candidate]["source_ids"]],
                "chunks": row["records"][candidate]["chunks"],
                "selection_audit": row["records"][candidate]["selection_audit"],
            } for candidate in ("text_mmr", "tagsam_chunk", "graph_role",
                                "anchor_mix")],
        })
    audit = {
        "constructor": f"cat_payload_{mode}",
        "objective": "fixed CaT X; current-delta local support; real text chunk payload",
        "payload_generator": mode,
        "class_shared": bool(shared_mode),
        "class_prototype_count": (2 if two_prototype_mode else
                                   (1 if shared_mode else 0)),
        "class_representative_rows": representative_by_class,
        "class_row_assignments": assignment_by_row,
        "payload_modes_compared": ["cctp", "text_mmr", "tagsam_chunk",
                                   "graph_role", "anchor_mix",
                                   "class_shared_tagsam",
                                   "class_shared_two_tagsam"],
        "support_pool_rule": "bounded union of CCTP anchor, CaT original-feature top-L, and class text-center candidates",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "x_updated": False,
        "variable_chunk_count": True,
        "text_source_ids": [record["source_ids"] for record in selected_records],
        "text_chunk_provenance": [record["chunks"] for record in selected_records],
        "candidate_bank": candidate_summary,
        "text_lengths": [len(record["text"]) for record in selected_records],
        "num_chunks": [len(record["chunks"]) for record in selected_records],
        "config": asdict(cfg),
    }
    return block, audit


def attach_class_shared_chunk_payload(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, base_texts: list[str],
        base_source_ids: list[list[int]], labels: torch.Tensor,
        encoder, cfg: MMConfig, mass: torch.Tensor | None,
        device: str) -> tuple[dict, dict]:
    """Build the class-shared TAGSAM-style payload without row-wise waste.

    The ordinary ``attach_chunk_payload(..., "class_shared_tagsam", ...)``
    first builds a candidate bank for every synthetic row and only then keeps
    one representative row per class.  For a class-shared payload that extra
    work cannot affect the emitted memory: the representative is selected in
    the CaT X space, and its own candidate bank is the only bank consulted for
    the final payload.  This implementation selects those representatives
    first, constructs their real chunk candidates, and broadcasts the selected
    materialized string/embedding to rows of the same class.  It is therefore
    output-equivalent to the row-wise class-shared rule while scaling with the
    number of classes rather than the number of synthetic rows.
    """
    labels = labels.detach().long().cpu()
    x_cpu = x0.detach().cpu().float()
    text_cpu = text0.detach().cpu().float()
    source_ids = [list(map(int, values)) for values in base_source_ids]
    rows_by_class: dict[int, list[int]] = {}
    for row, cls in enumerate(labels.tolist()):
        rows_by_class.setdefault(int(cls), []).append(int(row))

    representatives: dict[int, int] = {}
    for cls, rows in rows_by_class.items():
        normalized = F.normalize(x_cpu[rows], dim=1)
        center = F.normalize(normalized.mean(0, keepdim=True), dim=1)
        local = int((normalized @ center.T).argmax())
        representatives[int(cls)] = int(rows[local])

    representative_rows = sorted(representatives.values())
    representative_index = {row: i for i, row in
                            enumerate(representative_rows)}
    rep_labels = labels[representative_rows]
    rep_block, rep_audit = attach_chunk_payload(
        delta,
        source_features,
        x_cpu[representative_rows],
        text_cpu[representative_rows],
        [base_texts[row] for row in representative_rows],
        [source_ids[row] for row in representative_rows],
        rep_labels,
        encoder,
        cfg,
        "tagsam_chunk",
        None,
        device,
    )

    class_payload: dict[int, dict] = {}
    for row, cls in zip(representative_rows, rep_labels.tolist()):
        local = representative_index[row]
        class_payload[int(cls)] = {
            "representative_row": int(row),
            "text": rep_block["texts"][local],
            "text_emb": rep_block["text_emb"][local].clone(),
            "source_ids": list(rep_audit["text_source_ids"][local]),
            "chunks": list(rep_audit["text_chunk_provenance"][local]),
            "selection": rep_audit.get("candidate_bank", [])[local]
            if rep_audit.get("candidate_bank") else None,
        }

    selected_records = [class_payload[int(cls)] for cls in labels.tolist()]
    block = {
        "x": x_cpu,
        "text_emb": F.normalize(torch.stack(
            [record["text_emb"] for record in selected_records]), dim=1),
        "texts": [record["text"] for record in selected_records],
        "mass": (mass.detach().cpu() if mass is not None else
                 torch.ones(len(selected_records))),
        "depth": torch.zeros(len(selected_records), dtype=torch.long),
        "stage": torch.full((len(selected_records),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    audit = {
        "constructor": "cat_payload_class_shared_tagsam",
        "objective": "fixed CaT X; class-shared current-delta real text chunk payload",
        "payload_generator": "tagsam_chunk",
        "class_shared": True,
        "class_prototype_count": 1,
        "class_representative_rows": {
            str(cls): int(row) for cls, row in representatives.items()},
        "class_row_assignments": {
            str(row): int(representatives[int(cls)])
            for row, cls in enumerate(labels.tolist())},
        "representative_only_construction": True,
        "representative_rows_constructed": [int(v) for v in representative_rows],
        "text_source_ids": [record["source_ids"] for record in selected_records],
        "text_chunk_provenance": [record["chunks"] for record in selected_records],
        "text_lengths": [len(record["text"]) for record in selected_records],
        "num_chunks": [len(record["chunks"]) for record in selected_records],
        "candidate_bank": [record["selection"] for record in selected_records],
        "support_pool_rule": "bounded union of CCTP anchor, CaT original-feature top-L, and class text-center candidates",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "x_updated": False,
        "variable_chunk_count": True,
        "config": asdict(cfg),
        "source_audit": rep_audit,
    }
    return block, audit


def attach_class_assigned_chunk_payload(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, base_texts: list[str],
        base_source_ids: list[list[int]], labels: torch.Tensor,
        encoder, cfg: MMConfig, mass: torch.Tensor | None,
        device: str, candidate_anchor_count: int = 1,
        max_payloads: int = 1,
        use_prompt_margin: bool = False) -> tuple[dict, dict]:
    """Select bounded materialized payloads per class with an explicit objective.

    The candidate bank is still bounded and real: it is generated only from
    the current delta and contains the existing CCTP payload plus a few
    chunk-based alternatives.  Unlike the historical class-shared rule,
    selection is made with all synthetic rows of the class in the objective:

        J_c(p) = d_graph(X_c, g_p)
               + lambda * d_text(p, c)
               + gamma * Var_j[sim(X_j, g_p)]

    ``g_p`` is the mean source feature of the real rows used by payload ``p``.
    This keeps graph/text distances in their native, auditable spaces and
    avoids introducing a new learned graph-to-text map.  Candidate generation
    remains bounded by the existing support pool; no source combinations are
    enumerated.  When ``max_payloads`` is two, only classes above the fixed
    within-class X-dispersion median may use a second payload.  The two
    payloads are selected from the bounded candidate bank, and each synthetic
    row receives the one with higher graph compatibility.  The default value
    one keeps the original single-payload selection rule.
    """
    labels = labels.detach().long().cpu()
    x_cpu = x0.detach().cpu().float()
    text_cpu = text0.detach().cpu().float()
    source_ids = [list(map(int, values)) for values in base_source_ids]
    rows_by_class: dict[int, list[int]] = {}
    for row, cls in enumerate(labels.tolist()):
        rows_by_class.setdefault(int(cls), []).append(int(row))

    # Use one center anchor by default.  The optional second anchor is the
    # farthest row in the same class, which expands candidate coverage while
    # keeping the final payload count and assignment objective unchanged.
    anchor_count = max(1, min(2, int(candidate_anchor_count)))
    anchor_rows_by_class: dict[int, list[int]] = {}
    for cls, rows in rows_by_class.items():
        normalized = F.normalize(x_cpu[rows], dim=1)
        center = F.normalize(normalized.mean(0, keepdim=True), dim=1)
        local = int((normalized @ center.T).argmax())
        anchors = [int(rows[local])]
        if anchor_count > 1 and len(rows) > 1:
            distance = 1.0 - (normalized @ normalized[local])
            farthest = int(distance.argmax())
            if farthest != local:
                anchors.append(int(rows[farthest]))
        anchor_rows_by_class[int(cls)] = anchors
    representative_rows = sorted({
        row for anchors in anchor_rows_by_class.values() for row in anchors
    })
    representative_index = {row: i for i, row in
                            enumerate(representative_rows)}
    rep_labels = labels[representative_rows]

    bank = _build_chunk_payload_bank(
        delta, source_features, x_cpu[representative_rows],
        text_cpu[representative_rows], rep_labels,
        [base_texts[row] for row in representative_rows],
        [source_ids[row] for row in representative_rows],
        encoder, cfg)

    raw_x = delta.get("x", source_features).detach().cpu().float()
    if raw_x.ndim != 2 or raw_x.shape[1] != x_cpu.shape[1]:
        raw_x = source_features.detach().cpu().float()
    raw_x = F.normalize(raw_x, dim=1)
    raw_text = F.normalize(delta["text_emb"].detach().cpu().float(), dim=1)
    raw_labels = delta["y"].detach().cpu().long()
    id_to_local = {int(value): i for i, value in
                   enumerate(delta["ids"].detach().cpu().tolist())}
    class_centers: dict[int, torch.Tensor] = {}
    for cls in rows_by_class:
        class_values = raw_text[raw_labels == int(cls)]
        class_centers[int(cls)] = F.normalize(
            class_values.mean(0, keepdim=True), dim=1).squeeze(0)

    max_payloads = max(1, min(2, int(max_payloads)))
    class_dispersion: dict[int, float] = {}
    for cls, rows in rows_by_class.items():
        normalized = F.normalize(x_cpu[rows], dim=1)
        center = F.normalize(normalized.mean(0, keepdim=True), dim=1)
        class_dispersion[int(cls)] = float(
            (1.0 - (normalized @ center.T).squeeze(1).mean()).item())
    dispersion_values = torch.tensor(list(class_dispersion.values()),
                                     dtype=torch.float32)
    dispersion_threshold = float(torch.quantile(
        dispersion_values, 0.5).item()) if len(dispersion_values) > 1 else float("inf")

    selected_by_class: dict[int, list[dict]] = {}
    row_assignment_by_class: dict[int, list[int]] = {}
    assignment_audit: dict[str, dict] = {}
    for cls, rows in rows_by_class.items():
        records: list[dict] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for anchor in anchor_rows_by_class[int(cls)]:
            bank_row = bank["rows"][representative_index[anchor]]
            for mode in ("cctp", "text_mmr", "tagsam_chunk",
                         "graph_role", "anchor_mix"):
                record = bank_row["records"].get(mode)
                if record is None:
                    continue
                if mode == "cctp":
                    # The bounded bank intentionally leaves the exact CCTP
                    # string out because it is already available at the
                    # caller.  Fill it here for each anchor.
                    record = dict(record)
                    record["text"] = str(base_texts[anchor])
                    record["embedding"] = F.normalize(
                        text_cpu[anchor].view(1, -1), dim=1).squeeze(0)
                key = (str(record["text"]), tuple(int(v)
                                                   for v in record["source_ids"]))
                if key in seen:
                    continue
                seen.add(key)
                source_local = [int(v) for v in record.get(
                    "source_local_indices", [])]
                if not source_local:
                    source_local = [
                        id_to_local[int(v)] for v in record["source_ids"]
                        if int(v) in id_to_local]
                if source_local:
                    graph_proto = F.normalize(
                        raw_x[source_local].mean(0, keepdim=True),
                        dim=1).squeeze(0)
                else:
                    graph_proto = F.normalize(x_cpu[anchor], dim=0)
                graph_sim = F.normalize(x_cpu[rows], dim=1) @ graph_proto
                graph_cost = 1.0 - graph_sim.mean()
                text_cost = 1.0 - (
                    F.normalize(record["embedding"].view(1, -1), dim=1).squeeze(0)
                    @ class_centers[int(cls)])
                prompt_margin = graph_cost * 0.0
                prompt_emb = delta.get("prompt_emb")
                if use_prompt_margin and prompt_emb is not None:
                    prompt_cpu = F.normalize(
                        prompt_emb.detach().cpu().float(), dim=1)
                    if int(cls) < len(prompt_cpu):
                        prompt_scores = (
                            F.normalize(record["embedding"].view(1, -1), dim=1)
                            @ prompt_cpu.T).squeeze(0)
                        positive = prompt_scores[int(cls)]
                        negative = prompt_scores.clone()
                        negative[int(cls)] = -float("inf")
                        prompt_margin = F.relu(
                            float(cfg.prompt_margin) - positive
                            + negative.max())
                variance = (graph_sim.var(unbiased=False)
                            if len(graph_sim) > 1 else graph_sim.sum() * 0.0)
                objective = (graph_cost
                             + float(cfg.payload_assignment_text_weight) * text_cost
                             + float(cfg.payload_assignment_variance_weight) * variance)
                objective = (objective
                             + float(cfg.lambda_prompt_margin) * prompt_margin)
                records.append({
                    "mode": str(mode),
                    "anchor_row": int(anchor),
                    "record": record,
                    "graph_cost": float(graph_cost),
                    "text_cost": float(text_cost),
                    "prompt_margin": float(prompt_margin),
                    "compatibility_variance": float(variance),
                    "objective": float(objective),
                    "source_local_indices": source_local,
                    "_graph_sim": graph_sim,
                })
        if not records:
            raise ValueError(f"empty payload candidate bank for class {cls}")
        ordered = sorted(records, key=lambda value: (
            value["objective"], value["mode"], value["anchor_row"]))
        chosen_records = [ordered[0]]
        eligible_for_two = (
            max_payloads > 1
            and len(rows) > 1
            and class_dispersion[int(cls)] >= dispersion_threshold
            and len(ordered) > 1
        )
        if eligible_for_two:
            pair_candidates: list[dict] = []
            for left_index, left in enumerate(ordered[:-1]):
                for right in ordered[left_index + 1:]:
                    pair_sim = torch.stack((left["_graph_sim"],
                                             right["_graph_sim"]), dim=1)
                    best_sim = pair_sim.max(dim=1).values
                    pair_graph_cost = 1.0 - best_sim.mean()
                    pair_variance = (best_sim.var(unbiased=False)
                                     if len(best_sim) > 1 else
                                     best_sim.sum() * 0.0)
                    pair_text_cost = 0.5 * (left["text_cost"]
                                             + right["text_cost"])
                    pair_objective = (
                        pair_graph_cost
                        + float(cfg.payload_assignment_text_weight) * pair_text_cost
                        + float(cfg.payload_assignment_variance_weight) * pair_variance)
                    pair_candidates.append({
                        "left": left,
                        "right": right,
                        "graph_cost": float(pair_graph_cost),
                        "text_cost": float(pair_text_cost),
                        "compatibility_variance": float(pair_variance),
                        "objective": float(pair_objective),
                    })
            if pair_candidates:
                pair = min(pair_candidates, key=lambda value: (
                    value["objective"], value["left"]["mode"],
                    value["right"]["mode"]))
                chosen_records = [pair["left"], pair["right"]]

        selected_sim = torch.stack([value["_graph_sim"]
                                    for value in chosen_records], dim=1)
        row_assignment = selected_sim.argmax(dim=1).tolist()
        selected_by_class[int(cls)] = [value["record"]
                                       for value in chosen_records]
        row_assignment_by_class[int(cls)] = [int(v) for v in row_assignment]
        selected = chosen_records[0]
        assignment_audit[str(cls)] = {
            "representative_row": int(anchor_rows_by_class[int(cls)][0]),
            "candidate_anchor_rows": [int(v)
                                       for v in anchor_rows_by_class[int(cls)]],
            "row_count": len(rows),
            "class_dispersion": class_dispersion[int(cls)],
            "dispersion_threshold": dispersion_threshold,
            "used_two_payloads": len(chosen_records) == 2,
            "selected_mode": selected["mode"],
            "selected_modes": [value["mode"] for value in chosen_records],
            "selected_anchor_row": int(selected["anchor_row"]),
            "selected_anchor_rows": [int(value["anchor_row"])
                                     for value in chosen_records],
            "selected_objective": selected["objective"],
            "row_assignment": [int(v) for v in row_assignment],
            "row_assignment_counts": {
                str(index): int(row_assignment.count(index))
                for index in range(len(chosen_records))},
            "candidates": [{
                "mode": value["mode"],
                "anchor_row": int(value["anchor_row"]),
                "graph_cost": value["graph_cost"],
                "text_cost": value["text_cost"],
                "prompt_margin": value["prompt_margin"],
                "compatibility_variance": value["compatibility_variance"],
                "objective": value["objective"],
                "source_ids": [int(v) for v in value["record"]["source_ids"]],
            } for value in records],
        }

    # Recover a payload for each X row with deterministic per-class cursors.
    # ``rows_by_class`` stores row positions in the same order as ``labels``.
    class_cursor: dict[int, int] = {int(cls): 0 for cls in rows_by_class}
    selected_records = []
    for cls in labels.tolist():
        cls = int(cls)
        row_index = rows_by_class[cls][class_cursor[cls]]
        local_index = row_index - rows_by_class[cls][0]
        payload_index = row_assignment_by_class[cls][local_index]
        selected_records.append(selected_by_class[cls][payload_index])
        class_cursor[cls] += 1
    block = {
        "x": x_cpu,
        "text_emb": F.normalize(torch.stack(
            [record["embedding"] for record in selected_records]), dim=1),
        "texts": [record["text"] for record in selected_records],
        "mass": (mass.detach().cpu() if mass is not None else
                 torch.ones(len(selected_records))),
        "depth": torch.zeros(len(selected_records), dtype=torch.long),
        "stage": torch.full((len(selected_records),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    audit = {
        "constructor": "cat_payload_class_assigned_two_payload"
        if max_payloads > 1 else "cat_payload_class_assigned",
        "objective": "min graph cost + lambda text cost + gamma compatibility variance + optional prompt margin",
        "payload_generator": "class_level_materialized_assignment",
        "class_shared": True,
        "class_prototype_count": int(max_payloads),
        "max_payloads": int(max_payloads),
        "prompt_margin_enabled": bool(use_prompt_margin),
        "candidate_anchor_count": int(anchor_count),
        "class_representative_rows": {
            str(cls): [int(v) for v in rows]
            for cls, rows in anchor_rows_by_class.items()},
        "assignment_objective": {
            "graph_term": "1 - mean_j cosine(X_j, mean_graph_feature(source_ids(p)))",
            "text_term": "1 - cosine(payload_embedding, current_class_text_center)",
            "variance_term": "variance_j cosine(X_j, mean_graph_feature(source_ids(p)))",
            "prompt_margin_term": "relu(m - sim(payload, prompt_y) + max_{c!=y} sim(payload, prompt_c))",
            "text_weight": float(cfg.payload_assignment_text_weight),
            "variance_weight": float(cfg.payload_assignment_variance_weight),
            "prompt_margin_weight": float(cfg.lambda_prompt_margin)
            if use_prompt_margin else 0.0,
            "prompt_margin": float(cfg.prompt_margin),
            "prompt_margin_enabled": bool(use_prompt_margin),
        },
        "assignment_by_class": assignment_audit,
        "class_dispersion_rule": "allow two payloads at or above within-stage class-dispersion median",
        "selected_payload_count_by_class": {
            str(cls): len(selected_by_class[int(cls)])
            for cls in rows_by_class},
        "representative_only_construction": True,
        "representative_rows_constructed": [int(v) for v in representative_rows],
        "text_source_ids": [record["source_ids"]
                             for record in selected_records],
        "text_chunk_provenance": [record["chunks"]
                                   for record in selected_records],
        "text_lengths": [len(record["text"])
                         for record in selected_records],
        "num_chunks": [len(record["chunks"])
                       for record in selected_records],
        "support_pool_rule": "bounded current-delta candidate bank from CCTP, chunk MMR, TAGSAM-style chunk, graph-role and anchor-mix candidates",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "x_updated": False,
        "variable_chunk_count": True,
        "config": asdict(cfg),
    }
    return block, audit


def _fallback_group_indices(candidates: list[int], sources: int,
                            offset: int) -> list[int]:
    """Make a deterministic rotated group without enumerating combinations."""
    if not candidates:
        raise ValueError("cannot build a text group from an empty class")
    k = min(int(sources), len(candidates))
    return [candidates[(int(offset) + j) % len(candidates)]
            for j in range(k)]


def _build_text_group_bank(delta: dict, source_features: torch.Tensor,
                           encoder, cfg: MMConfig,
                           graph_text_map: torch.Tensor | None = None) -> dict:
    """Build a small bank of *materialized* candidate text groups per class.

    Every candidate is a real concatenated string encoded once by the frozen
    text encoder.  The bank is bounded by ``text_group_count`` and is built
    only from the current raw delta.  The first candidate follows the existing
    CCTP text-center selector; later candidates are fixed graph/text-aware
    alternatives plus deterministic rotations.  This avoids enumerating the
    combinatorial source space while making the optimization target identical
    to the object that can be stored in memory.
    """
    if "texts" not in delta or "ids" not in delta:
        raise ValueError("text-group construction requires delta texts and ids")
    count = max(1, int(cfg.text_group_count))
    source_features = source_features.detach().cpu().float()
    raw_text = F.normalize(delta["text_emb"].detach().cpu().float(), dim=1)
    raw_labels = delta["y"].detach().cpu().long()
    selector_cfg = M.Config(seed=int(cfg.seed), text_sources=int(cfg.text_sources),
                            text_chars=int(cfg.text_chars),
                            text_mode="text_center", text_diversity=0.15,
                            text_ridge=float(cfg.text_ridge))
    mode_order = ("text_center", "graph_conditioned", "nearest",
                  "graph_text_aligned")
    groups_by_class: dict[int, list[dict]] = {}
    all_embeddings, all_labels = [], []
    global_index = 0
    for cls in sorted(torch.unique(raw_labels).tolist()):
        candidates = torch.where(raw_labels == int(cls))[0].tolist()
        if not candidates:
            continue
        query = source_features[candidates].mean(0)
        seen: set[tuple[int, ...]] = set()
        records: list[dict] = []
        for mode in mode_order:
            if len(records) >= count:
                break
            if mode == "graph_text_aligned" and graph_text_map is None:
                continue
            cfg_mode = selector_cfg
            if mode != "text_center":
                cfg_mode = M.Config(
                    seed=int(cfg.seed), text_sources=int(cfg.text_sources),
                    text_chars=int(cfg.text_chars), text_mode=mode,
                    text_diversity=0.15, text_ridge=float(cfg.text_ridge))
            chosen, selector_audit = M._select_text_sources(
                delta, source_features, query, int(cls), cfg_mode,
                graph_text_map, row_index=len(records))
            indices = [int(v) for v in chosen.tolist()]
            key = _group_key(indices)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "group_index": len(records),
                "class": int(cls),
                "source_local_indices": indices,
                "source_ids": [int(delta["ids"][i]) for i in indices],
                "selector_mode": mode,
                "selector_audit": selector_audit,
            })
        # If the fixed selector views collapse to the same group, add only a
        # bounded number of deterministic rotations.  No pair enumeration.
        offset = 0
        while len(records) < count and offset < max(1, len(candidates) * 2):
            indices = _fallback_group_indices(candidates, cfg.text_sources,
                                               offset)
            offset += 1
            key = _group_key(indices)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "group_index": len(records),
                "class": int(cls),
                "source_local_indices": indices,
                "source_ids": [int(delta["ids"][i]) for i in indices],
                "selector_mode": "deterministic_rotation",
                "selector_audit": None,
            })
        texts = [_concat_text_group(delta, r["source_local_indices"],
                                     cfg.text_chars) for r in records]
        embeddings = F.normalize(encoder.encode(texts).cpu().float(), dim=1)
        for record, text, embedding in zip(records, texts, embeddings):
            record["text"] = text
            record["embedding"] = embedding
            record["global_index"] = global_index
            global_index += 1
            all_embeddings.append(embedding)
            all_labels.append(int(cls))
        groups_by_class[int(cls)] = records
    if not all_embeddings:
        raise ValueError("text-group bank is empty")
    return {
        "groups_by_class": groups_by_class,
        "embeddings": F.normalize(torch.stack(all_embeddings), dim=1),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "count_per_class": {str(cls): len(groups)
                             for cls, groups in groups_by_class.items()},
    }


def _soft_group_payload(raw_labels: torch.Tensor, x_value: torch.Tensor,
                        labels: torch.Tensor, graph_text_map: torch.Tensor,
                        group_bank: dict, temperature: float,
                        device: str) -> tuple[torch.Tensor, list[dict]]:
    """Transport each X row over a bounded bank of materialized text groups."""
    query = F.normalize(x_value.float(), dim=1)
    ones = torch.ones((len(query), 1), device=device, dtype=query.dtype)
    predicted_text = F.normalize(
        torch.cat((query, ones), dim=1) @ graph_text_map.to(device), dim=1)
    bank_embeddings = group_bank["embeddings"].to(device).float()
    bank_labels = group_bank["labels"].to(device).long()
    soft_rows, audits = [], []
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        group_ids = torch.where(bank_labels == int(cls))[0]
        if not len(group_ids):
            raise ValueError(f"no materialized text groups for class {cls}")
        group_vectors = bank_embeddings.index_select(0, group_ids)
        scores = (predicted_text[row:row + 1] @ group_vectors.T).squeeze(0)
        probabilities = F.softmax(scores / max(float(temperature), 1e-6), dim=0)
        soft_rows.append(probabilities @ group_vectors)
        with torch.no_grad():
            entropy = -(probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()).sum()
            order = torch.argsort(probabilities, descending=True)
            records = group_bank["groups_by_class"][int(cls)]
            audits.append({
                "class": int(cls),
                "group_indices": [int(group_ids[i]) for i in order.tolist()],
                "group_probabilities": probabilities[order].cpu().tolist(),
                "entropy": float(entropy.cpu()),
                "normalized_entropy": float(
                    entropy.cpu() / max(1.0, math.log(len(group_ids)))),
                "top1_probability": float(probabilities[order[0]].cpu()),
                "top1_group_local": int(order[0]),
                "top1_source_local_indices": records[int(order[0])][
                    "source_local_indices"],
            })
    return F.normalize(torch.stack(soft_rows), dim=1), audits


def _attention_group_payload(x_value: torch.Tensor, labels: torch.Tensor,
                             group_bank: dict, x_projection: torch.Tensor,
                             prompt_projection: torch.Tensor,
                             bias: torch.Tensor, prompt_emb: torch.Tensor | None,
                             temperature: float, device: str):
    """Score materialized groups with a differentiable X/prompt query."""
    x_norm = F.normalize(x_value.float(), dim=1)
    bank_embeddings = group_bank["embeddings"].to(device).float()
    bank_labels = group_bank["labels"].to(device).long()
    if prompt_emb is None:
        prompt_value = None
    else:
        prompt_value = F.normalize(prompt_emb.detach().to(device).float(), dim=1)
    soft_rows, query_rows, audits = [], [], []
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        if prompt_value is None or int(cls) >= len(prompt_value):
            class_prompt = torch.zeros_like(bias)
        else:
            class_prompt = prompt_value[int(cls)]
        query = (x_norm[row] @ x_projection
                 + class_prompt @ prompt_projection + bias)
        query = F.normalize(query.view(1, -1), dim=1).squeeze(0)
        group_ids = torch.where(bank_labels == int(cls))[0]
        if not len(group_ids):
            raise ValueError(f"no materialized text groups for class {cls}")
        group_vectors = bank_embeddings.index_select(0, group_ids)
        scores = (query.view(1, -1) @ group_vectors.T).squeeze(0)
        probabilities = F.softmax(scores / max(float(temperature), 1e-6), dim=0)
        soft_rows.append(probabilities @ group_vectors)
        query_rows.append(query)
        with torch.no_grad():
            entropy = -(probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()).sum()
            order = torch.argsort(probabilities, descending=True)
            records = group_bank["groups_by_class"][int(cls)]
            audits.append({
                "class": int(cls),
                "group_indices": [int(group_ids[i]) for i in order.tolist()],
                "group_probabilities": probabilities[order].cpu().tolist(),
                "entropy": float(entropy.cpu()),
                "normalized_entropy": float(
                    entropy.cpu() / max(1.0, math.log(len(group_ids)))),
                "top1_probability": float(probabilities[order[0]].cpu()),
                "top1_group_local": int(order[0]),
                "top1_source_local_indices": records[int(order[0])][
                    "source_local_indices"],
            })
    return (F.normalize(torch.stack(soft_rows), dim=1),
            F.normalize(torch.stack(query_rows), dim=1), audits)


def _prompt_margin_loss(text_value: torch.Tensor, labels: torch.Tensor,
                        prompt_emb: torch.Tensor | None, margin: float,
                        device: str) -> torch.Tensor:
    """Text-only class margin; returns zero when prompts are unavailable."""
    if prompt_emb is None:
        return text_value.sum() * 0.0
    prompts = F.normalize(prompt_emb.detach().to(device).float(), dim=1)
    if len(prompts) <= 1:
        return text_value.sum() * 0.0
    scores = F.normalize(text_value, dim=1) @ prompts.T
    positive = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    hardest_negative = masked.max(dim=1).values
    return F.relu(float(margin) - positive + hardest_negative).mean()


def _build_row_text_group_bank(delta: dict, source_features: torch.Tensor,
                               x0: torch.Tensor, labels: torch.Tensor,
                               base_source_ids: list[list[int]], encoder,
                               cfg: MMConfig, graph_text_map: torch.Tensor) -> dict:
    """Build a small candidate bank separately for every synthetic row.

    The first group is exactly the row's existing CCTP materialization.  The
    remaining groups are deterministic alternatives derived from the current
    delta.  This preserves a legal no-op path for the joint refinement and
    avoids replacing per-row CaT text with a class-shared prototype.
    """
    if "texts" not in delta or "ids" not in delta:
        raise ValueError("row text-group construction requires delta texts and ids")
    count = max(1, int(cfg.text_group_count))
    source_features = source_features.detach().cpu().float()
    raw_labels = delta["y"].detach().cpu().long()
    id_to_local = {int(value): i for i, value in enumerate(delta["ids"].tolist())}
    selector_base = M.Config(seed=int(cfg.seed), text_sources=int(cfg.text_sources),
                             text_chars=int(cfg.text_chars), text_mode="text_center",
                             text_diversity=0.15, text_ridge=float(cfg.text_ridge))
    mode_order = ("text_center", "graph_conditioned", "nearest",
                  "graph_text_aligned")
    rows = []
    all_embeddings, all_labels = [], []
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        candidates = torch.where(raw_labels == int(cls))[0].tolist()
        if not candidates:
            raise ValueError(f"no text candidates for class {cls}")
        records, seen = [], set()

        def add_record(indices: list[int], mode: str, selector_audit=None):
            key = _group_key(indices)
            if key in seen or len(records) >= count:
                return
            seen.add(key)
            records.append({
                "group_index": len(records),
                "class": int(cls),
                "source_local_indices": [int(i) for i in indices],
                "source_ids": [int(delta["ids"][i]) for i in indices],
                "selector_mode": mode,
                "selector_audit": selector_audit,
            })

        # Preserve the exact baseline text group for this synthetic row.
        base_ids = base_source_ids[row] if row < len(base_source_ids) else []
        base_indices = [id_to_local[int(value)] for value in base_ids
                        if int(value) in id_to_local]
        if not base_indices:
            base_indices = _fallback_group_indices(candidates, cfg.text_sources, 0)
        add_record(base_indices, "cctp_row_base")

        query = source_features[candidates].mean(0)
        for mode in mode_order:
            if len(records) >= count:
                break
            if mode == "graph_text_aligned" and graph_text_map is None:
                continue
            cfg_mode = selector_base if mode == "text_center" else M.Config(
                seed=int(cfg.seed), text_sources=int(cfg.text_sources),
                text_chars=int(cfg.text_chars), text_mode=mode,
                text_diversity=0.15, text_ridge=float(cfg.text_ridge))
            chosen, selector_audit = M._select_text_sources(
                delta, source_features, query, int(cls), cfg_mode,
                graph_text_map, row_index=row)
            add_record([int(v) for v in chosen.tolist()], mode, selector_audit)
        offset = 0
        while len(records) < count and offset < max(1, len(candidates) * 2):
            add_record(_fallback_group_indices(candidates, cfg.text_sources, offset),
                       "deterministic_rotation")
            offset += 1
        texts = [_concat_text_group(delta, r["source_local_indices"],
                                    cfg.text_chars) for r in records]
        embeddings = F.normalize(encoder.encode(texts).cpu().float(), dim=1)
        for record, text, embedding in zip(records, texts, embeddings):
            record["text"] = text
            record["embedding"] = embedding
            all_embeddings.append(embedding)
            all_labels.append(int(cls))
        rows.append(records)
    return {
        "rows": rows,
        "embeddings": F.normalize(torch.stack(all_embeddings), dim=1),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "count_per_row": [len(row) for row in rows],
    }


def _attention_row_group_payload(x_value: torch.Tensor, labels: torch.Tensor,
                                 row_bank: dict, x_projection: torch.Tensor,
                                 prompt_projection: torch.Tensor,
                                 bias: torch.Tensor,
                                 prompt_emb: torch.Tensor | None,
                                 temperature: float, device: str):
    """Attention payload for row-specific group candidates."""
    x_norm = F.normalize(x_value.float(), dim=1)
    if prompt_emb is None:
        prompt_value = None
    else:
        prompt_value = F.normalize(prompt_emb.detach().to(device).float(), dim=1)
    soft_rows, query_rows, audits = [], [], []
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        if prompt_value is None or int(cls) >= len(prompt_value):
            class_prompt = torch.zeros_like(bias)
        else:
            class_prompt = prompt_value[int(cls)]
        query = (x_norm[row] @ x_projection
                 + class_prompt @ prompt_projection + bias)
        query = F.normalize(query.view(1, -1), dim=1).squeeze(0)
        records = row_bank["rows"][row]
        group_vectors = torch.stack([r["embedding"] for r in records]).to(device)
        scores = query.view(1, -1) @ group_vectors.T
        probabilities = F.softmax(scores.squeeze(0) /
                                  max(float(temperature), 1e-6), dim=0)
        soft_rows.append(probabilities @ group_vectors)
        query_rows.append(query)
        with torch.no_grad():
            order = torch.argsort(probabilities, descending=True)
            entropy = -(probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()).sum()
            audits.append({
                "row": int(row),
                "class": int(cls),
                "group_indices": order.cpu().tolist(),
                "group_probabilities": probabilities[order].cpu().tolist(),
                "entropy": float(entropy.cpu()),
                "normalized_entropy": float(
                    entropy.cpu() / max(1.0, math.log(len(records)))),
                "top1_probability": float(probabilities[order[0]].cpu()),
                "top1_group_local": int(order[0]),
                "top1_source_local_indices": records[int(order[0])][
                    "source_local_indices"],
            })
    return (F.normalize(torch.stack(soft_rows), dim=1),
            F.normalize(torch.stack(query_rows), dim=1), audits)


def optimize_joint_cat_text_groups(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, texts: list[str], source_ids: list[list[int]],
        labels: torch.Tensor, parent: dict | None, encoder, cfg: MMConfig,
        mass: torch.Tensor | None, device: str) -> tuple[dict, dict]:
    """Jointly refine CaT X and a text query over real text groups.

    CaT supplies ``x0``.  The refinement trains only X and two small query
    projections; MiniLM, historical raw nodes, old memory, and topology stay
    frozen.  A trust-region anchor keeps the text branch from replacing the
    original CaT solution, while the final block stores an actual candidate
    group selected from the current raw delta.
    """
    source_features = source_features.detach().to(device).float()
    x0 = x0.detach().to(device).float()
    text0 = F.normalize(text0.detach().to(device).float(), dim=1)
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_labels = delta["y"].detach().long().to(device)
    labels = labels.detach().long().to(device)
    if parent is not None and parent["edge_index"].numel():
        raise ValueError("joint CaT-text requires edge-free parent topology")
    graph_text_map, fit_cosine = M._fit_graph_text_map(
        source_features, raw_text, float(cfg.text_ridge))
    group_bank = _build_row_text_group_bank(
        delta, source_features, x0, labels, source_ids, encoder, cfg,
        graph_text_map.detach().cpu())
    prompt_emb = delta.get("prompt_emb")
    text_dim = int(group_bank["embeddings"].shape[1])
    x_projection = torch.nn.Parameter(graph_text_map[:-1].clone())
    prompt_projection = torch.nn.Parameter(
        0.25 * torch.eye(text_dim, device=device))
    bias = torch.nn.Parameter(graph_text_map[-1].clone())
    x_param = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([
        {"params": [x_param], "lr": float(cfg.lr_x)},
        {"params": [x_projection, prompt_projection, bias],
         "lr": float(cfg.lr_text)},
    ])
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 9137)
    x_weight = torch.empty((source_features.shape[1], int(cfg.hidden)),
                           device=device)
    x_weight.uniform_(-_bound(source_features.shape[1], cfg.hidden),
                      _bound(source_features.shape[1], cfg.hidden),
                      generator=generator)
    text_weight = torch.empty((text_dim, int(cfg.hidden)), device=device)
    text_weight.uniform_(-_bound(text_dim, cfg.hidden), _bound(text_dim, cfg.hidden),
                         generator=generator)
    with torch.no_grad():
        baseline_soft, baseline_query, _ = _attention_row_group_payload(
            x0, labels, group_bank, x_projection, prompt_projection, bias,
            prompt_emb, cfg.assignment_temperature, device)
        baseline_text = torch.stack(
            [row[0]["embedding"] for row in group_bank["rows"]]
        ).to(device)
        scale_x = _class_mmd(source_features, x0, raw_labels, labels,
                             x_weight).clamp_min(1e-8)
        scale_group = _class_mmd(
            group_bank["embeddings"].to(device), baseline_text,
            group_bank["labels"].to(device), labels, text_weight
        ).clamp_min(1e-8)
        scale_cross = (1.0 - F.cosine_similarity(
            baseline_query, baseline_text, dim=1)).mean().clamp_min(1e-4)
    trace = []
    for step in range(max(1, int(cfg.steps))):
        soft_text, query, assignments = _attention_row_group_payload(
            x_param, labels, group_bank, x_projection, prompt_projection,
            bias, prompt_emb, cfg.assignment_temperature, device)
        loss_x = _class_mmd(source_features, x_param, raw_labels, labels,
                            x_weight)
        loss_group = _class_mmd(
            group_bank["embeddings"].to(device), soft_text,
            group_bank["labels"].to(device), labels, text_weight)
        cross_loss = (1.0 - F.cosine_similarity(query, soft_text, dim=1)).mean()
        prompt_loss = _prompt_margin_loss(
            soft_text, labels, prompt_emb, cfg.prompt_margin, device)
        query_anchor = (1.0 - F.cosine_similarity(
            query, baseline_query, dim=1)).mean()
        group_anchor = (1.0 - F.cosine_similarity(
            soft_text, baseline_text, dim=1)).mean()
        x_anchor = (x_param - x0).square().mean()
        entropy_terms = []
        for row, records in enumerate(group_bank["rows"]):
            bank_embeddings = torch.stack(
                [r["embedding"] for r in records]
            ).to(device)
            scores = query[row:row + 1] @ bank_embeddings.T
            probs = F.softmax(scores.squeeze(0) /
                              max(float(cfg.assignment_temperature), 1e-6), dim=0)
            entropy_terms.append(-(probs.clamp_min(1e-12)
                                   * probs.clamp_min(1e-12).log()).sum()
                                  / max(1.0, math.log(len(records))))
        entropy_loss = torch.stack(entropy_terms).mean()
        loss = (loss_x / scale_x
                + float(cfg.lambda_text) * loss_group / scale_group
                + float(cfg.lambda_cross) * cross_loss / scale_cross
                + float(cfg.lambda_prompt_margin) * prompt_loss
                + float(cfg.lambda_group_entropy) * entropy_loss
                + float(cfg.lambda_anchor_x) * x_anchor
                + float(cfg.lambda_anchor_text + cfg.lambda_cctp_anchor)
                * group_anchor
                + float(cfg.lambda_query_anchor) * query_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite joint CaT-text loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x_param, x_projection,
                                        prompt_projection, bias], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            delta_x = x_param - x0
            row_norm = delta_x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            allowed = float(cfg.x_trust_radius) * \
                x0.norm(dim=1, keepdim=True).clamp_min(1e-3)
            scale = (allowed / row_norm).clamp(max=1.0)
            x_param.copy_(x0 + delta_x * scale)
        if step == 0 or (step + 1) % 50 == 0 or step == cfg.steps - 1:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "x_mmd": float(loss_x.detach().cpu()),
                "group_mmd": float(loss_group.detach().cpu()),
                "cross": float(cross_loss.detach().cpu()),
                "prompt_margin": float(prompt_loss.detach().cpu()),
                "group_entropy": float(entropy_loss.detach().cpu()),
                "x_anchor": float(x_anchor.detach().cpu()),
                "query_anchor": float(query_anchor.detach().cpu()),
            })
    with torch.no_grad():
        final_soft, final_query, final_assignments = _attention_row_group_payload(
            x_param, labels, group_bank, x_projection, prompt_projection,
            bias, prompt_emb, cfg.assignment_temperature, device)
    final_x = x_param.detach().cpu()
    final_texts, final_ids, final_emb, hard_audit = [], [], [], []
    for row, audit_row in enumerate(final_assignments):
        group_id = int(audit_row["group_indices"][0])
        group_record = group_bank["rows"][row][group_id]
        final_texts.append(group_record["text"])
        final_ids.append(group_record["source_ids"])
        final_emb.append(group_record["embedding"])
        hard_audit.append({
            "row": int(row),
            "class": audit_row["class"],
            "selected_group_local": group_id,
            "source_local_indices": group_record["source_local_indices"],
            "source_ids": group_record["source_ids"],
            "text_length": len(group_record["text"]),
        })
    final_emb = F.normalize(torch.stack(final_emb), dim=1)
    block = {
        "x": final_x,
        "text_emb": final_emb,
        "texts": final_texts,
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x))),
        "depth": torch.zeros(len(final_x), dtype=torch.long),
        "stage": torch.full((len(final_x),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    soft_cpu = final_soft.detach().cpu()
    audit = {
        "constructor": "cat_joint_text_group",
        "objective": "CaT graph MMD + prompt/query text attention + materialized group selection",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "text_representation": "top-scoring real CCTP text group",
        "graph_text_fit_cosine": float(fit_cosine),
        "assignment_temperature": float(cfg.assignment_temperature),
        "row_classes": labels.cpu().tolist(),
        "text_source_ids": final_ids,
        "soft_group_assignment": final_assignments,
        "hard_group_materialization": hard_audit,
        "group_bank_count_per_row": group_bank["count_per_row"],
        "group_bank": [{
            "row": int(row),
            "class": int(r["class"]),
            "group_local": int(r["group_index"]),
            "source_local_indices": r["source_local_indices"],
            "source_ids": r["source_ids"],
            "selector_mode": r["selector_mode"],
            "text_length": len(r["text"]),
        } for row, records in enumerate(group_bank["rows"])
        for r in records],
        "soft_hard_cosine_mean": float(
            F.cosine_similarity(soft_cpu, final_emb, dim=1).mean()),
        "soft_hard_cosine_min": float(
            F.cosine_similarity(soft_cpu, final_emb, dim=1).min()),
        "config": asdict(cfg),
        "trace": trace,
        "normalization_scales": {
            "x_mmd": float(scale_x.cpu()),
            "group_mmd": float(scale_group.cpu()),
            "cross": float(scale_cross.cpu()),
        },
        "x_update_norm": float((final_x - x0.cpu()).norm()),
        "query_update_norm": float(
            (final_query.detach().cpu() - baseline_query.cpu()).norm()),
        "final_text_vs_base_cosine": float(
            F.cosine_similarity(final_emb, text0.cpu(), dim=1).mean()),
        "final_text_norm_mean": float(final_emb.norm(dim=1).mean()),
    }
    return block, audit


def _gated_row_group_payload(x_value: torch.Tensor, labels: torch.Tensor,
                             row_bank: dict, graph_text_map: torch.Tensor,
                             prompt_emb: torch.Tensor | None,
                             base_logit: float, temperature: float,
                             device: str):
    """Return a differentiable text payload with an explicit CCTP prior.

    Row zero in every bank is the exact CCTP source group.  The additive prior
    is deliberately attached to that row, rather than hidden in a learned
    selector, so the joint refinement has a reproducible no-op path.
    """
    query = F.normalize(x_value.float(), dim=1)
    ones = torch.ones((len(query), 1), device=device, dtype=query.dtype)
    graph_query = F.normalize(
        torch.cat((query, ones), dim=1) @ graph_text_map.to(device), dim=1)
    if prompt_emb is not None:
        prompts = F.normalize(prompt_emb.detach().to(device).float(), dim=1)
    else:
        prompts = None
    soft_rows, probabilities_rows, query_rows, audits = [], [], [], []
    for row, cls in enumerate(labels.detach().cpu().tolist()):
        class_prompt = (prompts[int(cls)] if prompts is not None
                        and int(cls) < len(prompts) else None)
        query_row = graph_query[row]
        if class_prompt is not None:
            query_row = F.normalize(query_row + 0.25 * class_prompt, dim=0)
        records = row_bank["rows"][row]
        group_vectors = torch.stack([r["embedding"] for r in records]).to(device)
        logits = query_row.view(1, -1) @ group_vectors.T
        logits = logits.squeeze(0)
        logits = logits.clone()
        logits[0] = logits[0] + float(base_logit)
        probabilities = F.softmax(logits / max(float(temperature), 1e-6), dim=0)
        soft_rows.append(probabilities @ group_vectors)
        probabilities_rows.append(probabilities)
        query_rows.append(query_row)
        with torch.no_grad():
            order = torch.argsort(logits, descending=True)
            entropy = -(probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()).sum()
            audits.append({
                "row": int(row),
                "class": int(cls),
                "logits": logits[order].cpu().tolist(),
                "group_indices": order.cpu().tolist(),
                "group_probabilities": probabilities[order].cpu().tolist(),
                "entropy": float(entropy.cpu()),
                "top1_probability": float(probabilities[order[0]].cpu()),
                "base_probability": float(probabilities[0].cpu()),
                "base_logit": float(logits[0].cpu()),
                "best_alternative_logit": float(logits[1:].max().cpu())
                if len(logits) > 1 else float("-inf"),
            })
    return (F.normalize(torch.stack(soft_rows), dim=1),
            torch.stack(probabilities_rows),
            F.normalize(torch.stack(query_rows), dim=1), audits)


def optimize_gated_joint_cat_text(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, texts: list[str], source_ids: list[list[int]],
        labels: torch.Tensor, parent: dict | None, encoder, cfg: MMConfig,
        mass: torch.Tensor | None, device: str) -> tuple[dict, dict]:
    """Joint CaT/text refinement with a bounded CCTP-preserving gate.

    The CaT random-feature objective remains the main X objective.  A frozen
    graph-to-text map turns the current X into an attention query over a small
    row-specific bank of real, MiniLM-encoded text groups.  The text loss
    backpropagates through this query into X, but the explicit base prior,
    baseline-selection penalty, and X trust region prevent a silent wholesale
    replacement of the original CaT+CCTP memory.
    """
    source_features = source_features.detach().to(device).float()
    x0 = x0.detach().to(device).float()
    text0 = F.normalize(text0.detach().to(device).float(), dim=1)
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_labels = delta["y"].detach().long().to(device)
    labels = labels.detach().long().to(device)
    if parent is not None and parent["edge_index"].numel():
        raise ValueError("gated joint CaT/text requires edge-free parent topology")
    graph_text_map, fit_cosine = M._fit_graph_text_map(
        source_features, raw_text, float(cfg.text_ridge))
    row_bank = _build_row_text_group_bank(
        delta, source_features, x0, labels, source_ids, encoder, cfg,
        graph_text_map.detach().cpu())
    prompt_emb = delta.get("prompt_emb")
    x_param = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([x_param], lr=float(cfg.lr_x))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 15211)
    x_weight = torch.empty((source_features.shape[1], int(cfg.hidden)),
                           device=device)
    x_weight.uniform_(-_bound(source_features.shape[1], cfg.hidden),
                      _bound(source_features.shape[1], cfg.hidden),
                      generator=generator)
    text_dim = int(row_bank["embeddings"].shape[1])
    with torch.no_grad():
        base_text = torch.stack([r[0]["embedding"] for r in row_bank["rows"]])
        base_text = F.normalize(base_text.to(device), dim=1)
        base_soft, base_prob, base_query, _ = _gated_row_group_payload(
            x0, labels, row_bank, graph_text_map, prompt_emb,
            cfg.joint_base_logit, cfg.assignment_temperature, device)
        scale_x = _class_mmd(source_features, x0, raw_labels, labels,
                             x_weight).clamp_min(1e-8)
        scale_cross = (1.0 - F.cosine_similarity(
            base_query, base_text, dim=1)).mean().clamp_min(1e-4)
    trace = []
    for step in range(max(1, int(cfg.steps))):
        soft_text, probabilities, query, assignments = _gated_row_group_payload(
            x_param, labels, row_bank, graph_text_map, prompt_emb,
            cfg.joint_base_logit, cfg.assignment_temperature, device)
        loss_x = _class_mmd(source_features, x_param, raw_labels, labels,
                            x_weight)
        cross_loss = (1.0 - F.cosine_similarity(query, soft_text, dim=1)).mean()
        prompt_loss = _prompt_margin_loss(
            soft_text, labels, prompt_emb, cfg.prompt_margin, device)
        text_anchor = (1.0 - F.cosine_similarity(
            soft_text, base_text, dim=1)).mean()
        base_nll = -probabilities[:, 0].clamp_min(1e-8).log().mean()
        x_anchor = (x_param - x0).square().mean()
        loss = (loss_x / scale_x
                + float(cfg.lambda_cross) * cross_loss / scale_cross
                + float(cfg.lambda_prompt_margin) * prompt_loss
                + float(cfg.lambda_cctp_anchor) * text_anchor
                + float(cfg.lambda_joint_base) * base_nll
                + float(cfg.lambda_anchor_x) * x_anchor)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite gated joint CaT/text loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x_param], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            delta_x = x_param - x0
            row_norm = delta_x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            allowed = float(cfg.joint_x_trust_radius) * \
                x0.norm(dim=1, keepdim=True).clamp_min(1e-3)
            scale = (allowed / row_norm).clamp(max=1.0)
            x_param.copy_(x0 + delta_x * scale)
        if step == 0 or (step + 1) % 50 == 0 or step == cfg.steps - 1:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "x_mmd": float(loss_x.detach().cpu()),
                "cross": float(cross_loss.detach().cpu()),
                "prompt_margin": float(prompt_loss.detach().cpu()),
                "text_anchor": float(text_anchor.detach().cpu()),
                "base_nll": float(base_nll.detach().cpu()),
                "x_anchor": float(x_anchor.detach().cpu()),
                "base_probability_mean": float(
                    probabilities[:, 0].mean().detach().cpu()),
            })
    with torch.no_grad():
        final_soft, final_prob, final_query, final_assignments = \
            _gated_row_group_payload(
                x_param, labels, row_bank, graph_text_map, prompt_emb,
                cfg.joint_base_logit, cfg.assignment_temperature, device)
    final_x = x_param.detach().cpu()
    final_texts, final_ids, final_emb, hard_audit = [], [], [], []
    changed_rows = 0
    for row, audit_row in enumerate(final_assignments):
        base_logit = float(audit_row["base_logit"])
        alternative = int(audit_row["group_indices"][0])
        best_alternative = float(audit_row["best_alternative_logit"])
        selected = (alternative if alternative != 0 and
                    best_alternative - base_logit >=
                    float(cfg.joint_selection_margin) else 0)
        if selected != 0:
            changed_rows += 1
        group_record = row_bank["rows"][row][selected]
        final_texts.append(group_record["text"])
        final_ids.append(group_record["source_ids"])
        final_emb.append(group_record["embedding"])
        hard_audit.append({
            "row": int(row),
            "selected_group_local": int(selected),
            "changed_from_cctp": bool(selected != 0),
            "source_local_indices": group_record["source_local_indices"],
            "source_ids": group_record["source_ids"],
            "text_length": len(group_record["text"]),
        })
    final_emb = F.normalize(torch.stack(final_emb), dim=1)
    block = {
        "x": final_x,
        "text_emb": final_emb,
        "texts": final_texts,
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x))),
        "depth": torch.zeros(len(final_x), dtype=torch.long),
        "stage": torch.full((len(final_x),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    soft_cpu = final_soft.detach().cpu()
    audit = {
        "constructor": "cat_gated_joint_text_group",
        "objective": "CaT graph MMD + gated X-conditioned text objective",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "text_representation": "CCTP group or margin-approved real alternative",
        "graph_text_fit_cosine": float(fit_cosine),
        "assignment_temperature": float(cfg.assignment_temperature),
        "joint_base_logit": float(cfg.joint_base_logit),
        "joint_selection_margin": float(cfg.joint_selection_margin),
        "row_classes": labels.cpu().tolist(),
        "text_source_ids": final_ids,
        "soft_group_assignment": final_assignments,
        "hard_group_materialization": hard_audit,
        "changed_rows": int(changed_rows),
        "changed_fraction": float(changed_rows / max(1, len(labels))),
        "group_bank_count_per_row": row_bank["count_per_row"],
        "group_bank": [{
            "row": int(row),
            "class": int(r["class"]),
            "group_local": int(r["group_index"]),
            "source_local_indices": r["source_local_indices"],
            "source_ids": r["source_ids"],
            "selector_mode": r["selector_mode"],
            "text_length": len(r["text"]),
        } for row, records in enumerate(row_bank["rows"])
        for r in records],
        "soft_hard_cosine_mean": float(
            F.cosine_similarity(soft_cpu, final_emb, dim=1).mean()),
        "soft_hard_cosine_min": float(
            F.cosine_similarity(soft_cpu, final_emb, dim=1).min()),
        "config": asdict(cfg),
        "trace": trace,
        "normalization_scales": {
            "x_mmd": float(scale_x.cpu()),
            "cross": float(scale_cross.cpu()),
        },
        "x_update_norm": float((final_x - x0.cpu()).norm()),
        "x_update_relative_max": float((final_x - x0.cpu()).norm(dim=1)
                                        .div(x0.cpu().norm(dim=1).clamp_min(1e-3))
                                        .max()),
        "final_text_vs_base_cosine": float(
            F.cosine_similarity(final_emb, text0.cpu(), dim=1).mean()),
        "final_text_norm_mean": float(final_emb.norm(dim=1).mean()),
    }
    return block, audit


def optimize_text_group_aware_cat(delta: dict, source_features: torch.Tensor,
                                  x0: torch.Tensor, text0: torch.Tensor,
                                  texts: list[str], source_ids: list[list[int]],
                                  labels: torch.Tensor, parent: dict | None,
                                  encoder, cfg: MMConfig,
                                  mass: torch.Tensor | None,
                                  device: str) -> tuple[dict, dict]:
    """Optimize CaT X against a finite, materializable text-group bank.

    This is the consistency-corrected variant of ``optimize_text_aware_cat``.
    The only trainable object is X.  Text affects X through a soft assignment
    over real CCTP strings, while the final block stores the top-scoring real
    group rather than an unrealizable convex combination.
    """
    source_features = source_features.detach().to(device).float()
    x0 = x0.detach().to(device).float()
    text0 = F.normalize(text0.detach().to(device).float(), dim=1)
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_labels = delta["y"].detach().long().to(device)
    labels = labels.detach().long().to(device)
    if parent is not None and parent["edge_index"].numel():
        raise ValueError("text-group CaT requires edge-free parent topology")
    graph_text_map, fit_cosine = M._fit_graph_text_map(
        source_features, raw_text, float(cfg.text_ridge))
    group_bank = _build_text_group_bank(delta, source_features, encoder, cfg,
                                        graph_text_map.detach().cpu())
    x_param = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([x_param], lr=float(cfg.lr_x))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 8801)
    scale_x_weight = torch.empty((source_features.shape[1], int(cfg.hidden)),
                                 device=device)
    scale_x_weight.uniform_(-_bound(source_features.shape[1], cfg.hidden),
                            _bound(source_features.shape[1], cfg.hidden),
                            generator=generator)
    scale_text_weight = torch.empty((group_bank["embeddings"].shape[1],
                                     int(cfg.hidden)), device=device)
    scale_text_weight.uniform_(-_bound(group_bank["embeddings"].shape[1],
                                       cfg.hidden),
                               _bound(group_bank["embeddings"].shape[1],
                                      cfg.hidden), generator=generator)
    with torch.no_grad():
        baseline_soft, _ = _soft_group_payload(
            raw_labels, x0, labels, graph_text_map, group_bank,
            cfg.assignment_temperature, device)
        scale_x = _class_mmd(source_features, x0, raw_labels, labels,
                             scale_x_weight).clamp_min(1e-8)
        scale_group = _class_mmd(
            group_bank["embeddings"].to(device), baseline_soft,
            group_bank["labels"].to(device), labels, scale_text_weight
        ).clamp_min(1e-8)
        base_query = F.normalize(x0, dim=1)
        predicted = F.normalize(
            torch.cat((base_query, torch.ones((len(base_query), 1),
                                               device=device)), dim=1)
            @ graph_text_map, dim=1)
        scale_cross = (1.0 - F.cosine_similarity(
            predicted, baseline_soft, dim=1)).mean().clamp_min(1e-4)
    trace, final_assignments = [], []
    for step in range(max(1, int(cfg.steps))):
        x_weight = torch.empty((source_features.shape[1], int(cfg.hidden)),
                               device=device)
        x_weight.uniform_(-_bound(source_features.shape[1], cfg.hidden),
                          _bound(source_features.shape[1], cfg.hidden),
                          generator=generator)
        soft_text, assignments = _soft_group_payload(
            raw_labels, x_param, labels, graph_text_map, group_bank,
            cfg.assignment_temperature, device)
        loss_x = _class_mmd(source_features, x_param, raw_labels, labels,
                            x_weight)
        loss_group = _class_mmd(
            group_bank["embeddings"].to(device), soft_text,
            group_bank["labels"].to(device), labels, scale_text_weight)
        query = F.normalize(x_param, dim=1)
        predicted = F.normalize(
            torch.cat((query, torch.ones((len(query), 1), device=device)), dim=1)
            @ graph_text_map, dim=1)
        cross_loss = (1.0 - F.cosine_similarity(
            predicted, soft_text, dim=1)).mean()
        # Recompute the differentiable entropy directly from the group scores;
        # the audit form above is detached by design.
        entropy_terms = []
        bank_embeddings = group_bank["embeddings"].to(device).float()
        bank_labels = group_bank["labels"].to(device).long()
        for row, cls in enumerate(labels.detach().cpu().tolist()):
            group_ids = torch.where(bank_labels == int(cls))[0]
            scores = predicted[row:row + 1] @ bank_embeddings[group_ids].T
            probs = F.softmax(scores.squeeze(0) /
                              max(float(cfg.assignment_temperature), 1e-6), dim=0)
            entropy_terms.append(-(probs.clamp_min(1e-12)
                                   * probs.clamp_min(1e-12).log()).sum()
                                  / max(1.0, math.log(len(group_ids))))
        entropy_loss = (torch.stack(entropy_terms).mean()
                        if entropy_terms else x_param.sum() * 0.0)
        anchor_x = (x_param - x0).square().mean()
        anchor_group = (1.0 - F.cosine_similarity(
            soft_text, baseline_soft, dim=1)).mean()
        loss = (loss_x / scale_x
                + float(cfg.lambda_text) * loss_group / scale_group
                + float(cfg.lambda_cross) * cross_loss / scale_cross
                + float(cfg.lambda_group_entropy) * entropy_loss
                + float(cfg.lambda_anchor_x) * anchor_x
                + float(cfg.lambda_anchor_text) * anchor_group)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite text-group CaT loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step == cfg.steps - 1:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "x_mmd": float(loss_x.detach().cpu()),
                "group_mmd": float(loss_group.detach().cpu()),
                "cross": float(cross_loss.detach().cpu()),
                "group_entropy": float(entropy_loss.detach().cpu()),
                "x_anchor": float(anchor_x.detach().cpu()),
                "group_anchor": float(anchor_group.detach().cpu()),
            })
    with torch.no_grad():
        final_soft, final_assignments = _soft_group_payload(
            raw_labels, x_param, labels, graph_text_map, group_bank,
            cfg.assignment_temperature, device)
    final_x = x_param.detach().cpu()
    final_texts, final_ids, final_emb = [], [], []
    hard_audit = []
    for audit_row in final_assignments:
        group_id = int(audit_row["group_indices"][0])
        group_record = None
        for record in group_bank["groups_by_class"][audit_row["class"]]:
            global_id = int(record["global_index"])
            if global_id == group_id:
                group_record = record
                break
        if group_record is None:
            # Global indices are attached below, but this fallback keeps the
            # error explicit if an audit or bank is malformed.
            raise RuntimeError(f"missing selected text group {group_id}")
        final_texts.append(group_record["text"])
        final_ids.append(group_record["source_ids"])
        final_emb.append(group_record["embedding"])
        hard_audit.append({
            "class": audit_row["class"],
            "selected_global_group": group_id,
            "source_local_indices": group_record["source_local_indices"],
            "source_ids": group_record["source_ids"],
            "text_length": len(group_record["text"]),
        })
    final_emb = F.normalize(torch.stack(final_emb), dim=1)
    block = {
        "x": final_x,
        "text_emb": final_emb,
        "texts": final_texts,
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x))),
        "depth": torch.zeros(len(final_x), dtype=torch.long),
        "stage": torch.full((len(final_x),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    soft_cpu = final_soft.detach().cpu()
    soft_hard_cosine = F.cosine_similarity(soft_cpu, final_emb, dim=1)
    audit = {
        "constructor": "cat_text_group_aware_x",
        "objective": "CaT graph MMD + materialized text-group MMD + graph-text consistency",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "text_representation": "top-scoring real CCTP text group",
        "graph_text_fit_cosine": float(fit_cosine),
        "assignment_temperature": float(cfg.assignment_temperature),
        "row_classes": labels.cpu().tolist(),
        "text_source_ids": final_ids,
        "soft_group_assignment": final_assignments,
        "hard_group_materialization": hard_audit,
        "group_bank_count_per_class": group_bank["count_per_class"],
        "group_bank": [{
            "class": int(r["class"]),
            "global_index": int(r["global_index"]),
            "source_local_indices": r["source_local_indices"],
            "source_ids": r["source_ids"],
            "selector_mode": r["selector_mode"],
            "text_length": len(r["text"]),
        } for records in group_bank["groups_by_class"].values() for r in records],
        "soft_hard_cosine_mean": float(soft_hard_cosine.mean()),
        "soft_hard_cosine_min": float(soft_hard_cosine.min()),
        "config": asdict(cfg),
        "trace": trace,
        "normalization_scales": {
            "x_mmd": float(scale_x.cpu()),
            "group_mmd": float(scale_group.cpu()),
            "cross": float(scale_cross.cpu()),
        },
        "x_update_norm": float((final_x - x0.cpu()).norm()),
        "soft_text_update_norm": float((soft_cpu - text0.cpu()).norm()),
        "final_text_vs_base_cosine": float(
            F.cosine_similarity(final_emb, text0.cpu(), dim=1).mean()),
        "final_text_norm_mean": float(final_emb.norm(dim=1).mean()),
    }
    return block, audit


def optimize_text_aware_cat(delta: dict, source_features: torch.Tensor,
                            x0: torch.Tensor, text0: torch.Tensor,
                            texts: list[str], source_ids: list[list[int]],
                            labels: torch.Tensor, parent: dict | None,
                            encoder, cfg: MMConfig,
                            mass: torch.Tensor | None,
                            device: str,
                            materialize: str = "base") -> tuple[dict, dict]:
    """Optimize CaT X with an X-conditioned differentiable text payload.

    Unlike ``optimize_text_payload``, this does not optimize a free text
    parameter and does not use Student response.  The current X predicts a
    soft distribution over legal same-class raw text sources; text MMD and a
    graph-text consistency term therefore backpropagate directly into X.
    """
    if materialize not in ("base", "soft", "hard"):
        raise ValueError("materialize must be 'base', 'soft', or 'hard'")
    source_features = source_features.detach().to(device).float()
    x0 = x0.detach().to(device).float()
    text0 = F.normalize(text0.detach().to(device).float(), dim=1)
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_labels = delta["y"].detach().long().to(device)
    labels = labels.detach().long().to(device)
    if parent is not None and parent["edge_index"].numel():
        raise ValueError("text-aware CaT requires edge-free parent topology")

    graph_text_map, fit_cosine = M._fit_graph_text_map(
        source_features, raw_text, float(cfg.text_ridge))
    x_param = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([x_param], lr=float(cfg.lr_x))
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    with torch.no_grad():
        baseline_soft_text, _ = _soft_text_payload(
            source_features, raw_text, raw_labels, x0, labels,
            graph_text_map, cfg.assignment_temperature)
    scale_x_weight = torch.empty((source_features.shape[1], int(cfg.hidden)),
                                 device=device)
    scale_x_weight.uniform_(-_bound(source_features.shape[1], cfg.hidden),
                            _bound(source_features.shape[1], cfg.hidden),
                            generator=generator)
    scale_text_weight = torch.empty((raw_text.shape[1], int(cfg.hidden)),
                                    device=device)
    scale_text_weight.uniform_(-_bound(raw_text.shape[1], cfg.hidden),
                               _bound(raw_text.shape[1], cfg.hidden),
                               generator=generator)
    with torch.no_grad():
        scale_x = _class_mmd(source_features, x0, raw_labels, labels,
                             scale_x_weight).clamp_min(1e-8)
        scale_text = _class_mmd(raw_text, baseline_soft_text,
                                raw_labels, labels, scale_text_weight).clamp_min(1e-8)
        base_query = F.normalize(x0, dim=1)
        base_predicted = F.normalize(
            torch.cat((base_query, torch.ones((len(base_query), 1),
                                               device=device)), dim=1)
            @ graph_text_map, dim=1)
        scale_cross = (1.0 - F.cosine_similarity(
            base_predicted, baseline_soft_text, dim=1)).mean().clamp_min(1e-4)
    trace = []
    final_assignments = []
    final_soft_text = None
    for step in range(max(1, int(cfg.steps))):
        x_weight = torch.empty((source_features.shape[1], int(cfg.hidden)),
                               device=device)
        x_weight.uniform_(-_bound(source_features.shape[1], cfg.hidden),
                          _bound(source_features.shape[1], cfg.hidden),
                          generator=generator)
        text_weight = torch.empty((raw_text.shape[1], int(cfg.hidden)),
                                  device=device)
        text_weight.uniform_(-_bound(raw_text.shape[1], cfg.hidden),
                             _bound(raw_text.shape[1], cfg.hidden),
                             generator=generator)
        soft_text, assignments = _soft_text_payload(
            source_features, raw_text, raw_labels, x_param, labels,
            graph_text_map, cfg.assignment_temperature)
        loss_x = _class_mmd(source_features, x_param, raw_labels, labels,
                            x_weight)
        loss_text = _class_mmd(raw_text, soft_text, raw_labels, labels,
                               text_weight)
        graph_query = F.normalize(x_param, dim=1)
        predicted_text = F.normalize(
            torch.cat((graph_query, torch.ones((len(graph_query), 1),
                                                device=device)), dim=1)
            @ graph_text_map, dim=1)
        cross_loss = (1.0 - F.cosine_similarity(
            predicted_text, soft_text, dim=1)).mean()
        anchor_x = (x_param - x0).square().mean()
        anchor_text = (1.0 - F.cosine_similarity(
            soft_text, baseline_soft_text, dim=1)).mean()
        loss = (loss_x / scale_x
                + float(cfg.lambda_text) * loss_text / scale_text
                + float(cfg.lambda_cross) * cross_loss / scale_cross
                + float(cfg.lambda_anchor_x) * anchor_x
                + float(cfg.lambda_anchor_text) * anchor_text)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite text-aware CaT loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step == cfg.steps - 1:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "x_mmd": float(loss_x.detach().cpu()),
                "text_mmd": float(loss_text.detach().cpu()),
                "cross": float(cross_loss.detach().cpu()),
                "x_mmd_normalized": float((loss_x / scale_x).detach().cpu()),
                "text_mmd_normalized": float((loss_text / scale_text).detach().cpu()),
                "cross_normalized": float((cross_loss / scale_cross).detach().cpu()),
                "x_anchor": float(anchor_x.detach().cpu()),
                "text_anchor": float(anchor_text.detach().cpu()),
            })

    with torch.no_grad():
        final_soft_text, final_assignments = _soft_text_payload(
            source_features, raw_text, raw_labels, x_param, labels,
            graph_text_map, cfg.assignment_temperature)
    final_x = x_param.detach().cpu()
    soft_text_cpu = final_soft_text.detach().cpu()
    if "texts" in delta and "ids" in delta:
        hard_texts, hard_ids, hard_emb, materialization_audit = \
            _materialize_soft_payload(delta, labels, final_assignments, encoder,
                                      cfg.text_sources, cfg.text_chars)
    else:
        # Small unit-test fixtures and analytical callers may provide only
        # embeddings.  The soft path does not require string materialization.
        hard_texts, hard_ids = list(texts), list(source_ids)
        hard_emb = soft_text_cpu.clone()
        materialization_audit = []
    if materialize == "soft":
        final_text = soft_text_cpu
        final_texts = list(texts)
        final_ids = [r["candidate_ids"][:cfg.text_sources]
                     for r in final_assignments]
    elif materialize == "hard":
        final_text = hard_emb
        final_texts = hard_texts
        final_ids = hard_ids
    else:
        final_text = text0.cpu()
        final_texts = list(texts)
        final_ids = list(source_ids)
    block = {
        "x": final_x,
        "text_emb": final_text,
        "texts": final_texts,
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x))),
        "depth": torch.zeros(len(final_x), dtype=torch.long),
        "stage": torch.full((len(final_x),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    soft_hard_cosine = F.cosine_similarity(soft_text_cpu, hard_emb, dim=1)
    audit = {
        "constructor": f"cat_text_aware_{materialize}",
        "objective": "CaT graph MMD + X-conditioned text MMD + graph-text consistency",
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "budget_updated": False,
        "text_representation": {
            "base": "X-conditioned soft transport used only in CaT objective; CCTP text retained",
            "soft": "X-conditioned soft same-class text transport",
            "hard": "X-conditioned soft transport plus hard top-k re-encode",
        }[materialize],
        "materialization": materialize,
        "graph_text_fit_cosine": float(fit_cosine),
        "assignment_temperature": float(cfg.assignment_temperature),
        "row_classes": labels.cpu().tolist(),
        "text_source_ids": final_ids,
        "soft_assignment": final_assignments,
        "hard_materialization": materialization_audit,
        "soft_hard_cosine_mean": float(soft_hard_cosine.mean()),
        "soft_hard_cosine_min": float(soft_hard_cosine.min()),
        "config": asdict(cfg),
        "trace": trace,
        "normalization_scales": {
            "x_mmd": float(scale_x.cpu()),
            "text_mmd": float(scale_text.cpu()),
            "cross": float(scale_cross.cpu()),
        },
        "x_update_norm": float((final_x - x0.cpu()).norm()),
        "text_update_norm": float((soft_text_cpu - text0.cpu()).norm()),
        "final_text_vs_base_cosine": float(
            F.cosine_similarity(final_text, text0.cpu(), dim=1).mean()),
        "final_text_norm_mean": float(final_text.norm(dim=1).mean()),
    }
    return block, audit


def optimize_text_payload(delta: dict, source_features: torch.Tensor,
                          x0: torch.Tensor, text0: torch.Tensor,
                          texts: list[str], source_ids: list[list[int]],
                          labels: torch.Tensor, parent: dict | None,
                          cfg: MMConfig, mode: str,
                          mass: torch.Tensor | None,
                          device: str) -> tuple[dict, dict]:
    """Optimize current-stage continuous text payloads.

    ``mode`` is ``text_only`` or ``joint``.  The raw text strings are kept as
    provenance only; the Student-facing payload is the optimized embedding.
    """
    if mode not in ("text_only", "joint"):
        raise ValueError(f"unknown multimodal mode: {mode}")
    source_features = source_features.detach().to(device).float()
    x0 = x0.detach().to(device).float()
    text0 = F.normalize(text0.detach().to(device).float(), dim=1)
    labels = labels.detach().long().to(device)
    raw_x = source_features
    raw_text = F.normalize(delta["text_emb"].detach().to(device).float(), dim=1)
    raw_labels = delta["y"].detach().to(device).long()

    x_param = torch.nn.Parameter(x0.clone())
    text_param = torch.nn.Parameter(text0.clone())
    params = [text_param] if mode == "text_only" else [x_param, text_param]
    optimizer = torch.optim.Adam([
        {"params": [text_param], "lr": float(cfg.lr_text)},
        *([{"params": [x_param], "lr": float(cfg.lr_x)}]
          if mode == "joint" else []),
    ])
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    trace = []
    for step in range(max(1, int(cfg.steps))):
        x_weight = torch.empty(
            (raw_x.shape[1], int(cfg.hidden)), device=device)
        x_weight.uniform_(-_bound(raw_x.shape[1], cfg.hidden),
                          _bound(raw_x.shape[1], cfg.hidden),
                          generator=generator)
        text_weight = torch.empty(
            (raw_text.shape[1], int(cfg.hidden)), device=device)
        text_weight.uniform_(-_bound(raw_text.shape[1], cfg.hidden),
                             _bound(raw_text.shape[1], cfg.hidden),
                             generator=generator)
        text_value = F.normalize(text_param, dim=1)
        loss_text = _class_mmd(raw_text, text_value, raw_labels, labels,
                               text_weight)
        anchor_text = (1.0 - F.cosine_similarity(
            text_value, text0, dim=1)).mean()
        if mode == "joint":
            loss_x = _class_mmd(raw_x, x_param, raw_labels, labels,
                                x_weight)
            anchor_x = (x_param - x0).square().mean()
            loss = (loss_x + float(cfg.lambda_text) * loss_text
                    + float(cfg.lambda_anchor_text) * anchor_text
                    + float(cfg.lambda_anchor_x) * anchor_x)
        else:
            loss_x = x_param.sum() * 0.0
            anchor_x = x_param.sum() * 0.0
            loss = (loss_text
                    + float(cfg.lambda_anchor_text) * anchor_text)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite multimodal CaT loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step == cfg.steps - 1:
            trace.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "text_mmd": float(loss_text.detach().cpu()),
                "x_mmd": float(loss_x.detach().cpu()),
                "text_anchor": float(anchor_text.detach().cpu()),
                "x_anchor": float(anchor_x.detach().cpu()),
            })

    final_x = x_param.detach().cpu() if mode == "joint" else x0.cpu()
    final_text = F.normalize(text_param.detach(), dim=1).cpu()
    block = {
        "x": final_x,
        "text_emb": final_text,
        "texts": list(texts),
        "mass": (mass.detach().cpu() if mass is not None
                 else torch.ones(len(final_x))),
        "depth": torch.zeros(len(final_x), dtype=torch.long),
        "stage": torch.full((len(final_x),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    audit = {
        "constructor": f"cat_multimodal_{mode}",
        "current_delta_only": True,
        "old_memory_updated": False,
        "text_representation": "continuous optimized MiniLM embedding",
        "text_provenance": "CCTP anchor strings retained for audit only",
        "row_classes": labels.cpu().tolist(),
        "text_source_ids": source_ids,
        "config": asdict(cfg),
        "trace": trace,
        "x_update_norm": float((final_x - x0.cpu()).norm()),
        "text_update_norm": float((final_text - text0.cpu()).norm()),
        "final_text_norm_mean": float(final_text.norm(dim=1).mean()),
    }
    return block, audit
