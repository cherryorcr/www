"""X-conditioned differentiable routing over materialized text payloads.

This module keeps CaT's synthetic features fixed and only changes how the
current-stage, real text candidates are assigned to synthetic rows.  It is a
controlled extension of the class-shared payload assignment:

* candidates are real strings made from the current raw delta;
* each class has a small candidate bank (slots), not one broadcast payload;
* a row-conditioned soft assignment is optimized from X/text compatibility;
* the stored memory uses the selected real candidate, not a synthetic text
  vector that cannot be audited or materialized.

The constructor never reads historical raw data, never changes old memory,
and never creates topology edges.  It is intentionally separate from the
downstream Student response objective.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import torch
import torch.nn.functional as F

import cat_multimodal_minimal as MM


def _truncate_text_boundary(text: str, limit: int) -> str:
    """Keep a bounded real payload without cutting through a word if possible.

    The 320-character cap is part of the frozen memory budget.  This helper
    only changes where the cap is applied: sentence boundaries are preferred,
    then whitespace, and a hard character cut is used only when one segment is
    longer than the complete budget.
    """
    limit = max(1, int(limit))
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized

    prefix = normalized[:limit]
    sentence_end = [match.end() for match in re.finditer(
        r"[.!?;。！？；]", prefix)]
    min_boundary = max(16, int(limit * 0.55))
    valid_sentence = [value for value in sentence_end
                      if min_boundary <= value <= limit]
    if valid_sentence:
        return prefix[:max(valid_sentence)].rstrip()

    word_boundary = prefix.rfind(" ")
    if word_boundary >= min_boundary:
        return prefix[:word_boundary].rstrip()
    return prefix.rstrip()


def _norm(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1)


def _candidate_key(record: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (str(record.get("text", "")), tuple(
        int(v) for v in record.get("source_ids", [])))


def _record_embedding(record: dict[str, Any], encoder) -> torch.Tensor:
    value = record.get("embedding")
    if value is None:
        value = encoder.encode([str(record.get("text", ""))])[0]
    return _norm(torch.as_tensor(value).view(1, -1)).squeeze(0).cpu()


def _batch_text_embeddings(encoder, texts: list[str]) -> dict[str, torch.Tensor]:
    """Encode each distinct text once.

    Large deltas can contain thousands of rows.  Calling MiniLM once per
    candidate makes coherent routing CPU-bound and repeats work for identical
    payload strings.  The mapping is deliberately keyed by the exact
    materialized string so this is an implementation optimization only; it
    does not change candidate selection or routing semantics.
    """
    unique = list(dict.fromkeys(str(value) for value in texts if str(value)))
    if not unique:
        return {}
    encoded = encoder.encode(unique)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu()
    encoded = torch.as_tensor(encoded).float()
    if encoded.ndim != 2 or len(encoded) != len(unique):
        raise ValueError(
            f"unexpected text embedding shape {tuple(encoded.shape)} for "
            f"{len(unique)} texts")
    return {
        text: _norm(value.view(1, -1)).squeeze(0).cpu()
        for text, value in zip(unique, encoded)
    }


def _record_graph_prototype(record: dict[str, Any], raw_x: torch.Tensor,
                            id_to_local: dict[int, int], fallback: torch.Tensor
                            ) -> torch.Tensor:
    indices = [int(v) for v in record.get("source_local_indices", [])]
    if not indices:
        indices = [id_to_local[int(v)] for v in record.get("source_ids", [])
                   if int(v) in id_to_local]
    if not indices:
        return _norm(fallback.view(1, -1)).squeeze(0).cpu()
    indices = [v for v in indices if 0 <= int(v) < len(raw_x)]
    if not indices:
        return _norm(fallback.view(1, -1)).squeeze(0).cpu()
    return _norm(raw_x[indices].mean(0, keepdim=True)).squeeze(0).cpu()


def _prompt_margin(embedding: torch.Tensor, cls: int,
                   prompt_emb: torch.Tensor | None, margin: float) -> float:
    if prompt_emb is None or int(cls) >= len(prompt_emb) or len(prompt_emb) <= 1:
        return 0.0
    prompts = _norm(prompt_emb)
    scores = _norm(embedding.view(1, -1)) @ prompts.T
    positive = scores[0, int(cls)]
    negative = scores.clone()
    negative[0, int(cls)] = -float("inf")
    return float(F.relu(float(margin) - positive + negative.max()).item())


def _select_diverse_slots(records: list[dict[str, Any]], rows: list[int],
                          x_value: torch.Tensor, raw_x: torch.Tensor,
                          id_to_local: dict[int, int], class_center: torch.Tensor,
                          prompt_emb: torch.Tensor | None, cls: int,
                          max_slots: int, margin: float,
                          candidate_mode: str = "standard") -> list[dict[str, Any]]:
    """Select a bounded, deterministic candidate bank for one class.

    The selection is only candidate-bank construction.  It does not inspect
    report/test nodes or final Student metrics.  A best-compatibility seed is
    followed by farthest-point choices in the joint graph/text signature so
    the routing layer has genuine alternatives instead of duplicate strings.
    """
    unique: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for record in records:
        key = _candidate_key(record)
        if key[0] and key not in unique:
            unique[key] = record
    candidates = list(unique.values())
    if not candidates:
        raise ValueError(f"empty payload bank for class {cls}")

    x_rows = _norm(x_value[rows])
    prepared: list[dict[str, Any]] = []
    for record in candidates:
        # ``embedding`` is filled by the caller before this helper is called.
        text_emb = _norm(record["embedding"].view(1, -1)).squeeze(0).cpu()
        graph_proto = _record_graph_prototype(
            record, raw_x, id_to_local, x_value[rows[0]])
        graph_sim = x_rows @ graph_proto
        text_sim = float(text_emb @ class_center)
        prompt_penalty = float(record.get("prompt_margin", 0.0))
        prepared.append({
            "record": record,
            "embedding": text_emb,
            "graph_proto": graph_proto,
            "graph_sim": graph_sim,
            "base_cost": float(1.0 - graph_sim.mean().item()
                                + 0.50 * (1.0 - text_sim)
                                + 0.15 * prompt_penalty),
        })

    selected: list[dict[str, Any]] = [min(
        prepared, key=lambda item: (item["base_cost"],
                                    item["record"].get("mode", ""),
                                    item["record"].get("source_ids", [])))]
    target = max(1, min(int(max_slots), len(prepared), len(rows)))
    while len(selected) < target:
        best = None
        best_score = None
        for item in prepared:
            if any(item is value for value in selected):
                continue
            text_dist = min(float(1.0 - item["embedding"] @ value["embedding"])
                            for value in selected)
            graph_dist = min(float(1.0 - item["graph_proto"] @
                                  value["graph_proto"])
                             for value in selected)
            if candidate_mode == "x_response_coverage":
                # Select candidates with complementary compatibility profiles
                # over the class's CaT X rows, not only diverse text strings.
                response_dist = min(
                    float(1.0 - F.cosine_similarity(
                        item["graph_sim"].view(1, -1),
                        value["graph_sim"].view(1, -1),
                        dim=1).item())
                    for value in selected)
                score = (0.55 * response_dist + 0.25 * text_dist
                         + 0.20 * graph_dist - 0.10 * item["base_cost"])
            else:
                score = (0.50 * text_dist + 0.35 * graph_dist
                         - 0.15 * item["base_cost"])
            tie = (score, -item["base_cost"],
                   str(item["record"].get("mode", "")))
            if best is None or tie > best_score:
                best, best_score = item, tie
        if best is None:
            break
        selected.append(best)
    return selected


def _balanced_hard_slots(probabilities: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Materialize a soft route with near-uniform slot capacities.

    The soft assignment already contains a load-balancing term, but taking an
    unconstrained row-wise argmax can still collapse on large classes.  This
    deterministic greedy projection keeps the highest-regret rows first and
    assigns them to their best remaining slot, with capacities differing by at
    most one.  It changes only hard materialization; the candidate costs and
    differentiable probabilities remain untouched.
    """
    if probabilities.ndim != 2 or not probabilities.numel():
        raise ValueError("probabilities must be a non-empty matrix")
    rows, slots = probabilities.shape
    base = rows // slots
    remainder = rows % slots
    capacities = [base + (1 if slot < remainder else 0)
                  for slot in range(slots)]
    selected = torch.full((rows,), -1, dtype=torch.long)
    if slots == 1:
        return torch.zeros(rows, dtype=torch.long), capacities

    top_values, top_indices = torch.topk(
        probabilities, k=min(2, slots), dim=1)
    regret = top_values[:, 0] - top_values[:, 1]
    order = torch.argsort(regret, descending=True, stable=True).tolist()
    for row in order:
        ranking = torch.argsort(probabilities[row], descending=True,
                                stable=True).tolist()
        slot = next((candidate for candidate in ranking
                     if capacities[candidate] > 0), None)
        if slot is None:
            raise AssertionError("balanced slot capacities were exhausted")
        selected[row] = int(slot)
        capacities[slot] -= 1
    used = torch.bincount(selected, minlength=slots).tolist()
    return selected, [int(value) for value in used]


def _routing_cost_matrix(x_rows: torch.Tensor,
                         candidates: list[dict[str, Any]],
                         raw_x: torch.Tensor,
                         id_to_local: dict[int, int],
                         prompt_emb: torch.Tensor | None, cls: int,
                         cfg: MM.MMConfig) -> torch.Tensor:
    """Recompute the fixed row-to-candidate cost used by the router."""
    x_rows = x_rows.detach().cpu().float()
    graph_protos = torch.stack([
        _record_graph_prototype(value, raw_x, id_to_local, x_rows[0])
        for value in candidates]).float()
    text_emb = _norm(torch.stack([
        _norm(value["embedding"].view(1, -1)).squeeze(0).cpu()
        for value in candidates]))
    graph_sim = _norm(x_rows) @ _norm(graph_protos).T
    text_center = _norm(text_emb.mean(0, keepdim=True)).squeeze(0)
    row_text_cost = 1.0 - text_emb @ text_center
    margin = torch.tensor([
        _prompt_margin(value, int(cls), prompt_emb, float(cfg.prompt_margin))
        for value in text_emb], dtype=torch.float32)
    return (1.0 - graph_sim
            + float(cfg.payload_assignment_text_weight)
            * row_text_cost.view(1, -1)
            + float(cfg.lambda_prompt_margin) * margin.view(1, -1))


def _collect_class_candidates(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor,
        base_texts: list[str], base_source_ids: list[list[int]],
        labels: torch.Tensor, encoder, cfg: MM.MMConfig,
        max_slots: int, candidate_mode: str = "standard",
        adaptive_slots: bool = False
        ) -> tuple[dict[int, list[dict[str, Any]]], dict]:
    """Build and select bounded real candidate slots for every class."""
    labels_cpu = labels.detach().cpu().long()
    x_cpu = x0.detach().cpu().float()
    source_features = source_features.detach().cpu().float()
    raw_x = delta.get("x", source_features).detach().cpu().float()
    if raw_x.ndim != 2 or raw_x.shape[1] != x_cpu.shape[1]:
        raw_x = source_features
    raw_x = raw_x.float()
    id_to_local = {int(v): i for i, v in enumerate(
        delta["ids"].detach().cpu().tolist())}
    prompt_emb = delta.get("prompt_emb")
    if prompt_emb is not None:
        prompt_emb = prompt_emb.detach().cpu().float()
    coherent_mode = candidate_mode in (
        "coherent_raw_medoid", "coherent_raw_medoid_anchor")
    bank = None
    if not coherent_mode:
        bank = MM._build_chunk_payload_bank(
            delta, source_features, x_cpu, text0.detach().cpu().float(),
            labels_cpu, base_texts, base_source_ids, encoder, cfg)

    # Build a single cache for all current-delta payload strings.  In the
    # coherent mode this replaces one MiniLM call per raw row; in the regular
    # mode it also removes repeated encoding of CCTP strings across rows.
    text_embedding_cache: dict[str, torch.Tensor] = {}
    texts_to_encode: list[str] = []
    if coherent_mode:
        texts_to_encode.extend(
            _truncate_text_boundary(value, int(cfg.text_chars))
            for value in delta["texts"])
    else:
        texts_to_encode.extend(str(value) for value in base_texts)
        for row_value in bank["rows"]:
            for record in row_value["records"].values():
                if record.get("embedding") is None:
                    texts_to_encode.append(str(record.get("text", "")))
    text_embedding_cache.update(_batch_text_embeddings(encoder, texts_to_encode))

    raw_text = _norm(delta["text_emb"].detach().cpu().float())
    raw_labels = delta["y"].detach().cpu().long()
    rows_by_class: dict[int, list[int]] = defaultdict(list)
    for row, cls in enumerate(labels_cpu.tolist()):
        rows_by_class[int(cls)].append(int(row))

    all_classes = sorted(torch.unique(raw_labels).tolist())
    class_centers = {}
    for cls in all_classes:
        class_centers[int(cls)] = _norm(
            raw_text[raw_labels == int(cls)].mean(0, keepdim=True)).squeeze(0)

    candidates_by_class: dict[int, list[dict[str, Any]]] = {}
    class_dispersion: dict[int, float] = {}
    class_slot_limits: dict[int, int] = {}
    if coherent_mode and adaptive_slots:
        # Use current-delta graph/text dispersion only.  Quantile ranks make
        # the rule dataset-scale independent and avoid tuning absolute
        # thresholds against downstream metrics.
        for cls in sorted(torch.unique(raw_labels).tolist()):
            local = torch.where(raw_labels == int(cls))[0]
            graph_center = _norm(raw_x[local].mean(0, keepdim=True)).squeeze(0)
            text_center = _norm(raw_text[local].mean(0, keepdim=True)).squeeze(0)
            graph_disp = float((1.0 - _norm(raw_x[local]) @ graph_center).mean())
            text_disp = float((1.0 - _norm(raw_text[local]) @ text_center).mean())
            class_dispersion[int(cls)] = 0.5 * graph_disp + 0.5 * text_disp
        values = sorted(class_dispersion.values())
        ranks = {value: (values.index(value) / max(1, len(values) - 1))
                 for value in values}
    else:
        ranks = {}
    audit: dict[str, Any] = {
        "candidate_selection": str(candidate_mode),
        "adaptive_slots": bool(adaptive_slots),
        "slot_rule": ("1/2/3 by within-delta graph-text dispersion rank"
                      if adaptive_slots and coherent_mode else "fixed"),
        "candidate_bank_by_class": {},
    }
    for cls, rows in rows_by_class.items():
        records = []
        if coherent_mode:
            # Use complete current-delta texts as coherent candidates.  The
            # selector still uses CaT X/text compatibility, but no longer
            # creates a candidate by concatenating unrelated source chunks.
            raw_class_rows = torch.where(raw_labels == int(cls))[0].tolist()
            for local_index in raw_class_rows:
                text = _truncate_text_boundary(
                    delta["texts"][int(local_index)], int(cfg.text_chars))
                embedding = text_embedding_cache[text]
                record = {
                    "mode": "coherent_raw_medoid",
                    "text": text,
                    "source_ids": [int(delta["ids"][int(local_index)])],
                    "source_local_indices": [int(local_index)],
                    "embedding": text_embedding_cache[text],
                }
                record["prompt_margin"] = _prompt_margin(
                    embedding, int(cls), prompt_emb,
                    float(cfg.prompt_margin))
                records.append(record)
        else:
            for row in rows:
                for mode, value in bank["rows"][row]["records"].items():
                    record = dict(value)
                    if mode == "cctp":
                        record["text"] = str(base_texts[row])
                        record["embedding"] = text_embedding_cache[
                            record["text"]]
                    elif record.get("embedding") is None:
                        record["embedding"] = text_embedding_cache[
                            str(record.get("text", ""))]
                    record["prompt_margin"] = _prompt_margin(
                        record["embedding"], int(cls), prompt_emb,
                        float(cfg.prompt_margin))
                    records.append(record)
        if adaptive_slots and coherent_mode:
            rank = ranks[class_dispersion[int(cls)]]
            if rank < 0.50:
                slot_limit = 1
            elif rank < 0.80:
                slot_limit = 2
            else:
                slot_limit = 3
            slot_limit = min(slot_limit, len(rows))
        else:
            slot_limit = int(max_slots)
        class_slot_limits[int(cls)] = int(max(1, slot_limit))
        effective_max_slots = (slot_limit if adaptive_slots and coherent_mode
                               else int(max_slots))
        selected = _select_diverse_slots(
            records, rows, x_cpu, raw_x, id_to_local,
            class_centers[int(cls)], prompt_emb, int(cls), effective_max_slots,
            float(cfg.prompt_margin), candidate_mode=candidate_mode)
        selected_records = [item["record"] for item in selected]
        candidates_by_class[int(cls)] = selected_records
        audit["candidate_bank_by_class"][str(cls)] = [{
            "mode": str(record.get("mode", "unknown")),
            "source_ids": [int(v) for v in record.get("source_ids", [])],
            "text_length": len(str(record.get("text", ""))),
            "prompt_margin": float(record.get("prompt_margin", 0.0)),
            "chunks": record.get("chunks", []),
        } for record in selected_records]
        audit.setdefault("class_slot_diagnostics", {})[str(cls)] = {
            "adaptive_slot_count": int(len(selected_records)),
            "class_dispersion": float(class_dispersion.get(int(cls), 0.0)),
            "class_dispersion_rank": float(ranks.get(
                class_dispersion.get(int(cls), 0.0), 0.0)),
        }
    audit["class_dispersion"] = {str(k): float(v)
                                  for k, v in class_dispersion.items()}
    audit["class_slot_limits"] = {str(k): int(v)
                                   for k, v in class_slot_limits.items()}
    return candidates_by_class, audit


def _route_class_rows(x_rows: torch.Tensor,
                      candidates: list[dict[str, Any]],
                      raw_x: torch.Tensor, id_to_local: dict[int, int],
                      prompt_emb: torch.Tensor | None, cls: int,
                      cfg: MM.MMConfig, steps: int,
                      temperature: float, load_weight: float,
                      anchor_weight: float,
                      balanced_materialization: bool = False
                      ) -> tuple[torch.Tensor, dict]:
    """Optimize differentiable row-to-slot assignment for one class."""
    x_rows = x_rows.detach().cpu().float()
    text_emb = _norm(torch.stack([
        _norm(value["embedding"].view(1, -1)).squeeze(0).cpu()
        for value in candidates]))
    cost = _routing_cost_matrix(
        x_rows, candidates, raw_x, id_to_local, prompt_emb, cls, cfg)
    temperature = max(float(temperature), 1e-4)
    initial = F.softmax(-cost / temperature, dim=1)
    logits = torch.nn.Parameter(torch.log(initial.clamp_min(1e-8)))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    target = torch.full((len(candidates),), 1.0 / len(candidates))
    trace = []
    for step in range(max(1, int(steps))):
        probabilities = F.softmax(logits, dim=1)
        expected_cost = (probabilities * cost).mean()
        usage = probabilities.mean(0).clamp_min(1e-8)
        load_loss = (usage * (usage / target).log()).sum()
        entropy_reg = (probabilities.clamp_min(1e-8)
                       * probabilities.clamp_min(1e-8).log()).mean()
        anchor = (probabilities * (probabilities.clamp_min(1e-8)
                                   / initial.clamp_min(1e-8)).log()).mean()
        loss = (expected_cost + float(load_weight) * load_loss
                + 0.01 * entropy_reg + float(anchor_weight) * anchor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, max(1, int(steps)) - 1):
            trace.append({
                "step": int(step),
                "loss": float(loss.detach()),
                "expected_cost": float(expected_cost.detach()),
                "load_loss": float(load_loss.detach()),
                "entropy": float((-entropy_reg).detach()),
            })
    probabilities = F.softmax(logits.detach(), dim=1)
    if balanced_materialization:
        selected, materialized_usage = _balanced_hard_slots(probabilities)
    else:
        selected = probabilities.argmax(dim=1)
        materialized_usage = torch.bincount(
            selected, minlength=len(candidates)).tolist()
    soft_text = F.normalize(probabilities @ text_emb, dim=1)
    hard_text = text_emb[selected]
    return probabilities, {
        "cost": cost.tolist(),
        "initial_probabilities": initial.tolist(),
        "probabilities": probabilities.tolist(),
        "selected_slots": selected.tolist(),
        "usage": probabilities.mean(0).tolist(),
        "materialized_usage": [int(value) for value in materialized_usage],
        "balanced_materialization": bool(balanced_materialization),
        "entropy": float((-(probabilities.clamp_min(1e-8)
                            * probabilities.clamp_min(1e-8).log()).sum(1)
                           ).mean()),
        "soft_hard_cosine": F.cosine_similarity(
            soft_text, hard_text, dim=1).tolist(),
        "trace": trace,
    }


def attach_x_conditioned_multi_payload(
        delta: dict, source_features: torch.Tensor, x0: torch.Tensor,
        text0: torch.Tensor, base_texts: list[str],
        base_source_ids: list[list[int]], labels: torch.Tensor,
        encoder, cfg: MM.MMConfig, mass: torch.Tensor | None,
        device: str, max_slots: int = 3, routing_steps: int = 30,
        routing_temperature: float = 0.15, load_weight: float = 0.03,
        anchor_weight: float = 0.10,
        candidate_mode: str = "standard",
        balanced_materialization: bool = False,
        cctp_anchor_swap: bool = False,
        adaptive_slots: bool = False) -> tuple[dict, dict]:
    """Attach per-row real text payloads through differentiable routing.

    CaT ``x0`` is frozen.  Only the assignment logits are optimized.  Rows of
    one class share a bounded candidate bank, while their final materialized
    texts may differ according to their X-conditioned probabilities.
    """
    labels_cpu = labels.detach().cpu().long()
    x_cpu = x0.detach().cpu().float()
    source_features = source_features.detach().cpu().float()
    raw_x = delta.get("x", source_features).detach().cpu().float()
    if raw_x.ndim != 2 or raw_x.shape[1] != x_cpu.shape[1]:
        raw_x = source_features
    id_to_local = {int(v): i for i, v in enumerate(
        delta["ids"].detach().cpu().tolist())}
    prompt_emb = delta.get("prompt_emb")
    if prompt_emb is not None:
        prompt_emb = prompt_emb.detach().cpu().float()
    candidates_by_class, candidate_audit = _collect_class_candidates(
        delta, source_features, x_cpu, text0, base_texts, base_source_ids,
        labels_cpu, encoder, cfg, max(1, int(max_slots)),
        candidate_mode=candidate_mode, adaptive_slots=adaptive_slots)

    rows_by_class: dict[int, list[int]] = defaultdict(list)
    for row, cls in enumerate(labels_cpu.tolist()):
        rows_by_class[int(cls)].append(int(row))
    selected_records: list[dict[str, Any] | None] = [None] * len(labels_cpu)
    routing_audit: dict[str, Any] = {}
    for cls, rows in rows_by_class.items():
        probabilities, row_audit = _route_class_rows(
            x_cpu[rows], candidates_by_class[int(cls)], raw_x, id_to_local,
            prompt_emb, int(cls), cfg, routing_steps, routing_temperature,
            load_weight, anchor_weight,
            balanced_materialization=balanced_materialization)
        slot_ids = row_audit["selected_slots"]
        for local, row in enumerate(rows):
            selected_records[row] = candidates_by_class[int(cls)][
                int(slot_ids[local])]
        if cctp_anchor_swap:
            global_cost = _routing_cost_matrix(
                x_cpu[rows], candidates_by_class[int(cls)], raw_x,
                id_to_local, prompt_emb, int(cls), cfg)
            chosen_cost = global_cost[
                torch.arange(len(rows)), torch.tensor(slot_ids)]
            anchor_embeddings = _norm(text0.detach().cpu().float()[rows])
            anchor_center = _norm(
                torch.stack([_norm(value["embedding"].view(1, -1)).squeeze(0)
                             for value in candidates_by_class[int(cls)]]).mean(
                                 0, keepdim=True)).squeeze(0)
            anchor_text_cost = 1.0 - anchor_embeddings @ anchor_center
            anchor_graph = torch.stack([
                _record_graph_prototype(
                    {"source_ids": base_source_ids[row]}, raw_x, id_to_local,
                    x_cpu[row]) for row in rows])
            anchor_graph_cost = 1.0 - _norm(x_cpu[rows]) @ _norm(anchor_graph).T
            anchor_margin = torch.tensor([
                _prompt_margin(value, int(cls), prompt_emb,
                               float(cfg.prompt_margin))
                for value in anchor_embeddings], dtype=torch.float32)
            anchor_cost = (anchor_graph_cost.diag()
                           + float(cfg.payload_assignment_text_weight)
                           * anchor_text_cost
                           + float(cfg.lambda_prompt_margin) * anchor_margin)
            swap_margin = 0.02
            swapped = []
            for local, row in enumerate(rows):
                if float(chosen_cost[local]) + swap_margin >= float(anchor_cost[local]):
                    selected_records[row] = {
                        "mode": "cctp_anchor",
                        "text": str(base_texts[row]),
                        "embedding": anchor_embeddings[local],
                        "source_ids": [int(v) for v in base_source_ids[row]],
                        "source_local_indices": [
                            int(id_to_local[v]) for v in base_source_ids[row]
                            if int(v) in id_to_local],
                    }
                    swapped.append(int(row))
            row_audit["cctp_anchor_fallback_rows"] = swapped
            row_audit["cctp_anchor_fallback_count"] = len(swapped)
        routing_audit[str(cls)] = {
            **row_audit,
            "rows": [int(v) for v in rows],
            "candidate_count": len(candidates_by_class[int(cls)]),
            "candidate_modes": [str(v.get("mode", "unknown"))
                                for v in candidates_by_class[int(cls)]],
            "candidate_source_ids": [
                [int(x) for x in v.get("source_ids", [])]
                for v in candidates_by_class[int(cls)]],
        }
    if any(value is None for value in selected_records):
        raise AssertionError("unassigned synthetic row")
    selected_records = [value for value in selected_records if value is not None]
    text_embeddings = _norm(torch.stack([
        _norm(value["embedding"].view(1, -1)).squeeze(0)
        for value in selected_records]))
    block = {
        "x": x_cpu,
        "text_emb": text_embeddings,
        "texts": [str(value["text"]) for value in selected_records],
        "mass": (mass.detach().cpu() if mass is not None else
                 torch.ones(len(selected_records))),
        "depth": torch.zeros(len(selected_records), dtype=torch.long),
        "stage": torch.full((len(selected_records),), int(delta["stage"]),
                             dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    unique_by_class = {
        str(cls): len({str(selected_records[row]["text"])
                       for row in rows})
        for cls, rows in rows_by_class.items()}
    audit = {
        "constructor": "CaT frozen X + X-conditioned differentiable payload routing",
        "objective": "row-slot cost + load balance + CCTP routing anchor",
        "candidate_selection": str(candidate_mode),
        "adaptive_slots": bool(adaptive_slots),
        "text_truncation": "sentence-boundary, then word-boundary, then hard cap",
        "class_shared": False,
        "max_slots": int(max_slots),
        "routing_steps": int(routing_steps),
        "routing_temperature": float(routing_temperature),
        "load_weight": float(load_weight),
        "anchor_weight": float(anchor_weight),
        "balanced_materialization": bool(balanced_materialization),
        "cctp_anchor_swap": bool(cctp_anchor_swap),
        "candidate_generation": "current-delta real chunks; bounded support pool; no text generation",
        "candidate_bank": candidate_audit,
        "routing_by_class": routing_audit,
        "unique_payloads_by_class": unique_by_class,
        "text_source_ids": [value.get("source_ids", [])
                            for value in selected_records],
        "text_chunk_provenance": [value.get("chunks", [])
                                  for value in selected_records],
        "text_lengths": [len(str(value["text"])) for value in selected_records],
        "current_delta_only": True,
        "historical_raw_reads": 0,
        "old_memory_updated": False,
        "topology_updated": False,
        "x_updated": False,
        "text_generated": False,
        "hard_materialization": True,
    }
    return block, audit

