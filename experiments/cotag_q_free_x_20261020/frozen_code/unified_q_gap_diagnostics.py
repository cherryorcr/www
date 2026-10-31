"""Read-only representation-feasibility diagnostics on the legal delta."""
import torch
import torch.nn.functional as F

import unified_q_multimodal as U


def class_geometry(raw_x, message, raw_labels, synth_x, labels, radius):
    records = []
    for cls in torch.unique(labels).tolist():
        raw = raw_x[raw_labels == cls].double()
        msg = message[raw_labels == cls].double()
        syn = synth_x[labels == cls].double()
        mu, target, actual = raw.mean(0), msg.mean(0), syn.mean(0)
        bound = float(radius) * (raw.norm(dim=1).mean() + 1e-3)
        distance = (target - mu).norm()
        records.append(dict(class_id=int(cls), raw_rows=len(raw), slots=len(syn),
            mean_distance=float(distance), residual_mean_bound=float(bound),
            distance_over_bound=float(distance / bound.clamp_min(1e-12)),
            raw_message_cosine=float(F.cosine_similarity(mu, target, dim=0)),
            raw_mean_norm=float(mu.norm()), message_mean_norm=float(target.norm()),
            synthetic_message_cosine=float(F.cosine_similarity(actual, target, dim=0)),
            synthetic_mean_error=float((actual - target).norm()),
            raw_within_variance=float((raw - mu).square().sum(1).mean()),
            message_within_variance=float((msg - target).square().sum(1).mean()),
            synthetic_within_variance=float((syn - actual).square().sum(1).mean())))
    # Descriptive nearest-centroid geometry on construction data, NOT a new
    # classifier fitted on report/test nodes or an independent zero-shot score.
    classes = torch.unique(labels)
    centers = torch.stack([synth_x[labels == c].mean(0) for c in classes])
    predicted = classes[(F.normalize(message, dim=1) @ F.normalize(centers, dim=1).T).argmax(1)]
    per_class = [float((predicted[raw_labels == c] == c).float().mean()) for c in classes]
    return dict(classes=records, construction_macro_centroid_accuracy=sum(per_class) / len(per_class),
                centroid_probe_scope="current construction delta only; descriptive, not report-dev")


def holdout_dm(message, raw_labels, synth_x, labels, cfg, stage):
    target = U._distribution_target(message, raw_labels, cfg,
        int(cfg.seed) + 1009 * int(stage) + 1999963)
    return float(U._distribution_loss(synth_x, labels, target))


def representation_diagnostic(state):
    q, delta, cfg = state["q"], state["delta"], state["config"]
    device = q.device
    raw = delta["x"].float().to(device)
    message = state["target_message"].float().to(device)
    labels = state["block"]["labels"].to(device)
    raw_labels = delta["y"].to(device)
    synth = state["block"]["x"].to(device)
    source = message if getattr(cfg, "feature_pooling", "raw") == "message" else raw
    pooled = q.T @ source / q.sum(0).clamp_min(1e-6)[:, None]
    residual = synth - pooled
    limit = cfg.trust_radius * pooled.norm(dim=1).clamp_min(1e-3)
    fractions = residual.norm(dim=1) / limit.clamp_min(1e-12)
    result = class_geometry(raw, message, raw_labels, synth, labels, cfg.trust_radius)
    result.update(residual_cap_saturation=float((fractions >= .99).float().mean()),
        residual_cap_fraction_mean=float(fractions.mean()), residual_cap_fraction_max=float(fractions.max()),
        holdout_dm=holdout_dm(message, raw_labels, synth, labels, cfg, delta["stage"]),
        diagnostic_holdout_used_for_training=False, feature_pooling=getattr(cfg, "feature_pooling", "raw"))
    for record in result["classes"]:
        c = record["class_id"]
        record["residual_cap_saturation"] = float((fractions[labels == c] >= .99).float().mean())
        record["pool_class_mean_error"] = float((pooled[labels == c].mean(0) - source[raw_labels == c].mean(0)).norm())
    return result


def edgefree_slice(memory, start, end):
    """Recover an immutable stage block from a verified frozen-prefix archive."""
    if memory["edge_index"].numel():
        raise ValueError("stage slicing supports edge-free archives only")
    block = {k: memory[k][start:end] for k in ("x", "text_emb", "labels", "mass", "depth", "stage", "texts", "text_source_ids")}
    block["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    if "support_ids" in memory:
        block["support_ids"] = memory["support_ids"][start:end]
    return block


def common_nearest_payloads(delta, features, base, encoder, ks=(1, 5)):
    """Same policy for both constructors, with batched float64 cosine queries.

    Native adapters use float32 row queries. Nearly tied sources can differ,
    so this control is never silently identified with native GECC payloads.
    """
    selected = [None] * len(base["x"])
    for cls in torch.unique(base["labels"]).tolist():
        ids = torch.where(delta["y"] == cls)[0]
        slots = torch.where(base["labels"] == cls)[0]
        graph = F.normalize(features[ids].cpu().double(), dim=1)
        queries = F.normalize(base["x"][slots].cpu().double(), dim=1)
        for start in range(0, len(slots), 128):
            score = queries[start:start + 128] @ graph.T
            order = torch.argsort(score, dim=1, descending=True, stable=True)[:, :min(max(ks), len(ids))]
            for slot, local in zip(slots[start:start + 128].tolist(), order):
                selected[slot] = ids[local].tolist()
    result = {}
    for k in ks:
        texts, sources = [], []
        for candidate_ids in selected:
            chosen = candidate_ids[:k]
            per = max(1, (320 - max(0, len(chosen) - 1)) // len(chosen))
            texts.append(" ".join(str(delta["texts"][i])[:per] for i in chosen)[:320])
            sources.append(delta["ids"][chosen].tolist())
        result[k] = dict(base, texts=texts, text_source_ids=sources,
                         text_emb=torch.as_tensor(encoder.encode(texts)).float().cpu())
    return result
