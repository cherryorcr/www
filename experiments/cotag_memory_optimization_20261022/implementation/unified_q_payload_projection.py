"""A real-text hard projection induced by the learned assignment Q."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def q_text_medoid(q, raw_text, raw_labels, labels, chunk=64):
    """Minimize Q-weighted cosine distortion over same-class real text rows."""
    text = F.normalize(raw_text, dim=1)
    means = F.normalize(q.T @ text / q.sum(0).clamp_min(1e-8)[:, None], dim=1)
    selected = torch.empty(len(labels), dtype=torch.long, device=q.device)
    for cls in torch.unique(labels, sorted=True):
        raw = torch.where(raw_labels == cls)[0]
        slots = torch.where(labels == cls)[0]
        if not len(raw):
            raise ValueError("a slot class has no current-delta text candidates")
        for start in range(0, len(slots), chunk):
            local = slots[start:start + chunk]
            selected[local] = raw[(means[local] @ text[raw].T).argmax(1)]
    return selected


@torch.no_grad()
def q_text_coverage(q, raw_text, raw_labels, labels,
                    coverage_weight=0.35, candidate_cap=32):
    """Select class-legal real texts with Q-conditioned semantic coverage.

    Q-weighted slot centers provide compatibility scores.  A deterministic
    greedy without-replacement projection rewards candidates that cover
    other Q-weighted slot centers in the same class.  The candidate shortlist
    bounds the cost when a class contains many raw texts.
    """
    if q.ndim != 2 or raw_text.ndim != 2:
        raise ValueError("q and raw_text must be matrices")
    if len(q) != len(raw_text) or len(raw_labels) != len(raw_text):
        raise ValueError("Q/text/labels must have aligned rows")
    text = F.normalize(raw_text.float(), dim=1)
    selected = torch.empty(len(labels), dtype=torch.long, device=q.device)
    weight = float(coverage_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("coverage_weight must lie in [0, 1]")
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be positive")

    for cls in torch.unique(labels, sorted=True):
        raw = torch.where(raw_labels.to(q.device) == cls)[0]
        slots = torch.where(labels.to(q.device) == cls)[0]
        if not len(raw) or not len(slots):
            raise ValueError("every slot class needs current-delta text rows")
        block_q = q.index_select(0, raw).index_select(1, slots)
        mass = block_q.sum(0).clamp_min(1e-8)
        centers = F.normalize(block_q.T @ text.index_select(0, raw) /
                              mass[:, None], dim=1)
        compatibility = centers @ text.index_select(0, raw).T
        entropy = -(block_q.clamp_min(1e-12) *
                    block_q.clamp_min(1e-12).log()).sum(0)
        slot_order = torch.argsort(entropy, stable=True).tolist()
        available = torch.ones(len(raw), dtype=torch.bool, device=q.device)
        best_cover = torch.full((len(slots),), -1.0, device=q.device)
        shortlist = compatibility.topk(min(int(candidate_cap), len(raw)), dim=1).indices
        for local_slot in slot_order:
            candidates = shortlist[local_slot]
            candidates = candidates[available[candidates]]
            if not len(candidates):
                candidates = torch.where(available)[0]
            if not len(candidates):
                candidates = shortlist[local_slot]
            candidates = candidates.sort().values
            gain = (compatibility[:, candidates] - best_cover[:, None]).clamp_min(0).mean(0)
            scores = (1.0 - weight) * compatibility[local_slot, candidates] + weight * gain
            choice = int(candidates[scores.argmax()])
            best_cover = torch.maximum(best_cover, compatibility[:, choice])
            available[choice] = False
            selected[slots[local_slot]] = raw[choice]
    return selected
