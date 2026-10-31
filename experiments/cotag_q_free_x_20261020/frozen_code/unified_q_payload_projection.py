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
