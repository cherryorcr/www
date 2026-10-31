"""Read-only attribution of text gradients and hard payload quantization."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from unified_q_payload_projection import q_text_medoid


@torch.no_grad()
def payload_metrics(q, raw_text, raw_labels, labels, chosen=None):
    text = F.normalize(raw_text, dim=1)
    mean = q.T @ text / q.sum(0).clamp_min(1e-8)[:, None]
    soft = F.normalize(mean, dim=1)
    selected = q.argmax(0) if chosen is None else chosen
    medoid = q_text_medoid(q, text, raw_labels, labels)
    hard, alternative = text[selected], text[medoid]
    slot_distortion = 1 - (mean * hard).sum(1)
    medoid_distortion = 1 - (mean * alternative).sum(1)
    per_class = []
    for cls in torch.unique(labels, sorted=True):
        raw = torch.where(raw_labels == cls)[0]
        slots = torch.where(labels == cls)[0]
        native_coverage, alternative_coverage, soft_coverage = [], [], []
        for start in range(0, len(raw), 1024):
            emb = text[raw[start:start + 1024]]
            native_coverage.append((emb @ hard[slots].T).max(1).values)
            alternative_coverage.append((emb @ alternative[slots].T).max(1).values)
            soft_coverage.append((emb @ soft[slots].T).max(1).values)
        per_class.append(dict(class_id=int(cls), raw_rows=len(raw), slots=len(slots),
            native_unique_sources=int(torch.unique(selected[slots]).numel()),
            medoid_unique_sources=int(torch.unique(medoid[slots]).numel()),
            native_distortion=float(slot_distortion[slots].mean()),
            medoid_distortion=float(medoid_distortion[slots].mean()),
            hard_soft_cosine=float((hard[slots] * soft[slots]).sum(1).mean()),
            native_coverage_cosine=float(torch.cat(native_coverage).mean()),
            medoid_coverage_cosine=float(torch.cat(alternative_coverage).mean()),
            soft_coverage_cosine=float(torch.cat(soft_coverage).mean())))
    keys = ("native_distortion", "medoid_distortion", "hard_soft_cosine",
            "native_coverage_cosine", "medoid_coverage_cosine", "soft_coverage_cosine")
    return dict(per_class=per_class, macro={k: sum(r[k] for r in per_class) / len(per_class) for k in keys},
        changed_to_medoid_fraction=float((selected != medoid).float().mean()),
        unique_source_fraction=len(torch.unique(selected)) / len(labels),
        medoid_unique_source_fraction=len(torch.unique(medoid)) / len(labels))


def _gradient(value, parameter):
    if not value.requires_grad:
        return torch.zeros_like(parameter)
    result = torch.autograd.grad(value, parameter, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(parameter) if result is None else result.detach()


def _cosine(left, right):
    denominator = left.norm() * right.norm()
    return float((left * right).sum() / denominator) if float(denominator) > 1e-20 else None


class TextPathDiagnostics:
    def __init__(self, steps):
        self.sample_steps = {0, max(0, steps // 2 - 1), max(0, steps - 1)}
        self.snapshots, self.switches = [], []
        self.previous = None

    def __call__(self, state):
        q, cfg, logits = state["q"], state["config"], state["q_logits"]
        selected = q.detach().argmax(0)
        self.switches.append(dict(step=state["step"], changed_fraction=(
            float((selected != self.previous).float().mean()) if self.previous is not None else None)))
        self.previous = selected.clone()
        if state["step"] not in self.sample_steps:
            return
        dm = _gradient(cfg.lambda_distribution * state["distribution"], logits)
        gm = _gradient(cfg.lambda_gradient * state["gm"], logits)
        text_new = state["text_new"]
        if text_new.requires_grad and state["gm"].requires_grad:
            upstream = torch.autograd.grad(cfg.lambda_gradient * state["gm"], text_new,
                retain_graph=True, allow_unused=True)[0]
            direct = (torch.autograd.grad(text_new, logits, upstream.detach(),
                retain_graph=True, allow_unused=True)[0] if upstream is not None else None)
            direct = torch.zeros_like(logits) if direct is None else direct.detach()
        else:
            direct = torch.zeros_like(logits)
        gradients = dict(dm_q_norm=float(dm.norm()), gm_q_norm=float(gm.norm()),
            direct_text_q_norm=float(direct.norm()),
            text_to_dm_norm_ratio=float(direct.norm() / dm.norm().clamp_min(1e-20)),
            gm_to_dm_norm_ratio=float(gm.norm() / dm.norm().clamp_min(1e-20)),
            text_dm_cosine=_cosine(direct, dm), gm_dm_cosine=_cosine(gm, dm),
            text_other_gm_cosine=_cosine(direct, gm - direct))
        del dm, gm, direct
        gradients["propagation_q_norm"] = float(_gradient(cfg.lambda_propagation * state["propagation"], logits).norm())
        self.snapshots.append(dict(step=state["step"], gradients=gradients,
            losses={k: float(state[k].detach()) for k in ("gm", "distribution", "propagation", "loss")},
            payload=payload_metrics(q.detach(), state["raw_text"], state["raw_labels"], state["labels"])))

    def final(self, state):
        delta, q = state["delta"], state["q"]
        labels = state["block"]["labels"].to(q.device)
        text = delta["text_emb"].to(q.device)
        raw_labels = delta["y"].to(q.device)
        return dict(snapshots=self.snapshots, switches=self.switches,
            final=payload_metrics(q, text, raw_labels, labels),
            initial=payload_metrics(state["q0"], text, raw_labels, labels),
            interpretation="text gradient is the direct text_new-to-Q path of total outer GM; not a separate additive text loss")
