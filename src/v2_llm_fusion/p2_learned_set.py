"""Small source-only DeepSets distillation with a source-holdout checkpoint."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DeepSetTarget(nn.Module):
    def __init__(self, dim=384, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.rho = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)
        nn.init.zeros_(self.rho[-1].weight)
        nn.init.zeros_(self.rho[-1].bias)

    def forward(self, bags, mask):
        z = bags + self.phi(bags)
        z = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(z + self.rho(z), dim=-1)


def fit_source_only(bags, masks, targets, seed=700, steps=300, device="cpu"):
    """Neither evaluation labels nor previous source examples enter fitting."""
    if len(bags) < 3:
        raise ValueError("at least three current-delta clusters are needed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    order = torch.randperm(len(bags))
    holdout = order[:max(1, len(bags) // 5)]
    train = order[len(holdout):]
    model = DeepSetTarget(dim=bags.shape[-1]).to(device)
    x, mask, y = bags.to(device), masks.to(device), targets.to(device)
    train, holdout = train.to(device), holdout.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    with torch.no_grad():
        initial = float((1 - (model(x[holdout], mask[holdout]) * y[holdout]).sum(-1)).mean())
    best_loss, best_step, best_state = initial, 0, copy.deepcopy(model.state_dict())
    curve = [{"step": 0, "source_holdout_cosine_loss": initial}]
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(x[train], mask[train])
        loss = (1 - (output * y[train]).sum(-1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                value = float((1 - (model(x[holdout], mask[holdout]) * y[holdout]).sum(-1)).mean())
            curve.append({"step": step, "source_holdout_cosine_loss": value})
            if value < best_loss:
                best_loss, best_step = value, step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        outputs = model(x, mask).cpu()
    return outputs, {key: value.cpu() for key, value in model.state_dict().items()}, {
        "construction_seed": int(seed), "steps": int(steps), "selected_step": best_step,
        "train_cluster_ids": train.cpu().tolist(), "holdout_cluster_ids": holdout.cpu().tolist(),
        "initial_holdout_loss": initial, "best_holdout_loss": best_loss,
        "parameters": sum(p.numel() for p in model.parameters()), "curve": curve,
        "fit_source": "current_delta_only", "fit_labels_used": False,
        "fit_target": "MiniLM_reencoded_same_bounded_snippets",
        "checkpoint_selection": "source_cluster_holdout_cosine_loss",
    }
