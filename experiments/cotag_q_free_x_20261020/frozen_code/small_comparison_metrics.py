"""Global-support metrics for class-incremental zero-shot evaluation."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def score_predictions(target, pred, stage_classes, stage):
    target, pred = np.asarray(target), np.asarray(pred)
    seen = [int(c) for group in stage_classes[:stage + 1] for c in group]
    active = np.isin(target, seen)
    target, pred = target[active], pred[active]
    if not len(target):
        raise ValueError("empty evaluation split")
    per_class = f1_score(target, pred, labels=seen, average=None,
                         zero_division=0) * 100.0
    class_scores = dict(zip(seen, per_class.tolist()))
    per_task = [float(np.mean([class_scores[int(c)] for c in group]))
                for group in stage_classes[:stage + 1]]
    old = [class_scores[int(c)] for group in stage_classes[:stage] for c in group]
    return {"cumulative_f1": float(np.mean(per_class)),
            "accuracy": float(np.mean(target == pred) * 100),
            "old_f1": float(np.mean(old)) if old else None,
            "new_f1": per_task[-1], "per_task_f1": per_task,
            "per_class_f1": class_scores, "n": len(target)}


def score_logits(logits, target, stage_classes, stage):
    allowed = sorted({int(c) for g in stage_classes[:stage + 1] for c in g})
    logits = torch.as_tensor(logits).detach().cpu()
    seen_pred = torch.tensor(allowed)[logits[:, allowed].argmax(1)]
    return {"seen": score_predictions(target, seen_pred.numpy(), stage_classes, stage),
            "all": score_predictions(target, logits.argmax(1).numpy(), stage_classes, stage)}


@torch.no_grad()
def evaluate(model, data, stage, split="report_dev_mask"):
    model.eval()
    mask = data[split]
    logits = model.logits(data["x_device"], data["real_adj"],
                          data["prompt_emb"].to(data["x_device"].device))
    return score_logits(logits.cpu()[mask], data["y"][mask].numpy(),
                        data["stage_classes"], stage)
