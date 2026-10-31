"""Confusion-based metrics, including predictions outside the seen classes."""
from __future__ import annotations

import numpy as np


def metrics(y, pred, seen, old, new, class_count):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if y.shape != pred.shape or not len(y):
        raise ValueError("nonempty, aligned predictions are required")
    if not set(y).issubset(seen):
        raise ValueError("evaluation cohort includes unseen truth labels")
    if min(y.min(), pred.min()) < 0 or max(y.max(), pred.max()) >= class_count:
        raise ValueError("label outside the complete prompt taxonomy")
    cm = np.bincount(y * class_count + pred, minlength=class_count**2).reshape(
        class_count, class_count)
    tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    divide = lambda num, den: np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
    precision = 100 * divide(tp, predicted)
    recall = 100 * divide(tp, support)
    f1 = 100 * divide(2 * tp, support + predicted)
    classes = [dict(class_id=int(c), support=int(support[c]), predicted=int(predicted[c]),
                    correct=int(tp[c]), precision=float(precision[c]), recall=float(recall[c]),
                    f1=float(f1[c])) for c in seen]
    return dict(final_f1=float(f1[seen].mean()), accuracy=float(100 * tp.sum() / len(y)),
                old_f1=float(f1[old].mean()) if old else None,
                new_f1=float(f1[new].mean()), per_class=classes, confusion=cm.tolist())


def frequency_groups(counts):
    if len(counts) % 3:
        raise ValueError("equal-size frequency tertiles require a multiple of three classes")
    ordered = sorted(range(len(counts)), key=lambda c: (int(counts[c]), c))
    size = len(counts) // 3
    return {c: ("low", "middle", "high")[rank // size] for rank, c in enumerate(ordered)}
