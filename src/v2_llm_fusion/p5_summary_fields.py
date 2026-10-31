"""Reporting-only definitions for the frozen P5 evaluator and cost fields."""

METRIC_DEFINITIONS = {
    "prompt_scope": "all classes, including unseen classes at intermediate stages",
    "cum_dev_f1_mean": (
        "Macro-F1 on pooled seen report-dev samples; labels present in targets; "
        "mean over student seeds"
    ),
    "old_mean_f1_mean": (
        "Task-slice Macro-F1: score each previous task on its own target samples, "
        "then average tasks equally and average student seeds; "
        "cross-task false positives are excluded"
    ),
    "current_f1_mean": (
        "Task-slice Macro-F1 on current-task target samples, averaged over "
        "student seeds; false positives from other tasks are excluded"
    ),
    "historical_comparison": (
        "Historical five-level old/new scores use class-weighted F1 on the "
        "pooled seen support and cannot be compared to task-slice old/new"
    ),
}


def optional_seconds(record, key):
    value = record.get(key)
    return None if value is None else float(value)


def complete_sum(values):
    """A missing cost makes the total unknown, not artificially smaller."""
    values = list(values)
    return None if not values or any(v is None for v in values) else sum(values)
