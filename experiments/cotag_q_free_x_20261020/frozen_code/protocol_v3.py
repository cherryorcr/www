"""Shared explicit protocol for reproducible TAG condensation reruns."""
from __future__ import annotations

from dataclasses import asdict, dataclass

PROTOCOL_NAME = "tagsam-unified-v3-2026-09-13"
ENUMERATION_PROTOCOL = "p0-canonical-class-then-source-id-v1"
STUDENT_SEEDS = (100, 101, 102, 103, 104, 105)
TEACHER_SEED = 9000
REFERENCE_SEED = 777
METHOD_SEED = 9001
BASELINE_RANDOM_SEED = 11
BASELINE_RANDOM_INIT_SEED = 123
JITTER_SEEDS = (9001, 9002, 9003, 9004, 9005)
ASM_STEPS = 100
ASM_INNER_STEPS = 10
ASM_BATCH = 16
ASM_OUTER_LR = 1e-2
TEACHER_STEPS = 4096
RAW_STEPS_CAP = 8192
FUSE_G = 5
FUSE_CHARS = 320
DEGREE = 1.0
NEW_ROWS_PER_CLASS = 15
MIN_DYNAMIC_ROWS_PER_CLASS = 8
MAX_REFRESH_RATIO = 0.20
DEV_MOD = 1
TEST_MOD = 2
REPORT_DEV_MOD = 3


@dataclass(frozen=True)
class StudentConfig:
    steps_cap: int = 2048
    steps_per_root: int = 16
    batch: int = 16
    lr: float = 1e-3

    def steps(self, roots: int, *, raw: bool = False,
              raw_steps_cap: int | None = None) -> int:
        if raw and raw_steps_cap is not None:
            return int(raw_steps_cap)
        return max(50, min(self.steps_cap, self.steps_per_root * int(roots)))


STUDENT = StudentConfig()


def student_steps(roots: int, *, override: int | None = None) -> int:
    """Return the canonical number of optimizer steps for an artifact."""
    if override is not None:
        return int(override)
    return STUDENT.steps(int(roots))


def raw_steps(*, override: int | None = None) -> int:
    """Return the canonical upper-reference training budget."""
    return RAW_STEPS_CAP if override is None else int(override)


def append_quota(total_budget: int, old_rows: int, new_class_count: int) -> tuple[int, int]:
    """Return (rows_to_keep, rows_for_new_delta) for the append rule."""
    total_budget = int(total_budget)
    old_rows = int(old_rows)
    new_class_count = int(new_class_count)
    if total_budget < 0 or old_rows < 0 or new_class_count < 0:
        raise ValueError("budget, old_rows, and new_class_count must be non-negative")
    new_rows = max(total_budget - old_rows,
                   NEW_ROWS_PER_CLASS * new_class_count)
    keep_rows = int(total_budget) - new_rows
    if keep_rows < 0:
        raise ValueError("new-class floor exceeds the total memory budget")
    return keep_rows, new_rows


def growing_budget(cumulative_roots: int, *, rate: float = 0.05,
                   mode: str = "fixed_rate", total_budget: int | None = None,
                   previous_budget: int = 0, min_increment: int = 0) -> int:
    """Resolve an explicit continual-memory budget policy.

    ``fixed_rate`` is the historical v3 rule: memory grows with cumulative
    source rows. ``fixed_total`` is retained as a stress-test control, while
    ``hybrid`` guarantees a minimum increment and otherwise follows the rate.
    """
    cumulative_roots = int(cumulative_roots)
    previous_budget = int(previous_budget)
    if cumulative_roots < 0 or previous_budget < 0:
        raise ValueError("cumulative_roots and previous_budget must be non-negative")
    if mode == "fixed_total":
        if total_budget is None:
            raise ValueError("fixed_total requires total_budget")
        budget = int(total_budget)
    elif mode == "fixed_rate":
        budget = int(cumulative_roots * float(rate))
    elif mode == "hybrid":
        rate_budget = int(cumulative_roots * float(rate))
        budget = max(rate_budget, previous_budget + int(min_increment))
    else:
        raise ValueError(f"unknown budget mode: {mode}")
    if budget < 0:
        raise ValueError("resolved budget must be non-negative")
    return budget


def dynamic_append_quota(total_budget: int, old_rows: int,
                         difficulty: float,
                         new_class_count: int,
                         min_rows_per_class: int = MIN_DYNAMIC_ROWS_PER_CLASS,
                         max_refresh_ratio: float = MAX_REFRESH_RATIO) -> tuple[int, int]:
    """Return a deterministic difficulty-aware append quota.

    The total memory budget is still hard.  Difficulty only decides how much
    of that budget is assigned to the incoming delta; it cannot exceed the
    configured refresh envelope or remove the per-class floor.
    """
    total_budget = int(total_budget)
    old_rows = int(old_rows)
    new_class_count = int(new_class_count)
    if total_budget < 0 or old_rows < 0 or new_class_count <= 0:
        raise ValueError("invalid dynamic append quota arguments")
    difficulty = max(0.0, min(1.0, float(difficulty)))
    base_new = max(0, total_budget - old_rows)
    floor_new = int(min_rows_per_class) * new_class_count
    refresh_cap = int(round(total_budget * max(0.0, min(1.0,
                                                        float(max_refresh_ratio)))))
    # A difficult delta may evict a bounded fraction of the retained history.
    new_rows = max(base_new, floor_new,
                   min(total_budget, base_new + int(round(refresh_cap * difficulty))))
    keep_rows = total_budget - new_rows
    if keep_rows < 0:
        raise ValueError("dynamic per-class floor exceeds total budget")
    return keep_rows, new_rows


def protocol_seeds(seeds=None) -> list[int]:
    """Normalize an optional CLI seed list without changing the default."""
    return list(STUDENT_SEEDS if seeds is None else seeds)


def canonicalize_class_groups(groups) -> list[list[int]]:
    """Sort classes within each stage while preserving stage order."""
    normalized = []
    seen = set()
    for group in groups:
        values = [int(v) for v in group]
        if not values or len(values) != len(set(values)):
            raise ValueError("each stage must contain unique non-empty classes")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"classes appear in multiple stages: {sorted(overlap)}")
        normalized.append(sorted(values))
        seen.update(values)
    if not normalized:
        raise ValueError("at least one stage is required")
    return normalized


def split_masks(ids):
    """Return the canonical (dev, test, pool) masks for original node ids."""
    dev = (ids % 10) == DEV_MOD
    test = (ids % 10) == TEST_MOD
    return dev, test, ~dev & ~test


def split_masks_v4(ids, report_dev_mod: int | None = REPORT_DEV_MOD):
    """Return disjoint selector-dev, report-dev, test, and construction masks."""
    selector_dev = (ids % 10) == DEV_MOD
    test = (ids % 10) == TEST_MOD
    report_dev = ((ids % 10) == int(report_dev_mod)
                  if report_dev_mod is not None
                  else selector_dev & False)
    pool = ~selector_dev & ~report_dev & ~test
    return selector_dev, report_dev, test, pool
def adaptive_rate_budget(cumulative_roots: int, previous_budget: int,
                         utility: float, *, base_rate: float = 0.05,
                         strength: float = 0.75, min_rate: float = 0.01,
                         max_rate: float = 0.10,
                         min_increment: int = 0) -> tuple[int, float]:
    """Resolve a source-only information-adaptive budget and its rate."""
    utility = max(0.0, min(1.0, float(utility)))
    rate = float(base_rate) * (1.0 + float(strength) * (utility - 0.5))
    rate = max(float(min_rate), min(float(max_rate), rate))
    budget = max(int(cumulative_roots * rate),
                 int(previous_budget) + int(min_increment))
    return int(budget), float(rate)


def replacement_quota(total_budget: int, old_rows: int,
                      delta_roots: int, cumulative_roots: int,
                      new_class_count: int) -> tuple[int, int]:
    """Allocate a fixed-size memory between retained old and new rows.

    This is the fixed-total continual control: new data still enters the
    memory, while a proportional number of old rows is evicted.
    """
    total_budget = int(total_budget)
    old_rows = int(old_rows)
    delta_roots = int(delta_roots)
    cumulative_roots = max(1, int(cumulative_roots))
    new_class_count = int(new_class_count)
    if min(total_budget, old_rows, delta_roots, new_class_count) < 0:
        raise ValueError("replacement quota arguments must be non-negative")
    proportional = int(round(total_budget * delta_roots / cumulative_roots))
    new_rows = min(total_budget, max(new_class_count, proportional))
    new_rows = min(new_rows, delta_roots)
    keep_rows = min(old_rows, total_budget - new_rows)
    new_rows = total_budget - keep_rows
    return int(keep_rows), int(new_rows)


def config_dict(**overrides):
    out = {
        "protocol": PROTOCOL_NAME,
        "student": asdict(STUDENT),
        "student_seeds": list(STUDENT_SEEDS),
        "teacher_seed": TEACHER_SEED,
        "reference_seed": REFERENCE_SEED,
        "method_seed": METHOD_SEED,
        "teacher_steps": TEACHER_STEPS,
        "raw_steps_cap": RAW_STEPS_CAP,
        "asm_steps": ASM_STEPS,
        "asm_inner_steps": ASM_INNER_STEPS,
        "asm_batch": ASM_BATCH,
        "asm_outer_lr": ASM_OUTER_LR,
        "fuse_g": FUSE_G,
        "fuse_chars": FUSE_CHARS,
        "degree": DEGREE,
        "new_rows_per_class": NEW_ROWS_PER_CLASS,
        "min_dynamic_rows_per_class": MIN_DYNAMIC_ROWS_PER_CLASS,
        "max_refresh_ratio": MAX_REFRESH_RATIO,
        "dev_mod": DEV_MOD,
        "report_dev_mod": REPORT_DEV_MOD,
        "test_mod": TEST_MOD,
        "pool_rule": "original_id % 10 not in {dev_mod, report_dev_mod, test_mod}",
        "test_source_sealed": True,
        "selector_dev_source_sealed": True,
        "report_dev_source_sealed": True,
        "student_steps_rule": "max(50, min(2048, 16 * roots))",
        "append_rule": "new_rows=max(budget-old_rows, 15*new_class_count); prune_old_by_mass",
        "dynamic_append_rule": "difficulty-aware delta allocation under fixed total budget",
    }
    out.update(overrides)
    return out
