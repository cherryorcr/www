"""Unit tests for v7-A gradient-compatible append."""
from copy import deepcopy

import pytest
import torch

import v7_gradient_append as V


def toy_memory(rows=20, dim=8):
    gen = torch.Generator().manual_seed(17)
    return {"x": torch.randn(rows, dim, generator=gen),
            "text_emb": torch.randn(rows, 384, generator=gen),
            "texts": [f"t{i}" for i in range(rows)],
            "mass": torch.ones(rows), "depth": torch.zeros(rows, dtype=torch.long),
            "stage": torch.zeros(rows, dtype=torch.long),
            "edge_index": torch.stack((torch.arange(rows - 1), torch.arange(1, rows)))}


def test_integer_budget_is_exact_and_deterministic():
    weights = torch.tensor([1., 2., 3.])
    assert V._integer_budget(weights, 11, 1) == V._integer_budget(weights, 11, 1)
    assert sum(V._integer_budget(weights, 11, 1)) == 11
    assert V._integer_budget(weights, 11, 1) == [2, 4, 5]


def test_budget_redistributes_when_a_subcluster_is_full():
    assert V._integer_budget(torch.tensor([100., 1., 1.]), 8, 1, [1, 5, 5]) == [1, 4, 3]
    with pytest.raises(ValueError, match="capacity"):
        V._integer_budget(torch.ones(2), 5, capacities=[2, 2])


def test_gradient_score_prefers_coverage_when_dynamics_match():
    base = {"observations": [{"gradient": torch.ones(4)} for _ in range(3)]}
    good = {"observations": [{"gradient": torch.ones(4)} for _ in range(3)]}
    score = V.gradient_score(good, base, 2., 1.)
    assert score["coverage_gain"] > 0 and score["dynamic_distance_mean"] == pytest.approx(0.)


def test_gradient_score_penalizes_direction_change():
    base = {"observations": [{"gradient": torch.ones(4)} for _ in range(3)]}
    bad = {"observations": [{"gradient": -torch.ones(4)} for _ in range(3)]}
    score = V.gradient_score(bad, base, 2., 1.9)
    assert score["dynamic_distance_mean"] == pytest.approx(2.)
    assert score["score"] < 0


def test_selection_requires_group_agreement_and_positive_score():
    scores = {name: {seed: {"score": (1. if name == "candidate" else 0.)}
                     for group in V.SELECTION_GROUPS for seed in group}
              for name in ("c0", "candidate")}
    result = V.select_gradient(scores, ["c0", "candidate"])
    assert result["selected"] == "candidate" and result["accepted"]
    for seed in V.SELECTION_GROUPS[1]:
        scores["candidate"][seed]["score"] = -1.
    assert V.select_gradient(scores, ["c0", "candidate"])["selected"] == "c0"


def test_gradient_curve_has_three_observations_and_does_not_mutate_memory():
    memory = toy_memory()
    before = deepcopy(memory["x"])
    curve = V.gradient_curve(memory, 33, "cpu")
    assert [x["step"] for x in curve["observations"]] == [0, 1, 2]
    assert torch.equal(memory["x"], before)


def test_gradient_prefix_is_the_common_student_and_batches_match():
    import run_v7_gradient_append as R
    result = R.prefix_audit(toy_memory(), "cpu")
    assert result["status"] == "passed"
    assert all(item["bitwise_equal"] for item in result["checks"])


def test_candidate_slot_alignment_removes_a_pure_permutation():
    data = {"x": torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.]]),
            "text_emb": torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])}
    base = [{"class": 0, "ids": torch.tensor([0, 1])}, {"class": 0, "ids": torch.tensor([2, 3])}]
    aligned = V.align_class_slots(data, base, list(reversed(base)))
    assert all(torch.equal(a["ids"], b["ids"]) for a, b in zip(aligned, base))


def test_multi_prototype_is_an_exact_partition_with_exact_quotas():
    gen = torch.Generator().manual_seed(91)
    data = {"x": torch.randn(80, 6, generator=gen), "text_emb": torch.randn(80, 8, generator=gen),
            "y": torch.tensor([0] * 40 + [1] * 40), "pool_mask": torch.ones(80, dtype=torch.bool)}
    ids = torch.arange(80)
    records = V.multi_prototype_records(data, ids, [0, 1], {0: 4, 1: 5}, "cpu")
    V.V6.validate_partition(data, ids, records, [0, 1])
    assert len(records) == 9
    assert sum(r["class"] == 0 for r in records) == 4


def test_invalid_observations_cannot_silently_win():
    bad = {"observations": [{"gradient": torch.tensor([float("nan")])}] * 3}
    with pytest.raises(ValueError, match="gradient"):
        V.gradient_score(bad, bad, 1., 1.)


def test_resume_does_not_rewrite_registered_json(tmp_path):
    import run_v7_gradient_append as R
    path = tmp_path / "freeze.json"
    R.stable_lock(path, {"groups": ((1, 2), (3, 4)), "version": "v7"})
    before = path.read_bytes()
    R.stable_lock(path, {"version": "v7", "groups": [[1, 2], [3, 4]]})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="mismatch"):
        R.stable_lock(path, {"version": "changed"})


def test_report_seed_overlap_is_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        V.select_gradient({}, ["c0"], groups=((9300, 9301),))
