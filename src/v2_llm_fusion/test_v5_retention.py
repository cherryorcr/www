"""Safety and causal-control tests for the retention-constrained append."""
from copy import deepcopy

import pytest
import torch

import retention_constrained_append as V
import run_v5_retention as R

torch.set_num_threads(2)


def block(n=20, seed=4, stage=0):
    gen = torch.Generator().manual_seed(seed)
    edges = torch.stack((torch.arange(n), torch.arange(n).roll(1)))
    return {"x": torch.rand((n, 8), generator=gen),
            "text_emb": torch.rand((n, 8), generator=gen),
            "texts": [f"synthetic-{stage}-{i}" for i in range(n)],
            "mass": torch.arange(1, n + 1).float(),
            "depth": torch.zeros(n, dtype=torch.long),
            "stage": torch.full((n,), stage, dtype=torch.long),
            "edge_index": torch.cat((edges, edges.flip(0)), 1)}


@pytest.mark.parametrize("arm", V.ARMS)
def test_append_invariants_across_multiple_stages(arm):
    old = None
    for stage in range(6):
        new = block(seed=stage + 4, stage=stage)
        previous = deepcopy(old)
        memory, meta = V.append(old, new, arm)
        assert V.audit_append(old, new, memory, arm)["passed"]
        if old:
            for key, value in previous.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, old[key]), key
            if arm in ("core", "full"):
                assert meta["refresh"]["refreshed_rows"]
            if arm in ("topology", "full"):
                assert 0 < meta["topology"]["lambda"] < 1
                a = V.support(memory).to_dense()
                assert torch.allclose(a, a.T)
                assert torch.linalg.matrix_norm(a, 2) <= 1.000001
        old = memory


def test_c0_matches_frozen_append_exactly():
    first, second = block(), block(seed=17, stage=1)
    old, _ = V.append(None, first, "c0")
    actual, _ = V.append(old, second, "c0")
    expected = V.SCA.append_memory(old, second, len(old["x"]), device="cpu")
    assert V.training_hash(actual) == V.training_hash(expected)
    V.C.assert_frozen(old, actual)


def test_zero_bounds_disable_both_changes():
    cfg = V.RetentionConfig(anchor_radius=0, refresh_response_bound=0, operator_bound=0, response_bound=0)
    old, _ = V.append(None, block(), "full", cfg)
    new, meta = V.append(old, block(seed=8, stage=1), "full", cfg)
    assert meta["topology"]["lambda"] == 0
    assert torch.equal(new["x"][:len(old["x"])], old["x"])
    assert V.drift(old, new)["two_hop"] < 1e-10


def test_metadata_does_not_affect_training_input_hash():
    hashes = [V.training_hash(V.append(None, block(), arm)[0]) for arm in V.ARMS]
    assert len(set(hashes)) == 1


def test_core_change_detected():
    old, _ = V.append(None, block(), "full")
    new, _ = V.append(old, block(seed=7, stage=1), "full")
    new["x"][19] += 0.1
    with pytest.raises(AssertionError):
        V.audit_append(old, block(seed=7, stage=1), new, "full")


def test_budget_anchor_membership_and_small_blocks():
    old, _ = V.append(None, block(1), "full")
    assert len(old["buffer_rows"]) == 0
    new, _ = V.append(old, block(1, 7, 1), "full")
    assert len(new["buffer_rows"]) == 0
    assert V.audit_append(old, block(1, 7, 1), new, "full")["passed"]


def test_new_order_depends_on_counts_and_excludes_developed_orders():
    reference = [[0, 1], [2, 3], [4, 5]]
    counts = [10, 40, 15, 35, 20, 30]
    excluded = {R.canonical_groups(reference)}
    groups, details = R.matched_order(counts, reference, excluded)
    assert all(a != b for a, b in zip(groups, reference))
    assert sorted(sum(groups, [])) == list(range(6))
    assert details["objective"] >= 0
    other, _ = R.matched_order(counts, reference, excluded | {R.canonical_groups(groups)})
    assert groups != other


def test_pooled_and_task_slice_scope_are_kept_separate():
    target = torch.tensor([0, 0, 1, 1])
    prediction = torch.tensor([0, 0, 0, 1])
    pooled = R.M.score_predictions(target, prediction, [[0], [1]], 1)
    sliced = R.legacy_from_predictions(target, prediction, [[0], [1]], 1)
    assert pooled["old_f1"] == pytest.approx(80.)
    assert sliced["old_mean_f1"] == pytest.approx(100.)


def test_gate_requires_confirmation_not_only_canonical_gains():
    from audit_v5_retention import evaluate_gates
    paired, costs = [], []
    for arm in V.ARMS[1:]:
        for dataset in ("cora", "citeseer"):
            for order in ("canonical", "matched_unseen"):
                for metric in ("final", "stage_average", "old", "new"):
                    value = 2.0 if order == "canonical" else 0.2
                    paired.append({"arm": arm, "dataset": dataset, "order": order,
                                   "metric": metric, "mean": value, "simultaneous_lower": -1.0})
                costs.append({"arm": arm, "dataset": dataset, "order": order,
                              "construction_ratio_to_c0": 1.1, "construction_seconds": 2.0})
    decision = evaluate_gates(paired, costs, True)
    assert decision["large_candidate"] is None
    assert "two_gains_including_confirmation" in decision["candidates"]["core"]["failed_gates"]


def test_gate_does_not_treat_wide_uncertainty_as_noninferiority():
    from audit_v5_retention import evaluate_gates, paired_stats
    assert paired_stats([-10., -5., 0., 5., 10., 12.])["simultaneous_lower"] < -2
    paired, costs = [], []
    for arm in V.ARMS[1:]:
        for dataset in ("cora", "citeseer"):
            for order in ("canonical", "matched_unseen"):
                for metric in ("final", "stage_average", "old", "new"):
                    paired.append({"arm": arm, "dataset": dataset, "order": order,
                                   "metric": metric, "mean": 2.0, "simultaneous_lower": -4.0})
                costs.append({"arm": arm, "dataset": dataset, "order": order,
                              "construction_ratio_to_c0": 1.1, "construction_seconds": 2.0})
    decision = evaluate_gates(paired, costs, True)
    assert decision["large_candidate"] is None
    assert "simultaneous_noninferiority_evidence" in decision["candidates"]["core"]["failed_gates"]


def test_cross_backend_drift_comparison_rejects_material_changes():
    from audit_v5_retention import compare_construction_checks
    actual = {"passed": True, "training_sha256": "same", "drift": {"two_hop": 0.3675662802937013}}
    expected = {"passed": True, "training_sha256": "same", "drift": {"two_hop": 0.3675662802937014}}
    assert compare_construction_checks(actual, expected) < 1e-15
    actual["drift"]["two_hop"] += 1e-8
    with pytest.raises(AssertionError):
        compare_construction_checks(actual, expected)


def test_cross_backend_check_does_not_relax_hash_equality():
    from audit_v5_retention import compare_construction_checks
    actual = {"passed": True, "training_sha256": "different", "drift": {"two_hop": 0.01}}
    expected = {"passed": True, "training_sha256": "same", "drift": {"two_hop": 0.01}}
    with pytest.raises(AssertionError):
        compare_construction_checks(actual, expected)
