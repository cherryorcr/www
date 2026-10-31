import copy
import json

import pytest
import torch

import tgc_selector_confirmation as M
import run_tgc_selector_confirmation as B
from test_tgc_student_revision import memory, support


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def responses(gains=(.1, .1, .1), old_rise=0.):
    records = []
    for seed, gain in zip(M.SELECTOR_SEEDS, gains):
        base = [{"step": s, "old": 2., "new": 2., "mixed": 2.} for s in M.CHECKPOINTS]
        candidate = [{"step": s, "old": 2. + old_rise, "new": 2. - gain, "mixed": 2. - gain}
                     for s in M.CHECKPOINTS]
        records.append({"seed": seed, "c0": base, "candidate": candidate})
    return records


def test_registered_groups_and_seed_roles():
    M.frozen_protocol()
    for dataset, orders in M.GROUPS.items():
        for order, groups in orders.items():
            assert M.R.A.stage_groups(dataset, 3, order) == groups
            assert sorted(sum(groups, [])) == list(range(7 if dataset == "cora" else 6))


@pytest.mark.parametrize("gains,accepted", [((.1, .1, -.05), True), ((.1, -.01, -.01), False),
                                           ((.01, .01, -1.), False), ((0., 0., 0.), False)])
def test_majority_and_mean_are_both_required(gains, accepted):
    policies = M.policy_scores(responses(gains))["policies"]
    assert all(p["accepted"] is accepted for p in policies.values())


def test_old_guard_and_fail_closed_nan():
    assert not M.policy_scores(responses(old_rise=.02))["policies"]["selected_curve"]["accepted"]
    with pytest.raises(ValueError, match="nonfinite"):
        M.policy_scores(responses(gains=(float("nan"), .1, .1)))


def test_gate_is_separate_and_cannot_override_primary():
    d = M.decision_summary({"accepted": True}, {"per_seed": responses((-.1, -.1, -.1))})
    assert d["policies"]["selected_curve"]["selected_arm"] == "c0"
    assert d["policies"]["selected_gate"]["selected_arm"] == "candidate"
    assert d["primary_policy"] == "selected_curve"


def test_duplicate_seeds_or_checkpoints_rejected():
    rows = responses()
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError, match="seeds"):
        M.policy_scores(rows)
    rows = responses()
    rows[0]["c0"].append(copy.deepcopy(rows[0]["c0"][0]))
    with pytest.raises(ValueError, match="checkpoints"):
        M.policy_scores(rows)


def test_probe_strata_and_mixed_cross_negatives():
    batches = M.probe_batches(12, 32, 1)
    assert all((x < 12).all() for x in batches["old"])
    assert all((x >= 12).all() for x in batches["new"])
    assert all(int((x < 12).sum()) == 8 and int((x >= 12).sum()) == 8 for x in batches["mixed"])
    assert all(torch.equal(a, b) for key in batches for a, b in zip(batches[key], M.probe_batches(12, 32, 1)[key]))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_scratch_curve_matches_uninterrupted_adam_and_is_immutable(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    base = memory(12)
    sup = support(base, memory(12, seed=8))
    before = (M.T.memory_hash(base), M.T.memory_hash(sup))
    trajectory = M.scratch_trajectory(base, sup, M.probe_batches(12, 24, 1), 7820, device, (2, 4, 8))
    for step in (2, 4, 8):
        ref = M.A.make_prefix(base["x"].to(device), M.R.C.training_adjacency(base, device),
                              base["text_emb"].to(device), 7820, step)
        saved = trajectory["states"][step]["prefix"]
        assert torch.equal(saved["rng"], ref["rng"])
        for group in ("params", "first", "second"):
            for name in ref[group]:
                torch.testing.assert_close(saved[group][name], ref[group][name].cpu(), atol=2e-5, rtol=2e-4)
    assert before == (M.T.memory_hash(base), M.T.memory_hash(sup))


def test_seal_required_and_detects_any_decision_change(tmp_path):
    with pytest.raises(FileNotFoundError):
        B.verify_seal(tmp_path)
    B.save(tmp_path / "protocol.json", {})
    hashes = {}
    for dataset in B.DATASETS:
        for order in B.ORDERS:
            for stage in (0, 1, 2):
                relative = f"{dataset}/{order}/s{stage}_decision.json"
                B.save(tmp_path / relative, {})
                hashes[relative] = M.R.C.sha(tmp_path / relative)
    B.save(tmp_path / "selection_seal.json", {"protocol_sha256": M.R.C.sha(tmp_path / "protocol.json"),
                                             "decision_hashes": hashes})
    B.verify_seal(tmp_path)
    B.save(tmp_path / relative, {"changed": True})
    with pytest.raises(ValueError, match="selection changed"):
        B.verify_seal(tmp_path)
