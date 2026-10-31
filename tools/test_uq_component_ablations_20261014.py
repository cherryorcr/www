"""Behavioral checks against the frozen constructor, not mirrored formulas."""
from dataclasses import replace

import torch

import unified_q_multimodal as U
import uq_ablation_assignment_20261014 as C
from test_unified_q_multimodal import DummyEncoder, _delta
from test_unified_q_distribution import config
import run_uq_component_ablations_20261014 as runner


def test_derived_constructor_keeps_primary_and_fixed_q_equivalence():
    torch.manual_seed(418)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    for train_q in (True, False):
        cfg = config(steps=3, train_q=train_q)
        primary, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
            cfg, "cpu", use_boundary=False, use_topology=False)
        derived, audit = C.optimize(delta, None, labels, DummyEncoder(), prompt, cfg, "cpu")
        for key in ("x", "text_emb", "mass"):
            torch.testing.assert_close(primary[key], derived[key], rtol=2e-5, atol=2e-6)
        assert primary["texts"] == derived["texts"]
        assert audit["q_update_count"] == (3 if train_q else 0)


def test_randomization_separates_initialization_from_learning():
    torch.manual_seed(811)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    outputs = []
    for train_q in (False, True):
        block, audit = C.optimize(delta, None, labels, DummyEncoder(), prompt,
            config(steps=3, train_q=train_q), "cpu", randomized_initial=True)
        assert audit["q_cost"]["moved_rows"] > 0
        assert audit["assignments"]["qx"]["cross_class_mass"] == 0
        assert audit["assignments"]["qx"]["row_error"] < 1e-4
        assert audit["residual_update_count"] == 3
        outputs.append(audit)
    assert outputs[0]["diagnostics"]["assignments"]["feature"]["relative_update"] < 1e-7
    assert outputs[1]["diagnostics"]["assignments"]["feature"]["relative_update"] > 0


def test_text_interventions_keep_features_and_budget():
    import run_unified_q_matrix_20261008 as R

    class Encoder(DummyEncoder):
        def encode(self, texts):
            return torch.tensor(super().encode(texts))

    torch.manual_seed(86)
    delta = _delta()
    delta["texts"] = [f"distinct source {i} " * 80 for i in range(12)]
    labels, prompt = torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    encoder = Encoder()
    block, _ = U.optimize_stage(delta, None, labels, encoder, prompt,
        config(steps=2), "cpu", use_boundary=False, use_topology=False)
    for arm in runner.PAYLOADS[1:]:
        altered, audit = runner.payload_block(block, delta, encoder, 271006, arm, R)
        torch.testing.assert_close(altered["x"], block["x"], rtol=0, atol=0)
        assert max(map(len, altered["texts"])) <= 320
        assert not audit["construction_forward_uses_altered_payload"]
        for cls, ids in zip(labels, altered["text_source_ids"]):
            assert all(delta["y"][int(i)] == cls for i in ids)
        if arm == "uq_class_shared_payload":
            assert altered["texts"][0] == altered["texts"][1]
            assert altered["texts"][2] == altered["texts"][3]
        else:
            assert all(len(ids) == 5 for ids in altered["text_source_ids"])


def test_diagnostics_preserve_next_stage_and_student_randomness():
    torch.manual_seed(25)
    delta, labels, prompt = _delta(), torch.tensor([0, 0, 1, 1]), torch.randn(2, 8)
    cfg = config(steps=3)
    block, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt,
        cfg, "cpu", use_boundary=False, use_topology=False)
    rng = torch.get_rng_state().clone()
    observed = {}
    replay, _ = U.optimize_stage(delta, None, labels, DummyEncoder(), prompt, cfg,
        "cpu", use_boundary=False, use_topology=False,
        diagnostic_callback=lambda state: observed.update(runner.slot_diagnostics(state, C, U)))
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(block["x"], replay["x"], rtol=0, atol=0)
    assert block["texts"] == replay["texts"]
    assert observed["assignments"]["feature"]["positive_mass_slot_coverage"] == 1
    assert observed["effective_support_size_mean"] > 1
