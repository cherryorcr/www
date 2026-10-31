from dataclasses import replace

import pytest
import torch

import shared_q_controls as C
import unified_q_multimodal as U
from test_unified_q_multimodal import DummyEncoder, _delta
from test_unified_q_distribution import config

LABELS = torch.tensor([0, 0, 0, 1, 1, 1])
torch.set_num_threads(2)


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_shared_reproduces_frozen_constructor(steps):
    torch.manual_seed(512)
    delta, prompt = _delta(), torch.randn(2, 8)
    cfg = config(steps=steps)
    full, audit = U.optimize_stage(delta, None, LABELS, DummyEncoder(), prompt, cfg,
                                  "cpu", use_boundary=False, use_topology=False)
    tied, control = C.optimize(delta, None, LABELS, DummyEncoder(), prompt, cfg, "cpu", shared=True)
    for key in ("x", "text_emb", "mass", "labels"):
        # Extra empty-prefix concatenation changes mixed-derivative reduction
        # order by a few float32 ulps; hard selections must remain identical.
        torch.testing.assert_close(full[key], tied[key], rtol=0, atol=1e-6)
    assert full["text_source_ids"] == tied["text_source_ids"]
    for a, b in zip(audit["trace"], control["trace"]):
        for key in ("loss", "gradient_match", "distribution_match", "propagation", "q_anchor"):
            assert a[key] == pytest.approx(b[key], abs=1e-6)
    assert control["qx_qt_row_l1_mean"] == 0


def test_split_has_same_initial_state_and_independent_gradients():
    torch.manual_seed(612)
    delta, prompt = _delta(), torch.randn(2, 8)
    cfg = config(steps=0)
    a, aa = C.optimize(delta, None, LABELS, DummyEncoder(), prompt, cfg, "cpu", shared=True)
    b, bb = C.optimize(delta, None, LABELS, DummyEncoder(), prompt, cfg, "cpu")
    for key in ("x", "text_emb", "mass"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    _, bb = C.optimize(delta, None, LABELS, DummyEncoder(), prompt, replace(cfg, steps=3), "cpu")
    assert bb["trace"][0]["qx_gradient_norm"] > 0
    assert bb["trace"][0]["qt_gradient_norm"] > 0
    assert bb["qx_qt_row_l1_mean"] > 0
    assert bb["qx_changed_norm"] > 0 and bb["qt_changed_norm"] > 0
    assert bb["assignment_parameter_count"] == 2 * aa["assignment_parameter_count"]
    assert bb["x_residual_ratio_max"] <= cfg.trust_radius + 1e-6
    for values in bb["assignments"].values():
        assert values["cross_class_mass"] == 0
        assert values["row_error"] < 1e-4 and values["column_relative_error"] < 1e-4


def test_split_freezes_parent_and_materializes_real_text():
    torch.manual_seed(713)
    delta, prompt = _delta(), torch.randn(2, 8)
    parent, _ = C.optimize(delta, None, LABELS, DummyEncoder(), prompt, config(steps=0), "cpu")
    old_x, old_t = parent["x"].clone(), parent["text_emb"].clone()
    new_delta = _delta(1, 12)
    block, audit = C.optimize(new_delta, parent, LABELS, DummyEncoder(), prompt, config(), "cpu")
    memory = U.append_memory(parent, block)
    assert torch.equal(old_x, memory["x"][:len(old_x)])
    assert torch.equal(old_t, memory["text_emb"][:len(old_x)])
    assert audit["text_hard_forward_max_error"] < 1e-6
    assert all(i in new_delta["ids"].tolist() for group in block["text_source_ids"] for i in group)
    assert not memory["edge_index"].numel()


def test_no_gm_split_text_parameters_have_no_direct_optimization_signal():
    torch.manual_seed(714)
    delta, prompt = _delta(), torch.randn(2, 8)
    _, audit = C.optimize(delta, None, LABELS, DummyEncoder(), prompt,
                          config(steps=3, lambda_gradient=0.), "cpu")
    assert audit["qt_changed_norm"] == 0
    assert audit["qx_changed_norm"] > 0
