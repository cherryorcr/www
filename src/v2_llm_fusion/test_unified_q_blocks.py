from dataclasses import replace

import pytest
import torch

import unified_q_multimodal as U
from unified_q_blocks import ClassBlocks
from test_unified_q_multimodal import DummyEncoder, _delta


def partition(value, labels, slots):
    rows = [torch.where(labels == c)[0] for c in labels.unique(sorted=True)]
    cols = [torch.where(slots == c)[0] for c in labels.unique(sorted=True)]
    return ClassBlocks(value.shape, rows, cols,
                       [value[r[:, None], s[None, :]] for r, s in zip(rows, cols)])


@pytest.mark.parametrize("steps", [4, 80])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_transport_pool_anchor_and_two_derivatives(steps, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(4)
    labels = torch.tensor([1, 0, 1, 2, 0, 2, 1, 0], device=device)
    slots = torch.tensor([2, 0, 1, 0, 1], device=device)
    dense = torch.randn(8, 5, requires_grad=True, device=device)
    other = dense.detach().clone().requires_grad_()
    q = U._masked_sinkhorn(dense, labels, slots, .7, steps)
    b = U._masked_sinkhorn(partition(other, labels, slots), labels, slots, .7, steps)
    torch.testing.assert_close(q, b.to_dense(), atol=0, rtol=0)
    raw = torch.randn(8, 7, device=device)
    initial = U._masked_sinkhorn(torch.zeros_like(dense), labels, slots, .7, steps)
    initial_block = partition(initial, labels, slots)
    for dim in (0, 1):
        torch.testing.assert_close(q.sum(dim), b.sum(dim))
        assert torch.equal(q.argmax(dim), b.argmax(dim))
    torch.testing.assert_close(q.T @ raw, b.T @ raw)
    dense_anchor = (q - initial).square().mean()
    torch.testing.assert_close(dense_anchor, b.anchor(initial_block))
    torch.testing.assert_close(-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(1).mean(), b.entropy())
    lhs = (q.T @ raw).square().mean() + dense_anchor
    rhs = (b.T @ raw).square().mean() + b.anchor(initial_block)
    dg = torch.autograd.grad(lhs, dense, create_graph=True)[0]
    bg = torch.autograd.grad(rhs, other, create_graph=True)[0]
    torch.testing.assert_close(dg, bg, atol=2e-6, rtol=2e-5)
    dh = torch.autograd.grad(dg.square().sum(), dense)[0]
    bh = torch.autograd.grad(bg.square().sum(), other)[0]
    torch.testing.assert_close(dh, bh, atol=2e-6, rtol=2e-5)
    assert bool((bg[labels[:, None] != slots[None, :]] == 0).all())


def test_ties_and_materialization_match():
    labels, slots = torch.tensor([1, 0, 1, 0]), torch.tensor([0, 1])
    q = (labels[:, None] == slots[None, :]).float()
    block = partition(q, labels, slots)
    for dim in (0, 1):
        assert torch.equal(block.argmax(dim), q.argmax(dim))
    assert block[torch.tensor([1, 3]), 0].tolist() == [1., 1.]
    assert block[2, 1].item() == 1.
    with pytest.raises(ValueError, match="cross-class"):
        block[0, 0]


def test_two_stages_full_objective_and_frozen_prefix():
    torch.manual_seed(47)
    delta0, delta1, prompt = _delta(), _delta(stage=1, offset=12), torch.randn(2, 8)
    labels = torch.tensor([0, 0, 1, 1])
    cfg = U.Config(steps=3, sinkhorn_steps=12, proxy_hidden=8, proxy_out=8,
                   lambda_distribution=1., distribution_hidden=12)
    parents = [None, None]
    for delta in (delta0, delta1):
        outputs = []
        for index, storage in enumerate(("dense", "class_blocks")):
            before = None if parents[index] is None else parents[index]["x"].clone()
            block, audit = U.optimize_stage(delta, parents[index], labels, DummyEncoder(),
                prompt, replace(cfg, assignment_storage=storage), "cpu",
                use_topology=False, use_boundary=False)
            memory = U.append_memory(parents[index], block)
            if before is not None:
                assert torch.equal(before, memory["x"][:len(before)])
            parents[index] = memory
            outputs.append((block, audit))
        a, b = outputs[0][0], outputs[1][0]
        torch.testing.assert_close(a["x"], b["x"], atol=2e-5, rtol=2e-4)
        assert a["texts"] == b["texts"]
        assert a["text_source_ids"] == b["text_source_ids"]
        assert a["support_ids"] == b["support_ids"]
        torch.testing.assert_close(a["text_emb"], b["text_emb"])
        for left, right in zip(outputs[0][1]["trace"], outputs[1][1]["trace"]):
            for key in ("loss", "gradient_match", "distribution_match", "propagation", "q_anchor"):
                assert left[key] == pytest.approx(right[key], abs=2e-5, rel=2e-4)
        assert outputs[1][1]["assignment_parameter_entries"] == 24


def test_unsupported_mode_fails_explicitly():
    with pytest.raises(ValueError, match="restricted"):
        U.optimize_stage(_delta(), None, torch.tensor([0, 1]), DummyEncoder(), torch.randn(2, 8),
            U.Config(assignment_storage="class_blocks"), "cpu", use_topology=True)
