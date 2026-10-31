"""The new default dense branch must reproduce the archived constructor on a fixture."""
from dataclasses import asdict

import torch

import run as A
from test_unified_q_multimodal import DummyEncoder, _delta


def test_archived_dense_regression():
    assert str(A.R.UQ.__file__).startswith(str(A.FROZEN))
    torch.manual_seed(47)
    deltas = [_delta(), _delta(stage=1, offset=12)]
    prompt = torch.randn(2, 8)
    labels = torch.tensor([0, 0, 1, 1])
    cfg = A.R.UQ.Config(steps=3, sinkhorn_steps=12, proxy_hidden=8, proxy_out=8,
                        lambda_distribution=1., distribution_hidden=12)
    parents = [None, None]
    for delta in deltas:
        outputs = []
        for i, (module, config) in enumerate(((A.R.UQ, cfg), (A.U, A.U.Config(**asdict(cfg))))):
            block, audit = module.optimize_stage(delta, parents[i], labels, DummyEncoder(),
                prompt, config, "cpu", use_topology=False, use_boundary=False)
            parents[i] = A.R.L.append_edge_free(parents[i], block)
            outputs.append((block, audit))
        assert A.R.V.memory_hash(outputs[0][0]) == A.R.V.memory_hash(outputs[1][0])
        for l, r in zip(outputs[0][1]["trace"], outputs[1][1]["trace"]):
            assert l == r
