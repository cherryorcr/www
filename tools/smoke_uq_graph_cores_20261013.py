"""Check native core gradients, finite materialization and source pinning."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/home/pc/research/TAG_condensation")
sys.path.insert(0, str(ROOT / "experiments/uq_final_evidence_20261012/code"))
import uq_native_graph_cores_20261013 as N


class Encoder:
    def encode(self, texts):
        return torch.ones((len(texts), 384), dtype=torch.float32)/np.sqrt(384)


torch.set_num_threads(2)
torch.manual_seed(111)
delta = dict(x=torch.randn(20, 8)*.1, y=torch.tensor([0]*10+[1]*10),
    ids=torch.arange(20), texts=[f"source {i}" for i in range(20)],
    text_emb=torch.ones(20, 384)/np.sqrt(384), stage=0,
    edge_index=torch.stack((torch.arange(19), torch.arange(1, 20))))
labels = torch.tensor([0, 0, 1, 1])
config = N.Config(gcond_steps=3, experts=2, teacher_epochs=20, snapshot_interval=1,
    start_epoch=5, expert_horizon=5, outer_steps=3, syn_steps=3, hidden=16)
for arm in ("gcond_x_core", "sfgc_core"):
    block, audit = N.build(arm, delta, None, labels, Encoder(), 150, "cuda:0", ROOT, config)
    assert block["x"].shape == (4, 8) and torch.isfinite(block["x"]).all()
    assert torch.equal(block["labels"], labels)
    assert not block["edge_index"].numel()
    assert any(step["feature_gradient_norm"] > 0 for step in audit["trace"])
    assert audit["feature_update_norm"] > 0
    print(arm, "native core smoke passed", flush=True)
