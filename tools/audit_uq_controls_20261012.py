"""Numerical contract checks before expensive attribution runs."""
from dataclasses import replace
import hashlib
import json

import torch

import run_unified_q_matrix_20261008 as R
import uq_controlled_assignment_20261012 as C


class Encoder:
    def encode(self, texts):
        return torch.stack([torch.randn(12, generator=torch.Generator().manual_seed(
            int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))) for t in texts])


def main():
    torch.set_num_threads(2)
    device = "cuda:0"
    encoder = Encoder()
    generator = torch.Generator().manual_seed(77)
    labels = torch.tensor([0] * 3 + [1] * 3)
    n = 30
    edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
    delta = dict(x=torch.randn(n, 9, generator=generator), y=torch.tensor([0] * 15 + [1] * 15),
                 ids=torch.arange(100, 100 + n), texts=[f"source {i}" for i in range(n)],
                 edge_index=torch.cat((edges, edges.flip(0)), 1), stage=0)
    prompt = encoder.encode(["class 0", "class 1", "class 2"])
    cfg = replace(R.q_config("uq_dm_edgefree", 271006, 3, 4096), proxy_hidden=8, proxy_out=8,
                  distribution_hidden=16, distribution_bank_size=2)
    parent = dict(x=torch.randn(2, 9, generator=generator), text_emb=R.UQ._norm(encoder.encode(["old 0", "old 1"])),
                  labels=torch.tensor([2, 2]), mass=torch.ones(2), depth=torch.zeros(2, dtype=torch.long),
                  stage=torch.zeros(2, dtype=torch.long), texts=["old 0", "old 1"],
                  edge_index=torch.empty((2, 0), dtype=torch.long), support_ids=[[], []], text_source_ids=[[], []])
    records = []
    for old in (None, parent):
        delta["stage"] = int(old is not None)
        native, _ = R.UQ.optimize_stage(delta, old, labels, encoder, prompt, cfg, device,
                                        use_boundary=False, use_topology=False)
        shared, audit = C.build("control_shared", delta, old, labels, encoder, prompt, cfg, device)
        errors = {key: float((native[key] - shared[key]).abs().max()) for key in ("x", "text_emb", "mass")}
        assert max(errors.values()) < 2e-5, errors
        assert native["texts"] == shared["texts"]
        equal_counts = []
        for arm in ("control_independent", "control_shared_no_residual", "control_independent_no_residual",
                    "control_shared_equal", "control_independent_equal", "control_soft_stored"):
            block, control = C.build(arm, delta, old, labels, encoder, prompt, cfg, device)
            assert R.assignment_checks(control)["sinkhorn_feasible"], control["assignments"]
            assert all(v["cross_class_mass"] == 0 for v in control["assignments"].values())
            assert control["initial_q_max_error"] == 0
            if "no_residual" in arm:
                assert control["residual_parameter_count"] == 0
                assert all(row["residual_relative_norm"] == 0 for row in control["trace"])
            if arm.endswith("equal"):
                equal_counts.append(control["assignment_parameter_count"])
            if arm == "control_soft_stored":
                assert control["stored_vs_reference_string_max_error"] > 1e-3
                assert control["forward_and_storage_match"]
            complete = R.L.append_edge_free(old, block)
            R.V.validate_memory(complete, old, len(labels) + (0 if old is None else len(old["x"])))
        assert equal_counts[0] == equal_counts[1], equal_counts
        records.append(dict(parent=old is not None, production_errors=errors, equal_counts=equal_counts))
    print(json.dumps(dict(status="passed", records=records)))


if __name__ == "__main__":
    main()
