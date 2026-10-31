"""Unit tests for the minimal Phase-2 topology-only append."""
from __future__ import annotations

import unittest

import torch

import joint_kernel_append as J
import run_unified_v3_continual as U
import run_p0_random_synthetic as R


class Encoder:
    def encode(self, texts):
        return torch.nn.functional.normalize(torch.tensor([
            [sum(map(ord, t)) % 19 + 1, len(t) + 1, 2, 3] for t in texts], dtype=torch.float32), dim=1)


def memory(offset: float, n: int) -> dict:
    return {
        "x": torch.arange(offset, offset + n * 3, dtype=torch.float32).reshape(n, 3),
        "text_emb": torch.arange(offset + 100, offset + 100 + n * 2,
                                  dtype=torch.float32).reshape(n, 2),
        "graph_signature": torch.arange(offset + 200, offset + 200 + n * 4,
                                         dtype=torch.float32).reshape(n, 4),
        "mass": torch.arange(1, n + 1, dtype=torch.float32),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.zeros(n, dtype=torch.long),
        "texts": [f"row-{offset}-{i}" for i in range(n)],
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        if n >= 3 else torch.empty((2, 0), dtype=torch.long),
    }


class JointKernelAppendTests(unittest.TestCase):
    def test_old_rows_and_old_edges_are_preserved(self):
        old = memory(0, 3)
        block = memory(1000, 2)
        out, audit = J.append_joint(old, block, keep=3, degree=1.0,
                                    cross_degree=1)
        kept = torch.tensor([0, 1, 2])
        self.assertTrue(torch.equal(out["x"][:3], old["x"][kept]))
        self.assertTrue(torch.equal(out["graph_signature"][:3],
                                    old["graph_signature"][kept]))
        self.assertEqual(out["texts"][:3], old["texts"])
        old_edges = old["edge_index"]
        got = out["edge_index"][:, (out["edge_index"] < 3).all(0)]
        self.assertTrue(torch.equal(got, old_edges))
        self.assertTrue(audit["old_old_edges_unchanged"])
        self.assertEqual(len(out["x"]), 5)

    def test_pruning_is_rejected(self):
        with self.assertRaises(ValueError):
            J.append_joint(memory(0, 3), memory(1000, 2), keep=2)

    def test_induced_graph_excludes_history_future_and_holdout(self):
        edge = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 3, 4]])
        got = J.induced_edges(edge, torch.tensor([1, 2]), 5)
        self.assertTrue(torch.equal(got, torch.tensor([[0], [1]])))

    def test_kernel_edges_are_deterministic(self):
        block = memory(0, 5)
        first = J.kernel_edges(block, 1.0)
        second = J.kernel_edges(block, 1.0)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape[0], 2)

    def test_append_is_deterministic(self):
        first, _ = J.append_joint(memory(0, 3), memory(1000, 2), 3)
        second, _ = J.append_joint(memory(0, 3), memory(1000, 2), 3)
        for key in ("x", "text_emb", "graph_signature", "edge_index"):
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertEqual(first["texts"], second["texts"])

    def test_same_c0_rows_no_hidden_source_access(self):
        gen = torch.Generator().manual_seed(7)
        data = {"x": torch.randn(12, 3, generator=gen),
                "text_emb": torch.randn(12, 4, generator=gen),
                "texts": [f"source {i}" for i in range(12)],
                "y": torch.tensor([0]*5 + [1]*5 + [2]*2),
                "pool_mask": torch.tensor([True]*10 + [False]*2),
                "stage_classes": [[0, 1], [2]], "encoder": Encoder(),
                "edge_index": torch.tensor([[0, 1, 2, 3, 6, 8, 9], [1, 2, 3, 4, 7, 9, 10]])}
        ids = torch.arange(10)
        base = U.build_stage_block(data, ids, 4, data["encoder"], 0,
                                  "joint_g5_c320", "concat", "cpu", "dispersion", [0, 1])
        first = J.build_joint_block(U, data, ids, 4, data["encoder"], 0, "cpu", base=base)
        for key in ("x", "text_emb", "mass", "stage", "depth"):
            self.assertTrue(torch.equal(base[key], first[key]))
        self.assertEqual(sorted(sum(first["audit_source_groups"], [])), ids.tolist())
        data["x"][10:] = 999
        data["text_emb"][10:] = -999
        data["y"][10:] = 999
        second = J.build_joint_block(U, data, ids, 4, data["encoder"], 0, "cpu", base=base)
        self.assertTrue(torch.equal(first["graph_signature"], second["graph_signature"]))
        random, _ = R.random_synthetic_block(data, ids, 4, 0, 12)
        repeated, _ = R.random_synthetic_block(data, ids, 4, 0, 12)
        self.assertTrue(torch.equal(random["x"], repeated["x"]))
        self.assertTrue(torch.equal(random["text_emb"], data["encoder"].encode(random["texts"])))
        self.assertEqual(float(random["mass"].sum()), 10.)


if __name__ == "__main__":
    unittest.main()
