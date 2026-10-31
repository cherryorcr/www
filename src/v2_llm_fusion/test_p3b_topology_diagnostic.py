"""Unit tests for the P3-B topology controls."""
from __future__ import annotations

import unittest

import torch

import run_p3b_topology_diagnostic as P3B


def memory(n: int, text_dim: int = 3, x_dim: int = 3):
    text = torch.eye(text_dim)[:n]
    x = torch.eye(x_dim)[:n]
    return {
        "x": x,
        "text_emb": text,
        "texts": [str(i) for i in range(n)],
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "mass": torch.ones(n),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.zeros(n, dtype=torch.long),
    }


class P3BTopologyTests(unittest.TestCase):
    def setUp(self):
        self.old = memory(2)
        self.block = {
            "x": torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]]),
            "text_emb": torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2]]),
            "texts": ["a", "b"],
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "mass": torch.ones(2),
            "depth": torch.zeros(2, dtype=torch.long),
            "stage": torch.ones(2, dtype=torch.long),
        }

    def test_all_variants_have_same_cross_count(self):
        counts = []
        for arm in P3B.ARMS:
            edges = P3B._make_cross_edges(self.old, self.block, arm, 1, "canonical")
            counts.append(edges.shape[1])
        self.assertEqual(counts, [4, 0, 4, 4])

    def test_random_is_deterministic(self):
        first = P3B._make_cross_edges(self.old, self.block, "random", 1, "canonical")
        second = P3B._make_cross_edges(self.old, self.block, "random", 1, "canonical")
        self.assertTrue(torch.equal(first, second))

    def test_append_preserves_old_old_edges(self):
        old_edges = self.old["edge_index"]
        for arm in P3B.ARMS:
            out = P3B.append_variant(self.old, self.block, 2, arm, 1, "canonical", "cpu")
            n = len(self.old["x"])
            old_part = out["edge_index"][:, (out["edge_index"] < n).all(0)]
            self.assertTrue(torch.equal(old_edges, old_part))

    def test_none_has_zero_structural_drift(self):
        out = P3B.append_variant(self.old, self.block, 2, "none", 1, "canonical", "cpu")
        metrics = P3B.topology_metrics(self.old, out, "cpu", "none")
        self.assertAlmostEqual(metrics["old_old_normalized_fro_drift"], 0.0, places=6)
        self.assertAlmostEqual(metrics["old_propagation_relative_drift"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
