"""Small graph tests for the independent topology/statistics checks."""
import unittest

import torch

from audit_topology_causal_ablation import graph_checks, nc_scores, metric_gap


def memory(n, pairs):
    edge = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    return dict(x=torch.zeros(n, 2), texts=["x"] * n,
                edge_index=torch.cat((edge, edge.flip(0)), 1))


class AuditTest(unittest.TestCase):
    def test_ring(self):
        info, degree = graph_checks(memory(3, [(0, 1), (1, 2), (0, 2)]), None, 0)
        self.assertEqual(info["pairs"], 3)
        self.assertEqual(info["degree_gini"], 0)
        self.assertEqual(info["new_components"], 1)
        self.assertTrue((degree == 2).all())

    def test_star_gini_and_components(self):
        info, _ = graph_checks(memory(4, [(0, 1), (0, 2)]), None, 0)
        self.assertAlmostEqual(info["degree_gini"], .375)
        self.assertEqual(info["isolated"], 1)
        self.assertEqual(info["new_components"], 2)

    def test_old_normalization_changes_without_rewriting(self):
        old = memory(2, [(0, 1)])
        current = memory(3, [(0, 1), (1, 2)])
        info, _ = graph_checks(current, old, 2)
        self.assertAlmostEqual(info["max_relative_load"], .5)
        self.assertGreater(info["old_block_row_l1"], 0)
        self.assertAlmostEqual(info["full_old_row_l1"],
                               info["old_block_row_l1"] + info["cross_norm_mass"])

    def test_duplicates_fail(self):
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            graph_checks(memory(2, [(0, 1), (0, 1)]), None, 0)

    def test_self_edges_fail(self):
        with self.assertRaises(AssertionError):
            graph_checks(memory(2, [(0, 0)]), None, 0)

    def test_metric_none_scope(self):
        import numpy as np
        score = nc_scores(torch.tensor([[1., 0.], [0., 1.]]), np.array([0, 1]), [[0], [1]], 0)
        self.assertEqual(score["all"]["cumulative_f1"], 100.)
        self.assertIsNone(score["all"]["old_f1"])
        self.assertEqual(metric_gap(score, score), 0)


if __name__ == "__main__":
    unittest.main()
