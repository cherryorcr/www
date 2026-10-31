import unittest
import numpy as np
import torch
import large_lp_support as B
import safe_joint_kernel_append as S


class LargeLPTests(unittest.TestCase):
    def test_exact_topology(self):
        for seed in range(8):
            gen = torch.Generator().manual_seed(seed)
            for n in (1, 2, 3, 8, 19, 37):
                block = {k: torch.randn(n, 12, generator=gen) for k in ("x", "text_emb", "graph_signature")}
                for weights in ((0, 1, 0), (4 / 7, 3 / 7, 0), (.4, .3, .3)):
                    for variant in ("degree_balanced", "coverage"):
                        self.assertTrue(torch.equal(B.select_internal(block, weights, variant),
                                                    S.select_internal(block, weights, variant)), (seed, n, weights, variant))

    def test_ties(self):
        for n in (3, 7, 20):
            for value in (0., 1.):
                block = {k: torch.full((n, 4), value) for k in ("x", "text_emb", "graph_signature")}
                for variant in ("degree_balanced", "coverage"):
                    self.assertTrue(torch.equal(B.select_internal(block, (.4, .3, .3), variant),
                                                S.select_internal(block, (.4, .3, .3), variant)))

    def test_five_stage_sparse_split(self):
        arrival = torch.arange(100) // 20
        edge = torch.tensor([(a, b) for a in range(100) for b in range(a + 1, 100) if (a * 37 + b) % 11 == 0]).T
        split = B.split_edges(edge, arrival, 39001)
        again = B.split_edges(edge.flip(0), arrival, 39001)
        for key in ("train_pos", "test_neg", "train_neg", "val_neg"):
            self.assertTrue(torch.equal(split[key], again[key]))
        B.L.validate_split(split)

    def test_nonedges_exclude_all_observed(self):
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        neg = B.training_negatives(edges, 10, 31)
        self.assertEqual(neg.shape[1], 42)
        self.assertFalse(B.L.pair_keys(neg, 10) & B.L.pair_keys(edges, 10))
        self.assertEqual(len(B.L.pair_keys(neg, 10)), 42)


if __name__ == "__main__":
    unittest.main()
