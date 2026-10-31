"""Unit tests for conserved-payload two-prototype slot permutations."""
from __future__ import annotations

import unittest

import torch

import run_tgc_two_prototype_role_permutation as P


class ConservedPermutationTests(unittest.TestCase):
    def setUp(self):
        self.row_classes = [0, 0, 0, 1]
        self.q = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.data = {"x": torch.arange(32, dtype=torch.float32).reshape(8, 4),
                     "text_emb": torch.arange(32, 64, dtype=torch.float32).reshape(8, 4)}
        self.ids = torch.arange(8)
        self.records = [
            {"class": 0, "source_positions": [0, 1], "mass": 2, "changed": True},
            {"class": 0, "source_positions": [2, 3], "mass": 2, "changed": True},
            {"class": 0, "source_positions": [4, 5], "mass": 2, "changed": True},
            {"class": 1, "source_positions": [6, 7], "mass": 2, "changed": False},
        ]
        self.roles = {
            i: {"degree": float(i + 1), "cross_degree": float(i),
                "new_new_degree": float(i % 2), "old_neighbor_ratio": 0.1 * i,
                "old_endpoint_load": float(2 * i)}
            for i in range(4)
        }
        self.block = {
            "x": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "text_emb": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4),
            "texts": ["a", "b", "c", "d"],
            "mass": torch.tensor([2., 2., 2., 2.]),
            "depth": torch.tensor([1, 1, 1, 1]),
            "stage": torch.tensor([1, 1, 1, 1]),
        }

    def test_mapping_is_a_classwise_bijection(self):
        mapping, _ = P.complete_mapping(
            self.row_classes, self.records, self.data, self.ids, self.q,
            self.roles, 0, "role_perm", 17
        )
        self.assertEqual(set(mapping), set(range(4)))
        self.assertEqual(set(mapping.values()), set(range(4)))

    def test_permutation_preserves_payload_and_source_coverage(self):
        mapping, _ = P.complete_mapping(
            self.row_classes, self.records, self.data, self.ids, self.q,
            self.roles, 0, "random_perm", 17
        )
        block, records = P.permute_block(
            self.block, self.records, self.row_classes, mapping
        )
        old_digest = sorted(P.row_digest(self.block, i) for i in range(4))
        new_digest = sorted(P.row_digest(block, i) for i in range(4))
        self.assertEqual(old_digest, new_digest)
        self.assertEqual(
            P.source_signature(self.records, self.row_classes),
            P.source_signature(records, self.row_classes),
        )
        self.assertEqual(sorted(block["texts"]), sorted(self.block["texts"]))


if __name__ == "__main__":
    unittest.main()
