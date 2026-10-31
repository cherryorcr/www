import unittest
from unittest.mock import patch

import torch

import protocol_v3 as P
import run_unified_v3_continual as U


class P0ComparisonControlTests(unittest.TestCase):
    def test_class_groups_are_canonical_but_stage_order_is_preserved(self):
        groups = P.canonicalize_class_groups([[5, 1], [4, 2], [0, 3]])
        self.assertEqual(groups, [[1, 5], [2, 4], [0, 3]])

    def test_duplicate_or_overlapping_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            P.canonicalize_class_groups([[1, 1], [2]])
        with self.assertRaises(ValueError):
            P.canonicalize_class_groups([[1, 2], [2, 3]])

    def test_allocation_is_invariant_to_input_class_order(self):
        data = {
            "y": torch.tensor([0, 0, 1, 1, 2, 2]),
            "x": torch.tensor([[1., 0.], [1., 1.], [2., 0.], [2., 1.],
                                [3., 0.], [3., 1.]]),
            "text_emb": torch.tensor([[1., 0.], [1., 1.], [2., 0.], [2., 1.],
                                       [3., 0.], [3., 1.]])
        }
        ids = torch.arange(6)
        left = U.allocate_rows_by_class(data, ids, [2, 0, 1], 6, "dispersion")
        right = U.allocate_rows_by_class(data, ids, [0, 1, 2], 6, "dispersion")
        self.assertEqual(left, right)

    def test_synthetic_row_order_is_invariant_to_class_list_order(self):
        data = {
            "y": torch.tensor([0, 0, 2, 2]),
            "x": torch.tensor([[0., 0.], [0., 1.], [2., 0.], [2., 1.]]),
            "text_emb": torch.tensor([[0., 0.], [0., 1.], [2., 0.], [2., 1.]]),
            "texts": ["a0", "a1", "b0", "b1"]
        }

        def fake_block(x, text_emb, texts, k, encoder, recipe, stage,
                       text_operator="concat", device=None,
                       diversity_lambda=0.5, dual_alpha=0.5):
            return {"x": x[:k].float(), "text_emb": text_emb[:k].float(),
                    "texts": list(texts[:k]), "mass": torch.ones(k),
                    "depth": torch.zeros(k, dtype=torch.long),
                    "stage": torch.full((k,), stage, dtype=torch.long)}

        with patch.object(U.SCA, "build_compact_block", side_effect=fake_block):
            left = U.build_stage_block(data, torch.arange(4), 2, None, 0,
                                       "joint_g5_c320", "concat", "cpu",
                                       "equal", [2, 0])
            right = U.build_stage_block(data, torch.arange(4), 2, None, 0,
                                        "joint_g5_c320", "concat", "cpu",
                                        "equal", [0, 2])
        self.assertTrue(torch.equal(left["x"], right["x"]))
        self.assertEqual(left["texts"], right["texts"])


if __name__ == "__main__":
    unittest.main()
