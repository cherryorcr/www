"""Regression checks for frozen-prefix append and legal rare-class splits."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_student_compatible_append as A
import uq_stad_native_core_20261012 as S


class FrozenAppendTests(unittest.TestCase):
    def memory(self, stage):
        return dict(x=torch.arange(8, dtype=torch.float).reshape(2, 4) + stage,
            text_emb=torch.eye(2, 4), texts=[f"stage {stage} first", f"stage {stage} second"],
            mass=torch.ones(2), depth=torch.zeros(2, dtype=torch.long),
            stage=torch.full((2,), stage, dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

    def test_frozen_append_preserves_inputs_and_edges_without_shrink_api(self):
        old, block = self.memory(0), self.memory(1)
        previous = copy.deepcopy(old)
        with patch.object(A.G, "shrink_by_mass", create=True, side_effect=AssertionError("must not shrink")):
            result = A.append_memory(old, block, 2, cross_degree=0, device="cpu")
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            self.assertTrue(torch.equal(result[key][:2], previous[key]), key)
            self.assertTrue(torch.equal(old[key], previous[key]), key)
        self.assertEqual(result["texts"], old["texts"] + block["texts"])
        self.assertTrue(torch.equal(result["edge_index"], torch.cat((old["edge_index"], block["edge_index"] + 2), 1)))
        result["x"][0, 0] = -100
        self.assertTrue(torch.equal(old["x"], previous["x"]))

    def test_actual_reduction_remains_delegated(self):
        old, block = self.memory(0), self.memory(1)
        shrunk = {k: (v[:1] if k != "edge_index" else torch.empty((2, 0), dtype=torch.long)) for k, v in old.items()}
        with patch.object(A.G, "shrink_by_mass", create=True, return_value=shrunk) as shrink:
            result = A.append_memory(old, block, 1, cross_degree=0, device="cpu")
        shrink.assert_called_once_with(old, 1)
        self.assertEqual(len(result["x"]), 3)


class SparseSplitTests(unittest.TestCase):
    def test_rare_classes_remain_trainable_without_overlap_or_external_rows(self):
        y = torch.tensor([0] + [1] * 2 + [2] * 3 + [3] * 50)
        train, val, unlabeled = S.internal_split(y, 271006)
        for left, right in ((train, val), (train, unlabeled), (val, unlabeled)):
            self.assertFalse(bool((left & right).any()))
        self.assertTrue(bool((train | val | unlabeled).all()))
        self.assertEqual(int(train[y == 0].sum()), 1)
        self.assertEqual(int(val[y == 0].sum()), 0)
        self.assertEqual(int(train[y == 1].sum()), 1)
        self.assertEqual(int(val[y == 1].sum()), 1)
        self.assertEqual(int(unlabeled[y <= 1].sum()), 0)
        self.assertEqual(sorted(torch.unique(y[train]).tolist()), [0, 1, 2, 3])
        for before, after in zip((train, val, unlabeled), S.internal_split(y, 271006)):
            self.assertTrue(torch.equal(before, after))

    def test_dense_classes_match_original_internal_split_exactly(self):
        y = torch.arange(4).repeat_interleave(torch.tensor([3, 20, 71, 100]))
        rng = np.random.RandomState(81)
        expected_train, expected_val = torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool)
        for cls in torch.unique(y):
            ids = rng.permutation(torch.where(y == cls)[0].numpy())
            n = max(1, round(.1 * len(ids)))
            expected_train[torch.as_tensor(ids[:n])] = True
            expected_val[torch.as_tensor(ids[n:2 * n])] = True
        actual = S.internal_split(y, 81)
        for left, right in zip(actual, (expected_train, expected_val, ~(expected_train | expected_val))):
            self.assertTrue(torch.equal(left, right))

    def test_no_global_validation_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "internal validation"):
            S.internal_split(torch.tensor([0, 1]), 17)

    def test_no_global_pseudo_pool_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "pseudo-label pool"):
            S.internal_split(torch.tensor([0, 0, 1, 1]), 17)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
