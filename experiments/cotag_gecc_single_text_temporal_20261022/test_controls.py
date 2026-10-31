import unittest
import torch

from controls import assert_numeric_equal, assert_frozen_prefix, assert_single_source, text_statistics


def memory():
    return dict(x=torch.ones(2, 3), text_emb=torch.ones(2, 3), labels=torch.tensor([0, 1]),
                mass=torch.ones(2), stage=torch.zeros(2), depth=torch.zeros(2),
                edge_index=torch.empty(2, 0), texts=["old a", "old b"])


class ControlsTest(unittest.TestCase):
    def test_text_only_intervention(self):
        old = memory()
        new = dict(old, texts=["new a", "new b"], text_emb=torch.zeros(2, 3))
        assert_numeric_equal(new, old)

    def test_feature_change_rejected(self):
        with self.assertRaises(AssertionError):
            assert_numeric_equal(dict(memory(), x=torch.zeros(2, 3)), memory())

    def test_frozen_text_and_embedding(self):
        assert_frozen_prefix(memory(), memory())
        with self.assertRaises(AssertionError):
            assert_frozen_prefix(dict(memory(), texts=["changed", "old b"]), memory())
        with self.assertRaises(AssertionError):
            assert_frozen_prefix(dict(memory(), text_emb=torch.zeros(2, 3)), memory())

    def test_first_source_and_access(self):
        assert_single_source([[10]], [[10, 11, 12]], [10, 11, 12])
        for bad in ([[11]], [[99]], [[10, 11]]):
            with self.assertRaises(AssertionError):
                assert_single_source(bad, [[10, 11, 12]], [10, 11, 12])

    def test_text_budget(self):
        self.assertEqual(text_statistics(["a"*320, "a"*320])["unique_text_ratio"], .5)
        with self.assertRaises(AssertionError):
            text_statistics(["a"*321])


if __name__ == "__main__":
    unittest.main()
