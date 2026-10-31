"""Focused tests for independent metrics and construction-level uncertainty."""
import math
import unittest

import numpy as np

from summarize import SEEDS, aggregate, paired
from verify_results import metrics


class AnalysisTest(unittest.TestCase):
    def test_fixed_taxonomy_zero_support(self):
        result = metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), 3)
        self.assertAlmostEqual(result["f1"], 100 * 4 / 9)
        self.assertAlmostEqual(result["accuracy"], 100 * 2 / 3)

    def test_empty_and_perfect(self):
        self.assertIsNone(metrics(np.array([], dtype=int), np.array([], dtype=int), 3))
        result = metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), 3)
        self.assertEqual(result["f1"], 100)
        self.assertEqual(result["accuracy"], 100)

    def test_paired_construction_unit(self):
        left, right = dict(zip(SEEDS, [2., 4., 6.])), dict(zip(SEEDS, [1., 3., 5.]))
        result = paired(left, right)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["ci95_low"], 1.)
        self.assertEqual(result["ci95_high"], 1.)
        self.assertEqual(result["positive_pairs"], 3)

    def test_nonconstant_interval(self):
        result = paired(dict(zip(SEEDS, [0., 1., 2.])), dict.fromkeys(SEEDS, 1.))
        self.assertAlmostEqual(result["difference_pp"], 0.)
        self.assertAlmostEqual(result["ci95_high"], 4.302652729911275 / math.sqrt(3))
        self.assertEqual(result["positive_pairs"], 1)
        self.assertEqual(result["worst_difference_pp"], -1.)

    def test_missing_or_duplicate_seed_rejected(self):
        with self.assertRaises(AssertionError):
            paired(dict(zip(SEEDS[:2], [1, 2])), dict(zip(SEEDS[:2], [1, 2])))
        rows = [dict(arm="a", construction_seed=s, f1=1.) for s in [SEEDS[0]] * 3]
        with self.assertRaises(AssertionError):
            aggregate(rows, ("arm",), ("f1",))


if __name__ == "__main__":
    unittest.main()
