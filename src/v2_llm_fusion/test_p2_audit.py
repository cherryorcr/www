import unittest

import numpy as np
from scipy.stats import t

from audit_p2_token_controls import effect


class PairedEffectTests(unittest.TestCase):
    def test_identical_results_have_zero_interval(self):
        result = effect([1, 4, 6, 7, 8, 2], [1, 4, 6, 7, 8, 2])
        self.assertEqual(result["paired_gain"], 0)
        self.assertEqual(result["ci_low"], 0)
        self.assertEqual(result["ci_high"], 0)

    def test_paired_not_unpaired_interval(self):
        base = np.arange(6) * 100
        result = effect(base + 1, base)
        self.assertEqual(result["paired_gain"], 1)
        self.assertEqual(result["ci_low"], 1)
        self.assertEqual(result["positive_seeds"], 6)

    def test_t_interval_uses_correct_degrees_of_freedom(self):
        for n in (6, 12):
            gain = np.arange(n)
            result = effect(gain, np.zeros(n))
            width = t.ppf(0.975, n - 1) * np.std(gain, ddof=1) / np.sqrt(n)
            self.assertAlmostEqual(result["ci_high"], np.mean(gain) + width)


if __name__ == "__main__":
    unittest.main()
