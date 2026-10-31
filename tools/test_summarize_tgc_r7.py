import unittest
from summarize_tgc_r7 import ranks, correlation


class RankingTests(unittest.TestCase):
    def test_ties_are_averaged(self):
        self.assertEqual(ranks([4, 1, 1, 9]), [3, 1.5, 1.5, 4])

    def test_lower_proxy_is_not_accidentally_reversed(self):
        self.assertAlmostEqual(correlation(ranks([-4, -3, -2, -1]), ranks([1, 2, 3, 4])), 1)
        self.assertAlmostEqual(correlation(ranks([-4, -3, -2, -1]), ranks([4, 3, 2, 1])), -1)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
