import unittest

import torch

from run_p2_numerical_guard import pool_parts
from test_p2_text_operators import Encoder


class CanonicalOrderTests(unittest.TestCase):
    def test_sorted_multisets_are_bitwise_equal(self):
        parts = ["third term", "first paper words", "second text"]
        for operator in ("mean_pool", "set_pool"):
            encoder = Encoder()
            a = pool_parts(sorted(parts), encoder, operator)
            b = pool_parts(sorted(reversed(parts)), encoder, operator)
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
