import unittest

import torch

import run_unified_v3_continual as U
from test_p2_text_operators import Encoder


class LegacyDefaultTests(unittest.TestCase):
    def test_default_concat_exactly_matches_v4_helper(self):
        torch.manual_seed(3)
        joint = torch.randn(8, 12)
        source = torch.randn(8, 4)
        members = torch.arange(8)
        texts = [(f"source{i} " * 100) for i in range(8)]
        for g, budget in ((3, 192), (5, 320), (8, 512)):
            encoder = Encoder()
            text, _ = U.SCA.encode_cluster_text(joint, members, texts, encoder,
                "concat", budget, g, source_text_emb=source)
            expected = U.G.fuse_new_only(joint, members, texts, budget_chars=budget, g=g)
            self.assertEqual(text, expected)


if __name__ == "__main__":
    unittest.main()
