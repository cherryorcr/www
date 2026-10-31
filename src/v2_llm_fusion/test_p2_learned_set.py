import unittest
import torch
import torch.nn.functional as F

from p2_learned_set import DeepSetTarget, fit_source_only


class LearnedSetTests(unittest.TestCase):
    def test_permutation_invariance_and_mask(self):
        torch.manual_seed(9)
        model = DeepSetTarget(dim=8)
        x, mask = torch.randn(4, 5, 8), torch.ones(4, 5)
        mask[:, -1] = 0
        output = model(x, mask)
        order = torch.tensor([4, 2, 0, 3, 1])
        self.assertTrue(torch.allclose(output, model(x[:, order], mask[:, order]), atol=1e-6))
        x[:, -1] = 1e6
        self.assertTrue(torch.allclose(output, model(x, mask), atol=1e-6))

    def test_source_holdout_and_determinism(self):
        torch.manual_seed(11)
        x = torch.randn(12, 5, 8)
        mask = torch.ones(12, 5)
        target = F.normalize(x.mean(1), dim=-1)
        a, _, audit = fit_source_only(x, mask, target, steps=10)
        b, _, audit_b = fit_source_only(x, mask, target, steps=10)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(audit, audit_b)
        self.assertFalse(set(audit["train_cluster_ids"]) & set(audit["holdout_cluster_ids"]))
        self.assertFalse(audit["fit_labels_used"])


if __name__ == "__main__":
    unittest.main()
