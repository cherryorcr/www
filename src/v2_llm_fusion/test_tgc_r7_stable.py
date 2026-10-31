"""Check mathematical equivalence and legal source matching for R7 diagnostics."""
import unittest
import torch
import tgc_r7_stable as Q


class StableTests(unittest.TestCase):
    def test_dense_objective_matches_sparse_value_and_gradient(self):
        gen = torch.Generator().manual_seed(41)
        old_n = 3
        initial = torch.rand(8, 4, generator=gen, dtype=torch.double)
        initial = initial / initial.sum(1, keepdim=True)
        h = torch.rand(11, 9, generator=gen, dtype=torch.double)
        a = torch.rand(11, 11, generator=gen, dtype=torch.double)
        a = (a+a.T)/2
        p = initial.clone().requires_grad_()
        dense, terms, coarse = Q.dense_objective(p, initial, h, a, old_n, Q.J.Config())
        sparse, other, projected = Q.J.objective(p, initial, h, a.to_sparse(), old_n, Q.J.Config())
        torch.testing.assert_close(dense, sparse, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(coarse, projected, rtol=1e-10, atol=1e-10)
        dg = torch.autograd.grad(dense, p)[0]
        sg = torch.autograd.grad(sparse, p)[0]
        torch.testing.assert_close(dg, sg, rtol=1e-9, atol=1e-10)

    def test_matched_batches_use_only_legal_unique_rows(self):
        q = torch.arange(8).repeat_interleave(5)
        a = Q.response_plan(q, 12, 20)
        b = Q.response_plan(q, 12, 20)
        for (c, s), (d, t) in zip(a, b):
            self.assertTrue(torch.equal(c, d) and torch.equal(s, t))
            self.assertEqual(len(s.unique()), len(s))
            for ci, si in zip(c.tolist(), s.tolist()):
                self.assertEqual(ci, si if ci < 12 else int(q[si-12])+12)

    def test_coherent_bound_keeps_class_mask_and_row_sums(self):
        q = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        initial, uniform, allowed, _ = Q.initial_assignment(q, labels, Q.J.Config(), torch.float32, "cpu")
        p0 = .98*initial+.02*uniform
        p = .5*p0+.5*uniform
        torch.testing.assert_close(p.sum(1), torch.ones(len(p)))
        self.assertEqual(int(torch.count_nonzero(p[~allowed])), 0)
        self.assertTrue((p.sum(0)>0).all())

    def test_numeric_context_restores_flags_on_error(self):
        flag = torch.are_deterministic_algorithms_enabled()
        with self.assertRaises(ValueError):
            with Q.N.deterministic(True):
                raise ValueError("intentional")
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), flag)


if __name__ == "__main__":
    unittest.main()
