"""Mechanism tests must pass before any formal report-dev screening."""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import p2b_student_attention as T
from run_p2b_student_attention import validate_seeds


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.evidence = dict(keys=F.normalize(torch.randn(4, 5, 384), dim=2),
            mask=torch.ones(4, 5, dtype=torch.bool), query=F.normalize(torch.randn(4, 384), dim=1),
            anchor=F.normalize(torch.randn(4, 384), dim=1))

    def test_seed_independent_initial_output(self):
        a = T.AnchoredAttention(seed=700)(**self.evidence)[0]
        b = T.AnchoredAttention(seed=701)(**self.evidence)[0]
        self.assertTrue(torch.equal(a, b))

    def test_mask_permutation_and_anchor(self):
        m = T.AnchoredAttention()
        self.evidence["mask"][0, 4] = False
        a, w = m(**self.evidence)
        self.assertEqual(float(w[0, 4]), 0.)
        perm = torch.tensor([2, 0, 4, 1, 3])
        e = dict(self.evidence, keys=self.evidence["keys"][:, perm], mask=self.evidence["mask"][:, perm])
        b, _ = m(**e)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.all((a * e["anchor"]).sum(1) > .94))

    def test_source_split_and_seed_isolation(self):
        a, b = T.source_split(100, 700, 1)
        self.assertFalse(set(a.tolist()) & set(b.tolist()))
        self.assertEqual(len(a) + len(b), 100)
        self.assertTrue(torch.equal(a, T.source_split(100, 700, 1)[0]))
        validate_seeds(list(range(100, 106)))
        with self.assertRaises(ValueError):
            validate_seeds([100, 310])

    def test_zero_moment_is_finite(self):
        value = torch.tensor([0., 1.], requires_grad=True)
        T.sqrt_zero_safe(value).sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([0., .5]))

    def test_unroll_matches_official_adam(self):
        for device in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
            x = torch.randn(20, 12).to(device)
            t = F.normalize(torch.randn(20, 384), dim=1).to(device)
            edges = torch.tensor([[i for i in range(19)], [i + 1 for i in range(19)]])
            adj = T.A.G.norm_adj(edges, len(x), device)
            params = T.unroll(x, adj, t, 300, 16, differentiable=False)
            torch.manual_seed(300)
            model = T.A.G.CLIPStudent(12).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            gen = torch.Generator().manual_seed(2101)
            for _ in range(16):
                rows = torch.randperm(len(x), generator=gen)[:16].to(device)
                g, tt = model.encode_graph(x, adj)[rows], model.encode_text(t[rows])
                logits = model.logit_scale.exp().clamp(max=100) * g @ tt.T
                target = torch.arange(len(rows), device=device)
                loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
            for k, v in model.named_parameters():
                torch.testing.assert_close(params[k], v, atol=2e-5, rtol=2e-4)

    def test_gradient_through_student_real_objective(self):
        model = T.AnchoredAttention()
        target, _ = model(**self.evidence)
        old = F.normalize(torch.randn(3, 384), dim=1)
        original = old.clone()
        x = torch.randn(7, 8)
        adj = torch.eye(7).to_sparse()
        params = T.unroll(x, adj, torch.cat((old.detach(), target)), 300, 3)
        real_x, real_t = torch.randn(9, 8), F.normalize(torch.randn(9, 384), dim=1)
        loss = T.contrastive(params, real_x, torch.eye(9).to_sparse(), real_t, torch.arange(9))
        loss.backward()
        self.assertIsNotNone(model.wq.grad)
        self.assertTrue(torch.isfinite(model.wq.grad).all())
        self.assertGreater(float(model.wq.grad.norm()), 0.)
        self.assertTrue(torch.equal(old, original))
        self.assertIsNone(old.grad)
        self.assertFalse(self.evidence["anchor"].requires_grad)

    def test_lock_refuses_resume_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "plan.json"
            T.A.O.lock_json(p, {"steps": 40})
            with self.assertRaises(ValueError):
                T.A.O.lock_json(p, {"steps": 41})


if __name__ == "__main__":
    unittest.main()
