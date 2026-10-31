import unittest

import torch

from tgc_deterministic_spmm import from_sparse
import tgc_r7_deterministic as R7
import tgc_r7_stable as DenseR7
import tgc_shared_assignment as Shared


class DeterministicSpMMTests(unittest.TestCase):
    def _matrix(self, device, dtype):
        indices = torch.tensor([
            [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 1],
            [0, 2, 0, 1, 4, 2, 1, 3, 0, 3, 4, 4],
        ], device=device)
        values = torch.tensor([1., .3, .3, 1., .2, 1., .4, 1., .2, .4, 1., .1],
                              device=device, dtype=dtype)
        return torch.sparse_coo_tensor(indices.cpu(), values.cpu(), (5, 5)).coalesce().to(device)

    def test_forward_and_backward_match_dense_reference(self):
        for dtype in (torch.float32, torch.float64):
            matrix = self._matrix("cpu", dtype)
            segmented = from_sparse(matrix, edge_limit=3)
            x1 = torch.randn(5, 7, dtype=dtype, requires_grad=True)
            x2 = x1.detach().clone().requires_grad_()
            weight = torch.randn(5, 7, dtype=dtype)
            actual = segmented @ x1
            expected = matrix.to_dense() @ x2
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(segmented.row_sums(), matrix.to_dense().sum(1),
                                       atol=2e-6, rtol=2e-6)
            (actual * weight).sum().backward()
            (expected * weight).sum().backward()
            torch.testing.assert_close(x1.grad, x2.grad, atol=2e-6, rtol=2e-6)

    def test_repeated_forward_and_backward_are_bitwise_identical(self):
        matrix = self._matrix("cpu", torch.float32)
        segmented = from_sparse(matrix, edge_limit=4)
        x = torch.randn(5, 13, requires_grad=True)
        outputs, gradients = [], []
        for _ in range(4):
            outputs.append((segmented @ x).detach().clone())
            grad, = torch.autograd.grad((segmented @ x).square().sum(), x, retain_graph=True)
            gradients.append(grad)
        for value in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], value))
        for value in gradients[1:]:
            self.assertTrue(torch.equal(gradients[0], value))

    def test_x_anchor_changes_only_new_synthetic_features(self):
        base = {"x": torch.zeros(4, 3), "text_emb": torch.ones(4, 2),
                "texts": ["a", "b", "c", "d"], "mass": torch.ones(4),
                "depth": torch.zeros(4, dtype=torch.long),
                "stage": torch.zeros(4, dtype=torch.long),
                "edge_index": torch.tensor([[0, 1], [1, 0]])}
        candidate = {key: value.clone() if isinstance(value, torch.Tensor) else list(value)
                     for key, value in base.items()}
        candidate["x"][:] = 2
        anchored = R7.apply_x_anchor(candidate, base, old_rows=2, lam=.25)
        self.assertTrue(torch.equal(anchored["x"][:2], candidate["x"][:2]))
        torch.testing.assert_close(anchored["x"][2:], torch.full((2, 3), .5))
        for key in ("text_emb", "mass", "depth", "stage", "edge_index"):
            self.assertTrue(torch.equal(anchored[key], candidate[key]))
        self.assertEqual(anchored["texts"], candidate["texts"])

    def test_assignment_objective_and_gradient_match_dense_reference(self):
        generator = torch.Generator().manual_seed(813)
        raw = torch.rand(9, 9, generator=generator, dtype=torch.float64)
        raw = (raw + raw.T) / 2
        raw = raw + torch.eye(9, dtype=raw.dtype)
        raw_sparse = raw.to_sparse().coalesce()
        adj = from_sparse(raw_sparse, edge_limit=7)
        h = torch.randn(9, 12, generator=generator, dtype=torch.float64)
        initial = torch.rand(7, 3, generator=generator, dtype=torch.float64)
        initial = initial / initial.sum(1, keepdim=True)
        p1 = initial.clone().requires_grad_()
        p2 = initial.clone().requires_grad_()
        cfg = Shared.Config()
        sparse_loss, _, sparse_projected = R7.objective(p1, initial, h, adj, 2, cfg)
        dense_loss, _, dense_projected = DenseR7.dense_objective(
            p2, initial, h, raw, 2, cfg)
        torch.testing.assert_close(sparse_loss, dense_loss, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(sparse_projected, dense_projected, atol=1e-10, rtol=1e-10)
        sparse_grad, = torch.autograd.grad(sparse_loss, p1)
        dense_grad, = torch.autograd.grad(dense_loss, p2)
        torch.testing.assert_close(sparse_grad, dense_grad, atol=1e-9, rtol=1e-9)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_repeated_forward_and_backward_are_bitwise_identical(self):
        device = "cuda"
        torch.use_deterministic_algorithms(True)
        matrix = self._matrix(device, torch.float32)
        segmented = from_sparse(matrix, edge_limit=4)
        x = torch.randn(5, 13, device=device, requires_grad=True)
        outputs, gradients = [], []
        for _ in range(4):
            outputs.append((segmented @ x).detach().clone())
            grad, = torch.autograd.grad((segmented @ x).square().sum(), x, retain_graph=True)
            gradients.append(grad)
        torch.cuda.synchronize()
        for value in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], value))
        for value in gradients[1:]:
            self.assertTrue(torch.equal(gradients[0], value))


if __name__ == "__main__":
    unittest.main()
