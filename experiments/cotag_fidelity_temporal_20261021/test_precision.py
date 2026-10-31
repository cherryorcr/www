import unittest
import common as C
from common import R, torch
import solver_precision


class PrecisionTests(unittest.TestCase):
    def test_value_and_gradient_agree_at_a_resolved_small_problem(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        x = torch.randn(24, 6, requires_grad=True)
        masses = torch.full((6,), 4.)
        original = R.UQ._log_sinkhorn
        before = original(x, masses, 30, 1e-4, 4096)
        g0 = torch.autograd.grad(before.square().sum(), x)[0]
        solver_precision.install(R.UQ)
        try:
            after = R.UQ._log_sinkhorn(x, masses, 30, 1e-4, 4096)
            g1 = torch.autograd.grad(after.square().sum(), x)[0]
            self.assertEqual(R.UQ._cotag_precision_calls, 1)
        finally:
            R.UQ._log_sinkhorn = original
        self.assertEqual(after.dtype, x.dtype)
        self.assertLess(float((before-after).abs().max()), 1e-6)
        self.assertLess(float((g0-g1).abs().max()), 1e-6)
        self.assertLess(float((after.sum(1)-1).abs().max()), 1e-4)

    def test_audit_has_double_precision_error_floor(self):
        original = R.UQ._log_sinkhorn
        solver_precision.install(R.UQ)
        audits = []
        try:
            x = torch.randn(24, 6)
            # An exact unit column target permits a much stricter double audit.
            R.UQ._log_sinkhorn(x, torch.full((6,), 4.), 64, 1e-12, 4096, audits)
        finally:
            R.UQ._log_sinkhorn = original
        self.assertLess(audits[-1]['row_error'], 1e-12)
        self.assertLess(audits[-1]['column_relative_error'], 1e-12)


if __name__ == '__main__':
    unittest.main()
