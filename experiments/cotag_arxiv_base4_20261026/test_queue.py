"""Regression tests for warning polarity and the larger-base registration."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('base4_queue', Path(__file__).with_name('queue.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class ValidationTests(unittest.TestCase):
    def checks(self):
        return dict(budget_ok=True, prefix_ok=True, finite_ok=True,
            numerical_warning=False, sinkhorn_feasible=True,
            sinkhorn_block_error=9.989738464355469e-5,
            sinkhorn_full_error=9.72747802734375e-5,
            sinkhorn_solver_tolerance=1e-4, full_matrix_rounding_allowance=9.5367431640625e-7)

    def test_no_warning_is_success(self):
        queue.validate_checks(self.checks())

    def test_real_warning_fails(self):
        with self.assertRaises(AssertionError):
            queue.validate_checks(dict(self.checks(), numerical_warning=True))

    def test_failed_invariant_fails(self):
        for key in ['budget_ok', 'prefix_ok', 'finite_ok', 'sinkhorn_feasible']:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                queue.validate_checks(dict(self.checks(), **{key: False}))

    def test_excessive_error_fails(self):
        with self.assertRaises(AssertionError):
            queue.validate_checks(dict(self.checks(), sinkhorn_block_error=2e-4))

    def test_nonfinite_error_fails(self):
        with self.assertRaises(AssertionError):
            queue.validate_checks(dict(self.checks(), sinkhorn_full_error=float('nan')))

    def test_baseline_invariants(self):
        queue.validate_checks(dict(budget_ok=True, prefix_ok=True, finite_ok=True))

    def test_larger_base(self):
        self.assertEqual(queue.CLASS_COUNTS, [16, 6, 6, 6, 6])
        self.assertEqual(sum(queue.CLASS_COUNTS), 40)


if __name__ == '__main__':
    unittest.main()
