import unittest

import numpy as np

from semantic_topology_controls import budget_knn, local_cycles


def score(n, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x @ x.T


class SemanticControlsTest(unittest.TestCase):
    def test_equal_budget_and_determinism(self):
        for n in range(1, 40):
            for seed in range(4):
                s = score(n, seed)
                for selector in (budget_knn, local_cycles):
                    p, info = selector(s)
                    self.assertEqual(p.shape, (2, min(n, n * (n - 1) // 2)))
                    self.assertTrue(np.array_equal(p, selector(s)[0]))
                    self.assertEqual(len(set(map(tuple, p.T))), p.shape[1])
                    self.assertTrue((p[0] < p[1]).all())
                    if n > 1:
                        self.assertTrue((np.bincount(p.ravel(), minlength=n) >= 1).all())
                    if selector == local_cycles and n >= 3:
                        self.assertTrue((np.bincount(p.ravel(), minlength=n) == 2).all())

    def test_cycles_allow_disconnection(self):
        s = np.zeros((6, 6))
        s[:3, :3] = s[3:, 3:] = 1
        p, info = local_cycles(s)
        self.assertTrue(((p[0] < 3) == (p[1] < 3)).all())
        self.assertEqual(info["repairs"], 0)

    def test_singleton_repair(self):
        s = np.zeros((4, 4)); s[:3, :3] = 1
        p, info = local_cycles(s)
        self.assertEqual(info["repairs"], 1)
        self.assertTrue((np.bincount(p.ravel(), minlength=4) == 2).all())

    def test_pair_repair(self):
        s = np.zeros((5, 5)); s[:3, :3] = 1; s[3:, 3:] = 2
        p, info = local_cycles(s)
        self.assertEqual(info["repairs"], 1)
        self.assertTrue((np.bincount(p.ravel(), minlength=5) == 2).all())

    def test_knn_subset_and_mandatory(self):
        s = score(32); ranked = s.copy(); np.fill_diagonal(ranked, -np.inf)
        neighbors = np.argsort(-ranked, axis=1, kind="stable")[:, :2]
        required = {tuple(sorted((i, int(row[0])))) for i, row in enumerate(neighbors)}
        allowed = {tuple(sorted((i, int(j)))) for i, row in enumerate(neighbors) for j in row}
        selected = set(map(tuple, budget_knn(s)[0].T))
        self.assertTrue(required <= selected <= allowed)

    def test_ties_and_invalid(self):
        for n in (3, 4, 5, 12):
            for selector in (budget_knn, local_cycles):
                p, _ = selector(np.zeros((n, n)))
                self.assertEqual(p.shape[1], n)
        for selector in (budget_knn, local_cycles):
            with self.assertRaises(ValueError): selector(np.ones((2, 3)))
            with self.assertRaises(ValueError): selector(np.full((2, 2), np.nan))


if __name__ == "__main__":
    unittest.main()
