"""CPU-only contract tests for the minimal TGC-R revision gate."""
from __future__ import annotations

import unittest

import torch

import tgc_revision_gate as T


def block(n: int, offset: float = 0.0, stage: int = 0) -> dict:
    x = torch.arange(offset, offset + n * 4, dtype=torch.float32).reshape(n, 4)
    text = torch.arange(offset + 100, offset + 100 + n * 384,
                        dtype=torch.float32).reshape(n, 384)
    return {
        "x": x,
        "text_emb": text,
        "texts": [f"text-{stage}-{i}" for i in range(n)],
        "mass": torch.arange(1, n + 1, dtype=torch.float32),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.full((n,), stage, dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "current_delta_ids": torch.arange(n, dtype=torch.long),
    }


class TGCRevisionGateTests(unittest.TestCase):
    def test_empty_delta_is_noop(self):
        old = block(6)
        revised, audit = T.revise_old_memory(
            old, block(0, stage=1), T.RevisionConfig(), "cpu")
        self.assertEqual(audit["updated_rows"], [])
        self.assertTrue(torch.equal(old["x"], revised["x"]))

    def test_zero_radius_is_allowed_and_noop(self):
        old = block(8)
        cfg = T.RevisionConfig(protected_fraction=0.5,
                               revision_fraction_of_old=0.5,
                               revision_radius=0.0,
                               teacher_steps=1,
                               teacher_batch=4)
        revised, audit = T.revise_old_memory(old, block(4, 1000, 1), cfg, "cpu")
        T.assert_revision_invariants(old, revised, audit, cfg)
        self.assertEqual(audit["updated_rows"], [])

    def test_revision_preserves_immutable_fields_and_budget(self):
        old = block(8)
        cfg = T.RevisionConfig(protected_fraction=0.5,
                               revision_fraction_of_old=0.5,
                               revision_radius=0.05,
                               teacher_steps=2,
                               teacher_batch=4,
                               preserve_tolerance=1.0)
        revised, audit = T.revise_old_memory(old, block(4, 1000, 1), cfg, "cpu")
        T.assert_revision_invariants(old, revised, audit, cfg)
        self.assertEqual(len(old["x"]), len(revised["x"]))
        self.assertTrue(torch.equal(old["text_emb"], revised["text_emb"]))
        self.assertTrue(torch.equal(old["edge_index"], revised["edge_index"]))
        self.assertLessEqual(len(audit["updated_rows"]),
                             audit["revision_quota"])
        for row in audit["protected_rows"]:
            self.assertTrue(torch.equal(old["x"][row], revised["x"][row]))

    def test_support_contains_only_old_memory_and_current_delta(self):
        old = block(3)
        delta = block(2, 200, 1)
        support = T._concat_support(old, delta)
        self.assertEqual(len(support["x"]), 5)
        self.assertEqual(support["old_rows"], 3)
        self.assertEqual(support["current_delta_ids"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
