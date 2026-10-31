"""CPU-only contracts for C0-M; no dataset or CUDA is required."""
from __future__ import annotations

import unittest

import torch

import c0m_memory_maintenance as M
import run_student_compatible_append as SCA


def block(n: int, offset: float = 0.0, stage: int = 0) -> dict:
    x = torch.arange(offset, offset + n * 4, dtype=torch.float32).reshape(n, 4)
    text = torch.arange(offset + 100, offset + 100 + n * 3,
                        dtype=torch.float32).reshape(n, 3)
    return {
        "x": x,
        "text_emb": text,
        "texts": [f"text-{stage}-{i}" for i in range(n)],
        "mass": torch.arange(1, n + 1, dtype=torch.float32),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.full((n,), stage, dtype=torch.long),
    }


class C0MMaintenanceTests(unittest.TestCase):
    def test_novelty_is_per_new_row_against_core(self):
        old = block(3)
        old["core_rows"] = torch.tensor([0, 1], dtype=torch.long)
        candidate = block(2, 100)
        values = M._novelty_scores(old, candidate)
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(torch.isfinite(values).all())

    def test_core_buffer_and_append_contracts(self):
        cfg = M.MaintenanceConfig(buffer_fraction=0.5,
                                  refresh_fraction_of_old=0.5,
                                  blend_alpha=0.25,
                                  anchor_radius=0.1)
        first = block(6, stage=0)
        old, first_audit = M.append(None, first, 0, cfg, device="cpu")
        self.assertEqual(len(old["x"]), 6)
        self.assertEqual(len(old["core_rows"]), 3)
        self.assertEqual(len(old["buffer_rows"]), 3)
        self.assertTrue(first_audit["old_old_edges_unchanged"])

        second = block(4, 1000, stage=1)
        c0 = SCA.append_memory(None, first, 0, topology_mode="frozen_old",
                               cross_degree=1, refresh_mode="frozen_old",
                               device="cpu")
        c0 = SCA.append_memory(c0, second, 6, topology_mode="frozen_old",
                               cross_degree=1, refresh_mode="frozen_old",
                               device="cpu")
        updated, audit = M.append(old, second, 6, cfg, device="cpu")
        M.assert_invariants(old, second, updated, audit, cfg)

        self.assertEqual(len(updated["x"]), 10)
        self.assertTrue(torch.equal(updated["x"][6:], second["x"]))
        self.assertTrue(torch.equal(updated["text_emb"][6:], second["text_emb"]))
        self.assertEqual(updated["texts"][6:], second["texts"])
        self.assertTrue(torch.equal(updated["core_rows"], old["core_rows"]))
        self.assertTrue(torch.equal(updated["edge_index"][:,
                              (updated["edge_index"] < 6).all(0)],
                                    old["edge_index"]))
        self.assertEqual(updated["edge_index"].shape,
                         c0["edge_index"].shape)
        self.assertLessEqual(audit["max_anchor_relative"],
                             cfg.anchor_radius + 1e-6)
        self.assertEqual(len(updated["buffer_rows"]), 7)

    def test_zero_alpha_is_content_noop_but_still_appends(self):
        cfg = M.MaintenanceConfig(buffer_fraction=0.5,
                                  refresh_fraction_of_old=1.0,
                                  blend_alpha=0.0,
                                  anchor_radius=0.25)
        old, _ = M.append(None, block(4), 0, cfg, device="cpu")
        second = block(3, 200, stage=1)
        updated, audit = M.append(old, second, 4, cfg, device="cpu")
        self.assertEqual(audit["maintenance"]["count"], len(old["buffer_rows"]))
        for row in old["buffer_rows"].tolist():
            self.assertTrue(torch.equal(updated["x"][row], old["x"][row]))
            self.assertTrue(torch.equal(updated["text_emb"][row],
                                        old["text_emb"][row]))


if __name__ == "__main__":
    unittest.main()
