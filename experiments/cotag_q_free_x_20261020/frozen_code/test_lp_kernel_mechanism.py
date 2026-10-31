import unittest
from unittest.mock import patch

import torch

import run_lp_kernel_mechanism as M
import safe_joint_kernel_append as S


def block(n, stage):
    generator = torch.Generator().manual_seed(42 + stage)
    return {"x": torch.randn(n, 8, generator=generator),
            "text_emb": torch.randn(n, 8, generator=generator),
            "mass": torch.ones(n), "depth": torch.zeros(n).long(),
            "stage": torch.full((n,), stage).long(), "texts": [f"synthetic {stage} {i}" for i in range(n)]}


class MechanismTests(unittest.TestCase):
    def test_no_signature_required_and_old_blocks_frozen(self):
        for arm in M.SPECS:
            old = M.append_control(None, block(7, 0), arm)
            new = M.append_control(old, block(9, 1), arm)
            M.check_memory(new, new, old, arm)
            self.assertNotIn("graph_signature", new)
            self.assertTrue(torch.equal(old["x"], new["x"][:7]))

    def test_illegal_signature_rejected(self):
        incoming = block(7, 0)
        incoming["graph_signature"] = torch.randn(7, 4)
        with self.assertRaises(AssertionError):
            M.append_control(None, incoming, "xt_degree")

    def test_exact_equivalence_to_joint_when_weights_are_overridden(self):
        for arm, (weights, variant) in M.SPECS.items():
            a, b = block(6, 0), block(8, 1)
            old = M.append_control(None, a, arm)
            current = M.append_control(old, b, arm)
            a["graph_signature"] = torch.randn(6, 5)
            b["graph_signature"] = torch.randn(8, 5)
            with patch.dict(S.WEIGHTS, {variant: weights}):
                ref_old, _ = S.append_variant(None, a, 0, variant)
                ref, _ = S.append_variant(ref_old, b, 6, variant)
            self.assertTrue(torch.equal(ref_old["edge_index"], old["edge_index"]))
            self.assertTrue(torch.equal(ref["edge_index"], current["edge_index"]))

    def test_text_only_topology_ignores_x(self):
        incoming = block(10, 0)
        changed = {k: S.copy_value(v) for k, v in incoming.items()}
        changed["x"] = torch.randn_like(changed["x"]) * 100
        for arm in ("t_degree", "t_coverage"):
            self.assertTrue(torch.equal(M.append_control(None, incoming, arm)["edge_index"],
                                        M.append_control(None, changed, arm)["edge_index"]))

    def test_deterministic_and_edge_budget_for_small_blocks(self):
        for n in (1, 2, 3, 10):
            for arm in M.SPECS:
                a = M.append_control(None, block(n, 0), arm)
                b = M.append_control(None, block(n, 0), arm)
                self.assertTrue(torch.equal(a["edge_index"], b["edge_index"]))
                self.assertEqual(a["edge_index"].shape[1], 2 * min(n, n * (n - 1) // 2))
                M.check_memory(a, a, None, arm)

    def test_c0_suffix_excludes_edges_and_metadata(self):
        incoming = block(10, 0)
        incoming.update(graph_signature=torch.randn(10, 4), source_groups=[[1]], edge_index=torch.zeros(2, 3).long())
        result = M.block_from_c0(incoming, 3)
        self.assertEqual(set(result), set(M.ROW_KEYS) | {"texts"})
        self.assertTrue(torch.equal(result["x"], incoming["x"][3:]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
