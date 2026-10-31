from __future__ import annotations

import unittest

import torch

import safe_joint_kernel_append as S


def memory(n: int, offset: float = 0.0) -> dict:
    return {
        "x": torch.arange(offset, offset + n * 4, dtype=torch.float32).reshape(n, 4),
        "text_emb": torch.arange(offset + 100, offset + 100 + n * 3,
                                  dtype=torch.float32).reshape(n, 3),
        "graph_signature": torch.arange(offset + 200, offset + 200 + n * 5,
                                         dtype=torch.float32).reshape(n, 5),
        "mass": torch.ones(n),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.zeros(n, dtype=torch.long),
        "texts": [f"t{i}" for i in range(n)],
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }


class SafeJointKernelTests(unittest.TestCase):
    def test_same_rows_and_old_edges(self):
        old = memory(5)
        old["edge_index"] = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        block = memory(4, 1000)
        for arm in S.VARIANTS:
            out, audit = S.append_variant(old, block, 5, arm)
            self.assertTrue(torch.equal(out["x"][:5], old["x"]))
            old_part = out["edge_index"][:, (out["edge_index"] < 5).all(0)]
            self.assertTrue(torch.equal(old_part, old["edge_index"]))
            self.assertEqual(audit["old_rows"], 5)

    def test_edge_budget_operator_stable_matches_bidir_cross(self):
        old, block = memory(6), memory(5, 1000)
        out_degree, _ = S.append_variant(old, block, 6, "degree_balanced")
        out_stable, audit = S.append_variant(old, block, 6, "operator_stable")
        self.assertEqual(out_degree["edge_index"].shape,
                         out_stable["edge_index"].shape)
        self.assertEqual(audit["cross_edges"], 2 * len(block["x"]))
        self.assertEqual(audit["cross_direction"], "new_receives_old")

    def test_component_weights(self):
        old, block = memory(4), memory(3, 1000)
        old["x"] = torch.tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
        block["x"] = torch.tensor([[.9, .1, 0., 0.],
                                   [.1, .9, 0., 0.],
                                   [0., 0., .1, .9]])
        old["text_emb"] = torch.tensor([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.],
                                         [1., 1., 0.]])
        block["text_emb"] = torch.tensor([[.1, .9, 0.],
                                          [.9, .1, 0.],
                                          [0., .1, .9]])
        old["graph_signature"] = torch.tensor([[1., 0., 0., 0., 0.],
                                               [0., 1., 0., 0., 0.],
                                               [0., 0., 1., 0., 0.],
                                               [0., 0., 0., 1., 0.]])
        block["graph_signature"] = torch.tensor([[0., 0., 1., 0., 0.],
                                                 [0., 0., 0., 1., 0.],
                                                 [1., 0., 0., 0., 0.]])
        outputs = []
        for arm in ("kernel_x", "kernel_t", "kernel_g"):
            out, _ = S.append_variant(old, block, 4, arm)
            outputs.append(out["edge_index"])
        self.assertTrue(any(not torch.equal(outputs[0], item)
                            for item in outputs[1:]))

    def test_x_plus_text_variant_is_registered_and_deterministic(self):
        old, block = memory(6), memory(5, 1000)
        first, _ = S.append_variant(old, block, 6, "kernel_xt")
        second, _ = S.append_variant(old, block, 6, "kernel_xt")
        self.assertTrue(torch.equal(first["edge_index"], second["edge_index"]))
        self.assertEqual(S.WEIGHTS["kernel_xt"], (4.0 / 7.0, 3.0 / 7.0, 0.0))

    def test_degree_balanced_hard_cap_and_edge_count(self):
        old, block = memory(7), memory(9, 1000)
        out, _ = S.append_variant(old, block, 7, "degree_balanced")
        internal = out["edge_index"][:, (out["edge_index"] >= 7).all(0)] - 7
        degrees = torch.bincount(internal[0], minlength=9)
        self.assertLessEqual(int(degrees.max()), 2)
        self.assertEqual(int(internal.shape[1]), 18)

    def test_signature_compression_is_fixed_and_smaller(self):
        block = memory(8)
        block["graph_signature"] = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
        fp16 = S.compress_signature(block, "fp16")
        sketch = S.compress_signature(block, "sketch64")
        self.assertEqual(fp16["graph_signature"].dtype, torch.float16)
        self.assertEqual(sketch["graph_signature"].shape[1], 64)
        self.assertEqual(S.compress_signature(block, "sketch64")["graph_signature"].tolist(),
                         sketch["graph_signature"].tolist())
        self.assertLess(sketch["graph_signature"].numel() * sketch["graph_signature"].element_size(),
                        block["graph_signature"].numel() * block["graph_signature"].element_size())


if __name__ == "__main__":
    unittest.main()
