"""Contract tests for the follow-up; no GPU or real dataset required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import safe_joint_kernel_append as S
import run_safe_joint_kernel_experiments as R
import diagnose_joint_kernel_topology as D
import summarize_safe_joint_kernel as H
from test_safe_joint_kernel_append import memory


def random_memory(n, seed=1):
    result = memory(n)
    generator = torch.Generator().manual_seed(seed)
    for field, dimension in (("x", 4), ("text_emb", 3), ("graph_signature", 5)):
        result[field] = torch.randn(n, dimension, generator=generator)
    return result


class Contracts(unittest.TestCase):
    def test_hard_caps_adversarial_sizes_and_determinism(self):
        for n in range(1, 17):
            for seed in (3, 21):
                block = random_memory(n, seed)
                for variant in ("degree_balanced", "coverage"):
                    edges = S.select_internal(block, S.WEIGHTS["joint_kernel"], variant)
                    count = min(n, n * (n - 1) // 2)
                    self.assertEqual(edges.shape[1], 2 * count)
                    self.assertEqual(len(set(map(tuple, edges.T.tolist()))), 2 * count)
                    self.assertLessEqual(int(torch.bincount(edges[0], minlength=n).max()), 2)
                    self.assertTrue(torch.equal(edges, S.select_internal(block, S.WEIGHTS["joint_kernel"], variant)))

    def test_cross_capacity_with_many_new_rows(self):
        old, block = random_memory(3), random_memory(19, 7)
        for arm in ("degree_balanced", "coverage"):
            out, audit = S.append_variant(old, block, 3, arm)
            cross = out["edge_index"]
            cross = cross[:, (cross[0] < 3) & (cross[1] >= 3)]
            self.assertEqual(cross.shape[1], 19)
            self.assertLessEqual(int(torch.bincount(cross[0], minlength=3).max()), 7)
            self.assertEqual(audit["cross_capacity"], 7)

    def test_operator_preserves_old_normalized_operator_exactly(self):
        old = random_memory(6)
        old["edge_index"] = S.edge_tensor([(0, 1), (1, 2), (0, 4)])
        block = random_memory(9, 22)
        out, _ = S.append_variant(old, block, 6, "operator_stable")
        joint, _ = S.append_variant(old, block, 6, "joint_kernel")
        old_adj = R.C.training_adjacency(old, "cpu").to_dense()
        new_adj = R.C.training_adjacency(out, "cpu").to_dense()
        self.assertTrue(torch.equal(old_adj, new_adj[:6, :6]))
        self.assertEqual(int(torch.count_nonzero(new_adj[:6, 6:])), 0)
        self.assertEqual(out["edge_index"].shape, joint["edge_index"].shape)
        for source in (out, joint):
            edge = source["edge_index"]
            source["internal_only"] = edge[:, (edge >= 6).all(0)]
        self.assertTrue(torch.equal(out["internal_only"], joint["internal_only"]))
        self.assertEqual(R.topology_metrics(old, out)["old_effective_message_drift"], 0.)

    def test_tiny_operator_condition_rejected_explicitly(self):
        with self.assertRaises(ValueError):
            S.append_variant(memory(1), memory(5), 1, "operator_stable")

    def test_component_cross_is_not_balanced(self):
        old, block = memory(4), memory(8)
        old["x"] = torch.eye(4)
        block["x"] = torch.tensor([[1., 0., 0., 0.]] * 8)
        out, audit = S.append_variant(old, block, 4, "kernel_x")
        edge = out["edge_index"]
        cross = edge[:, (edge[0] < 4) & (edge[1] >= 4)]
        self.assertEqual(cross[0].tolist(), [0] * 8)
        self.assertIsNone(audit["cross_capacity"])

    def test_redundancy_not_trivially_self_similarity(self):
        score = torch.eye(4)
        self.assertAlmostEqual(S.edge_redundancy(score, 0, 2, [(0, 1)]), .5)
        self.assertAlmostEqual(S.edge_redundancy(score, 2, 3, [(0, 1)]), 0.)

    def test_sketch_does_not_change_global_rng_and_freezes_old(self):
        torch.manual_seed(43)
        before = torch.random.get_rng_state()
        old = S.compress_signature(random_memory(4), "sketch64")
        block = S.compress_signature(random_memory(8, 3), "sketch64")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        out, _ = S.append_variant(old, block, 4, "joint_kernel")
        self.assertTrue(torch.equal(old["graph_signature"], out["graph_signature"][:4]))
        self.assertEqual(out["signature_config"], old["signature_config"])
        wrong = S.compress_signature(random_memory(8, 3), "fp32")
        # The frozen implementation rejects incompatible tensor widths at cat.
        with self.assertRaises(RuntimeError):
            S.append_variant(old, wrong, 4, "joint_kernel")
        wrong = dict(block, signature_config=dict(block["signature_config"], seed=0))
        with self.assertRaises(ValueError):
            S.append_variant(old, wrong, 4, "joint_kernel")

    def test_drift_gate_compares_same_previous_history(self):
        old = random_memory(4)
        old["edge_index"] = S.edge_tensor([(0, 1), (1, 2), (2, 3), (3, 0)])
        block = random_memory(6, 2)
        output, audit = R.gated_append(old, block, "cpu")
        candidate, _ = S.append_variant(old, block, 4, "joint_kernel")
        reference = R.c0_append(old, block, "cpu")
        cm, rm = R.topology_metrics(old, candidate), R.topology_metrics(old, reference)
        expected = any(cm[k] > rm[k] + 1e-8 for k in ("old_normalized_drift", "old_effective_message_drift"))
        self.assertEqual(expected, audit["gate_fallback"])
        self.assertTrue(torch.equal(output["edge_index"], (reference if expected else candidate)["edge_index"]))

    def test_internal_degree_excludes_cross_edges_and_selfloops(self):
        old, block = memory(4), memory(5)
        out, _ = S.append_variant(old, block, 4, "degree_balanced")
        metrics = R.topology_metrics(old, out)
        self.assertEqual(metrics["max_new_internal_degree"], 2)
        self.assertEqual(metrics["new_internal_degree_gini"], 0.)
        self.assertEqual(metrics["cross_unique_pairs"], 5)
        self.assertEqual(metrics["cross_edge_entries"], 10)
        self.assertEqual(metrics["max_new_degree"], 3)

    def test_summary_null_and_scalar_values(self):
        self.assertIsNone(H.value({"all": {"old_f1": None}}, "all", "old_f1"))
        self.assertEqual(H.value({"all": {"old_f1": 12.}}, "all", "old_f1"), 12.)
        self.assertEqual(H.value({"all": {"old_f1": {"mean": 13.}}}, "all", "old_f1"), 13.)

    def test_diagnostic_does_not_mix_datasets_with_same_cell_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, edges in (("cora", [(0, 1), (0, 2), (0, 3), (0, 4)]),
                                   ("citeseer", [(0, 1), (1, 2), (2, 3), (3, 4)])):
                cell = root / "matrix" / dataset / "s1_canonical_r05"
                cell.mkdir(parents=True)
                (cell / "manifest.json").write_text(json.dumps({"stage_classes": [[0]], "rate": .05}))
                (cell / "results.json").write_text(json.dumps({"results": [
                    {"arm": arm, "stage": 0, "metrics": {"all": {"cumulative_f1": {"mean": 1.}}}}
                    for arm in ("c0", "joint_kernel")]}))
                support = memory(5)
                support["edge_index"] = S.edge_tensor(edges)
                for arm in ("c0", "joint_kernel"):
                    torch.save(support, cell / f"{arm}_s0_memory.pt")
            with patch("sys.argv", ["diagnose", "--root", str(root / "matrix"), "--output", str(root / "out")]):
                D.main()
            import csv
            with (root / "out/final_topology_with_f1.csv").open() as handle:
                rows = {r["dataset"]: r for r in csv.DictReader(handle)}
            self.assertNotEqual(rows["cora"]["final_new_internal_degree_gini"],
                                rows["citeseer"]["final_new_internal_degree_gini"])


if __name__ == "__main__":
    unittest.main()
