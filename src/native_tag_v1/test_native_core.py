"""Small, meaningful CPU checks; no dataset downloads or real training."""

import io
import math
import unittest

import torch
from torch.nn import functional as F

from native_core import (GCNLayer, asm_loss, clean_data, condense_asm, create_model,
                         epoch_batches, evaluate_logits, normalized_adjacency,
                         select_reference_rows, symmetric_infonce, train_model)


def tiny_data(n=7, offset=0):
    generator = torch.Generator().manual_seed(71 + offset)
    nodes = torch.arange(n)
    edge = torch.stack((torch.cat((nodes, nodes.roll(1))),
                        torch.cat((nodes.roll(1), nodes))))
    return {"x": torch.randn(n, 5, generator=generator), "edge_index": edge,
            "text_emb": torch.randn(n, 8, generator=generator),
            "texts": [f"actual selected text {offset + i}" for i in range(n)],
            "weights": torch.ones(n), "origin_ids": nodes + offset,
            "stage": torch.zeros(n, dtype=torch.long)}


class NativeCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_gcn_matches_explicit_formula_and_bias_location(self):
        edge = torch.tensor([[0, 1, 1, 1, 2, 2], [1, 0, 0, 1, 2, 3]])
        x = torch.tensor([[1., 2.], [3., 1.], [-1., 2.], [2., 0.]], dtype=torch.double)
        raw = torch.zeros(4, 4, dtype=x.dtype)
        for source, target in edge.T.tolist():
            if source != target:
                raw[target, source] += 1
        raw += torch.eye(4, dtype=x.dtype)
        inv = raw.sum(1).pow(-0.5)
        expected_adj = inv[:, None] * raw * inv[None, :]
        adj = normalized_adjacency(edge, 4, dtype=x.dtype)
        torch.testing.assert_close(adj.to_dense(), expected_adj)
        layer = GCNLayer(2, 3).double()
        with torch.no_grad():
            layer.bias.copy_(torch.tensor([0.3, -0.2, 0.9]))
        expected = expected_adj @ F.linear(x, layer.lin.weight) + layer.bias
        torch.testing.assert_close(layer(x, adj), expected)

    def test_gcn_matches_pyg_when_available(self):
        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            self.skipTest("PyG unavailable; explicit GCN formula is independently checked")
        data = tiny_data()
        data["edge_index"] = torch.tensor([[0, 1, 1, 1, 2, 2], [1, 0, 0, 1, 2, 3]])
        ours = GCNLayer(5, 4)
        theirs = GCNConv(5, 4)
        with torch.no_grad():
            theirs.lin.weight.copy_(ours.lin.weight)
            theirs.bias.copy_(ours.bias)
        actual = ours(data["x"], normalized_adjacency(data["edge_index"], 7))
        torch.testing.assert_close(actual, theirs(data["x"], data["edge_index"]), atol=1e-6, rtol=1e-5)

    def test_model_and_asm_match_released_formulas(self):
        data = tiny_data()
        torch.manual_seed(2)
        model = create_model(5, 8, 6, 4).eval()
        projected = model.text_encoder.projection(data["text_emb"])
        text = model.text_encoder.layer_norm(projected + model.text_encoder.fc(F.gelu(projected)))
        text = text / (text.norm(dim=1, keepdim=True) + 1e-10)
        graph = model.encode_graph(data)
        torch.testing.assert_close(model(data), graph @ text.T / .07)
        target = torch.randn(7, 7)
        expected = (F.binary_cross_entropy(model(data).softmax(1), target.softmax(1)) +
                    F.binary_cross_entropy(model(data).softmax(0), target.softmax(0))) / 2
        torch.testing.assert_close(asm_loss(model(data), target), expected)

    def test_tail_batch_coverage_and_training(self):
        for n, batch_size in ((7, 3), (8, 3), (2, 20), (10, 3)):
            batches = epoch_batches(n, batch_size, torch.Generator().manual_seed(5))
            self.assertEqual(sorted(torch.cat(batches).tolist()), list(range(n)))
            self.assertTrue(all(len(batch) >= 2 for batch in batches))
        _, logs = train_model(tiny_data(), 3, 2, .001, .001, 3, "cpu", hidden_dim=6, out_dim=4)
        self.assertEqual(logs[-1]["unique_rows_seen"], 7)
        self.assertEqual(logs[-1]["examples_seen"], 7)

    def test_weighted_loss_uniform_equivalence(self):
        logits = torch.randn(4, 4)
        torch.testing.assert_close(symmetric_infonce(logits), symmetric_infonce(logits, torch.ones(4)))

    def test_asm_has_nonzero_outer_gradient_and_preserves_fixed_inputs(self):
        source, initial = tiny_data(7), tiny_data(4, 100)
        source_copy, initial_copy = clean_data(source), clean_data(initial)
        torch.manual_seed(4)
        teacher = create_model(5, 8, 6, 4).eval()
        teacher_state = {k: v.clone() for k, v in teacher.state_dict().items()}
        result, logs = condense_asm(source, initial, teacher,
                                   {"outer_steps": 2, "inner_steps": 2, "batch_size": 3,
                                    "reference_size": 5, "outer_lr": .01}, 5, "cpu")
        outer = [row for row in logs if row["event"] == "outer"]
        self.assertEqual(len(outer), 2)
        self.assertTrue(all(row["x_grad_l2"] > 0 for row in outer))
        self.assertFalse(torch.equal(result["x"], initial["x"]))
        for key in source_copy:
            if isinstance(source_copy[key], torch.Tensor):
                torch.testing.assert_close(source[key], source_copy[key], rtol=0, atol=0)
                torch.testing.assert_close(initial[key], initial_copy[key], rtol=0, atol=0)
            else:
                self.assertEqual(source[key], source_copy[key])
                self.assertEqual(initial[key], initial_copy[key])
        for key in initial_copy:
            if key != "x":
                if isinstance(initial_copy[key], torch.Tensor):
                    torch.testing.assert_close(result[key], initial_copy[key], rtol=0, atol=0)
                else:
                    self.assertEqual(result[key], initial_copy[key])
        for key, value in teacher_state.items():
            torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)

    def test_validation_selects_initial_without_test_or_source_payload(self):
        source, initial = tiny_data(), tiny_data(4, 100)
        source["y"] = object()
        initial["y"] = object()
        initial["metadata"] = {"illegal_source": source}
        teacher = create_model(5, 8, 6, 4).eval()
        visited = []

        def validator(candidate, step):
            visited.append(step)
            self.assertNotIn("y", candidate)
            return {"score": 1.0 - step}

        result, logs = condense_asm(source, initial, teacher,
                                   {"outer_steps": 2, "inner_steps": 1, "batch_size": 3,
                                    "eval_every": 1, "source_texts": source["texts"],
                                    "source": source}, 5, "cpu", validator)
        self.assertEqual(visited, [0, 1, 2])
        self.assertEqual(result["metadata"]["selected_outer_step"], 0)
        torch.testing.assert_close(result["x"], initial["x"], rtol=0, atol=0)
        buffer = io.BytesIO()
        torch.save(result, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        self.assertEqual(set(loaded), set(clean_data(initial)) | {"metadata"})
        self.assertNotIn("illegal_source", loaded["metadata"])
        self.assertNotIn("source_texts", loaded["metadata"]["config"])
        self.assertNotIn("source", loaded["metadata"]["config"])
        self.assertFalse(loaded["metadata"]["source_payload_embedded"])

    def test_protected_rows_remain_exact_under_updates_and_norm_projection(self):
        initial = tiny_data(4, 100)
        teacher = create_model(5, 8, 6, 4).eval()
        mask = torch.tensor([False, False, True, True])
        result, logs = condense_asm(tiny_data(), initial, teacher,
                                   {"outer_steps": 2, "inner_steps": 2, "batch_size": 3,
                                    "feature_norm_cap": .1, "optimize_rows": mask}, 15, "cpu")
        torch.testing.assert_close(result["x"][:2], initial["x"][:2], rtol=0, atol=0)
        self.assertFalse(torch.equal(result["x"][2:], initial["x"][2:]))
        self.assertTrue(bool((result["x"][2:].norm(dim=1) <= .10001).all()))
        self.assertEqual(result["metadata"]["optimized_x_rows"], mask.tolist())
        self.assertTrue(all(row["x_grad_l2"] > 0 for row in logs if row["event"] == "outer"))
        with self.assertRaisesRegex(ValueError, "boolean row mask"):
            condense_asm(tiny_data(), initial, teacher,
                         {"outer_steps": 0, "optimize_rows": [0, 1]}, 15, "cpu")

    def test_bounded_learning_rates_and_independent_classification(self):
        teacher = create_model(5, 8, 6, 4).eval()
        result, logs = condense_asm(tiny_data(), tiny_data(4, 100), teacher,
                                   {"outer_steps": 1, "batch_size": 3, "inner_lr_mode": "bounded",
                                    "inner_lr_bounds": (.0001, .02)}, 9, "cpu")
        row = next(row for row in logs if row["event"] == "outer")
        self.assertTrue(.0001 < row["inner_lr_graph"] < .02)
        self.assertTrue(.0001 < row["inner_lr_text"] < .02)
        student, _ = train_model(result, 11, 2, .001, .001, 3, "cpu", hidden_dim=6, out_dim=4)
        logits = evaluate_logits(student, tiny_data(), torch.randn(3, 8))
        self.assertEqual(logits.shape, (7, 3))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_unbounded_learning_rates_receive_outer_gradient(self):
        teacher = create_model(5, 8, 6, 4).eval()
        result, logs = condense_asm(tiny_data(), tiny_data(4, 100), teacher,
                                   {"outer_steps": 2, "batch_size": 3,
                                    "inner_lr_mode": "unbounded"}, 12, "cpu")
        outer = [row for row in logs if row["event"] == "outer"]
        self.assertEqual(len(outer), 2)
        self.assertNotEqual(outer[-1]["inner_lr_graph"], .005)
        self.assertNotEqual(outer[-1]["inner_lr_text"], .005)
        self.assertEqual(result["metadata"]["inner_lr_graph"], outer[-1]["inner_lr_graph"])

    def test_evaluation_adjacency_casts_to_model_dtype(self):
        model = create_model(5, 8, 6, 4).eval()
        data = tiny_data()
        data["x"] = data["x"].double()
        data["text_emb"] = data["text_emb"].double()
        logits = evaluate_logits(model, data, torch.randn(2, 8, dtype=torch.double))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.shape, (7, 2))

    def test_reference_sampling_covers_all_rows_and_retained_stages(self):
        source = tiny_data(10)
        source["stage"] = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        source["weights"] = torch.tensor([20., 20., 1., 1., 1., 1., 1., 1., 1., 1.])
        for policy in ("uniform", "mass_stratified"):
            rows, audit = select_reference_rows(source, 50, torch.Generator().manual_seed(7), policy)
            self.assertEqual(sorted(rows.tolist()), list(range(10)))
            self.assertEqual(audit["source_mass"], 48.)
            self.assertEqual(audit["selected_mass"], 48.)
            self.assertTrue(audit["all_source_stages_covered"])
        rows, audit = select_reference_rows(source, 6, torch.Generator().manual_seed(7), "mass_stratified", mass_weighted=True)
        self.assertEqual(len(rows.unique()), 6)
        self.assertTrue(audit["all_source_stages_covered"])
        self.assertEqual(audit["stage_summary"][0]["selected_rows"], 2)
        self.assertAlmostEqual(sum(row["query_loss_fraction"] for row in audit["stage_summary"]), 1.)
        again, _ = select_reference_rows(source, 6, torch.Generator().manual_seed(7), "mass_stratified")
        self.assertTrue(torch.equal(rows, again))
        with self.assertRaisesRegex(ValueError, "too small"):
            select_reference_rows(source, 2, torch.Generator().manual_seed(7), "mass_stratified")

    def test_reference_quota_redistributes_mass_after_capacity_saturation(self):
        source = tiny_data(15)
        source["stage"] = torch.tensor([0, 0] + [1] * 10 + [2] * 3)
        source["weights"] = torch.tensor([10000., 10000.] + [100.] * 10 + [1.] * 3)
        _, audit = select_reference_rows(source, 8, torch.Generator().manual_seed(7), "mass_stratified")
        # Heavy stage zero caps at its two rows; residual allocation follows
        # the 1000:3 masses, while stage two retains its required one row.
        self.assertEqual([row["selected_rows"] for row in audit["stage_summary"]], [2, 5, 1])

    def test_reference_mass_weighting_matches_uniform_and_manual_query_formula(self):
        torch.manual_seed(6)
        logits = torch.randn(5, 5, requires_grad=True)
        target = torch.randn(5, 5)
        torch.testing.assert_close(asm_loss(logits, target), asm_loss(logits, target, torch.ones(5)))
        weights = torch.tensor([20., 3., 1., 2., 1.])
        row = F.binary_cross_entropy(logits.softmax(1), target.softmax(1), reduction="none").mean(1)
        col = F.binary_cross_entropy(logits.softmax(0), target.softmax(0), reduction="none").mean(0)
        expected = ((row + col) * weights).sum() / (2 * weights.sum())
        actual = asm_loss(logits, target, weights)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(actual, logits)[0]).all()))

    def test_mass_reference_probe_preserves_fixed_targets(self):
        source = tiny_data(7)
        source["stage"] = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        source["weights"][:2] = 10
        teacher = create_model(5, 8, 6, 4).eval()
        result, logs = condense_asm(source, tiny_data(4, 100), teacher,
                                   {"outer_steps": 1, "batch_size": 3, "reference_size": 4,
                                    "reference_policy": "mass_stratified", "reference_mass_weighted": True}, 15, "cpu")
        audit = result["metadata"]["reference_audit"]
        self.assertTrue(audit["all_source_stages_covered"])
        self.assertTrue(audit["reference_mass_weighted"])
        self.assertEqual(audit["selected_count"], 4)
        self.assertGreater(next(row["x_grad_l2"] for row in logs if row["event"] == "outer"), 0)

    def test_text_anchor_logs_components_and_freezes_teacher(self):
        source, initial = tiny_data(7), tiny_data(4, 100)
        initial["stage"] = torch.tensor([0, 0, 1, 1])
        mask = initial["stage"] == 1
        torch.manual_seed(4)
        teacher = create_model(5, 8, 6, 4).train()
        teacher_state = {key: value.clone() for key, value in teacher.state_dict().items()}
        result, logs = condense_asm(
            source, initial, teacher,
            {"outer_steps": 2, "inner_steps": 2, "batch_size": 3,
             "reference_size": 5, "outer_lr": .01, "optimize_rows": mask,
             "text_anchor_weight": .2, "text_anchor_scope": "optimized"},
            5, "cpu")
        outer = [row for row in logs if row["event"] == "outer"]
        self.assertEqual(len(outer), 2)
        for row in outer:
            self.assertTrue(math.isfinite(row["asm_loss"]))
            self.assertTrue(math.isfinite(row["text_anchor_loss"]))
            self.assertAlmostEqual(row["total_loss"],
                                   row["asm_loss"] + .2 * row["text_anchor_loss"], places=6)
        self.assertEqual(result["metadata"]["text_anchor"]["pair_count"], 2)
        self.assertEqual(result["metadata"]["text_anchor"]["scope"], "optimized")
        torch.testing.assert_close(result["x"][:2], initial["x"][:2], rtol=0, atol=0)
        self.assertTrue(teacher.training)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        for key, value in teacher_state.items():
            torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)

    def test_zero_weight_text_anchor_preserves_legacy_numerics(self):
        source, initial = tiny_data(7), tiny_data(4, 100)
        torch.manual_seed(4)
        teacher = create_model(5, 8, 6, 4).eval()
        base = {"outer_steps": 2, "inner_steps": 2, "batch_size": 3,
                "reference_size": 5, "outer_lr": .01}
        legacy, legacy_logs = condense_asm(source, initial, teacher, base, 5, "cpu")
        disabled, disabled_logs = condense_asm(
            source, initial, teacher,
            {**base, "text_anchor_weight": 0.0, "text_anchor_scope": "optimized"},
            5, "cpu")
        torch.testing.assert_close(disabled["x"], legacy["x"], rtol=0, atol=0)
        legacy_outer = [row for row in legacy_logs if row["event"] == "outer"]
        disabled_outer = [row for row in disabled_logs if row["event"] == "outer"]
        self.assertEqual([row["asm_loss"] for row in disabled_outer],
                         [row["asm_loss"] for row in legacy_outer])
        self.assertTrue(all(row["text_anchor_loss"] is None for row in disabled_outer))
        self.assertTrue(all(row["total_loss"] == row["asm_loss"] for row in disabled_outer))
        self.assertEqual(disabled["metadata"]["algorithm"], "native-tagsam-asm-x-only-v1")
        self.assertNotIn("text_anchor", disabled["metadata"])

    def test_text_anchor_configuration_validation(self):
        source, initial = tiny_data(7), tiny_data(4, 100)
        teacher = create_model(5, 8, 6, 4).eval()
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            condense_asm(source, initial, teacher,
                         {"outer_steps": 0, "text_anchor_weight": -1}, 5, "cpu")
        with self.assertRaisesRegex(ValueError, "all or optimized"):
            condense_asm(source, initial, teacher,
                         {"outer_steps": 0, "text_anchor_scope": "new"}, 5, "cpu")
        with self.assertRaisesRegex(ValueError, "at least two"):
            condense_asm(source, initial, teacher,
                         {"outer_steps": 0, "text_anchor_weight": .1,
                          "text_anchor_scope": "optimized",
                          "optimize_rows": torch.tensor([False, False, False, True])},
                         5, "cpu")


if __name__ == "__main__":
    unittest.main()
