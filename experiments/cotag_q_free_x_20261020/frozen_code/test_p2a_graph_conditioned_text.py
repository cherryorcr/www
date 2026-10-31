import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import p2a_graph_text as T
import p2_text_operators as BT
from test_p2_text_operators import Encoder
import run_p2a_graph_conditioned_text as R


class GraphTextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.x = torch.randn(6, 7)
        self.emb = F.normalize(torch.randn(6, 4), dim=1)
        self.clusters = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])]
        self.edges = torch.tensor([[0, 2, 1, 4], [2, 0, 4, 1]])
        self.encoder = Encoder()
        self.texts = [" ".join(f"node{i}word{j}" for j in range(100)) for i in range(6)]
        self.bank = T.snippet_bank(self.texts, self.encoder)
        self.queries = T.make_queries(self.x, self.emb, self.clusters, self.edges, 123)
        self.joint = torch.cat((F.normalize(self.x, dim=1), self.emb), dim=1)

    def test_all_arms_strict_budget_and_finite_target(self):
        for arm in T.ARMS:
            for budget in (1, 7, 32, 64):
                self.encoder.inputs = []
                text, target, audit = T.construct(arm, self.joint, self.emb, self.clusters[0],
                    self.texts, self.encoder, self.bank, self.queries, 0, budget)
                actual = sum(BT.token_count(self.encoder.tokenizer, s) for s in self.encoder.inputs)
                self.assertLessEqual(actual, budget, arm)
                self.assertLessEqual(BT.token_count(self.encoder.tokenizer, text), budget)
                self.assertEqual(actual, audit["encoder_content_tokens"])
                self.assertTrue(torch.isfinite(target).all())
                self.assertTrue(set(audit["selected_source_local_ids"]).issubset({0, 1}))

    def test_graph_pool_weights_and_selection_match(self):
        outputs = [T.construct(a, self.joint, self.emb, self.clusters[0], self.texts,
                              self.encoder, self.bank, self.queries, 0)
                   for a in ("graph_selection", "graph_pool")]
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][2]["snippet_ids"], outputs[1][2]["snippet_ids"])
        self.assertAlmostEqual(sum(outputs[1][2]["attention_weights"]), 1., places=6)

    def test_shuffle_derangement_and_determinism(self):
        again = T.make_queries(self.x, self.emb, self.clusters, self.edges, 123)
        self.assertTrue(torch.equal(again["shuffled"], self.queries["shuffled"]))
        self.assertTrue((again["permutation"] != torch.arange(3)).all())
        expected = F.normalize(again["attribute"] + again["neighbors"][again["permutation"]], dim=1)
        self.assertTrue(torch.equal(expected, again["shuffled"]))

    def test_isolation_falls_back_to_attribute(self):
        query = T.make_queries(self.x, self.emb, self.clusters, torch.empty(2, 0, dtype=torch.long), 1)
        self.assertTrue(torch.allclose(query["graph"], query["attribute"], atol=1e-6))

    def test_graph_perturbation_does_not_change_attribute(self):
        query = T.make_queries(self.x, self.emb, self.clusters, self.edges.flip(1), 123)
        self.assertTrue(torch.equal(query["graph"], self.queries["graph"]))
        query = T.make_queries(self.x, self.emb, self.clusters, torch.empty(2, 0, dtype=torch.long), 123)
        self.assertTrue(torch.equal(query["attribute"], self.queries["attribute"]))
        self.assertFalse(torch.equal(query["graph"], self.queries["graph"]))

    def test_x_and_text_dimensions_need_not_match(self):
        self.assertEqual(self.queries["graph"].shape, (3, 4))
        with self.assertRaises(ValueError):
            T.make_queries(self.x, self.emb, self.clusters, torch.tensor([[0], [6]]), 1)

    def test_source_cap_and_order_invariance(self):
        self.assertTrue(all(BT.token_count(self.encoder.tokenizer, t) <= T.SNIPPET_TOKENS
                            for t in self.bank["texts"]))
        for row, members in enumerate(self.clusters):
            left = T.select_snippets(self.bank, members, self.queries["graph"][row], self.encoder.tokenizer)
            right = T.select_snippets(self.bank, members.flip(0), self.queries["graph"][row], self.encoder.tokenizer)
            self.assertEqual(left, right)

    def test_legacy_pooling_controls_are_exact_reuse(self):
        for arm, legacy in [("center_concat", "concat"), ("mean_pool", "mean_pool"), ("text_attention", "set_pool")]:
            a = T.construct(arm, self.joint, self.emb, self.clusters[0], self.texts,
                            self.encoder, self.bank, self.queries, 0)
            b = BT.construct(self.joint, self.emb, self.clusters[0], self.texts, self.encoder, legacy)
            self.assertEqual(a[0], b[0])
            self.assertTrue(torch.equal(a[1], b[1]))

    def test_legal_delta_rejects_holdouts_and_slices_edges(self):
        data = dict(x=self.x, y=torch.arange(6), text_emb=self.emb, texts=self.texts,
                    pool_mask=torch.tensor([True, True, False, False, False, False]),
                    edge_index=self.edges)
        delta = R.legal_delta(data, torch.tensor([0, 1]))
        self.assertEqual(delta["texts"], self.texts[:2])
        self.assertEqual(delta["edge_index"].numel(), 0)
        with self.assertRaises(ValueError):
            R.legal_delta(data, torch.tensor([0, 2]))

    def test_append_preserves_every_control(self):
        def block(n, stage):
            return dict(x=torch.randn(n, 7), text_emb=torch.randn(n, 4), texts=["text"]*n,
                        mass=torch.ones(n), depth=torch.zeros(n, dtype=torch.long),
                        stage=torch.full((n,), stage), edge_index=torch.empty(2, 0, dtype=torch.long))
        old, new = block(2, 0), block(3, 1)
        reference = {k: old[k]+new[k] if k == "texts" else torch.cat((old[k], new[k])) for k in R.FIELDS}
        reference["edge_index"] = torch.tensor([[0, 2], [2, 0]])
        result = R.append_fixed_edges(old, new, reference)
        self.assertTrue(torch.equal(result["text_emb"][:2], old["text_emb"]))
        self.assertTrue(torch.equal(result["edge_index"], reference["edge_index"]))
        changed = dict(new, x=new["x"]+1)
        with self.assertRaises(AssertionError):
            R.append_fixed_edges(old, changed, reference)


if __name__ == "__main__":
    unittest.main()
