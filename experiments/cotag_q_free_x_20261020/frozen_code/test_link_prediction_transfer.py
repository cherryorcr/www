import unittest

import torch

import run_link_prediction_transfer as L


class LinkPredictionTests(unittest.TestCase):
    def fixture(self):
        arrival = torch.arange(36) // 12
        arrival[-2:] = -1
        edge = torch.combinations(torch.arange(36), r=2).T
        edge = edge[:, (edge.sum(0) % 4) == 0]
        return edge, arrival

    def test_split_disjoint_deterministic_and_pool_only(self):
        edge, arrival = self.fixture()
        a = L.split_edges(L.bidirectional(edge), arrival, 123)
        b = L.split_edges(edge.flip(0), arrival, 123)
        for key in a:
            if isinstance(a[key], torch.Tensor):
                self.assertTrue(torch.equal(a[key], b[key]), key)
        L.validate_split(a)

    def test_negative_contamination_rejected(self):
        edge, arrival = self.fixture()
        split = L.split_edges(edge, arrival, 123)
        split["test_neg"] = split["test_pos"].clone()
        with self.assertRaises(AssertionError):
            L.validate_split(split)

    def test_hidden_both_directions_removed_and_future_excluded(self):
        edge, arrival = self.fixture()
        split = L.split_edges(edge, arrival, 124)
        ids = torch.where((arrival >= 0) & (arrival <= 1))[0]
        train = L.localize(split["train_pos"], ids, len(arrival))
        test = L.localize(split["test_pos"], ids, len(arrival))
        self.assertFalse(L.pair_keys(train, len(ids)) & L.pair_keys(test, len(ids)))
        self.assertTrue(bool((train < len(ids)).all()))
        self.assertTrue(torch.equal(L.pairs(L.bidirectional(train), len(ids)), L.pairs(train, len(ids))))

    def test_slices_partition_and_balanced(self):
        edge, arrival = self.fixture()
        split = L.split_edges(edge, arrival, 125)
        for stage in range(3):
            ids = torch.where((arrival >= 0) & (arrival <= stage))[0]
            pos = L.localize(split["test_pos"], ids, len(arrival))
            neg = L.localize(split["test_neg"], ids, len(arrival))
            pm = L.masks(pos, arrival[ids], stage)
            nm = L.masks(neg, arrival[ids], stage)
            self.assertTrue(torch.equal(pm["old"] | pm["new"], pm["all"]))
            self.assertTrue(torch.equal(pm["cross"] | pm["within_new"], pm["new"]))
            for name in pm:
                self.assertEqual(int(pm[name].sum()), int(nm[name].sum()))

    def test_synthetic_negatives_exclude_edges_and_loops(self):
        edges = torch.tensor([[0, 1], [1, 2]])
        neg = L.synthetic_negatives(edges, 4)
        self.assertEqual(neg.shape[1], 4)
        self.assertFalse(L.pair_keys(edges, 4) & L.pair_keys(neg, 4))
        self.assertTrue(bool((neg[0] < neg[1]).all()))

    def test_metrics_known_values(self):
        pos = torch.tensor([[0, 1], [1, 2]])
        neg = torch.tensor([[0, 0], [2, 3]])
        result = L.metrics(torch.tensor([2., 3.]), torch.tensor([0., 1.]), pos, neg, torch.zeros(4).long(), 0)
        self.assertEqual(result["all"]["auc"], 1.)
        self.assertEqual(result["all"]["ap"], 1.)
        self.assertIsNone(result["old"]["auc"])

    def test_fit_checkpoint_reproduces_scores(self):
        torch.set_num_threads(1)
        x = L.features(torch.randn(6, 4), torch.randn(6, 3))
        pos = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        neg = L.synthetic_negatives(pos, 6)
        model, ps, ns, meta = L.fit(x, pos, neg, x, L.bidirectional(pos), pos, neg, 100, 2, "cpu")
        clone = L.GAE(L.Encoder(x.shape[1]))
        clone.load_state_dict(model.state_dict())
        clone.eval()
        with torch.no_grad():
            z = clone.encode(x, L.bidirectional(pos))
            self.assertTrue(torch.equal(clone.decoder(z, pos, sigmoid=False), ps))
            self.assertTrue(torch.equal(clone.decoder(z, neg, sigmoid=False), ns))
        self.assertEqual(meta["steps"], 2)


if __name__ == "__main__":
    unittest.main()
