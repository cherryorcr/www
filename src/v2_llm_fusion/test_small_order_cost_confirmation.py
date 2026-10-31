import unittest
from unittest.mock import patch

import torch
import run_small_order_cost_confirmation as R


class ConfirmationTests(unittest.TestCase):
    def test_orders_are_fixed_partitions(self):
        for dataset, n in [("cora", 7), ("citeseer", 6)]:
            orders = R.stage_orders(dataset)
            self.assertEqual(orders, R.stage_orders(dataset))
            self.assertEqual(len({str(x) for x in orders.values()}), 3)
            for groups in orders.values():
                self.assertEqual(sorted(sum(groups, [])), list(range(n)))
                self.assertEqual([len(g) for g in groups], [2, 2, n - 4])

    def test_selector_is_frozen(self):
        a = R.v4_args()
        self.assertEqual(a.selection_seeds, [300, 301, 302])
        self.assertEqual(a.selection_batches, [8, 16, 32])
        self.assertEqual(a.response_marginal_rounds, 2)
        self.assertEqual(a.response_min_horizon_gain, 0)

    def test_all_stage_globals_change_together(self):
        data = {"y": torch.arange(6).repeat_interleave(200),
                "pool_mask": torch.ones(1200, dtype=torch.bool)}
        with patch.object(R.G, "STAGE_CLASSES", R.G.STAGE_CLASSES):
            groups = [[5, 1], [2, 4], [0, 3]]
            ids, budgets = R.set_stages(data, groups)
            self.assertEqual(R.SCA.G.STAGE_CLASSES,
                             [[1, 5], [2, 4], [0, 3]])
            self.assertEqual(budgets, [20, 40, 60])
            self.assertEqual(len(torch.unique(torch.cat(ids))), 1200)

    def test_invalid_partition_rejected(self):
        data = {"y": torch.arange(6).repeat_interleave(200),
                "pool_mask": torch.ones(1200, dtype=torch.bool)}
        with self.assertRaises(ValueError):
            R.set_stages(data, [[0, 1], [2, 3], [4, 4]])

    def test_payload_counts_all_tensor_fields(self):
        mem = {"x": torch.zeros(2, 4), "edge_index": torch.tensor([[0, 1], [1, 0]]),
               "text_emb": torch.zeros(2, 4), "texts": ["a", "bc"]}
        stat = R.payload_stats(mem)
        self.assertEqual(stat["logical_payload_bytes"], 32 + 32 + 32 + 3)
        self.assertEqual(stat["edge_entries"], 2)


class RegressionTests(unittest.TestCase):
    def test_raw_numerical_drift_is_recorded_but_inputs_remain_strict(self):
        row = {"support_sha256": "same", "steps": 50, "rows": 100,
               "start_state_sha256": "same", "end_state_sha256": "new", "seed": 100,
               "metrics": {"all": {"cumulative_f1": 42.}}}
        ref = dict(row, end_state_sha256="old")
        self.assertFalse(R.check_archived_fit(row, ref, "raw_equal", 1)["end_state_bitwise_equal"])
        with self.assertRaises(AssertionError):
            R.check_archived_fit(row, ref, "v4", 1)
        with self.assertRaises(AssertionError):
            R.check_archived_fit(row, dict(ref, support_sha256="other"), "raw_equal", 1)


if __name__ == "__main__":
    unittest.main()
