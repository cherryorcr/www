import unittest

import report_safe_joint_kernel_v2 as R


class Reporting(unittest.TestCase):
    def test_pairs_by_seed_not_position(self):
        def row(seed, value):
            return {"seed": seed, "metrics": {"all": {"cumulative_f1": value}}}
        result = R.paired([row(101, 8), row(100, 2)], [row(100, 1), row(101, 5)], "cumulative_f1")
        self.assertEqual(result, {100: 1, 101: 3})
        with self.assertRaises(ValueError):
            R.paired([row(100, 2)], [row(101, 1)], "cumulative_f1")

    def test_cost_is_relative_to_same_cell(self):
        fields = ("logical_bytes", "serialized_bytes", "chain_build_seconds", "chain_train_mean_seconds", "chain_total_seconds")
        rows = [dict(dataset="cora", cell="a", arm="c0", **dict.fromkeys(fields, 2)),
                dict(dataset="citeseer", cell="a", arm="c0", **dict.fromkeys(fields, 4)),
                dict(dataset="cora", cell="a", arm="joint_kernel", **dict.fromkeys(fields, 3))]
        R.relative_cost(rows)
        self.assertEqual(rows[-1]["logical_bytes_ratio_c0"], 1.5)

    def test_gates_refuse_partial_matrix(self):
        with self.assertRaises(ValueError):
            R.gates([])

    def test_gates_do_not_hide_one_old_class_regression(self):
        rows = []
        numeric = ("f1", "old", "new", "f1_delta", "old_delta", "new_delta", "stage_average_f1",
                   "logical_bytes_ratio_c0", "chain_build_seconds", "chain_train_mean_seconds",
                   "chain_total_seconds", "chain_train_mean_seconds_ratio_c0", "chain_total_seconds_ratio_c0")
        for arm in R.MAIN_ARMS[2:]:
            for dataset in ("cora", "citeseer"):
                for order in ("canonical", "order19102"):
                    for i in range(4):
                        rows.append(dict.fromkeys(numeric, 1.) | dict(arm=arm, dataset=dataset,
                            order=order, cell=str(i), f1_delta=2.))
        self.assertTrue(all(g["quantitative_gates_pass"] for g in R.gates(rows)))
        rows[0]["old_delta"] = -3.
        result = R.gates(rows)
        self.assertFalse(result[0]["quantitative_gates_pass"])
        self.assertTrue(all(g["quantitative_gates_pass"] for g in result[1:]))


if __name__ == "__main__":
    unittest.main()
