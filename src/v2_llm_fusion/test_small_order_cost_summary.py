import unittest
import summarize_small_order_cost as S


class SummaryTests(unittest.TestCase):
    def fixture(self, arm):
        stages = []
        for s in range(3):
            fits = []
            for seed in range(100, 106):
                def metrics(value):
                    item = {"cumulative_f1": value, "per_task_f1": [value] * (s + 1),
                            "old_f1": value, "new_f1": value}
                    return {"all": item, "seen": dict(item)}
                fits.append({"seed": seed, "steps": 8192 if arm == "raw_long" else 50,
                    "metrics": metrics(20.), "selected_metrics": metrics(40.),
                    "start_state_sha256": f"{s-1}:{seed}", "end_state_sha256": f"{s}:{seed}"})
            stages.append({"stage": s, "rows": 10, "per_seed": fits})
        cost = {"fresh_chain_measurement": True, "checkpoint_rechecks": 18, "student_fits": 18,
            "build_chain_seconds_with_io": 3., "selector_seconds": 2., "selector_runs": 6,
            "selector_steps": 60, "student_seconds": 60., "student_wall_seconds": 66.,
            "student_steps": 900, "build_peak_allocated_mb": 100., "student_peak_allocated_mb": 120.,
            "memory_payload": [{"logical_payload_bytes": 1000, "edge_entries": 20}] * 3}
        audits = {"v4": [{"build_seconds": 4., "selector_seconds": 3., "selector_runs": 9, "selector_steps": 90}],
                  "raw_equal": [{"build_seconds": 1.}]}
        manifest = {"student_steps": [50] * 3, "budgets": [10] * 3,
                    "stage_classes": [[0, 1], [2, 3], [4, 5]]}
        return stages, cost, audits, manifest

    def test_raw_long_reports_selected_not_endpoint(self):
        stages, cost, audits, manifest = self.fixture("raw_long")
        row = S.summarize_arm("cora", "canonical", "raw_long", stages, cost, audits, {}, manifest)
        self.assertEqual(row["final_f1"], 40.)
        self.assertEqual(row["stage_average_f1"], 40.)

    def test_construction_not_divided_by_report_seeds(self):
        stages, cost, audits, manifest = self.fixture("v4")
        row = S.summarize_arm("cora", "canonical", "v4", stages, cost, audits, {}, manifest)
        self.assertEqual(row["student_seconds_one_seed_trajectory"], 10.)
        self.assertEqual(row["total_seconds_one_seed"], 14.)

    def test_finetune_initializer_cost_and_state_chain(self):
        stages, cost, audits, manifest = self.fixture("v4_init_finetune")
        row = S.summarize_arm("cora", "canonical", "v4_init_finetune", stages, cost, audits, {}, manifest)
        self.assertEqual(row["total_seconds_one_seed"], 18.)
        self.assertEqual(row["response_seconds"], 5.)
        self.assertEqual(row["selector_runs"], 15)
        stages[1]["per_seed"][0]["start_state_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            S.summarize_arm("cora", "canonical", "v4_init_finetune", stages, cost, audits, {}, manifest)

    def test_unmeasured_cached_construction_rejected(self):
        stages, cost, audits, manifest = self.fixture("v4")
        cost["fresh_chain_measurement"] = False
        with self.assertRaises(ValueError):
            S.summarize_arm("cora", "canonical", "v4", stages, cost, audits, {}, manifest)

    def test_ratio_of_order_means_not_average_ratios(self):
        rows = []
        fields = ["stage_average_seen_f1", "old_f1", "new_f1", "forgetting_pp", "build_seconds",
                  "response_seconds", "selector_runs", "selector_steps",
                  "student_seconds_one_seed_trajectory", "total_seconds_one_seed", "payload_bytes", "edges"]
        for dataset in ("cora", "citeseer"):
            for order, full, compressed in [("canonical", 20., 30.), ("order1701", 40., 30.), ("order1702", 60., 30.)]:
                for arm, value in [("v4", compressed), ("raw_long", full)]:
                    row = {"dataset": dataset, "order": order, "arm": arm,
                           "final_f1": value, "stage_average_f1": value}
                    row.update({key: 1. for key in fields})
                    rows.append(row)
        result = S.aggregate_orders(rows)
        for row in result:
            self.assertAlmostEqual(row["retention_percent"], 75. if row["arm"] == "v4" else 100.)
        self.assertEqual(rows[0]["retention_percent"], 150.)

    def test_partial_orders_not_published(self):
        rows = [{"dataset": "cora", "order": "canonical", "arm": "raw_long",
                 "final_f1": 50., "stage_average_f1": 50.}]
        with self.assertRaises(ValueError):
            S.aggregate_orders(rows)


if __name__ == "__main__":
    unittest.main()
