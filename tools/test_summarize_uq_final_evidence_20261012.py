"""Regression checks for matched adapted-baseline reporting."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_uq_final_evidence_20261012 import confirmation_completion, plots, summarize


def entry(category, arm, f1, accuracy):
    protocol = dict(dataset="bookhis", order_seed=9101, construction_seed=271006,
                    groups=[[0]], budgets=[5], student={"name": "common"},
                    student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
                    pool_mask_hash="pool", report_mask_hash="report", test_mask_hash="test",
                    stage_id_hashes=["stage"], stage_raw_rows=[100], memory_rate=.05,
                    budget_mode="fixed_rate_current_legal_pool", actual_final_rate=.05,
                    source_hashes={"graphtext_prompt_student.py": "student-hash"}, arms=[arm])
    scores = {}
    for split in ("report_dev", "sealed_test"):
        values = dict(final_f1=f1, accuracy=accuracy, old_f1=f1, new_f1=f1)
        scores[split] = dict(**values, student_seeds=[150, 151, 152],
                             per_student=[dict(seed=seed, **values) for seed in (150, 151, 152)])
    result = dict(protocol=protocol, status="complete", rows=[dict(
        arm=arm, stage=0, memory_rows=5, new_rows=5, checks={"budget_ok": True},
        scores=scores, audit={}, construction_seconds=1., construction_peak_mb=128.)])
    return dict(path=f"{category}/{arm}/results.json", source_sha256="a" * 64,
                bytes=42, complete_arms=[arm], errors=[], result=result)


class MatchedBaselineTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(entries=[entry("confirmation", "uq_dm_edgefree", 50., 70.),
                                      entry("baselines", "gecc_soft", 47.5, 69.)])

    def test_single_cell_has_no_confidence_interval(self):
        pairs = summarize(self.snapshot)["pairs"]
        matched = [row for row in pairs if row["category"] == "adapted_baseline"
                   and row["split"] == "sealed_test"]
        self.assertEqual({row["metric"] for row in matched},
                         {"final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1"})
        self.assertEqual(next(row["mean_delta"] for row in matched if row["metric"] == "final_f1"), 2.5)
        self.assertTrue(all(row["cells"] == 1 and row["ci_low"] is None
                            and row["ci_high"] is None for row in matched))

    def test_unmatched_seed_is_not_paired(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["entries"][1]["result"]["protocol"]["construction_seed"] = 271108
        self.assertFalse([row for row in summarize(snapshot)["pairs"]
                          if row["category"] == "adapted_baseline"])

    def test_completion_grid_includes_missing_files_and_partial_arms(self):
        snapshot = copy.deepcopy(self.snapshot)
        pending = entry("confirmation", "gecc_hard", 49., 70.)
        pending.update(path="confirmation/partial/results.json", complete_arms=[])
        pending["result"]["status"] = "running"
        snapshot["entries"].append(pending)
        grid = summarize(snapshot)["confirmation_completion"]
        self.assertEqual(len(grid), 81)
        statuses = {(r["dataset"], r["arm"], r["order_seed"], r["construction_seed"]): r for r in grid}
        self.assertEqual(statuses[("bookhis", "uq_dm_edgefree", 9101, 271006)]["status"], "complete")
        self.assertEqual(statuses[("bookhis", "gecc_hard", 9101, 271006)]["status"], "partial")
        self.assertEqual(statuses[("bookhis", "uq_dm_edgefree", 9103, 271109)]["status"], "not_started")
        self.assertEqual(sum(r["completed_student_runs"] for r in grid), 3)

    def test_completion_requires_both_score_splits(self):
        protocol = dict(dataset="arxiv", order_seed=9101, construction_seed=271006,
                        arms=["uq_dm_edgefree"])
        cells = [dict(category="confirmation", arm="uq_dm_edgefree", split="sealed_test",
                      dataset="arxiv", order_seed=9101, construction_seed=271006)]
        grid = confirmation_completion(cells, [dict(path="confirmation/arxiv/results.json", protocol=protocol)])
        row = next(r for r in grid if (r["dataset"], r["arm"], r["order_seed"], r["construction_seed"]) ==
                   ("arxiv", "uq_dm_edgefree", 9101, 271006))
        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["completed_student_runs"], 0)

    def test_nine_unregistered_cells_do_not_complete_confirmation(self):
        entries = []
        for order in (9000, 9101, 9102):
            for seed in (271006, 271108, 271109):
                item = entry("confirmation", "uq_dm_edgefree", 50., 70.)
                item["path"] = f"confirmation/o{order}_c{seed}/results.json"
                item["result"]["protocol"].update(order_seed=order, construction_seed=seed)
                entries.append(item)
        evidence = summarize(dict(entries=entries))
        self.assertTrue(all(r["cells"] == 9 for r in evidence["summary"]))
        self.assertFalse(any(r["confirmation_complete"] for r in evidence["summary"]))
        self.assertEqual(sum(r["status"] == "complete" for r in evidence["confirmation_completion"]), 6)

    def test_mismatched_common_protocol_is_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["entries"][1]["result"]["protocol"]["student_steps"] = 512
        with self.assertRaisesRegex(ValueError, "common protocol mismatch"):
            summarize(snapshot)

    def test_mismatched_legal_pool_or_student_implementation_is_rejected(self):
        for field, value in (("stage_raw_rows", [101]),
                             ("actual_final_rate", .051),
                             ("budget_mode", "reference_budget"),
                             ("source_hashes", {"graphtext_prompt_student.py": "other-hash"})):
            with self.subTest(field=field):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot["entries"][1]["result"]["protocol"][field] = value
                with self.assertRaisesRegex(ValueError, "common protocol mismatch"):
                    summarize(snapshot)

    def test_single_cell_plot_has_no_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            plots(Path(directory), summarize(self.snapshot))
            self.assertTrue((Path(directory) / "adapted_baselines.png").is_file())

    def test_inspected_stad_metadata_normalization_requires_access_audits(self):
        main = entry("confirmation", "uq_dm_edgefree", 50., 70.)
        main["result"]["protocol"].update(
            name="uq-final-frozen-confirmation-20261012-v1",
            class_floor_adjustments=[],
            evaluation_split="report_dev and sealed_test stored separately; protocol frozen before either",
            access="current-delta-only; own frozen synthetic prefix; zero historical raw reads")
        other = copy.deepcopy(main)
        other.update(path="stad/stad/results.json", complete_arms=["stad_native_local"])
        other["result"]["protocol"].update(name="stad-native-current-delta-20261012-v1",
            arms=["stad_native_local"], method_selection_from_test=False,
            access="current delta only; own frozen synthetic prefix; no historical raw reads")
        other["result"]["protocol"].pop("budget_mode")
        other["result"]["protocol"].pop("evaluation_split")
        row = other["result"]["rows"][0]
        row.update(arm="stad_native_local", checks={"budget_ok": True, "prefix_ok": True},
            audit=dict(historical_raw_reads=0, report_dev_used_for_selection=False,
                       test_used_for_selection=False))
        evidence = summarize(dict(entries=[main, other]))
        self.assertTrue(any(r["right"] == "stad_native_local" for r in evidence["pairs"]))
        row["audit"]["test_used_for_selection"] = True
        with self.assertRaisesRegex(ValueError, "selection audit"):
            summarize(dict(entries=[main, other]))

    def test_stad_v2_requires_audited_sparse_class_split(self):
        main = entry("confirmation", "uq_dm_edgefree", 50., 70.)
        main["result"]["protocol"].update(
            name="uq-final-frozen-confirmation-20261012-v1",
            class_floor_adjustments=[],
            evaluation_split="report_dev and sealed_test stored separately; protocol frozen before either",
            access="current-delta-only; own frozen synthetic prefix; zero historical raw reads")
        other = copy.deepcopy(main)
        other.update(path="stad/retry/results.json", complete_arms=["stad_native_local"])
        other["result"]["protocol"].update(name="stad-native-current-delta-20261013-v2",
            arms=["stad_native_local"], method_selection_from_test=False,
            internal_split_policy="classwise-10pct-sparse-v1",
            access="current delta only; own frozen synthetic prefix; no historical raw reads")
        other["result"]["protocol"].pop("budget_mode")
        other["result"]["protocol"].pop("evaluation_split")
        row = other["result"]["rows"][0]
        row.update(arm="stad_native_local", checks={"budget_ok": True, "prefix_ok": True},
            audit=dict(historical_raw_reads=0, report_dev_used_for_selection=False,
                       test_used_for_selection=False,
                       internal_split=dict(policy="classwise-10pct-sparse-v1",
                                           disjoint=True, external_nodes_used=False)))
        evidence = summarize(dict(entries=[main, other]))
        self.assertTrue(any(r["right"] == "stad_native_local" for r in evidence["pairs"]))
        for field, value in (("policy", "unregistered"), ("disjoint", False),
                             ("external_nodes_used", True)):
            invalid = copy.deepcopy(other)
            invalid["result"]["rows"][0]["audit"]["internal_split"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "split audit"):
                summarize(dict(entries=[main, invalid]))

    def test_duplicate_completed_retry_cannot_double_count(self):
        duplicate = copy.deepcopy(self.snapshot["entries"][1])
        duplicate["path"] = "baselines/retry/results.json"
        with self.assertRaisesRegex(ValueError, "duplicate cells"):
            summarize(dict(entries=self.snapshot["entries"] + [duplicate]))


if __name__ == "__main__":
    unittest.main()
