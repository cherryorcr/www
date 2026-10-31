"""Check the evidence unit and protocol guards for coverage confirmation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_uq_coverage_20261012 import plots, summarize


def fixture(arms=("uq_dm_edgefree", "uq_qcoverage"), seed=271108, gain=2.):
    protocol = dict(name="uq-q-coverage-confirmation-20261012-v2", dataset="wikics",
        order_seed=9101, construction_seed=seed, arms=list(arms), groups=[[0], [1]],
        budgets=[5, 10], stage_raw_rows=[100, 100], memory_rate=.05,
        actual_final_rate=.05, max_stages=5, method_selection_from_test=False,
        evaluation_split="report-dev only; sealed test not used",
        pool_mask_hash="pool", report_mask_hash="report", test_mask_hash="test",
        stage_id_hashes=["stage0", "stage1"], student={"name": "common"},
        student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        q_configs={"base": {}}, source_hashes={"unified_q_payload_projection.py": "projection"})
    rows, stages = [], []
    for arm in arms:
        f1 = 60. + (gain if arm == "uq_qcoverage" else 0.)
        score = dict(final_f1=f1, accuracy=70., old_f1=50., new_f1=65.,
            final_f1_std=1., accuracy_std=1., student_seeds=[150, 151, 152],
            student_steps=2048, student_batch=24,
            per_student=[dict(seed=s, final_f1=f1+d, accuracy=70.+d,
                old_f1=50.+d, new_f1=65.+d) for s, d in zip((150, 151, 152), (-1., 0., 1.))])
        rows.append(dict(arm=arm, stage=1, memory_rows=10, score=score,
            method_class="primary" if arm == "uq_dm_edgefree" else "internal_text_projection",
            construction_seconds=3., construction_peak_mb=128., unique_texts=8,
            memory_hash="memory"))
        for stage in (0, 1):
            stages.append(dict(arm=arm, stage=stage, new_rows=5, audit={},
                construction_seconds_including_reused=1.5, reused_cat_seconds=0.,
                diagnostic_seconds=0.))
    return dict(protocol=protocol, status="complete", rows=rows, stages=stages)


class CoverageSummaryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, result):
        folder = self.root / name
        folder.mkdir()
        (folder / "results.json").write_text(json.dumps(result), encoding="utf-8")

    def test_student_seeds_do_not_create_independent_cells(self):
        self.write("one", fixture())
        evidence = summarize(self.root)
        pair = next(r for r in evidence["pairs"] if r["metric"] == "final_f1")
        self.assertEqual((pair["cells"], pair["delta"]), (1, 2.))
        self.assertIsNone(pair["ci_low"])
        self.assertEqual(len(evidence["student_rows"]), 6)

    def test_paired_cells_use_intersection_and_cell_means(self):
        self.write("one", fixture())
        self.write("two", fixture(seed=271109, gain=4.))
        self.write("unmatched", fixture(arms=("uq_qcoverage",), seed=271110, gain=30.))
        pair = next(r for r in summarize(self.root)["pairs"] if r["metric"] == "final_f1")
        self.assertEqual((pair["cells"], pair["delta"]), (2, 3.))
        self.assertLess(pair["ci_low"], 3.)
        self.assertGreater(pair["ci_high"], 3.)

    def test_wrong_means_and_duplicate_cells_are_rejected(self):
        bad = fixture()
        bad["rows"][0]["score"]["accuracy"] = 80.
        self.write("bad", bad)
        with self.assertRaisesRegex(ValueError, "mean mismatch"):
            summarize(self.root)
        (self.root / "bad" / "results.json").write_text(json.dumps(fixture()), encoding="utf-8")
        self.write("duplicate", fixture())
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            summarize(self.root)

    def test_protocol_mismatch_is_rejected_even_for_distinct_arms(self):
        self.write("base", fixture(arms=("uq_dm_edgefree",)))
        other = fixture(arms=("uq_qcoverage",))
        other["protocol"]["report_mask_hash"] = "different"
        self.write("candidate", other)
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            summarize(self.root)

    def test_diagnostics_use_slot_raw_and_class_denominators(self):
        result = fixture(arms=("uq_dm_edgefree",))
        for stage, slots, raw, gap, coverage, entropy, unique in (
                (0, 2, 10, .6, .9, .2, .5), (1, 8, 90, .2, .3, .6, .75)):
            row = result["stages"][stage]
            row["new_rows"] = slots
            row["audit"] = dict(coverage_diagnostics=dict(
                hard_slot_coverage=.5 if stage == 0 else 1., positive_mass_slot_coverage=1.,
                unique_source_ratio=unique, slot_source_entropy=1., row_entropy=entropy,
                normalized_row_entropy=entropy, soft_hard_cosine_gap_mean=gap,
                soft_hard_cosine_gap_max=gap, q_relative_update=2.,
                per_class=[dict(class_id=stage, raw_rows=raw, slots=slots,
                                class_coverage_cosine=coverage)]))
        self.write("weighted", result)
        row = summarize(self.root)["diagnostic_cells"][0]
        self.assertEqual((row["slots"], row["raw_rows"], row["classes"]), (10, 100, 2))
        self.assertAlmostEqual(row["hard_slot_coverage"], .9)
        self.assertAlmostEqual(row["normalized_row_entropy"], .56)
        self.assertAlmostEqual(row["unique_source_ratio"], .7)
        self.assertAlmostEqual(row["soft_hard_cosine_gap_mean"], .28)
        self.assertAlmostEqual(row["class_coverage_cosine"], .6)
        self.assertEqual(row["unique_text_ratio"], .8)

    def test_smoke_results_are_excluded_and_plot_renders(self):
        smoke = fixture()
        smoke["status"] = "smoke_complete"
        self.write("smoke", smoke)
        self.write("formal", fixture())
        evidence = summarize(self.root)
        self.assertEqual(len(evidence["cells"]), 2)
        self.assertEqual(len(evidence["excluded"]), 1)
        plots(self.root, evidence)
        self.assertTrue((self.root / "report_dev_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
