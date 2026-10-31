import copy
import tempfile
import unittest
from pathlib import Path

import report_semantic_topology_controls as R
import summarize_topology_causal_ablation as S


def supplement_fixture():
    manifest = dict(dataset="toy", orders={"1": [[0], [1]]}, sources={"runner.py": "digest"})
    structural, timings = [], []
    for stage in range(2):
        for arm in R.NEW_ARMS:
            structural.append(dict(dataset="toy", order=1, stage=stage, arm=arm,
                                   matched_baseline="t_degree" if arm.endswith("capacity") else "internal_only",
                                   same_current_cross=True, same_full_degree=arm.startswith("cycles"),
                                   max_new_internal_degree=2 if arm.startswith("cycles") else 3,
                                   memory_file_bytes=2100 + stage, tensor_bytes=1000 + stage * 100,
                                   text_utf8_bytes=200))
        for op in ("c0", "single_ring", "knn", "cycles"):
            timings.append(dict(dataset="toy", order=1, stage=stage, operator=op, new_rows=10 + stage,
                                seconds=[v * (stage + 1) for v in (1., 3., 2.)],
                                median_seconds=2. * (stage + 1), repeats=3))
    data = dict(dataset="toy", passed=True, frozen_sources_checked=manifest["sources"].copy(),
                memory_checks=8, cycle_degree_checks=4, structural=structural,
                internal_microbenchmark=timings)
    return data, manifest


class ReportingTest(unittest.TestCase):
    def test_paired_order_and_seed(self):
        rows = [dict(task="nc", order=order, split=None, seed=seed, horizon="final",
                     metric="all_cumulative_f1", arm=arm, value=value)
                for order in (1, 2) for seed in (10, 11)
                for arm, value in (("c0", seed + order), ("new", seed + order + 2))]
        result = R.paired_contrasts(rows, {"new-c0": {"new": 1, "c0": -1}})
        pooled = next(r for r in result if r["order"] == "all")
        self.assertEqual((pooled["mean_pp"], pooled["n"], pooled["positive"]), (2, 4, 4))
        with self.assertRaisesRegex(ValueError, "unpaired"):
            R.paired_contrasts(rows[:-1], {"new-c0": {"new": 1, "c0": -1}})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            R.paired_contrasts(rows + rows[:1], {"new-c0": {"new": 1, "c0": -1}})

    def test_trajectory_endpoint_and_undefined_old(self):
        manifest = dict(orders={"1": [[0], [1], [2]]})
        records = [dict(task="nc", order=1, seed=10, arm="c0", stage=i,
                        metrics={"all": {"cumulative_f1": value,
                                         "old_f1": None if i == 0 else value}})
                   for i, value in enumerate((10, 20, 30))]
        rows = S.trajectories(records, manifest)
        old = next(r for r in rows if r["horizon"] == "stage_mean" and r["metric"] == "all_old_f1")
        self.assertEqual((old["value"], old["stages"]), (25, 2))
        partial = S.trajectories(records[:2], manifest)
        self.assertFalse(any(r["horizon"] == "final" for r in partial))

    def test_exact_matrix_rejects_bad_cells(self):
        manifest = dict(dataset="toy", orders={"1": [[0]]}, seeds=[10], splits=[20], arms=["c0"])
        records = [dict(task="nc", dataset="toy", order=1, stage=0, seed=10, arm="c0",
                        metrics={"all": {"cumulative_f1": 12}}),
                   dict(task="lp", dataset="toy", order=1, stage=0, seed=10, split=20,
                        arm="c0", metrics={"all": {"auc": .8, "ap": .7}})]
        self.assertEqual(S.validate(records, manifest), 2)
        for bad in (records[:1], records + records[:1]):
            with self.assertRaises(ValueError):
                S.validate(bad, manifest)
        bad = copy.deepcopy(records)
        bad[0]["dataset"] = "other"
        with self.assertRaisesRegex(ValueError, "foreign"):
            S.validate(bad, manifest)
        bad = copy.deepcopy(records)
        bad[1]["metrics"]["all"]["auc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            S.validate(bad, manifest)

    def test_supplement_summary_preserves_units_and_scope(self):
        data, manifest = supplement_fixture()
        original = copy.deepcopy(data)
        timing, storage = R.supplement_rows(data, manifest)
        self.assertEqual(data, original)
        self.assertEqual(len(timing), 4)
        self.assertEqual(len(storage), 4)
        self.assertEqual(timing[0]["mean_block_median_seconds"], 3.)
        self.assertEqual(timing[0]["measured_runs"], 6)
        self.assertEqual(timing[0]["blocks"], 2)
        self.assertEqual(storage[0]["final_file_mib_max"], 2101 / 2**20)
        self.assertEqual(storage[0]["final_memories"], 1)
        self.assertEqual([r["full_degree_matches"] for r in storage], [0, 0, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            R.write_markdown(path, [], [], timing, storage)
            report = path.read_text(encoding="utf-8")
            self.assertIn("Not end-to-end speedup evidence", report)
            self.assertIn("not process RAM", report)
            self.assertIn("| toy | c0 | 2 | 3.000000 |", report)

    def test_supplement_rejects_missing_duplicate_and_foreign_cells(self):
        for field in ("structural", "internal_microbenchmark"):
            for mode in ("missing", "duplicate", "foreign"):
                with self.subTest(field=field, mode=mode):
                    data, manifest = supplement_fixture()
                    if mode == "missing":
                        data[field].pop()
                    elif mode == "duplicate":
                        data[field].append(data[field][0].copy())
                    else:
                        data[field][0]["dataset"] = "other"
                    with self.assertRaisesRegex(ValueError, "matrix mismatch"):
                        R.supplement_rows(data, manifest)

    def test_supplement_rejects_bad_timing_and_different_blocks(self):
        cases = ({"seconds": []}, {"seconds": [1., float("nan"), 2.]},
                 {"seconds": [-1., 2., 3.]}, {"median_seconds": float("inf")},
                 {"median_seconds": 1.}, {"repeats": 2}, {"new_rows": 0})
        for changes in cases:
            with self.subTest(changes=changes):
                data, manifest = supplement_fixture()
                data["internal_microbenchmark"][0].update(changes)
                with self.assertRaisesRegex(ValueError, "invalid internal timing"):
                    R.supplement_rows(data, manifest)
        data, manifest = supplement_fixture()
        data["internal_microbenchmark"][0]["new_rows"] = 99
        with self.assertRaisesRegex(ValueError, "different blocks"):
            R.supplement_rows(data, manifest)

    def test_supplement_rejects_broken_structural_guards(self):
        cases = ((0, {"same_current_cross": False}, "cross policy"),
                 (0, {"matched_baseline": "c0"}, "cross policy"),
                 (2, {"same_full_degree": False}, "cycle degree"),
                 (2, {"max_new_internal_degree": 3}, "cycle degree"),
                 (0, {"memory_file_bytes": -1}, "storage size"),
                 (0, {"text_utf8_bytes": 201}, "payload sizes"))
        for index, changes, message in cases:
            with self.subTest(changes=changes):
                data, manifest = supplement_fixture()
                data["structural"][index].update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    R.supplement_rows(data, manifest)

    def test_supplement_rejects_wrong_provenance_and_counts(self):
        cases = ({"passed": False}, {"dataset": "other"},
                 {"frozen_sources_checked": {}}, {"memory_checks": 99},
                 {"cycle_degree_checks": 99})
        for changes in cases:
            with self.subTest(changes=changes):
                data, manifest = supplement_fixture()
                data.update(changes)
                with self.assertRaises(ValueError):
                    R.supplement_rows(data, manifest)


if __name__ == "__main__":
    unittest.main()
