"""Lightweight reporting checks without torch/GPU dependencies."""
import copy
import unittest

import summarize_topology_causal_ablation as S


class ReportingTest(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(dataset="toy", orders={"1": [[0], [1]]}, splits=[3],
                             arms=list(S.ARMS), seeds=[4, 5])
        self.records = []
        for task in ("nc", "lp"):
            for stage in (0, 1):
                for arm in S.ARMS:
                    for seed in (4, 5):
                        value = (60 if task == "nc" else .6) + (.02 if arm == "t_degree" else 0)
                        metrics = ({"cumulative_f1": value, "old_f1": None if stage == 0 else value}
                                   if task == "nc" else {"auc": value, "ap": value})
                        self.records.append(dict(task=task, dataset="toy", order=1, stage=stage,
                                                 split=None if task == "nc" else 3, arm=arm, seed=seed,
                                                 metrics={"all": metrics}))

    def test_complete_and_missing(self):
        self.assertEqual(S.validate(self.records, self.manifest), 48)
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            S.validate(self.records[:-1], self.manifest)

    def test_duplicate_replacement(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            S.validate(self.records[:-1] + [self.records[0]], self.manifest)

    def test_invalid_primary_metric(self):
        records = copy.deepcopy(self.records)
        records[0]["metrics"]["all"]["cumulative_f1"] = None
        with self.assertRaisesRegex(ValueError, "primary metric"):
            S.validate(records, self.manifest)

    def test_lp_pp_and_interaction(self):
        rows = S.trajectories(self.records, self.manifest)
        contrasts = S.paired_contrasts(rows)
        for name in ("t_degree-c0", "interaction"):
            r = next(r for r in contrasts if r["task"] == "lp" and r["order"] == "all"
                     and r["horizon"] == "final" and r["metric"] == "all_auc" and r["contrast"] == name)
            self.assertAlmostEqual(r["mean"], 2.)
            self.assertEqual(r["n"], 2)
            self.assertEqual(r["positive"], 2)

    def test_pair_before_sd(self):
        rows = S.trajectories(self.records, self.manifest)
        for r in rows:
            r["value"] += r["seed"] * 100
        contrast = S.paired_contrasts(rows)[0]
        self.assertAlmostEqual(contrast["sd"], 0.)

    def test_null_not_zero(self):
        rows = S.trajectories(self.records, self.manifest)
        old = [r for r in rows if r["metric"] == "all_old_f1" and r["horizon"] == "stage_mean"]
        self.assertTrue(all(r["stages"] == 1 for r in old))
        self.assertIsNone(S.avg([]))
        self.assertIsNone(S.sd([]))
        self.assertIsNone(S.sd([1]))

    def test_missing_subgroup_arm(self):
        rows = S.trajectories(self.records, self.manifest)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            S.paired_contrasts([r for r in rows if r["arm"] != "random_ring"])


if __name__ == "__main__":
    unittest.main()
