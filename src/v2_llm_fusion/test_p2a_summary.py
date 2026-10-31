"""Regression tests for P2-A aggregation, independent of GPU training."""
from pathlib import Path
import tempfile
import unittest

import summarize_p2a_graph_conditioned_text as S


class SummaryTests(unittest.TestCase):
    def stages(self):
        tasks = [[50., 0., 0.], [60., 55., 0.], [40., 65., 70.]]
        stages = []
        for stage, values in enumerate(tasks):
            metrics = dict(cumulative_f1=30.+10*stage, old_f1=40.,
                           new_f1=70., per_task_f1=values)
            other = dict(metrics, cumulative_f1=99.)
            rows = [dict(seed=100, metrics={"all": metrics, "seen": other}),
                    dict(seed=101, metrics={"all": other, "seen": metrics})]
            stages.append(dict(stage=stage, per_seed=rows[::(-1 if stage % 2 else 1)]))
        return stages

    def paired(self):
        return [dict(dataset=d, arm="graph_selection", control=c, final=.5,
                     stage_mean=0., old=-2., new=-2., per_order_final={"canonical": -2.})
                for d in ("cora", "citeseer") for c in S.CONTROLS]

    def test_seed_pairing_uses_ids_not_list_position(self):
        rows = self.stages()
        self.assertEqual(S.seed_summary(rows, 100)["final"], 50.)
        self.assertEqual(S.seed_summary(rows, 100)["stage_mean"], 40.)
        self.assertEqual(S.seed_summary(rows, 101)["final"], 99.)
        self.assertEqual(S.seed_summary(rows, 100, scope="seen")["final"], 99.)

    def test_forgetting_uses_historical_max_and_allows_backward_gain(self):
        # Task 0: max(50, 60)-40 = 20. Task 1: 55-65 = -10.
        self.assertEqual(S.seed_summary(self.stages(), 100)["forgetting"], 5.)
        rows = self.stages()
        for row in rows[-1]["per_seed"]:
            if row["seed"] == 100:
                row["metrics"]["all"]["per_task_f1"][0] = 80.
        self.assertEqual(S.seed_summary(rows, 100)["forgetting"], -15.)

    def test_screen_boundary_is_eligible_not_promoted(self):
        result = S.screen([], self.paired())
        self.assertTrue(result["pass_screen"])
        self.assertFalse(result["promotion"])
        self.assertEqual(result["independent_confirmation"], "eligible, not yet executed")

    def test_screen_rejects_each_prespecified_regression(self):
        for key, value in (("final", .49), ("stage_mean", -.01),
                           ("old", -2.01), ("new", -2.01)):
            with self.subTest(key=key):
                paired = self.paired()
                paired[0][key] = value
                result = S.screen([], paired)
                self.assertFalse(result["pass_screen"])
                self.assertEqual(result["independent_confirmation"], "not triggered")
        paired = self.paired()
        paired[-1]["per_order_final"]["canonical"] = -2.01
        self.assertFalse(S.screen([], paired)["pass_screen"])

    def test_missing_control_cannot_pass_screen(self):
        with self.assertRaises(StopIteration):
            S.screen([], self.paired()[:-1])

    def test_missing_matrix_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "cora"
            (dataset / "canonical").mkdir(parents=True)
            S.save(dataset / "plan.json", dict(steps_override=None, arms=list(S.ARMS),
                   seeds=list(range(100, 106)), orders=["canonical", "order1701", "order1702"]))
            S.save(dataset / "acceptance.json", dict(complete=True))
            S.save(dataset / "canonical" / "results.json", dict(results=[]))
            with self.assertRaisesRegex(AssertionError, "missing arm/stage cell"):
                S.summarize(root, root / "summary")
            self.assertFalse((root / "summary" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
