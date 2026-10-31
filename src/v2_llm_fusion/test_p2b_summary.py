import copy
import unittest

import summarize_p2b_student_attention as S


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(classification="known-order screen", arms=list(S.ARMS),
            seeds=list(range(100, 106)), orders={k: [] for k in ("canonical", "order1701", "order1702")},
            outer_steps=40, inner_steps=16, validation_steps=32, stages=3, report_steps_override=None)
        self.pairs = [dict(dataset=d, control=c, final=1., stage_mean=.1, old=0., new=1.,
            per_order_final={"canonical": 1., "order1701": 1., "order1702": 1.})
            for d in ("cora", "citeseer") for c in ("center_concat", "fixed_anchor", "semantic")]
        self.gains = [dict(dataset=d, method_seed=s, gain=1.)
                      for d in ("cora", "citeseer") for s in (700, 701)]

    def test_valid_and_smoke_matrix(self):
        S.check_matrix(self.plan, {"complete": True})
        self.plan["report_steps_override"] = 16
        with self.assertRaises(AssertionError):
            S.check_matrix(self.plan, {"complete": True})

    def test_missing_seed_rejected(self):
        self.plan["seeds"].pop()
        with self.assertRaises(AssertionError):
            S.check_matrix(self.plan, {"complete": True})

    def test_screen_not_promotion(self):
        result = S.screen(self.pairs, self.gains)
        self.assertTrue(result["pass_screen"])
        self.assertFalse(result["promotion"])

    def test_missing_controls_cannot_pass(self):
        with self.assertRaises(AssertionError):
            S.screen(self.pairs[:-1], self.gains)
        with self.assertRaises(AssertionError):
            S.screen(self.pairs, self.gains[:-1])

    def test_both_datasets_and_all_controls(self):
        self.pairs[-1]["final"] = .49
        self.assertFalse(S.screen(self.pairs, self.gains)["pass_screen"])

    def test_method_seed_not_cherry_picked(self):
        self.gains[-1]["gain"] = -.1
        self.assertFalse(S.screen(self.pairs, self.gains)["pass_screen"])

    def test_order_and_plasticity_guard(self):
        p = copy.deepcopy(self.pairs)
        p[0]["per_order_final"]["order1702"] = -2.01
        self.assertFalse(S.screen(p, self.gains)["pass_screen"])
        self.pairs[0]["new"] = -2.01
        self.assertFalse(S.screen(self.pairs, self.gains)["pass_screen"])

    def test_stage_average_and_forgetting(self):
        stages = []
        for s, final in enumerate((20., 40., 60.)):
            m = dict(cumulative_f1=final, old_f1=50., new_f1=70.,
                     per_task_f1=[[50.], [40., 60.], [30., 50., 70.]][s])
            stages.append(dict(per_seed=[dict(seed=100, metrics=dict(all=m))]))
        result = S.seed_summary(stages, 100)
        self.assertEqual(result["final"], 60.)
        self.assertEqual(result["stage_mean"], 40.)
        self.assertEqual(result["forgetting"], 15.)


if __name__ == "__main__":
    unittest.main()
