import copy
import unittest

from summarize_tgr_cross_edge_confirmation import paired, summarize_trajectory


class SummaryTest(unittest.TestCase):
    def test_paired_seed_mismatch_is_rejected(self):
        a = [{"seed": i, "f1": i} for i in range(6)]
        b = [{"seed": i + 1, "f1": i} for i in range(6)]
        with self.assertRaises(ValueError):
            paired(a, b, "f1")

    def test_trajectory_uses_stage_scores_not_group_scores(self):
        stages = []
        for stage in range(3):
            ref = {"training_only_seconds": 1.0, "rows": [
                {"seed": seed, "cumulative_macro_f1": 10.0 * stage,
                 "cumulative_accuracy": 20.0 * stage,
                 "per_stage": [{"macro_f1": 99.0}]} for seed in range(150, 156)]}
            cand = copy.deepcopy(ref)
            for r in cand["rows"]:
                r["cumulative_macro_f1"] += stage
            stages.append({"stage": stage, "arms": {"semantic_recursive": ref,
                                                      "message_balanced": cand}})
        result = summarize_trajectory(stages)
        self.assertEqual(result["semantic_recursive"]["cumulative_macro_f1"]["mean"], 10)
        self.assertEqual(result["message_balanced"]["cumulative_macro_f1"]["mean"], 11)
        self.assertEqual(result["paired"]["cumulative_macro_f1"]["delta_pp"], 1)


if __name__ == "__main__":
    unittest.main()
