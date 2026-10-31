import unittest

from summarize_phase12_frozen import evaluate_gates


class PromotionTests(unittest.TestCase):
    def cells(self, delta=1.2):
        return [dict(joint_minus_c0_pp=delta, joint_minus_c0_old_pp=0.0,
                     joint_minus_c0_new_pp=delta, joint_minus_kcenter_pp=0.1) for _ in range(36)]

    def test_incomplete_cannot_promote_even_when_every_score_wins(self):
        result = evaluate_gates(self.cells(), [{"build_plus_one_student_ratio": 1.1}], 1.0, False)
        self.assertFalse(result["promote"])

    def test_worst_old_cell_is_a_gate_even_when_mean_increases(self):
        cells = self.cells()
        cells[0]["joint_minus_c0_old_pp"] = -2.01
        self.assertFalse(evaluate_gates(cells, [{"build_plus_one_student_ratio": 1.1}], 1.0, True)["promote"])

    def test_thresholds_and_per_dataset_cost(self):
        self.assertTrue(evaluate_gates(self.cells(), [{"build_plus_one_student_ratio": 2.0}], 0.0, True)["promote"])
        self.assertFalse(evaluate_gates(self.cells(), [{"build_plus_one_student_ratio": 1.0},
            {"build_plus_one_student_ratio": 2.001}], 1.0, True)["promote"])


if __name__ == "__main__":
    unittest.main()
