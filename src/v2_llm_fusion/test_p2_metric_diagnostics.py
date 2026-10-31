import unittest

import numpy as np

from run_p2_metric_diagnostics import class_metrics


class ClassSupportTests(unittest.TestCase):
    def test_cross_task_false_positives_count_against_old_class(self):
        result = class_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]),
                               [0, 1], [0], [1], 2)
        self.assertAlmostEqual(result["old_class_f1_global_support"], 80)
        self.assertAlmostEqual(result["new_class_f1_global_support"], 200 / 3)
        self.assertAlmostEqual(result["cumulative_f1"], (80 + 200 / 3) / 2)


if __name__ == "__main__":
    unittest.main()
