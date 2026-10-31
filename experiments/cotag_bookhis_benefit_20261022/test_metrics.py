import unittest

import numpy as np
from sklearn.metrics import f1_score

from metrics import frequency_groups, metrics


class MetricTests(unittest.TestCase):
    def test_outside_seen_predictions_count_as_false_negatives(self):
        y, pred = np.array([0, 0, 1, 1]), np.array([0, 2, 1, 0])
        result = metrics(y, pred, [0, 1], [0], [1], 3)
        self.assertAlmostEqual(result["final_f1"], 100 * f1_score(y, pred, labels=[0, 1], average="macro"))
        self.assertAlmostEqual(result["accuracy"], 50)
        self.assertAlmostEqual(result["old_f1"], 50)

    def test_decomposition(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        a = metrics(y, [0, 0, 1, 1, 2, 2], [0, 1, 2], [0, 1], [2], 3)
        b = metrics(y, [0, 2, 1, 1, 1, 0], [0, 1, 2], [0, 1], [2], 3)
        parts = list(zip(a["per_class"], b["per_class"]))
        self.assertAlmostEqual(sum(x["f1"] - z["f1"] for x, z in parts) / 3, a["final_f1"] - b["final_f1"])
        self.assertAlmostEqual(sum(x["support"] / len(y) * (x["recall"] - z["recall"]) for x, z in parts),
                               a["accuracy"] - b["accuracy"])

    def test_frequency_ties_use_id(self):
        self.assertEqual(frequency_groups([1, 1, 1, 1, 3, 5]),
                         {0: "low", 1: "low", 2: "middle", 3: "middle", 4: "high", 5: "high"})


if __name__ == "__main__":
    unittest.main()
