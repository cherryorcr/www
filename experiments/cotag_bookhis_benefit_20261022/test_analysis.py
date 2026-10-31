import unittest

import numpy as np
import pandas as pd

from analyze import crossed, interval


class AnalysisTests(unittest.TestCase):
    def test_three_constructions_not_nine_students(self):
        students = pd.DataFrame({"construction": np.repeat([1, 2, 3], 3),
                                 "delta": [1, 2, 3, 2, 3, 4, 3, 4, 5]})
        result = interval(students.groupby("construction").delta.mean())
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 3)
        self.assertEqual(result["sd"], 1)

    def test_crossed_bootstrap_constant(self):
        frame = pd.DataFrame([dict(order=o, construction=c, delta=2.) for o in range(3) for c in range(3)])
        result = crossed(frame, "delta")
        self.assertEqual(result["crossed_low"], 2)
        self.assertEqual(result["crossed_high"], 2)

    def test_crossed_bootstrap_requires_complete_design(self):
        frame = pd.DataFrame([dict(order=o, construction=c, delta=2.) for o in range(3) for c in range(3)][:-1])
        with self.assertRaises(AssertionError):
            crossed(frame, "delta")


if __name__ == "__main__":
    unittest.main()
