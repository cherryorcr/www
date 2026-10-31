import copy
import unittest

import torch

import analyze as A


class AnalysisTest(unittest.TestCase):
    def test_students_are_nested_within_construction(self):
        cohort = lambda f1: dict(f1=f1, accuracy=f1+1)
        score = dict(cumulative=cohort(20), current=cohort(20), historical=None,
                     per_student=[dict(seed=s, cumulative=cohort(f), current=cohort(f), historical=None)
                                  for s, f in zip([150, 151, 152], [10, 20, 30])])
        value = dict(status="complete", protocol=dict(construction_seed=271006),
                     rows=[dict(stage=0, scores=dict(prediction={"arrived_snapshot": {"historical_test": score}}))])
        rows = A.collect(value, "arxiv", "gecc_nearest1")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["f1"], 20)
        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(rows[0]["source_split"], "historical_test")

    def test_stage_average_requires_all_five_stages(self):
        rows = [dict(dataset="acm", variant="gecc_nearest1", construction_seed=271006,
                     mode="arrived_snapshot", split="test", cohort="cumulative",
                     stage=str(s), f1=s, accuracy=s+1) for s in range(5)]
        self.assertEqual(A.stage_averages(rows)[0]["f1"], 2)
        with self.assertRaises(AssertionError):
            A.stage_averages(rows[:-1])
        with self.assertRaises(AssertionError):
            A.stage_averages(rows+[rows[0]])

    def test_paired_direction_and_missing_seed(self):
        left = dict(zip(A.S.SEEDS, [3, 4, 5]))
        right = dict(zip(A.S.SEEDS, [2, 4, 6]))
        result = A.S.paired(left, right)
        self.assertEqual(result["difference_pp"], 0)
        self.assertEqual(result["positive_pairs"], 1)
        self.assertEqual(result["worst_difference_pp"], -1)
        self.assertLess(result["ci95_low"], 0)
        self.assertGreater(result["ci95_high"], 0)
        with self.assertRaises(AssertionError):
            A.S.paired(left, {271006: 2})

    def test_raw_prediction_metric_and_missing_mode(self):
        data = dict(arrival=torch.tensor([0, 0, 1]), labels=torch.tensor([0, 1, 0]),
                    classes=2, masks={s: torch.tensor([True, True, True]) for s in ("test", "report_dev")})
        predictions = {s: {m: dict(ids=torch.tensor([0, 1]), pred=torch.tensor([0, 1]))
                           for m in ("arrived_snapshot", "current_snapshot")} for s in (150, 151, 152)}
        metric = dict(nodes=2, f1=100., accuracy=100.)
        aggregate = dict(metric, f1_sd=0., accuracy_sd=0.)
        score = dict(cumulative=aggregate, current=aggregate, historical=None,
                     per_student=[dict(seed=s, cumulative=metric, current=metric, historical=None)
                                  for s in (150, 151, 152)])
        scores = dict(prediction={m: {s: copy.deepcopy(score) for s in ("test", "report_dev")}
                                 for m in ("arrived_snapshot", "current_snapshot")})
        self.assertEqual(A.check_metrics(scores, predictions, data, 0), 48)
        broken = copy.deepcopy(scores)
        broken["prediction"].pop("current_snapshot")
        with self.assertRaises(AssertionError):
            A.check_metrics(broken, predictions, data, 0)
        wrong = copy.deepcopy(predictions)
        wrong[150]["arrived_snapshot"]["pred"] = torch.tensor([1, 0])
        with self.assertRaises(AssertionError):
            A.check_metrics(scores, wrong, data, 0)


if __name__ == "__main__":
    unittest.main()
