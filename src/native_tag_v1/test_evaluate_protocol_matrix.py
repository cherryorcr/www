"""Tests for fixed-label-space protocol-matrix metrics."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluate_protocol_matrix import evaluate_model, score


def _logits(predictions: list[int], num_classes: int = 7) -> torch.Tensor:
    logits = torch.full((len(predictions), num_classes), -1.0)
    logits[torch.arange(len(predictions)), torch.tensor(predictions)] = 1.0
    return logits


class ProtocolMatrixMetricTests(unittest.TestCase):
    def test_score_counts_absent_protocol_labels_in_macro_f1(self):
        metric = score(np.asarray([0, 0]), np.asarray([0, 1]), [0, 1])

        self.assertAlmostEqual(metric["macro_f1"], 100.0 / 3.0)
        self.assertEqual(metric["support"], {"0": 2, "1": 0})
        self.assertEqual(metric["accuracy"], 50.0)

    def test_evaluate_model_uses_stage_seen_and_future_label_spaces(self):
        stage_classes = [[0, 1], [2, 3], [4, 5, 6]]
        stages = []
        predictions = {}
        for stage, present_label in enumerate((0, 2, 4)):
            train_graph = {"key": f"stage-{stage}-train"}
            eval_graph = {"key": f"stage-{stage}-eval"}
            predictions[train_graph["key"]] = _logits([present_label])
            predictions[eval_graph["key"]] = _logits([present_label, present_label])
            stages.append({
                "train": train_graph,
                "train_y": np.asarray([present_label]),
                "evaluation": {
                    "graph": eval_graph,
                    "y": torch.tensor([present_label, present_label]),
                    "masks": {
                        "checkpoint": torch.tensor([True, False]),
                        "test": torch.tensor([False, True]),
                    },
                },
            })

        def fake_evaluate_logits(_model, graph, _classes):
            return predictions[graph["key"]]

        with patch("evaluate_protocol_matrix.evaluate_logits", side_effect=fake_evaluate_logits):
            matrix = evaluate_model(object(), torch.empty(7, 1), stages, 1, stage_classes)

        cells = {(cell["eval_stage"], cell["partition"]): cell["metric"]
                 for cell in matrix["cells"]}
        self.assertEqual(cells[(0, "train")]["support"], {"0": 1, "1": 0})
        self.assertEqual(cells[(1, "checkpoint")]["support"], {"2": 1, "3": 0})
        self.assertEqual(cells[(2, "future")]["support"], {"4": 1, "5": 0, "6": 0})
        self.assertEqual(matrix["cumulative_seen"]["test"]["support"],
                         {"0": 1, "1": 0, "2": 1, "3": 0})
        self.assertEqual(matrix["cumulative_future_test"]["support"],
                         {"4": 1, "5": 0, "6": 0})
        self.assertEqual(matrix["cumulative_seen"]["test"]["macro_f1"], 50.0)
        self.assertAlmostEqual(matrix["cumulative_future_test"]["macro_f1"], 100.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
