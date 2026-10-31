"""Numerical diagnostics and safe output handling for coverage comparisons."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_q_coverage_diagnostics import CoverageDiagnostics, assignment_metrics


class CoverageDiagnosticsTest(unittest.TestCase):
    def test_known_assignment_entropy_and_gap(self):
        q = torch.tensor([[.5, .5], [.5, .5], [1., 0.], [0., 1.]])
        text = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        metrics = assignment_metrics(q, text, torch.zeros(4, dtype=torch.long),
            torch.zeros(2, dtype=torch.long), torch.tensor([2, 3]), ["a", "b"])
        self.assertAlmostEqual(metrics["normalized_row_entropy"], .5, places=6)
        self.assertEqual(metrics["hard_slot_coverage"], 1.)
        self.assertEqual(metrics["unique_source_ratio"], 1.)
        self.assertAlmostEqual(metrics["class_coverage_cosine"], 1.)
        self.assertAlmostEqual(metrics["soft_hard_cosine_gap_mean"],
                               1. - 1.5 / (2.5 ** .5), places=6)

    def test_one_slot_class_and_actual_source_id_mapping(self):
        q = torch.ones(2, 1)
        callback = CoverageDiagnostics()
        callback(dict(q=q, q0=q.clone(), delta=dict(ids=torch.tensor([10, 20]),
            text_emb=torch.tensor([[1., 0.], [1., 0.]]), y=torch.zeros(2, dtype=torch.long)),
            block=dict(labels=torch.zeros(1, dtype=torch.long),
                text_source_ids=[[20]], texts=["payload"])))
        self.assertEqual(callback.result["q_relative_update"], 0.)
        self.assertEqual(callback.result["normalized_row_entropy"], 0.)
        self.assertEqual(callback.result["soft_hard_cosine_gap_mean"], 0.)
        self.assertIsNone(callback.result["per_class"][0]["within_class_text_cosine"])


class CoverageRunnerGuardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load only the runner guards; legacy dataset imports are tested remotely.
        fake = types.ModuleType("run_unified_q_attribution_20261008")
        for name in ("R", "L", "V", "M", "UQ"):
            setattr(fake, name, types.SimpleNamespace())
        path = Path(__file__).with_name("run_uq_coverage_confirmation_20261012.py")
        spec = importlib.util.spec_from_file_location("coverage_runner_guards", path)
        cls.runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {fake.__name__: fake}):
            spec.loader.exec_module(cls.runner)

    def test_subset_does_not_require_primary_arm(self):
        self.runner.validate_arms(["uq_qcoverage"])
        self.runner.validate_arms(["cat_cctp", "adaptive_coherent"])
        for arms in ([], ["uq_qcoverage", "uq_qcoverage"], ["adaptive_coherent"],
                     ["adaptive_coherent", "cat_cctp"]):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                self.runner.validate_arms(arms)
        self.assertEqual(self.runner.method_class("adaptive_coherent"), "internal_historical_candidate")

    def test_partial_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            payload = json.dumps({"status": "running", "rows": ["existing"]})
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "partial output preserved"):
                self.runner.protect_existing(path)
            self.assertEqual(path.read_text(encoding="utf-8"), payload)


if __name__ == "__main__":
    unittest.main()
