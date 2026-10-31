"""Small CPU checks for checkpoint and continual diagnostic semantics."""

import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import torch

from protocol_data import subset_graph
from run_native_experiments import Experiment
from run_native_attribution import run_preserved
from run_outer_trajectory import build_args
from test_native_core import tiny_data


def metric(value):
    return {"macro_f1": value, "accuracy": value, "n": 4,
            "per_stage": [{"macro_f1": value, "accuracy": value, "n": 4}],
            "old_mean_f1": None, "current_f1": value}


class NativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_zero_steps_is_initial_checkpoint_and_none_uses_default(self):
        experiment = Experiment.__new__(Experiment)
        experiment.args = Namespace(student_steps=2, rate=.5, batch_size=3,
                                    check_every=1, lr=.001, hidden=4)
        experiment.counts = [7]
        experiment.device = "cpu"
        for requested, expected_steps, values, expected_gain in (
                (0, 0, [20., 20.], 0.),
                (None, 2, [20., 30., 25., 25.], 10.)):
            with self.subTest(steps=requested):
                experiment.metric = Mock(side_effect=[metric(value) for value in values])
                _, info = experiment.fit(tiny_data(), 0, 3, steps=requested)
                self.assertEqual(info["steps"], expected_steps)
                self.assertEqual(info["step0_checkpoint"], metric(20.))
                self.assertEqual(info["best_minus_step0"], expected_gain)
                self.assertEqual(info["curve"][0]["step"], 0)
                self.assertEqual(info["curve"][0]["examples_seen"], 0)
                self.assertEqual(info["curve"][0]["per_stage"], metric(20.)["per_stage"])
                self.assertIsNone(info["curve"][0]["old_mean_f1"])
                self.assertEqual(info["curve"][0]["current_f1"], 20.)
                self.assertEqual(info["selected_step"], 0 if expected_steps == 0 else 1)
                if expected_steps == 0:
                    self.assertEqual(info["examples_seen"], 0)
                    self.assertEqual(info["unique_rows_seen"], 0)

    def test_trajectory_build_args_records_text_anchor_configuration(self):
        args = build_args({}, "out", "prepared", "cpu", 44, [100, 101], 32,
                          100, 20, .05, "optimized")
        self.assertEqual(args.text_anchor_weight, .05)
        self.assertEqual(args.text_anchor_scope, "optimized")

    def test_diagnostics_reuse_previous_union_and_actual_delta_without_condensation(self):
        experiment = Experiment.__new__(Experiment)
        experiment.args = Namespace(stages=2, rate=.5, phase="continual",
                                    arms=["union", "old_only", "delta_only", "union_no_recondense"])
        experiment.counts = []
        experiment.started = time.time()
        deltas = [tiny_data(4, 10 * stage) for stage in range(2)]
        for stage, delta in enumerate(deltas):
            delta["stage"].fill_(stage)
        records = {}

        def load_stage(stage):
            experiment.counts.append(len(deltas[stage]["x"]))
            return deltas[stage]

        def record(data, stage, arm, access, weighted=False):
            records[(arm, stage)] = (data, access)

        experiment.load_stage = load_stage
        experiment.record = record
        experiment.teacher = Mock(return_value=object())
        experiment.initializer = lambda source, k, *args: subset_graph(source, torch.arange(k))
        experiment.optimize = Mock(side_effect=lambda source, initial, *args, **kwargs: initial)
        with tempfile.TemporaryDirectory() as output:
            experiment.output = Path(output)
            experiment.run()
            self.assertTrue((experiment.output / "completed.json").exists())

        self.assertNotIn(("old_only", 0), records)
        self.assertIs(records[("old_only", 1)][0], records[("union", 0)][0])
        self.assertIs(records[("delta_only", 0)][0], deltas[0])
        self.assertIs(records[("delta_only", 1)][0], deltas[1])
        combined = records[("union_no_recondense", 1)][0]
        expected_ids = torch.cat((records[("union", 0)][0]["origin_ids"], deltas[1]["origin_ids"]))
        torch.testing.assert_close(combined["origin_ids"], expected_ids)
        self.assertEqual(experiment.optimize.call_count, 2)
        self.assertIn("diagnostic", records[("old_only", 1)][1])
        self.assertIn("over_budget", records[("delta_only", 1)][1])

    def test_history_diagnostics_require_union_memory(self):
        for arm in ("old_only", "union_no_recondense"):
            with self.subTest(arm=arm):
                experiment = Experiment.__new__(Experiment)
                experiment.args = Namespace(arms=[arm])
                with self.assertRaisesRegex(ValueError, "require the union arm"):
                    experiment.run()

    def test_preserved_comparison_shares_teacher_for_identical_source(self):
        experiment = Experiment.__new__(Experiment)
        experiment.args = Namespace(rate=.5)
        experiment.counts = [4]
        stage0, delta = tiny_data(4, 0), tiny_data(4, 10)
        delta["stage"].fill_(1)
        anchor = subset_graph(stage0, torch.arange(2))
        experiment.load_stage = Mock(return_value=delta)
        experiment.record = Mock()
        experiment.teacher = Mock(side_effect=[object(), object()])
        experiment.initializer = lambda source, k, *args: subset_graph(source, torch.arange(k))
        experiment.optimize = Mock(side_effect=lambda source, initial, *args, **kwargs: initial)
        with tempfile.TemporaryDirectory() as output:
            experiment.output = Path(output)
            run_preserved(experiment, stage0, anchor, 2)
        self.assertEqual(experiment.teacher.call_count, 2)
        union, append, joint, protected = experiment.optimize.call_args_list
        self.assertIs(union.args[2], joint.args[2])
        self.assertIs(joint.args[2], protected.args[2])
        self.assertIsNot(append.args[2], joint.args[2])
        torch.testing.assert_close(joint.args[1]["x"], protected.args[1]["x"], rtol=0, atol=0)
        self.assertEqual(protected.kwargs["extra_config"]["optimize_rows"].tolist(),
                         [False, False, True, True])


if __name__ == "__main__":
    unittest.main()
