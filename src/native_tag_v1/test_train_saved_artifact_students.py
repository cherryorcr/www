"""CPU tests for the immutable saved-artifact student runner."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from evaluate_protocol_matrix import discover_models
from run_native_experiments import file_sha
from test_native_core import tiny_data
from train_saved_artifact_students import (
    train_saved_artifact_students,
    validate_artifact,
)


class SavedArtifactStudentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _inputs(self, root: Path) -> Namespace:
        prepared = root / "prepared"
        prepared.mkdir()
        train = tiny_data(4, 0)
        evaluation_graph = tiny_data(4, 100)
        evaluation = {
            "graph": evaluation_graph,
            "origin_ids": evaluation_graph["origin_ids"],
            "y": torch.tensor([0, 1, 0, 1]),
            "masks": {
                "checkpoint": torch.tensor([True, True, False, False]),
                "test": torch.tensor([False, False, True, True]),
            },
        }
        class_emb = torch.randn(2, train["text_emb"].shape[1],
                                generator=torch.Generator().manual_seed(9))
        torch.save(train, prepared / "stage_0_train.pt")
        torch.save(evaluation, prepared / "stage_0_eval.pt")
        torch.save(class_emb, prepared / "class_text_emb.pt")
        files = {}
        for path in prepared.glob("*.pt"):
            files[path.name] = {"sha256": file_sha(path)}
        manifest = {"stage_classes": [[0, 1]], "files": files}
        (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        manifest_sha = file_sha(prepared / "manifest.json")

        base = {
            "prepared": "original-location-is-provenance-only",
            "output": "original-run",
            "phase": "continual",
            "stages": 1,
            "rate": 0.5,
            "seed": 44,
            "model_seeds": [100, 101],
            "arms": [],
            "device": "cpu",
            "threads": 1,
            "hidden": 4,
            "lr": 0.001,
            "student_steps": 0,
            "check_every": 1,
            "batch_size": 2,
            "source_manifest_sha256": manifest_sha,
        }
        base_path = root / "source_run_config.json"
        base_path.write_text(json.dumps(base), encoding="utf-8")

        artifact = tiny_data(2, 200)
        artifact["metadata"] = {
            "stage": 0,
            "training_labels_used": False,
        }
        artifact_path = root / "selected.pt"
        torch.save(artifact, artifact_path)
        return Namespace(
            base_config=str(base_path), prepared=str(prepared), artifact=str(artifact_path),
            output=str(root / "new-run"), stage=0, label="selected_outer20",
            device="cpu", model_seeds=[100, 101], check_every=1,
        )

    def test_end_to_end_saves_two_matrix_discoverable_students_and_audit_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root)
            completed = train_saved_artifact_students(args)
            output = Path(args.output)

            self.assertEqual(completed["status"], "complete")
            self.assertFalse(completed["weighted"])
            self.assertEqual([row["seed"] for row in completed["models"]], [100, 101])
            for seed in args.model_seeds:
                model_path = output / f"selected_outer20_s0_m{seed}_model.pt"
                train_path = output / f"selected_outer20_s0_m{seed}_train.json"
                self.assertTrue(model_path.is_file())
                self.assertTrue(train_path.is_file())
                saved = torch.load(model_path, map_location="cpu", weights_only=True)
                self.assertFalse(saved["training_labels_used"])
                self.assertEqual(saved["stage"], 0)
                log = json.loads(train_path.read_text(encoding="utf-8"))
                self.assertFalse(log["weighted"])
                self.assertEqual(log["selected_step"], 0)

            run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["artifact_sha256"], file_sha(Path(args.artifact)))
            self.assertEqual(run_config["source_config_sha256"], file_sha(Path(args.base_config)))
            self.assertTrue((output / "artifact_inputs.json").is_file())
            self.assertTrue((output / "student_runs.json").is_file())
            discovered = discover_models([output], None)
            self.assertEqual([(row["arm"], row["stage"], row["seed"]) for row in discovered],
                             [("selected_outer20", 0, 100), ("selected_outer20", 0, 101)])

    def test_rejects_labels_wrong_stage_and_existing_output_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root)
            artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
            artifact["y"] = torch.zeros(len(artifact["x"]), dtype=torch.long)
            with self.assertRaisesRegex(ValueError, "forbidden label fields"):
                validate_artifact(artifact, "correct-manifest", 0)

            artifact.pop("y")
            artifact["stage"].fill_(1)
            with self.assertRaisesRegex(ValueError, "end at the requested stage"):
                validate_artifact(artifact, "correct-manifest", 0)

            artifact["stage"].fill_(0)
            artifact["metadata"]["source_manifest_sha256"] = "wrong-manifest"
            with self.assertRaisesRegex(ValueError, "different prepared manifest"):
                validate_artifact(artifact, "correct-manifest", 0)

            base_path = Path(args.base_config)
            base = json.loads(base_path.read_text(encoding="utf-8"))
            expected_manifest = base["source_manifest_sha256"]
            base["source_manifest_sha256"] = "wrong-manifest"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base config belongs"):
                train_saved_artifact_students(args)
            self.assertFalse(Path(args.output).exists())
            base["source_manifest_sha256"] = expected_manifest
            base_path.write_text(json.dumps(base), encoding="utf-8")

            Path(args.output).mkdir()
            with self.assertRaisesRegex(FileExistsError, "never overwrite"):
                train_saved_artifact_students(args)
            self.assertEqual(list(Path(args.output).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
