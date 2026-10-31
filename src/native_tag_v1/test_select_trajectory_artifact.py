import json
import tempfile
import unittest
from pathlib import Path

import torch

from run_native_experiments import file_sha
from select_trajectory_artifact import export_selected, select_row


class SelectTrajectoryArtifactTests(unittest.TestCase):
    def test_tie_prefers_lower_outer_step(self):
        trajectory = {"trajectory": [
            {"outer_step": 20, "artifact": "b.pt", "selected_student_mean_macro_f1": 42.0},
            {"outer_step": 0, "artifact": "a.pt", "selected_student_mean_macro_f1": 42.0},
        ]}
        selected = select_row(trajectory, "selected_student_mean_macro_f1")
        self.assertEqual(selected["outer_step"], 0)

    def test_export_restores_recursive_provenance_without_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact_path = root / "trajectory_joint_s1_outer_0.pt"
            torch.save({
                "x": torch.zeros(2, 3),
                "edge_index": torch.empty(2, 0, dtype=torch.long),
                "text_emb": torch.zeros(2, 4),
                "texts": ["a", "b"],
                "weights": torch.ones(2),
                "origin_ids": torch.arange(2),
                "stage": torch.ones(2, dtype=torch.long),
                "metadata": {"training_labels_used": False},
            }, artifact_path)
            (root / "trajectory.json").write_text(json.dumps({
                "outer_steps": [0],
                "trajectory": [{
                    "outer_step": 0,
                    "artifact": artifact_path.name,
                    "artifact_sha256": file_sha(artifact_path),
                    "selected_student_mean_macro_f1": 50.0,
                }],
            }))
            (root / "diagnostic_inputs.json").write_text(json.dumps({
                "confirmation_accessed": False,
                "training_labels_used": False,
                "outer_selector": "disabled; fixed pre-registered snapshots",
                "prepared_manifest_sha256": "manifest-sha",
                "stage": 1,
                "model_seeds": [100, 101],
            }))
            (root / "completed.json").write_text(json.dumps({"status": "complete"}))

            output = root / "selected.pt"
            result = export_selected(root, output, "selected_student_mean_macro_f1")
            exported = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(exported["metadata"]["source_manifest_sha256"], "manifest-sha")
            self.assertEqual(exported["metadata"]["stage"], 1)
            self.assertFalse(exported["metadata"]["selection"]["test_or_future_used"])
            self.assertNotIn("y", exported)
            self.assertEqual(result["selected_outer_step"], 0)


if __name__ == "__main__":
    unittest.main()
