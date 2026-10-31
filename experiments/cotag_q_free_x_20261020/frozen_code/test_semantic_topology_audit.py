import tempfile
import unittest
from pathlib import Path

import torch

import audit_semantic_topology_controls as A


class AuditTest(unittest.TestCase):
    def test_checkpoint_hash_uses_topology_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fit.pt"
            evaluation = Path(directory) / "evaluation.pt"
            context = dict(stage=0, arm="knn_nearest", seed=210)
            torch.save(dict(context=context), checkpoint)
            torch.save(dict(ids=torch.arange(2)), evaluation)
            record = dict(context, checkpoint_sha256=A.T.sha(checkpoint),
                          evaluation_sha256=A.T.sha(evaluation), memory_sha256="memory")
            construction = [dict(arms={"knn_nearest": {"sha256": "memory"}})]
            self.assertEqual(A.check_state(checkpoint, evaluation, record, construction)["context"], context)
            record["checkpoint_sha256"] = "modified"
            with self.assertRaises(AssertionError):
                A.check_state(checkpoint, evaluation, record, construction)


if __name__ == "__main__":
    unittest.main()
