import tempfile
import unittest
from pathlib import Path

import torch

import run_semantic_topology_controls as R


class SemanticRunnerTest(unittest.TestCase):
    def test_combine_preserves_fields_and_old_edges(self):
        old = dict(x=torch.zeros(2, 3), text_emb=torch.zeros(2, 3), mass=torch.ones(2),
                   depth=torch.zeros(2), stage=torch.zeros(2, dtype=torch.long), texts=["a", "b"],
                   edge_index=torch.tensor([[0, 1], [1, 0]]))
        block = dict(x=torch.ones(2, 3), text_emb=torch.ones(2, 3), mass=torch.ones(2),
                     depth=torch.ones(2), stage=torch.ones(2, dtype=torch.long), texts=["c", "d"])
        memory = R.combine(old, block, 2, torch.tensor([[0, 1], [1, 0]]),
                           torch.tensor([[0], [2]]))
        self.assertTrue(torch.equal(memory["x"][:2], old["x"]))
        self.assertTrue(torch.equal(memory["edge_index"][:, :2], old["edge_index"]))
        self.assertEqual(len(R.L.pairs(memory["edge_index"], 4).T), 3)

    def test_source_hashes_exist(self):
        root = Path(__file__).resolve().parents[2]
        hashes = R.source_hashes(root)
        self.assertEqual(len(hashes), 3)
        self.assertTrue(all(len(v) == 64 for v in hashes.values()))


if __name__ == "__main__":
    unittest.main()
