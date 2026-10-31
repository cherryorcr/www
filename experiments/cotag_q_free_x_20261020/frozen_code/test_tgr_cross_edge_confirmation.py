import copy
import unittest

import torch

from run_tgr_cross_edge_confirmation import audit_transport


class TransportAuditTest(unittest.TestCase):
    def fixture(self):
        parent = {key: torch.ones(2, 3) if key in ("x", "text_emb") else torch.ones(2)
                  for key in ("x", "text_emb", "mass", "depth", "stage")}
        parent["texts"] = ["a", "b"]
        parent["edge_index"] = torch.tensor([[0, 1], [1, 0]])
        base = {key: torch.cat((value, value.clone()), dim=0)
                for key, value in parent.items() if key not in ("texts", "edge_index")}
        base["texts"] = ["a", "b", "c", "d"]
        base["edge_index"] = torch.tensor([[0, 1, 2, 3, 0, 2, 1, 3],
                                           [1, 0, 3, 2, 2, 0, 3, 1]])
        return parent, base

    def test_degree_preserving_swap_is_valid(self):
        parent, base = self.fixture()
        candidate = copy.deepcopy(base)
        candidate["edge_index"] = torch.tensor([[0, 1, 2, 3, 1, 2, 0, 3],
                                                [1, 0, 3, 2, 2, 1, 3, 0]])
        self.assertEqual(audit_transport(base, candidate, parent)["changed_cross_pairs"], 2)

    def test_same_edge_count_with_old_overload_is_rejected(self):
        parent, base = self.fixture()
        candidate = copy.deepcopy(base)
        candidate["edge_index"] = torch.tensor([[0, 1, 2, 3, 0, 2, 0, 3],
                                                [1, 0, 3, 2, 2, 0, 3, 0]])
        with self.assertRaises(AssertionError):
            audit_transport(base, candidate, parent)

    def test_silent_content_change_is_rejected(self):
        parent, base = self.fixture()
        candidate = copy.deepcopy(base)
        candidate["x"][0, 0] += 0.01
        with self.assertRaises(AssertionError):
            audit_transport(base, candidate, parent)


if __name__ == "__main__":
    unittest.main()
