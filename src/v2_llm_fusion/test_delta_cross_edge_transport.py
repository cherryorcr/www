import unittest

import torch

from delta_cross_edge_transport import (
    build_raw_to_old_slot,
    materialize_cross_only,
    materialize_cross_only_global,
    make_support,
)


def memory(n, d=4):
    return {
        "x": torch.arange(n * d, dtype=torch.float32).reshape(n, d) + 1,
        "text_emb": torch.nn.functional.normalize(
            torch.arange(n * d, dtype=torch.float32).reshape(n, d) + 2, dim=1),
        "texts": [f"old-{i}" for i in range(n)],
        "mass": torch.ones(n),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.zeros(n, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        "class_id": torch.tensor([0, 0, 1], dtype=torch.long),
    }


def delta():
    return {
        "x": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
        "text_emb": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
        "texts": ["new-0", "new-1"],
        "mass": torch.ones(2),
        "depth": torch.zeros(2, dtype=torch.long),
        "stage": torch.ones(2, dtype=torch.long),
        "class_id": torch.tensor([2, 2], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "current_delta_ids": torch.tensor([10, 11]),
    }


class DeltaCrossEdgeTransportTest(unittest.TestCase):
    def test_none_keeps_old_and_delta_internal_edges_only(self):
        old, new = memory(3), delta()
        support, audit = make_support(old, new, mode="none")
        self.assertEqual(audit["cross_edge_pairs"], 0)
        self.assertEqual(support["edge_index"].shape[1], 6)
        self.assertFalse(bool(((support["edge_index"][0] < 3) ^
                               (support["edge_index"][1] < 3)).any()))

    def test_random_and_semantic_have_same_bounded_budget(self):
        old, new = memory(3), delta()
        random_a, audit_a = make_support(old, new, mode="random", seed=8)
        random_b, _ = make_support(old, new, mode="random", seed=8)
        semantic, audit_s = make_support(old, new, mode="semantic")
        self.assertTrue(torch.equal(random_a["edge_index"], random_b["edge_index"]))
        self.assertEqual(audit_a["cross_edge_pairs"], 2)
        self.assertEqual(audit_s["cross_edge_pairs"], 2)
        self.assertEqual(random_a["edge_index"].shape[1], semantic["edge_index"].shape[1])

    def test_message_balanced_limits_support_load(self):
        old, new = memory(3), delta()
        support, audit = make_support(old, new, mode="message_balanced",
                                      cross_degree=1)
        self.assertEqual(audit["cross_edge_pairs"], len(new["x"]))
        self.assertLessEqual(audit["selection"]["max_old_load"],
                             audit["selection"]["capacity_cap"])

    def test_incident_mapping_uses_only_current_incident_edges(self):
        old, new = memory(3), delta()
        records = [
            {"source_original_ids": [20]},
            {"source_original_ids": [21]},
            {"source_original_ids": [22]},
        ]
        provenance = build_raw_to_old_slot(records, 3)
        raw_edges = torch.tensor([[10, 11, 10, 99], [20, 22, 21, 10]])
        support, audit = make_support(
            old, new, mode="incident", raw_edge_index=raw_edges,
            current_ids=torch.tensor([10, 11]), raw_to_old_slot=provenance)
        self.assertEqual(audit["cross_edge_pairs"], 2)
        cross = support["edge_index"][:, ((support["edge_index"][0] < 3) ^
                                           (support["edge_index"][1] < 3))]
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in cross.T.tolist()}
        self.assertEqual(pairs, {(0, 3), (2, 4)})

    def test_incident_without_provenance_is_explicit(self):
        with self.assertRaises(ValueError):
            make_support(memory(3), delta(), mode="incident",
                         raw_edge_index=torch.tensor([[0], [1]]),
                         current_ids=torch.tensor([0, 1]))

    def test_cross_only_materializer_preserves_old_and_new_blocks(self):
        old, new = memory(3), delta()
        support, _ = make_support(old, new, mode="semantic")
        projected = torch.zeros((5, 5))
        projected[2, 3] = 10.0
        projected[1, 4] = 9.0
        edge, audit = materialize_cross_only(
            torch.tensor([[0, 1, 1, 0, 2, 3, 3, 4, 0, 3, 1, 4],
                          [1, 0, 2, 1, 1, 4, 3, 3, 3, 0, 4, 1]]),
            projected, 3)
        old_base = {(0, 1), (1, 0), (1, 2), (2, 1)}
        old_new = {(a, b) for a, b in edge.T.tolist()
                   if a < 3 and b < 3}
        self.assertEqual(old_new, old_base)
        self.assertEqual(audit["cross_pairs_before"], 2)
        self.assertEqual(audit["cross_pairs_after"], 2)
        self.assertTrue(audit["per_new_cross_degree_preserved"])

    def test_global_materializer_uses_capacity_optimum(self):
        # Two new rows share two old endpoint slots.  Greedy row order would
        # consume old row 0 for both high local scores; global assignment
        # must use each C0 capacity exactly once.
        base_edge = torch.tensor([
            [0, 1, 2, 3, 0, 2, 1, 3],
            [1, 0, 3, 2, 2, 0, 3, 1],
        ], dtype=torch.long)
        projected = torch.zeros((4, 4), dtype=torch.float32)
        projected[0, 2] = 10.0
        projected[0, 3] = 9.0
        projected[1, 2] = 9.0
        projected[1, 3] = 0.0
        edge, audit = materialize_cross_only_global(base_edge, projected, 2)
        cross = {(a, b) for a, b in edge.T.tolist()
                 if a < 2 <= b}
        self.assertEqual(cross, {(1, 2), (0, 3)})
        self.assertTrue(audit["old_capacity_exact"])
        self.assertEqual(audit["global_objective"], 18.0)
        self.assertEqual(audit["assignment_method"],
                         "global_capacity_linear_sum_assignment")


if __name__ == "__main__":
    unittest.main()
