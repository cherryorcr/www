import json
import math
import unittest
from pathlib import Path

import torch

from student_consistent_edge_swap import (
    Config,
    enumerate_two_switches,
    materialize_cross_swap,
    evaluate_one_step,
    select_swaps,
    topology_audit,
)


def base_edges():
    # Old rows 0,1; new rows 2,3.  Each block is represented symmetrically.
    return torch.tensor([
        [0, 1, 2, 3, 0, 2, 1, 3],
        [1, 0, 3, 2, 2, 0, 3, 1],
    ], dtype=torch.long)


def candidate_edges():
    # The candidate pool contains the endpoint permutation (1,2),(0,3).
    return torch.tensor([
        [0, 1, 2, 3, 1, 2, 0, 3],
        [1, 0, 3, 2, 2, 1, 3, 0],
    ], dtype=torch.long)


def multi_degree_edges(extra=False):
    # Old rows 0,1,2; new rows 3,4.  The two new rows each have two
    # cross edges, while the old endpoint capacities are [1,2,1].
    pairs = [(0, 1), (1, 2), (3, 4),
             (0, 3), (1, 3), (1, 4), (2, 4)]
    if extra:
        pairs.extend([(2, 3), (0, 4)])
    directed = pairs + [(right, left) for left, right in pairs]
    return torch.tensor(directed, dtype=torch.long).T


class StudentConsistentSwapTest(unittest.TestCase):
    def test_enumeration_is_capacity_preserving(self):
        swaps = enumerate_two_switches(base_edges(), candidate_edges(), 2)
        self.assertEqual(len(swaps), 1)
        edge = materialize_cross_swap(base_edges(), swaps[0], 2)
        audit = topology_audit(base_edges(), edge, 2)
        self.assertTrue(all(audit[key] for key in (
            "old_old_preserved", "new_new_preserved",
            "new_row_set_preserved", "per_new_cross_degree_preserved",
            "old_endpoint_capacity_preserved")))
        self.assertEqual(audit["changed_cross_pairs"], 2)

    def test_fallback_when_new_response_does_not_improve(self):
        memory = {"edge_index": base_edges(), "x": torch.zeros(4, 2)}

        def evaluate(value):
            changed = not torch.equal(value["edge_index"], base_edges())
            return {"new_gain": 0.1 if not changed else 0.09,
                    "old_gain": 0.1, "total_after": 1.0}

        result, audit = select_swaps(
            memory, candidate_edges(), 2,
            Config(max_swaps=1, new_gain_margin=0.0), evaluate)
        self.assertTrue(audit["fallback_to_input"])
        self.assertTrue(torch.equal(result["edge_index"], base_edges()))

    def test_accepts_only_guarded_student_improvement(self):
        memory = {"edge_index": base_edges(), "x": torch.zeros(4, 2)}

        def evaluate(value):
            changed = not torch.equal(value["edge_index"], base_edges())
            return {"new_gain": 0.2 if changed else 0.1,
                    "old_gain": 0.1 if changed else 0.1,
                    "total_after": 0.9 if changed else 1.0}

        result, audit = select_swaps(
            memory, candidate_edges(), 2,
            Config(max_swaps=1), evaluate)
        self.assertFalse(audit["fallback_to_input"])
        self.assertEqual(audit["accepted_count"], 1)
        self.assertFalse(torch.equal(result["edge_index"], base_edges()))

    def test_majority_seed_guard_rejects_average_only_gain(self):
        memory = {"edge_index": base_edges(), "x": torch.zeros(4, 2)}

        def evaluate(value):
            changed = not torch.equal(value["edge_index"], base_edges())
            if not changed:
                gains = [0.10, 0.10, 0.10]
            else:
                # The mean improves, but two of three selector seeds regress.
                gains = [0.30, 0.00, 0.00]
            return {
                "new_gain": sum(gains) / len(gains),
                "old_gain": 0.10,
                "total_after": 0.90,
                "records": [
                    {"seed": seed, "new_gain": gain,
                     "old_gain": 0.10, "total_after": 0.90}
                    for seed, gain in zip((17, 23, 29), gains)
                ],
            }

        result, audit = select_swaps(
            memory, candidate_edges(), 2,
            Config(max_swaps=1, min_seed_fraction=2 / 3), evaluate)
        self.assertTrue(audit["fallback_to_input"])
        self.assertTrue(torch.equal(result["edge_index"], base_edges()))

    def test_multi_degree_cross_slots_are_swapped_without_changing_capacity(self):
        base = multi_degree_edges()
        candidate = multi_degree_edges(extra=True)
        swaps = enumerate_two_switches(base, candidate, 3)
        self.assertEqual(len(swaps), 1)
        edge = materialize_cross_swap(base, swaps[0], 3)
        audit = topology_audit(base, edge, 3)
        self.assertTrue(all(audit[key] for key in (
            "old_old_preserved", "new_new_preserved", "new_row_set_preserved",
            "per_new_cross_degree_preserved", "old_endpoint_capacity_preserved")))
        self.assertEqual(audit["base_cross_pairs"], 4)
        self.assertEqual(audit["result_cross_pairs"], 4)
        self.assertEqual(audit["changed_cross_pairs"], 2)
        cross = {(left, right) for left, right in edge.T.tolist()
                 if left < 3 <= right}
        self.assertEqual(cross, {(0, 4), (1, 3), (1, 4), (2, 3)})

    def test_real_sealed_small_memory_student_smoke(self):
        """Run one canonical Student response on a checked-in experiment artifact."""
        project = Path(__file__).resolve().parents[2]
        cell = (project / "results" / "tgc_fresh_semantic_recursive_20260926"
                / "cora_order19611_s1")
        protocol_path = cell / "protocol.json"
        memory_path = cell / "semantic_perm_memory.pt"
        if not protocol_path.exists() or not memory_path.exists():
            self.skipTest("sealed small-memory artifact is not available")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        memory = torch.load(memory_path, map_location="cpu", weights_only=False)
        stats = evaluate_one_step(
            memory, int(protocol["old_rows"]), (17,), device="cpu")
        for key in ("old_gain", "new_gain", "total_after"):
            self.assertTrue(math.isfinite(float(stats[key])))
        self.assertEqual(stats["student_steps"], 1)
        self.assertEqual(stats["student_seeds"], [17])


if __name__ == "__main__":
    unittest.main()
