import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import run_degree_cross_task_confirmation as R
import summarize_degree_confirmation as S


class ConfirmationTests(unittest.TestCase):
    def test_orders(self):
        groups = [[0, 1, 2, 3], [4, 5], [6, 7], [8], [9]]
        a = R.fresh_groups(groups, 49011)
        self.assertEqual(a, R.fresh_groups(groups, 49011))
        self.assertNotEqual(a, R.fresh_groups(groups, 49012))
        self.assertEqual(list(map(len, a)), list(map(len, groups)))
        self.assertEqual(sorted(sum(a, [])), list(range(10)))
        with self.assertRaises(ValueError):
            R.fresh_groups([[0], [0]], 1)

    def test_construct_uses_only_pool_and_no_edges(self):
        gen = torch.Generator().manual_seed(91)
        D = {"x": torch.randn(220, 8, generator=gen),
             "text_emb": torch.randn(220, 8, generator=gen), "texts": ["a"] * 220,
             "y": torch.arange(220) // 110, "pool_mask": torch.arange(220) % 11 != 0,
             "stage_classes": [[0], [1]], "encoder": None, "edge_index": "forbidden"}

        def block(source, ids, k, encoder, stage, *args):
            self.assertEqual(set(source), {"x", "text_emb", "texts", "y"})
            self.assertTrue(bool(D["pool_mask"][ids].all()))
            return {"x": D["x"][ids[:k]], "text_emb": D["text_emb"][ids[:k]],
                    "texts": ["a"] * k, "mass": torch.ones(k), "depth": torch.zeros(k).long(),
                    "stage": torch.full((k,), stage), "allocation": {stage: k}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(R.U, "build_stage_block", block):
            root = Path(tmp)
            result = R.construct(D, root, "cpu")
            for arm in R.ARMS:
                old = torch.load(root / result[0]["arms"][arm]["path"], weights_only=False)
                new = torch.load(root / result[1]["arms"][arm]["path"], weights_only=False)
                R.frozen(old, new)
                new["x"][0, 0] += 1
                with self.assertRaises(AssertionError):
                    R.frozen(old, new)

    def test_matrix_rejects_missing_duplicate_or_different_memory(self):
        records = []
        for order, stage, arm, seed in itertools.product(S.ORDERS, range(5), S.ARMS, S.SEEDS):
            for task, split in (("nc", None), ("lp", 49101), ("lp", 49102)):
                records.append(dict(task=task, split=split, order=order, stage=stage, arm=arm,
                                    seed=seed, memory_sha256=str((order, stage, arm))))
        S.validate(records)
        for bad in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                S.validate(bad)
        records[0]["memory_sha256"] = "different"
        with self.assertRaises(ValueError):
            S.validate(records)

    def test_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.pt"
            a = R.fixed_evaluation(path, {"x": torch.ones(2)})
            self.assertEqual(a, R.fixed_evaluation(path, {"x": torch.ones(2)}))
            with self.assertRaises(AssertionError):
                R.fixed_evaluation(path, {"x": torch.zeros(2)})

    def test_paired_summary_and_decision_do_not_pick_xt(self):
        records = []
        for order, stage, arm, seed in itertools.product(S.ORDERS, range(5), S.ARMS, S.SEEDS):
            improvement = {"c0": 0., "t_degree": 1., "xt_degree": 5.}[arm]
            nc = dict(cumulative_f1=40 + improvement, new_f1=30 + improvement,
                      old_f1=50 + improvement if stage else None)
            lp = dict(auc=.8 + improvement / 100, ap=.7 + improvement / 100)
            for task, split in (("nc", None), ("lp", 49101), ("lp", 49102)):
                metrics = ({scope: nc for scope in ("all", "seen")} if task == "nc" else
                           {scope: lp for scope in ("all", "old", "new", "cross", "within_new")})
                records.append(dict(task=task, split=split, order=order, stage=stage, arm=arm,
                                    seed=seed, memory_sha256=str((order, stage, arm)), metrics=metrics))
        rows = S.summarize(records)
        verdict = S.decision("bookchild", rows)
        self.assertTrue(verdict["lp_passed"] and verdict["nc_passed"])
        self.assertEqual(verdict["candidate"], "t_degree")
        for row in rows:
            if row["task"] == "nc" and row["arm"] == "t_degree":
                row["delta_pp"]["mean"] = -3.
        self.assertFalse(S.decision("bookchild", rows)["nc_passed"])
        self.assertTrue(S.decision("bookchild", rows)["lp_passed"])


if __name__ == "__main__":
    unittest.main()
