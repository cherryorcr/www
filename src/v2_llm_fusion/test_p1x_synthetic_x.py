"""Unit tests for the controlled P1-X synthetic feature operators."""
from __future__ import annotations

import unittest

import torch

import run_student_compatible_append as SCA


class P1XSyntheticXTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [-0.7, 0.9],
            [-0.5, 1.1],
        ])
        self.text = torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ])
        self.joint = torch.cat((
            torch.nn.functional.normalize(self.x, dim=1), self.text), dim=1)
        self.members = torch.arange(len(self.x))

    def test_mean_matches_legacy_definition(self):
        actual = SCA.synthesize_x(self.x, self.joint, self.members, "mean")
        expected = self.x.mean(0)
        self.assertTrue(torch.equal(actual, expected))

    def test_weighted_is_finite_and_convex(self):
        actual = SCA.synthesize_x(
            self.x, self.joint, self.members, "weighted")
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.all(actual >= self.x.min(0).values - 1e-6))
        self.assertTrue(torch.all(actual <= self.x.max(0).values + 1e-6))

    def test_moment_is_deterministic_and_not_a_source_copy(self):
        first = SCA.synthesize_x(self.x, self.joint, self.members, "moment")
        second = SCA.synthesize_x(self.x, self.joint, self.members, "moment")
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(any(torch.allclose(first, row) for row in self.x))

    def test_block_keeps_non_x_payload_fixed(self):
        common = dict(
            x=self.x,
            text_emb=self.text,
            texts=["a", "b", "c", "d"],
            k=2,
            encoder=None,
            recipe_name="joint_g5_c320",
            stage=1,
            text_operator="no_reencode",
            device="cpu",
        )
        mean = SCA.build_compact_block(**common, x_operator="mean")
        weighted = SCA.build_compact_block(**common, x_operator="weighted")
        moment = SCA.build_compact_block(**common, x_operator="moment")
        self.assertEqual(mean["x"].shape, weighted["x"].shape)
        self.assertEqual(mean["x"].shape, moment["x"].shape)
        self.assertTrue(torch.equal(mean["text_emb"], weighted["text_emb"]))
        self.assertTrue(torch.equal(mean["text_emb"], moment["text_emb"]))
        self.assertTrue(torch.equal(mean["mass"], weighted["mass"]))
        self.assertTrue(torch.equal(mean["mass"], moment["mass"]))
        self.assertEqual(mean["texts"], weighted["texts"])
        self.assertEqual(mean["texts"], moment["texts"])
        self.assertEqual(mean["x_operator"], "mean")
        self.assertEqual(weighted["x_operator"], "weighted")
        self.assertEqual(moment["x_operator"], "moment")

    def test_invalid_operator_is_rejected(self):
        with self.assertRaises(ValueError):
            SCA.synthesize_x(self.x, self.joint, self.members, "copy_real")


if __name__ == "__main__":
    unittest.main()
