import unittest

import numpy as np
import torch
import torch.nn.functional as F

from small_comparison_metrics import score_logits, score_predictions


class MetricTests(unittest.TestCase):
    def test_noncontiguous_prompt_mapping(self):
        logits = [[0., 9., 0., 1.], [0., 1., 0., 9.]]
        result = score_logits(logits, np.array([1, 3]), [[1, 3], [0, 2]], 0)
        self.assertEqual(result["seen"]["cumulative_f1"], 100.)

    def test_cross_task_false_positives_are_counted(self):
        result = score_predictions([0, 0, 1, 1], [0, 0, 0, 1], [[0], [1]], 1)
        self.assertAlmostEqual(result["old_f1"], 80.)
        self.assertAlmostEqual(result["new_f1"], 200 / 3)

    def test_final_prompt_scopes_equal(self):
        logits = [[9., 2., 1.], [1., 9., 2.], [9., 1., 2.]]
        result = score_logits(logits, np.array([0, 1, 2]), [[2], [0, 1]], 1)
        self.assertEqual(result["seen"], result["all"])

    def test_class_weighting_not_task_weighting(self):
        result = score_predictions([0, 1, 2], [0, 0, 0], [[0], [1, 2]], 1)
        self.assertAlmostEqual(result["cumulative_f1"], 50 / 3)
        self.assertNotAlmostEqual(result["cumulative_f1"], np.mean(result["per_task_f1"]))


class ProtocolTests(unittest.TestCase):
    def test_induced_support_excludes_holdout_endpoints(self):
        from run_small_comparison_audited import induced_edges
        edges = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
        compact = induced_edges(edges, torch.tensor([2, 0]), 4)
        self.assertTrue(torch.equal(compact, torch.tensor([[1], [0]])))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression")
    def test_student_matches_canonical_parameters(self):
        import run_small_comparison_audited as C
        torch.set_num_threads(2)
        torch.manual_seed(7)
        x, text = torch.randn(20, 8), torch.randn(20, 384)
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        memory = dict(x=x, text_emb=text, edge_index=edges)
        data = dict(x_device=x.cuda(), real_adj=C.G.norm_adj(edges, len(x), "cuda:0"),
                    prompt_emb=text[:2], y=torch.arange(20) % 2,
                    stage_classes=[[0, 1]], report_dev_mask=torch.ones(20, dtype=torch.bool))
        canonical = C.G.train_student(memory, 100, "cuda:0", 64)
        row = C.fit(memory, 100, 64, "cuda:0", data, 0)
        self.assertEqual(row["end_state_sha256"], C.tensor_hash(canonical.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA bilevel smoke")
    def test_official_response_gradient_preserves_old_memory(self):
        from pathlib import Path
        import run_tagsam_graphclip_comparison as A
        loss_fn, _ = A.official_loss(Path("/home/pc/research/TAGSAM_official"))
        torch.manual_seed(9)
        def block(n, stage):
            return dict(x=torch.randn(n, 8), text_emb=F.normalize(torch.randn(n, 384), dim=1),
                        texts=["test"] * n, mass=torch.ones(n),
                        depth=torch.zeros(n, dtype=torch.long),
                        stage=torch.full((n,), stage, dtype=torch.long),
                        edge_index=torch.empty((2, 0), dtype=torch.long))
        old, new, raw = block(4, 0), block(5, 1), block(12, 1)
        teacher = A.G.CLIPStudent(8).cuda().eval()
        for arm in ("tagsam_selfloop", "tagsam_semantic", "asm_rowcol_semantic"):
            memory, audit = A.optimize(new, old, raw, teacher, 1, arm, "cuda:0", loss_fn, 1)
            A.C.assert_frozen(old, memory)
            self.assertGreater(audit["curve"][0]["x_shift_rms"], 0.)
            self.assertTrue(torch.isfinite(memory["x"]).all())


if __name__ == "__main__":
    unittest.main()
