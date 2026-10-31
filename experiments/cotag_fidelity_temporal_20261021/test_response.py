import unittest

import common as C
from common import R, torch
import tagsam_response as T


class ResponseTests(unittest.TestCase):
    def test_epoch_coverage(self):
        batches = T.inner_batches(41, 15, 20, 17, 'epochs')
        self.assertEqual(len(batches), 30)
        for epoch in range(15):
            joined = torch.cat(batches[epoch*2:epoch*2+2])
            self.assertEqual(len(joined.unique()), 40)

    def test_update_count(self):
        self.assertEqual(len(T.inner_batches(1000, 15, 16, 17, 'updates')), 15)

    def test_original_response_equivalence(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        n = 24
        initial = dict(x=torch.randn(n, 8), text_emb=torch.randn(n, 384),
            texts=[str(i) for i in range(n)], labels=torch.arange(n)%3,
            mass=torch.ones(n), depth=torch.zeros(n, dtype=torch.long),
            stage=torch.zeros(n, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))
        teacher = T.T.G.CLIPStudent(8)
        loss_fn, _ = T.T.official_loss(T.A.Path('/home/pc/research/TAGSAM_official'))
        actual, audit = T.optimize(initial, None, initial, teacher, 0, 'cpu', loss_fn, 2)
        expected, original = T.T.optimize(initial, None, initial, teacher, 0,
                                          'tagsam_selfloop', 'cpu', loss_fn, 2)
        self.assertTrue(torch.allclose(actual['x'], expected['x'], atol=1e-6, rtol=1e-6))
        self.assertEqual(audit['inner_steps_total'], original['inner_steps_total'])
        local, _ = T.optimize(initial, None, initial, teacher, 0, 'cpu', loss_fn, 2, batch_local=True)
        self.assertTrue(torch.allclose(local['x'], expected['x'], atol=2e-5, rtol=2e-5))


if __name__ == '__main__':
    unittest.main()
