import tempfile
from pathlib import Path
import unittest

import common as C
from common import torch, R
import native_tagsam as N


class NativeEpochTests(unittest.TestCase):
    def test_literal_native_epoch_and_frozen_prefix(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        native = N.components('cpu', 8)

        def block(n, stage):
            return dict(x=torch.randn(n, 8), text_emb=torch.randn(n, 384),
                labels=torch.arange(n)%3, mass=torch.ones(n),
                texts=[f'text {stage} {i}' for i in range(n)],
                depth=torch.zeros(n, dtype=torch.long),
                stage=torch.full((n,), stage, dtype=torch.long),
                edge_index=torch.empty((2, 0), dtype=torch.long))

        parent, initial = block(20, 0), block(20, 1)
        raw = R.L.append_edge_free(parent, initial)
        bert_text = torch.randn(40, 768)
        teacher = native.model(native.args).eval()
        with tempfile.TemporaryDirectory() as folder:
            memory, audit = N.optimize(native, initial, parent, bert_text, raw,
                bert_text, teacher, 1, 'cpu', 1, Path(folder))
        self.assertTrue(torch.equal(memory['x'][:20], parent['x']))
        self.assertEqual(audit['inner_updates_per_outer'], 30)
        self.assertEqual(audit['omitted_tail_per_epoch'], 0)
        self.assertTrue(bool(torch.isfinite(memory['x']).all()))
        self.assertFalse(torch.equal(memory['x'][20:], initial['x']))


if __name__ == '__main__':
    unittest.main()
