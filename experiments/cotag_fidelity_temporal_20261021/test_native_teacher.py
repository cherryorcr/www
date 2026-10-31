import unittest

from common import torch
import native_tagsam as N
from run_native_teacher_budget import train_teacher


class NativeTeacherTests(unittest.TestCase):
    def test_fixed_update_count_and_frozen_input(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        native = N.components('cpu', 8)
        raw = dict(x=torch.randn(12, 8), edge_index=torch.empty((2, 0), dtype=torch.long))
        before = raw['x'].clone()
        model, audit = train_teacher(native, raw, torch.randn(12, 768), 2, 'cpu')
        self.assertEqual(audit['updates'], 2048)
        self.assertEqual(len(audit['losses']), 2048)
        self.assertTrue(torch.equal(raw['x'], before))
        self.assertFalse(model.training)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
