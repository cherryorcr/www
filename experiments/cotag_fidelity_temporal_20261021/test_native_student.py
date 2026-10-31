import unittest

import common as C
from common import torch
import native_tagsam as N
from native_student_diagnostic import train_native


class NativeStudentTests(unittest.TestCase):
    def test_fixed_epochs_keep_final_short_batch(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        native = N.components('cpu', 8)
        memory = dict(x=torch.randn(35, 8))
        before = memory['x'].clone()
        model, losses = train_native(native, memory, torch.randn(35, 768), 150,
                                     'cpu', .005, .005, 2)
        self.assertEqual(len(losses), 2)
        self.assertEqual([len(v['batch_losses']) for v in losses], [2, 2])
        self.assertTrue(torch.equal(memory['x'], before))
        self.assertFalse(model.training)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
