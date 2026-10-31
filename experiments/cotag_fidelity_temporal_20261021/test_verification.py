import copy
import unittest

from verify_results import check_scores
from supplement import compare_protocols


class VerificationTests(unittest.TestCase):
    def test_score_mean_and_sample_sd(self):
        scores = dict(report_dev=dict(per_student=[dict(seed=s, final_f1=v, accuracy=v,
                old_f1=None, new_f1=v) for s, v in zip((150, 151, 152), (1., 2., 3.))],
            final_f1=2., final_f1_std=1., accuracy=2., accuracy_std=1.,
            old_f1=None, old_f1_std=0., new_f1=2., new_f1_std=1.))
        self.assertEqual(check_scores(scores, False), 6)
        bad = copy.deepcopy(scores)
        bad['report_dev']['final_f1'] = 2.1
        with self.assertRaises(AssertionError):
            check_scores(bad, False)

    def test_incompatible_order_rejected(self):
        fields = ('groups', 'budgets', 'stage_id_hashes', 'pool_mask_hash',
                  'report_mask_hash', 'test_mask_hash', 'student_seeds',
                  'student_steps', 'student_batch', 'order_seed', 'construction_seed')
        left = {name: 1 for name in fields}
        right = dict(left, order_seed=2)
        with self.assertRaises(AssertionError):
            compare_protocols(left, right)


if __name__ == '__main__':
    unittest.main()
