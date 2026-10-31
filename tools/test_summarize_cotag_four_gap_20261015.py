"""Data-contract and pairing regressions for the research evidence aggregator."""
from copy import deepcopy
import unittest

import numpy as np

import summarize_cotag_four_gap_20261015 as S


def fixture():
    protocol = dict(dataset='arxiv', order_seed=9101, construction_seed=271006,
        groups=[[0], [1]], budgets=[5, 10], stage_raw_rows=[100, 100],
        student={'name': 'prompt'}, student_steps=2048, student_batch=24,
        student_seeds=[150, 151, 152], pool_mask_hash='pool',
        report_mask_hash='dev', test_mask_hash='test', stage_id_hashes=['a', 'b'])
    rows = []
    for stage in (0, 1):
        scores = {}
        for split in S.SPLITS:
            students = [dict(seed=seed, final_f1=40.+i, accuracy=60.+i,
                old_f1=None if stage == 0 else 30.+i, new_f1=50.+i)
                for i, seed in enumerate(protocol['student_seeds'])]
            score = dict(split=split, per_student=students,
                **{key: protocol[key] for key in ('student_steps', 'student_batch', 'student_seeds')})
            for metric in S.METRICS[:-1]:
                score[metric] = (None if metric == 'old_f1' and stage == 0 else
                                 float(np.mean([s[metric] for s in students])))
            scores[split] = score
        rows.append(dict(stage=stage, arm='full', memory_rows=protocol['budgets'][stage],
            checks=dict(budget_ok=True, prefix_ok=True, finite_ok=True), scores=scores))
    return dict(protocol=protocol, rows=rows, status='complete')


class EvidenceContracts(unittest.TestCase):
    def test_valid_source_and_unordered_rows(self):
        data = fixture()
        data['rows'].reverse()
        self.assertEqual([r['stage'] for r in S.validate_rows(data, 'full')], [0, 1])

    def test_missing_or_duplicate_stage_is_rejected(self):
        for mode in ('missing', 'duplicate'):
            data = fixture()
            data['rows'] = data['rows'][:1] if mode == 'missing' else data['rows']*2
            with self.subTest(mode=mode), self.assertRaises(AssertionError):
                S.validate_rows(data)

    def test_changed_protocol_is_rejected(self):
        primary = fixture()
        for key in S.MATCH + ('dataset', 'order_seed', 'construction_seed'):
            data = deepcopy(primary)
            data['protocol'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(AssertionError):
                S.validate_protocol(data, primary)

    def test_bad_seed_budget_split_and_aggregate_are_rejected(self):
        mutations = {
            'duplicate_seed': lambda r: r['scores']['sealed_test']['per_student'][0].update(seed=151),
            'missing_seed': lambda r: r['scores']['sealed_test']['per_student'].pop(),
            'training_budget': lambda r: r['scores']['sealed_test'].update(student_steps=100),
            'row_budget': lambda r: r.update(memory_rows=9),
            'split': lambda r: r['scores']['sealed_test'].update(split='report_dev'),
            'mean': lambda r: r['scores']['sealed_test'].update(final_f1=99.),
            'nan': lambda r: r['scores']['sealed_test']['per_student'][0].update(final_f1=float('nan')),
            'percentage': lambda r: r['scores']['sealed_test']['per_student'][0].update(final_f1=101.),
            'prefix': lambda r: r['checks'].update(prefix_ok=False),
            'missing_check': lambda r: r['checks'].pop('finite_ok'),
        }
        for name, mutate in mutations.items():
            data = fixture()
            mutate(data['rows'][-1])
            with self.subTest(name=name), self.assertRaises(AssertionError):
                S.validate_rows(data)

    def test_retention_pairs_by_student_after_construction_average(self):
        reference = dict(per_student=[dict(seed=152, final_f1=80.),
            dict(seed=150, final_f1=20.), dict(seed=151, final_f1=50.)])
        ours = [dict(per_student=[dict(seed=seed, final_f1=value+offset)
            for seed, value in ((151, 60.), (152, 40.), (150, 30.))]) for offset in (-2., 0., 2.)]
        pairs = S.paired_retention(ours, reference, 'final_f1', [150, 151, 152])
        self.assertEqual([p['retention'] for p in pairs], [150., 120., 50.])
        self.assertEqual(len(pairs), 3)
        self.assertNotAlmostEqual(np.mean([p['retention'] for p in pairs]),
                                  100*sum((30., 60., 40.))/sum((20., 50., 80.)))

    def test_construction_cells_define_uncertainty(self):
        result = S.stats([1., 2., 3.])
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['sd'], 1.)
        self.assertAlmostEqual(result['ci_low'], -0.48413771171954556)
        for values in ([], [1., float('nan')], [float('inf')]):
            with self.assertRaises(AssertionError):
                S.stats(values)


if __name__ == '__main__':
    unittest.main()
