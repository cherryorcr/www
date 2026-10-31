import unittest

from summarize import group_temporal, interval, paired_temporal, stage_average_temporal


def row(arm, seed, value, complete=True):
    return dict(variant=arm, construction_seed=seed, run_complete=complete,
                stage=4, mode='arrived_snapshot', split='historical_test',
                cohort='cumulative', f1=value, accuracy=value+10)


class SummaryTests(unittest.TestCase):
    def test_aggregation_unit_is_construction(self):
        values = [row('uq_dm_edgefree_float64', s, x) for s, x in zip((1,2,3),(1.,3.,5.))]
        values.append(row('uq_dm_edgefree_float64', 4, 1000., False))
        result = group_temporal(values)[0]
        self.assertEqual(result['constructions'], 3)
        self.assertAlmostEqual(result['f1'], 3.)
        self.assertAlmostEqual(result['f1_construction_sd'], 2.)

    def test_paired_seeds_not_student_replicates(self):
        values = [row('uq_dm_edgefree_float64', s, x) for s, x in zip((1,2,3),(5.,7.,9.))]
        values += [row('gecc_hard', s, x) for s, x in zip((1,2,3),(4.,5.,6.))]
        result = paired_temporal(values)[0]
        self.assertEqual(result['n'], 3)
        self.assertAlmostEqual(result['difference_pp'], 2.)
        self.assertEqual(result['positive_count'], 3)
        self.assertLess(result['ci95'][0], 0)

    def test_no_interval_for_one_memory(self):
        self.assertIsNone(interval([2.]))

    def test_duplicate_construction_rejected(self):
        with self.assertRaises(AssertionError):
            group_temporal([row('x', 1, 2.), row('x', 1, 2.)])

    def test_stage_mean_uses_construction_not_fifteen_independent_stages(self):
        values = [dict(row('uq_dm_edgefree_float64', seed, seed+stage), stage=stage)
                  for seed in (1, 3, 5) for stage in range(5)]
        values += [dict(row('uq_dm_edgefree_float64', 8, 1000., False), stage=0)]
        result = group_temporal(stage_average_temporal(values))[0]
        self.assertEqual(result['constructions'], 3)
        self.assertEqual(result['stage'], 'five_stage_mean')
        self.assertAlmostEqual(result['f1'], 5.)
        self.assertAlmostEqual(result['f1_construction_sd'], 2.)

    def test_stage_mean_rejects_missing_or_duplicate_stage(self):
        values = [dict(row('x', 1, stage), stage=stage) for stage in range(5)]
        for bad in (values[:4], values+[values[0]]):
            with self.assertRaises(AssertionError):
                stage_average_temporal(bad)

    def test_stage_mean_paired_interval_uses_matching_constructions(self):
        values = [dict(row(arm, seed, seed+stage+offset), stage=stage)
                  for arm, offset in (('uq_dm_edgefree_float64', 2), ('gecc_hard', 0))
                  for seed in (1, 2, 3) for stage in range(5)]
        result = paired_temporal(stage_average_temporal(values), stage='five_stage_mean')[0]
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['difference_pp'], 2.)
        self.assertEqual(result['ci95'], [2., 2.])


if __name__ == '__main__':
    unittest.main()
